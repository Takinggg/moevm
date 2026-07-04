#include <fcntl.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if defined(__APPLE__)
#ifndef F_NOCACHE
#define F_NOCACHE 48
#endif
#endif

namespace {

struct Options {
    std::string index_path;
    std::string sidecar_path;
    std::string trace_path;
    std::string out_path;
    std::string prefetch_policy = "oracle";
    double budget_gib = 8.0;
    int max_tokens = -1;
    int prefetch_window = 0;
    int prefetch_threads = 1;
    int predictor_topk = 10;
    int predictor_min_count = 1;
    int candidate_topk = 10;
    int candidate_delay_us = 0;
    int compute_us_per_access = 0;
    double prefetch_max_inflight_mib = 512.0;
    bool nocache = false;
    bool no_read = false;
};

struct Header {
    char magic[8];
    uint32_t version;
    uint32_t entry_count;
    uint32_t layer_count;
    uint32_t expert_count;
    uint64_t reserved[4];
};

struct Entry {
    uint64_t offset;
    uint32_t bytes;
    uint32_t reserved;
};

static_assert(sizeof(Header) == 56, "unexpected Header packing");
static_assert(sizeof(Entry) == 16, "unexpected Entry packing");

struct Metrics {
    uint64_t tokens = 0;
    uint64_t accesses = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t missing_index = 0;
    uint64_t read_ops = 0;
    uint64_t read_bytes = 0;
    uint64_t demand_read_ops = 0;
    uint64_t demand_read_bytes = 0;
    uint64_t prefetch_enqueued = 0;
    uint64_t prefetch_dropped = 0;
    uint64_t prefetch_deduped = 0;
    uint64_t prefetch_completed = 0;
    uint64_t prefetch_read_ops = 0;
    uint64_t prefetch_read_bytes = 0;
    uint64_t prefetch_waits = 0;
    uint64_t prefetch_wait_bytes = 0;
    uint64_t candidate_records = 0;
    uint64_t candidate_prefetch_keys = 0;
    uint64_t arena_evictions = 0;
    uint64_t arena_used_bytes = 0;
    uint64_t arena_high_water_bytes = 0;
    uint64_t arena_resident_records = 0;
    uint64_t arena_free_spans = 0;
    uint64_t arena_largest_free_span_bytes = 0;
    double prefetch_wait_s = 0.0;
    double elapsed_s = 0.0;
};

struct Access {
    int64_t token = 0;
    uint32_t key = 0;
    std::vector<uint32_t> candidates;
};

struct TraceData {
    std::vector<Access> accesses;
    uint64_t tokens = 0;
    uint64_t missing_index = 0;
    uint64_t candidate_records = 0;
};

struct PrefetchStats {
    uint64_t enqueued = 0;
    uint64_t dropped = 0;
    uint64_t deduped = 0;
    uint64_t completed = 0;
    uint64_t read_ops = 0;
    uint64_t read_bytes = 0;
};

uint64_t route_key(int64_t token, int64_t layer) {
    return (static_cast<uint64_t>(token) << 32) | static_cast<uint32_t>(layer);
}

class PredictorState {
  public:
    PredictorState(uint32_t layer_count, uint32_t expert_count, int topk, int min_count)
        : layer_count_(layer_count),
          expert_count_(expert_count),
          topk_(topk),
          min_count_(min_count),
          recent_by_layer_(layer_count),
          freq_top_by_layer_(layer_count),
          counts_(layer_count, std::vector<uint32_t>(expert_count, 0)) {}

    void observe(uint32_t key) {
        const uint32_t layer = key / expert_count_;
        const uint32_t expert = key % expert_count_;
        if (layer >= layer_count_ || expert >= expert_count_) {
            return;
        }

        auto &recent = recent_by_layer_[layer];
        recent.erase(std::remove(recent.begin(), recent.end(), key), recent.end());
        recent.insert(recent.begin(), key);
        if (recent.size() > static_cast<size_t>(topk_)) {
            recent.resize(static_cast<size_t>(topk_));
        }

        ++counts_[layer][expert];
        rebuild_freq_top(layer);
    }

    std::vector<uint32_t> predict(const std::string &policy, uint32_t layer) const {
        std::vector<uint32_t> out;
        if (layer >= layer_count_) {
            return out;
        }
        if (policy == "last-layer" || policy == "hybrid") {
            append_recent_with_threshold(out, layer);
        }
        if (policy == "freq-layer" || policy == "hybrid") {
            append_unique(out, freq_top_by_layer_[layer]);
        }
        return out;
    }

  private:
    static void append_unique(std::vector<uint32_t> &out, const std::vector<uint32_t> &values) {
        for (uint32_t value : values) {
            if (std::find(out.begin(), out.end(), value) == out.end()) {
                out.push_back(value);
            }
        }
    }

    void append_recent_with_threshold(std::vector<uint32_t> &out, uint32_t layer) const {
        for (uint32_t key : recent_by_layer_[layer]) {
            const uint32_t expert = key % expert_count_;
            if (counts_[layer][expert] < static_cast<uint32_t>(min_count_)) {
                continue;
            }
            if (std::find(out.begin(), out.end(), key) == out.end()) {
                out.push_back(key);
            }
        }
    }

    void rebuild_freq_top(uint32_t layer) {
        std::vector<uint32_t> experts(expert_count_);
        for (uint32_t i = 0; i < expert_count_; ++i) {
            experts[i] = i;
        }
        const auto &counts = counts_[layer];
        const size_t keep = std::min(static_cast<size_t>(topk_), experts.size());
        std::partial_sort(
            experts.begin(), experts.begin() + static_cast<std::ptrdiff_t>(keep), experts.end(),
            [&](uint32_t a, uint32_t b) {
                if (counts[a] != counts[b]) {
                    return counts[a] > counts[b];
                }
                return a < b;
            });

        auto &top = freq_top_by_layer_[layer];
        top.clear();
        for (size_t i = 0; i < keep; ++i) {
            if (counts[experts[i]] < static_cast<uint32_t>(min_count_)) {
                break;
            }
            top.push_back(layer * expert_count_ + experts[i]);
        }
    }

    uint32_t layer_count_ = 0;
    uint32_t expert_count_ = 0;
    int topk_ = 10;
    int min_count_ = 1;
    std::vector<std::vector<uint32_t>> recent_by_layer_;
    std::vector<std::vector<uint32_t>> freq_top_by_layer_;
    std::vector<std::vector<uint32_t>> counts_;
};

[[noreturn]] void fail(const std::string &msg) {
    throw std::runtime_error(msg);
}

void usage(const char *argv0) {
    std::cerr
        << "usage: " << argv0 << " --index IDX --sidecar FILE --trace TRACE [options]\n"
        << "\n"
        << "options:\n"
        << "  --budget-gib N     logical LRU cache budget, default 8\n"
        << "  --max-tokens N     stop after N unique trace tokens\n"
        << "  --prefetch-window N\n"
        << "                     oracle: future accesses; predictor policies: layers ahead, default 0\n"
        << "  --prefetch-policy NAME\n"
        << "                     oracle, candidates, last-layer, freq-layer, or hybrid; default oracle\n"
        << "  --prefetch-threads N\n"
        << "                     background pread worker count, default 1\n"
        << "  --predictor-topk N\n"
        << "                     predicted experts per layer for freq/hybrid policies, default 10\n"
        << "  --predictor-min-count N\n"
        << "                     minimum historical hits before an expert can be predicted, default 1\n"
        << "  --candidate-topk N\n"
        << "                     candidates to prefetch per routed layer when policy=candidates, default 10\n"
        << "  --candidate-delay-us N\n"
        << "                     sleep after candidate enqueue before expert demand, default 0\n"
        << "  --prefetch-max-inflight-mib N\n"
        << "                     cap queued/running prefetch bytes, default 512\n"
        << "  --compute-us-per-access N\n"
        << "                     sleep after each access to model compute overlap, default 0\n"
        << "  --nocache          request F_NOCACHE on the sidecar fd\n"
        << "  --no-read          simulate misses without pread\n"
        << "  --out PATH         write JSON summary\n";
}

Options parse_args(int argc, char **argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need_value = [&](const std::string &name) -> std::string {
            if (i + 1 >= argc) {
                fail("missing value for " + name);
            }
            return argv[++i];
        };
        if (a == "--index") {
            opt.index_path = need_value(a);
        } else if (a == "--sidecar") {
            opt.sidecar_path = need_value(a);
        } else if (a == "--trace") {
            opt.trace_path = need_value(a);
        } else if (a == "--budget-gib") {
            opt.budget_gib = std::stod(need_value(a));
        } else if (a == "--max-tokens") {
            opt.max_tokens = std::stoi(need_value(a));
        } else if (a == "--prefetch-policy") {
            opt.prefetch_policy = need_value(a);
        } else if (a == "--prefetch-window") {
            opt.prefetch_window = std::stoi(need_value(a));
        } else if (a == "--prefetch-threads") {
            opt.prefetch_threads = std::stoi(need_value(a));
        } else if (a == "--predictor-topk") {
            opt.predictor_topk = std::stoi(need_value(a));
        } else if (a == "--predictor-min-count") {
            opt.predictor_min_count = std::stoi(need_value(a));
        } else if (a == "--candidate-topk") {
            opt.candidate_topk = std::stoi(need_value(a));
        } else if (a == "--candidate-delay-us") {
            opt.candidate_delay_us = std::stoi(need_value(a));
        } else if (a == "--prefetch-max-inflight-mib") {
            opt.prefetch_max_inflight_mib = std::stod(need_value(a));
        } else if (a == "--compute-us-per-access") {
            opt.compute_us_per_access = std::stoi(need_value(a));
        } else if (a == "--out") {
            opt.out_path = need_value(a);
        } else if (a == "--nocache") {
            opt.nocache = true;
        } else if (a == "--no-read") {
            opt.no_read = true;
        } else if (a == "--help" || a == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            fail("unknown argument: " + a);
        }
    }

    if (opt.index_path.empty() || opt.sidecar_path.empty() || opt.trace_path.empty()) {
        usage(argv[0]);
        fail("missing required argument");
    }
    if (opt.budget_gib <= 0.0) {
        fail("--budget-gib must be positive");
    }
    if (opt.prefetch_window < 0) {
        fail("--prefetch-window must be >= 0");
    }
    if (opt.prefetch_threads <= 0) {
        fail("--prefetch-threads must be positive");
    }
    if (opt.predictor_topk <= 0) {
        fail("--predictor-topk must be positive");
    }
    if (opt.predictor_min_count <= 0) {
        fail("--predictor-min-count must be positive");
    }
    if (opt.candidate_topk <= 0) {
        fail("--candidate-topk must be positive");
    }
    if (opt.candidate_delay_us < 0) {
        fail("--candidate-delay-us must be >= 0");
    }
    if (opt.prefetch_policy != "oracle" && opt.prefetch_policy != "candidates" && opt.prefetch_policy != "last-layer" &&
        opt.prefetch_policy != "freq-layer" && opt.prefetch_policy != "hybrid") {
        fail("--prefetch-policy must be oracle, candidates, last-layer, freq-layer, or hybrid");
    }
    if (opt.compute_us_per_access < 0) {
        fail("--compute-us-per-access must be >= 0");
    }
    if (opt.prefetch_max_inflight_mib <= 0.0) {
        fail("--prefetch-max-inflight-mib must be positive");
    }
    return opt;
}

int open_readonly(const std::string &path, bool nocache) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        fail("open failed for " + path + ": " + std::strerror(errno));
    }
#if defined(__APPLE__)
    if (nocache) {
        (void)::fcntl(fd, F_NOCACHE, 1);
    }
#else
    (void)nocache;
#endif
    return fd;
}

std::vector<Entry> load_index(const std::string &path, Header &header) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        fail("could not open index: " + path);
    }
    in.read(reinterpret_cast<char *>(&header), sizeof(header));
    if (in.gcount() != static_cast<std::streamsize>(sizeof(header))) {
        fail("short index header");
    }
    if (std::memcmp(header.magic, "MOEVMIDX", 8) != 0) {
        fail("bad index magic");
    }
    if (header.version != 1) {
        fail("unsupported index version");
    }
    if (header.entry_count == 0 || header.layer_count == 0 || header.expert_count == 0) {
        fail("invalid empty index");
    }
    if (header.entry_count != header.layer_count * header.expert_count) {
        fail("index entry_count does not match layer_count * expert_count");
    }

    std::vector<Entry> entries(header.entry_count);
    in.read(reinterpret_cast<char *>(entries.data()), static_cast<std::streamsize>(entries.size() * sizeof(Entry)));
    if (in.gcount() != static_cast<std::streamsize>(entries.size() * sizeof(Entry))) {
        fail("short index entry table");
    }
    return entries;
}

bool parse_number_after(const std::string &line, const std::string &key, int64_t &value) {
    size_t p = line.find(key);
    if (p == std::string::npos) {
        return false;
    }
    p = line.find(':', p + key.size());
    if (p == std::string::npos) {
        return false;
    }
    ++p;
    while (p < line.size() && (line[p] == ' ' || line[p] == '\t')) {
        ++p;
    }
    bool neg = false;
    if (p < line.size() && line[p] == '-') {
        neg = true;
        ++p;
    }
    if (p >= line.size() || line[p] < '0' || line[p] > '9') {
        return false;
    }
    int64_t out = 0;
    while (p < line.size() && line[p] >= '0' && line[p] <= '9') {
        out = out * 10 + (line[p] - '0');
        ++p;
    }
    value = neg ? -out : out;
    return true;
}

bool parse_int_array_after(const std::string &line, const std::string &key, std::vector<int> &values) {
    values.clear();
    size_t p = line.find(key);
    if (p == std::string::npos) {
        return false;
    }
    p = line.find('[', p);
    if (p == std::string::npos) {
        return false;
    }
    ++p;
    while (p < line.size()) {
        while (p < line.size() && (line[p] == ' ' || line[p] == '\t' || line[p] == ',')) {
            ++p;
        }
        if (p >= line.size()) {
            return false;
        }
        if (line[p] == ']') {
            return true;
        }
        if (line[p] < '0' || line[p] > '9') {
            return false;
        }
        int value = 0;
        while (p < line.size() && line[p] >= '0' && line[p] <= '9') {
            value = value * 10 + (line[p] - '0');
            ++p;
        }
        values.push_back(value);
    }
    return false;
}

bool parse_experts(const std::string &line, std::vector<int> &experts) {
    return parse_int_array_after(line, "\"experts\"", experts);
}

bool parse_candidates(const std::string &line, std::vector<int> &candidates) {
    return parse_int_array_after(line, "\"candidates\"", candidates);
}

class LruCache {
  public:
    explicit LruCache(uint64_t capacity_bytes) : capacity_bytes_(capacity_bytes) {
        free_spans_.push_back({0, capacity_bytes_});
    }

    bool touch(uint32_t key) {
        auto it = items_.find(key);
        if (it != items_.end()) {
            lru_.splice(lru_.end(), lru_, it->second.where);
            return true;
        }
        return false;
    }

    bool contains(uint32_t key) const {
        return items_.find(key) != items_.end();
    }

    void insert(uint32_t key, uint32_t bytes) {
        auto it = items_.find(key);
        if (it != items_.end()) {
            lru_.splice(lru_.end(), lru_, it->second.where);
            return;
        }
        const uint64_t arena_bytes = align_up(bytes, kArenaAlignment);
        while (!has_free_span(arena_bytes) && !lru_.empty()) {
            evict_one();
        }
        if (!has_free_span(arena_bytes)) {
            fail("arena allocation failed: record is larger than available cache budget");
        }
        const uint64_t offset = allocate(arena_bytes);
        lru_.push_back(key);
        Item item;
        item.bytes = bytes;
        item.arena_bytes = arena_bytes;
        item.offset = offset;
        item.where = std::prev(lru_.end());
        items_.emplace(key, item);
        used_bytes_ += arena_bytes;
        high_water_bytes_ = std::max(high_water_bytes_, used_bytes_);
    }

    bool access(uint32_t key, uint32_t bytes) {
        if (touch(key)) {
            return true;
        }
        insert(key, bytes);
        return false;
    }

    uint64_t evictions() const {
        return evictions_;
    }

    uint64_t used_bytes() const {
        return used_bytes_;
    }

    uint64_t high_water_bytes() const {
        return high_water_bytes_;
    }

    uint64_t resident_records() const {
        return items_.size();
    }

    uint64_t free_span_count() const {
        return free_spans_.size();
    }

    uint64_t largest_free_span_bytes() const {
        uint64_t largest = 0;
        for (const Span &span : free_spans_) {
            largest = std::max(largest, span.bytes);
        }
        return largest;
    }

  private:
    static constexpr uint64_t kArenaAlignment = 16 * 1024;

    struct Span {
        uint64_t offset = 0;
        uint64_t bytes = 0;
    };

    struct Item {
        uint32_t bytes = 0;
        uint64_t arena_bytes = 0;
        uint64_t offset = 0;
        std::list<uint32_t>::iterator where;
    };

    static uint64_t align_up(uint64_t value, uint64_t align) {
        return ((value + align - 1) / align) * align;
    }

    bool has_free_span(uint64_t bytes) const {
        for (const Span &span : free_spans_) {
            if (span.bytes >= bytes) {
                return true;
            }
        }
        return false;
    }

    uint64_t allocate(uint64_t bytes) {
        for (size_t i = 0; i < free_spans_.size(); ++i) {
            Span &span = free_spans_[i];
            if (span.bytes < bytes) {
                continue;
            }
            const uint64_t offset = span.offset;
            span.offset += bytes;
            span.bytes -= bytes;
            if (span.bytes == 0) {
                free_spans_.erase(free_spans_.begin() + static_cast<std::ptrdiff_t>(i));
            }
            return offset;
        }
        fail("arena allocator internal error");
    }

    void release(uint64_t offset, uint64_t bytes) {
        Span span{offset, bytes};
        auto it = std::lower_bound(
            free_spans_.begin(), free_spans_.end(), span, [](const Span &a, const Span &b) {
                return a.offset < b.offset;
            });
        free_spans_.insert(it, span);
        merge_free_spans();
    }

    void merge_free_spans() {
        if (free_spans_.empty()) {
            return;
        }
        std::vector<Span> merged;
        merged.reserve(free_spans_.size());
        merged.push_back(free_spans_[0]);
        for (size_t i = 1; i < free_spans_.size(); ++i) {
            Span &last = merged.back();
            const Span &cur = free_spans_[i];
            if (last.offset + last.bytes == cur.offset) {
                last.bytes += cur.bytes;
            } else {
                merged.push_back(cur);
            }
        }
        free_spans_ = std::move(merged);
    }

    void evict_one() {
        if (lru_.empty()) {
            return;
        }
        uint32_t victim = lru_.front();
        lru_.pop_front();
        auto victim_it = items_.find(victim);
        if (victim_it == items_.end()) {
            return;
        }
        release(victim_it->second.offset, victim_it->second.arena_bytes);
        if (used_bytes_ >= victim_it->second.arena_bytes) {
            used_bytes_ -= victim_it->second.arena_bytes;
        } else {
            used_bytes_ = 0;
        }
        items_.erase(victim_it);
        ++evictions_;
    }

    uint64_t capacity_bytes_ = 0;
    uint64_t used_bytes_ = 0;
    uint64_t high_water_bytes_ = 0;
    uint64_t evictions_ = 0;
    std::list<uint32_t> lru_;
    std::unordered_map<uint32_t, Item> items_;
    std::vector<Span> free_spans_;
};

void read_exact(int fd, uint64_t offset, uint32_t bytes, std::vector<uint8_t> &buffer) {
    if (buffer.size() < bytes) {
        buffer.resize(bytes);
    }
    uint32_t done = 0;
    while (done < bytes) {
        ssize_t n = ::pread(fd, buffer.data() + done, bytes - done, static_cast<off_t>(offset + done));
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            fail(std::string("pread failed: ") + std::strerror(errno));
        }
        if (n == 0) {
            fail("short sidecar read");
        }
        done += static_cast<uint32_t>(n);
    }
}

class AsyncPrefetcher {
  public:
    enum class EnqueueResult { Queued, Dropped, Deduped };

    AsyncPrefetcher(int fd, bool no_read, uint64_t max_inflight_bytes, int thread_count)
        : fd_(fd), no_read_(no_read), max_inflight_bytes_(max_inflight_bytes), thread_count_(thread_count) {}

    ~AsyncPrefetcher() {
        stop();
    }

    void start() {
        for (int i = 0; i < thread_count_; ++i) {
            workers_.emplace_back([this] { worker_loop(); });
        }
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        for (std::thread &worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        workers_.clear();
    }

    EnqueueResult enqueue(uint32_t key, const Entry &entry) {
        std::lock_guard<std::mutex> lock(mu_);
        check_error_locked();
        if (queued_or_running_.count(key) || completed_.count(key)) {
            ++stats_.deduped;
            return EnqueueResult::Deduped;
        }
        if (inflight_bytes_ + entry.bytes > max_inflight_bytes_) {
            ++stats_.dropped;
            return EnqueueResult::Dropped;
        }
        Task task;
        task.key = key;
        task.entry = entry;
        queue_.push_back(task);
        queued_or_running_[key] = entry.bytes;
        inflight_bytes_ += entry.bytes;
        ++stats_.enqueued;
        cv_.notify_one();
        return EnqueueResult::Queued;
    }

    bool wait_for(uint32_t key, double &wait_s) {
        auto start = std::chrono::steady_clock::now();
        std::unique_lock<std::mutex> lock(mu_);
        check_error_locked();
        if (completed_.count(key)) {
            completed_.erase(key);
            erase_completed_order_locked(key);
            wait_s = 0.0;
            return true;
        }
        if (!queued_or_running_.count(key)) {
            wait_s = 0.0;
            return false;
        }
        cv_.wait(lock, [&] { return completed_.count(key) || !queued_or_running_.count(key) || !error_.empty(); });
        check_error_locked();
        if (!completed_.count(key)) {
            wait_s = 0.0;
            return false;
        }
        completed_.erase(key);
        erase_completed_order_locked(key);
        auto end = std::chrono::steady_clock::now();
        wait_s = std::chrono::duration<double>(end - start).count();
        return true;
    }

    std::vector<std::pair<uint32_t, uint32_t>> drain_completed() {
        std::lock_guard<std::mutex> lock(mu_);
        check_error_locked();
        std::vector<std::pair<uint32_t, uint32_t>> out;
        out.reserve(completed_order_.size());
        for (uint32_t key : completed_order_) {
            auto it = completed_.find(key);
            if (it != completed_.end()) {
                out.emplace_back(key, it->second);
            }
        }
        completed_order_.clear();
        completed_.clear();
        return out;
    }

    PrefetchStats stats() const {
        std::lock_guard<std::mutex> lock(mu_);
        return stats_;
    }

  private:
    struct Task {
        uint32_t key = 0;
        Entry entry{};
    };

    void check_error_locked() const {
        if (!error_.empty()) {
            fail(error_);
        }
    }

    void erase_completed_order_locked(uint32_t key) {
        auto it = std::find(completed_order_.begin(), completed_order_.end(), key);
        if (it != completed_order_.end()) {
            completed_order_.erase(it);
        }
    }

    void worker_loop() {
        std::vector<uint8_t> buffer;
        while (true) {
            Task task;
            {
                std::unique_lock<std::mutex> lock(mu_);
                cv_.wait(lock, [&] { return stop_ || !queue_.empty(); });
                if (stop_) {
                    return;
                }
                task = queue_.front();
                queue_.pop_front();
            }

            try {
                if (!no_read_) {
                    read_exact(fd_, task.entry.offset, task.entry.bytes, buffer);
                }
            } catch (const std::exception &e) {
                std::lock_guard<std::mutex> lock(mu_);
                error_ = e.what();
                stop_ = true;
                cv_.notify_all();
                return;
            }

            {
                std::lock_guard<std::mutex> lock(mu_);
                queued_or_running_.erase(task.key);
                if (inflight_bytes_ >= task.entry.bytes) {
                    inflight_bytes_ -= task.entry.bytes;
                } else {
                    inflight_bytes_ = 0;
                }
                completed_[task.key] = task.entry.bytes;
                completed_order_.push_back(task.key);
                ++stats_.completed;
                ++stats_.read_ops;
                stats_.read_bytes += task.entry.bytes;
            }
            cv_.notify_all();
        }
    }

    int fd_ = -1;
    bool no_read_ = false;
    uint64_t max_inflight_bytes_ = 0;
    int thread_count_ = 1;

    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::deque<Task> queue_;
    std::unordered_map<uint32_t, uint32_t> queued_or_running_;
    std::unordered_map<uint32_t, uint32_t> completed_;
    std::vector<uint32_t> completed_order_;
    uint64_t inflight_bytes_ = 0;
    bool stop_ = false;
    std::string error_;
    PrefetchStats stats_;
    std::vector<std::thread> workers_;
};

TraceData load_trace_data(const Options &opt, const Header &header, const std::vector<Entry> &entries) {
    std::ifstream trace(opt.trace_path);
    if (!trace) {
        fail("could not open trace: " + opt.trace_path);
    }

    TraceData data;
    std::unordered_set<int64_t> seen_tokens;
    std::vector<int> experts;
    std::vector<int> candidates;
    std::unordered_map<uint64_t, std::vector<uint32_t>> candidates_by_route;
    std::string line;
    while (std::getline(trace, line)) {
        if (line.empty()) {
            continue;
        }
        if (line.find("\"kind\":\"candidates\"") != std::string::npos ||
            line.find("\"kind\": \"candidates\"") != std::string::npos) {
            int64_t token = 0;
            int64_t layer = 0;
            if (!parse_number_after(line, "\"token\"", token) || !parse_number_after(line, "\"layer\"", layer) ||
                !parse_candidates(line, candidates)) {
                fail("could not parse candidate trace line: " + line);
            }
            if (layer < 0 || static_cast<uint64_t>(layer) >= header.layer_count) {
                continue;
            }
            std::vector<uint32_t> keys;
            keys.reserve(candidates.size());
            for (int expert : candidates) {
                if (expert < 0 || static_cast<uint64_t>(expert) >= header.expert_count) {
                    continue;
                }
                const uint32_t key = static_cast<uint32_t>(layer) * header.expert_count + static_cast<uint32_t>(expert);
                if (entries[key].bytes != 0) {
                    keys.push_back(key);
                }
            }
            candidates_by_route[route_key(token, layer)] = std::move(keys);
            ++data.candidate_records;
            continue;
        }
        int64_t token = 0;
        int64_t layer = 0;
        if (!parse_number_after(line, "\"token\"", token) || !parse_number_after(line, "\"layer\"", layer) ||
            !parse_experts(line, experts)) {
            fail("could not parse trace line: " + line);
        }
        if (!seen_tokens.count(token)) {
            if (opt.max_tokens >= 0 && static_cast<int>(seen_tokens.size()) >= opt.max_tokens) {
                break;
            }
            seen_tokens.insert(token);
        }
        if (layer < 0 || static_cast<uint64_t>(layer) >= header.layer_count) {
            data.missing_index += experts.size();
            continue;
        }
        for (int expert : experts) {
            if (expert < 0 || static_cast<uint64_t>(expert) >= header.expert_count) {
                ++data.missing_index;
                continue;
            }
            uint32_t key = static_cast<uint32_t>(layer) * header.expert_count + static_cast<uint32_t>(expert);
            if (entries[key].bytes == 0) {
                ++data.missing_index;
                continue;
            }
            Access access;
            access.token = token;
            access.key = key;
            if (data.accesses.empty() || data.accesses.back().token != token || data.accesses.back().key / header.expert_count != static_cast<uint32_t>(layer)) {
                auto cand_it = candidates_by_route.find(route_key(token, layer));
                if (cand_it != candidates_by_route.end()) {
                    const size_t keep = std::min(cand_it->second.size(), static_cast<size_t>(opt.candidate_topk));
                    access.candidates.assign(cand_it->second.begin(), cand_it->second.begin() + static_cast<std::ptrdiff_t>(keep));
                }
            }
            data.accesses.push_back(access);
        }
    }
    data.tokens = seen_tokens.size();
    return data;
}

Metrics replay(const Options &opt, const Header &header, const std::vector<Entry> &entries, int sidecar_fd) {
    const uint64_t budget_bytes = static_cast<uint64_t>(opt.budget_gib * 1024.0 * 1024.0 * 1024.0);
    LruCache cache(budget_bytes);
    TraceData trace = load_trace_data(opt, header, entries);

    Metrics m;
    m.missing_index = trace.missing_index;
    m.candidate_records = trace.candidate_records;
    std::vector<uint8_t> io_buffer;
    PredictorState predictor(header.layer_count, header.expert_count, opt.predictor_topk, opt.predictor_min_count);
    const uint64_t max_inflight_bytes =
        static_cast<uint64_t>(opt.prefetch_max_inflight_mib * 1024.0 * 1024.0);
    std::unique_ptr<AsyncPrefetcher> prefetcher;
    if (opt.prefetch_window > 0 || opt.prefetch_policy == "candidates") {
        prefetcher.reset(new AsyncPrefetcher(sidecar_fd, opt.no_read, max_inflight_bytes, opt.prefetch_threads));
        prefetcher->start();
    }

    auto start = std::chrono::steady_clock::now();
    for (size_t i = 0; i < trace.accesses.size(); ++i) {
        const Access &access = trace.accesses[i];
        const Entry &entry = entries[access.key];

        if (prefetcher) {
            for (const auto &done : prefetcher->drain_completed()) {
                cache.insert(done.first, done.second);
            }
        }

        if (prefetcher && opt.prefetch_policy == "candidates" && !access.candidates.empty()) {
            for (uint32_t candidate_key : access.candidates) {
                if (cache.contains(candidate_key)) {
                    continue;
                }
                (void)prefetcher->enqueue(candidate_key, entries[candidate_key]);
                ++m.candidate_prefetch_keys;
            }
            if (opt.candidate_delay_us > 0) {
                std::this_thread::sleep_for(std::chrono::microseconds(opt.candidate_delay_us));
                for (const auto &done : prefetcher->drain_completed()) {
                    cache.insert(done.first, done.second);
                }
            }
        }

        ++m.accesses;
        if (cache.touch(access.key)) {
            ++m.hits;
        } else {
            ++m.misses;
            bool satisfied_by_prefetch = false;
            if (prefetcher) {
                double wait_s = 0.0;
                if (prefetcher->wait_for(access.key, wait_s)) {
                    satisfied_by_prefetch = true;
                    ++m.prefetch_waits;
                    m.prefetch_wait_bytes += entry.bytes;
                    m.prefetch_wait_s += wait_s;
                    cache.insert(access.key, entry.bytes);
                }
            }
            if (!satisfied_by_prefetch) {
                if (!opt.no_read) {
                    read_exact(sidecar_fd, entry.offset, entry.bytes, io_buffer);
                }
                ++m.demand_read_ops;
                m.demand_read_bytes += entry.bytes;
                cache.insert(access.key, entry.bytes);
            }
        }

        if (prefetcher && opt.prefetch_policy == "oracle") {
            const size_t end = std::min(trace.accesses.size(), i + 1 + static_cast<size_t>(opt.prefetch_window));
            for (size_t j = i + 1; j < end; ++j) {
                uint32_t future_key = trace.accesses[j].key;
                if (cache.contains(future_key)) {
                    continue;
                }
                (void)prefetcher->enqueue(future_key, entries[future_key]);
            }
        }

        predictor.observe(access.key);

        if (prefetcher && opt.prefetch_policy != "oracle" && opt.prefetch_policy != "candidates") {
            const uint32_t current_layer = access.key / header.expert_count;
            for (int ahead = 1; ahead <= opt.prefetch_window; ++ahead) {
                const uint32_t layer = (current_layer + static_cast<uint32_t>(ahead)) % header.layer_count;
                for (uint32_t predicted_key : predictor.predict(opt.prefetch_policy, layer)) {
                    if (cache.contains(predicted_key)) {
                        continue;
                    }
                    (void)prefetcher->enqueue(predicted_key, entries[predicted_key]);
                }
            }
        }

        if (opt.compute_us_per_access > 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(opt.compute_us_per_access));
        }
    }
    if (prefetcher) {
        for (const auto &done : prefetcher->drain_completed()) {
            cache.insert(done.first, done.second);
        }
        PrefetchStats stats = prefetcher->stats();
        m.prefetch_enqueued = stats.enqueued;
        m.prefetch_dropped = stats.dropped;
        m.prefetch_deduped = stats.deduped;
        m.prefetch_completed = stats.completed;
        m.prefetch_read_ops = stats.read_ops;
        m.prefetch_read_bytes = stats.read_bytes;
        prefetcher->stop();
    }
    auto end = std::chrono::steady_clock::now();
    m.tokens = trace.tokens;
    m.read_ops = m.demand_read_ops + m.prefetch_read_ops;
    m.read_bytes = m.demand_read_bytes + m.prefetch_read_bytes;
    m.arena_evictions = cache.evictions();
    m.arena_used_bytes = cache.used_bytes();
    m.arena_high_water_bytes = cache.high_water_bytes();
    m.arena_resident_records = cache.resident_records();
    m.arena_free_spans = cache.free_span_count();
    m.arena_largest_free_span_bytes = cache.largest_free_span_bytes();
    m.elapsed_s = std::chrono::duration<double>(end - start).count();
    return m;
}

std::string json_escape(const std::string &s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        if (c == '"' || c == '\\') {
            out.push_back('\\');
            out.push_back(c);
        } else if (c == '\n') {
            out += "\\n";
        } else {
            out.push_back(c);
        }
    }
    return out;
}

std::string to_json(const Options &opt, const Header &header, const Metrics &m) {
    const double hit_rate = m.accesses ? static_cast<double>(m.hits) / static_cast<double>(m.accesses) : 0.0;
    const double read_gib = static_cast<double>(m.read_bytes) / (1024.0 * 1024.0 * 1024.0);
    const double read_gbps = m.elapsed_s > 0.0 ? static_cast<double>(m.read_bytes) / m.elapsed_s / 1e9 : 0.0;
    const double tok_s = m.elapsed_s > 0.0 ? static_cast<double>(m.tokens) / m.elapsed_s : 0.0;

    std::ostringstream os;
    os << std::fixed << std::setprecision(6);
    os << "{\n";
    os << "  \"index\": \"" << json_escape(opt.index_path) << "\",\n";
    os << "  \"sidecar\": \"" << json_escape(opt.sidecar_path) << "\",\n";
    os << "  \"trace\": \"" << json_escape(opt.trace_path) << "\",\n";
    os << "  \"budget_gib\": " << opt.budget_gib << ",\n";
    os << "  \"max_tokens\": " << opt.max_tokens << ",\n";
    os << "  \"prefetch_policy\": \"" << json_escape(opt.prefetch_policy) << "\",\n";
    os << "  \"prefetch_window_accesses\": " << opt.prefetch_window << ",\n";
    os << "  \"prefetch_threads\": " << opt.prefetch_threads << ",\n";
    os << "  \"predictor_topk\": " << opt.predictor_topk << ",\n";
    os << "  \"predictor_min_count\": " << opt.predictor_min_count << ",\n";
    os << "  \"candidate_topk\": " << opt.candidate_topk << ",\n";
    os << "  \"candidate_delay_us\": " << opt.candidate_delay_us << ",\n";
    os << "  \"prefetch_max_inflight_mib\": " << opt.prefetch_max_inflight_mib << ",\n";
    os << "  \"compute_us_per_access\": " << opt.compute_us_per_access << ",\n";
    os << "  \"layers\": " << header.layer_count << ",\n";
    os << "  \"experts\": " << header.expert_count << ",\n";
    os << "  \"tokens\": " << m.tokens << ",\n";
    os << "  \"accesses\": " << m.accesses << ",\n";
    os << "  \"hits\": " << m.hits << ",\n";
    os << "  \"misses\": " << m.misses << ",\n";
    os << "  \"hit_rate\": " << hit_rate << ",\n";
    os << "  \"missing_index\": " << m.missing_index << ",\n";
    os << "  \"read_ops\": " << m.read_ops << ",\n";
    os << "  \"read_bytes\": " << m.read_bytes << ",\n";
    os << "  \"demand_read_ops\": " << m.demand_read_ops << ",\n";
    os << "  \"demand_read_bytes\": " << m.demand_read_bytes << ",\n";
    os << "  \"prefetch_enqueued\": " << m.prefetch_enqueued << ",\n";
    os << "  \"prefetch_dropped\": " << m.prefetch_dropped << ",\n";
    os << "  \"prefetch_deduped\": " << m.prefetch_deduped << ",\n";
    os << "  \"prefetch_completed\": " << m.prefetch_completed << ",\n";
    os << "  \"prefetch_read_ops\": " << m.prefetch_read_ops << ",\n";
    os << "  \"prefetch_read_bytes\": " << m.prefetch_read_bytes << ",\n";
    os << "  \"prefetch_waits\": " << m.prefetch_waits << ",\n";
    os << "  \"prefetch_wait_bytes\": " << m.prefetch_wait_bytes << ",\n";
    os << "  \"prefetch_wait_s\": " << m.prefetch_wait_s << ",\n";
    os << "  \"candidate_records\": " << m.candidate_records << ",\n";
    os << "  \"candidate_prefetch_keys\": " << m.candidate_prefetch_keys << ",\n";
    os << "  \"arena_evictions\": " << m.arena_evictions << ",\n";
    os << "  \"arena_used_bytes\": " << m.arena_used_bytes << ",\n";
    os << "  \"arena_high_water_bytes\": " << m.arena_high_water_bytes << ",\n";
    os << "  \"arena_resident_records\": " << m.arena_resident_records << ",\n";
    os << "  \"arena_free_spans\": " << m.arena_free_spans << ",\n";
    os << "  \"arena_largest_free_span_bytes\": " << m.arena_largest_free_span_bytes << ",\n";
    os << "  \"read_gib\": " << read_gib << ",\n";
    os << "  \"elapsed_s\": " << m.elapsed_s << ",\n";
    os << "  \"read_gbps_decimal\": " << read_gbps << ",\n";
    os << "  \"tok_s_replay\": " << tok_s << ",\n";
    os << "  \"nocache\": " << (opt.nocache ? "true" : "false") << ",\n";
    os << "  \"no_read\": " << (opt.no_read ? "true" : "false") << "\n";
    os << "}\n";
    return os.str();
}

} // namespace

int main(int argc, char **argv) {
    try {
        Options opt = parse_args(argc, argv);
        Header header{};
        std::vector<Entry> entries = load_index(opt.index_path, header);
        int sidecar_fd = open_readonly(opt.sidecar_path, opt.nocache);
        Metrics m;
        try {
            m = replay(opt, header, entries, sidecar_fd);
        } catch (...) {
            ::close(sidecar_fd);
            throw;
        }
        ::close(sidecar_fd);

        std::string json = to_json(opt, header, m);
        if (!opt.out_path.empty()) {
            std::ofstream out(opt.out_path);
            if (!out) {
                fail("could not write output: " + opt.out_path);
            }
            out << json;
        }
        std::cout << json;
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
