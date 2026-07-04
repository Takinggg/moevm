#include "moevm_runtime.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <fstream>
#include <iterator>
#include <list>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(__APPLE__)
#include <CommonCrypto/CommonDigest.h>
#ifndef F_NOCACHE
#define F_NOCACHE 48
#endif
#endif

namespace moevm {
namespace {

struct IndexHeader {
    char magic[8];
    uint32_t version;
    uint32_t entry_count;
    uint32_t layer_count;
    uint32_t expert_count;
    uint64_t reserved[4];
};

struct IndexEntry {
    uint64_t offset;
    uint32_t bytes;
    uint32_t reserved;
};

struct ComponentIndexHeader {
    char magic[8];
    uint32_t version;
    uint32_t entry_count;
    uint32_t layer_count;
    uint32_t expert_count;
    uint32_t component_count;
    uint32_t reserved0;
    uint64_t reserved[3];
};

struct ComponentSlice {
    uint32_t rel_offset;
    uint32_t bytes;
};

struct ComponentIndexEntry {
    ComponentSlice components[3];
};

static_assert(sizeof(IndexHeader) == 56, "unexpected IndexHeader packing");
static_assert(sizeof(IndexEntry) == 16, "unexpected IndexEntry packing");
static_assert(sizeof(ComponentIndexHeader) == 56, "unexpected ComponentIndexHeader packing");
static_assert(sizeof(ComponentIndexEntry) == 24, "unexpected ComponentIndexEntry packing");

[[noreturn]] void fail(const std::string &msg) {
    throw std::runtime_error(msg);
}

uint64_t align_up(uint64_t value, uint64_t align) {
    return ((value + align - 1) / align) * align;
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

void read_exact_into(int fd, uint64_t offset, uint32_t bytes, uint8_t *dst) {
    uint32_t done = 0;
    while (done < bytes) {
        ssize_t n = ::pread(fd, dst + done, bytes - done, static_cast<off_t>(offset + done));
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

uint64_t file_size(int fd) {
    struct stat st {};
    if (::fstat(fd, &st) != 0) {
        fail(std::string("fstat failed: ") + std::strerror(errno));
    }
    if (st.st_size < 0) {
        fail("negative file size");
    }
    return static_cast<uint64_t>(st.st_size);
}

bool any_nonzero(const uint64_t * values, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (values[i] != 0) {
            return true;
        }
    }
    return false;
}

uint64_t load_le64(const uint8_t * data) {
    uint64_t out = 0;
    for (int i = 7; i >= 0; --i) {
        out = (out << 8) | data[i];
    }
    return out;
}

std::vector<uint64_t> sha256_prefix_words(int fd, uint64_t expected_size, size_t word_count) {
#if defined(__APPLE__)
    CC_SHA256_CTX ctx;
    CC_SHA256_Init(&ctx);
    std::vector<uint8_t> buffer(8 * 1024 * 1024);
    uint64_t offset = 0;
    while (offset < expected_size) {
        const size_t want = static_cast<size_t>(std::min<uint64_t>(buffer.size(), expected_size - offset));
        ssize_t n = ::pread(fd, buffer.data(), want, static_cast<off_t>(offset));
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            fail(std::string("sidecar hash read failed: ") + std::strerror(errno));
        }
        if (n == 0) {
            fail("sidecar hash read was short");
        }
        CC_SHA256_Update(&ctx, buffer.data(), static_cast<CC_LONG>(n));
        offset += static_cast<uint64_t>(n);
    }
    uint8_t digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256_Final(digest, &ctx);
    std::vector<uint64_t> words;
    words.reserve(word_count);
    for (size_t i = 0; i < word_count; ++i) {
        words.push_back(load_le64(digest + i * sizeof(uint64_t)));
    }
    return words;
#else
    (void)fd;
    (void)expected_size;
    (void)word_count;
    fail("sidecar integrity hash checking is only implemented on Apple builds");
#endif
}

} // namespace

class Runtime::Impl {
  public:
    explicit Impl(RuntimeOptions options) : options_(std::move(options)) {
        if (options_.index_path.empty() || options_.sidecar_path.empty()) {
            fail("RuntimeOptions requires index_path and sidecar_path");
        }
        if (options_.arena_bytes == 0) {
            fail("arena_bytes must be positive");
        }
        if (options_.max_inflight_bytes == 0) {
            fail("max_inflight_bytes must be positive");
        }
        if (options_.prefetch_threads == 0) {
            fail("prefetch_threads must be positive");
        }
        load_index();
        if (!options_.component_index_path.empty()) {
            load_component_index();
        }
        sidecar_fd_ = open_readonly(options_.sidecar_path, options_.nocache);
        if (options_.verify_integrity) {
            verify_sidecar_integrity();
        } else {
            verify_sidecar_size_only();
        }
        if (options_.direct_mmap) {
            sidecar_mmap_bytes_ = file_size(sidecar_fd_);
            void * mapped = ::mmap(nullptr, static_cast<size_t>(sidecar_mmap_bytes_), PROT_READ, MAP_PRIVATE, sidecar_fd_, 0);
            if (mapped == MAP_FAILED) {
                fail(std::string("mmap failed for sidecar: ") + std::strerror(errno));
            }
            sidecar_mmap_ = static_cast<const uint8_t *>(mapped);
        } else {
            arena_.resize(static_cast<size_t>(options_.arena_bytes));
            free_spans_.push_back({0, options_.arena_bytes});
        }
    }

    ~Impl() {
        stop();
        if (sidecar_mmap_ != nullptr) {
            ::munmap(const_cast<uint8_t *>(sidecar_mmap_), static_cast<size_t>(sidecar_mmap_bytes_));
            sidecar_mmap_ = nullptr;
            sidecar_mmap_bytes_ = 0;
        }
        if (sidecar_fd_ >= 0) {
            ::close(sidecar_fd_);
            sidecar_fd_ = -1;
        }
    }

    void start() {
        std::lock_guard<std::mutex> lock(mu_);
        if (started_) {
            return;
        }
        started_ = true;
        stop_ = false;
        if (options_.direct_mmap) {
            return;
        }
        for (uint32_t i = 0; i < options_.prefetch_threads; ++i) {
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
        started_ = false;
    }

    uint32_t layer_count() const {
        return header_.layer_count;
    }

    uint32_t expert_count() const {
        return header_.expert_count;
    }

    void prefetch_candidates(uint32_t layer, const uint32_t *experts, size_t count) {
        if (experts == nullptr && count != 0) {
            fail("prefetch_candidates got null experts pointer");
        }
        std::lock_guard<std::mutex> lock(mu_);
        check_error_locked();
        if (!started_) {
            fail("Runtime::start must be called before prefetch_candidates");
        }
        for (size_t i = 0; i < count; ++i) {
            if (layer >= header_.layer_count || experts[i] >= header_.expert_count) {
                ++metrics_.prefetch_dropped;
                continue;
            }
            const uint32_t key = flat_key(layer, experts[i]);
            const IndexEntry &entry = entries_[key];
            if (entry.bytes == 0) {
                ++metrics_.prefetch_dropped;
                continue;
            }
            if (options_.direct_mmap) {
                if (entry.offset + entry.bytes <= sidecar_mmap_bytes_) {
#if defined(MADV_WILLNEED)
                    (void)::madvise(const_cast<uint8_t *>(sidecar_mmap_ + entry.offset), entry.bytes, MADV_WILLNEED);
#endif
                    ++metrics_.prefetch_enqueued;
                    ++metrics_.prefetch_completed;
                } else {
                    ++metrics_.prefetch_dropped;
                }
                continue;
            }
            if (states_.count(key)) {
                ++metrics_.prefetch_deduped;
                continue;
            }
            if (inflight_bytes_ + entry.bytes > options_.max_inflight_bytes) {
                ++metrics_.prefetch_dropped;
                continue;
            }

            Slot slot;
            if (!allocate_slot_locked(key, entry.bytes, true, slot)) {
                ++metrics_.prefetch_dropped;
                continue;
            }

            State state;
            state.status = Status::Loading;
            state.offset = slot.offset;
            state.bytes = entry.bytes;
            state.arena_bytes = slot.arena_bytes;
            state.from_prefetch = true;
            states_.emplace(key, state);

            Task task;
            task.key = key;
            task.entry = entry;
            task.offset = slot.offset;
            queue_.push_back(task);
            inflight_bytes_ += entry.bytes;
            ++metrics_.prefetch_enqueued;
            cv_.notify_one();
        }
    }

    RecordView wait_selected(uint32_t layer, uint32_t expert) {
        if (layer >= header_.layer_count || expert >= header_.expert_count) {
            fail("selected expert key out of range");
        }
        const uint32_t key = flat_key(layer, expert);
        const IndexEntry &entry = entries_[key];
        if (entry.bytes == 0) {
            fail("selected expert key is missing from index");
        }

        if (options_.direct_mmap) {
            if (entry.offset + entry.bytes > sidecar_mmap_bytes_) {
                fail("selected expert sidecar range is outside mmap");
            }
            uint64_t generation = 0;
            {
                std::lock_guard<std::mutex> lock(mu_);
                ++metrics_.hits;
                generation = ++generation_;
            }
            RecordView view;
            view.data = sidecar_mmap_ + entry.offset;
            view.bytes = entry.bytes;
            view.layer = layer;
            view.expert = expert;
            view.generation = generation;
            view.came_from_prefetch = true;
            return view;
        }

        std::unique_lock<std::mutex> lock(mu_);
        check_error_locked();

        while (true) {
            auto state_it = states_.find(key);
            if (state_it == states_.end()) {
                break;
            }
            State &state = state_it->second;
            if (state.status == Status::Resident) {
                touch_locked(state);
                ++metrics_.hits;
                return make_view_locked(key, state);
            }

            const auto start = std::chrono::steady_clock::now();
            cv_.wait(lock, [&] {
                auto it = states_.find(key);
                return !error_.empty() || it == states_.end() || it->second.status == Status::Resident;
            });
            check_error_locked();
            const auto end = std::chrono::steady_clock::now();
            ++metrics_.prefetch_waits;
            metrics_.prefetch_wait_s += std::chrono::duration<double>(end - start).count();
        }

        ++metrics_.misses;
        Slot slot;
        if (!allocate_slot_locked(key, entry.bytes, true, slot)) {
            fail("arena allocation failed for selected expert");
        }

        State loading;
        loading.status = Status::Loading;
        loading.offset = slot.offset;
        loading.bytes = entry.bytes;
        loading.arena_bytes = slot.arena_bytes;
        loading.from_prefetch = false;
        states_.emplace(key, loading);

        lock.unlock();
        try {
            inject_delay(static_cast<uint64_t>(key) ^ 0xd0e0a0dull);
            read_exact_into(sidecar_fd_, entry.offset, entry.bytes, arena_.data() + slot.offset);
        } catch (...) {
            lock.lock();
            error_ = "demand read failed";
            cv_.notify_all();
            throw;
        }
        lock.lock();

        auto state_it = states_.find(key);
        if (state_it == states_.end()) {
            fail("selected expert state disappeared during demand read");
        }
        State &state = state_it->second;
        state.status = Status::Resident;
        state.generation = ++generation_;
        lru_.push_back(key);
        state.where = std::prev(lru_.end());
        ++metrics_.demand_read_ops;
        metrics_.demand_read_bytes += entry.bytes;
        cv_.notify_all();
        return make_view_locked(key, state);
    }

    ComponentView component_view(const RecordView &record, ExpertComponent component) const {
        if (component_entries_.empty()) {
            fail("component index was not loaded");
        }
        if (record.data == nullptr) {
            fail("record view is empty");
        }
        if (record.layer >= header_.layer_count || record.expert >= header_.expert_count) {
            fail("record view key out of range");
        }
        const uint32_t idx = flat_key(record.layer, record.expert);
        const uint32_t component_idx = static_cast<uint32_t>(component);
        if (component_idx >= 3) {
            fail("invalid expert component");
        }
        const ComponentSlice &slice = component_entries_[idx].components[component_idx];
        if (static_cast<uint64_t>(slice.rel_offset) + slice.bytes > record.bytes) {
            fail("component slice is outside record view");
        }
        ComponentView out;
        out.data = record.data + slice.rel_offset;
        out.bytes = slice.bytes;
        out.component = component;
        return out;
    }

    bool contains(uint32_t layer, uint32_t expert) const {
        if (layer >= header_.layer_count || expert >= header_.expert_count) {
            return false;
        }
        if (options_.direct_mmap) {
            return entries_[flat_key(layer, expert)].bytes != 0;
        }
        const uint32_t key = flat_key(layer, expert);
        std::lock_guard<std::mutex> lock(mu_);
        auto it = states_.find(key);
        return it != states_.end() && it->second.status == Status::Resident;
    }

    RuntimeMetrics metrics() const {
        std::lock_guard<std::mutex> lock(mu_);
        RuntimeMetrics out = metrics_;
        out.delay_injections = delay_injections_.load(std::memory_order_relaxed);
        out.delay_us = delay_us_.load(std::memory_order_relaxed);
        if (options_.direct_mmap) {
            return out;
        }
        out.arena_used_bytes = arena_used_bytes_;
        out.arena_high_water_bytes = arena_high_water_bytes_;
        out.arena_resident_records = lru_.size();
        out.arena_free_spans = free_spans_.size();
        out.arena_largest_free_span_bytes = largest_free_span_locked();
        return out;
    }

  private:
    static constexpr uint64_t kArenaAlignment = 16 * 1024;

    enum class Status { Loading, Resident };

    struct Span {
        uint64_t offset = 0;
        uint64_t bytes = 0;
    };

    struct Slot {
        uint64_t offset = 0;
        uint64_t arena_bytes = 0;
    };

    struct State {
        Status status = Status::Loading;
        uint64_t offset = 0;
        uint32_t bytes = 0;
        uint64_t arena_bytes = 0;
        uint64_t generation = 0;
        uint32_t pin_count = 0;
        bool from_prefetch = false;
        std::list<uint32_t>::iterator where;
    };

    struct Task {
        uint32_t key = 0;
        IndexEntry entry{};
        uint64_t offset = 0;
    };

    uint32_t flat_key(uint32_t layer, uint32_t expert) const {
        return layer * header_.expert_count + expert;
    }

    void load_index() {
        std::ifstream in(options_.index_path, std::ios::binary);
        if (!in) {
            fail("could not open index: " + options_.index_path);
        }
        in.read(reinterpret_cast<char *>(&header_), sizeof(header_));
        if (in.gcount() != static_cast<std::streamsize>(sizeof(header_))) {
            fail("short index header");
        }
        if (std::memcmp(header_.magic, "MOEVMIDX", 8) != 0) {
            fail("bad index magic");
        }
        if (header_.version != 1) {
            fail("unsupported index version");
        }
        if (header_.entry_count != header_.layer_count * header_.expert_count) {
            fail("index entry_count does not match layer_count * expert_count");
        }
        index_sidecar_size_ = header_.reserved[0];
        index_sidecar_hash_words_[0] = header_.reserved[1];
        index_sidecar_hash_words_[1] = header_.reserved[2];
        index_sidecar_hash_words_[2] = header_.reserved[3];
        index_has_integrity_ = index_sidecar_size_ != 0 || any_nonzero(index_sidecar_hash_words_, 3);
        if (index_has_integrity_ && (index_sidecar_size_ == 0 || !any_nonzero(index_sidecar_hash_words_, 3))) {
            fail("index sidecar integrity fields are incomplete");
        }
        entries_.resize(header_.entry_count);
        in.read(reinterpret_cast<char *>(entries_.data()), static_cast<std::streamsize>(entries_.size() * sizeof(IndexEntry)));
        if (in.gcount() != static_cast<std::streamsize>(entries_.size() * sizeof(IndexEntry))) {
            fail("short index entry table");
        }
    }

    void load_component_index() {
        std::ifstream in(options_.component_index_path, std::ios::binary);
        if (!in) {
            fail("could not open component index: " + options_.component_index_path);
        }
        ComponentIndexHeader header{};
        in.read(reinterpret_cast<char *>(&header), sizeof(header));
        if (in.gcount() != static_cast<std::streamsize>(sizeof(header))) {
            fail("short component index header");
        }
        if (std::memcmp(header.magic, "MOEVMCIX", 8) != 0) {
            fail("bad component index magic");
        }
        if (header.version != 1 || header.component_count != 3) {
            fail("unsupported component index version");
        }
        if (header.layer_count != header_.layer_count || header.expert_count != header_.expert_count ||
            header.entry_count != header_.entry_count) {
            fail("component index shape does not match main index");
        }
        component_sidecar_size_ = header.reserved[0];
        component_sidecar_hash_words_[0] = header.reserved[1];
        component_sidecar_hash_words_[1] = header.reserved[2];
        component_has_integrity_ = component_sidecar_size_ != 0 || any_nonzero(component_sidecar_hash_words_, 2);
        if (component_has_integrity_ && (component_sidecar_size_ == 0 || !any_nonzero(component_sidecar_hash_words_, 2))) {
            fail("component index sidecar integrity fields are incomplete");
        }
        component_entries_.resize(header.entry_count);
        in.read(reinterpret_cast<char *>(component_entries_.data()),
                static_cast<std::streamsize>(component_entries_.size() * sizeof(ComponentIndexEntry)));
        if (in.gcount() != static_cast<std::streamsize>(component_entries_.size() * sizeof(ComponentIndexEntry))) {
            fail("short component index entry table");
        }
    }

    void verify_sidecar_integrity() const {
        const bool need_index_hash = index_has_integrity_;
        const bool need_component_hash = component_has_integrity_;
        if (!need_index_hash && !need_component_hash) {
            return;
        }

        const uint64_t actual_size = file_size(sidecar_fd_);
        if (need_index_hash && actual_size != index_sidecar_size_) {
            fail("sidecar size does not match index integrity header");
        }
        if (need_component_hash && actual_size != component_sidecar_size_) {
            fail("sidecar size does not match component index integrity header");
        }

        const size_t word_count = need_index_hash ? 3 : 2;
        const std::vector<uint64_t> actual_hash = sha256_prefix_words(sidecar_fd_, actual_size, word_count);
        if (need_index_hash) {
            for (size_t i = 0; i < 3; ++i) {
                if (actual_hash[i] != index_sidecar_hash_words_[i]) {
                    fail("sidecar sha256 prefix does not match index integrity header");
                }
            }
        }
        if (need_component_hash) {
            for (size_t i = 0; i < 2; ++i) {
                if (actual_hash[i] != component_sidecar_hash_words_[i]) {
                    fail("sidecar sha256 prefix does not match component index integrity header");
                }
            }
        }
    }

    void verify_sidecar_size_only() const {
        const uint64_t actual_size = file_size(sidecar_fd_);
        if (index_has_integrity_ && actual_size != index_sidecar_size_) {
            fail("sidecar size does not match index integrity header");
        }
        if (component_has_integrity_ && actual_size != component_sidecar_size_) {
            fail("sidecar size does not match component index integrity header");
        }
    }

    void check_error_locked() const {
        if (!error_.empty()) {
            fail(error_);
        }
    }

    bool has_free_span_locked(uint64_t bytes) const {
        for (const Span &span : free_spans_) {
            if (span.bytes >= bytes) {
                return true;
            }
        }
        return false;
    }

    bool allocate_slot_locked(uint32_t key, uint32_t bytes, bool allow_evict, Slot &slot) {
        (void)key;
        const uint64_t arena_bytes = align_up(bytes, kArenaAlignment);
        if (arena_bytes > options_.arena_bytes) {
            return false;
        }
        while (!has_free_span_locked(arena_bytes) && allow_evict && !lru_.empty()) {
            if (!evict_one_locked()) {
                break;
            }
        }
        if (!has_free_span_locked(arena_bytes)) {
            return false;
        }
        for (size_t i = 0; i < free_spans_.size(); ++i) {
            Span &span = free_spans_[i];
            if (span.bytes < arena_bytes) {
                continue;
            }
            slot.offset = span.offset;
            slot.arena_bytes = arena_bytes;
            span.offset += arena_bytes;
            span.bytes -= arena_bytes;
            if (span.bytes == 0) {
                free_spans_.erase(free_spans_.begin() + static_cast<std::ptrdiff_t>(i));
            }
            arena_used_bytes_ += arena_bytes;
            arena_high_water_bytes_ = std::max(arena_high_water_bytes_, arena_used_bytes_);
            return true;
        }
        return false;
    }

    void release_slot_locked(uint64_t offset, uint64_t bytes) {
        Span span{offset, bytes};
        auto it = std::lower_bound(free_spans_.begin(), free_spans_.end(), span, [](const Span &a, const Span &b) {
            return a.offset < b.offset;
        });
        free_spans_.insert(it, span);
        merge_free_spans_locked();
    }

    void merge_free_spans_locked() {
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

    bool evict_one_locked() {
        if (lru_.empty()) {
            return false;
        }

        auto victim_it = lru_.end();
        auto state_it = states_.end();
        for (auto it = lru_.begin(); it != lru_.end(); ++it) {
            auto candidate = states_.find(*it);
            if (candidate == states_.end()) {
                victim_it = it;
                break;
            }
            if (candidate->second.pin_count == 0) {
                victim_it = it;
                state_it = candidate;
                break;
            }
        }
        if (victim_it == lru_.end()) {
            return false;
        }

        const uint32_t victim = *victim_it;
        lru_.erase(victim_it);
        if (state_it == states_.end()) {
            state_it = states_.find(victim);
        }
        if (state_it == states_.end()) {
            return true;
        }
        release_slot_locked(state_it->second.offset, state_it->second.arena_bytes);
        if (arena_used_bytes_ >= state_it->second.arena_bytes) {
            arena_used_bytes_ -= state_it->second.arena_bytes;
        } else {
            arena_used_bytes_ = 0;
        }
        states_.erase(state_it);
        ++metrics_.evictions;
        return true;
    }

    void touch_locked(State &state) {
        lru_.splice(lru_.end(), lru_, state.where);
        state.where = std::prev(lru_.end());
    }

    RecordView make_view_locked(uint32_t key, State &state) const {
        ++state.pin_count;
        RecordView view;
        view.data = arena_.data() + state.offset;
        view.bytes = state.bytes;
        view.layer = key / header_.expert_count;
        view.expert = key % header_.expert_count;
        view.generation = state.generation;
        view.came_from_prefetch = state.from_prefetch;
        return view;
    }

  public:
    void release_record(const RecordView &record) {
        if (record.data == nullptr || options_.direct_mmap) {
            return;
        }
        if (record.layer >= header_.layer_count || record.expert >= header_.expert_count) {
            fail("record release key out of range");
        }
        const uint32_t key = flat_key(record.layer, record.expert);
        std::lock_guard<std::mutex> lock(mu_);
        auto state_it = states_.find(key);
        if (state_it == states_.end() || state_it->second.generation != record.generation) {
            fail("record release after eviction or generation mismatch");
        }
        if (state_it->second.pin_count == 0) {
            fail("record release without matching pin");
        }
        --state_it->second.pin_count;
    }

  private:
    uint64_t largest_free_span_locked() const {
        uint64_t largest = 0;
        for (const Span &span : free_spans_) {
            largest = std::max(largest, span.bytes);
        }
        return largest;
    }

    void worker_loop() {
        while (true) {
            Task task;
            {
                std::unique_lock<std::mutex> lock(mu_);
                cv_.wait(lock, [&] { return stop_ || !queue_.empty(); });
                if (stop_ && queue_.empty()) {
                    return;
                }
                if (queue_.empty()) {
                    continue;
                }
                task = queue_.front();
                queue_.pop_front();
            }

            try {
                inject_delay(static_cast<uint64_t>(task.key) ^ 0x51dec4full);
                read_exact_into(sidecar_fd_, task.entry.offset, task.entry.bytes, arena_.data() + task.offset);
            } catch (const std::exception &e) {
                std::lock_guard<std::mutex> lock(mu_);
                error_ = e.what();
                cv_.notify_all();
                return;
            }

            {
                std::lock_guard<std::mutex> lock(mu_);
                if (inflight_bytes_ >= task.entry.bytes) {
                    inflight_bytes_ -= task.entry.bytes;
                } else {
                    inflight_bytes_ = 0;
                }
                auto state_it = states_.find(task.key);
                if (state_it != states_.end() && state_it->second.status == Status::Loading) {
                    State &state = state_it->second;
                    state.status = Status::Resident;
                    state.generation = ++generation_;
                    lru_.push_back(task.key);
                    state.where = std::prev(lru_.end());
                    ++metrics_.prefetch_completed;
                    ++metrics_.prefetch_read_ops;
                    metrics_.prefetch_read_bytes += task.entry.bytes;
                }
            }
            cv_.notify_all();
        }
    }

    RuntimeOptions options_;
    IndexHeader header_{};
    std::vector<IndexEntry> entries_;
    std::vector<ComponentIndexEntry> component_entries_;
    bool index_has_integrity_ = false;
    uint64_t index_sidecar_size_ = 0;
    uint64_t index_sidecar_hash_words_[3] = {0, 0, 0};
    bool component_has_integrity_ = false;
    uint64_t component_sidecar_size_ = 0;
    uint64_t component_sidecar_hash_words_[2] = {0, 0};
    int sidecar_fd_ = -1;
    const uint8_t *sidecar_mmap_ = nullptr;
    uint64_t sidecar_mmap_bytes_ = 0;
    std::vector<uint8_t> arena_;

    mutable std::mutex mu_;
    std::condition_variable cv_;
    bool started_ = false;
    bool stop_ = false;
    std::string error_;
    std::deque<Task> queue_;
    std::vector<std::thread> workers_;
    uint64_t inflight_bytes_ = 0;
    uint64_t generation_ = 0;

    std::vector<Span> free_spans_;
    uint64_t arena_used_bytes_ = 0;
    uint64_t arena_high_water_bytes_ = 0;
    std::list<uint32_t> lru_;
    std::unordered_map<uint32_t, State> states_;
    RuntimeMetrics metrics_;
    std::atomic<uint64_t> delay_counter_{0};
    std::atomic<uint64_t> delay_injections_{0};
    std::atomic<uint64_t> delay_us_{0};

    void inject_delay(uint64_t salt) {
        if (options_.inject_delay_max_us == 0) {
            return;
        }
        uint64_t x = options_.inject_delay_seed ^ salt ^
            (delay_counter_.fetch_add(1, std::memory_order_relaxed) + 0x9e3779b97f4a7c15ull);
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        const uint64_t us = (x * 2685821657736338717ull) % ((uint64_t) options_.inject_delay_max_us + 1);
        if (us == 0) {
            return;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(us));
        delay_injections_.fetch_add(1, std::memory_order_relaxed);
        delay_us_.fetch_add(us, std::memory_order_relaxed);
    }
};

Runtime::Runtime(RuntimeOptions options) : impl_(new Impl(std::move(options))) {}

Runtime::~Runtime() = default;

void Runtime::start() {
    impl_->start();
}

void Runtime::stop() {
    impl_->stop();
}

uint32_t Runtime::layer_count() const {
    return impl_->layer_count();
}

uint32_t Runtime::expert_count() const {
    return impl_->expert_count();
}

void Runtime::prefetch_candidates(uint32_t layer, const uint32_t *experts, size_t count) {
    impl_->prefetch_candidates(layer, experts, count);
}

RecordView Runtime::wait_selected(uint32_t layer, uint32_t expert) {
    return impl_->wait_selected(layer, expert);
}

void Runtime::release_record(const RecordView &record) {
    impl_->release_record(record);
}

ComponentView Runtime::component_view(const RecordView &record, ExpertComponent component) const {
    return impl_->component_view(record, component);
}

bool Runtime::contains(uint32_t layer, uint32_t expert) const {
    return impl_->contains(layer, expert);
}

RuntimeMetrics Runtime::metrics() const {
    return impl_->metrics();
}

} // namespace moevm
