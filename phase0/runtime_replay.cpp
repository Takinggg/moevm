#include "moevm_runtime.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

struct Options {
    std::string index_path;
    std::string sidecar_path;
    std::string trace_path;
    std::string out_path;
    double budget_gib = 8.0;
    double inflight_mib = 64.0;
    uint32_t threads = 4;
    size_t candidate_topk = 10;
    int max_tokens = -1;
    double trim_gib_after_replay = -1.0;
    double hold_seconds_after_replay = 0.0;
    bool nocache = true;
    bool verify_integrity = true;
};

struct ReplayMetrics {
    uint64_t tokens = 0;
    uint64_t records = 0;
    uint64_t candidate_records = 0;
    uint64_t accesses = 0;
    uint64_t selected_bytes = 0;
    uint64_t missing_candidates = 0;
    uint64_t missing_index = 0;
    uint64_t trim_evictions_after_replay = 0;
    double elapsed_s = 0.0;
};

[[noreturn]] void fail(const std::string &msg) {
    throw std::runtime_error(msg);
}

void usage(const char *argv0) {
    std::cerr
        << "usage: " << argv0 << " --index IDX --sidecar FILE --trace TRACE [options]\n"
        << "options:\n"
        << "  --budget-gib N                  default 8\n"
        << "  --prefetch-max-inflight-mib N   default 64\n"
        << "  --prefetch-threads N            default 4\n"
        << "  --candidate-topk N              default 10\n"
        << "  --max-tokens N                  default all\n"
        << "  --trim-gib-after-replay N       evict arena down to this used size before metrics\n"
        << "  --hold-seconds-after-replay N   keep process alive after replay for footprint inspection\n"
        << "  --no-nocache                    do not request F_NOCACHE\n"
        << "  --no-integrity-check            verify sidecar size only, not SHA prefix\n"
        << "  --out PATH                      write JSON summary\n";
}

Options parse_args(int argc, char **argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const std::string &name) -> std::string {
            if (i + 1 >= argc) {
                fail("missing value for " + name);
            }
            return argv[++i];
        };
        if (a == "--index") {
            opt.index_path = need(a);
        } else if (a == "--sidecar") {
            opt.sidecar_path = need(a);
        } else if (a == "--trace") {
            opt.trace_path = need(a);
        } else if (a == "--budget-gib") {
            opt.budget_gib = std::stod(need(a));
        } else if (a == "--prefetch-max-inflight-mib") {
            opt.inflight_mib = std::stod(need(a));
        } else if (a == "--prefetch-threads") {
            opt.threads = static_cast<uint32_t>(std::stoul(need(a)));
        } else if (a == "--candidate-topk") {
            opt.candidate_topk = static_cast<size_t>(std::stoul(need(a)));
        } else if (a == "--max-tokens") {
            opt.max_tokens = std::stoi(need(a));
        } else if (a == "--trim-gib-after-replay") {
            opt.trim_gib_after_replay = std::stod(need(a));
        } else if (a == "--hold-seconds-after-replay") {
            opt.hold_seconds_after_replay = std::stod(need(a));
        } else if (a == "--out") {
            opt.out_path = need(a);
        } else if (a == "--no-nocache") {
            opt.nocache = false;
        } else if (a == "--no-integrity-check") {
            opt.verify_integrity = false;
        } else if (a == "-h" || a == "--help") {
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
    if (opt.budget_gib <= 0.0 || opt.inflight_mib <= 0.0 || opt.threads == 0 || opt.candidate_topk == 0) {
        fail("numeric options must be positive");
    }
    return opt;
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

bool parse_int_array_after(const std::string &line, const std::string &key, std::vector<uint32_t> &values) {
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
        uint32_t value = 0;
        while (p < line.size() && line[p] >= '0' && line[p] <= '9') {
            value = value * 10 + static_cast<uint32_t>(line[p] - '0');
            ++p;
        }
        values.push_back(value);
    }
    return false;
}

uint64_t route_key(int64_t token, int64_t layer) {
    return (static_cast<uint64_t>(token) << 32) | static_cast<uint32_t>(layer);
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

std::string to_json(const Options &opt, const ReplayMetrics &r, const moevm::RuntimeMetrics &m) {
    const double tok_s = r.elapsed_s > 0.0 ? static_cast<double>(r.tokens) / r.elapsed_s : 0.0;
    const uint64_t read_bytes = m.demand_read_bytes + m.prefetch_read_bytes;
    const double read_gib = static_cast<double>(read_bytes) / (1024.0 * 1024.0 * 1024.0);
    const double read_gbps = r.elapsed_s > 0.0 ? static_cast<double>(read_bytes) / r.elapsed_s / 1e9 : 0.0;
    const uint64_t runtime_accesses = m.hits + m.misses;
    const double hit_rate = runtime_accesses ? static_cast<double>(m.hits) / static_cast<double>(runtime_accesses) : 0.0;
    const double demand_miss_rate = r.accesses ? static_cast<double>(m.demand_read_ops) / static_cast<double>(r.accesses) : 0.0;
    const double prefetch_wait_rate = r.accesses ? static_cast<double>(m.prefetch_waits) / static_cast<double>(r.accesses) : 0.0;

    std::ostringstream os;
    os << std::fixed << std::setprecision(6);
    os << "{\n";
    os << "  \"index\": \"" << json_escape(opt.index_path) << "\",\n";
    os << "  \"sidecar\": \"" << json_escape(opt.sidecar_path) << "\",\n";
    os << "  \"trace\": \"" << json_escape(opt.trace_path) << "\",\n";
    os << "  \"budget_gib\": " << opt.budget_gib << ",\n";
    os << "  \"prefetch_threads\": " << opt.threads << ",\n";
    os << "  \"candidate_topk\": " << opt.candidate_topk << ",\n";
    os << "  \"prefetch_max_inflight_mib\": " << opt.inflight_mib << ",\n";
    os << "  \"max_tokens\": " << opt.max_tokens << ",\n";
    os << "  \"tokens\": " << r.tokens << ",\n";
    os << "  \"records\": " << r.records << ",\n";
    os << "  \"candidate_records\": " << r.candidate_records << ",\n";
    os << "  \"accesses\": " << r.accesses << ",\n";
    os << "  \"selected_bytes\": " << r.selected_bytes << ",\n";
    os << "  \"missing_candidates\": " << r.missing_candidates << ",\n";
    os << "  \"missing_index\": " << r.missing_index << ",\n";
    os << "  \"trim_evictions_after_replay\": " << r.trim_evictions_after_replay << ",\n";
    os << "  \"runtime_hits\": " << m.hits << ",\n";
    os << "  \"runtime_misses\": " << m.misses << ",\n";
    os << "  \"runtime_hit_rate\": " << hit_rate << ",\n";
    os << "  \"demand_miss_rate\": " << demand_miss_rate << ",\n";
    os << "  \"prefetch_wait_rate\": " << prefetch_wait_rate << ",\n";
    os << "  \"demand_read_ops\": " << m.demand_read_ops << ",\n";
    os << "  \"demand_read_bytes\": " << m.demand_read_bytes << ",\n";
    os << "  \"prefetch_enqueued\": " << m.prefetch_enqueued << ",\n";
    os << "  \"prefetch_dropped\": " << m.prefetch_dropped << ",\n";
    os << "  \"prefetch_deduped\": " << m.prefetch_deduped << ",\n";
    os << "  \"prefetch_completed\": " << m.prefetch_completed << ",\n";
    os << "  \"prefetch_read_ops\": " << m.prefetch_read_ops << ",\n";
    os << "  \"prefetch_read_bytes\": " << m.prefetch_read_bytes << ",\n";
    os << "  \"prefetch_waits\": " << m.prefetch_waits << ",\n";
    os << "  \"prefetch_wait_s\": " << m.prefetch_wait_s << ",\n";
    os << "  \"read_ops\": " << (m.demand_read_ops + m.prefetch_read_ops) << ",\n";
    os << "  \"read_bytes\": " << read_bytes << ",\n";
    os << "  \"read_gib\": " << read_gib << ",\n";
    os << "  \"read_gbps_decimal\": " << read_gbps << ",\n";
    os << "  \"evictions\": " << m.evictions << ",\n";
    os << "  \"arena_used_bytes\": " << m.arena_used_bytes << ",\n";
    os << "  \"arena_high_water_bytes\": " << m.arena_high_water_bytes << ",\n";
    os << "  \"arena_resident_records\": " << m.arena_resident_records << ",\n";
    os << "  \"arena_free_spans\": " << m.arena_free_spans << ",\n";
    os << "  \"arena_largest_free_span_bytes\": " << m.arena_largest_free_span_bytes << ",\n";
    os << "  \"arena_reclaim_advises\": " << m.arena_reclaim_advises << ",\n";
    os << "  \"arena_reclaim_bytes\": " << m.arena_reclaim_bytes << ",\n";
    os << "  \"arena_reclaim_live_bytes\": " << m.arena_reclaim_live_bytes << ",\n";
    os << "  \"arena_reuse_advises\": " << m.arena_reuse_advises << ",\n";
    os << "  \"arena_reuse_bytes\": " << m.arena_reuse_bytes << ",\n";
    os << "  \"elapsed_s\": " << r.elapsed_s << ",\n";
    os << "  \"tok_s_replay\": " << tok_s << ",\n";
    os << "  \"nocache\": " << (opt.nocache ? "true" : "false") << "\n";
    os << "}\n";
    return os.str();
}

ReplayMetrics run_replay(const Options &opt, moevm::Runtime &runtime) {
    std::ifstream trace(opt.trace_path);
    if (!trace) {
        fail("could not open trace: " + opt.trace_path);
    }

    ReplayMetrics metrics;
    std::unordered_map<uint64_t, std::vector<uint32_t>> candidates_by_route;
    std::unordered_set<int64_t> seen_tokens;
    std::vector<uint32_t> values;
    std::string line;

    runtime.start();
    const auto start = std::chrono::steady_clock::now();
    while (std::getline(trace, line)) {
        if (line.empty()) {
            continue;
        }
        if (line.find("\"kind\":\"candidates\"") != std::string::npos ||
            line.find("\"kind\": \"candidates\"") != std::string::npos) {
            int64_t token = 0;
            int64_t layer = 0;
            if (!parse_number_after(line, "\"token\"", token) ||
                !parse_number_after(line, "\"layer\"", layer) ||
                !parse_int_array_after(line, "\"candidates\"", values)) {
                fail("could not parse candidate trace line");
            }
            if (values.size() > opt.candidate_topk) {
                values.resize(opt.candidate_topk);
            }
            candidates_by_route[route_key(token, layer)] = values;
            ++metrics.candidate_records;
            continue;
        }

        int64_t token = 0;
        int64_t layer = 0;
        if (!parse_number_after(line, "\"token\"", token) ||
            !parse_number_after(line, "\"layer\"", layer) ||
            !parse_int_array_after(line, "\"experts\"", values)) {
            fail("could not parse topk trace line");
        }

        if (!seen_tokens.count(token)) {
            if (opt.max_tokens >= 0 && static_cast<int>(seen_tokens.size()) >= opt.max_tokens) {
                break;
            }
            seen_tokens.insert(token);
        }

        ++metrics.records;
        const auto candidate_it = candidates_by_route.find(route_key(token, layer));
        if (candidate_it != candidates_by_route.end()) {
            runtime.prefetch_candidates(static_cast<uint32_t>(layer), candidate_it->second.data(), candidate_it->second.size());
            candidates_by_route.erase(candidate_it);
        } else {
            ++metrics.missing_candidates;
        }

        for (uint32_t expert : values) {
            moevm::RecordView view = runtime.wait_selected(static_cast<uint32_t>(layer), expert);
            metrics.selected_bytes += view.bytes;
            ++metrics.accesses;
            runtime.release_record(view);
        }
    }
    runtime.stop();
    const auto end = std::chrono::steady_clock::now();
    metrics.tokens = seen_tokens.size();
    metrics.elapsed_s = std::chrono::duration<double>(end - start).count();
    return metrics;
}

} // namespace

int main(int argc, char **argv) {
    try {
        Options opt = parse_args(argc, argv);
        moevm::RuntimeOptions runtime_opt;
        runtime_opt.index_path = opt.index_path;
        runtime_opt.sidecar_path = opt.sidecar_path;
        runtime_opt.arena_bytes = static_cast<uint64_t>(opt.budget_gib * 1024.0 * 1024.0 * 1024.0);
        runtime_opt.max_inflight_bytes = static_cast<uint64_t>(opt.inflight_mib * 1024.0 * 1024.0);
        runtime_opt.prefetch_threads = opt.threads;
        runtime_opt.nocache = opt.nocache;
        runtime_opt.verify_integrity = opt.verify_integrity;

        moevm::Runtime runtime(runtime_opt);
        ReplayMetrics replay_metrics = run_replay(opt, runtime);
        if (opt.trim_gib_after_replay >= 0.0) {
            const uint64_t target = static_cast<uint64_t>(opt.trim_gib_after_replay * 1024.0 * 1024.0 * 1024.0);
            replay_metrics.trim_evictions_after_replay = runtime.trim_to_bytes(target);
        }
        moevm::RuntimeMetrics runtime_metrics = runtime.metrics();
        std::string json = to_json(opt, replay_metrics, runtime_metrics);
        if (!opt.out_path.empty()) {
            std::ofstream out(opt.out_path);
            if (!out) {
                fail("could not write output: " + opt.out_path);
            }
            out << json;
        }
        std::cout << json;
        std::cout.flush();
        if (opt.hold_seconds_after_replay > 0.0) {
            std::this_thread::sleep_for(std::chrono::duration<double>(opt.hold_seconds_after_replay));
        }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
