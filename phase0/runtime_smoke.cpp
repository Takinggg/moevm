#include "moevm_runtime.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string index_path;
    std::string component_index_path;
    std::string sidecar_path;
    std::string trace_path;
    uint64_t arena_mib = 64;
    uint64_t inflight_mib = 16;
    uint32_t threads = 2;
    size_t candidate_topk = 10;
    bool verify_integrity = true;
};

[[noreturn]] void fail(const std::string &msg) {
    throw std::runtime_error(msg);
}

void usage(const char *argv0) {
    std::cerr
        << "usage: " << argv0 << " --index IDX --sidecar FILE --trace TRACE [options]\n"
        << "options:\n"
        << "  --component-index IDX optional gate/up/down component index\n"
        << "  --arena-mib N       default 64\n"
        << "  --inflight-mib N    default 16\n"
        << "  --threads N         default 2\n"
        << "  --candidate-topk N  default 10\n"
        << "  --no-integrity-check verify sidecar size only, not SHA prefix\n";
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
        } else if (a == "--component-index") {
            opt.component_index_path = need(a);
        } else if (a == "--sidecar") {
            opt.sidecar_path = need(a);
        } else if (a == "--trace") {
            opt.trace_path = need(a);
        } else if (a == "--arena-mib") {
            opt.arena_mib = std::stoull(need(a));
        } else if (a == "--inflight-mib") {
            opt.inflight_mib = std::stoull(need(a));
        } else if (a == "--threads") {
            opt.threads = static_cast<uint32_t>(std::stoul(need(a)));
        } else if (a == "--candidate-topk") {
            opt.candidate_topk = static_cast<size_t>(std::stoul(need(a)));
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
    int64_t out = 0;
    if (p >= line.size() || line[p] < '0' || line[p] > '9') {
        return false;
    }
    while (p < line.size() && line[p] >= '0' && line[p] <= '9') {
        out = out * 10 + (line[p] - '0');
        ++p;
    }
    value = out;
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

struct FirstRoute {
    uint32_t layer = 0;
    std::vector<uint32_t> candidates;
    std::vector<uint32_t> experts;
};

FirstRoute load_first_route(const std::string &path, size_t candidate_topk) {
    std::ifstream in(path);
    if (!in) {
        fail("could not open trace: " + path);
    }
    std::string line;
    int64_t candidate_token = -1;
    int64_t candidate_layer = -1;
    std::vector<uint32_t> candidates;
    while (std::getline(in, line)) {
        if (line.find("\"kind\":\"candidates\"") != std::string::npos ||
            line.find("\"kind\": \"candidates\"") != std::string::npos) {
            if (!parse_number_after(line, "\"token\"", candidate_token) ||
                !parse_number_after(line, "\"layer\"", candidate_layer) ||
                !parse_int_array_after(line, "\"candidates\"", candidates)) {
                fail("could not parse candidate line");
            }
            if (candidates.size() > candidate_topk) {
                candidates.resize(candidate_topk);
            }
            continue;
        }
        if (candidate_token < 0) {
            continue;
        }
        int64_t token = -1;
        int64_t layer = -1;
        std::vector<uint32_t> experts;
        if (!parse_number_after(line, "\"token\"", token) ||
            !parse_number_after(line, "\"layer\"", layer) ||
            !parse_int_array_after(line, "\"experts\"", experts)) {
            continue;
        }
        if (token == candidate_token && layer == candidate_layer) {
            FirstRoute out;
            out.layer = static_cast<uint32_t>(layer);
            out.candidates = std::move(candidates);
            out.experts = std::move(experts);
            return out;
        }
    }
    fail("trace did not contain a candidate/topk pair");
}

} // namespace

int main(int argc, char **argv) {
    try {
        Options opt = parse_args(argc, argv);
        FirstRoute route = load_first_route(opt.trace_path, opt.candidate_topk);

        moevm::RuntimeOptions runtime_opt;
        runtime_opt.index_path = opt.index_path;
        runtime_opt.component_index_path = opt.component_index_path;
        runtime_opt.sidecar_path = opt.sidecar_path;
        runtime_opt.arena_bytes = opt.arena_mib * 1024ull * 1024ull;
        runtime_opt.max_inflight_bytes = opt.inflight_mib * 1024ull * 1024ull;
        runtime_opt.prefetch_threads = opt.threads;
        runtime_opt.nocache = true;
        runtime_opt.verify_integrity = opt.verify_integrity;

        moevm::Runtime runtime(runtime_opt);
        runtime.start();
        runtime.prefetch_candidates(route.layer, route.candidates.data(), route.candidates.size());
        uint64_t selected_bytes = 0;
        uint64_t component_bytes = 0;
        uint64_t prefetched_selected = 0;
        for (uint32_t expert : route.experts) {
            moevm::RecordView view = runtime.wait_selected(route.layer, expert);
            selected_bytes += view.bytes;
            if (!opt.component_index_path.empty()) {
                component_bytes += runtime.component_view(view, moevm::ExpertComponent::Gate).bytes;
                component_bytes += runtime.component_view(view, moevm::ExpertComponent::Up).bytes;
                component_bytes += runtime.component_view(view, moevm::ExpertComponent::Down).bytes;
            }
            if (view.came_from_prefetch) {
                ++prefetched_selected;
            }
            runtime.release_record(view);
        }
        runtime.stop();

        moevm::RuntimeMetrics m = runtime.metrics();
        std::cout
            << "{\n"
            << "  \"layer\": " << route.layer << ",\n"
            << "  \"candidate_count\": " << route.candidates.size() << ",\n"
            << "  \"selected_count\": " << route.experts.size() << ",\n"
            << "  \"selected_bytes\": " << selected_bytes << ",\n"
            << "  \"component_bytes\": " << component_bytes << ",\n"
            << "  \"prefetched_selected\": " << prefetched_selected << ",\n"
            << "  \"prefetch_enqueued\": " << m.prefetch_enqueued << ",\n"
            << "  \"prefetch_completed\": " << m.prefetch_completed << ",\n"
            << "  \"prefetch_dropped\": " << m.prefetch_dropped << ",\n"
            << "  \"prefetch_deduped\": " << m.prefetch_deduped << ",\n"
            << "  \"prefetch_waits\": " << m.prefetch_waits << ",\n"
            << "  \"prefetch_wait_s\": " << m.prefetch_wait_s << ",\n"
            << "  \"demand_read_ops\": " << m.demand_read_ops << ",\n"
            << "  \"demand_read_bytes\": " << m.demand_read_bytes << ",\n"
            << "  \"hits\": " << m.hits << ",\n"
            << "  \"misses\": " << m.misses << ",\n"
            << "  \"evictions\": " << m.evictions << ",\n"
            << "  \"arena_high_water_bytes\": " << m.arena_high_water_bytes << "\n"
            << "}\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
