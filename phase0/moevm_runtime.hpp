#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace moevm {

struct RuntimeOptions {
    std::string index_path;
    std::string component_index_path;
    std::string sidecar_path;
    uint64_t arena_bytes = 8ull * 1024ull * 1024ull * 1024ull;
    uint64_t max_inflight_bytes = 64ull * 1024ull * 1024ull;
    uint32_t prefetch_threads = 4;
    bool nocache = true;
    bool direct_mmap = false;
    bool verify_integrity = true;
    uint32_t inject_delay_max_us = 0;
    uint64_t inject_delay_seed = 0x9e3779b97f4a7c15ull;
};

enum class ExpertComponent : uint32_t {
    Gate = 0,
    Up = 1,
    Down = 2,
};

struct RecordKey {
    uint32_t layer = 0;
    uint32_t expert = 0;
};

struct RecordView {
    const uint8_t *data = nullptr;
    uint32_t bytes = 0;
    uint32_t layer = 0;
    uint32_t expert = 0;
    uint64_t generation = 0;
    bool came_from_prefetch = false;
};

struct ComponentView {
    const uint8_t *data = nullptr;
    uint32_t bytes = 0;
    ExpertComponent component = ExpertComponent::Gate;
};

struct RuntimeMetrics {
    uint64_t prefetch_enqueued = 0;
    uint64_t prefetch_dropped = 0;
    uint64_t prefetch_deduped = 0;
    uint64_t prefetch_completed = 0;
    uint64_t prefetch_read_ops = 0;
    uint64_t prefetch_read_bytes = 0;
    uint64_t prefetch_waits = 0;
    double prefetch_wait_s = 0.0;
    uint64_t demand_read_ops = 0;
    uint64_t demand_read_bytes = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t evictions = 0;
    uint64_t arena_used_bytes = 0;
    uint64_t arena_high_water_bytes = 0;
    uint64_t arena_resident_records = 0;
    uint64_t arena_free_spans = 0;
    uint64_t arena_largest_free_span_bytes = 0;
    uint64_t arena_reclaim_advises = 0;
    uint64_t arena_reclaim_bytes = 0;
    uint64_t arena_reclaim_live_bytes = 0;
    uint64_t arena_reuse_advises = 0;
    uint64_t arena_reuse_bytes = 0;
    uint64_t delay_injections = 0;
    uint64_t delay_us = 0;
};

class Runtime {
  public:
    explicit Runtime(RuntimeOptions options);
    ~Runtime();

    Runtime(const Runtime &) = delete;
    Runtime &operator=(const Runtime &) = delete;

    void start();
    void stop();

    uint32_t layer_count() const;
    uint32_t expert_count() const;

    void prefetch_candidates(uint32_t layer, const uint32_t *experts, size_t count);
    RecordView wait_selected(uint32_t layer, uint32_t expert);
    void release_record(const RecordView &record);
    ComponentView component_view(const RecordView &record, ExpertComponent component) const;
    bool contains(uint32_t layer, uint32_t expert) const;
    uint64_t trim_to_bytes(uint64_t target_used_bytes);

    RuntimeMetrics metrics() const;

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace moevm
