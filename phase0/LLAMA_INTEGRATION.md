# MoEVM llama.cpp Integration Boundary

This note captures the current integration target for Qwen3-Coder-Next Q3 on the M5 Pro 48GB.

## Current Runtime Artifacts

- Packed sidecar: `$MOEVM_MODELS_DIR/qwen3-coder-next/qwen3-next-q3.moevm`
- Main index: `$MOEVM_MODELS_DIR/qwen3-coder-next/qwen3-next-q3.moevm.idx`
- Component index: `$MOEVM_MODELS_DIR/qwen3-coder-next/qwen3-next-q3.moevm.cidx`
- Runtime API: `moevm/phase0/moevm_runtime.hpp`
- Runtime implementation: `moevm/phase0/moevm_runtime.cpp`
- llama.cpp hook patch:
  - `external/llama.cpp/ggml/include/ggml-backend.h`
  - `external/llama.cpp/ggml/src/ggml-backend.cpp`
- MoEVM-enabled local example:
  - `external/llama.cpp/examples/moe-trace/moe-trace.cpp`
  - `external/llama.cpp/examples/moe-trace/CMakeLists.txt`

The runtime API now supports:

- `prefetch_candidates(layer, experts, count)`
- `wait_selected(layer, expert)`
- `component_view(record, ExpertComponent::Gate/Up/Down)`

The component index is needed because the sidecar packs one `(layer, expert)` record, while llama.cpp consumes separate routed tensors.

## llama.cpp Points

Router/top-k construction is in:

- `external/llama.cpp/src/llama-graph.cpp`
- Function: `llm_graph_context::build_moe_ffn`
- Candidate/top-k tensors:
  - `ffn_moe_argsort-<layer>`
  - `ffn_moe_topk-<layer>`

Qwen3Next calls MoE through:

- `external/llama.cpp/src/models/qwen3next.cpp`
- Function: `llama_model_qwen3next::graph::build_layer_ffn`

The selected expert matmuls are created by:

- `build_lora_mm_id(...)`
- `ggml_mul_mat_id(ctx, weights, cur, selected_experts)`

The existing weight-copy optimization is in:

- `external/llama.cpp/ggml/src/ggml-backend.cpp`
- Branch comment: `when offloading MoE weights, we can reduce the amount of data copied by copying only the experts that are used`

That branch already:

1. Detects `GGML_OP_MUL_MAT_ID`.
2. Reads the selected expert IDs tensor.
3. Builds a bitset of used experts.
4. Copies only selected expert slices from host weights to the split backend.

This is the first replacement point for MoEVM.

## Implemented Hook

`ggml-backend` now exposes:

```c
ggml_backend_set_moe_weight_copy_callback(callback, user_data);
```

The callback receives:

- destination backend
- destination tensor copy
- source expert tensor
- sorted unique selected expert IDs
- expert slice size

If the callback returns `true`, the normal llama.cpp selected-expert copy is skipped. If no callback is registered or it returns `false`, llama.cpp falls back to the original code path.

`llama-moe-trace` now accepts:

- `--moevm-index`
- `--moevm-component-index`
- `--moevm-sidecar`
- `--moevm-arena-gib`
- `--moevm-inflight-mib`
- `--moevm-threads`
- `--moevm-prefetch-topk`
- `--moevm-copy-debug`
- `--moevm-copy-mode async|memcpy`
- `--moevm-cache`
- `--moevm-nocache`
- `--moevm-direct-mmap`
- `--moevm-shadow-ids`
- `--moevm-compact-layer <layer|all|-1>`
- `--moevm-compact-k-max`

Its eval callback prefetches candidates/top-k IDs through `moevm::Runtime`. Its weight-copy hook parses routed tensor names and maps:

- `blk.<layer>.ffn_gate_exps.weight` -> `ExpertComponent::Gate`
- `blk.<layer>.ffn_up_exps.weight` -> `ExpertComponent::Up`
- `blk.<layer>.ffn_down_exps.weight` -> `ExpertComponent::Down`

Then it copies the resident component view into the backend tensor copy.

Copy modes:

- `async`: original experimental path using `ggml_backend_tensor_set_async`.
- `memcpy`: preferred on this Apple Silicon setup when Metal uses shared buffers; it writes directly to `dst_tensor->data + expert_offset` after bounds checks.

The hook also prefetches the selected expert set it receives from llama.cpp before waiting on each expert. This is important because scheduler input copies happen before the eval callback can see same-graph router tensors.

## Smoke Results

CPU-only Qwen3Next Q3 MoEVM smoke:

- Command log: `moevm/runs/qwen3_next_q3_llama_moevm_cpu_smoke.log`
- Trace: `moevm/runs/qwen3_next_q3_llama_moevm_cpu_smoke.jsonl`
- Mode: `-ngl 0 --no-repack`, 1GiB MoEVM arena, 64MiB in-flight, 4 prefetch threads.
- Prompt tokens: 10.
- Trace records: 940.
- MoEVM prefetch records: 942.
- Runtime demand reads: 0.
- Runtime prefetch reads: 2,116.
- Runtime waits: 0.
- Arena high-water: 1.00GiB.
- Copy hook calls: 0, expected for CPU-only because no backend offload copy branch runs.

Limited Metal/offload probe:

- Command log: `moevm/runs/qwen3_next_q3_llama_moevm_ngl4_smoke.log`
- Mode: `-ngl 4`, 1GiB MoEVM arena.
- Result: stopped by `safe_run.py` before decode.
- Stop reason: wired memory exceeded 34GiB.
- No trace records were produced.

Working Qwen3Next Q3 Metal/offload path:

- Command log: `moevm/runs/qwen3_next_q3_llama_moevm_ngl99_hookprefetch_decode20_arena1_t8.log`
- Trace: `moevm/runs/qwen3_next_q3_llama_moevm_ngl99_hookprefetch_decode20_arena1_t8.jsonl`
- Required env: `GGML_OP_OFFLOAD_MIN_BATCH=1`
- Required placement: `-ngl 99 -ot '.*ffn_.*_exps.weight=CPU' --op-offload --no-repack`
- MoEVM runtime: 1GiB arena, 128MiB in-flight, 8 workers, prefetch top10.
- Prompt tokens: 10.
- Generated tokens: 20.
- Decode speed: 5.78 tok/s.
- Copy hook calls: 3,024.
- Copy hook handled: 3,024.
- Callback fallback: 0.
- Copied expert components: 35,613.
- Copied bytes: 16.26GB.
- Runtime demand reads: 0.
- Runtime prefetch reads: 8,648.
- Runtime waits: 1,496, 0.532s total.
- Arena high-water: 1.00GiB.
- Backend normal fallback copies: 63.
- Safe-run end wired: 32.30GiB.

Short sweep:

| Mode | Arena | Workers / in-flight | Decode speed | Demand reads | Normal fallback |
|---|---:|---:|---:|---:|---:|
| `-ngl 4` | 1GiB | 4 / 64MiB | 3.12 tok/s | 111 | 6,228 |
| `-ngl 99` | 1GiB | 4 / 64MiB | 5.50 tok/s | 96 | 18 |
| `-ngl 99` | 2GiB | 4 / 64MiB | 5.07 tok/s | 130 | 18 |
| `-ngl 99` | 1GiB | 8 / 128MiB | 5.77 tok/s | 0 | 18 |
| `-ngl 99` | 1GiB | 12 / 256MiB | 5.25 tok/s | 0 | 18 |

Conclusion: the llama.cpp MoEVM path is now end-to-end working for Qwen3Next Q3. The selected expert copy hook is not just registered; it handles Metal-bound expert component copies from the sidecar with zero fallback. The current preferred local setting is `-ngl 99`, CPU override for routed experts, 1GiB MoEVM arena, `--moevm-copy-mode memcpy`, and `--moevm-cache`.

Parity check:

- Added generated token ID logging in `llama-moe-trace`.
- Added `moevm/phase0/compare_decode_logs.py`.
- Baseline log: `moevm/runs/qwen3_next_q3_baseline_ot_cpu_decode20_parity.log`.
- MoEVM log: `moevm/runs/qwen3_next_q3_moevm_decode20_parity.log`.
- Compare JSON: `moevm/runs/qwen3_next_q3_decode20_parity_compare.json`.
- Result: `same_tokens=true`.
- Baseline: 20 generated tokens at 6.45 tok/s.
- MoEVM: 20 generated tokens at 5.30 tok/s in the parity run.

The matching generated token IDs are:

```text
[198,750,1186,2877,11,293,1648,470,264,481,293,198,750,15602,2877,11,293,1648,470,264]
```

Additional optimization probes:

| Probe | Result | Interpretation |
|---|---:|---|
| Copy-prefetch dedup by `(decode_base, layer, ids)` | 1,008 copy prefetches, 2,016 skips, 5.32 tok/s | Reduces duplicate scheduling but not the dominant cost. |
| `--moevm-direct-mmap` sidecar | 3.08 tok/s, same tokens | OS paging is slower than explicit arena prefetch. |
| Hook-only prefetch with `--moevm-prefetch-topk 0` | 5.03 tok/s, same tokens | Eval-callback prefetch is still useful. |
| `--moevm-copy-mode memcpy` | 20 tokens at 8.29 tok/s, same tokens | Avoids async tensor-set overhead. |
| `--moevm-copy-mode memcpy --moevm-cache` | 20 tokens at 9.99 tok/s, same tokens | Page cache beats `F_NOCACHE` for this decode path. |
| 100-token baseline vs best MoEVM | Baseline 8.10 tok/s; MoEVM 11.06 tok/s; same tokens | Stable +36.5% decode gain. |

Best 100-token result:

- Baseline log: `moevm/runs/qwen3_next_q3_baseline_ot_cpu_decode100_parity.log`
- MoEVM log: `moevm/runs/qwen3_next_q3_moevm_decode100_memcpy_cache_arena1_t12_i256.log`
- Compare JSON: `moevm/runs/qwen3_next_q3_decode100_memcpy_cache_arena1_t12_i256_compare.json`
- Baseline: 100 tokens in 12.345s, 8.10 tok/s.
- MoEVM: 100 tokens in 9.041s, 11.06 tok/s.
- Result: `same_tokens=true`.
- MoEVM settings: 1GiB arena, 256MiB in-flight, 12 workers, prefetch top10, `memcpy`, page cache.

## Shadow / Compact Path

The scheduler patch now supports temporary graph surgery for MoE nodes:

- Save original `node->src[0]` and `node->src[2]` before the callback mutates them.
- Let the callback replace `src[2]` with a shadow IDs tensor, or replace both `src[0]` and `src[2]` with compact weights/remapped IDs.
- Restore all patched pointers after the split compute returns.

This restore is required because llama.cpp graph reuse can re-run the same node objects across decode tokens.

Validated checkpoints:

- Shadow IDs only, graph reuse enabled: `same_tokens=true`, `graph_patches=graph_patch_restores=14403`.
- Shadow IDs only, `LLAMA_GRAPH_REUSE_DISABLE=1`: `same_tokens=true`.
- Compact layer 0: `same_tokens=true`, `compact_applied=300`, `compact_fallback=3`.
- Compact all decode layers: `same_tokens=true`, `compact_applied=14403`, `compact_fallback=141`.
- Compact all logits check: `max_abs=0.0`, `argmax_match_count=20`.

Key logs:

- Shadow IDs: `moevm/runs/qwen3_next_q3_moevm_decode100_memcpy_cache_shadow_ids.log`
- Shadow IDs no graph reuse: `moevm/runs/qwen3_next_q3_moevm_decode100_memcpy_cache_shadow_ids_no_reuse.log`
- Compact layer 0: `moevm/runs/qwen3_next_q3_moevm_decode100_memcpy_cache_compact_l0.log`
- Compact all decode layers: `moevm/runs/qwen3_next_q3_moevm_decode100_memcpy_cache_compact_all.log`

Compact is not the preferred performance path. It still copies the same expert bytes and adds remap/swap overhead. Its purpose was to establish the correct graph mechanism for the persistent expert pool, where only missing compact slots are copied.

## Persistent Expert Pool

`llama-moe-trace` now supports:

- `--moevm-pool-layer <layer|all|-1>`
- `--moevm-pool-capacity <slots>`

The pool path allocates small backend tensors per layer/component, assigns experts to LRU slots, copies only slot misses from the sidecar, remaps IDs to slot IDs, patches `MUL_MAT_ID` inputs for the split, then restores the graph pointers after compute.

Validated checkpoints:

- Pool layer 0, cap64: `same_tokens=true`, logits `max_abs=0.0`.
- Pool all, cap64: `same_tokens=true`, logits `max_abs=0.0`, but only 0.79 tok/s because the larger pool tensor is too expensive for Metal `MUL_MAT_ID`.
- Pool all, cap10, logits: `same_tokens=true`, logits `max_abs=0.0`.
- Pool all, cap10, 100 tokens: `same_tokens=true`, 11.95 tok/s, repeat 11.76 tok/s.
- Pool all, cap10, 256 tokens: 11.74 tok/s, `callback_fallback=0`.
- Pool all, cap11, 100 tokens: `same_tokens=true`, 11.81 tok/s; lower copy volume but slower than cap10.

Current best 100-token result:

- Baseline same placement without MoEVM: 8.10 tok/s.
- Previous best sparse MoEVM: 11.06 tok/s, copied 68.86GB.
- Persistent pool all cap10: 11.76-11.95 tok/s over two runs, copied 42.66GB.
- Gain vs baseline: about +47.5%.
- Gain vs previous best sparse MoEVM: about +8.0%.

Key logs:

- Pool layer 0 logits: `moevm/runs/qwen3_next_q3_moevm_decode20_pool_l0_logits.log`
- Pool all cap64 logits: `moevm/runs/qwen3_next_q3_moevm_decode20_pool_all_logits.log`
- Pool all cap10 logits: `moevm/runs/qwen3_next_q3_moevm_decode20_pool_all_cap10_logits.log`
- Pool all cap10 100-token: `moevm/runs/qwen3_next_q3_moevm_decode100_pool_all_cap10.log`

Engineering note: pool capacity is performance-critical. On Qwen3Next Q3, active experts per token are 10, and `--moevm-pool-capacity 10` is the current best setting. Larger capacities reduce copied bytes but can cost more in Metal compute than they save in copy time.

Rejected experiment: a dense resident cache using llama.cpp's original `input_cpy` tensor was tested behind `--moevm-resident-cache-unsafe`. It reached 13.65 tok/s on the 20-token prompt and copied only 3.75GB, but failed parity immediately: generated IDs became `[198,0,0,...]`. This means the scheduler copy buffer is not safe to treat as durable expert storage across splits. Keep the compact/pool tensor path as the correctness baseline.

## Full-Resident Baseline

Qwen3Next Q3 does fit full-resident on this Mac in the standard llama.cpp/Metal path:

- Log: `moevm/runs/qwen3_next_q3_fullresident_llamabench_decode128.log`
- `llama-bench` model type: `qwen3next 80B.A3B Q3_K - Medium`.
- Params: 79,674,391,296.
- Model size: 36,276,696,064 bytes.
- Placement: `-ngl 99`, `n_cpu_moe=0`, no expert CPU override.
- Generation bench: 49.69 tok/s for 128 tokens.
- Safe-run end wired memory: 31.36GiB.

This reframes the 11-12 tok/s MoEVM pool result. It is a win over the same CPU-expert paging baseline, but not over full-resident Metal. For this Q3 model, MoEVM's purpose is now R&D and portability to larger non-resident targets, not beating the already-fitting resident path. The next optimization target is reducing paging overhead enough to approach resident behavior, or moving to a model/quant where full resident is impossible.

## Code Benchmark Harness

`llama-moe-trace` supports `--generated-out <path>`. `moevm/phase0/code_bench.py` uses it to run small Python code-generation tasks and local unit tests.

Smoke result:

- Summary: `moevm/runs/codebench_smoke/summary.md`
- Baseline CPU-expert placement: 2/3 passed, 7.95 tok/s average.
- MoEVM pool cap10: 2/3 passed, 10.85 tok/s average.
- Pool output matched baseline exactly on tokens and text for all tasks.

## Integration Shape

Phase 1 should be a local experimental flag, not default llama.cpp behavior.

1. At eval-callback time, watch `ffn_moe_topk-<layer>` or `ffn_moe_argsort-<layer>`.
2. Extract expert IDs and call `runtime.prefetch_candidates(layer, ids, count)`.
3. In `ggml-backend.cpp`'s selected-expert copy branch, detect routed expert tensor names:
   - `blk.<layer>.ffn_gate_exps.weight` -> `ExpertComponent::Gate`
   - `blk.<layer>.ffn_up_exps.weight` -> `ExpertComponent::Up`
   - `blk.<layer>.ffn_down_exps.weight` -> `ExpertComponent::Down`
4. For each selected expert:
   - `record = runtime.wait_selected(layer, expert)`
   - `component = runtime.component_view(record, kind)`
   - copy `component.data/component.bytes` into `input_cpy` at the same expert offset llama.cpp would have used.

## Important Caveats

- The eval callback path synchronizes the backend around requested tensors. It is good for an experimental integration and correctness testing, but may not be the final lowest-overhead production hook.
- The backend copy branch sees selected top-k IDs. It is enough for the working system, but wider candidates mostly help future-token prefetch, not current-token scheduler copies.
- Qwen3Next Q3 uses separate `gate`, `up`, and `down` expert tensors. The current component index covers that path.
- If a future GGUF uses fused `ffn_gate_up_exps`, the bridge needs a fused component copy that concatenates Gate + Up into the destination expert slot.
- Existing llama.cpp groups consecutive expert IDs for fewer copies. MoEVM resident records are not guaranteed contiguous in arena memory, so the first implementation should copy per expert/component and optimize coalescing later.
- Keep all first full runs behind `safe_run.py`. Do not test this on `gpt-oss-120b` + Metal.
- `GGML_OP_OFFLOAD_MIN_BATCH=1` is required for short decode tests; otherwise Metal's default opportunistic-offload threshold can leave `MUL_MAT_ID` on CPU.
- Mmap should stay enabled for Qwen3Next Q3 in this setup. `--no-mmap` would load the full GGUF expert tensors into RAM in addition to the sidecar-backed arena.

## Acceptance Gate

The first llama.cpp integration is acceptable only if:

- It compiles behind a local experimental path.
- It runs a tiny Qwen3Next prompt under `safe_run.py` in CPU-only mode.
- It logs MoEVM demand reads, prefetch reads, waits, evictions, and arena high-water.
- Offload copy-hook testing is attempted only when wired memory stays below the configured guardrail.
- It generates at least 20 tokens on Qwen3Next Q3 with copy hook handled count equal to callback attempts and callback fallback equal to zero.
- Its outputs are numerically close to normal llama.cpp for a short deterministic prompt once parity checking is added.
