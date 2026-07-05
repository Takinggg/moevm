# gpt-oss-120b Q4_K_M MoEVM Summary

Date: 2026-07-04

Updated: 2026-07-05 with cold-replay caveat, 512-token traces, and Belady cache bounds.

Machine: MacBook Pro M5 Pro, 48GB RAM.

## Safety

`gpt-oss-120b` unsafe Metal shapes remain blocked on this Mac. Prior unconstrained `-ngl` / FlashAttention tuning hit Metal OOM and a watchdog panic path. The bounded MoEVM graph-pool path below now runs on Metal only through explicit `safe_run.py` gpt-oss gates, FlashAttention off, expert source tensors on CPU, and pressure-trim armed for scale tests.

The validated path is:

- CPU-only route trace with `llama-moe-trace`.
- Multi-shard sidecar planning and packing.
- Packed sidecar replay.
- C++ MoEVM runtime replay with a real RAM arena and candidate prefetch.

## Model Layout

- Source model: `$MOEVM_MODELS_DIR/gpt-oss-120b/Q4_K_M/gpt-oss-120b-Q4_K_M-00001-of-00002.gguf`
- Shards: 2 GGUF files, 58 GiB total.
- Architecture: `gpt-oss`
- Layers: 36
- Experts: 128
- Active experts/token: 4
- Routed expert pool: 56.88 GiB
- Non-routed resident estimate: 1.57 GiB
- Record size: 12.64 MiB per `(layer, expert)`

## Sidecar

- Plan: `moevm/runs/gpt_oss_120b_q4_sidecar_plan.json`
- Packed sidecar: `$MOEVM_MODELS_DIR/gpt-oss-120b/Q4_K_M/gpt-oss-120b-q4.moevm`
- Index: `$MOEVM_MODELS_DIR/gpt-oss-120b/Q4_K_M/gpt-oss-120b-q4.moevm.idx`
- Component index: `$MOEVM_MODELS_DIR/gpt-oss-120b/Q4_K_M/gpt-oss-120b-q4.moevm.cidx`
- Records: 4,608
- Payload: 56.74 GiB
- Pack time: 55.72s, 1.02 GiB/s
- File SHA-256: `ac311e633a1b555054f6c687a253cdccf93615f8c1715b88f0f13008109d5f64`

Multi-shard fix: `TensorInfo.source_path` is now tracked by `gguf_inspect.py`, and `sidecar_plan.py` validates slice offsets against the owning shard. This was required because 128 layer-28 records span both source shards.

## Trace

Trace: `moevm/runs/gpt_oss_120b_q4_cpu_trace_code_n128.jsonl`

- Prompt tokens: 67
- Generated tokens: 128
- Total trace tokens: 195
- Topk records: 6,954
- Candidate records: 6,954
- Expert accesses: 27,816
- CPU-only decode: 128 tokens in 11.796s, 10.85 tok/s
- Safe-run memory stayed stable.

## Cache Simulation

Decode-only window starts after prompt token 67.

| Pool | Policy | Hit Rate | Miss MB/token | Sync tok/s est | Overlap tok/s est |
|---:|---|---:|---:|---:|---:|
| 24 GiB | LRU | 0.859 | 256.8 | 15.30 | 19.47 |
| 28 GiB | LRU | 0.924 | 138.2 | 24.02 | 36.19 |
| 32 GiB | LRU | 0.947 | 95.6 | 30.20 | 52.31 |

`cache_sim.py` now also supports `belady`, an offline-optimal eviction policy. Belady is not deployable online; it gives the upper bound for what a better cache/prefetch policy could extract from the same trace.

## Packed Replay

No-prefetch Python replay over prompt+decode. These replay tok/s numbers are page-cache sensitive and should not be used as cold-SSD claims.

| Pool | Hit Rate | Reads | Read GiB | Replay tok/s |
|---:|---:|---:|---:|---:|
| 28 GiB | 0.870 | 3,614 | 44.49 | 53.29 |
| 32 GiB | 0.885 | 3,195 | 39.33 | 58.00 |

Packed sidecar reduces read operations 3x versus source-GGUF slices on this trace: 3,614 packed reads vs 10,842 source-slice reads at 28 GiB.

## Runtime Replay

C++ runtime replay uses the real MoEVM arena, sidecar index, and candidate prefetch. `candidate_topk=4` is the correct setting for this model because `gpt-oss` selects 4 active experts/token.

| Pool | candidate_topk | Demand Reads | Read GiB | Evictions | Arena High Water | Replay tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 24 GiB | 4 | 0 | 59.87 | 2,914 | 24.00 GiB | 37.80 |
| 28 GiB | 4 | 0 | 44.54 | 1,345 | 27.99 GiB | 50.89 |
| 32 GiB | 4 | 0 | 39.35 | 598 | 31.99 GiB | 31.65 |
| 28 GiB | 10 | 728 | 108.01 | 6,500 | 27.99 GiB | 24.45 |

Best current point: 28 GiB arena, candidate top4, 4 prefetch threads, 64 MiB in-flight.

`--no-integrity-check` keeps size checks but skips the full sidecar SHA read after the sidecar has already been validated. It reduces the 28 GiB/top4 safe-run wall time from about 28s to 5.57s while preserving the same internal replay behavior.

Cold-replay caveat:

- `sudo purge` could not be run non-interactively on this machine, so a true cold page-cache replay is not available from this pass.
- A best-effort eviction read of the Qwen Q4 GGUF before replay still produced 46.22 replay tok/s while reading 44.54 GiB. That remains too cache/storage-dependent to publish as a cold result.
- Public claim should use replay throughput only as warm-runtime evidence. The defensible result is the demand-read behavior and cache locality, not the 46-50+ replay tok/s number.

## Long Traces

Three 512-token coding traces were collected CPU-only with gpt-oss Metal still disabled:

| Trace | Prompt Tok | Gen Tok | CPU Decode tok/s | Records | Expert Accesses |
|---|---:|---:|---:|---:|---:|
| Python backend | 47 | 512 | 14.18 | 40,156 | 80,312 |
| TypeScript type utility | 48 | 512 | 13.77 | 40,226 | 80,452 |
| Rust tokenizer | 51 | 512 | 14.36 | 40,436 | 80,872 |

Decode-only LRU vs Belady cache bounds:

| Trace | Pool | LRU Hit | LRU Miss MB/tok | LRU Sync tok/s est | Belady Hit | Belady Sync tok/s est | 512-token Working Set |
|---|---:|---:|---:|---:|---:|---:|---:|
| Python | 16 GiB | 84.97% | 273.6 | 14.55 | 93.44% | 26.41 | 41.49 GiB |
| Python | 24 GiB | 93.28% | 122.3 | 26.00 | 96.78% | 38.88 | 41.49 GiB |
| Python | 28 GiB | 95.32% | 85.3 | 32.20 | 97.42% | 42.74 | 41.49 GiB |
| TypeScript | 16 GiB | 85.86% | 257.3 | 15.27 | 93.32% | 26.09 | 42.27 GiB |
| TypeScript | 24 GiB | 93.26% | 122.7 | 25.95 | 96.62% | 38.01 | 42.27 GiB |
| TypeScript | 28 GiB | 95.10% | 89.2 | 31.41 | 97.15% | 41.05 | 42.27 GiB |
| Rust | 16 GiB | 84.16% | 288.3 | 13.95 | 93.27% | 25.97 | 39.98 GiB |
| Rust | 24 GiB | 93.63% | 116.0 | 26.88 | 96.92% | 39.64 | 39.98 GiB |
| Rust | 28 GiB | 95.75% | 77.3 | 33.93 | 97.49% | 43.23 | 39.98 GiB |

Research conclusion: LRU is quasi-optimal on these coding traces. The 28 GiB gap between LRU and offline-optimal Belady is only about 1.7-2.1 hit-rate points, so new online cache policies are unlikely to repay their engineering cost for this workload. Future work should focus on prefetch timing, compute/orchestration, and pressure-triggered trimming rather than S3-FIFO/LFU-style policy swaps.

Runtime replay at 28 GiB/candidate top4 on the same long traces:

| Trace | Tokens | Demand Reads | Read GiB | Evictions | Arena High Water | Replay tok/s |
|---|---:|---:|---:|---:|---:|---:|
| Python | 559 | 1 | 64.92 | 3,000 | 27.99 GiB | 89.11 |
| TypeScript | 560 | 0 | 66.28 | 3,111 | 27.99 GiB | 88.12 |
| Rust | 563 | 0 | 61.64 | 2,734 | 27.99 GiB | 94.82 |

These replay tok/s numbers are warm-cache internal-runtime numbers. The important result is that candidate top4 prefetch gives 0, 0, and 1 demand read across about 241k expert accesses.

## Reclaim/Trim Guard

On macOS, the runtime now uses `MADV_FREE_REUSABLE` on evicted full pages and `MADV_FREE_REUSE` before writing into them again. New metrics:

- `arena_reclaim_advises`
- `arena_reclaim_bytes`
- `arena_reclaim_live_bytes`
- `arena_reuse_advises`
- `arena_reuse_bytes`

Smoke checks on the Python long trace, 160 tokens, run sequentially:

| Pool | Demand Reads | Evictions | Cumulative Reclaim GiB | Live Reclaim GiB | Arena High Water | Replay tok/s | safe_run End Wired |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 GiB | 4 | 3,800 | 46.79 | 0.00 | 16.00 GiB | 22.81 | 2.79 GiB |
| 20 GiB | 2 | 2,440 | 30.05 | 0.00 | 20.00 GiB | 29.46 | 2.79 GiB |
| 24 GiB | 0 | 1,125 | 13.85 | 0.00 | 24.00 GiB | 40.90 | 2.80 GiB |

The cumulative reclaim number is not the pressure-relief metric. On normal replay, evicted slots are immediately reused, so `arena_reclaim_live_bytes` returns to zero. The real pressure guard is explicit trimming.

`Runtime::trim_to_bytes(target_used_bytes)` evicts LRU records without replacing them, leaving pages marked reusable. Footprint validation on the same 16 GiB replay held alive after completion:

| Mode | Runtime Used | Live Reclaim | `footprint` Reclaimable | `phys_footprint` | Peak |
|---|---:|---:|---:|---:|---:|
| No trim | 16.00 GiB | 0.00 GiB | 0.00 GB | 17.19 GB | 17.19 GB |
| Trim to 8 GiB | 7.99 GiB | 8.00 GiB | 8.59 GB | 8.60 GB | 17.19 GB |

So `MADV_FREE_REUSABLE` does decrement the relevant footprint ledger when paired with an actual trim. For gpt-oss Metal, the safety mechanism should be pressure-triggered trim/shrink; eviction-only advice is not enough.

`safe_run.py` now keeps gpt-oss Metal blocked by default and exposes narrow explicit gates:

- `--allow-gptoss-metal-smoke`: `-n <= 8`, MoEVM arena <= 16 GiB.
- `--allow-gptoss-metal-parity-gate`: `-n <= 128`, MoEVM arena <= 16 GiB.
- `--allow-gptoss-metal-scale-gate`: `-n <= 128`, MoEVM arena <= 24 GiB, and `--moevm-pressure-trim` is mandatory above 16 GiB.

`llama-moe-trace` now has an opt-in Darwin memory-pressure listener. With `--moevm-pressure-trim`, `DISPATCH_MEMORYPRESSURE_WARN` trims the runtime arena down by one configured step, and `CRITICAL` trims by two steps, bounded by `--moevm-pressure-trim-min-gib`. Pressure events and reclaimed bytes are logged as `MoEVM pressure_trim ...`.

## Metal Smoke

Prerequisites checked:

- `llama-moe-trace` was rebuilt after the runtime reclaim/trim changes.
- The sparse-mmap loader content is present (`llama_buf_map_find` and post-load `unmap_fragment` of unused mmap ranges).
- `--moevm-load-only` was added so a true load-only smoke can skip prompt prefill/decode.
- gpt-oss `ADD_ID` nodes remain unpatched by graph-pool; only `MUL_MAT_ID` pool nodes are rewritten.

Load-only smoke:

- Command shape: `-ngl 99`, expert weights overridden to CPU, `--no-op-offload`, `--no-repack`, `-fa off`, `--moevm-graph-mode pool`, arena 16 GiB, pool cap16, `--moevm-load-only`.
- Result: pass, no decode, graph-mode installed on `MTL0`, safe-run end wired 3.49 GiB.

First decode smoke, same prompt as CPU baseline, `-n 8`:

| Run | Placement | Tokens | Decode tok/s | Generated IDs | Result |
|---|---|---:|---:|---|---|
| CPU baseline | `-ngl 0 --device none` | 8 | 5.74 | `[279,1314,57388,6271,11,287,3127,622]` | Reference CPU |
| Metal baseline | `-ngl 99`, expert weights CPU | 8 | 4.65 | `[279,1314,71973,6271,11,287,3127,622]` | Metal placement reference |
| MoEVM graph pool | `-ngl 99`, sidecar pool cap16 | 8 | 8.71 | `[279,1314,71973,6271,11,287,3127,622]` | Smoke pass vs Metal baseline |

Short logits comparisons:

- CPU baseline vs MoEVM: 1 token/argmax mismatch at row 2, `max_abs=5.33`. This is not a MoEVM-only failure because CPU baseline vs Metal baseline shows the same token/argmax mismatch.
- Metal baseline vs MoEVM: 0 token mismatches, 0 argmax mismatches, `max_abs=1.36`, `mean_abs=0.0705`. This is not bit-exact, likely because the baseline uses CPU expert matmuls while MoEVM copies selected experts into Metal pool tensors.

Same-backend Metal-vs-Metal parity gate:

| Run | Placement | Tokens | Decode tok/s | Result |
|---|---|---:|---:|---|
| Metal op-offload baseline | `-ngl 99`, experts CPU, `--op-offload`, `--no-repack` | 64 | 2.84 | Slow same-compute reference |
| MoEVM graph pool | `-ngl 99`, sidecar pool cap16 | 64 | 11.54 | `demand_reads=0`, `runtime_errors=0` |

Logits comparison, op-offload Metal baseline vs MoEVM graph pool:

- Rows: 64.
- Token mismatches: 0.
- Argmax mismatches: 0.
- `max_abs=0.0`, `mean_abs=0.0`, `rms_abs=0.0`.

This is the discriminating correction test for the earlier `max_abs=1.36`: old op-offload Metal path vs MoEVM pool, same Metal backend and same MXFP4 dequant path. It rules out a graph-pool `ADD_ID` bias corruption on this tested path.

Smoke/parity verdict: gpt-oss-120b Q4 now runs a bounded Metal/MoEVM decode on the Mac without jetsam/watchdog. The CPU-vs-Metal drift is real, but at backend-equal placement MoEVM is bit-exact over 64 decoded tokens, so the gpt-oss `ADD_ID` bias path is not currently corrupted by graph-pool.

## Metal Scale

All runs below use FlashAttention off, expert weights overridden to CPU in the source model, MoEVM graph pool, `--moevm-pressure-trim`, and `safe_run.py`.

Short prompt, `c=128`, `n=64`, same op-offload baseline logits:

| Arena | Decode tok/s | Demand Reads | Runtime Evictions | Arena Used | Pressure Events | Logits vs op-offload |
|---:|---:|---:|---:|---:|---:|---|
| 16 GiB | 11.44 | 0 | 511 | 16.00 GiB | 0 | `max_abs=0.0` |
| 20 GiB | 13.34 | 0 | 35 | 20.00 GiB | 0 | `max_abs=0.0` |
| 24 GiB | 13.24 | 0 | 0 | 20.38 GiB | 0 | `max_abs=0.0` |

The 24 GiB short-prompt run only used 20.38 GiB, so the 20->24 delta is not meaningful on that workload.

Real-context prompt, `c=4096`, 990 prompt tokens from `prompt_observability_512plus.txt`, `n=128`:

| Mode | Arena | Batch | Wall s | Decode tok/s | Demand Reads | Prefetch Reads | Wait s | Runtime Evictions | Logits |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| MoEVM | 16 GiB | `b=1 ub=1` | 86.99 | 9.37 | 11 | 18,764 | 19.42 | 17,476 | Reference |
| MoEVM | 24 GiB | `b=1 ub=1` | 76.42 | 9.94 | 0 | 5,279 | 6.80 | 3,330 | `max_abs=0.0` vs 16 GiB |
| MoEVM | 24 GiB | `b=4 ub=4` | 64.05 | 10.19 | 360 | 5,070 | 5.49 | 3,481 | Batch mode |

Median-of-5 launch rerun, same real-context prompt and same `24 GiB`, `b=1 ub=1`, `n=128` shape:

- Summary: `moevm/runs/gpt_oss_120b_median5_observability_24g_20260705.md`.
- Decode tok/s runs: 8.72, 10.64, 10.17, 7.65, 7.87.
- Median decode throughput: 8.72 tok/s; range 7.65-10.64 tok/s.
- Median wall time: 88.59 s; median decode time: 14.67 s.
- Demand reads: 0/5. Runtime errors: 0/5. Pressure events/trims: 0/5. Token prefix identical across all five runs.
- Publishable launch number for this path is 8.72 tok/s median. The earlier 10-13 tok/s values remain valid single-run observations but should not be used as the headline.

Parallel multi-agent decode, same real-context prompt, `c=4096`, `n=32` per sequence:

| Mode | Arena | Batch | Parallel seqs | Aggregate tokens | Decode tok/s aggregate | Tok/s/seq | Decode Demand Reads | Runtime Errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MoEVM parallel | 24 GiB | `b=4 ub=4` | 4 | 128 | 26.81 | 6.70 | 0 | 0 |

Summary: `moevm/runs/gpt_oss_120b_parallel4_aggregate_20260705.md`.
This reaches the 20-25+ tok/s band as aggregate throughput for multi-agent serving. It is not a single-stream latency number.

Interpretation:

- Larger arena fixes most I/O pressure on real context: demand reads 11 -> 0 and runtime wait 19.42s -> 6.80s.
- Throughput improves only modestly because scheduler/readback orchestration dominates once I/O is controlled.
- Timestamp diagnostic on `24 GiB`, `b=1 ub=1`, same real context, `n=64`: wall 71.88s, `metal_gpu_busy_us=20.07s`, `ids_readback_us=25.79s`, `callback_us=29.18s`, `backend_sync_us=26.73s`. This points to rendez-vous/ids readback rather than raw GPU compute as the next bottleneck.
- `b=4 ub=4` is bit-exact against same-backend op-offload over 16 rows (`max_abs=0.0`) and improves full-run wall time, but it does not reproduce the `b=1 ub=1` token stream. Use it as a batch/prefill mode, not as a bit-identical replacement for `b=1`.
- `b=8 ub=8` with pool cap16 fails safely at graph fill with `fallback_k_overflow=1`: the top-k union exceeds pool capacity. Cap32+ is a separate memory-risk experiment, not yet validated.

## Graph/ADD_ID Check

gpt-oss has per-expert bias tensors that use `GGML_OP_ADD_ID`. The current graph-pool path does not patch those nodes:

- Build-time `moevm_graph_transform_cb` returns early unless `node->op == GGML_OP_MUL_MAT_ID`.
- Scheduler graph-pool dispatch uses `ggml_backend_moe_is_graph_pool_node()`, which also requires `GGML_OP_MUL_MAT_ID` and a `moevm_pool_` weight tensor.
- The legacy sparse-copy hook rejects first nodes that are not `GGML_OP_MUL_MAT_ID`; its existing comment says `GGML_OP_ADD_ID` weights are not split.

So the current gpt-oss-specific invariant is: MoEVM may rewrite routed expert matrix multiply weights and their shadow ids, but it must leave `ADD_ID` bias ids untouched.

## Interpretation

This validates the core MoEVM thesis on a model larger than RAM: the runtime can serve a 56.88 GiB routed expert pool while loading only selected experts, and the first bounded Metal decode is bit-exact against a same-backend reference.

Next gates:

1. Run 6-8h launch soak with pressure trim armed and periodic logits snapshots.
2. Freeze `moevm-launch-v0` if soak stays clean.
3. Post-launch: optimize the graph-pool rendez-vous path; fewer ids readbacks/callbacks matter more than larger arenas now.
4. Do not publish replay tok/s as a cold SSD number until a real `sudo purge` or reboot-cold protocol is run.
