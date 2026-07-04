# Qwen3-Coder-Next Q3 Trace Summary

| Trace | Tokens | Records | Working set GiB | LRU@8 hit | LRU@8 sync tok/s | LRU@12 hit | LRU@12 sync tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| code2_b16 | 70 | 3290 | 9.81 | 0.763 | 23.14 | 0.767 | 23.42 |
| frontend1 | 43 | 2021 | 6.79 | 0.737 | 21.58 | 0.737 | 21.58 |
| infra1 | 39 | 1833 | 6.72 | 0.714 | 20.29 | 0.714 | 20.29 |

## Averages

| Budget GiB | LRU hit avg | LRU miss MB/tok avg | LRU sync tok/s avg | S3FIFO hit avg | S3FIFO sync tok/s avg |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 0.738 | 161.4 | 21.67 | 0.692 | 19.30 |
| 12 | 0.739 | 160.5 | 21.76 | 0.702 | 19.78 |
| 20 | 0.739 | 160.5 | 21.76 | 0.722 | 20.80 |
| 28 | 0.739 | 160.5 | 21.76 | 0.737 | 21.62 |

## Packed Sidecar Replay

Sidecar:

- Plan: `moevm/runs/qwen3_next_q3_sidecar_plan.json`
- Packed file: `$MOEVM_MODELS_DIR/qwen3-coder-next/qwen3-next-q3.moevm`
- Records: 24,576 = 48 layers x 512 experts.
- Payload: 31.35 GiB.
- Pack time: 31.71 s, 0.99 GiB/s.

The replay below measures only expert I/O against recorded routes. It is not full model inference speed.

| Trace | Mode | Tokens | Hit | Misses | Read ops | Read GiB | Elapsed s | Replay tok/s | Read GB/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| code2_b16 | GGUF source slices | 70 | 0.763 | 7,792 | 23,376 | 9.93 | 3.788 | 18.5 | 2.82 |
| code2_b16 | packed sidecar | 70 | 0.763 | 7,792 | 7,792 | 9.93 | 0.353 | 198.5 | 30.24 |
| frontend1 | GGUF source slices | 43 | 0.737 | 5,307 | 15,921 | 6.76 | 2.513 | 17.1 | 2.89 |
| frontend1 | packed sidecar | 43 | 0.737 | 5,307 | 5,307 | 6.76 | 0.297 | 144.7 | 24.43 |
| infra1 | GGUF source slices | 39 | 0.714 | 5,251 | 15,753 | 6.69 | 2.463 | 15.8 | 2.92 |
| infra1 | packed sidecar | 39 | 0.714 | 5,251 | 5,251 | 6.69 | 0.293 | 132.9 | 24.50 |

Result:

- Packed sidecar reduces read operations exactly 3.0x because one miss becomes one contiguous `(layer, expert)` record instead of gate/up/down source tensor slices.
- Elapsed replay time improved 8.4x-10.7x on these traces.
- The packed read GB/s is probably helped by OS/file-cache effects despite `F_NOCACHE`, so use the elapsed-vs-source improvement as a prototype signal, not as a final hardware limit.

## Native C++ Pager Replay

Runtime index:

- Indexer: `moevm/phase0/sidecar_index.py`
- Index file: `$MOEVM_MODELS_DIR/qwen3-coder-next/qwen3-next-q3.moevm.idx`
- Index size: 393,272 bytes.
- Layout: dense `layer * expert_count + expert -> pack_offset, bytes`.

C++ replay:

- Source: `moevm/phase0/pager_replay.cpp`
- Binary: `moevm/phase0/pager_replay`
- Mode: sync `pread` from packed sidecar, same LRU policy as Python replay.

| Trace | Tokens | Hit | Read ops | Read GiB | Elapsed s | Replay tok/s | Read GB/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| code2_b16 | 70 | 0.763 | 7,792 | 9.93 | 0.336 | 208.1 | 31.70 |
| frontend1 | 43 | 0.737 | 5,307 | 6.76 | 0.236 | 182.4 | 30.80 |
| infra1 | 39 | 0.714 | 5,251 | 6.69 | 0.229 | 170.6 | 31.44 |

The C++ path matches Python on hit-rate and read bytes, while removing JSON plan parsing from the runtime path.

## Native Async Prefetch Replay

`pager_replay.cpp` now supports:

- `--prefetch-window`
- `--prefetch-threads`
- `--prefetch-max-inflight-mib`
- `--compute-us-per-access`

These runs are still expert I/O replay, not full inference. `compute-us-per-access` is synthetic overlap time, so compare demand reads and wait time separately from wall-time.

Code trace, `window=47`, `inflight=64MiB`, `compute=5us/access`:

| Threads | Elapsed s | Hit | Demand ops | Prefetch ops | Waits | Wait s | Dropped | Read GB/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.590 | 0.880 | 1 | 7,791 | 3,953 | 0.287 | 0 | 18.09 |
| 2 | 0.415 | 0.951 | 1 | 7,793 | 1,604 | 0.104 | 0 | 25.70 |
| 4 | 0.397 | 0.975 | 1 | 7,793 | 820 | 0.069 | 0 | 26.88 |
| 8 | 0.441 | 0.977 | 1 | 7,793 | 754 | 0.067 | 0 | 24.21 |

Selected profile: `window=47`, `threads=4`, `inflight=64MiB`, `compute=5us/access`.

| Trace | Elapsed s | Replay tok/s | Hit | Demand ops | Prefetch ops | Waits | Wait s | Read GB/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| code2_b16 | 0.397 | 176.4 | 0.975 | 1 | 7,793 | 820 | 0.069 | 26.88 |
| frontend1 | 0.278 | 154.6 | 0.973 | 1 | 5,306 | 543 | 0.046 | 26.11 |
| infra1 | 0.258 | 151.0 | 0.969 | 1 | 5,250 | 574 | 0.048 | 27.82 |

Interpretation: a small 64MiB async prefetch queue plus 4 `pread` workers masks most blocking I/O on these traces. 8 workers did not improve the code trace, so the first useful CPU setting is 4 workers, not "all cores".

## Arena Allocator Model

`pager_replay.cpp` now models resident expert storage with a 16KiB-aligned arena allocator instead of only counting LRU bytes. It tracks arena offsets, evictions, free spans, high-water bytes, and resident records. It does not allocate the full 8GiB buffer yet.

Selected async profile rerun with arena metrics: `window=47`, `threads=4`, `inflight=64MiB`, `compute=5us/access`.

| Trace | Elapsed s | Hit | Demand ops | Waits | Wait s | Evictions | Arena high GiB | Resident records | Free spans | Largest free MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| code2_b16 | 0.443 | 0.978 | 1 | 739 | 0.061 | 1,551 | 8.00 | 6,247 | 34 | 1.30 |
| frontend1 | 0.277 | 0.973 | 1 | 552 | 0.046 | 0 | 6.76 | 5,307 | 1 | 1267.34 |
| infra1 | 0.259 | 0.969 | 1 | 576 | 0.048 | 0 | 6.69 | 5,251 | 1 | 1337.70 |

Interpretation: the 8GiB arena model behaves cleanly on these traces. Code reaches the 8GiB cap and evicts, but fragmentation does not become pathological in this short run. Frontend/infra fit below 8GiB and do not evict.

## Long Trace Predictor Gate

Long trace:

- Trace: `moevm/runs/qwen3_next_q3_trace_code_long1.jsonl`
- Prompt tokens: 633.
- Records: 29,751.
- Expert accesses: 297,510.
- Trace mode: CPU-only, `-ngl 0 --no-repack -c 1024 -b 16 -ub 16`.
- Trace runtime: 17.97s under `safe_run.py`.

`pager_replay.cpp` now supports realistic predictor policies:

- `--prefetch-policy oracle`: future trace upper bound, not realistic.
- `--prefetch-policy last-layer`: repeats recently seen experts per layer.
- `--prefetch-policy freq-layer`: predicts frequent experts per layer.
- `--prefetch-policy hybrid`: recent + frequent.
- `--predictor-topk`
- `--predictor-min-count`

No-read long trace summary:

| Budget GiB | Policy | Hit | Demand ops | Prefetch ops | Est read GiB | Evictions |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 2 | LRU | 0.503 | 147,754 | 0 | 188.4 | 146,395 |
| 2 | oracle | 1.000 | 1 | 147,753 | 188.4 | 146,407 |
| 2 | hybrid min-count 5 | 0.579 | 125,163 | 111,619 | 301.9 | 235,343 |
| 4 | LRU | 0.517 | 143,833 | 0 | 183.3 | 141,034 |
| 4 | oracle | 1.000 | 9 | 144,332 | 184.0 | 141,541 |
| 4 | hybrid min-count 5 | 0.585 | 123,344 | 29,427 | 194.8 | 149,817 |
| 8 | LRU | 0.824 | 52,289 | 0 | 66.7 | 46,302 |
| 8 | oracle | 1.000 | 9 | 52,673 | 67.2 | 46,785 |
| 8 | hybrid min-count 5 | 0.823 | 52,546 | 532 | 67.7 | 47,126 |

Real sidecar reads:

| Budget GiB | Policy | Elapsed s | Replay tok/s | Hit | Demand ops | Prefetch ops | Read GiB | Wait s | Evictions |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | LRU | 34.108 | 18.6 | 0.517 | 143,833 | 0 | 183.3 | 0.000 | 141,034 |
| 4 | hybrid min-count 5 | 39.053 | 16.2 | 0.585 | 123,396 | 29,431 | 194.8 | 0.000 | 149,865 |
| 8 | LRU | 12.706 | 49.8 | 0.824 | 52,289 | 0 | 66.7 | 0.000 | 46,302 |
| 8 | oracle | 5.450 | 116.1 | 0.895 | 6 | 52,371 | 66.8 | 2.134 | 46,471 |

Verdict:

- LRU + 8GiB arena is currently the strongest realistic baseline.
- The oracle result proves that prefetch can roughly 2.3x the long-trace I/O replay if the runtime can get a useful signal.
- The first realistic predictors are not good enough. At 4GiB, hybrid improves hit-rate but increases total reads and wall time. At 8GiB, it does not beat LRU.
- Next predictor work should not be "history only"; it needs route-logit/topk information from the current forward pass or a cheap layer-transition predictor.

## Router Candidate Trace

Local `llama-moe-trace` now supports `--trace-candidates N`.

Smoke trace:

- File: `moevm/runs/qwen3_next_q3_trace_candidates_smoke.jsonl`
- Prompt tokens: 13.
- Topk records: 611.
- Candidate records: 611.
- Routed layers/token: 47.
- Candidates/record: 32.

Format:

```json
{"kind":"candidates","token":0,"layer":0,"candidates":[426,292,21]}
{"token":0,"layer":0,"experts":[426,292,21]}
```

Long candidate trace:

- Trace: `moevm/runs/qwen3_next_q3_trace_code_long1_candidates.jsonl`
- Prompt tokens: 633.
- Topk records: 29,751.
- Candidate records: 29,751.
- Total JSONL records: 59,502.
- Candidates/record: 16.
- Trace runtime: 35.28s under `safe_run.py`.

This gives the first realistic prefetch signal: a router-split runtime where router candidate IDs are known before expert matmuls. It is not an oracle because it does not read future tokens; it uses the current layer's router output before loading the selected expert records.

## Candidate-Aware Replay

`pager_replay.cpp` now supports `--prefetch-policy candidates`, `--candidate-topk`, and `--candidate-delay-us`.

Long trace, 8GiB arena, real sidecar reads:

| Policy | Threads | Candidate topk | Delay us | Elapsed s | Replay tok/s | Hit | Demand ops | Prefetch ops | Read GiB | Wait s | Evictions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LRU | 0 | 0 | 0 | 12.703 | 49.8 | 0.824 | 52,289 | 0 | 66.7 | 0.000 | 46,302 |
| candidates | 4 | 10 | 0 | 9.141 | 69.3 | 0.852 | 29 | 52,407 | 66.9 | 8.783 | 46,501 |
| candidates | 4 | 10 | 10 | 9.245 | 68.5 | 0.852 | 20 | 52,421 | 66.9 | 8.424 | 46,519 |
| candidates | 4 | 10 | 50 | 9.723 | 65.1 | 0.854 | 24 | 52,518 | 67.0 | 7.382 | 46,625 |
| candidates | 4 | 16 | 0 | 12.100 | 52.3 | 0.824 | 1 | 104,708 | 133.5 | 11.680 | 98,878 |

Worker sweep, `candidate_topk=10`, delay 0, 8GiB arena:

| Threads | Elapsed s | Replay tok/s | Demand ops | Prefetch ops | Wait s | Read GB/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 12.883 | 49.1 | 10 | 52,279 | 12.583 | 5.56 |
| 2 | 9.875 | 64.1 | 10 | 52,279 | 9.550 | 7.25 |
| 4 | 9.182 | 68.9 | 18 | 52,430 | 8.823 | 7.82 |
| 8 | 9.377 | 67.5 | 21 | 52,382 | 8.984 | 7.65 |
| 12 | 9.534 | 66.4 | 28 | 52,644 | 9.038 | 7.56 |

Verdict:

- Candidate-aware prefetch is the first realistic predictor that beats LRU on real sidecar reads.
- Best current setting: `candidate_topk=10`, `prefetch_threads=4`, `prefetch_max_inflight_mib=64`, `candidate_delay_us=0`.
- It improves the long-trace replay from 12.703s to 9.141s, or about 1.39x.
- It almost eliminates synchronous demand reads, but many accesses still wait on in-flight prefetch. The next bottleneck is wait scheduling and earlier overlap with router/attention work.
- `candidate_topk=16` is rejected: it doubles read traffic to 133.5GiB, causes 98,878 evictions, and falls back near LRU wall-time.

## Native Runtime API Smoke

New files:

- `moevm/phase0/moevm_runtime.hpp`
- `moevm/phase0/moevm_runtime.cpp`
- `moevm/phase0/runtime_smoke.cpp`
- `moevm/phase0/runtime_replay.cpp`
- `moevm/phase0/sidecar_component_index.py`

Runtime contract:

1. Open sidecar and compact index.
2. Create a fixed RAM arena.
3. Call `prefetch_candidates(layer, candidates)`.
4. Call `wait_selected(layer, expert)` for the actual routed experts.
5. Return a resident record pointer for the expert payload.

Smoke run:

- Trace: `moevm/runs/qwen3_next_q3_trace_candidates_smoke.jsonl`.
- Arena: 64MiB.
- In-flight cap: 16MiB.
- Threads: 2.
- Candidate topk: 10.
- Result: 10 selected experts, 10 served from prefetch, 0 demand reads, 0 evictions, 13,598,720 bytes high-water.
- Component index: `$MOEVM_MODELS_DIR/qwen3-coder-next/qwen3-next-q3.moevm.cidx`, 589,880 bytes.
- Component result: `component_bytes == selected_bytes == 13,598,720`, so `Gate + Up + Down` views cover the resident records exactly.

This proves the standalone runtime boundary can do the real operation the final llama.cpp integration needs. It is still a smoke test, not full inference.

Runtime-driven long replay:

- Source: `moevm/phase0/runtime_replay.cpp`.
- Trace: `moevm/runs/qwen3_next_q3_trace_code_long1_candidates.jsonl`.
- Arena: 8GiB.
- In-flight cap: 64MiB.
- Threads: 4.
- Candidate topk: 10.
- Internal elapsed: 9.386s.
- Safe-run wall elapsed including setup: 10.11s.
- Replay speed: 67.4 tok/s.
- Read: 66.56GiB, 52,207 ops.
- Demand reads: 23 ops / 33.65MiB.
- Prefetch reads: 52,184 ops.
- Prefetch waits: 40,632, 9.208s total.
- Demand miss rate: 0.0077%.
- Prefetch wait rate: 13.66%.
- Evictions: 46,207.
- Arena high-water: 8.00GiB.

Comparison against benchmark-only `pager_replay.cpp` candidate top10:

- `pager_replay`: 9.141s, 69.3 replay tok/s, 66.9GiB read.
- `runtime_replay`: 9.386s, 67.4 replay tok/s, 66.56GiB read.
- The reusable runtime API adds about 2-3% internal replay overhead while switching to real resident arena pointers. That is acceptable for the next integration gate.
- Metric caveat: runtime hit-rate counts records after waiting for in-flight prefetch. Compare demand reads, wait rate, read bytes, and wall-time rather than raw hit-rate.

## Oracle Prefetch Simulation

Tool: `moevm/phase0/prefetch_sim.py`.

This is an upper bound, not a real decoder guarantee. It assumes the runtime can look ahead in the recorded route stream.

| Trace | Window accesses | Demand misses | Demand MB/tok | Total MB/tok | Prefetch loads |
| --- | ---: | ---: | ---: | ---: | ---: |
| code2_b16 | 0 | 7,792 | 152.364 | 152.364 | 0 |
| code2_b16 | 47 | 2 | 0.039 | 152.422 | 7,793 |
| code2_b16 | 470 | 2 | 0.039 | 152.830 | 7,814 |
| code2_b16 | 1,880 | 1 | 0.019 | 154.035 | 7,877 |
| frontend1 | 0 | 5,307 | 168.861 | 168.861 | 0 |
| frontend1 | 47 | 1 | 0.032 | 168.861 | 5,306 |
| infra1 | 0 | 5,251 | 184.288 | 184.288 | 0 |
| infra1 | 47 | 1 | 0.035 | 184.288 | 5,250 |

Interpretation: if the runtime can prefetch about one layer of future expert records, the blocking demand I/O almost disappears on these traces while total bytes remain roughly the same. The hard problem is making that prefetch realistic without oracle future routes.

## Decision

- Qwen3Next Q3 is a strong MoEVM lab target.
- The observed short-prompt working set is 6.7-9.8 GiB, far below the 31.35 GiB routed pool.
- Even an 8 GiB arena projects about 20-23 sync tok/s on these traces with a 5 GB/s SSD assumption.
- The sidecar packer and replay path validate that compact `(layer, expert)` records are much cheaper to stream than dispersed GGUF tensor slices.
- The long trace proves the important R&D direction: current-layer router candidates can prefetch selected expert records and beat pure LRU without future-token oracle access.
- The first native runtime API smoke passes with real sidecar reads and a real RAM arena.
- Runtime-driven replay matches the benchmark-only pager closely enough to move on.
- The component index maps the current packed record format onto llama.cpp's per-tensor `MUL_MAT_ID` copies.
- Next implementation step: integrate at `ggml-backend.cpp`'s existing “copy only used experts” branch and replace host tensor slice copies with MoEVM component views behind a local flag.
