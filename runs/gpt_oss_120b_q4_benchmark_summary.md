# gpt-oss-120b Q4_K_M MoEVM Summary

Date: 2026-07-04

Machine: MacBook Pro M5 Pro, 48GB RAM.

## Safety

`gpt-oss-120b` Metal inference is still blocked on this Mac. Prior `-ngl` / FlashAttention tuning hit Metal OOM and a watchdog panic path. All tests below avoid gpt-oss Metal execution.

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

## Packed Replay

No-prefetch Python replay over prompt+decode:

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

## Interpretation

This validates the core MoEVM thesis on a model larger than RAM: the runtime can serve a 56.88 GiB routed expert pool using a bounded 28 GiB arena and load only selected experts. The test is not yet full gpt-oss Metal inference; it is the paging/runtime proof that avoids the known unsafe gpt-oss Metal path on this Mac.

Next gates:

1. Get a longer gpt-oss trace, ideally 1k-5k generated tokens or multiple real coding prompts.
2. Re-run runtime replay at 24/28 GiB on the longer traces.
3. Only attempt full inference after a safer gpt-oss Metal strategy exists, or on a machine with more unified memory.
