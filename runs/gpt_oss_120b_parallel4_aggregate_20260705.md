# gpt-oss-120b Parallel Decode Aggregate Probe

Date: 2026-07-05

Code change: `llama-moe-trace` now supports `--moevm-parallel-seqs N`, which prefills N independent sequences and then decodes one token per sequence in each target batch. This is an aggregate-throughput mode for multi-agent workloads, not a single-stream latency mode.

## GPT-OSS 120B Q4_K_M

Common shape:

- Target: `$MOEVM_MODELS_DIR/gpt-oss-120b/Q4_K_M/gpt-oss-120b-Q4_K_M-00001-of-00002.gguf`
- MoEVM sidecar: `$MOEVM_MODELS_DIR/gpt-oss-120b/Q4_K_M/gpt-oss-120b-q4.moevm`
- `-ngl 99`, expert source tensors overridden to CPU, `--no-op-offload`, `--no-repack`, `-fa off`
- MoEVM graph pool, arena 24 GiB, pool cap16, prefetch top4, 4 runtime threads
- `--moevm-pressure-trim`, `safe_run.py`

| Prompt | Context | Parallel seqs | Decode steps | Aggregate tokens | Decode s | Aggregate tok/s | Tok/s/seq | Decode demand reads | Runtime errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Short coding prompt | 512 | 4 | 32 | 128 | 3.923 | 32.63 | 8.16 | 0 | 0 |
| `prompt_observability_512plus.txt` (990 prompt tokens) | 4096 | 4 | 32 | 128 | 4.775 | 26.81 | 6.70 | 0 | 0 |

Real-context run details:

- Safe-run elapsed: 158.25 s.
- Prefill is currently sequential per sequence: 139.35 s for 4 x 989-token prefixes.
- Decode phase: `prefetch_reads=759`, `wait_s=0.942277`, `evictions=759`, `arena_used=24.00GiB`, `high_water=24.00GiB`.
- Decode backend: `ids_readback_calls=1152`, `ids_readback_us=1.791973s`, `callback_us=2.624278s`, `backend_sync_us=2.044452s`.
- Pressure events/trims: 0.

Interpretation:

- The single-stream real-context median remains 8.72 tok/s.
- Parallel decode reaches the requested 20-25+ tok/s band as aggregate throughput: 26.81 tok/s on the real 990-token prompt.
- This is the correct claim for multi-agent serving: four active agents share the same 120B target process and amortize graph-pool rendez-vous.
- It does not reduce single-agent latency; that still needs the post-launch scheduler/readback refactor.
