# Evidence for the README results table

Raw outputs behind each headline number, copied from the original run
directory. Everything else from those runs (traces, `.logits` dumps, hundreds of
exploratory logs) stays out of the repository because of size.

The only edit made to these files is path redaction: absolute paths on the
author's machine were rewritten to repo-relative form (`runs/...`,
`external/llama.cpp/...`) and model paths to `$MOEVM_MODELS_DIR/...`. Numbers,
token ids and counters are untouched.

## Qwen3-Coder-Next Q4_K_M (45.2 GiB GGUF)

Head-to-head: both runs use the fork's `llama-moe-trace` with routing traces
disabled (`records=0`), the same prompt (`Write a Python function that parses
log lines into records.`), `-c 1536`, `-b 1 -ub 1`, `-n 1024 --ignore-eos`,
`--temp 0 -s 1`, and `-ngl 99 --op-offload` with routed experts overridden to
CPU. The only difference is the `--moevm-*` flags.

| File | What it shows |
|---|---|
| `qwen3_next_q4_p0d_parity_stock_ignoreeos_n1024_v2.log`, `.summary` | MoEVM off: 1024 tokens in 159.706 s, 6.41 tok/s |
| `qwen3_next_q4_p0d_parity_moevm_cap128_ignoreeos_n1024.log`, `.summary` | MoEVM pool, cap 128: 1024 tokens in 51.462 s, 19.90 tok/s |
| `../qwen3_next_q4_p0d_parity_stock_vs_moevm_n1024.json` | Logits of the two runs: 1024 rows x 151,936 vocab, `max_abs=0.0`, zero token/argmax mismatches |
| `qwen3_next_q4_graph_parity_off_vs_sched_n50.json` | Later graph-build path, pool off vs on: 50 rows, `max_abs=0.0`, zero token/argmax mismatches |
| `qwen3_next_q3_fullresident_llamabench_decode128.log` | Machine limit: `recommendedMaxWorkingSetSize = 40200.90 MB` reported by ggml-metal |

## gpt-oss-120b Q4_K_M (58 GiB GGUF, 2 shards)

| File | What it shows |
|---|---|
| `gpt_oss_120b_median5_observability_24g_20260705/` | Five runs at `c=4096`, 990-token prompt, 128 decoded tokens, 24 GiB arena. Median 8.72 tok/s; `demand_reads=0`, `runtime_errors=0` and no pressure events in every run (`results.jsonl`, `summary.json`). |
| `gpt_oss_120b_parallel4_observability_n32.log` | Four parallel sequences on the same 990-token prompt at `c=4096`: 128 tokens (4 x 32 decode steps) in 4.775 s, 26.81 aggregate tok/s, 6.70 tok/s per sequence |
| `gpt_oss_120b_q4_metal_opoffload_baseline_n64.log` | Same-backend Metal reference (`--op-offload`) for the parity gate |
| `gpt_oss_120b_q4_metal_moevm_pool16_n64.log` | MoEVM graph pool, cap 16, same prompt and placement |
| `gpt_oss_120b_q4_metal_opoffload_vs_moevm_n64_logits_compare.json` | Reference vs MoEVM: 64 rows, `max_abs=0.0`, zero token/argmax mismatches |

The single-run tok/s printed in the two n64 logs come from a short prompt and
are not headline numbers; the median-of-5 run above is.
