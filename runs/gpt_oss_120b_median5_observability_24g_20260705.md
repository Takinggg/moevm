# GPT-OSS 120B Median-of-5 Launch Benchmark

- Run dir: `moevm/runs/gpt_oss_120b_median5_observability_24g_20260705/`
- Date: 2026-07-05
- Model: `gpt-oss-120b Q4_K_M`
- Prompt: `prompt_observability_512plus.txt`
- Shape: `c=4096`, 990 prompt tokens, `n=128`, `b=1 ub=1`
- Runtime: MoEVM graph pool, arena 24 GiB, pool cap16, prefetch top4, 4 runtime threads, pressure-trim armed, FlashAttention off
- Logits dump: disabled for throughput measurement

| Iteration | Decode tok/s | Wall s | Decode s | Demand reads | Prefetch reads | Runtime errors | Pressure events |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 8.72 | 81.42 | 14.67 | 0 | 5,279 | 0 | 0 |
| 1 | 10.64 | 88.59 | 12.03 | 0 | 5,279 | 0 | 0 |
| 2 | 10.17 | 85.35 | 12.58 | 0 | 5,279 | 0 | 0 |
| 3 | 7.65 | 102.98 | 16.73 | 0 | 5,279 | 0 | 0 |
| 4 | 7.87 | 92.90 | 16.26 | 0 | 5,279 | 0 | 0 |

Summary:

- Median decode throughput: 8.72 tok/s.
- Range: 7.65-10.64 tok/s.
- Median wall time: 88.59 s.
- Median decode time: 14.67 s.
- Demand reads: 0/5 runs.
- Runtime errors: 0/5 runs.
- Pressure warn/critical/trim events: 0/5 runs.
- Token prefix was identical across all five runs.
- Jetsam log scan after the run did not show a Jetsam event.

Interpretation:

The publishable launch number for this real-context GPT-OSS path is the median 8.72 tok/s, not the earlier best single-run 10-13 tok/s. The important invariant held across the full median run: no demand reads, no runtime errors, no memory-pressure trims, and stable output prefix. This confirms the current bottleneck is orchestration/decode variance, not disk paging.
