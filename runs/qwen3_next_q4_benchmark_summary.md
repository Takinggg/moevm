# Qwen3-Coder-Next Q4 MoEVM Benchmark Summary

Date: 2026-07-04

## Model And Sidecar

- Model: `$MOEVM_MODELS_DIR/qwen3-coder-next-q4/Qwen3-Coder-Next-Q4_K_M.gguf`
- GGUF size: about 45GiB
- Routed expert pool: 43.59GiB
- Packed MoEVM sidecar: `$MOEVM_MODELS_DIR/qwen3-coder-next-q4/qwen3-next-q4.moevm`
- Sidecar payload: 43.69GiB
- Sidecar pack time: 42.07s, 1.04GiB/s
- Indexes: `.idx` 384KiB, `.cidx` 576KiB
- Audit-hardened sidecar integrity: Q4 `.idx` and `.cidx` now include sidecar size plus SHA-256 prefix and were regenerated on 2026-07-04. Runtime verifies those fields when present.

## Safety Fixes Before Runtime

- `gguf_inspect.py` now streams GGUF headers instead of using `path.read_bytes()`.
- `safe_run.py` blocks concurrent model I/O jobs by default.
- `sidecar_pack.py` is required to run with `--nocache` under `safe_run.py`.
- `safe_run.py` catches Metal OOM patterns and handles post-kill child timeout safely.
- Audit hardening: `safe_run.py` now reads child output with nonblocking `os.read()` and fails closed after repeated `memory_pressure` failures.
- Audit hardening: backend MoE stats use internal relaxed atomics, `debug_remaining` is an atomic claim counter, low-level timing hooks are gated outside MoEVM, and `MOEVM_FORCE_STATS=1` can force diagnostic timing. Stats snapshots/resets are documented as relaxed per-field diagnostics. Runtime errors are counted/logged via `runtime_errors` with first/last messages retained.
- `llama-model.cpp` / `llama-model-loader.cpp` now map sparse mmap tensor ranges to Metal buffers instead of creating one `buffer_from_host_ptr` span from the first to last Metal tensor in a GGUF file.

## Legacy Working Runtime Shape

```bash
env GGML_OP_OFFLOAD_MIN_BATCH=1 \
external/llama.cpp/build/bin/llama-moe-trace \
  -m $MOEVM_MODELS_DIR/qwen3-coder-next-q4/Qwen3-Coder-Next-Q4_K_M.gguf \
  --moevm-index $MOEVM_MODELS_DIR/qwen3-coder-next-q4/qwen3-next-q4.moevm.idx \
  --moevm-component-index $MOEVM_MODELS_DIR/qwen3-coder-next-q4/qwen3-next-q4.moevm.cidx \
  --moevm-sidecar $MOEVM_MODELS_DIR/qwen3-coder-next-q4/qwen3-next-q4.moevm \
  --moevm-arena-gib 1 \
  --moevm-inflight-mib 64 \
  --moevm-threads 4 \
  --moevm-prefetch-topk 10 \
  --moevm-copy-mode memcpy \
  --moevm-cache \
  --moevm-pool-layer all \
  --moevm-pool-capacity 256 \
  -ngl 0 \
  -ot '.*ffn_.*_exps.weight=CPU' \
  --op-offload \
  --no-repack \
  -fa off \
  -nkvo \
  -c 192 -b 1 -ub 1 -n 100 \
  --temp 0 -s 1 \
  -p 'def add(a, b): return a + b'
```

## Current Best Runtime Shape

After the sparse mmap Metal loader patch:

```bash
env GGML_OP_OFFLOAD_MIN_BATCH=1 \
external/llama.cpp/build/bin/llama-moe-trace \
  -m $MOEVM_MODELS_DIR/qwen3-coder-next-q4/Qwen3-Coder-Next-Q4_K_M.gguf \
  --moevm-index $MOEVM_MODELS_DIR/qwen3-coder-next-q4/qwen3-next-q4.moevm.idx \
  --moevm-component-index $MOEVM_MODELS_DIR/qwen3-coder-next-q4/qwen3-next-q4.moevm.cidx \
  --moevm-sidecar $MOEVM_MODELS_DIR/qwen3-coder-next-q4/qwen3-next-q4.moevm \
  --moevm-arena-gib 1 \
  --moevm-inflight-mib 64 \
  --moevm-threads 4 \
  --moevm-prefetch-topk 10 \
  --moevm-copy-mode memcpy \
  --moevm-cache \
  --moevm-pool-layer all \
  --moevm-pool-capacity 128 \
  -ngl 99 \
  -ot '.*ffn_.*_exps.weight=CPU' \
  --op-offload \
  --no-repack \
  -c 192 -b 1 -ub 1 -n 100 \
  --temp 0 -s 1 \
  -p 'def add(a, b): return a + b'
```

## Results

| Mode | Log | Decoded | Seconds | tok/s | Token parity |
|---|---|---:|---:|---:|---|
| Baseline, same placement, no sidecar | `qwen3_next_q4_baseline_decode100_ngl0_opoffload_b1.log` | 64 | 21.094 | 3.03 | Reference |
| MoEVM cap10 | `qwen3_next_q4_moevm_decode100_ngl0_opoffload_b1.log` | 64 | 12.895 | 4.96 | Same |
| MoEVM cap64 | `qwen3_next_q4_moevm_decode100_ngl0_opoffload_b1_cap64.log` | 64 | 12.546 | 5.10 | Same |
| MoEVM cap128 | `qwen3_next_q4_moevm_decode100_ngl0_opoffload_b1_cap128.log` | 64 | 12.549 | 5.10 | Same |
| MoEVM cap256 | `qwen3_next_q4_moevm_decode100_ngl0_opoffload_b1_cap256.log` | 64 | 12.511 | 5.12 | Same |
| Baseline `-ngl99 --op-offload`, sparse mmap | `qwen3_next_q4_baseline_decode100_ngl99_sparse_mmap_opoffload_kvfaauto.log` | 92 | 21.712 | 4.24 | Reference |
| MoEVM `-ngl99`, cap32, sparse mmap | `qwen3_next_q4_moevm_decode100_ngl99_sparse_mmap_cap32_kvfaauto.log` | 92 | 7.423 | 12.39 | Same |
| MoEVM `-ngl99`, cap64, sparse mmap | `qwen3_next_q4_moevm_decode100_ngl99_sparse_mmap_cap64_kvfaauto.log` | 92 | 7.033 | 13.08 | Same |
| MoEVM `-ngl99`, cap128, sparse mmap | `qwen3_next_q4_moevm_decode100_ngl99_sparse_mmap_cap128_kvfaauto.log` | 92 | 7.100 | 12.96 | Same |
| MoEVM `-ngl99`, cap64, P0.b defer input wait prototype | `qwen3_next_q4_p0b_notrace_cap64_n200.log` | 200 | 9.934 | 20.13 | Same vs pre-P0.b no-trace |
| MoEVM `-ngl99`, cap64, P0.b hardened defer input wait | `qwen3_next_q4_hardened_defer_on_cap64_n200.log` | 200 | 10.729 | 18.64 | Same vs P0.b prototype and fallback |
| MoEVM `-ngl99`, cap64, P0.c dynamic-K `ub=1` | `qwen3_next_q4_dynamick_ub1_cap64_n100.log` | 100 | 5.178 | 19.31 | Same first 100 IDs vs hardened |
| MoEVM `-ngl99`, cap128, P0.c dynamic-K long `ub=8` | `qwen3_next_q4_dynamick_long_ub8_cap128_n4_shadowfix.log` | 4 | 0.206 | 19.38 | Same IDs; bit-exact vs old `ub=8` fallback logits |
| MoEVM speculative, Qwen3-0.6B Q8 draft1 | `qwen3_next_q4_moevm_spec_qwen06b_q8_draft1_n32.log` | 34 | 2.537 | 13.40 | Accept 100%; below same-prompt non-spec |
| MoEVM speculative, DFlash Q8 draft8 p_min=0.5 | `qwen3_next_q4_moevm_spec_dflash_q8_draft8_pmin05_n32_guard.log` | 33 | 2.380 | 13.86 | Accept 76.19%; below same-prompt non-spec |
| MoEVM speculative, DFlash Q8 draft8 p_min=0.5 longer | `qwen3_next_q4_moevm_spec_dflash_q8_draft8_pmin05_n100.log` | 101 | 5.536 | 18.24 | Accept 81.71%; +5.2% vs same-prompt non-spec n100 |
| Audit-hardened MoEVM `-ngl99`, cap64, same prompt | `qwen3_next_q4_audit_normal_cap64_n100.log` | 100 | 5.631 | 17.76 | `pool_fallback=0`, `fallback_k_overflow=0` |
| Audit-hardened MoEVM `-ngl99`, cap128, same prompt | `qwen3_next_q4_audit_normal_cap128_n100.log` | 100 | 5.492 | 18.21 | `pool_fallback=0`, `fallback_k_overflow=0` |
| Audit-hardened DFlash cap64 draft8 p_min=0.5 | `qwen3_next_q4_audit_spec_dflash_cap64_draft8_pmin05_n100.log` | 101 | 7.068 | 14.29 | Accept 81.71%; `fallback_k_overflow=39`, cap too small |
| Audit-hardened DFlash cap128 draft8 p_min=0.5 | `qwen3_next_q4_audit_spec_dflash_cap128_draft8_pmin05_n100.log` | 101 | 5.031 | 20.08 | Accept 81.71%; `pool_fallback=0`, `fallback_k_overflow=0`; +10.3% vs audit cap128 non-spec |
| Audit second-pass smoke | `qwen3_next_q4_auditfix2_smoke_n1.log` | 1 | 0.058 | 17.28 | `pool_fallback=0`, `fallback_k_overflow=0`, `runtime_errors=0` |
| Audit third-pass smoke | `qwen3_next_q4_auditfix3_smoke_n1.log` | 1 | 0.064 | 15.63 | `pool_fallback=0`, `fallback_k_overflow=0`, `runtime_errors=0` |
| P0.d real context `-c4096`, cap128, `ub=8` | `qwen3_next_q4_p0d_c4096_ub8_cap128_n50.log` | 50 | 2.285 | 21.88 | 960-token prompt; `pool_fallback=0`, `fallback_k_overflow=0`, `runtime_errors=0` |
| P0.d no-trace logits parity, MoEVM cap128 | `qwen3_next_q4_p0d_parity_moevm_cap128_ignoreeos_n1024.log` | 1024 | 51.462 | 19.90 | vs stock `qwen3_next_q4_p0d_parity_stock_ignoreeos_n1024_v2.log`: 1024 rows, `max_abs=0.0`, zero token/argmax mismatches |
| P0.d jitter/fence logits stress, cap16 | `qwen3_next_q4_p0d_jitter_delay_cap16_n50.log` | 50 | 20.743 | 2.41 | Delay stress only; `pool_forced_fences=1`, `pool_pinned_eviction_skips=16`, logits `max_abs=0.0` vs no-delay cap16 |

Best old `-ngl0` comparison:

- JSON: `moevm/runs/qwen3_next_q4_decode100_ngl0_opoffload_b1_cap256_compare.json`
- Speedup: 5.12 / 3.03 = 1.69x
- `same_tokens=true`
- `callback_fallback=0`
- `pool_applied=10629`
- `pool_fallback=0`
- `pool_evictions=0`
- `copied_bytes=12659417088`

Best current `-ngl99` comparison:

- JSON: `moevm/runs/qwen3_next_q4_decode100_ngl99_sparse_mmap_cap64_kvfaauto_compare.json`
- Speedup: 13.08 / 4.24 = 3.08x
- `same_tokens=true`
- `callback_fallback=0`
- `pool_applied=14661`
- `pool_fallback=0`
- `pool_evictions=8269`
- `copied_bytes=21796110336`
- `normal_fallback=333`

P0.b scheduler wait comparison:

- Pre-P0.b log: `qwen3_next_q4_site_timing_notrace_cap64_n200.log`
- P0.b prototype log: `qwen3_next_q4_p0b_notrace_cap64_n200.log`
- P0.b hardened log: `qwen3_next_q4_hardened_defer_on_cap64_n200.log`
- Token parity: `same_tokens=true` across 200 generated token IDs.
- Throughput: 12.93 -> 20.13 tok/s prototype, 18.64 tok/s hardened.
- Decode wall time: 15.465s -> 9.934s prototype, 10.729s hardened.
- Hardened `split_input_wait_us`: 623, with `defer_input_wait_skips=33456`.
- Hardened `backend_sync_us`: 5,373,235; `ids_readback_us`: 5,335,033.
- Hardened pool guard counters: `pool_submit_epoch=33456`, `pool_completed_epoch=33453`, `pool_epoch_drains=11152`, `pool_forced_fences=0`, `pool_pinned_eviction_skips=0`.
- Mechanism: deferring the split input wait until a real fallback copy is needed removes the dominant scheduler drain for pool-patched MoE inputs.
- Hardening added after review: scheduler event mode disables defer, pool slot eviction is guarded by explicit submit/completed epochs, and pool/compact shadow IDs are patched atomically only after all fills succeed.

P0.b eviction stress:

- Log: `qwen3_next_q4_hardened_defer_on_cap16_n100.log`.
- Mode: same as hardened P0.b but `--moevm-pool-capacity 16`.
- Result: 100 tokens in 6.187s, 16.16 tok/s.
- Token parity: same first 100 generated IDs as the cap64 P0.b reference.
- Pool counters: `pool_evictions=32637`, `pool_forced_fences=0`, `pool_pinned_eviction_skips=0`.

P0.b pinning/fence validation:

- Self-test: `external/llama.cpp/build/bin/llama-moe-trace --moevm-self-test-pool-pinning`, passed.
- Delay/fence stress log: `qwen3_next_q4_delay_pinning_cap16_n50.log`.
- Mode: cap16, `--moevm-inject-delay-us 500`, plus one-shot `--moevm-test-force-pinned-fence` to exercise the no-evictable pinned path.
- Result: 50 tokens in 21.059s, 2.37 tok/s; this is a correctness stress, not a perf number.
- Token parity: same first 50 generated IDs as hardened cap64 reference.
- Guard counters: `pool_forced_fences=1`, `pool_pinned_eviction_skips=16`, `delay_injections=59203`, callback `delay_us=14818365`, runtime `delay_us=7421004`.

P0.c dynamic-K validation:

- Capacity guard is now applied to the deduplicated expert union (`expert_id_count`) instead of the logical `[k, n_tokens]` count. This prevents false fallback when the ubatch has repeated experts.
- `ub=1` gate: `qwen3_next_q4_dynamick_ub1_cap64_n100.log`, 100 tokens in 5.178s, 19.31 tok/s, `fallback_k_overflow=0`, `pool_fallback=0`.
- `ub=8` long prefill gate: `qwen3_next_q4_dynamick_long_ub8_cap128_n4_shadowfix.log`, 960-token prompt, 4 decode tokens in 0.206s, 19.38 tok/s, `fallback_k_overflow=0`, `pool_fallback=0`.
- Bit-exact replacement check: old `ub=8` fallback logits vs dynamic-K `ub=8` logits have `max_abs=0.0`. `ub=1` vs `ub=8` has a numeric backend delta (`max_abs=1.711`) but same argmax/tokens; that delta is not introduced by dynamic-K.
- Delay/fence `ub=8` stress: `qwen3_next_q4_dynamick_long_ub8_delay_cap128_n2_rss65g.log`, `pool_forced_fences=1`, `pool_pinned_eviction_skips=89`, `fallback_k_overflow=0`, `pool_fallback=0`.

MoEVM speculative validation:

- `llama-moe-trace` now has `--moevm-speculative` and uses llama.cpp `--spec-*` flags while keeping the target on the MoEVM sidecar path.
- DFlash-specific fixes: pass `ctx_other=ctx_tgt`, call `common_speculative_process()` after target decode, and skip `common_speculative_accept()` when high `p_min` produces no draft tokens.
- Same-prompt non-spec baselines: `qwen3_next_q4_moevm_normal_sameprompt_n32.log`, 32 tokens in 2.095s, 15.28 tok/s; `qwen3_next_q4_moevm_normal_sameprompt_n100.log`, 100 tokens in 5.764s, 17.35 tok/s.
- Best Qwen3-0.6B Q8 draft-simple result: `draft1`, 13.40 tok/s, accept 100%.
- Best short DFlash Q8 result: `draft8 p_min=0.5`, 13.86 tok/s, accept 76.19%, still below same-prompt non-spec n32.
- Audit rerun:
  - cap64: `qwen3_next_q4_audit_spec_dflash_cap64_draft8_pmin05_n100.log`, 101 tokens in 7.068s, 14.29 tok/s, accept 81.71%, `fallback_k_overflow=39`; cap64 is too small for DFlash verification on this prompt.
  - cap128: `qwen3_next_q4_audit_spec_dflash_cap128_draft8_pmin05_n100.log`, 101 tokens in 5.031s, 20.08 tok/s, accept 81.71%, `pool_fallback=0`, `fallback_k_overflow=0`.
- Verdict: speculation is integrated and starts to win on longer generation. After audit hardening, cap128 DFlash is 20.08 vs 18.21 tok/s on same-prompt non-spec n100 (+10.3%). Cap64 must not be used for DFlash because fallback dominates.

Graph-build scheduler hook validation:

- Branch: `codex/moevm-graph-build`.
- Mechanism: post-build graph transform rewrites routed `MUL_MAT_ID` expert weights to backend-resident `moevm_pool_*` tensors while preserving the original topk ids dependency. A scheduler-level graph-pool hook then forces a split before the first pool `MUL_MAT_ID` for each ids tensor, reads topk ids once, fills all layer component pools, patches `src[2]` to a shadow ids tensor for the split, and restores the graph after compute.
- Failed prototype to avoid: `qwen3_next_q4_graph_pool_c192_n50.log` used `cb_eval` for topk readback. It was functionally correct but only 2.07 tok/s because scheduler eval created thousands of tiny graph computes and 22.94s of backend sync.
- Smoke: `qwen3_next_q4_graph_sched_smoke_n1.log`, token identical to P0.d reference, `copy_hook_calls=0`, `graph_patch_applied=864`, `graph_topk_fills=519`, `runtime_errors=0`.
- Throughput: `qwen3_next_q4_graph_sched_c192_n50.log`, 50 tokens in 2.284s, 21.90 tok/s, `copy_hook_calls=0`, `callback_handled=8640`, `graph_fill_fallback=0`, `runtime_errors=0`.
- Logits parity: `qwen3_next_q4_graph_parity_off_n50.logits` vs `qwen3_next_q4_graph_parity_sched_n50.logits`, 50 rows, `max_abs=0.0`, zero token/argmax mismatches.
- Jitter/fence: `qwen3_next_q4_graph_sched_jitter_delay_cap16_n20.log`, cap16 with `--moevm-inject-delay-us 500` and `--moevm-test-force-pinned-fence`, `pool_forced_fences=1`, `pool_pinned_eviction_skips=15`, `runtime_errors=0`. Logits vs no-delay cap16 reference: `max_abs=0.0`.
- Interpretation: graph-build no longer depends on op-offload or the old expert copy hook, and it keeps correctness under the moved drain point. Current throughput is roughly tied with the best warm P0.d/P0.c runs, not yet the 25+ tok/s target; the next gain must come from reducing per-layer ids readback/sync cost or fusing layer rendezvous more aggressively.

Fast non-MoEVM baseline:

- Log: `qwen3_next_q4_baseline_decode100_ngl99_sparse_mmap_noopoffload_kvfaauto.log`
- Mode: `-ngl 99 -ot '.*ffn_.*_exps.weight=CPU' --no-op-offload`, default KV/Flash Attention behavior.
- Result: 100 tokens in 9.174s, 10.90 tok/s.
- This does not exercise MoEVM (`copy_hook_calls=0` in the equivalent sidecar run), but it is an important hybrid CPU-expert baseline unlocked by the sparse mmap loader patch.

## Negative Runs

- Before the sparse mmap loader patch, `-ngl 99` with expert CPU override hit Metal `kIOGPUCommandBufferCallbackErrorOutOfMemory`.
- `-ngl 4` also hit Metal OOM, including `-fa off -nkvo` and `-b 1 -ub 1`.
- `-ngl 0 --no-op-offload` runs but does not exercise the MoEVM copy hook.

## Next Technical Target

The Q4 sidecar path is correct and now faster than both the old `-ngl0` baseline and the same-placement `-ngl99 --op-offload` baseline. P0.b moved the prototype best to 20.13 tok/s; after epoch/pinning hardening, the mergeable best was 18.64 tok/s. P0.c removes the false `ub > 1` pool fallback, with the audit fix applying capacity to the deduplicated expert union. The graph-build scheduler hook removes dependency on op-offload/old copy-hook and reaches 21.67-21.90 tok/s with logits bit-exactness and jitter/fence validation. The audited same-prompt cap128 baseline is 18.21 tok/s, and audited DFlash cap128 reaches 20.08 tok/s with `fallback_k_overflow=0`. Next work should reduce per-layer ids readback/sync overhead, then rerun DFlash on top of graph-build before treating speculation as a 25-30 tok/s path.
