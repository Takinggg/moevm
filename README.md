# MoEVM

MoEVM is a runtime experiment for running Mixture-of-Experts LLMs whose routed
expert weights are larger than the machine's RAM.

Instead of loading a MoE model as one monolithic blob, MoEVM stores routed expert
weights in a packed sidecar and keeps only the experts needed by the current
tokens in a persistent Metal pool.

The practical claim is narrow and testable:

> For MoE models, model size can exceed RAM when routed experts are paged on
> demand. Throughput depends on locality, prefetch, and orchestration, not only
> on the total parameter count.

## Current Launch Candidate

Machine: MacBook Pro M5 Pro, 48 GB RAM.

| Model | Source size | Working RAM budget | Result | Correction gate |
|---|---:|---:|---:|---|
| Qwen3-Coder-Next Q4 | about 45 GiB | 48 GB MacBook | 4.24 tok/s stock op-offload -> 21.67-21.90 tok/s MoEVM graph-build | same-backend logits `max_abs=0.0`; jitter/fence validated |
| gpt-oss-120b Q4_K_M | 58 GiB | 48 GB MacBook, arena 16-24 GiB | 11.44-13.34 tok/s short prompt; 9.5-10.4 tok/s at `c=4096` with about 990 prompt tokens | Metal-vs-Metal op-offload baseline vs MoEVM `max_abs=0.0` over 64 rows |

The gpt-oss path is the launch proof: a 58 GiB MoE model decoding on a 48 GB
laptop without loading the full routed expert pool into RAM.

## How It Works

1. Inspect the GGUF tensor table and identify routed expert tensors.
2. Build a model capability profile so modules are enabled by tensor layout, not
   by hardcoded model names.
3. Pack experts into a sidecar indexed by `(layer, expert, component)`.
4. Build a persistent Metal pool and shadow ids for selected experts.
5. Patch `MUL_MAT_ID` expert nodes in the graph to read from the pool.
6. Fill the pool from the sidecar with LRU residency, candidate prefetch,
   epochs, pinning, and atomic graph commits.
7. Use `MADV_FREE_REUSABLE` plus pressure-triggered trimming to reduce macOS
   footprint before jetsam pressure becomes fatal.
8. Treat logits parity as the correctness gate, not only token matching.

For gpt-oss specifically, `ADD_ID` expert bias nodes are intentionally not
rewritten; same-backend Metal-vs-Metal logits parity rules out the bias-corruption
failure mode on the tested path.

## Reproduce

Build the local llama.cpp fork:

```bash
cmake -S external/llama.cpp -B external/llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON
cmake --build external/llama.cpp/build -j 16
```

Run the bounded gpt-oss launch path through the safety wrapper:

```bash
python3 moevm/phase0/safe_run.py \
  --allow-gptoss-metal-scale-gate \
  --log moevm/runs/repro_gptoss_120b.log \
  --summary moevm/runs/repro_gptoss_120b.safe_summary \
  --max-seconds 900 \
  --preflight-min-free-pct 25 \
  --min-free-pct 8 \
  --max-rss-mib 42000 \
  --max-wired-gib 38 \
  --max-log-mb 96 \
  -- /usr/bin/env GGML_OP_OFFLOAD_MIN_BATCH=1 \
  external/llama.cpp/build/bin/llama-moe-trace \
  -m $MOEVM_MODELS_DIR/gpt-oss-120b/Q4_K_M/gpt-oss-120b-Q4_K_M-00001-of-00002.gguf \
  --moevm-index $MOEVM_MODELS_DIR/gpt-oss-120b/Q4_K_M/gpt-oss-120b-q4.moevm.idx \
  --moevm-component-index $MOEVM_MODELS_DIR/gpt-oss-120b/Q4_K_M/gpt-oss-120b-q4.moevm.cidx \
  --moevm-sidecar $MOEVM_MODELS_DIR/gpt-oss-120b/Q4_K_M/gpt-oss-120b-q4.moevm \
  --moevm-arena-gib 24 \
  --moevm-inflight-mib 64 \
  --moevm-threads 4 \
  --moevm-prefetch-topk 4 \
  --moevm-copy-mode memcpy \
  --moevm-pool-layer all \
  --moevm-pool-capacity 16 \
  --moevm-graph-mode pool \
  --moevm-no-integrity-check \
  --moevm-pressure-trim \
  --moevm-pressure-trim-step-gib 2 \
  --moevm-pressure-trim-min-gib 12 \
  -ngl 99 \
  -ot '.*ffn_.*_exps.weight=CPU' \
  --no-op-offload \
  --no-repack \
  -fa off \
  -c 4096 \
  -b 1 \
  -ub 1 \
  -n 64 \
  --ignore-eos \
  --temp 0 \
  -s 7000 \
  -f moevm/runs/prompt_observability_512plus.txt
```

Run the launch soak:

```bash
python3 moevm/phase0/gptoss_soak.py \
  --hours 6 \
  --tokens 64 \
  --logits-every 20 \
  --out-dir moevm/runs/gpt_oss_120b_launch_soak
```

Validate it:

```bash
python3 moevm/phase0/validate_soak.py \
  moevm/runs/gpt_oss_120b_launch_soak \
  --final
```

## Limits

- This is for routed MoE weights, not dense models.
- It does not make a 1T dense model interactive on a 48 GB laptop.
- Current production packing supports split `gate/up/down` routed expert
  tensors. Fused `gate_up` or custom expert layouts need a layout adapter before
  graph-pool is enabled.
- Every new model requires `model_profile.py` plus same-backend logits parity;
  benchmarked models are examples, not hardcoded special cases.
- Throughput is workload-dependent. Expert locality matters.
- The launch proof is 58 GiB of source weights on 48 GB RAM, not an arbitrary
  proof that every MoE can run at any RAM multiple.
- Replay tok/s can be page-cache contaminated; publish live decode tok/s and
  correction gates instead.
- Batch modes can be useful for aggregate throughput, but they may not produce
  the exact same token stream as `b=1 ub=1`.
- gpt-oss speculation is not part of the current launch path.
- Unsafe gpt-oss Metal shapes remain blocked by `safe_run.py` after earlier OOM
  and watchdog failures on this machine.

## Scaling Notes

The current gpt-oss launch point is:

- 58 GiB source model.
- 56.88 GiB routed expert pool.
- 1.57 GiB non-routed resident estimate.
- 12.64 MiB per `(layer, expert)` record.
- 4 active experts per token.
- 24 GiB launch arena.

Measured 512-token coding traces show that LRU is already close to offline
optimal Belady on this workload. At 28 GiB, LRU hit rate is about 95.1-95.8% and
Belady is about 97.1-97.5%. That means new cache policies are unlikely to be the
main source of a large speedup. The better targets are prefetch timing,
orchestration/readback, and reducing copied expert bytes.

Rule of thumb for this implementation:

- More RAM helps until the active expert working set fits.
- Past that point, larger arenas can hurt by taking memory away from the OS page
  cache and Metal working set.
- Bigger-than-RAM MoE inference is viable when expert locality is high enough
  and the active set per token stays small.
- Dense models do not have this sparsity, so they do not get the same scaling
  behavior.

The current launch numbers should be read as a measured point in that scaling
law, not as a universal promise for all MoE architectures.

## Launch Gates

See `moevm/launch/LAUNCH_CRITERIA.md`.

The launch branch requires:

- same-backend logits parity for the headline path;
- a 6h gpt-oss soak with `runtime_errors=0`;
- no pressure-critical events;
- no jetsam events;
- stable throughput across the first and last hour.

## Roadmap

The next large tok/s gains are measured but not yet all productized:

- gpt-oss phase counters and token-ahead prefetch.
- Qwen graph-build DFlash follow-up; current measured DFlash graph path is
  functional but not yet a major accelerator.
- ids-readback/orchestration surgery after launch. GPU busy is far below full
  utilization on gpt-oss, but this path moves the epoch fence and needs a full
  correctness campaign.
- Demand-shaping/top-k quality gates after launch.

See `moevm/launch/WOW_SPRINT.md` for the bounded pre-launch scope.

## Launch Package

- `moevm/launch/FAQ.md`
- `moevm/launch/VIDEO_STORYBOARD.md`
- `moevm/launch/POST_DRAFTS.md`
