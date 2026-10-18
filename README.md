# MoEVM

**Expert paging for Mixture-of-Experts LLMs on Apple Silicon.** MoEVM runs MoE
models whose routed experts do not fit in GPU memory on a 48 GB Mac. Routed
expert weights are packed into a sidecar file and paged on demand into a
persistent Metal pool, inside a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp).
The output is checked bit for bit against llama.cpp with MoEVM turned off.

- **Qwen3-Coder-Next Q4_K_M (45.2 GiB, more than the 40.2 GB Metal working-set
  limit of this Mac):** **6.41 → 19.90 tok/s** decode (3.1×) against the same
  llama.cpp build with MoEVM off: same prompt, 1024 tokens, routing traces off
  on both sides. Logits are bit-identical at all 1024 positions over the full
  151,936-token vocabulary.
- **gpt-oss-120b Q4_K_M (58 GiB, more than the 48 GB of RAM):** **8.72 tok/s**
  decode, median of 5 runs at a 4096-token context with a 990-token prompt, with
  zero demand reads and no memory-pressure events. Four parallel sequences
  reach 26.81 tok/s in aggregate.
- **Correctness gate:** logits parity `max_abs = 0.0` on both models.

Status: research prototype, measured on one machine. The 6-hour stability soak
was never completed (see [Status](#status)).

## Results

Machine: MacBook Pro, Apple M5 Pro, 48 GB unified memory. Greedy decoding
(`--temp 0`); single-sequence rows use `-b 1 -ub 1`. Throughput is the decode
tok/s printed by `llama-moe-trace`, with routing traces off.

| Model | Configuration | Decode tok/s | Correctness | Source |
|---|---|---:|---|---|
| Qwen3-Coder-Next Q4_K_M, 45.2 GiB | llama.cpp `--op-offload`, MoEVM off, 1024 tokens | 6.41 | reference | [log](runs/evidence/qwen3_next_q4_p0d_parity_stock_ignoreeos_n1024_v2.log) |
| | **MoEVM pool**, same command plus `--moevm-*` flags | **19.90** | logits bit-identical, 1024 rows | [log](runs/evidence/qwen3_next_q4_p0d_parity_moevm_cap128_ignoreeos_n1024.log), [parity](runs/qwen3_next_q4_p0d_parity_stock_vs_moevm_n1024.json) |
| gpt-oss-120b Q4_K_M, 58 GiB | **MoEVM graph pool**, 24 GiB arena, 4096 context, 1 sequence | **8.72** (median of 5) | `demand_reads=0`, `runtime_errors=0` in all 5 runs | [runs](runs/evidence/gpt_oss_120b_median5_observability_24g_20260705/), [summary](runs/gpt_oss_120b_median5_observability_24g_20260705.md) |
| | MoEVM graph pool, 4 parallel sequences, same prompt | 26.81 aggregate (6.70 per sequence) | `runtime_errors=0` | [log](runs/evidence/gpt_oss_120b_parallel4_observability_n32.log), [summary](runs/gpt_oss_120b_parallel4_aggregate_20260705.md) |

Logits parity (full-vocabulary logits compared at every decoded position):

| Model | Comparison | Rows | `max_abs` | Source |
|---|---|---:|---:|---|
| Qwen3-Coder-Next Q4_K_M | MoEVM off vs on (the head-to-head above) | 1024 | 0.0 | [json](runs/qwen3_next_q4_p0d_parity_stock_vs_moevm_n1024.json) |
| Qwen3-Coder-Next Q4_K_M | Later graph-build pool, off vs on | 50 | 0.0 | [json](runs/evidence/qwen3_next_q4_graph_parity_off_vs_sched_n50.json) |
| gpt-oss-120b Q4_K_M | Same-backend Metal reference vs MoEVM | 64 | 0.0 | [json](runs/evidence/gpt_oss_120b_q4_metal_opoffload_vs_moevm_n64_logits_compare.json) |

How to read these numbers:

- The Qwen pair differs only by the `--moevm-*` flags: same fork binary
  (`llama-moe-trace`), prompt, seed, 1536-token context, and `-ngl 99
  --op-offload` with routed experts on CPU. The baseline relies on the
  sparse-mmap loader fix from
  [#25309](https://github.com/ggml-org/llama.cpp/pull/25309); without it this
  placement ran out of Metal memory.
- MoEVM does not beat llama.cpp in general. On this machine, llama.cpp on CPU
  only (`-ngl 0`, 512–1024-token context, prompts of 47–67 tokens, even with
  routing traces on) decodes gpt-oss-120b at 10.85–14.36 tok/s
  ([summary](runs/gpt_oss_120b_q4_benchmark_summary.md#trace)). That is a
  shorter context than the 8.72 tok/s run, so the two are not directly
  comparable. MoEVM's gain is against the GPU/Metal paths, with bounded memory.
- The parallel figure is aggregate throughput over 32 decode steps per
  sequence; it does not lower single-sequence latency.
- The lab notes in [`phase0/RUNBOOK.md`](phase0/RUNBOOK.md) and
  [`runs/*.md`](runs/) record every intermediate measurement, including runs
  with routing traces on, warm-cache replays and superseded figures. The tables
  above are the reference numbers. In particular, the best earlier gpt-oss
  single runs were faster than 8.72 tok/s; the median of five at 4096 context
  replaces them.
- [`runs/evidence/`](runs/evidence/) holds the raw logs behind each row, with
  only local paths redacted.

## How it works

1. Inspect the GGUF tensor table and identify the routed expert tensors.
2. Build a model capability profile, so modules are enabled by tensor layout
   rather than by hardcoded model names
   ([`phase0/MODEL_CAPABILITIES.md`](phase0/MODEL_CAPABILITIES.md)).
3. Pack the experts into a sidecar indexed by `(layer, expert, component)`,
   with size and SHA-256 prefix checks in the index.
4. Allocate a persistent Metal pool and shadow expert ids for the selected
   experts.
5. Rewrite the routed `MUL_MAT_ID` nodes in the compute graph to read from the
   pool. A scheduler hook reads the top-k expert ids once per layer, fills the
   pool, patches the graph for that split and restores it after compute.
6. Fill the pool from the sidecar with LRU residency, candidate prefetch,
   epochs, pinning and atomic graph commits.
7. Use `MADV_FREE_REUSABLE` and pressure-triggered trimming to shrink the macOS
   footprint before memory pressure becomes fatal.
8. Treat logits parity as the correctness gate, not only token matching.

On 512-token coding traces of gpt-oss with a 28 GiB budget, plain LRU hits
95.10–95.75% of expert accesses against 97.15–97.49% for the offline-optimal
Belady policy ([RUNBOOK](phase0/RUNBOOK.md)), so further gains are expected from
prefetch timing, orchestration and copied bytes rather than from a smarter
cache policy.
Metal timestamp diagnostics show the GPU far from saturated: the bottleneck is
CPU and scheduler rendezvous, mostly expert-id readback.

## Repository layout

| Path | Contents |
|---|---|
| [`phase0/`](phase0/) | Python tooling (GGUF inspection, capability profile, sidecar plan/pack/index, cache and prefetch simulators, benchmark and quality harnesses, `safe_run.py` memory guard) and the standalone C++ runtime (`moevm_runtime.*`, replay tools). Standard library only. |
| [`runs/`](runs/) | Benchmark summaries, comparison JSONs, prompts, and [`runs/evidence/`](runs/evidence/). |
| [`docs/`](docs/) | [FAQ](docs/FAQ.md) and the pre-registered [soak criteria](docs/SOAK_CRITERIA.md). |
| llama.cpp fork | Runtime integration: graph rewrite, scheduler hook, sidecar runtime, `llama-moe-trace`. |

## llama.cpp fork and upstream work

- Fork branch: [`Takinggg/llama.cpp@codex/moevm-graph-build`](https://github.com/Takinggg/llama.cpp/tree/codex/moevm-graph-build).
  The tag [`moevm-p0d-audit-hardened`](https://github.com/Takinggg/llama.cpp/releases/tag/moevm-p0d-audit-hardened)
  marks the audited P0.d runtime, the `--op-offload`-based path that produced
  the Qwen head-to-head, before the graph-build pool.
- Upstream: draft PR [ggml-org/llama.cpp#25309](https://github.com/ggml-org/llama.cpp/pull/25309),
  *loader: map sparse mmap tensor ranges*. It maps only the tensor ranges a
  Metal buffer needs instead of one span from the first to the last tensor of a
  GGUF file. Without it, `-ngl 99` with experts on CPU ran out of Metal memory
  on this machine, so the Qwen baseline above depends on it.

## Reproduce

Requirements: Apple Silicon Mac with 48 GB of unified memory, Xcode command
line tools, CMake, Python 3, the Hugging Face CLI (`hf`) for downloads, and
about 90 GiB of free disk for the Qwen GGUF plus its packed sidecar. Run
everything from the repository root.

Paths are configurable: `MOEVM_MODELS_DIR` (default `~/models`) points at the
models and sidecars, and `MOEVM_LLAMA_BIN` overrides the default
`external/llama.cpp/build/bin/llama-moe-trace`.

### 1. Build the fork

```bash
git clone https://github.com/Takinggg/moevm.git && cd moevm
git clone --branch codex/moevm-graph-build https://github.com/Takinggg/llama.cpp external/llama.cpp
cmake -S external/llama.cpp -B external/llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON
cmake --build external/llama.cpp/build -j
```

### 2. Download Qwen3-Coder-Next and build its sidecar

```bash
export MOEVM_MODELS_DIR=~/models
Q="$MOEVM_MODELS_DIR/qwen3-coder-next-q4"
mkdir -p "$Q"
hf download unsloth/Qwen3-Coder-Next-GGUF Qwen3-Coder-Next-Q4_K_M.gguf --local-dir "$Q"

python3 phase0/model_profile.py "$Q/Qwen3-Coder-Next-Q4_K_M.gguf" --md runs/qwen3_next_q4_profile.md
python3 phase0/sidecar_plan.py --out runs/qwen3_next_q4_sidecar_plan.json "$Q/Qwen3-Coder-Next-Q4_K_M.gguf"
python3 phase0/safe_run.py --log runs/qwen3_next_q4_pack.log -- \
  python3 phase0/sidecar_pack.py --plan runs/qwen3_next_q4_sidecar_plan.json \
  --out "$Q/qwen3-next-q4.moevm" --nocache
python3 phase0/sidecar_index.py --plan runs/qwen3_next_q4_sidecar_plan.json \
  --sidecar "$Q/qwen3-next-q4.moevm" --out "$Q/qwen3-next-q4.moevm.idx"
python3 phase0/sidecar_component_index.py --plan runs/qwen3_next_q4_sidecar_plan.json \
  --sidecar "$Q/qwen3-next-q4.moevm" --out "$Q/qwen3-next-q4.moevm.cidx"
```

`sidecar_pack.py` must run with `--nocache` under `safe_run.py`; otherwise
writing 43.69 GiB fills the unified-memory file cache.

### 3. Run the Qwen head-to-head

The head-to-head was measured on the audited P0.d runtime. To rebuild that
exact runtime first:

```bash
git -C external/llama.cpp checkout moevm-p0d-audit-hardened
cmake --build external/llama.cpp/build -j
```

MoEVM off (the 6.41 tok/s row):

```bash
python3 phase0/safe_run.py --log runs/repro_qwen_q4_off.log --max-seconds 900 -- \
  /usr/bin/env GGML_OP_OFFLOAD_MIN_BATCH=1 external/llama.cpp/build/bin/llama-moe-trace \
  -m "$Q/Qwen3-Coder-Next-Q4_K_M.gguf" \
  --logits-out runs/repro_qwen_q4_off.logits \
  -ngl 99 -ot '.*ffn_.*_exps.weight=CPU' --op-offload --no-repack \
  -c 1536 -b 1 -ub 1 -n 1024 --ignore-eos --temp 0 -s 1 \
  -p 'Write a Python function that parses log lines into records.'
```

MoEVM on (the 19.90 tok/s row): the same command plus the sidecar flags.

```bash
python3 phase0/safe_run.py --log runs/repro_qwen_q4_moevm.log --max-seconds 900 -- \
  /usr/bin/env GGML_OP_OFFLOAD_MIN_BATCH=1 external/llama.cpp/build/bin/llama-moe-trace \
  -m "$Q/Qwen3-Coder-Next-Q4_K_M.gguf" \
  --logits-out runs/repro_qwen_q4_moevm.logits \
  --moevm-index "$Q/qwen3-next-q4.moevm.idx" \
  --moevm-component-index "$Q/qwen3-next-q4.moevm.cidx" \
  --moevm-sidecar "$Q/qwen3-next-q4.moevm" \
  --moevm-arena-gib 1 --moevm-inflight-mib 64 --moevm-threads 4 --moevm-prefetch-topk 10 \
  --moevm-copy-mode memcpy --moevm-cache --moevm-pool-layer all --moevm-pool-capacity 128 \
  -ngl 99 -ot '.*ffn_.*_exps.weight=CPU' --op-offload --no-repack \
  -c 1536 -b 1 -ub 1 -n 1024 --ignore-eos --temp 0 -s 1 \
  -p 'Write a Python function that parses log lines into records.'
```

Compare the logits:

```bash
python3 phase0/compare_logits.py runs/repro_qwen_q4_off.logits runs/repro_qwen_q4_moevm.logits --json runs/repro_parity.json
```

Each `.logits` file is about 600 MiB. The exact command lines of the reference
runs are the first line of each log in [`runs/evidence/`](runs/evidence/).
Switch back to the branch (`git -C external/llama.cpp checkout
codex/moevm-graph-build` and rebuild) for the gpt-oss runs.

### 4. gpt-oss-120b median-of-5

Download the Unsloth gpt-oss-120b Q4_K_M GGUF (two shards) into
`$MOEVM_MODELS_DIR/gpt-oss-120b/Q4_K_M/`, build its sidecar as in step 2
(`gpt-oss-120b-q4.moevm`, `.idx`, `.cidx`), then:

```bash
python3 phase0/gptoss_soak.py --iterations 5 --tokens 128 --ctx 4096 \
  --arena-gib 24 --trim-min-gib 12 --pool-capacity 16 --prefetch-topk 4 \
  --moevm-threads 4 --copy-mode memcpy --logits-every 0 --prompt-names observability \
  --out-dir runs/repro_gptoss_median5
```

The harness runs a model capability gate first, wraps every run in
`safe_run.py`, and writes `results.jsonl` and `summary.json`. Metal shapes that
crashed this machine earlier stay blocked by `safe_run.py`.

## Limits

- Routed MoE weights only. Dense models have no small active set per token, so
  the same paging does not apply; a dense 1T model will not run interactively on
  a 48 GB laptop.
- Measured on one machine. Other Apple Silicon configurations are untested.
- MoEVM is not faster than llama.cpp in every setup: CPU-only llama.cpp decodes
  gpt-oss-120b at 10.85–14.36 tok/s with short contexts on this machine. The
  measured win is against the GPU/Metal paths, with bounded memory.
- Qwen3-Coder-Next (45.2 GiB) fits in the 48 GB of RAM; what it exceeds is
  Metal's 40.2 GB working-set limit. Only gpt-oss-120b is larger than RAM.
- The packer supports split `gate/up/down` routed expert tensors. Fused
  `gate_up` or custom expert layouts need a layout adapter first.
- Every new model needs a capability profile and a same-backend logits parity
  run before any throughput claim.
- Throughput depends on the workload's expert locality.
- The headline proof is 58 GiB of weights on 48 GB of RAM, not a claim about
  arbitrary RAM multiples.
- Larger batch modes can raise aggregate throughput but do not reproduce the
  exact `b=1` token stream.
- Speculative decoding is not part of the measured path for gpt-oss.

## Status

Research prototype; the latest measurements date from July 2026.

Done:

- sidecar packing with integrity-checked indexes;
- graph-build Metal pool with scheduler hook, epochs and pinning;
- logits parity gates, plus jitter and fence stress tests;
- `safe_run.py` memory guard for macOS, after earlier OOM and watchdog crashes;
- gpt-oss-120b (58 GiB) decoding on a 48 GB Mac.

Not done:

- **The 6-hour soak was never completed.** Only a pilot of about 30 minutes
  ran on 2026-07-05 before it was stopped on purpose, and its validation was
  never finished. The final soak defined in
  [`docs/SOAK_CRITERIA.md`](docs/SOAK_CRITERIA.md) has not been run, so
  long-run stability is not established.
- Rewriting expert-id readback and orchestration, the main remaining
  throughput lever. It moves the epoch fence and needs a full parity and soak
  campaign.
- Upstreaming beyond the loader PR.

## License

MIT, see [LICENSE](LICENSE). The llama.cpp fork keeps llama.cpp's MIT license;
its notice is reproduced in the same file.
