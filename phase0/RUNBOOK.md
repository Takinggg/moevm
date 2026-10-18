# MoEVM Phase 0 Runbook

This runbook is for the current MacBook Pro M5 Pro 48GB.

## Current Local Facts

- macOS 26.5.1.
- Apple M5 Pro, 18 cores.
- RAM: 48GB.
- Free disk after model cleanup: about 310GiB.
- `iogpu.wired_limit_mb=0`.
- Native Metal build reports `recommendedMaxWorkingSetSize = 40200.90 MB`.
- Ollama 0.31.1.
- No `llama-server` / `llama-cli` in PATH before Phase 0 setup.
- Native llama.cpp has now been built in `external/llama.cpp`.
- `llama-moe-trace` has been added locally under `external/llama.cpp/examples/moe-trace` and built into `external/llama.cpp/build/bin/llama-moe-trace`.
- Ollama local model list is empty.
- Docker VM memory reported around 31.3GiB, too high for paging experiments.
- `gpt-oss-120b Q4_K_M` is downloaded in `$MOEVM_MODELS_DIR/gpt-oss-120b/Q4_K_M`.
- `gpt-oss-120b Q4_K_M` corrected multi-shard expert pool: 56.88GiB; `(layer, expert)` slices are uniform at 12.64MiB.
- `Qwen3-Coder-Next-UD-Q3_K_XL.gguf` is downloaded in `$MOEVM_MODELS_DIR/qwen3-coder-next`.
- Packed Qwen3Next Q3 sidecar: `$MOEVM_MODELS_DIR/qwen3-coder-next/qwen3-next-q3.moevm`, 31.35GiB.
- `Qwen3-Coder-Next-Q4_K_M.gguf` is downloaded in `$MOEVM_MODELS_DIR/qwen3-coder-next-q4`, about 45GiB.

## Safety Status

Do not continue Metal runtime tuning on `gpt-oss-120b Q4_K_M` by trial and error on this Mac.

Observed failures:

- `llama-bench` does not expose `--no-repack`; CPU-MoE tests can trigger large anonymous-memory copies for MXFP4 experts.
- `llama-cli --no-repack -ngl 99 -ncmoe 36 -fa on` avoided the earlier global Jetsam pattern but hit Metal `kIOGPUCommandBufferCallbackErrorOutOfMemory`.
- A reduced `llama-cli --no-repack -ngl 16 -ncmoe 36 -fa on` run was followed by a macOS watchdog kernel panic. The panic report names `WindowServer`, not llama.cpp directly, but the test path is unsafe.

Allowed next steps:

- Offline GGUF inspection.
- SSD read measurement only when cache state is declared.
- Route tracing on smaller/lab models first.
- Cache simulation.
- Controlled CPU-expert arena prototype that avoids large Metal residency.
- Run risky commands only through `phase0/safe_run.py`.

Blocked for now:

- Further `gpt-oss-120b` Metal `-ngl` / `-fa` tuning.
- Any run likely to allocate near the Metal recommended working set.
- Any benchmark without a pre-flight check proving Docker VM is down and no LLM process is already running.
- Parallel full-model Python reads on GGUF files above 30GiB. This caused severe macOS memory pressure on 2026-07-04 when `gguf_inspect.py` and `sidecar_plan.py` both touched the Q4 GGUF before the parser was changed to stream the header.

## Safe Runner

Use `safe_run.py` for every risky benchmark or prototype run:

```bash
python3 phase0/safe_run.py \
  --log runs/<run>.log \
  --summary runs/<run>.summary \
  --max-seconds 600 \
  --min-free-pct 8 \
  --max-wired-gib 34 \
  --max-rss-mib 38000 \
  --max-log-mb 64 \
  -- <command> <args...>
```

What it does:

- Refuses to start if Docker/Virtualization VM is running, unless `--allow-docker` is passed.
- Refuses to start when another model I/O job is already active, unless `--allow-concurrent-model-io` is passed for tiny controlled tests.
- Requires `sidecar_pack.py` to use `--nocache`, so packed-sidecar creation does not intentionally fill the unified-memory file cache.
- Refuses known-dangerous `gpt-oss-120b` + Metal command shapes unless `--allow-known-unsafe` is passed.
- Allows bounded gpt-oss Metal only through narrow explicit gates:
  - `--allow-gptoss-metal-smoke`: `-n <= 8`, MoEVM arena <= 16GiB.
  - `--allow-gptoss-metal-parity-gate`: `-n <= 128`, MoEVM arena <= 16GiB.
  - `--allow-gptoss-metal-scale-gate`: `-n <= 128`, MoEVM arena <= 24GiB, and `--moevm-pressure-trim` is required above 16GiB.
- Kills the whole process group on low free memory, high wired memory, excessive RSS, excessive swapout delta, timeout, oversized logs, or Metal OOM log patterns.
- Writes a compact summary with the stop reason and log tail.

Limits:

- It cannot guarantee protection from an instantaneous kernel/Metal driver panic.
- A process in macOS uninterruptible kernel I/O wait may remain visible briefly after `SIGKILL`; the wrapper records this instead of hanging forever.
- It is a safety fuse, not proof that a command is safe.
- For `gpt-oss-120b`, keep the static block enabled. Do not pass `--allow-known-unsafe` on this Mac unless the command is intentionally part of a controlled crash reproduction.

## Docker Memory

During MoEVM inference tests, stop Docker Desktop or cap it to 4-6GiB.

Safer stop/start:

```bash
docker desktop stop
docker desktop start
```

Cap setting when Docker is stopped:

```bash
python3 - <<'PY'
import json, pathlib
p = pathlib.Path.home() / "Library/Group Containers/group.com.docker/settings-store.json"
d = json.loads(p.read_text())
d["MemoryMiB"] = 6144
p.write_text(json.dumps(d, indent=1))
print("MemoryMiB ->", d["MemoryMiB"])
PY
```

## Inspect Hugging Face Without Full Model Download

List candidate repos:

```bash
curl -sL "https://huggingface.co/api/models?search=Qwen3-Coder-Next&limit=50" \
  | jq -r '.[].id'
```

List GGUF files in the Unsloth repo:

```bash
curl -sL "https://huggingface.co/api/models/unsloth/Qwen3-Coder-Next-GGUF/tree/main?recursive=true" \
  | jq -r '.[] | select(.path|endswith(".gguf")) | "\(.size/1000000000)GB\t\(.path)"' \
  | sort -n
```

Download only the first 64MiB of a GGUF:

```bash
mkdir -p phase0/headers
curl -L -r 0-67108863 \
  -o phase0/headers/qwen3-coder-next-q4km.head.gguf \
  "https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF/resolve/main/Qwen3-Coder-Next-Q4_K_M.gguf"
python3 phase0/gguf_inspect.py \
  phase0/headers/qwen3-coder-next-q4km.head.gguf \
  --json phase0/headers/qwen3-coder-next-q4km.inspect.json
```

If the tensor-info table is larger than 64MiB, repeat with 128MiB:

```bash
curl -L -r 0-134217727 -o phase0/headers/q4.head128.gguf "<GGUF_URL>"
```

## Build llama.cpp Native Metal

Recommended location in this workspace:

```bash
mkdir -p external
git clone --depth 1 https://github.com/ggml-org/llama.cpp external/llama.cpp
cmake -S external/llama.cpp -B external/llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON
cmake --build external/llama.cpp/build -j 16
external/llama.cpp/build/bin/llama-cli --version
```

Current build:

- commit: `2d97363`
- Metal backend included.
- `qwen3next.cpp` compiled.
- useful binaries:
  - `external/llama.cpp/build/bin/llama-cli`
  - `external/llama.cpp/build/bin/llama-server`
  - `external/llama.cpp/build/bin/llama-bench`
  - `external/llama.cpp/build/bin/llama-eval-callback`

## Download Strategy

Do not download Q4 first.

1. Download a Q3-class quant to validate kernels and quality with lower pressure.
2. Trace routing and simulate cache.
3. Download Q4 only if the gates pass.

Example:

```bash
mkdir -p ~/models/qwen3-coder-next
hf download unsloth/Qwen3-Coder-Next-GGUF Qwen3-Coder-Next-UD-Q3_K_XL.gguf \
  --local-dir ~/models/qwen3-coder-next
```

## Smoke Test Template

This template is for lab models such as Qwen3-Coder-Next Q3, not for `gpt-oss-120b Q4_K_M`.

```bash
B=external/llama.cpp/build/bin
M=~/models/qwen3-coder-next/Qwen3-Coder-Next-UD-Q3_K_XL.gguf
mkdir -p runs

$B/llama-bench -m "$M" -fa 1 -p 2048 -n 128 -r 3 --no-warmup -o md \
  | tee runs/q3_bench.md

# `llama-bench` uses -ncmoe / --n-cpu-moe rather than --cpu-moe.
# To test MoE CPU placement for all 48 layers:
$B/llama-bench -m "$M" -fa 1 -p 2048 -n 128 -r 3 --no-warmup -o md -ncmoe 48 \
  | tee runs/q3_bench_ncpu_moe.md

/usr/bin/time -l $B/llama-cli -m "$M" -ngl 99 -fa on -c 8192 \
  -ctk q8_0 -ctv q8_0 --no-warmup -n 256 \
  -p "Write a concise Python function that parses key=value lines." \
  2>&1 | tee runs/q3_cli.log

# `llama-cli` exposes -cmoe / --cpu-moe:
/usr/bin/time -l $B/llama-cli -m "$M" -ngl 99 -fa on -c 8192 \
  -ctk q8_0 -ctv q8_0 --no-warmup --cpu-moe -n 256 \
  -p "Write a concise Python function that parses key=value lines." \
  2>&1 | tee runs/q3_cli_cpu_moe.log
```

In parallel:

```bash
iostat -d disk0 1 | tee runs/q3_iostat.log
vm_stat 5 | tee runs/q3_vmstat.log
```

Measure SSD random reads once a GGUF is downloaded:

```bash
python3 phase0/ssd_bench.py "$M" --size-mb 2 --threads 6
```

## MoE Route Tracing

Local trace binary:

```bash
external/llama.cpp/build/bin/llama-moe-trace --help
```

Trace output format is JSONL:

```json
{"token":0,"layer":0,"experts":[12,44,91,103]}
```

Always run trace jobs through `safe_run.py`:

```bash
python3 phase0/safe_run.py \
  --log runs/<name>.log \
  --summary runs/<name>.summary \
  --max-seconds 600 \
  --max-rss-mib 16000 \
  -- external/llama.cpp/build/bin/llama-moe-trace \
    --trace-out runs/<name>.jsonl \
    -m /path/to/model.gguf \
    -ngl 0 -c 512 -b 64 -ub 64 \
    -p "def quicksort(arr):"
```

For trace correctness in v0, set `-b` equal to `-ub`. If llama.cpp internally splits a logical batch into smaller ubatches, the eval callback does not expose a reliable token offset. Traces produced with `-b 32 -ub 16` are invalid for per-token cache simulation unless repaired.

Validation run:

- Model: `$MOEVM_MODELS_DIR/moe-trace-lab/Mistral-7B-v0.3-2x-MoE-v1.Q2_K.gguf`
- Mode: CPU-only, `-ngl 0`.
- Prompt tokens traced: 5.
- Records: 155.
- Explanation: this model has 32 layers, but only layers 0-30 emitted `ffn_moe_topk`, so integrity is `tokens × routed_layers`, not `tokens × total_layers`.
- Trace: `runs/mistral_2x_moe_trace.jsonl`.
- Cache sim: `runs/mistral_2x_moe_cache_sim.json`.

This lab model is not a useful MoEVM target because it has 2 active experts out of 2. It only validates the trace hook and downstream simulator.

Qwen3-Coder-Next Q3 validation:

- Model: `$MOEVM_MODELS_DIR/qwen3-coder-next/Qwen3-Coder-Next-UD-Q3_K_XL.gguf`
- Mode: CPU-only, `-ngl 0 --no-repack`.
- Correct trace runs use `-b 16 -ub 16`.
- Summary: `runs/qwen3_next_q3_trace_summary.md`.
- Observed routed layers: 47 (`layers 0-46`; layer 47 emitted no `ffn_moe_topk`).
- Three short coding traces project about 20-23 sync tok/s with an 8GiB LRU arena under the current 5GB/s SSD assumption.

Packed sidecar replay:

- Plan: `runs/qwen3_next_q3_sidecar_plan.json`.
- Packed file: `$MOEVM_MODELS_DIR/qwen3-coder-next/qwen3-next-q3.moevm`.
- Records: 24,576 = 48 layers x 512 experts.
- Payload: 31.35GiB.
- Pack time: 31.71s, 0.99GiB/s.
- Replay source-vs-packed at 8GiB arena:
  - `code2_b16`: source 3.788s / 18.5 replay tok/s; packed 0.353s / 198.5 replay tok/s.
  - `frontend1`: source 2.513s / 17.1 replay tok/s; packed 0.297s / 144.7 replay tok/s.
  - `infra1`: source 2.463s / 15.8 replay tok/s; packed 0.293s / 132.9 replay tok/s.
- This measures expert I/O replay only, not full inference speed.
- Packed sidecar reduces miss read operations 3.0x by reading one contiguous `(layer, expert)` record instead of separate gate/up/down slices.
- The packed GB/s figure is probably helped by OS/file-cache behavior despite `F_NOCACHE`; use the source-vs-packed elapsed improvement as the current signal.

Native C++ pager replay:

- Indexer: `phase0/sidecar_index.py`.
- Index: `$MOEVM_MODELS_DIR/qwen3-coder-next/qwen3-next-q3.moevm.idx`, 393,272 bytes.
- Source: `phase0/pager_replay.cpp`.
- Binary: `phase0/pager_replay`.
- Mode: sync `pread` from packed sidecar with the same LRU policy as Python.
- Results at 8GiB arena:
  - `code2_b16`: 0.336s, 208.1 replay tok/s, 31.70GB/s, hit 0.763.
  - `frontend1`: 0.236s, 182.4 replay tok/s, 30.80GB/s, hit 0.737.
  - `infra1`: 0.229s, 170.6 replay tok/s, 31.44GB/s, hit 0.714.
- This still measures expert I/O replay only, not full model inference.

Native async prefetch replay:

- `pager_replay.cpp` now supports `--prefetch-window`, `--prefetch-threads`, `--prefetch-max-inflight-mib`, and `--compute-us-per-access`.
- `compute-us-per-access` is synthetic overlap time. Use demand reads and wait time as the important signal; wall-time includes the artificial sleep.
- Code trace, `window=47`, `inflight=64MiB`, `compute=5us/access`:
  - 1 worker: 0.590s, hit 0.880, waits 3,953, wait 0.287s, 18.09GB/s.
  - 2 workers: 0.415s, hit 0.951, waits 1,604, wait 0.104s, 25.70GB/s.
  - 4 workers: 0.397s, hit 0.975, waits 820, wait 0.069s, 26.88GB/s.
  - 8 workers: 0.441s, hit 0.977, waits 754, wait 0.067s, 24.21GB/s.
- Selected profile for next work: `window=47`, `threads=4`, `inflight=64MiB`, `compute=5us/access`.
- Selected profile results:
  - `code2_b16`: 0.397s, 176.4 replay tok/s, hit 0.975, demand reads 1, prefetch reads 7,793, wait 0.069s.
  - `frontend1`: 0.278s, 154.6 replay tok/s, hit 0.973, demand reads 1, prefetch reads 5,306, wait 0.046s.
  - `infra1`: 0.258s, 151.0 replay tok/s, hit 0.969, demand reads 1, prefetch reads 5,250, wait 0.048s.
- Interpretation: 4 `pread` workers are useful on this Mac; 8 workers did not improve the code trace. The useful next design is a small in-flight queue, not unbounded CPU fanout.

Arena allocator model:

- `pager_replay.cpp` now models resident expert storage with a 16KiB-aligned arena allocator.
- It tracks arena offsets, evictions, free spans, high-water bytes, and resident record count.
- It does not allocate the full 8GiB buffer yet; this is the allocator model before real resident buffers.
- Selected async profile rerun with arena metrics:
  - `code2_b16`: hit 0.978, wait 0.061s, evictions 1,551, arena high-water 8.00GiB, resident records 6,247, free spans 34, largest free span 1.30MiB.
  - `frontend1`: hit 0.973, wait 0.046s, evictions 0, arena high-water 6.76GiB, resident records 5,307, free spans 1.
  - `infra1`: hit 0.969, wait 0.048s, evictions 0, arena high-water 6.69GiB, resident records 5,251, free spans 1.
- Interpretation: the 8GiB arena behaves cleanly on short traces. Code reaches the cap and evicts, but fragmentation is not pathological yet. Longer traces are still needed.

Long trace predictor gate:

- Long trace: `runs/qwen3_next_q3_trace_code_long1.jsonl`.
- Prompt tokens: 633.
- Records: 29,751.
- Expert accesses: 297,510.
- Trace mode: CPU-only, `-ngl 0 --no-repack -c 1024 -b 16 -ub 16`.
- Trace runtime: 17.97s under `safe_run.py`.

`pager_replay.cpp` realistic predictor flags:

- `--prefetch-policy oracle`: future trace upper bound, not realistic.
- `--prefetch-policy last-layer`: repeats recently seen experts per layer.
- `--prefetch-policy freq-layer`: predicts frequent experts per layer.
- `--prefetch-policy hybrid`: recent + frequent.
- `--predictor-topk N`.
- `--predictor-min-count N`.

Long trace real sidecar reads:

- 4GiB LRU: 34.108s, 18.6 replay tok/s, hit 0.517, demand reads 143,833, read 183.3GiB.
- 4GiB hybrid min-count 5: 39.053s, 16.2 replay tok/s, hit 0.585, demand reads 123,396, prefetch reads 29,431, read 194.8GiB.
- 8GiB LRU: 12.706s, 49.8 replay tok/s, hit 0.824, demand reads 52,289, read 66.7GiB.
- 8GiB oracle: 5.450s, 116.1 replay tok/s, hit 0.895, demand reads 6, prefetch reads 52,371, read 66.8GiB, wait 2.134s.

Verdict:

- LRU + 8GiB arena is the strongest realistic baseline today.
- The oracle result proves prefetch can roughly 2.3x the long-trace I/O replay if the runtime gets a good signal.
- The first realistic predictors are rejected for now. At 4GiB, hybrid improves hit-rate but increases total reads and wall time. At 8GiB, it does not beat LRU.
- Next predictor work needs route-logit/topk information from the current forward pass or a cheap learned layer-transition predictor, not just historical expert IDs.

Router candidate trace:

- Local `llama-moe-trace` now supports `--trace-candidates N`.
- It captures `ffn_moe_argsort` candidate expert IDs in addition to `ffn_moe_topk`.
- Candidate records are emitted as `{"kind":"candidates","token":...,"layer":...,"candidates":[...]}`.
- Existing topk records keep the old format: `{"token":...,"layer":...,"experts":[...]}`.
- Smoke file: `runs/qwen3_next_q3_trace_candidates_smoke.jsonl`.
- Smoke result: 13 prompt tokens, 611 topk records, 611 candidate records, 47 routed layers/token, 32 candidates/record.
- Long candidate trace: `runs/qwen3_next_q3_trace_code_long1_candidates.jsonl`.
- Long candidate result: 633 prompt tokens, 29,751 topk records, 29,751 candidate records, 16 candidates/record, 35.28s trace runtime.

Candidate trace command:

```bash
python3 phase0/safe_run.py \
  --log runs/qwen3_next_q3_trace_candidates_smoke.log \
  --summary runs/qwen3_next_q3_trace_candidates_smoke.summary \
  --max-seconds 180 \
  --preflight-min-free-pct 50 \
  --min-free-pct 20 \
  --max-rss-mib 40000 \
  --max-wired-gib 34 \
  --max-log-mb 8 \
  -- external/llama.cpp/build/bin/llama-moe-trace \
    --trace-out runs/qwen3_next_q3_trace_candidates_smoke.jsonl \
    --trace-candidates 32 \
    -m ~/models/qwen3-coder-next/Qwen3-Coder-Next-UD-Q3_K_XL.gguf \
    -ngl 0 --no-repack -c 256 -b 16 -ub 16 \
    -p "Implement a small Python task queue with retry scheduling and deterministic tests."
```

Candidate-aware replay:

- `pager_replay.cpp` supports `--prefetch-policy candidates`, `--candidate-topk N`, and `--candidate-delay-us N`.
- The candidate predictor uses current-layer router candidates before expert matmul. It is not an oracle and does not use future-token route data.
- Long trace, 8GiB arena, real sidecar reads:
  - LRU baseline: 12.703s, 49.8 replay tok/s, hit 0.824, demand reads 52,289, read 66.7GiB.
  - Candidates top10, 4 workers, delay 0: 9.141s, 69.3 replay tok/s, hit 0.852, demand reads 29, prefetch reads 52,407, read 66.9GiB.
  - Candidates top10, 4 workers, delay 10us: 9.245s, 68.5 replay tok/s, demand reads 20, read 66.9GiB.
  - Candidates top10, 4 workers, delay 50us: 9.723s, 65.1 replay tok/s, demand reads 24, read 67.0GiB.
  - Candidates top16, 4 workers, delay 0: 12.100s, 52.3 replay tok/s, demand reads 1, prefetch reads 104,708, read 133.5GiB.
- Worker sweep at top10/delay0: 1 worker 12.883s, 2 workers 9.875s, 4 workers 9.182s, 8 workers 9.377s, 12 workers 9.534s.
- Current selected candidate profile: `candidate_topk=10`, `prefetch_threads=4`, `prefetch_max_inflight_mib=64`, `candidate_delay_us=0`.
- Result: candidate prefetch is the first realistic predictor that beats LRU on real sidecar reads, about 1.39x on the long trace.
- Rejected: `candidate_topk=16`, because it doubles read traffic and causes 98,878 evictions without improving wall-time.

Oracle prefetch simulation:

- Tool: `phase0/prefetch_sim.py`.
- This is an upper bound, not a real decoder guarantee.
- At 8GiB arena, a 47-access lookahead nearly eliminates blocking demand reads on the three Qwen3Next Q3 traces:
  - `code2_b16`: demand misses 7,792 -> 2; demand MB/token 152.364 -> 0.039.
  - `frontend1`: demand misses 5,307 -> 1; demand MB/token 168.861 -> 0.032.
  - `infra1`: demand misses 5,251 -> 1; demand MB/token 184.288 -> 0.035.
- Total bytes stay roughly the same; prefetch changes when bytes are loaded, not how many useful records are needed.
- Research implication: the key optimization is a realistic bounded prefetch signal, not raw SSD bandwidth alone.

Sidecar commands:

```bash
python3 phase0/sidecar_plan.py \
  --model ~/models/qwen3-coder-next/Qwen3-Coder-Next-UD-Q3_K_XL.gguf \
  --tensors-json runs/qwen3_next_q3_tensors.json \
  --out runs/qwen3_next_q3_sidecar_plan.json

python3 phase0/sidecar_pack.py \
  --plan runs/qwen3_next_q3_sidecar_plan.json \
  --out ~/models/qwen3-coder-next/qwen3-next-q3.moevm \
  --summary runs/qwen3_next_q3_pack_full.json \
  --nocache \
  --progress-every 2048

python3 phase0/sidecar_replay.py \
  --plan runs/qwen3_next_q3_sidecar_plan.json \
  --trace runs/qwen3_next_q3_trace_code2_b16.jsonl \
  --budget-gib 8 \
  --packed ~/models/qwen3-coder-next/qwen3-next-q3.moevm \
  --nocache \
  --out runs/qwen3_next_q3_replay_code2_b16_8g_packed.json
```

Native index + replay commands:

```bash
python3 phase0/sidecar_index.py \
  --plan runs/qwen3_next_q3_sidecar_plan.json \
  --out ~/models/qwen3-coder-next/qwen3-next-q3.moevm.idx

clang++ -O3 -std=c++17 -Wall -Wextra -pedantic \
  phase0/pager_replay.cpp \
  -o phase0/pager_replay

python3 phase0/safe_run.py \
  --log runs/qwen3_next_q3_pager_cpp_code2_b16_8g.log \
  --summary runs/qwen3_next_q3_pager_cpp_code2_b16_8g.safe_summary \
  --max-seconds 300 \
  --preflight-min-free-pct 50 \
  --min-free-pct 20 \
  --max-rss-mib 2000 \
  --max-log-mb 8 \
  -- phase0/pager_replay \
    --index ~/models/qwen3-coder-next/qwen3-next-q3.moevm.idx \
    --sidecar ~/models/qwen3-coder-next/qwen3-next-q3.moevm \
    --trace runs/qwen3_next_q3_trace_code2_b16.jsonl \
    --budget-gib 8 \
    --nocache \
    --out runs/qwen3_next_q3_pager_cpp_code2_b16_8g.json
```

Async prefetch replay command:

```bash
python3 phase0/safe_run.py \
  --log runs/qwen3_next_q3_pager_cpp_code2_b16_8g_pf47_c5_t4.log \
  --summary runs/qwen3_next_q3_pager_cpp_code2_b16_8g_pf47_c5_t4.safe_summary \
  --max-seconds 300 \
  --preflight-min-free-pct 50 \
  --min-free-pct 20 \
  --max-rss-mib 2000 \
  --max-log-mb 8 \
  -- phase0/pager_replay \
    --index ~/models/qwen3-coder-next/qwen3-next-q3.moevm.idx \
    --sidecar ~/models/qwen3-coder-next/qwen3-next-q3.moevm \
    --trace runs/qwen3_next_q3_trace_code2_b16.jsonl \
    --budget-gib 8 \
    --prefetch-window 47 \
    --prefetch-threads 4 \
    --prefetch-max-inflight-mib 64 \
    --compute-us-per-access 5 \
    --nocache \
    --out runs/qwen3_next_q3_pager_cpp_code2_b16_8g_pf47_c5_t4.json
```

Native runtime API smoke:

- Source: `phase0/moevm_runtime.hpp`, `phase0/moevm_runtime.cpp`.
- Smoke source: `phase0/runtime_smoke.cpp`.
- Runtime replay source: `phase0/runtime_replay.cpp`.
- Component indexer: `phase0/sidecar_component_index.py`.
- Component index: `$MOEVM_MODELS_DIR/qwen3-coder-next/qwen3-next-q3.moevm.cidx`, 589,880 bytes.
- Contract: `prefetch_candidates(layer, candidates)` then `wait_selected(layer, expert)` returns a resident record pointer in the RAM arena; `component_view(record, Gate/Up/Down)` returns tensor-specific slices inside that resident record.
- Smoke result on the candidate trace: 10 selected experts, 10 served from prefetch, 0 demand reads, 0 evictions, 13,598,720 bytes high-water.
- Component smoke result: `component_bytes == selected_bytes == 13,598,720`, proving `gate + up + down` views cover the resident record exactly.
- `safe_run.py` was fixed to drain remaining stdout after very short processes exit, so smoke logs are now complete.
- Bug caught and fixed: the first `.cidx` header writer used a 60-byte Python struct while C++ expected 56 bytes. The script now asserts header and entry sizes before writing.

Build:

```bash
clang++ -O3 -std=c++17 -Wall -Wextra -pedantic \
  phase0/moevm_runtime.cpp \
  phase0/runtime_smoke.cpp \
  -o phase0/runtime_smoke
```

Run:

```bash
python3 phase0/sidecar_component_index.py \
  --plan runs/qwen3_next_q3_sidecar_plan.json \
  --out ~/models/qwen3-coder-next/qwen3-next-q3.moevm.cidx

python3 phase0/safe_run.py \
  --log runs/qwen3_next_q3_runtime_smoke.log \
  --summary runs/qwen3_next_q3_runtime_smoke.safe_summary \
  --max-seconds 60 \
  --preflight-min-free-pct 20 \
  --min-free-pct 15 \
  --max-rss-mib 1000 \
  --max-wired-gib 34 \
  --max-log-mb 8 \
  -- phase0/runtime_smoke \
    --index ~/models/qwen3-coder-next/qwen3-next-q3.moevm.idx \
    --component-index ~/models/qwen3-coder-next/qwen3-next-q3.moevm.cidx \
    --sidecar ~/models/qwen3-coder-next/qwen3-next-q3.moevm \
    --trace runs/qwen3_next_q3_trace_candidates_smoke.jsonl \
    --arena-mib 64 \
    --inflight-mib 16 \
    --threads 2 \
    --candidate-topk 10
```

Runtime replay build:

```bash
clang++ -O3 -std=c++17 -Wall -Wextra -pedantic \
  phase0/moevm_runtime.cpp \
  phase0/runtime_replay.cpp \
  -o phase0/runtime_replay
```

Runtime replay long trace:

```bash
python3 phase0/safe_run.py \
  --log runs/qwen3_next_q3_runtime_replay_long1_cand10_8g.log \
  --summary runs/qwen3_next_q3_runtime_replay_long1_cand10_8g.safe_summary \
  --max-seconds 240 \
  --preflight-min-free-pct 20 \
  --min-free-pct 15 \
  --max-rss-mib 14000 \
  --max-wired-gib 34 \
  --max-log-mb 8 \
  -- phase0/runtime_replay \
    --index ~/models/qwen3-coder-next/qwen3-next-q3.moevm.idx \
    --sidecar ~/models/qwen3-coder-next/qwen3-next-q3.moevm \
    --trace runs/qwen3_next_q3_trace_code_long1_candidates.jsonl \
    --budget-gib 8 \
    --prefetch-max-inflight-mib 64 \
    --prefetch-threads 4 \
    --candidate-topk 10 \
    --out runs/qwen3_next_q3_runtime_replay_long1_cand10_8g.json
```

Runtime replay long result:

- Internal replay elapsed: 9.386s.
- Safe-run wall elapsed including setup: 10.11s.
- Replay speed: 67.4 tok/s.
- Read: 66.56GiB, 52,207 ops.
- Demand reads: 23 ops / 33.65MiB.
- Prefetch reads: 52,184 ops / 71.43GB.
- Prefetch waits: 40,632, 9.208s total.
- Demand miss rate: 0.0077%.
- Prefetch wait rate: 13.66%.
- Evictions: 46,207.
- Arena high-water: 8.00GiB.
- Memory stayed stable under `safe_run.py`.

Comparison against `pager_replay.cpp` candidate top10:

- `pager_replay`: 9.141s, 69.3 replay tok/s, 66.9GiB read.
- `runtime_replay`: 9.386s, 67.4 replay tok/s, 66.56GiB read.
- Runtime API overhead is about 2-3% on internal replay time, acceptable because this path uses a real RAM arena and returns resident pointers.
- Metric caveat: `runtime_hit_rate` counts records after waiting for in-flight prefetch, so compare `demand_read_ops`, `demand_miss_rate`, `prefetch_wait_rate`, and wall-time against `pager_replay`, not raw hit-rate.

Oracle prefetch command:

```bash
python3 phase0/prefetch_sim.py \
  --plan runs/qwen3_next_q3_sidecar_plan.json \
  --trace runs/qwen3_next_q3_trace_code2_b16.jsonl \
  --budget-gib 8 \
  --windows 0,47,470,1880 \
  --out runs/qwen3_next_q3_prefetch_code2_b16_8g.json
```

## Cache Simulator

Expected trace JSONL format:

```json
{"token":0,"layer":0,"experts":[12,44,91,103,120,201,260,311,400,511]}
```

Run:

```bash
python3 phase0/cache_sim.py runs/router_trace.jsonl \
  --slice-mb 1.90 --budgets 20,24,28 --bssd-gbps 5 --compute-ms 14 \
  --json runs/cache_sim.json
```

## Kill Switches

- Qwen3-Coder-Next quality does not beat or materially complement `qwen3-coder:30b`.
- Q3 resident/near-resident is good enough, making Q4 paging unnecessary for this model.
- SSD random 2MiB reads are below 3GB/s, or below 1.5GB/s as a hard stop.
- Simulated hit-rate at 28GiB is too low for >=10 tok/s.
- Q3 smoke test triggers memory pressure or Jetsam with Docker stopped.
- Prefill is too slow for agentic repo tasks even if decode is acceptable.

## Next Engineering Step

Move from runtime replay to llama.cpp integration boundary:

1. Use `ggml-backend.cpp`'s existing “copy only used experts” branch as the first integration point, because it already has selected expert IDs immediately before `MUL_MAT_ID` weight transfer.
2. Replace source tensor slice reads for routed expert tensors with `wait_selected` plus `component_view(Gate/Up/Down)` when MoEVM is enabled.
3. Keep LRU + 8GiB arena as the fallback baseline and candidate top10 as the current winning prefetch policy.
4. Extend candidate records with scores/weights if IDs alone stall below the oracle gap.
5. Integrate behind a local llama.cpp flag only after the boundary is mapped and protected by `safe_run.py`.

## llama.cpp MoEVM Hook Status

Implemented local hook:

- Header: `external/llama.cpp/ggml/include/ggml-backend.h`
- Backend: `external/llama.cpp/ggml/src/ggml-backend.cpp`
- API: `ggml_backend_set_moe_weight_copy_callback(callback, user_data)`.
- Behavior: if the callback handles selected expert copies, llama.cpp skips the normal host tensor slice copy. Without a callback, behavior is unchanged.

MoEVM-enabled local example:

- `external/llama.cpp/examples/moe-trace/moe-trace.cpp`
- `external/llama.cpp/examples/moe-trace/CMakeLists.txt`
- Build target: `llama-moe-trace`.

Added local options:

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

CPU-only smoke:

- Log: `runs/qwen3_next_q3_llama_moevm_cpu_smoke.log`.
- Trace: `runs/qwen3_next_q3_llama_moevm_cpu_smoke.jsonl`.
- Mode: `-ngl 0 --no-repack`.
- Prompt tokens: 10.
- Trace records: 940.
- MoEVM prefetch records: 942.
- Demand reads: 0.
- Prefetch reads: 2,116.
- Waits: 0.
- Arena high-water: 1.00GiB.
- Copy hook calls: 0, expected for CPU-only mode.

Limited offload probe:

- Log: `runs/qwen3_next_q3_llama_moevm_ngl4_smoke.log`.
- Mode: `-ngl 4`.
- Result: stopped by `safe_run.py` before decode.
- Stop reason: wired memory exceeded 34GiB.
- No trace records were produced.

Working offload path:

- Key env: `GGML_OP_OFFLOAD_MIN_BATCH=1`. The Metal default is 32, which keeps short `MUL_MAT_ID` decode steps on CPU.
- Key llama.cpp args: `-ngl 99 -ot '.*ffn_.*_exps.weight=CPU' --op-offload --no-repack`.
- Keep mmap enabled for Qwen3Next Q3; `--no-mmap` would load the full GGUF expert tensors into RAM in addition to the MoEVM sidecar.
- This mode keeps routed experts as CPU/mmap tensors for scheduler placement, runs selected expert matmuls on Metal, and replaces the normal selected-expert host copy with MoEVM sidecar component views.

Lab MoE decode smoke:

- Log: `runs/mistral_2x_moe_llama_moevm_decode3_hook.log`.
- Mode: `GGML_OP_OFFLOAD_MIN_BATCH=1`, `-ngl 99`, expert tensors overridden to CPU, `--no-mmap`.
- Generated 3 tokens at 1.37 tok/s.
- Copy hook calls: 384.
- Copy hook handled: 384.
- Callback fallback: 0.

Qwen3Next Q3 prompt + real decode:

- Best stable command family:

```bash
python3 phase0/safe_run.py \
  --log runs/qwen3_next_q3_llama_moevm_ngl99_hookprefetch_decode20_arena1_t8.log \
  --summary runs/qwen3_next_q3_llama_moevm_ngl99_hookprefetch_decode20_arena1_t8.summary \
  --max-rss-mib 44032 \
  --max-wired-gib 40.5 \
  --min-free-pct 5 \
  --max-seconds 240 \
  -- env GGML_OP_OFFLOAD_MIN_BATCH=1 \
    external/llama.cpp/build/bin/llama-moe-trace \
      -m ~/models/qwen3-coder-next/Qwen3-Coder-Next-UD-Q3_K_XL.gguf \
      --trace-out runs/qwen3_next_q3_llama_moevm_ngl99_hookprefetch_decode20_arena1_t8.jsonl \
      --trace-candidates 10 \
      --moevm-index ~/models/qwen3-coder-next/qwen3-next-q3.moevm.idx \
      --moevm-component-index ~/models/qwen3-coder-next/qwen3-next-q3.moevm.cidx \
      --moevm-sidecar ~/models/qwen3-coder-next/qwen3-next-q3.moevm \
      --moevm-arena-gib 1 \
      --moevm-inflight-mib 128 \
      --moevm-threads 8 \
      --moevm-prefetch-topk 10 \
      --moevm-copy-debug 0 \
      -ngl 99 \
      -ot '.*ffn_.*_exps.weight=CPU' \
      --op-offload \
      --no-repack \
      -c 128 -b 16 -ub 16 -n 20 \
      -p 'def add(a, b): return a + b'
```

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
- Evictions: 7,893.
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

Current verdict:

- MoEVM has a working llama.cpp path for Qwen3Next Q3 on this Mac.
- The runtime can serve selected routed expert components from the sidecar into Metal `MUL_MAT_ID` copies with zero callback fallback.
- `-ngl 99` plus CPU expert tensor override is materially better than partial `-ngl 4`, because non-expert weights stay on Metal and normal scheduler copies nearly disappear.
- `--moevm-copy-mode memcpy` is now the preferred copy path on this Apple Silicon setup. It writes directly into the host-visible Metal shared buffer and avoids the `ggml_backend_tensor_set_async` overhead.
- `--moevm-cache` is now preferred for decode tests. macOS page cache is faster than `F_NOCACHE` for this access pattern, while explicit direct mmap remains slower.
- Best measured 100-token setting: `--moevm-arena-gib 1 --moevm-inflight-mib 256 --moevm-threads 12 --moevm-prefetch-topk 10 --moevm-copy-mode memcpy --moevm-cache`.
- Balanced setting: `--moevm-arena-gib 1 --moevm-inflight-mib 128 --moevm-threads 8 --moevm-copy-mode memcpy --moevm-cache`; it is almost as fast and uses fewer prefetch workers.
- The next bottleneck is reducing copy volume and scheduler overhead, not proving feasibility.

Parity check:

- Tool: `phase0/compare_decode_logs.py`.
- Baseline log: `runs/qwen3_next_q3_baseline_ot_cpu_decode20_parity.log`.
- MoEVM log: `runs/qwen3_next_q3_moevm_decode20_parity.log`.
- Compare JSON: `runs/qwen3_next_q3_decode20_parity_compare.json`.
- Baseline mode: same `GGML_OP_OFFLOAD_MIN_BATCH=1`, `-ngl 99`, expert CPU override, no MoEVM sidecar.
- Baseline generated 20 tokens at 6.45 tok/s.
- MoEVM generated 20 tokens at 5.30 tok/s in the parity run.
- `same_tokens=true`.
- Generated IDs:

```text
[198,750,1186,2877,11,293,1648,470,264,481,293,198,750,15602,2877,11,293,1648,470,264]
```

Optimization probes after parity:

| Probe | Result | Verdict |
|---|---:|---|
| Copy-prefetch dedup by `(decode_base, layer, ids)` | `copy_prefetches` 3,024 -> 1,008, `copy_prefetch_skips` 2,016, 5.32 tok/s | Correct, but not a speed win; keep if code simplicity is acceptable. |
| `--moevm-direct-mmap` sidecar | 3.08 tok/s, `same_tokens=true`, no arena reads | Slower; OS page faults are worse than explicit arena prefetch. Useful only as a negative baseline. |
| Hook-only prefetch, `--moevm-prefetch-topk 0` | 5.03 tok/s, 5 demand reads, more waits | Slower; eval-callback candidate/topk prefetch still helps. |
| `--moevm-copy-mode memcpy` with `F_NOCACHE` | 20 tokens at 8.29 tok/s, `same_tokens=true`, `memcpy_ops=35,613` | Big win; async tensor-set overhead was dominant. |
| `--moevm-copy-mode memcpy --moevm-cache` | 20 tokens at 9.99 tok/s, `same_tokens=true` | Best short decode result; page cache beats `F_NOCACHE`. |
| 100-token baseline vs best MoEVM | Baseline 8.10 tok/s; MoEVM 11.06 tok/s; `same_tokens=true` | Stable +36.5% decode gain over same placement without sidecar hook. |
| 2GiB arena with memcpy/cache | 9.13 tok/s; fewer reads but higher wait | Rejected; larger arena is not automatically better here. |
| 16 workers / 512MiB in-flight | 10.83 tok/s | Rejected; too much prefetch parallelism starts to cost. |

Current best 100-token artifacts:

- Baseline log: `runs/qwen3_next_q3_baseline_ot_cpu_decode100_parity.log`.
- Best MoEVM log: `runs/qwen3_next_q3_moevm_decode100_memcpy_cache_arena1_t12_i256.log`.
- Compare JSON: `runs/qwen3_next_q3_decode100_memcpy_cache_arena1_t12_i256_compare.json`.
- Result: `same_tokens=true`.
- Baseline: 100 tokens in 12.345s, 8.10 tok/s.
- MoEVM: 100 tokens in 9.041s, 11.06 tok/s.
- MoEVM copied expert components: 150,813.
- MoEVM copied bytes: 68.86GB.
- Runtime demand reads: 0.
- Runtime prefetch reads: 29,612.
- Runtime waits: 6,725, 1.467s total.
- Arena high-water: 1.00GiB.

## Shadow IDs and Compact Tensor Milestones

Fable review identified a graph-reuse hazard: if `node->src[0]` or `node->src[2]` is patched and not restored after a split compute, graph reuse can silently run the next token with stale expert tensors/IDs. The local hook now records original pointers and restores them after each split compute.

Implemented flags:

- `--moevm-shadow-ids`: allocate a backend shadow IDs tensor, copy original IDs into it, temporarily swap `node->src[2]`, then restore after compute.
- `--moevm-compact-layer 0`: compact/remap only one layer, with all other layers using the validated sparse copy path.
- `--moevm-compact-layer all`: compact/remap all decode layers; prompt/prefill currently falls back when IDs are not compatible with the compact path.
- `--moevm-compact-k-max 64`: maximum compact expert slots.

Validation results:

| Mode | Log | Result | Notes |
|---|---|---:|---|
| Shadow IDs, graph reuse enabled | `runs/qwen3_next_q3_moevm_decode100_memcpy_cache_shadow_ids.log` | `same_tokens=true`, 8.86 tok/s | `shadow_ids_applied=14403`, `fallback=141`, `graph_patches=graph_patch_restores=14403`. |
| Shadow IDs, `LLAMA_GRAPH_REUSE_DISABLE=1` | `runs/qwen3_next_q3_moevm_decode100_memcpy_cache_shadow_ids_no_reuse.log` | `same_tokens=true`, 8.59 tok/s | Confirms restore is not hiding a graph-reuse-only failure. |
| Compact layer 0 | `runs/qwen3_next_q3_moevm_decode100_memcpy_cache_compact_l0.log` | `same_tokens=true`, 8.91 tok/s | `compact_applied=300`, `fallback=3`; only layer 0 decode was compacted. |
| Compact all decode layers | `runs/qwen3_next_q3_moevm_decode100_memcpy_cache_compact_all.log` | `same_tokens=true`, 8.66 tok/s | `compact_applied=14403`, `fallback=141`; all decode MoE matmuls used compact weights/remapped IDs. |

Interpretation:

- Compact/shadow graph surgery is now correct at token-parity level across 100 generated tokens.
- The compact path is intentionally slower today because it still recopies every selected expert each token and adds ID remap/tensor patch overhead.
- The value of compact is architectural: it creates the exact mechanism needed for a persistent compact expert pool where only cache misses are copied.
- Logits-delta checker is implemented in `phase0/compare_logits.py`.
- Compact all decode with logits: `runs/qwen3_next_q3_decode20_compact_all_logits_compare.json`, `max_abs=0.0`, `argmax_match_count=20`.

## Persistent Expert Pool Milestone

`llama-moe-trace` now supports:

- `--moevm-pool-layer <layer|all|-1>`
- `--moevm-pool-capacity <slots>`

The persistent pool keeps a small backend tensor per layer/component and remaps selected expert IDs into pool slots. Only slot misses are copied from the MoEVM sidecar. Patched `node->src[0]` and `node->src[2]` are restored after each split compute, same as compact.

Validation results:

| Mode | Log | Result | Notes |
|---|---|---:|---|
| Pool layer 0, cap64, logits | `runs/qwen3_next_q3_moevm_decode20_pool_l0_logits.log` | `same_tokens=true`, logits `max_abs=0.0`, 10.06 tok/s | First pool correctness checkpoint. |
| Pool all, cap64, logits | `runs/qwen3_next_q3_moevm_decode20_pool_all_logits.log` | `same_tokens=true`, logits `max_abs=0.0`, 0.79 tok/s | Correct but too slow; Metal appears to pay for the larger `ne[2]=64` pool tensor. |
| Pool all, cap10 | `runs/qwen3_next_q3_moevm_decode20_pool_all_cap10.log` | `same_tokens=true`, 11.65 tok/s | Best short sweep result. |
| Pool all, cap11 | `runs/qwen3_next_q3_moevm_decode20_pool_all_cap11.log` | `same_tokens=true`, 11.17 tok/s | Fewer copies than cap10, but lower speed. |
| Pool all, cap16 | `runs/qwen3_next_q3_moevm_decode20_pool_all_cap16.log` | `same_tokens=true`, 10.67 tok/s | Fewer copies but lower speed than cap10. |
| Pool all, cap12 | `runs/qwen3_next_q3_moevm_decode20_pool_all_cap12.log` | `same_tokens=true`, 10.49 tok/s | Still slower than cap10. |
| Pool all, cap10, logits | `runs/qwen3_next_q3_moevm_decode20_pool_all_cap10_logits.log` | `same_tokens=true`, logits `max_abs=0.0`, 11.01 tok/s | Current bit-exact pool proof. |
| Pool all, cap10, 100 tokens | `runs/qwen3_next_q3_moevm_decode100_pool_all_cap10.log` | `same_tokens=true`, 11.95 tok/s | New best result. |
| Pool all, cap10, 100 tokens repeat | `runs/qwen3_next_q3_moevm_decode100_pool_all_cap10_r2.log` | `same_tokens=true`, 11.76 tok/s | Confirms the gain is repeatable. |
| Pool all, cap11, 100 tokens | `runs/qwen3_next_q3_moevm_decode100_pool_all_cap11.log` | `same_tokens=true`, 11.81 tok/s | More hits than cap10, but slower overall. |
| Pool all, cap10, 256 tokens | `runs/qwen3_next_q3_moevm_decode256_pool_all_cap10.log` | 11.74 tok/s | Stability run, `callback_fallback=0`, copied 107.05GB. |

Current best 100-token comparison:

- Baseline same placement without MoEVM: 8.10 tok/s, 12.345s.
- Previous best MoEVM sparse copy: 11.06 tok/s, 9.041s, copied 68.86GB.
- Persistent pool all cap10: 11.76-11.95 tok/s over two 100-token runs, best 8.371s, copied 42.66GB.
- Gain vs baseline: about +47.5%.
- Gain vs previous best MoEVM: about +8.0%.
- Pool stats on the 100-token run: `pool_applied=14403`, `pool_fallback=141`, `pool_component_hits=57384`, `pool_component_misses=86646`, `pool_evictions=28402`.
- Longer 256-token stability run stayed at 11.74 tok/s with `pool_applied=36867`, `callback_fallback=0`, and `high_water=1.00GiB`.

Interpretation:

- Pooling works and is bit-exact on the 20-token logits check.
- Pool capacity must stay close to active experts per token on this Metal path. Larger pools reduce copy volume but can lose much more in `MUL_MAT_ID` compute overhead.

## Qwen3-Coder-Next Q4 MoEVM Checkpoint

Files:

- GGUF: `$MOEVM_MODELS_DIR/qwen3-coder-next-q4/Qwen3-Coder-Next-Q4_K_M.gguf`, about 45GiB.
- Sidecar plan: `runs/qwen3_next_q4_sidecar_plan.json`, 32MiB.
- Packed sidecar: `$MOEVM_MODELS_DIR/qwen3-coder-next-q4/qwen3-next-q4.moevm`, 43.69GiB.
- Index: `$MOEVM_MODELS_DIR/qwen3-coder-next-q4/qwen3-next-q4.moevm.idx`, 384KiB.
- Component index: `$MOEVM_MODELS_DIR/qwen3-coder-next-q4/qwen3-next-q4.moevm.cidx`, 576KiB.

Safety fixes made before Q4 runtime:

- `gguf_inspect.py` now streams GGUF headers instead of reading full GGUF files into RAM.
- `safe_run.py` now blocks concurrent model I/O jobs by default.
- `safe_run.py` now handles children that remain alive after `SIGKILL` without hanging or crashing.
- `llama-model.cpp` / `llama-model-loader.cpp` now support sparse mmap Metal buffers. Tensors assigned to a default device buffer are grouped into offset ranges instead of mapping one giant `first..last` file span, so CPU-overridden routed experts are no longer included in the Metal no-copy buffer.

Q4 inspection:

- Architecture: `qwen3next`.
- Layers: 48.
- Experts: 512.
- Active experts per token: 10.
- Routed expert pool: 43.59GiB.
- Resident non-routed estimate: 1.60GiB.
- Average bytes per `(layer, expert)`: 1.82MiB.
- Active expert bytes per token if all miss: 871.88MiB.

Packing:

- Command used `sidecar_pack.py --nocache` under `safe_run.py`.
- Packed 24,576 records.
- Payload: 43.69GiB.
- Pack time: 42.07s.
- Write throughput: 1.04GiB/s.

Runtime findings:

- Before the sparse mmap loader patch, `-ngl 99` with expert CPU override triggered Metal `kIOGPUCommandBufferCallbackErrorOutOfMemory` before a successful decode.
- `-ngl 4` also triggers Metal OOM, even with `-fa off -nkvo` and `-b 1 -ub 1`.
- `-ngl 0 --no-op-offload` runs but does not exercise the MoEVM copy hook.
- Old working Q4 path: `-ngl 0 --op-offload -fa off -nkvo -b 1 -ub 1`, with expert tensors overridden to CPU and MoEVM pool enabled.
- Current working Q4 path: `-ngl 99 --op-offload --no-repack -b 1 -ub 1`, default KV/Flash Attention behavior, expert tensors overridden to CPU and MoEVM pool enabled.

Best current Q4 comparison:

| Mode | Log | Decoded | tok/s | Notes |
|---|---|---:|---:|---|
| Baseline, same placement, no sidecar | `runs/qwen3_next_q4_baseline_decode100_ngl0_opoffload_b1.log` | 64 | 3.03 | EOG at 64/100. |
| MoEVM pool cap10 | `runs/qwen3_next_q4_moevm_decode100_ngl0_opoffload_b1.log` | 64 | 4.96 | `same_tokens=true`, `pool_fallback=0`. |
| MoEVM pool cap64 | `runs/qwen3_next_q4_moevm_decode100_ngl0_opoffload_b1_cap64.log` | 64 | 5.10 | Lower copy volume, stable. |
| MoEVM pool cap128 | `runs/qwen3_next_q4_moevm_decode100_ngl0_opoffload_b1_cap128.log` | 64 | 5.10 | Higher wired memory, no speed gain over cap64. |
| MoEVM pool cap256 | `runs/qwen3_next_q4_moevm_decode100_ngl0_opoffload_b1_cap256.log` | 64 | 5.12 | Best measured; zero pool evictions, copied 12.66GB. |
| Baseline `-ngl99 --op-offload`, sparse mmap | `runs/qwen3_next_q4_baseline_decode100_ngl99_sparse_mmap_opoffload_kvfaauto.log` | 92 | 4.24 | Same placement baseline for current MoEVM. |
| MoEVM `-ngl99`, pool cap32, sparse mmap | `runs/qwen3_next_q4_moevm_decode100_ngl99_sparse_mmap_cap32_kvfaauto.log` | 92 | 12.39 | Correct but below cap64. |
| MoEVM `-ngl99`, pool cap64, sparse mmap | `runs/qwen3_next_q4_moevm_decode100_ngl99_sparse_mmap_cap64_kvfaauto.log` | 92 | 13.08 | Pre-P0.b best. |
| MoEVM `-ngl99`, pool cap128, sparse mmap | `runs/qwen3_next_q4_moevm_decode100_ngl99_sparse_mmap_cap128_kvfaauto.log` | 92 | 12.96 | Fewer copies, slightly slower and higher wired. |
| MoEVM `-ngl99`, pool cap64, P0.b defer input wait prototype | `runs/qwen3_next_q4_p0b_notrace_cap64_n200.log` | 200 | 20.13 | Unguarded prototype; `same_tokens=true` vs pre-P0.b no-trace. |
| MoEVM `-ngl99`, pool cap64, P0.b hardened defer input wait | `runs/qwen3_next_q4_hardened_defer_on_cap64_n200.log` | 200 | 18.64 | Current mergeable best; epoch/pinning guard, `same_tokens=true`. |

Cap256 comparison JSON:

- `runs/qwen3_next_q4_decode100_ngl0_opoffload_b1_cap256_compare.json`
- `same_tokens=true`.
- Baseline: 64 tokens in 21.094s, 3.03 tok/s.
- MoEVM: 64 tokens in 12.511s, 5.12 tok/s.
- Speedup vs same placement baseline: about 1.69x.
- `callback_fallback=0`.
- `pool_applied=10629`, `pool_fallback=0`.
- `pool_component_hits=86427`, `pool_component_misses=19863`, `pool_evictions=0`.
- `copied_bytes=12659417088`.

Current best sparse mmap comparison JSON:

- `runs/qwen3_next_q4_decode100_ngl99_sparse_mmap_cap64_kvfaauto_compare.json`
- `same_tokens=true`.
- Baseline same placement: 92 tokens in 21.712s, 4.24 tok/s.
- MoEVM: 92 tokens in 7.033s, 13.08 tok/s.
- Speedup vs same placement baseline: about 3.08x.
- `callback_fallback=0`.
- `pool_applied=14661`, `pool_fallback=0`.
- `pool_component_hits=112587`, `pool_component_misses=34023`, `pool_evictions=8269`.
- `copied_bytes=21796110336`.
- `normal_fallback=333`.

P0.b deferred input wait:

- Pre-P0.b site timing: `runs/qwen3_next_q4_site_timing_notrace_cap64_n200.log`, 200 tokens in 15.465s, 12.93 tok/s.
- P0.b prototype run: `runs/qwen3_next_q4_p0b_notrace_cap64_n200.log`, 200 tokens in 9.934s, 20.13 tok/s.
- P0.b hardened run: `runs/qwen3_next_q4_hardened_defer_on_cap64_n200.log`, 200 tokens in 10.729s, 18.64 tok/s.
- Token parity: `same_tokens=true` over all 200 generated IDs.
- Hardened token parity: `same_tokens=true` vs both the prototype and fallback 200-token runs.
- `split_input_wait_us`: 9,211,215 -> 698 prototype, 623 hardened.
- `backend_sync_us`: 11,110,584 -> 4,769,352 prototype, 5,373,235 hardened.
- Hardened counters: `defer_input_wait_skips=33456`, `defer_input_wait_disabled_events=0`, `pool_submit_epoch=33456`, `pool_completed_epoch=33453`, `pool_epoch_drains=11152`, `pool_forced_fences=0`.
- Mechanism: MoEVM sets `defer_input_wait=1` for pool/compact mode. The backend delays the split input wait until a real fallback write to `input_cpy` is required; pool-patched `MUL_MAT_ID` inputs skip that drain entirely.
- Guardrail: in deferred mode, the MoEVM callback returns `false` instead of doing a classic copy if pool/compact cannot patch the graph, so the backend falls back after performing the normal wait.
- Hardening: defer is refused when scheduler events are active, pool slot eviction is guarded by explicit submit/completed epochs, and pool/compact shadow IDs are copied into graph-visible tensors only after all required fills succeed.
- Stress probe: `runs/qwen3_next_q4_hardened_defer_on_cap16_n100.log`, pool cap16, 100 tokens in 6.187s, 16.16 tok/s, same first 100 IDs as cap64 reference, `pool_evictions=32637`, `pool_forced_fences=0`.
- Deterministic pinning self-test: `external/llama.cpp/build/bin/llama-moe-trace --moevm-self-test-pool-pinning`, passed. It verifies that pinned slots are skipped during eviction and that advancing `completed_epoch` makes the retry plan succeed.
- Delay/fence stress: `runs/qwen3_next_q4_delay_pinning_cap16_n50.log`, cap16, `--moevm-inject-delay-us 500`, one-shot `--moevm-test-force-pinned-fence`, 50 tokens in 21.059s, same first 50 IDs as hardened cap64 reference, `pool_forced_fences=1`, `pool_pinned_eviction_skips=16`, callback `delay_us=14818365`, runtime `delay_us=7421004`. This is a correctness stress, not a throughput number.

P0.c dynamic-K pool:

- Implementation: pool/compact now support logical K = `ids.ne[0] * ids.ne[1]` for `ub > 1`, while the capacity guard is applied to the deduplicated expert union (`expert_id_count`). If the unique union exceeds capacity, overflow falls back and increments `fallback_k_overflow`.
- The IDs readback path now reads the full strided tensor span, and pool/compact remap walks both dimensions of the original `[k, n_tokens]` IDs tensor with source/destination strides.
- Added pool fallback reason counters: shape, IDs read, allocation, shadow allocation/bytes, plan, remap, and fill.
- Added deterministic stride-aware remap coverage to `--moevm-self-test-pool-pinning`.
- `ub=1` gate: `runs/qwen3_next_q4_dynamick_ub1_cap64_n100.log`, 100 tokens in 5.178s, 19.31 tok/s, same first 100 IDs as the hardened reference, `fallback_k_overflow=0`, `pool_fallback=0`.
- `ub=8` prefill gate: `runs/qwen3_next_q4_dynamick_long_ub8_cap128_n4_shadowfix.log`, 960-token prompt, 4 decode tokens in 0.206s, 19.38 tok/s, generated IDs `[31075,264,5670,11972]`, `fallback_k_overflow=0`, `pool_fallback=0`.
- Bit-exact replacement check: `qwen3_next_q4_dynamick_long_ub8_cap128_n4.logits` vs `qwen3_next_q4_dynamick_long_ub8_cap128_n4_shadowfix.logits` has `max_abs=0.0`, `argmax_mismatch_count=0`.
- Note: `ub=1` vs `ub=8` logits differ numerically (`max_abs=1.711`) while tokens/argmax match; the same delta exists on the old `ub=8` fallback, so it is not caused by dynamic-K remap.
- Delay/fence `ub=8` stress: `runs/qwen3_next_q4_dynamick_long_ub8_delay_cap128_n2_rss65g.log`, 960-token prompt, delay max 500us, one-shot forced fence, same first two generated IDs `[31075,264]`, `pool_forced_fences=1`, `pool_pinned_eviction_skips=89`, `fallback_k_overflow=0`, `pool_fallback=0`. Throughput is not meaningful because 65.48s of artificial delay was injected.

P0.d Fable audit hardening:

- Code branch: `codex/moevm-audit-fixes` in `external/llama.cpp`; P0.d code commit `db94a94`, tag `moevm-p0d-audit-hardened`.
- Build gate: `cmake --build external/llama.cpp/build --target llama-moe-trace -j 8`, passed.
- Runtime pinning: arena-backed `RecordView` now increments a per-record pin count and must be released by the caller; eviction skips pinned resident records. Direct mmap records remain no-op releases.
- `moe-trace.cpp` uses RAII record guards around every `wait_selected()` result.
- Shadow IDs are keyed by patch kind (`pool`, `compact`, `verbatim`), so a fallback path cannot reuse and overwrite a graph-visible shadow tensor from another mode.
- Pool/compact weight tensors allocate one extra expert slot as over-read padding while graph-visible `ne[2]` remains the logical active capacity.
- The backend callback contract now documents process-global scope, unregister-before-destroy, `false` with no mutation, and no `dst_tensor` writes while deferred input wait is active.
- Deferred input wait is gated by `sched->n_copies == 1` and no split event; otherwise it falls back to normal waits and increments `defer_input_wait_disabled_events`.
- Pool state is keyed by backend and buffer type, not just layer, so future multi-backend placement cannot share one LRU/epoch state accidentally.
- Backend MoE copy stats now use internal relaxed atomic counters and the low-level backend timing hooks are gated, so non-MoEVM paths no longer pay unconditional `ggml_time_us()` calls.
- `debug_remaining` is now an atomic claim counter and `MOEVM_FORCE_STATS=1` can force backend timing stats without registering a MoE callback. The public header documents that stats snapshots/resets are relaxed per-field diagnostics, not coherent transaction snapshots.
- Direct-mmap runtime hits now update metrics and record generations under the runtime mutex.
- MoEVM callback runtime failures now increment `runtime_errors`, log the first error immediately as `LOG_ERR`, and retain first/last messages under a mutex for final diagnostics.
- Pool fill failures now unwind through the existing touched-slot rollback instead of returning from the middle of the fill loop.
- `llama-moe-trace` now permits no-trace stock runs and respects `--ignore-eos` in normal and speculative loops, so long logits parity tests do not require trace JSON and can force fixed token counts.
- `safe_run.py` now reads child output via nonblocking `os.read()` and fails closed after repeated `memory_pressure` failures.
- Sidecar pack summaries include SHA-256. `sidecar_index.py` and `sidecar_component_index.py` can bind indexes to a `.moevm` payload by size + SHA-256 prefix; the runtime verifies these fields when present.
- Q4 `.idx` and `.cidx` were regenerated with integrity fields:
  - `runs/qwen3_next_q4_sidecar_index_integrity.log`, sidecar size `46909087744`, sha256 prefix24 `afbf1ec420fa72a8a5f4852d49fa4233c0d76875213ee123`.
  - `runs/qwen3_next_q4_sidecar_component_index_integrity.log`, sha256 prefix16 `afbf1ec420fa72a8a5f4852d49fa4233`.
- Smoke with integrity verification: `runs/qwen3_next_q4_audit_integrity_smoke_n1.log`, 1 token at 17.08 tok/s, `pool_fallback=0`, `fallback_k_overflow=0`, `defer_input_wait_disabled_events=0`.
- Second-pass smoke after atomic stats/runtime error cleanup: `runs/qwen3_next_q4_auditfix2_smoke_n1.log`, 1 token at 17.28 tok/s, `pool_fallback=0`, `fallback_k_overflow=0`, `runtime_errors=0`.
- Third-pass smoke after Fable nits: `runs/qwen3_next_q4_auditfix3_smoke_n1.log`, 1 token at 15.63 tok/s, `pool_fallback=0`, `fallback_k_overflow=0`, `runtime_errors=0`.
- Real-context run: `runs/qwen3_next_q4_p0d_c4096_ub8_cap128_n50.log`, `-c 4096`, 960-token prompt, 50 decode tokens in 2.285s, 21.88 tok/s, `pool_fallback=0`, `fallback_k_overflow=0`, `runtime_errors=0`.
- Long no-trace logits parity: stock `qwen3_next_q4_p0d_parity_stock_ignoreeos_n1024_v2.log` vs MoEVM `qwen3_next_q4_p0d_parity_moevm_cap128_ignoreeos_n1024.log`, 1024 rows, `max_abs=0.0`, `argmax_mismatch_count=0`, `token_mismatch_count=0`.
- Jitter/fence logits stress: `qwen3_next_q4_p0d_jitter_delay_cap16_n50.log`, cap16, `--moevm-inject-delay-us 500`, `--moevm-test-force-pinned-fence`, 50 rows, `max_abs=0.0` vs no-delay cap16 reference, `pool_forced_fences=1`, `pool_pinned_eviction_skips=16`, `runtime_errors=0`.
- `safe_run` nonblocking smoke: `runs/safe_run_audit_echo.summary`, process exited cleanly after emitting partial stdout without a newline.

Fast hybrid non-MoEVM baseline:

- `runs/qwen3_next_q4_baseline_decode100_ngl99_sparse_mmap_noopoffload_kvfaauto.log`
- Mode: `-ngl 99 -ot '.*ffn_.*_exps.weight=CPU' --no-op-offload`.
- Result: 100 tokens in 9.174s, 10.90 tok/s.
- This does not exercise MoEVM, but it is an important baseline unlocked by sparse mmap.

Interpretation:

- This is the first Q4 checkpoint where `-ngl 99` runs without Metal OOM while preserving token parity against the same placement baseline.
- The current bottleneck is no longer the giant `first..last` mmap span, the pre-copy split wait, or `ub > 1` pool fallback. The bottlenecks are readback/verification sync, callback/pool copy cost, graph patch split count, and speculative rollback overhead on Qwen3Next recurrent state.
- The next R&D targets are graph-level pool integration to reduce split count, then recurrent-state rollback/partial sequence removal so speculative decode can avoid checkpoint save/restore costs.
- The current Qwen3Next Q3 default pool capacity in `llama-moe-trace` is 10.

DeepSpec / DFlash R&D note:

- DeepSeek DeepSpec is a relevant speculative-decoding framework containing DSpark, DFlash, and Eagle3.
- z-lab publishes `Qwen3-Coder-Next-DFlash`, a 0.5B DFlash drafter paired with `Qwen/Qwen3-Coder-Next`.
- This is relevant because speculative verification can amortize MoEVM's fixed scheduler/readback costs across several accepted tokens.
- The compatible GGUF drafter tested locally is `transmutator/Qwen3-Coder-Next-DFlash-GGUF:Q8_0`, a GGUF quant of `z-lab/Qwen3-Coder-Next-DFlash`.
- Dynamic-K is now implemented and verified. The remaining local prerequisite for a real speedup is avoiding checkpoint-heavy rollback or using a drafter with much longer accepted blocks.

Speculative rollback probe:

- Log: `runs/qwen3_next_q4_spec_probe_qwen06b_q8_n32.log`.
- Target: local `Qwen3-Coder-Next-Q4_K_M.gguf`, `-ngl 99`, routed experts on CPU, `--no-op-offload`.
- Draft: `Qwen/Qwen3-0.6B-GGUF:Q8_0`, downloaded through llama.cpp HF cache.
- Result: process exited successfully.
- llama.cpp reported: `speculative decoding will use checkpoints (context does not support partial sequence removal)`.
- Generation stats: encoded 11 prompt tokens, decoded 33 tokens, `n_draft=4`, `n_drafted=44`, `n_accept=22`, acceptance 50%.
- Throughput was only 6.16 tok/s in this non-MoEVM probe, so this is not a performance result. It is a gating result: Qwen3Next speculative decode does not immediately fail; rollback is handled by checkpointing rather than partial sequence removal.

MoEVM speculative probe:

- Integration: `llama-moe-trace` now supports `--moevm-speculative` and the normal llama.cpp `--spec-*` flags. The target model runs through the MoEVM sidecar callback; the draft model is loaded separately.
- DFlash fix: draft contexts now receive `ctx_other=ctx_tgt`, and the loop calls `common_speculative_process()` after each target decode so DFlash can inject target hidden states into the draft KV cache.
- Empty-draft guard: if a high `p_min` causes no draft tokens to be produced, the loop no longer calls `common_speculative_accept()` with a null last implementation.
- Same-prompt non-spec baseline:
  - `runs/qwen3_next_q4_moevm_normal_sameprompt_n32.log`, 32 tokens in 2.095s, 15.28 tok/s.
  - `runs/qwen3_next_q4_moevm_normal_sameprompt_n100.log`, 100 tokens in 5.764s, 17.35 tok/s.
- Qwen3-0.6B Q8 draft-simple matrix on the same prompt:
  - `draft1`: `qwen3_next_q4_moevm_spec_qwen06b_q8_draft1_n32.log`, 34 tokens in 2.537s, 13.40 tok/s, accept 100%.
  - `draft2`: `qwen3_next_q4_moevm_spec_qwen06b_q8_draft2_n32.log`, 33 tokens in 2.803s, 11.77 tok/s, accept 76.92%.
  - `draft4`: `qwen3_next_q4_moevm_spec_qwen06b_q8_n32.log`, 33 tokens in 3.104s, 10.63 tok/s, accept 50%.
  - `draft8`: `qwen3_next_q4_moevm_spec_qwen06b_q8_draft8_n32.log`, 33 tokens in 4.533s, 7.28 tok/s, accept 25%.
- DFlash Q8 matrix on the same prompt:
  - `draft8 p_min=0.0`: `qwen3_next_q4_moevm_spec_dflash_q8_draft8_n32_process.log`, 33 tokens in 4.041s, 8.17 tok/s, accept 25%.
  - `draft8 p_min=0.2`: `qwen3_next_q4_moevm_spec_dflash_q8_draft8_pmin02_n32.log`, 34 tokens in 3.474s, 9.79 tok/s, accept 40.39%.
  - `draft8 p_min=0.5`: `qwen3_next_q4_moevm_spec_dflash_q8_draft8_pmin05_n32_guard.log`, 33 tokens in 2.380s, 13.86 tok/s, accept 76.19%.
  - `draft15 p_min=0.5`: `qwen3_next_q4_moevm_spec_dflash_q8_draft15_pmin05_n32.log`, 33 tokens in 2.722s, 12.12 tok/s, accept 69.57%.
  - `draft8 p_min=0.5`, longer run: `qwen3_next_q4_moevm_spec_dflash_q8_draft8_pmin05_n100.log`, 101 tokens in 5.536s, 18.24 tok/s, accept 81.71%, `pool_fallback=0`, `fallback_k_overflow=0`.
- Audit rerun:
  - Normal cap64: `qwen3_next_q4_audit_normal_cap64_n100.log`, 100 tokens in 5.631s, 17.76 tok/s, `pool_fallback=0`, `fallback_k_overflow=0`.
  - Normal cap128: `qwen3_next_q4_audit_normal_cap128_n100.log`, 100 tokens in 5.492s, 18.21 tok/s, `pool_fallback=0`, `fallback_k_overflow=0`.
  - DFlash cap64: `qwen3_next_q4_audit_spec_dflash_cap64_draft8_pmin05_n100.log`, 101 tokens in 7.068s, 14.29 tok/s, accept 81.71%, `fallback_k_overflow=39`; cap64 is too small for this verification shape.
  - DFlash cap128: `qwen3_next_q4_audit_spec_dflash_cap128_draft8_pmin05_n100.log`, 101 tokens in 5.031s, 20.08 tok/s, accept 81.71%, `pool_fallback=0`, `fallback_k_overflow=0`.
- Verdict: speculation is integrated and correct enough to benchmark. On short `n=32`, checkpoint/draft overhead loses to non-spec. On audited `n=100`, DFlash cap128 is a real but still modest win over same-prompt non-spec: 20.08 vs 18.21 tok/s (+10.3%). Cap64 should not be used for DFlash because fallback dominates.
- Interpretation: DFlash needs enough generation length, cap128 pool capacity, and a conservative `p_min` to amortize checkpoint/draft overhead. Bigger gains need better accepted length, lower checkpoint overhead, or a non-checkpoint rollback path for Qwen3Next recurrent state.

Rejected experiment:

- Dense resident cache using llama.cpp's original `input_cpy` tensor was tested as `--moevm-resident-cache-unsafe`.
- It was fast on 20 tokens, 13.65 tok/s, and reduced copied bytes to 3.75GB, but it failed token parity: generated IDs became `[198,0,0,...]`.
- Token compare: `runs/qwen3_next_q3_decode20_resident_cache_keyed_token_compare.json`, `same_tokens=false`.
- Interpretation: the scheduler copy buffer cannot be treated as durable resident expert storage across splits. The safe production path remains graph-restored compact/pool tensors.

## Full-Resident Speed Check

Question: if Qwen3Next Q3 fits on the Mac, why is MoEVM only at 11-12 tok/s?

Result: full-resident llama.cpp/Metal is much faster than the MoEVM paging path.

- Command log: `runs/qwen3_next_q3_fullresident_llamabench_decode128.log`
- Model reported by `llama-bench`: `qwen3next 80B.A3B Q3_K - Medium`.
- Model size: 36,276,696,064 bytes.
- Params: 79,674,391,296.
- Placement: `-ngl 99`, `n_cpu_moe=0`, no tensor overrides.
- Prompt eval: 178.83 tok/s for 16 prompt tokens.
- Generation: 49.69 tok/s for 128 generated tokens.
- Safe-run memory: end wired 31.36GiB.

Interpretation:

- The public 25-30 tok/s expectation is plausible; this local full-resident run is even faster on a synthetic bench.
- The 11-12 tok/s MoEVM result is not the raw model limit. It is the current cost of the paging architecture: sidecar reads, expert copies, remapped compact pool tensors, scheduler graph patches, and trace callbacks.
- MoEVM is still valuable for models/quantizations that do not fit full-resident, or for proving expert-paging mechanics. For this exact Q3 model, full resident is the performance baseline to beat, not 8.10 tok/s.

## Code Bench Smoke

`llama-moe-trace` now supports `--generated-out <path>`, so benchmark harnesses can score generated text instead of only token IDs.

Local harness:

- Script: `phase0/code_bench.py`
- Summary: `runs/codebench_smoke/summary.md`
- JSON: `runs/codebench_smoke/summary.json`

Smoke result over three small Python tasks:

- Baseline same CPU-expert placement: 2/3 passed, 7.95 tok/s average.
- MoEVM pool cap10: 2/3 passed, 10.85 tok/s average.
- Pool generated exactly the same tokens and text as baseline on all 3 tasks.
- The failed task is a model/prompt quality miss on relative Unix paths, not a MoEVM divergence.

## Graph-Build Scheduler Hook

Goal: remove the old op-offload expert copy hook from the hot path by patching the built graph to use stable backend-resident expert pool tensors.

Implementation checkpoint:

- Branch: `codex/moevm-graph-build`.
- `llama_context` has an experimental graph transform hook called after `model.build_graph()` and before scheduler allocation.
- `llama-moe-trace --moevm-graph-mode pool` rewrites routed expert `MUL_MAT_ID` `src[0]` to `moevm_pool_*` tensors but keeps the original topk ids dependency at build time.
- `ggml-backend` has an experimental scheduler graph-pool callback. It forces a split before the first pool `MUL_MAT_ID` for each ids tensor, reads ids once, calls back into MoEVM to fill the layer pool, patches `src[2]` to shadow ids for the split, then restores graph sources after compute.
- The early `cb_eval` prototype is rejected for performance: `qwen3_next_q4_graph_pool_c192_n50.log` was correct but only 2.07 tok/s because it forced a scheduler sync around every topk callback.

Validation:

- Smoke: `runs/qwen3_next_q4_graph_sched_smoke_n1.log`, token matches reference, `copy_hook_calls=0`, `graph_patch_applied=864`, `graph_topk_fills=519`, `runtime_errors=0`.
- Throughput: `runs/qwen3_next_q4_graph_sched_c192_n50.log`, 50 tokens in 2.284s, 21.90 tok/s, `copy_hook_calls=0`, `callback_handled=8640`, `graph_fill_fallback=0`, `runtime_errors=0`.
- Logits parity: `runs/qwen3_next_q4_graph_parity_off_vs_sched_n50.json`, 50 rows, `max_abs=0.0`, zero token/argmax mismatches.
- Jitter/fence stress: `runs/qwen3_next_q4_graph_sched_jitter_delay_cap16_n20.log`, cap16, `--moevm-inject-delay-us 500`, `--moevm-test-force-pinned-fence`, `pool_forced_fences=1`, `pool_pinned_eviction_skips=15`, `runtime_errors=0`.
- Jitter logits parity: `runs/qwen3_next_q4_graph_sched_jitter_cap16_n20_compare.json`, `max_abs=0.0`.
- Phase A timestamp diagnostic: use `MOEVM_METAL_TIMESTAMPS=1` only for diagnostics, not headline throughput. `runs/qwen3_next_q4_graph_sched_timestamps_oldprompt_c192_n50.log` reports `metal_cmd_buffers=8799`, `metal_gpu_busy_us=1304967`, and zero missing timestamps over a 2.472s decode, so GPU busy is about 52.8%. The same run reports `ids_readback_us=1564882`, `backend_sync_us=1671520`, and `callback_us=1587336`.
- Timestamp control: `runs/qwen3_next_q4_graph_sched_no_timestamps_oldprompt2_c192_n50.log`, same tokens, 50 decode tokens in 2.427s, 20.60 tok/s, `metal_cmd_buffers=0`.
- Graph-mode prefetch fix: graph-build initially lost the legacy `prefetch_candidates()` call. It now reuses the same cached prefetch helper in graph-mode after deduplicating topk ids. This changes DFlash graph-mode from `copy_prefetches=0` / 18.17 tok/s to `copy_prefetches=2063` / 18.85 tok/s on the same DFlash run.
- Phase A DFlash rerun: best current graph-build DFlash is `runs/qwen3_next_q4_graph_dflash_prefetch_cap128_draft8_pmin05_n100.log`, 101 tokens in 5.359s, 18.85 tok/s, accept 81.71%, `graph_fill_fallback=0`. Same-prompt graph normal is `runs/qwen3_next_q4_graph_normal_sameprompt_cap128_n100.log`, 100 tokens in 5.368s, 18.63 tok/s. Legacy-path DFlash rerun is 19.58 tok/s.
- Negative DFlash A/Bs: `draft16 p_min=0.5` is 15.89 tok/s because it creates more topk fills/readbacks; `draft8 p_min=0.7` is 18.48 tok/s because it drafts less even though accepted draft tokens are cleaner.
- Fused graph-pool callbacks: the scheduler now calls the graph-pool callback once per topk ids tensor and reuses the returned shadow ids for the other component nodes in the same split. This reduces graph callback attempts from three per layer to one per layer.
- Expert-major graph fill: graph-mode fills all component pool tensors for one expert while holding a single pinned sidecar record, instead of pinning the same expert once per component.
- Validation: `runs/qwen3_next_q4_graph_expertmajor_parity_compare_n50.json` vs the previous graph-build logits has 50 rows, `max_abs=0.0`, zero token/argmax mismatches.
- Result: `runs/qwen3_next_q4_graph_expertmajor_normal_cap128_n100.log`, 100 tokens in 5.061s, 19.76 tok/s. `runs/qwen3_next_q4_graph_expertmajor_dflash_cap128_draft8_pmin05_n100.log`, 101 tokens in 5.039s, 20.04 tok/s, accept 81.71%.
- Negative capacity check: `runs/qwen3_next_q4_graph_expertmajor_cap256_sameprompt_c192_n50.log`, 50 tokens in 2.539s, 19.69 tok/s. Cap256 removed pool evictions but was slower than cap128; keep cap128 as the default.

Interpretation:

- Graph-build is now functionally real: it runs without the old MoEVM copy hook and without op-offload dependence.
- It is not yet a 25+ tok/s breakthrough. The current graph scheduler hook is roughly tied with the best warm P0.d/P0.c runs.
- A.1 says the GPU is not saturated; the practical bottleneck is still scheduler/CPU rendezvous around ids readback and graph-pool callbacks.
- DFlash is correct and fallback-free on graph-build, and after callback fusion it matches the old audited DFlash headline without the old copy-hook path. Next useful optimization is reducing per-layer ids readback/sync itself or reducing copied expert bytes.

## gpt-oss-120b Q4_K_M Runtime Replay

Safety boundary:

- Full `gpt-oss-120b` Metal inference remains blocked on this Mac. Prior Metal / FlashAttention tuning hit a Metal OOM path and a watchdog panic route.
- The validated work here is CPU-only route tracing, multi-shard sidecar packing, packed replay, and C++ MoEVM runtime replay with a real RAM arena.
- Do not run gpt-oss `-ngl` / `-fa` experiments without a new safety plan.

Layout:

- Source: `$MOEVM_MODELS_DIR/gpt-oss-120b/Q4_K_M/gpt-oss-120b-Q4_K_M-00001-of-00002.gguf`
- Shards: 2.
- Routed expert pool: 56.88GiB.
- Resident non-routed estimate: 1.57GiB.
- Layers: 36.
- Experts: 128.
- Active experts per token: 4.
- Record size: 12.64MiB per `(layer, expert)`.

Tooling fixes needed for gpt-oss:

- `gguf_inspect.py` now tracks `TensorInfo.source_path`.
- `sidecar_plan.py` now uses the owning shard for every tensor and validates slice bounds against the source shard size.
- This matters because layer 28 has 128 mixed records where gate/up live in shard 2 while down still lives in shard 1.
- `cache_sim.py` now supports `--stats-token-min` / `--stats-token-max`, so prefill-warmed decode can be measured separately from cold prefill.
- `sidecar_replay.py` skips `{"kind":"candidates"}` trace rows and replays only selected expert rows.
- `runtime_replay.cpp` and `runtime_smoke.cpp` now release pinned `RecordView`s after use. Without this, the arena cannot evict and gpt-oss replay fails with `arena allocation failed for selected expert`.
- `sidecar_pack.py` now includes alignment padding in its SHA-256 summary. Existing gpt-oss sidecar hash was recomputed from the actual file.
- `RuntimeOptions.verify_integrity` defaults to true. `runtime_replay --no-integrity-check`, `runtime_smoke --no-integrity-check`, and `llama-moe-trace --moevm-no-integrity-check` skip the full SHA read while keeping sidecar size checks.

Sidecar:

- Plan: `runs/gpt_oss_120b_q4_sidecar_plan.json`.
- Packed sidecar: `$MOEVM_MODELS_DIR/gpt-oss-120b/Q4_K_M/gpt-oss-120b-q4.moevm`.
- Index: `$MOEVM_MODELS_DIR/gpt-oss-120b/Q4_K_M/gpt-oss-120b-q4.moevm.idx`.
- Component index: `$MOEVM_MODELS_DIR/gpt-oss-120b/Q4_K_M/gpt-oss-120b-q4.moevm.cidx`.
- Records: 4,608.
- Payload: 56.74GiB.
- Pack time: 55.72s, 1.02GiB/s.
- Actual file SHA-256: `ac311e633a1b555054f6c687a253cdccf93615f8c1715b88f0f13008109d5f64`.

Trace:

- Trace: `runs/gpt_oss_120b_q4_cpu_trace_code_n128.jsonl`.
- Prompt tokens: 67.
- Generated tokens: 128.
- Total trace tokens: 195.
- Topk records: 6,954.
- Candidate records: 6,954.
- Expert accesses: 27,816.
- CPU-only decode: 128 tokens in 11.796s, 10.85 tok/s.

Decode-only cache simulation after prompt token 67:

- 24GiB LRU: hit 0.859, miss 256.8MB/tok, sync estimate 15.30 tok/s.
- 28GiB LRU: hit 0.924, miss 138.2MB/tok, sync estimate 24.02 tok/s.
- 32GiB LRU: hit 0.947, miss 95.6MB/tok, sync estimate 30.20 tok/s.

Packed replay over cold prompt+decode:

- 28GiB: hit 0.870, 3,614 reads, 44.49GiB read, 53.29 replay tok/s.
- 32GiB: hit 0.885, 3,195 reads, 39.33GiB read, 58.00 replay tok/s.
- At 28GiB, packed sidecar reduces read ops from 10,842 source-slice reads to 3,614 packed record reads.

C++ runtime replay with real arena and candidate prefetch:

- `candidate_topk=4` is the correct gpt-oss setting because active experts/token is 4. `candidate_topk=10` over-prefetches badly.
- 24GiB, top4: demand reads 0, read 59.87GiB, evictions 2,914, arena high-water 24.00GiB, 37.80 replay tok/s.
- 28GiB, top4: demand reads 0, read 44.54GiB, evictions 1,345, arena high-water 27.99GiB, 50.89 replay tok/s.
- 32GiB, top4: demand reads 0, read 39.35GiB, evictions 598, arena high-water 31.99GiB, 31.65 replay tok/s on this run. More RAM reduced reads but hurt wall-time, likely due memory pressure / lower read throughput.
- 28GiB, top10: demand reads 728, read 108.01GiB, evictions 6,500, 24.45 replay tok/s.

Cold-replay caveat:

- `sudo purge` was not available non-interactively, so no true cold page-cache replay was produced.
- A best-effort eviction read before replay still produced 46.22 replay tok/s while reading 44.54GiB. Treat that as warm/cache-contaminated evidence, not a publishable cold SSD number.
- Public claims should use demand-read behavior and cache locality. Replay tok/s is currently internal warm-runtime evidence.

Long gpt-oss traces:

- `cache_sim.py` supports `belady` as an offline-optimal upper bound.
- Three CPU-only coding traces at 512 generated tokens were collected: Python backend, TypeScript type utility, and Rust tokenizer.
- CPU-only decode stayed at 13.77-14.36 tok/s.
- 28GiB LRU decode-only hit rate was 95.10-95.75%; Belady at 28GiB was 97.15-97.49%.
- 24GiB LRU already gives 93.26-93.63% hit rate with 25.95-26.88 sync tok/s estimate.
- Runtime replay at 28GiB/candidate top4 on the three long traces had demand reads 1, 0, and 0 across about 241k expert accesses. Warm replay tok/s was 88.12-94.82 and is not a cold-public number.
- Research conclusion: LRU is quasi-optimal here. The 28GiB gap to offline-optimal Belady is only about 1.7-2.1 hit-rate points, so policy swaps like S3-FIFO/LFU are deprioritized.

Host-arena reclaim/trim guard:

- On macOS, evicted full pages use `MADV_FREE_REUSABLE`; pages get `MADV_FREE_REUSE` before being written again.
- `Runtime::trim_to_bytes(target_used_bytes)` evicts LRU records without replacing them, which is the path that creates real pressure relief.
- New runtime metrics: `arena_reclaim_advises`, `arena_reclaim_bytes`, `arena_reclaim_live_bytes`, `arena_reuse_advises`, and `arena_reuse_bytes`.
- Sequential smoke paliers on the Python long trace, 160 tokens:
  - 16GiB: 4 demand reads, 3,800 evictions, 46.79GiB cumulative reclaim, 0.00GiB live reclaim, 22.81 replay tok/s, safe-run end wired 2.79GiB.
  - 20GiB: 2 demand reads, 2,440 evictions, 30.05GiB cumulative reclaim, 0.00GiB live reclaim, 29.46 replay tok/s, safe-run end wired 2.79GiB.
  - 24GiB: 0 demand reads, 1,125 evictions, 13.85GiB cumulative reclaim, 0.00GiB live reclaim, 40.90 replay tok/s, safe-run end wired 2.80GiB.
- Footprint validation: no-trim 16GiB replay had `phys_footprint=17.19GB` and 0 reclaimable; trim-to-8GiB had `phys_footprint=8.60GB`, `footprint` reclaimable 8.59GB, and `arena_reclaim_live_bytes=8.59GB`.
- Conclusion: eviction-only advice is not a safety mechanism because pages are immediately reused. A pressure-triggered trim/shrink path is the safety mechanism.
- Do not run these paliers in parallel: a parallel 20GiB+24GiB attempt was correctly killed by `safe_run.py` on swapout delta.
- `safe_run.py` now includes `runtime_replay`, `runtime_smoke`, and `pager_replay` in model-I/O detection, and holds a nonblocking `/tmp/moevm_model_io.lock` flock for model-I/O commands. This closes the startup race where two safe_run processes could pass process-list preflight at the same time.
- gpt-oss Metal remains blocked by default. Explicit gates:
  - `--allow-gptoss-metal-smoke`: `-n <= 8`, MoEVM arena <= 16GiB.
  - `--allow-gptoss-metal-parity-gate`: `-n <= 128`, MoEVM arena <= 16GiB.
  - `--allow-gptoss-metal-scale-gate`: `-n <= 128`, MoEVM arena <= 24GiB, and `--moevm-pressure-trim` required above 16GiB.
  - For comparable Metal baselines without MoEVM: FlashAttention off, expert weight CPU override, explicit `--op-offload` or `--no-op-offload`, and `--no-repack`.

Metal smoke:

- `llama-moe-trace` has `--moevm-load-only` for true load-only checks that initialize model/context/runtime/graph hook and skip prompt prefill/decode.
- `llama-moe-trace` has `--moevm-pressure-trim`, backed by Darwin memory-pressure dispatch events. WARN trims by one configured step, CRITICAL by two, bounded by `--moevm-pressure-trim-min-gib`.
- Sparse-mmap loader content is present in the current branch (`llama_buf_map_find` plus post-load `unmap_fragment` of unused mmap ranges). The exact sparse-mmap commit is patch-equivalent but not an ancestor of this branch.
- Load-only: `gpt_oss_120b_q4_metal_loadonly_pool16.log`, `-ngl 99`, expert weights CPU, graph pool, arena 16GiB, pool cap16, FA off. Passed; graph-mode installed on `MTL0`; safe-run end wired 3.49GiB.
- Load-only with pressure trim armed: `gpt_oss_120b_q4_metal_moevm_pool16_load_only_pressure_trim.log`, passed; pressure listener armed; no pressure events.
- CPU baseline n8: `gpt_oss_120b_q4_cpu_baseline_n8_v2.log`, 5.74 tok/s, IDs `[279,1314,57388,6271,11,287,3127,622]`.
- Metal baseline n8, expert weights CPU: `gpt_oss_120b_q4_metal_baseline_cpu_experts_n8.log`, 4.65 tok/s, IDs `[279,1314,71973,6271,11,287,3127,622]`.
- MoEVM graph pool n8: `gpt_oss_120b_q4_metal_moevm_pool16_n8.log`, 8.71 tok/s, IDs `[279,1314,71973,6271,11,287,3127,622]`, demand reads 0, runtime high-water 12.63GiB, graph patch applied 540, runtime errors 0.
- CPU baseline vs MoEVM logits: 1 token/argmax mismatch, `max_abs=5.33`. The same mismatch exists CPU baseline vs Metal baseline, so this is CPU-vs-Metal numeric drift, not isolated MoEVM failure.
- Metal baseline vs MoEVM logits: 0 token mismatches, 0 argmax mismatches, `max_abs=1.36`, `mean_abs=0.0705`. This is not bit-exact because the baseline uses CPU expert matmuls while MoEVM uses Metal pool matmuls for selected experts.
- Same-backend Metal-vs-Metal parity gate: `gpt_oss_120b_q4_metal_opoffload_baseline_n64.log` vs `gpt_oss_120b_q4_metal_moevm_pool16_n64.log` gives 64 rows, 0 token mismatches, 0 argmax mismatches, `max_abs=0.0`. This is the discriminating test for the earlier `max_abs=1.36`: old op-offload Metal path vs MoEVM pool, same Metal backend and same MXFP4 dequant path. It rules out the suspected gpt-oss `ADD_ID` bias corruption on this tested graph-pool path.
- Pressure-trim regression gate: `gpt_oss_120b_q4_metal_moevm_pool16_n64_pressure_trim.log` remains bit-exact vs op-offload (`max_abs=0.0`), 11.44 tok/s, demand reads 0, no pressure events.

Metal scale:

- Short prompt, `c=128`, `n=64`, same op-offload baseline:
  - 16GiB: 11.44 tok/s, demand reads 0, runtime evictions 511, `max_abs=0.0`.
  - 20GiB: 13.34 tok/s, demand reads 0, runtime evictions 35, `max_abs=0.0`.
  - 24GiB: 13.24 tok/s, demand reads 0, runtime evictions 0, arena used only 20.38GiB, `max_abs=0.0`.
- Real-context prompt: `prompt_observability_512plus.txt`, 990 prompt tokens, `c=4096`, `n=128`.
  - 16GiB, `b=1 ub=1`: wall 86.99s, decode 9.37 tok/s, demand reads 11, prefetch reads 18,764, wait 19.42s, evictions 17,476.
  - 24GiB, `b=1 ub=1`: wall 76.42s, decode 9.94 tok/s, demand reads 0, prefetch reads 5,279, wait 6.80s, evictions 3,330, bit-exact vs 16GiB (`max_abs=0.0` over 128 rows).
  - 24GiB, `b=4 ub=4`: wall 64.05s, decode 10.19 tok/s, demand reads 360, prefetch reads 5,070, wait 5.49s, evictions 3,481.
- Timestamp diagnostic, real-context 24GiB `b=1 ub=1`, `n=64`: wall 71.88s, `metal_gpu_busy_us=20,067,878`, `ids_readback_us=25,785,434`, `callback_us=29,181,138`, `backend_sync_us=26,733,346`. The next bottleneck is rendez-vous/ids readback, not arena capacity.
- `b=4 ub=4` is same-backend correct: op-offload `ub=4` vs MoEVM `ub=4`, 16 rows, `max_abs=0.0`. It does not reproduce the `b=1 ub=1` token stream, so treat it as a validated batch mode, not a bit-identical replacement.
- `b=8 ub=8` with cap16 fails safely with `fallback_k_overflow=1` because the top-k union exceeds pool capacity. Do not increase pool cap above 16 without a separate narrower memory gate.

gpt-oss `ADD_ID` invariant:

- gpt-oss has per-expert bias tensors using `GGML_OP_ADD_ID`.
- Graph-pool build and scheduler paths currently patch only `GGML_OP_MUL_MAT_ID` nodes.
- Legacy sparse-copy also rejects non-`MUL_MAT_ID` first nodes.
- Keep `ADD_ID` bias ids untouched when moving toward Metal smoke tests.

Best current paging point:

- 28GiB arena.
- `candidate_topk=4`.
- 4 prefetch threads.
- 64MiB max in-flight.
- `--no-integrity-check` after initial sidecar validation.

Interpretation:

- This is now full bounded gpt-oss Metal/MoEVM inference on this Mac, not just replay: 56.88GiB of routed experts, selected expert paging, same-backend bit-exact gates, no jetsam/watchdog in bounded runs.
- Current publishable interactive number is the median-of-5 launch rerun: 8.72 tok/s on `c=4096`, 990-token prompt, `n=128`, arena 24GiB, `b=1 ub=1`, with zero demand reads, zero runtime errors, zero pressure events, and stable output prefix. The earlier 9.94 tok/s is a valid single-run observation, not the launch headline.
- Current best throughput mode for real context is `b=4 ub=4` at 64.05s wall for prompt+128 decode, but it changes the token stream relative to `b=1`; label it batch mode.
- Next engineering target is scheduler/readback reduction. Larger arena alone is no longer the main lever.
