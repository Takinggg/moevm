#!/usr/bin/env python3
"""Launch gpt-oss MoEVM soak runs through safe_run.py.

The soak intentionally runs many bounded processes instead of one huge decode:
it exercises model load/unload, runtime startup, pressure-trim arming, prompt
prefill, decode, and cleanup repeatedly while keeping each run small enough for
safe_run.py to kill cleanly if a guard trips.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "moevm/runs"
SAFE_RUN = ROOT / "moevm/phase0/safe_run.py"
LLAMA_MOE_TRACE = ROOT / "external/llama.cpp/build/bin/llama-moe-trace"
DEFAULT_MODEL = Path("$MOEVM_MODELS_DIR/gpt-oss-120b/Q4_K_M/gpt-oss-120b-Q4_K_M-00001-of-00002.gguf")
DEFAULT_INDEX = Path("$MOEVM_MODELS_DIR/gpt-oss-120b/Q4_K_M/gpt-oss-120b-q4.moevm.idx")
DEFAULT_COMPONENT_INDEX = Path("$MOEVM_MODELS_DIR/gpt-oss-120b/Q4_K_M/gpt-oss-120b-q4.moevm.cidx")
DEFAULT_SIDECAR = Path("$MOEVM_MODELS_DIR/gpt-oss-120b/Q4_K_M/gpt-oss-120b-q4.moevm")
OBS_PROMPT = RUNS / "prompt_observability_512plus.txt"


@dataclass(frozen=True)
class Prompt:
    name: str
    text: str | None = None
    file: Path | None = None


PROMPTS = [
    Prompt("observability", file=OBS_PROMPT),
    Prompt(
        "python_queue",
        text=(
            "You are a senior backend engineer. Implement a production-grade Python module for an "
            "idempotent task queue with retry scheduling, exponential backoff, dead-letter handling, "
            "JSON persistence, deterministic tests, and careful error handling. Return code only."
        ),
    ),
    Prompt(
        "typescript_grid",
        text=(
            "You are a senior TypeScript engineer. Build a small client-side data grid engine with "
            "sorting, filtering, pagination, stable row identity, column definitions, async data loading, "
            "cancellation, and unit tests. Avoid external dependencies. Return code only."
        ),
    ),
    Prompt(
        "rust_scheduler",
        text=(
            "You are a senior Rust systems engineer. Implement a bounded multi-producer multi-consumer "
            "work scheduler with cancellation, backpressure, graceful shutdown, error propagation, and "
            "tests for race-prone edge cases. Use only the Rust standard library. Return code only."
        ),
    ),
]


DECODE_RE = re.compile(r"decoded\s+(\d+) / \1 tokens in\s+([0-9.]+)s,\s+([0-9.]+) tok/s")
TOKENS_RE = re.compile(r"generated_token_ids=\[([^\]]*)\]")
RUNTIME_RE = re.compile(
    r"MoEVM runtime demand_reads=(\d+) prefetch_reads=(\d+) waits=(\d+) wait_s=([0-9.]+) "
    r"evictions=(\d+) high_water=([0-9.]+)GiB"
)
PRESSURE_RE = re.compile(
    r"MoEVM pressure_trim warn_events=(\d+) critical_events=(\d+) trim_calls=(\d+) "
    r"trim_records=(\d+) trim_bytes=([0-9.]+)GiB"
)
ERROR_RE = re.compile(r"runtime_errors=(\d+)")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_show_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_text(cmd: list[str], timeout: float = 20.0) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=timeout)
    except Exception as exc:
        return f"command failed: {exc}\n"


def jetsam_snapshot(since: str, out_path: Path, fallback_minutes: int) -> None:
    predicate = (
        'eventMessage CONTAINS[c] "Jetsam" OR '
        'eventMessage CONTAINS[c] "memorystatus" OR '
        'eventMessage CONTAINS[c] "jettisoned"'
    )
    cmd = ["log", "show", "--start", since, "--style", "compact", "--predicate", predicate]
    text = run_text(cmd, timeout=60.0)
    if "returned non-zero exit status" in text or "Invalid" in text:
        text = run_text(
            ["log", "show", "--last", f"{fallback_minutes}m", "--style", "compact", "--predicate", predicate],
            timeout=60.0,
        )
    out_path.write_text(text, encoding="utf-8", errors="replace")


def parse_log(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    out: dict[str, object] = {}
    if m := DECODE_RE.search(text):
        out["decoded_tokens"] = int(m.group(1))
        out["decode_s"] = float(m.group(2))
        out["tok_s"] = float(m.group(3))
    if m := TOKENS_RE.search(text):
        ids = [int(x) for x in m.group(1).split(",") if x]
        out["token_prefix"] = ids[:16]
        out["token_count_logged"] = len(ids)
    if m := RUNTIME_RE.search(text):
        out.update(
            {
                "demand_reads": int(m.group(1)),
                "prefetch_reads": int(m.group(2)),
                "waits": int(m.group(3)),
                "wait_s": float(m.group(4)),
                "evictions": int(m.group(5)),
                "high_water_gib": float(m.group(6)),
            }
        )
    if m := PRESSURE_RE.search(text):
        out.update(
            {
                "pressure_warn_events": int(m.group(1)),
                "pressure_critical_events": int(m.group(2)),
                "pressure_trim_calls": int(m.group(3)),
                "pressure_trim_records": int(m.group(4)),
                "pressure_trim_gib": float(m.group(5)),
            }
        )
    errors = ERROR_RE.findall(text)
    if errors:
        out["runtime_errors"] = int(errors[-1])
    return out


def build_command(args: argparse.Namespace, prompt: Prompt, run_dir: Path, iteration: int, logits: bool) -> list[str]:
    stem = f"iter_{iteration:05d}_{prompt.name}"
    log = run_dir / f"{stem}.log"
    summary = run_dir / f"{stem}.safe_summary"
    generated = run_dir / f"{stem}.txt"
    logits_path = run_dir / f"{stem}.logits"

    cmd = [
        sys.executable,
        str(SAFE_RUN),
        "--allow-gptoss-metal-scale-gate",
        "--log",
        str(log),
        "--summary",
        str(summary),
        "--max-seconds",
        str(args.per_run_timeout),
        "--poll-seconds",
        "0.5",
        "--preflight-min-free-pct",
        str(args.preflight_min_free_pct),
        "--min-free-pct",
        str(args.min_free_pct),
        "--max-rss-mib",
        str(args.max_rss_mib),
        "--max-wired-gib",
        str(args.max_wired_gib),
        "--max-log-mb",
        str(args.max_log_mb),
        "--",
        "/usr/bin/env",
        "GGML_OP_OFFLOAD_MIN_BATCH=1",
        str(LLAMA_MOE_TRACE),
        "-m",
        str(args.model),
        "--moevm-index",
        str(args.index),
        "--moevm-component-index",
        str(args.component_index),
        "--moevm-sidecar",
        str(args.sidecar),
        "--moevm-arena-gib",
        str(args.arena_gib),
        "--moevm-inflight-mib",
        "64",
        "--moevm-threads",
        "4",
        "--moevm-prefetch-topk",
        "4",
        "--moevm-copy-mode",
        "memcpy",
        "--moevm-pool-layer",
        "all",
        "--moevm-pool-capacity",
        "16",
        "--moevm-graph-mode",
        "pool",
        "--moevm-no-integrity-check",
        "--moevm-pressure-trim",
        "--moevm-pressure-trim-step-gib",
        "2",
        "--moevm-pressure-trim-min-gib",
        str(args.trim_min_gib),
        "-ngl",
        "99",
        "-ot",
        ".*ffn_.*_exps.weight=CPU",
        "--no-op-offload",
        "--no-repack",
        "-fa",
        "off",
        "-c",
        str(args.ctx),
        "-b",
        str(args.batch),
        "-ub",
        str(args.ubatch),
        "-n",
        str(args.tokens),
        "--ignore-eos",
        "--temp",
        "0",
        "-s",
        str(args.seed_base + iteration),
    ]
    if prompt.file:
        cmd += ["-f", str(prompt.file)]
    else:
        cmd += ["-p", prompt.text or ""]
    if logits:
        cmd += ["--logits-out", str(logits_path)]
    cmd += ["--generated-out", str(generated)]
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=6.0)
    parser.add_argument("--iterations", type=int, help="override duration and run exactly N iterations")
    parser.add_argument("--out-dir", type=Path, default=RUNS / "gpt_oss_120b_launch_soak")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--component-index", type=Path, default=DEFAULT_COMPONENT_INDEX)
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument("--arena-gib", type=float, default=24)
    parser.add_argument("--trim-min-gib", type=float, default=12)
    parser.add_argument("--ctx", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--ubatch", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--logits-every", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=7000)
    parser.add_argument("--per-run-timeout", type=float, default=900)
    parser.add_argument("--preflight-min-free-pct", type=float, default=25)
    parser.add_argument("--min-free-pct", type=float, default=8)
    parser.add_argument("--max-rss-mib", type=float, default=42000)
    parser.add_argument("--max-wired-gib", type=float, default=38)
    parser.add_argument("--max-log-mb", type=float, default=96)
    parser.add_argument("--continue-on-failure", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.out_dir / "results.jsonl"
    manifest_path = args.out_dir / "manifest.json"
    jetsam_path = args.out_dir / "jetsam_after.log"
    start_iso = utc_now()
    start_log_show = log_show_time()
    start_monotonic = time.monotonic()
    deadline = start_monotonic + args.hours * 3600.0

    manifest = {
        "started_at": start_iso,
        "log_show_start": start_log_show,
        "hours": args.hours,
        "iterations": args.iterations,
        "arena_gib": args.arena_gib,
        "trim_min_gib": args.trim_min_gib,
        "ctx": args.ctx,
        "batch": args.batch,
        "ubatch": args.ubatch,
        "tokens": args.tokens,
        "logits_every": args.logits_every,
        "prompts": [{"name": p.name, "file": str(p.file) if p.file else None, "text": p.text} for p in PROMPTS],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    iteration = 0
    failures = 0
    with results_path.open("a", encoding="utf-8") as results:
        while True:
            if args.iterations is not None:
                if iteration >= args.iterations:
                    break
            elif time.monotonic() >= deadline:
                break

            prompt = PROMPTS[iteration % len(PROMPTS)]
            take_logits = args.logits_every > 0 and iteration % args.logits_every == 0
            cmd = build_command(args, prompt, args.out_dir, iteration, take_logits)
            stdout_path = args.out_dir / f"iter_{iteration:05d}_{prompt.name}.safe_stdout"
            run_started = time.time()
            with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout:
                proc = subprocess.run(cmd, stdout=stdout, stderr=subprocess.STDOUT)
            elapsed = time.time() - run_started

            log_path = args.out_dir / f"iter_{iteration:05d}_{prompt.name}.log"
            row = {
                "iteration": iteration,
                "prompt": prompt.name,
                "started_at": datetime.fromtimestamp(run_started, timezone.utc).isoformat(),
                "elapsed_s": elapsed,
                "returncode": proc.returncode,
                "log": str(log_path),
                "safe_stdout": str(stdout_path),
                "logits": str(args.out_dir / f"iter_{iteration:05d}_{prompt.name}.logits") if take_logits else None,
            }
            if log_path.exists():
                row.update(parse_log(log_path))
            if proc.returncode != 0:
                failures += 1
                row["failure"] = True
            results.write(json.dumps(row, sort_keys=True) + "\n")
            results.flush()

            print(json.dumps(row, sort_keys=True), flush=True)
            iteration += 1
            if proc.returncode != 0 and not args.continue_on_failure:
                break

    fallback_minutes = max(1, int((time.monotonic() - start_monotonic) / 60.0) + 2)
    jetsam_snapshot(start_log_show, jetsam_path, fallback_minutes)

    summary = {
        "started_at": start_iso,
        "finished_at": utc_now(),
        "iterations": iteration,
        "failures": failures,
        "results": str(results_path),
        "jetsam_log": str(jetsam_path),
        "jetsam_bytes": jetsam_path.stat().st_size if jetsam_path.exists() else None,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
