#!/usr/bin/env python3
"""Quality gate for Qwen active-expert demand shaping.

This compares the validated Qwen graph-pool path against the same path with a
smaller active top-k. It is a small generated-code gate: the goal is to reject
obvious correctness regressions before treating active-k as a launchable
quality/speed knob.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from compare_decode_logs import parse_log
from gptoss_activek_quality import (
    TASKS,
    candidate_source,
    mode_active_experts,
    mode_renorm,
    run_python_tests,
    summarize,
)


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "moevm/runs"
SAFE_RUN = ROOT / "moevm/phase0/safe_run.py"
LLAMA_BIN = ROOT / "external/llama.cpp/build/bin/llama-moe-trace"

DEFAULT_MODEL = Path("$MOEVM_MODELS_DIR/qwen3-coder-next-q4/Qwen3-Coder-Next-Q4_K_M.gguf")
DEFAULT_INDEX = Path("$MOEVM_MODELS_DIR/qwen3-coder-next-q4/qwen3-next-q4.moevm.idx")
DEFAULT_COMPONENT_INDEX = Path("$MOEVM_MODELS_DIR/qwen3-coder-next-q4/qwen3-next-q4.moevm.cidx")
DEFAULT_SIDECAR = Path("$MOEVM_MODELS_DIR/qwen3-coder-next-q4/qwen3-next-q4.moevm")


def build_command(task, mode: str, args: argparse.Namespace, log: Path, summary: Path, trace: Path, generated: Path) -> list[str]:
    active_experts = mode_active_experts(mode, args.active_experts)
    return [
        sys.executable,
        str(SAFE_RUN),
        "--log",
        str(log),
        "--summary",
        str(summary),
        "--max-seconds",
        str(args.max_seconds),
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
        "96",
        "--",
        "/usr/bin/env",
        "GGML_OP_OFFLOAD_MIN_BATCH=1",
        str(LLAMA_BIN),
        "-m",
        str(args.model),
        "--trace-out",
        str(trace),
        "--generated-out",
        str(generated),
        "--trace-candidates",
        "10",
        "--moevm-index",
        str(args.index),
        "--moevm-component-index",
        str(args.component_index),
        "--moevm-sidecar",
        str(args.sidecar),
        "--moevm-arena-gib",
        str(args.arena_gib),
        "--moevm-inflight-mib",
        str(args.inflight_mib),
        "--moevm-threads",
        str(args.moevm_threads),
        "--moevm-prefetch-topk",
        str(args.prefetch_topk),
        *(["--moevm-active-experts", str(active_experts)] if active_experts is not None else []),
        *(["--moevm-active-experts-renorm"] if mode_renorm(mode) else []),
        "--moevm-copy-mode",
        "memcpy",
        "--moevm-cache",
        "--moevm-pool-layer",
        "all",
        "--moevm-pool-capacity",
        str(args.pool_capacity),
        "--moevm-graph-mode",
        "pool",
        "--moevm-no-integrity-check",
        "-ngl",
        "99",
        "-ot",
        ".*ffn_.*_exps.weight=CPU",
        "--no-op-offload",
        "--no-repack",
        "-c",
        str(args.context),
        "-b",
        str(args.batch),
        "-ub",
        str(args.ubatch),
        "-n",
        str(args.n_predict),
        "--ignore-eos",
        "--temp",
        "0",
        "-s",
        str(args.seed),
        "-p",
        task.prompt,
    ]


def run_task(task, mode: str, args: argparse.Namespace) -> dict:
    prefix = f"qwen_activek_quality_{task.name}_{mode}"
    log = args.out_dir / f"{prefix}.log"
    summary = args.out_dir / f"{prefix}.safe_summary"
    trace = args.out_dir / f"{prefix}.jsonl"
    generated = args.out_dir / f"{prefix}.txt"
    source_path = args.out_dir / f"{prefix}.candidate.py"

    cmd = build_command(task, mode, args, log, summary, trace, generated)
    if args.reuse_existing and log.exists() and generated.exists():
        returncode = 0
        safe_stdout = ""
        safe_stderr = ""
    else:
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        returncode = proc.returncode
        safe_stdout = proc.stdout[-4000:]
        safe_stderr = proc.stderr[-4000:]

    try:
        parsed = parse_log(log) if log.exists() else {}
    except Exception as exc:
        parsed = {"parse_error": str(exc)}

    generated_text = generated.read_text(errors="replace") if generated.exists() else ""
    source = candidate_source(task, generated_text)
    source_path.write_text(source, encoding="utf-8")
    test_result = run_python_tests(source, task.tests, args.test_timeout_s)

    return {
        "task": task.name,
        "mode": mode,
        "returncode": returncode,
        "safe_stdout": safe_stdout,
        "safe_stderr": safe_stderr,
        "log": str(log),
        "summary": str(summary),
        "trace": str(trace),
        "generated": str(generated),
        "candidate": str(source_path),
        "generated_preview": generated_text[:1200],
        "parsed": parsed,
        "tests": test_result,
    }


def write_markdown(summary: dict, path: Path) -> None:
    lines = ["# Qwen Active-K Quality Gate", ""]
    lines.append(f"- gate: `{summary['gate']}`")
    lines.append(f"- base-passing tasks protected: `{summary['base_passes']}`")
    lines.append(f"- active passes on base-passing tasks: `{summary['active_passes_when_base_passes']}`")
    lines.append(f"- quality regressions: `{summary['quality_regressions']}`")
    lines.append("")
    lines.append("| Mode | Passed | Tasks | Avg tok/s | Speedup vs base | Demand reads | Runtime errors |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for mode, row in summary["modes"].items():
        avg = row["avg_tok_s"]
        speedup = summary["speedups_vs_base"].get(mode)
        avg_s = f"{avg:.2f}" if isinstance(avg, (int, float)) else "n/a"
        speed_s = f"{speedup:.2f}x" if isinstance(speedup, (int, float)) else "-"
        demand = row["demand_reads_total"]
        demand_s = str(demand) if demand is not None else "n/a"
        lines.append(
            f"| {mode} | {row['passed']} | {row['tasks']} | {avg_s} | {speed_s} | "
            f"{demand_s} | {row['runtime_errors']} |"
        )

    lines.append("")
    lines.append("## Comparisons")
    lines.append("")
    lines.append("| Task | Mode | Base passed | Candidate passed | Regression | Same tokens |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for row in summary["comparisons"]:
        lines.append(
            f"| {row['task']} | {row['mode']} | `{row['base_passed']}` | "
            f"`{row['candidate_passed']}` | `{row['quality_regression']}` | `{row['same_tokens_as_base']}` |"
        )

    lines.append("")
    lines.append("## Outputs")
    for row in summary["results"]:
        decoded = row["parsed"].get("decoded", {})
        lines.append("")
        lines.append(f"### {row['task']} / {row['mode']}")
        lines.append(f"- passed: `{row['tests']['passed']}`")
        lines.append(f"- tok/s: `{decoded.get('tok_s')}`")
        lines.append(f"- generated: `{row['generated']}`")
        lines.append(f"- candidate: `{row['candidate']}`")
        if not row["tests"]["passed"] and row["tests"].get("stderr"):
            lines.append("")
            lines.append("```text")
            lines.append(row["tests"]["stderr"][-1200:])
            lines.append("```")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", default="base,active8,active6", help="comma-separated: base,activeN,activeNr where r forces top-k weight renorm")
    parser.add_argument("--tasks", default=",".join(task.name for task in TASKS), help="comma-separated task names")
    parser.add_argument("--out-dir", type=Path, default=RUNS / "qwen_activek_quality")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--component-index", type=Path, default=DEFAULT_COMPONENT_INDEX)
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument("--active-experts", type=int, default=8, help="default N for mode active")
    parser.add_argument("--arena-gib", type=float, default=1)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--ubatch", type=int, default=1)
    parser.add_argument("--n-predict", type=int, default=192)
    parser.add_argument("--inflight-mib", type=float, default=64)
    parser.add_argument("--moevm-threads", type=int, default=4)
    parser.add_argument("--prefetch-topk", type=int, default=10)
    parser.add_argument("--pool-capacity", type=int, default=128)
    parser.add_argument("--seed", type=int, default=8001)
    parser.add_argument("--max-seconds", type=float, default=600)
    parser.add_argument("--preflight-min-free-pct", type=float, default=5)
    parser.add_argument("--min-free-pct", type=float, default=4)
    parser.add_argument("--max-rss-mib", type=float, default=70000)
    parser.add_argument("--max-wired-gib", type=float, default=42)
    parser.add_argument("--test-timeout-s", type=float, default=5)
    parser.add_argument("--reuse-existing", action="store_true", help="reuse existing log/generated files when present")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()

    if args.active_experts <= 0:
        raise SystemExit("--active-experts must be positive")
    if args.n_predict < 1:
        raise SystemExit("--n-predict must be positive")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    task_names = {name for name in args.tasks.split(",") if name}
    selected_tasks = [task for task in TASKS if task.name in task_names]
    if not selected_tasks:
        raise SystemExit("no tasks selected")
    modes = [mode for mode in args.modes.split(",") if mode]

    results: list[dict] = []
    for task in selected_tasks:
        for mode in modes:
            results.append(run_task(task, mode, args))

    summary = summarize(results)
    json_path = args.json or args.out_dir / "summary.json"
    md_path = args.markdown or args.out_dir / "summary.md"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_markdown(summary, md_path)
    print(
        json.dumps(
            {
                "json": str(json_path),
                "markdown": str(md_path),
                "gate": summary["gate"],
                "modes": summary["modes"],
                "speedups_vs_base": summary["speedups_vs_base"],
            },
            indent=2,
        )
    )
    if any(row["returncode"] != 0 for row in results):
        return 1
    return 0 if summary["gate"] != "fail" else 2


if __name__ == "__main__":
    raise SystemExit(main())
