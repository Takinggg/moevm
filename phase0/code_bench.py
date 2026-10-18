#!/usr/bin/env python3
"""Run small deterministic code-generation benchmarks through llama-moe-trace.

This is not a replacement for HumanEval/MBPP. It is a local smoke benchmark
that exercises real generated text, token parity, speed, and simple Python
correctness before spending time on larger benchmark suites.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from compare_decode_logs import parse_log


from moevm_paths import LLAMA_BIN, MODELS_DIR, RUNS, SAFE_RUN


MODEL = MODELS_DIR / "qwen3-coder-next/Qwen3-Coder-Next-UD-Q3_K_XL.gguf"
INDEX = MODELS_DIR / "qwen3-coder-next/qwen3-next-q3.moevm.idx"
COMPONENT_INDEX = MODELS_DIR / "qwen3-coder-next/qwen3-next-q3.moevm.cidx"
SIDECAR = MODELS_DIR / "qwen3-coder-next/qwen3-next-q3.moevm"


@dataclass(frozen=True)
class Task:
    name: str
    prompt: str
    tests: str


TASKS = [
    Task(
        name="is_prime",
        prompt=(
            "from typing import *\n\n"
            "def is_prime(n: int) -> bool:\n"
            "    \"\"\"Return True if n is a prime number, otherwise False.\"\"\"\n"
            "    "
        ),
        tests=(
            "assert is_prime(2) is True\n"
            "assert is_prime(3) is True\n"
            "assert is_prime(4) is False\n"
            "assert is_prime(1) is False\n"
            "assert is_prime(97) is True\n"
            "assert is_prime(100) is False\n"
        ),
    ),
    Task(
        name="merge_sorted",
        prompt=(
            "from typing import *\n\n"
            "def merge_sorted(a: list[int], b: list[int]) -> list[int]:\n"
            "    \"\"\"Merge two sorted integer lists into one sorted list.\"\"\"\n"
            "    "
        ),
        tests=(
            "assert merge_sorted([], []) == []\n"
            "assert merge_sorted([1, 3], [2, 4]) == [1, 2, 3, 4]\n"
            "assert merge_sorted([1, 1, 5], [1, 2]) == [1, 1, 1, 2, 5]\n"
            "assert merge_sorted([-3, 0], [-2, 2]) == [-3, -2, 0, 2]\n"
        ),
    ),
    Task(
        name="normalize_path",
        prompt=(
            "from typing import *\n\n"
            "def normalize_path(path: str) -> str:\n"
            "    \"\"\"Normalize a Unix path by resolving '.', '..', and duplicate slashes.\"\"\"\n"
            "    "
        ),
        tests=(
            "assert normalize_path('/a//b/./c/../d') == '/a/b/d'\n"
            "assert normalize_path('/../') == '/'\n"
            "assert normalize_path('/home/user/') == '/home/user'\n"
            "assert normalize_path('a/b/../c') == 'a/c'\n"
        ),
    ),
]


def mode_args(mode: str, pool_capacity: int) -> list[str]:
    if mode == "baseline":
        return []
    args = [
        "--moevm-index",
        str(INDEX),
        "--moevm-component-index",
        str(COMPONENT_INDEX),
        "--moevm-sidecar",
        str(SIDECAR),
        "--moevm-arena-gib",
        "1",
        "--moevm-inflight-mib",
        "256",
        "--moevm-threads",
        "12",
        "--moevm-prefetch-topk",
        "10",
        "--moevm-copy-debug",
        "0",
        "--moevm-copy-mode",
        "memcpy",
        "--moevm-cache",
    ]
    if mode == "pool":
        args += ["--moevm-pool-layer", "all", "--moevm-pool-capacity", str(pool_capacity)]
    elif mode != "sparse":
        raise ValueError(f"unknown mode: {mode}")
    return args


def clean_completion(text: str) -> str:
    lines = []
    in_fence = False
    seen_indented_body = False
    for line in text.replace("\r\n", "\n").splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if line.startswith("if __name__"):
            break
        if line.strip() and not line.startswith((" ", "\t")) and seen_indented_body:
            break
        if line.startswith((" ", "\t")):
            seen_indented_body = True
        lines.append(line)
    cleaned = "\n".join(lines).rstrip() + "\n"
    return cleaned


def run_python_tests(source: str, tests: str, timeout_s: float) -> dict:
    with tempfile.TemporaryDirectory(prefix="moevm-codebench-") as tmp:
        script = Path(tmp) / "candidate.py"
        script.write_text(source + "\n\n" + tests + "\n", encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, str(script)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
        )
    return {
        "passed": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-2000:],
        "stderr": proc.stderr[-4000:],
    }


def run_task(task: Task, mode: str, args: argparse.Namespace) -> dict:
    prefix = f"codebench_{task.name}_{mode}"
    log = args.out_dir / f"{prefix}.log"
    summary = args.out_dir / f"{prefix}.summary"
    trace = args.out_dir / f"{prefix}.jsonl"
    generated = args.out_dir / f"{prefix}.txt"

    cmd = [
        sys.executable,
        str(SAFE_RUN),
        "--log",
        str(log),
        "--summary",
        str(summary),
        "--max-rss-mib",
        "44032",
        "--max-wired-gib",
        "40.5",
        "--min-free-pct",
        "5",
        "--preflight-min-free-pct",
        "5",
        "--max-seconds",
        str(args.max_seconds),
        "--",
        "env",
        "GGML_OP_OFFLOAD_MIN_BATCH=1",
        str(LLAMA_BIN),
        "-m",
        str(MODEL),
        "--trace-out",
        str(trace),
        "--generated-out",
        str(generated),
        "--trace-candidates",
        "10",
        *mode_args(mode, args.pool_capacity),
        "-ngl",
        "99",
        "-ot",
        ".*ffn_.*_exps.weight=CPU",
        "--op-offload",
        "--no-repack",
        "-c",
        str(args.context),
        "-b",
        "16",
        "-ub",
        "16",
        "-n",
        str(args.n_predict),
        "-p",
        task.prompt,
    ]
    if args.reuse_existing and log.exists() and generated.exists():
        proc_returncode = 0
        safe_stdout = ""
        safe_stderr = ""
    else:
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        proc_returncode = proc.returncode
        safe_stdout = proc.stdout[-4000:]
        safe_stderr = proc.stderr[-4000:]
    parsed = parse_log(log) if log.exists() else {}
    generated_text = generated.read_text(errors="replace") if generated.exists() else ""
    candidate_source = task.prompt + clean_completion(generated_text)
    test_result = run_python_tests(candidate_source, task.tests, args.test_timeout_s)
    return {
        "task": task.name,
        "mode": mode,
        "returncode": proc_returncode,
        "safe_stdout": safe_stdout,
        "safe_stderr": safe_stderr,
        "log": str(log),
        "summary": str(summary),
        "trace": str(trace),
        "generated": str(generated),
        "generated_preview": generated_text[:1000],
        "parsed": parsed,
        "tests": test_result,
    }


def summarize(results: list[dict]) -> dict:
    by_mode: dict[str, list[dict]] = {}
    for row in results:
        by_mode.setdefault(row["mode"], []).append(row)
    modes = {}
    for mode, rows in by_mode.items():
        speeds = [row["parsed"].get("decoded", {}).get("tok_s") for row in rows]
        speeds = [speed for speed in speeds if isinstance(speed, (int, float))]
        modes[mode] = {
            "tasks": len(rows),
            "passed": sum(1 for row in rows if row["tests"]["passed"]),
            "avg_tok_s": sum(speeds) / len(speeds) if speeds else None,
            "all_returncode_zero": all(row["returncode"] == 0 for row in rows),
        }
    parity = []
    by_task: dict[str, dict[str, dict]] = {}
    for row in results:
        by_task.setdefault(row["task"], {})[row["mode"]] = row
    for task, rows in by_task.items():
        baseline = rows.get("baseline")
        if not baseline:
            continue
        baseline_tokens = baseline["parsed"].get("generated_token_ids")
        baseline_text_path = baseline.get("generated")
        baseline_text = Path(baseline_text_path).read_text(errors="replace") if baseline_text_path and Path(baseline_text_path).exists() else None
        for mode, row in rows.items():
            if mode == "baseline":
                continue
            tokens = row["parsed"].get("generated_token_ids")
            text_path = row.get("generated")
            text = Path(text_path).read_text(errors="replace") if text_path and Path(text_path).exists() else None
            parity.append({
                "task": task,
                "mode": mode,
                "same_tokens_as_baseline": baseline_tokens == tokens,
                "same_text_as_baseline": baseline_text == text,
            })
    return {"modes": modes, "parity": parity, "results": results}


def write_markdown(summary: dict, path: Path) -> None:
    lines = ["# MoEVM Code Bench", ""]
    lines.append("| Mode | Passed | Tasks | Avg tok/s |")
    lines.append("|---|---:|---:|---:|")
    for mode, row in summary["modes"].items():
        avg = row["avg_tok_s"]
        lines.append(f"| {mode} | {row['passed']} | {row['tasks']} | {avg:.2f} |" if avg is not None else f"| {mode} | {row['passed']} | {row['tasks']} | n/a |")
    lines.append("")
    if summary.get("parity"):
        lines.append("## Parity")
        lines.append("")
        lines.append("| Task | Mode | Same tokens | Same text |")
        lines.append("|---|---|---:|---:|")
        for row in summary["parity"]:
            lines.append(f"| {row['task']} | {row['mode']} | `{row['same_tokens_as_baseline']}` | `{row['same_text_as_baseline']}` |")
        lines.append("")
    lines.append("## Tasks")
    for row in summary["results"]:
        decoded = row["parsed"].get("decoded", {})
        lines.append("")
        lines.append(f"### {row['task']} / {row['mode']}")
        lines.append(f"- passed: `{row['tests']['passed']}`")
        lines.append(f"- tok/s: `{decoded.get('tok_s')}`")
        lines.append(f"- generated: `{row['generated']}`")
        if not row["tests"]["passed"]:
            err = row["tests"].get("stderr", "").strip()
            if err:
                lines.append("")
                lines.append("```text")
                lines.append(err[-1200:])
                lines.append("```")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", default="baseline,pool", help="comma-separated: baseline,sparse,pool")
    parser.add_argument("--tasks", default=",".join(task.name for task in TASKS), help="comma-separated task names")
    parser.add_argument("--out-dir", type=Path, default=RUNS / "codebench")
    parser.add_argument("--n-predict", type=int, default=192)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--pool-capacity", type=int, default=10)
    parser.add_argument("--max-seconds", type=float, default=420)
    parser.add_argument("--test-timeout-s", type=float, default=5)
    parser.add_argument("--reuse-existing", action="store_true", help="reuse existing log/generated files when present")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    task_names = {name for name in args.tasks.split(",") if name}
    selected_tasks = [task for task in TASKS if task.name in task_names]
    modes = [mode for mode in args.modes.split(",") if mode]

    results = []
    for task in selected_tasks:
        for mode in modes:
            results.append(run_task(task, mode, args))

    summary = summarize(results)
    json_path = args.json or args.out_dir / "summary.json"
    md_path = args.markdown or args.out_dir / "summary.md"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_markdown(summary, md_path)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "modes": summary["modes"]}, indent=2))
    return 0 if all(row["returncode"] == 0 for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
