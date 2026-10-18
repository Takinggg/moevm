#!/usr/bin/env python3
"""Quality gate for GPT-OSS active-expert demand shaping.

This benchmark compares MoEVM graph-pool with the model's normal top-k against
the same runtime with --moevm-active-experts=N. It is intentionally small: the
goal is to catch obvious code-quality regressions before treating active-k as a
launchable speed/quality knob.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from compare_decode_logs import parse_log


from moevm_paths import LLAMA_BIN, MODELS_DIR, RUNS, SAFE_RUN


DEFAULT_MODEL = MODELS_DIR / "gpt-oss-120b/Q4_K_M/gpt-oss-120b-Q4_K_M-00001-of-00002.gguf"
DEFAULT_INDEX = MODELS_DIR / "gpt-oss-120b/Q4_K_M/gpt-oss-120b-q4.moevm.idx"
DEFAULT_COMPONENT_INDEX = MODELS_DIR / "gpt-oss-120b/Q4_K_M/gpt-oss-120b-q4.moevm.cidx"
DEFAULT_SIDECAR = MODELS_DIR / "gpt-oss-120b/Q4_K_M/gpt-oss-120b-q4.moevm"


@dataclass(frozen=True)
class Task:
    name: str
    function_name: str
    prompt: str
    tests: str


TASKS = [
    Task(
        name="clamp",
        function_name="clamp",
        prompt=(
            "# Complete the function body. Output only valid Python code, no markdown.\n"
            "from typing import *\n\n"
            "def clamp(x: int, lo: int, hi: int) -> int:\n"
            "    \"\"\"Return x clipped to the inclusive range [lo, hi].\"\"\"\n"
            "    "
        ),
        tests=(
            "assert clamp(5, 0, 10) == 5\n"
            "assert clamp(-1, 0, 10) == 0\n"
            "assert clamp(11, 0, 10) == 10\n"
            "assert clamp(3, 3, 3) == 3\n"
        ),
    ),
    Task(
        name="count_vowels",
        function_name="count_vowels",
        prompt=(
            "# Complete the function body. Output only valid Python code, no markdown.\n"
            "from typing import *\n\n"
            "def count_vowels(text: str) -> int:\n"
            "    \"\"\"Count ASCII vowels in text, case-insensitively.\"\"\"\n"
            "    "
        ),
        tests=(
            "assert count_vowels('') == 0\n"
            "assert count_vowels('abc') == 1\n"
            "assert count_vowels('AEIOU xyz') == 5\n"
            "assert count_vowels('rhythm') == 0\n"
        ),
    ),
    Task(
        name="is_prime",
        function_name="is_prime",
        prompt=(
            "# Complete the function body. Output only valid Python code, no markdown.\n"
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
            "assert is_prime(0) is False\n"
            "assert is_prime(97) is True\n"
            "assert is_prime(100) is False\n"
        ),
    ),
    Task(
        name="merge_sorted",
        function_name="merge_sorted",
        prompt=(
            "# Complete the function body. Output only valid Python code, no markdown.\n"
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
        function_name="normalize_path",
        prompt=(
            "# Complete the function body. Output only valid Python code, no markdown.\n"
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
            "assert normalize_path('./a/./b') == 'a/b'\n"
        ),
    ),
]


def strip_fences(text: str) -> str:
    lines: list[str] = []
    in_fence = False
    for line in text.replace("\r\n", "\n").splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        lines.append(line)
    return "\n".join(lines)


def extract_full_function(text: str, function_name: str) -> str | None:
    clean = strip_fences(text)
    lines = clean.splitlines()
    start = None
    for i, line in enumerate(lines):
        if re.match(rf"^def\s+{re.escape(function_name)}\s*\(", line):
            start = i
            break
    if start is None:
        return None

    out: list[str] = ["from typing import *", ""]
    seen_body = False
    for line in lines[start:]:
        if line.startswith("if __name__"):
            break
        if line.strip() and not line.startswith((" ", "\t")) and seen_body:
            break
        if line.startswith((" ", "\t")):
            seen_body = True
        out.append(line)
    return "\n".join(out).rstrip() + "\n"


def extract_indented_body(text: str) -> str:
    clean = strip_fences(text)
    lines: list[str] = []
    seen_body = False
    for raw in clean.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not seen_body:
            if not stripped:
                continue
            if line.startswith((" ", "\t")):
                seen_body = True
                lines.append(line)
            continue
        if line.startswith("if __name__"):
            break
        if stripped and not line.startswith((" ", "\t")):
            break
        lines.append(line)
    return "\n".join(lines).rstrip() + "\n"


def candidate_source(task: Task, generated_text: str) -> str:
    full_function = extract_full_function(generated_text, task.function_name)
    if full_function is not None:
        return full_function

    body = extract_indented_body(generated_text)
    if body.strip():
        return task.prompt + body

    return task.prompt + strip_fences(generated_text).rstrip() + "\n"


def run_python_tests(source: str, tests: str, timeout_s: float) -> dict:
    with tempfile.TemporaryDirectory(prefix="moevm-gptoss-activek-") as tmp:
        script = Path(tmp) / "candidate.py"
        script.write_text(source + "\n\n" + tests + "\n", encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(script)],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "passed": False,
                "returncode": None,
                "stdout": (exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else "",
                "stderr": "test timeout",
            }
    return {
        "passed": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-2000:],
        "stderr": proc.stderr[-4000:],
    }


def mode_active_experts(mode: str, active_experts: int) -> int | None:
    if mode == "base":
        return None
    if mode.startswith("active"):
        suffix = mode.removeprefix("active").removesuffix("r")
        return int(suffix) if suffix else active_experts
    raise ValueError(f"unknown mode: {mode}")


def mode_renorm(mode: str) -> bool:
    return mode.startswith("active") and mode.endswith("r")


def build_command(task: Task, mode: str, args: argparse.Namespace, log: Path, summary: Path, trace: Path, generated: Path) -> list[str]:
    active_experts = mode_active_experts(mode, args.active_experts)
    return [
        sys.executable,
        str(SAFE_RUN),
        "--allow-gptoss-metal-scale-gate",
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
        "--moevm-pool-layer",
        "all",
        "--moevm-pool-capacity",
        str(args.pool_capacity),
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


def run_task(task: Task, mode: str, args: argparse.Namespace) -> dict:
    prefix = f"gptoss_activek_quality_{task.name}_{mode}"
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


def summarize(results: list[dict]) -> dict:
    by_mode: dict[str, list[dict]] = {}
    by_task: dict[str, dict[str, dict]] = {}
    for row in results:
        by_mode.setdefault(row["mode"], []).append(row)
        by_task.setdefault(row["task"], {})[row["mode"]] = row

    modes: dict[str, dict] = {}
    for mode, rows in by_mode.items():
        speeds = [row["parsed"].get("decoded", {}).get("tok_s") for row in rows]
        speeds = [speed for speed in speeds if isinstance(speed, (int, float))]
        demand_reads = [row["parsed"].get("runtime", {}).get("demand_reads") for row in rows]
        demand_reads = [value for value in demand_reads if isinstance(value, int)]
        runtime_errors = []
        for row in rows:
            log_path = Path(row["log"])
            if not log_path.exists():
                continue
            matches = re.findall(r"runtime_errors=(\d+)", log_path.read_text(errors="replace"))
            runtime_errors.append(int(matches[-1]) if matches else 0)
        modes[mode] = {
            "tasks": len(rows),
            "passed": sum(1 for row in rows if row["tests"]["passed"]),
            "avg_tok_s": sum(speeds) / len(speeds) if speeds else None,
            "all_returncode_zero": all(row["returncode"] == 0 for row in rows),
            "runtime_errors": sum(runtime_errors),
            "demand_reads_total": sum(demand_reads) if demand_reads else None,
        }

    comparisons: list[dict] = []
    regressions = 0
    base_passes = 0
    active_passes_when_base_passes = 0
    for task, rows in by_task.items():
        base = rows.get("base")
        if not base:
            continue
        base_pass = bool(base["tests"]["passed"])
        if base_pass:
            base_passes += 1
        base_tokens = base["parsed"].get("generated_token_ids")
        base_text = Path(base["generated"]).read_text(errors="replace") if Path(base["generated"]).exists() else None
        for mode, row in sorted(rows.items()):
            if mode == "base":
                continue
            active_pass = bool(row["tests"]["passed"])
            if base_pass and active_pass:
                active_passes_when_base_passes += 1
            if base_pass and not active_pass:
                regressions += 1
            active_tokens = row["parsed"].get("generated_token_ids")
            active_text = Path(row["generated"]).read_text(errors="replace") if Path(row["generated"]).exists() else None
            comparisons.append(
                {
                    "task": task,
                    "mode": mode,
                    "base_passed": base_pass,
                    "candidate_passed": active_pass,
                    "quality_regression": base_pass and not active_pass,
                    "same_tokens_as_base": base_tokens == active_tokens,
                    "same_text_as_base": base_text == active_text,
                }
            )

    gate = "pass"
    if regressions:
        gate = "fail"
    elif base_passes == 0:
        gate = "inconclusive"

    speedups: dict[str, float | None] = {}
    base_speed = modes.get("base", {}).get("avg_tok_s")
    for mode, row in modes.items():
        if mode == "base" or not isinstance(base_speed, (int, float)) or not isinstance(row.get("avg_tok_s"), (int, float)):
            continue
        speedups[mode] = row["avg_tok_s"] / base_speed if base_speed > 0 else None

    return {
        "gate": gate,
        "base_passes": base_passes,
        "active_passes_when_base_passes": active_passes_when_base_passes,
        "quality_regressions": regressions,
        "modes": modes,
        "speedups_vs_base": speedups,
        "comparisons": comparisons,
        "results": results,
    }


def write_markdown(summary: dict, path: Path) -> None:
    lines = ["# GPT-OSS Active-K Quality Gate", ""]
    lines.append(f"- gate: `{summary['gate']}`")
    lines.append(f"- base-passing tasks protected: `{summary['base_passes']}`")
    lines.append(f"- active passes on base-passing tasks: `{summary['active_passes_when_base_passes']}`")
    lines.append(f"- quality regressions: `{summary['quality_regressions']}`")
    lines.append("")
    lines.append("| Mode | Passed | Tasks | Avg tok/s | Speedup vs base | Demand reads |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for mode, row in summary["modes"].items():
        avg = row["avg_tok_s"]
        speedup = summary["speedups_vs_base"].get(mode)
        avg_s = f"{avg:.2f}" if isinstance(avg, (int, float)) else "n/a"
        speed_s = f"{speedup:.2f}x" if isinstance(speedup, (int, float)) else "-"
        demand = row["demand_reads_total"]
        demand_s = str(demand) if demand is not None else "n/a"
        lines.append(f"| {mode} | {row['passed']} | {row['tasks']} | {avg_s} | {speed_s} | {demand_s} |")

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
    parser.add_argument("--modes", default="base,active3", help="comma-separated: base,activeN,activeNr where r forces top-k weight renorm")
    parser.add_argument("--tasks", default=",".join(task.name for task in TASKS), help="comma-separated task names")
    parser.add_argument("--out-dir", type=Path, default=RUNS / "gptoss_activek_quality")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--component-index", type=Path, default=DEFAULT_COMPONENT_INDEX)
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument("--active-experts", type=int, default=3, help="default N for mode active")
    parser.add_argument("--arena-gib", type=float, default=24)
    parser.add_argument("--trim-min-gib", type=float, default=12)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--ubatch", type=int, default=1)
    parser.add_argument("--n-predict", type=int, default=128)
    parser.add_argument("--inflight-mib", type=float, default=64)
    parser.add_argument("--moevm-threads", type=int, default=4)
    parser.add_argument("--prefetch-topk", type=int, default=4)
    parser.add_argument("--pool-capacity", type=int, default=16)
    parser.add_argument("--seed", type=int, default=8001)
    parser.add_argument("--max-seconds", type=float, default=900)
    parser.add_argument("--preflight-min-free-pct", type=float, default=20)
    parser.add_argument("--min-free-pct", type=float, default=8)
    parser.add_argument("--max-rss-mib", type=float, default=42000)
    parser.add_argument("--max-wired-gib", type=float, default=38)
    parser.add_argument("--test-timeout-s", type=float, default=5)
    parser.add_argument("--reuse-existing", action="store_true", help="reuse existing log/generated files when present")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()

    if args.active_experts <= 0:
        raise SystemExit("--active-experts must be positive")
    if args.n_predict < 1 or args.n_predict > 128:
        raise SystemExit("--n-predict must be between 1 and 128 for safe_run GPT-OSS scale gate")
    if args.arena_gib > 24:
        raise SystemExit("--arena-gib must be <= 24 for safe_run GPT-OSS scale gate")
    if args.pool_capacity > 16:
        raise SystemExit("--pool-capacity must be <= 16 for safe_run GPT-OSS scale gate")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    task_names = {name for name in args.tasks.split(",") if name}
    selected_tasks = [task for task in TASKS if task.name in task_names]
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
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "gate": summary["gate"], "modes": summary["modes"], "speedups_vs_base": summary["speedups_vs_base"]}, indent=2))
    if any(row["returncode"] != 0 for row in results):
        return 1
    return 0 if summary["gate"] != "fail" else 2


if __name__ == "__main__":
    raise SystemExit(main())
