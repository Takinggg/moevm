#!/usr/bin/env python3
"""Validate a MoEVM launch soak directory against predeclared gates."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from compare_logits import compare as compare_logits  # noqa: E402


SAFE_END_FREE_RE = re.compile(r"^end_free_pct:\s+([0-9.]+)", re.MULTILINE)
SAFE_END_WIRED_RE = re.compile(r"^end_wired_gib:\s+([0-9.]+)", re.MULTILINE)
REAL_JETSAM_RE = re.compile(
    r"JetsamEvent|jettisoned|memorystatus: killing|highwater|per-process-limit|killed process",
    re.IGNORECASE,
)
BENIGN_JETSAM_RE = re.compile(
    r"Ignoring jetsam update|not RunningBoard jetsam managed|memorystatus_control error|"
    r"Kernel call to get coalition roles",
    re.IGNORECASE,
)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def parse_started_at(row: dict) -> datetime | None:
    value = row.get("started_at")
    if not isinstance(value, str):
        return None
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value)


def safe_summary_metrics(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    out: dict[str, float] = {}
    if m := SAFE_END_FREE_RE.search(text):
        out["end_free_pct"] = float(m.group(1))
    if m := SAFE_END_WIRED_RE.search(text):
        out["end_wired_gib"] = float(m.group(1))
    return out


def manifest_start(run_dir: Path) -> datetime | None:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    started_at = manifest.get("started_at")
    if not isinstance(started_at, str):
        return None
    return datetime.fromisoformat(started_at)


def new_jetsam_reports(started_at: datetime | None) -> list[str]:
    if started_at is None:
        return []
    roots = [
        Path.home() / "Library/Logs/DiagnosticReports",
        Path("/Library/Logs/DiagnosticReports"),
    ]
    out: list[str] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.glob("JetsamEvent*.ips"):
            try:
                mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=started_at.tzinfo)
            except OSError:
                continue
            if mtime >= started_at:
                out.append(str(path))
    return sorted(out)


def real_jetsam_lines(text: str) -> list[str]:
    lines = []
    for line in text.splitlines():
        if not REAL_JETSAM_RE.search(line):
            continue
        if BENIGN_JETSAM_RE.search(line):
            continue
        lines.append(line)
    return lines


def rel_hours(rows: list[dict]) -> list[tuple[dict, float]]:
    starts = [(row, parse_started_at(row)) for row in rows]
    starts = [(row, ts) for row, ts in starts if ts is not None]
    if not starts:
        return []
    first = min(ts for _, ts in starts)
    return [(row, (ts - first).total_seconds() / 3600.0) for row, ts in starts]


def median(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.median(values)


def drift_checks(rows: list[dict], threshold: float) -> tuple[list[dict], list[dict]]:
    """Return failures and pending checks for first-hour vs last-hour tok/s drift."""
    timed = rel_hours(rows)
    if not timed:
        return [], [{"check": "tok_s_drift", "reason": "no started_at timestamps"}]

    duration_h = max(hour for _, hour in timed)
    if duration_h < 2.0:
        return [], [{"check": "tok_s_drift", "reason": f"need >=2h, have {duration_h:.2f}h"}]

    first_rows = [row for row, hour in timed if hour < 1.0]
    last_start = max(0.0, duration_h - 1.0)
    last_rows = [row for row, hour in timed if hour >= last_start]

    failures: list[dict] = []
    pending: list[dict] = []
    prompts = sorted({row.get("prompt") for row in rows if row.get("prompt")})
    for prompt in prompts:
        first = [float(row["tok_s"]) for row in first_rows if row.get("prompt") == prompt and "tok_s" in row]
        last = [float(row["tok_s"]) for row in last_rows if row.get("prompt") == prompt and "tok_s" in row]
        if len(first) < 2 or len(last) < 2:
            pending.append({"check": "tok_s_drift", "prompt": prompt, "reason": "insufficient samples"})
            continue
        first_m = median(first)
        last_m = median(last)
        assert first_m is not None and last_m is not None
        drift = abs(last_m - first_m) / first_m if first_m else 0.0
        if drift > threshold:
            failures.append(
                {
                    "check": "tok_s_drift",
                    "prompt": prompt,
                    "first_hour_median": first_m,
                    "last_hour_median": last_m,
                    "drift": drift,
                    "threshold": threshold,
                }
            )
    return failures, pending


def logits_checks(rows: list[dict], max_abs_threshold: float) -> tuple[list[dict], list[dict]]:
    """Compare periodic logits snapshots for the same prompt against the first snapshot."""
    by_prompt: dict[str, list[Path]] = defaultdict(list)
    for row in rows:
        prompt = row.get("prompt")
        logits = row.get("logits")
        if not prompt or not logits:
            continue
        path = Path(str(logits))
        if path.exists():
            by_prompt[str(prompt)].append(path)

    failures: list[dict] = []
    pending: list[dict] = []
    if not by_prompt:
        return failures, [{"check": "logits_snapshots", "reason": "no logits snapshots yet"}]

    for prompt, paths in sorted(by_prompt.items()):
        if len(paths) < 2:
            pending.append({"check": "logits_snapshots", "prompt": prompt, "reason": "need at least two snapshots"})
            continue
        baseline = paths[0]
        for candidate in paths[1:]:
            summary = compare_logits(baseline, candidate)
            if (
                summary["token_mismatch_count"] != 0
                or summary["argmax_mismatch_count"] != 0
                or summary["max_abs"] > max_abs_threshold
            ):
                failures.append(
                    {
                        "check": "logits_snapshots",
                        "prompt": prompt,
                        "baseline": str(baseline),
                        "candidate": str(candidate),
                        "token_mismatch_count": summary["token_mismatch_count"],
                        "argmax_mismatch_count": summary["argmax_mismatch_count"],
                        "max_abs": summary["max_abs"],
                        "threshold": max_abs_threshold,
                    }
                )
    return failures, pending


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expected-tokens", type=int, default=64)
    parser.add_argument("--tok-s-drift-threshold", type=float, default=0.10)
    parser.add_argument("--logits-max-abs-threshold", type=float, default=0.0)
    parser.add_argument("--max-end-wired-gib", type=float, default=38.0)
    parser.add_argument("--max-end-wired-growth-gib", type=float, default=4.0)
    parser.add_argument("--final", action="store_true", help="require summary.json and jetsam_after.log")
    parser.add_argument("--json", type=Path, help="write validation result as JSON")
    args = parser.parse_args()

    run_dir = args.run_dir
    results_path = run_dir / "results.jsonl"
    summary_path = run_dir / "summary.json"
    jetsam_path = run_dir / "jetsam_after.log"
    started_at = manifest_start(run_dir)

    failures: list[dict] = []
    warnings: list[dict] = []
    pending: list[dict] = []

    if not results_path.exists():
        raise SystemExit(f"missing results: {results_path}")
    rows = read_jsonl(results_path)
    if not rows:
        failures.append({"check": "results", "reason": "no completed iterations"})

    if args.final:
        if not summary_path.exists():
            failures.append({"check": "summary", "reason": "summary.json missing"})
        if not jetsam_path.exists():
            failures.append({"check": "jetsam", "reason": "jetsam_after.log missing"})

    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if int(summary.get("failures", -1)) != 0:
            failures.append({"check": "summary.failures", "value": summary.get("failures")})

    for row in rows:
        iteration = row.get("iteration")
        if row.get("returncode") != 0:
            failures.append({"check": "returncode", "iteration": iteration, "value": row.get("returncode")})
        if int(row.get("runtime_errors", 0)) != 0:
            failures.append({"check": "runtime_errors", "iteration": iteration, "value": row.get("runtime_errors")})
        if int(row.get("pressure_critical_events", 0)) != 0:
            failures.append(
                {"check": "pressure_critical_events", "iteration": iteration, "value": row.get("pressure_critical_events")}
            )
        if int(row.get("pressure_warn_events", 0)) != 0:
            warnings.append({"check": "pressure_warn_events", "iteration": iteration, "value": row.get("pressure_warn_events")})
        if row.get("decoded_tokens") is not None and int(row["decoded_tokens"]) != args.expected_tokens:
            failures.append({"check": "decoded_tokens", "iteration": iteration, "value": row.get("decoded_tokens")})
        if row.get("token_count_logged") is not None and int(row["token_count_logged"]) != args.expected_tokens:
            failures.append({"check": "token_count_logged", "iteration": iteration, "value": row.get("token_count_logged")})

    safe_metrics = []
    for row in rows:
        safe_stdout = row.get("safe_stdout")
        if not safe_stdout:
            continue
        safe_path = Path(str(safe_stdout)).with_suffix(".safe_summary")
        if not safe_path.exists() and row.get("log"):
            safe_path = Path(str(row["log"])).with_suffix(".safe_summary")
        if safe_path.exists():
            metrics = safe_summary_metrics(safe_path)
            if metrics:
                metrics["iteration"] = row.get("iteration")
                safe_metrics.append(metrics)

    wired = [float(m["end_wired_gib"]) for m in safe_metrics if "end_wired_gib" in m]
    if wired:
        max_wired = max(wired)
        if max_wired > args.max_end_wired_gib:
            failures.append({"check": "end_wired_gib", "max": max_wired, "threshold": args.max_end_wired_gib})
        quarter = max(1, len(wired) // 4)
        start_m = median(wired[:quarter])
        end_m = median(wired[-quarter:])
        assert start_m is not None and end_m is not None
        growth = end_m - start_m
        if growth > args.max_end_wired_growth_gib:
            failures.append(
                {
                    "check": "end_wired_growth_gib",
                    "start_median": start_m,
                    "end_median": end_m,
                    "growth": growth,
                    "threshold": args.max_end_wired_growth_gib,
                }
            )
    elif args.final:
        warnings.append({"check": "end_wired_gib", "reason": "no safe_summary metrics found"})

    drift_failures, drift_pending = drift_checks(rows, args.tok_s_drift_threshold)
    failures.extend(drift_failures)
    pending.extend(drift_pending)

    logits_failures, logits_pending = logits_checks(rows, args.logits_max_abs_threshold)
    failures.extend(logits_failures)
    pending.extend(logits_pending)

    if jetsam_path.exists():
        jetsam_text = jetsam_path.read_text(encoding="utf-8", errors="replace")
        lines = real_jetsam_lines(jetsam_text)
        if lines:
            failures.append({"check": "jetsam_after", "matches": lines[:10], "bytes": jetsam_path.stat().st_size})
    elif not args.final:
        pending.append({"check": "jetsam_after", "reason": "not written until soak completion"})

    reports = new_jetsam_reports(started_at)
    if reports:
        failures.append({"check": "jetsam_reports", "reports": reports[:10], "count": len(reports)})

    result = {
        "run_dir": str(run_dir),
        "iterations": len(rows),
        "status": "fail" if failures else ("pending" if pending else "pass"),
        "failures": failures,
        "warnings": warnings,
        "pending": pending,
        "safe_summary_count": len(safe_metrics),
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.json:
        args.json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
