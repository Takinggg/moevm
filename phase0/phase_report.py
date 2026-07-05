#!/usr/bin/env python3
"""Summarize MoEVM prefill/decode phase counters from gptoss_soak results."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def number(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = round((len(ordered) - 1) * pct)
    return float(ordered[idx])


def ratio(num: float, den: float) -> float | None:
    if den <= 0:
        return None
    return num / den


def phase_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        phases = row.get("phases")
        if not isinstance(phases, dict):
            continue
        for name, phase in phases.items():
            if isinstance(name, str) and isinstance(phase, dict):
                item = dict(phase)
                item["_prompt"] = row.get("prompt")
                item["_iteration"] = row.get("iteration")
                out.setdefault(name, []).append(item)
    return out


def summarize_phase(rows: list[dict[str, Any]]) -> dict[str, Any]:
    elapsed_us = [number(row.get("elapsed_us")) for row in rows]
    total_elapsed_us = sum(elapsed_us)
    total_hits = sum(number(row.get("hits")) for row in rows)
    total_misses = sum(number(row.get("misses")) for row in rows)
    total_gpu_busy_us = sum(number(row.get("metal_gpu_busy_us")) for row in rows)
    total_graph_compute_us = sum(number(row.get("graph_compute_us")) for row in rows)
    total_backend_sync_us = sum(number(row.get("backend_sync_us")) for row in rows)
    total_ids_readback_us = sum(number(row.get("ids_readback_us")) for row in rows)
    total_callback_us = sum(number(row.get("callback_us")) for row in rows)
    total_demand_reads = sum(number(row.get("demand_reads")) for row in rows)
    total_prefetch_reads = sum(number(row.get("prefetch_reads")) for row in rows)
    total_wait_s = sum(number(row.get("wait_s")) for row in rows)
    total_cmd_buffers = sum(number(row.get("metal_cmd_buffers")) for row in rows)

    return {
        "rows": len(rows),
        "elapsed_s_p50": seconds(median(elapsed_us)),
        "elapsed_s_p90": seconds(percentile(elapsed_us, 0.90)),
        "elapsed_s_total": seconds(total_elapsed_us),
        "wait_s_total": total_wait_s,
        "demand_reads": int(total_demand_reads),
        "prefetch_reads": int(total_prefetch_reads),
        "evictions": int(sum(number(row.get("evictions")) for row in rows)),
        "hit_rate": ratio(total_hits, total_hits + total_misses),
        "graph_compute_s": seconds(total_graph_compute_us),
        "backend_sync_s": seconds(total_backend_sync_us),
        "ids_readback_s": seconds(total_ids_readback_us),
        "callback_s": seconds(total_callback_us),
        "metal_cmd_buffers": int(total_cmd_buffers),
        "gpu_busy_ratio": ratio(total_gpu_busy_us, total_elapsed_us),
        "gpu_timestamp_missing": int(sum(number(row.get("metal_gpu_timestamp_missing")) for row in rows)),
        "arena_used_gib_p50": median([number(row.get("arena_used_gib")) for row in rows]),
        "high_water_gib_max": max([number(row.get("high_water_gib")) for row in rows], default=0.0),
    }


def seconds(us: float | None) -> float | None:
    if us is None:
        return None
    return us / 1_000_000.0


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    phases = phase_rows(rows)
    return {
        "rows": len(rows),
        "phase_rows": sum(len(items) for items in phases.values()),
        "phases": {name: summarize_phase(items) for name, items in sorted(phases.items())},
    }


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown_table(summary: dict[str, Any]) -> str:
    phases = summary["phases"]
    lines = [
        "| Phase | Rows | Elapsed p50 | Wait total | GPU busy | Graph compute | Backend sync | IDs readback | Demand reads | Hit rate | Cmd buffers |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if not phases:
        lines.append("| no phase counters found | 0 | - | - | - | - | - | - | - | - | - |")
        return "\n".join(lines)
    for name, row in phases.items():
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    str(row["rows"]),
                    fmt(row["elapsed_s_p50"]) + "s",
                    fmt(row["wait_s_total"]) + "s",
                    fmt(100.0 * row["gpu_busy_ratio"], 1) + "%" if row["gpu_busy_ratio"] is not None else "-",
                    fmt(row["graph_compute_s"]) + "s",
                    fmt(row["backend_sync_s"]) + "s",
                    fmt(row["ids_readback_s"]) + "s",
                    str(row["demand_reads"]),
                    fmt(100.0 * row["hit_rate"], 1) + "%" if row["hit_rate"] is not None else "-",
                    str(row["metal_cmd_buffers"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="gptoss_soak results.jsonl")
    parser.add_argument("--json", type=Path, help="write JSON summary")
    parser.add_argument("--markdown", type=Path, help="write Markdown table")
    args = parser.parse_args()

    summary = summarize(read_jsonl(args.results))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    table = markdown_table(summary)
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(table + "\n", encoding="utf-8")
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
