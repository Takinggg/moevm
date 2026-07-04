#!/usr/bin/env python3
"""Simulate oracle lookahead prefetch on a MoE route trace.

This is intentionally labeled as an oracle/upper-bound tool. A real decoder does
not know arbitrary future routes. The goal is to quantify how much demand I/O
could be converted into background I/O if the runtime learns useful prefetch
signals.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any


Key = tuple[int, int]


class LRU:
    def __init__(self, capacity_bytes: int) -> None:
        self.capacity_bytes = capacity_bytes
        self.bytes = 0
        self.items: collections.OrderedDict[Key, int] = collections.OrderedDict()

    def contains(self, key: Key) -> bool:
        return key in self.items

    def touch(self, key: Key) -> bool:
        if key not in self.items:
            return False
        self.items.move_to_end(key)
        return True

    def insert(self, key: Key, size: int) -> None:
        if key in self.items:
            self.items.move_to_end(key)
            return
        while self.bytes + size > self.capacity_bytes and self.items:
            _, old_size = self.items.popitem(last=False)
            self.bytes -= old_size
        self.items[key] = size
        self.bytes += size


def load_sizes(plan_path: Path) -> dict[Key, int]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    return {(int(r["layer"]), int(r["expert"])): int(r["bytes"]) for r in plan["records"]}


def load_accesses(trace_path: Path, sizes: dict[Key, int]) -> tuple[list[Key], list[int]]:
    keys: list[Key] = []
    tokens: list[int] = []
    with trace_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            token = int(row["token"])
            layer = int(row["layer"])
            for expert in row["experts"]:
                key = (layer, int(expert))
                if key not in sizes:
                    raise ValueError(f"missing key in plan: {key}")
                keys.append(key)
                tokens.append(token)
    return keys, tokens


def simulate(keys: list[Key], tokens: list[int], sizes: dict[Key, int], budget_gib: float, window: int) -> dict[str, Any]:
    cache = LRU(int(budget_gib * 1024**3))
    demand_hits = 0
    demand_misses = 0
    demand_bytes = 0
    prefetch_loads = 0
    prefetch_bytes = 0
    total_prefetch_attempts = 0

    for i, key in enumerate(keys):
        if cache.touch(key):
            demand_hits += 1
        else:
            demand_misses += 1
            size = sizes[key]
            demand_bytes += size
            cache.insert(key, size)

        if window > 0:
            end = min(len(keys), i + 1 + window)
            for future in keys[i + 1 : end]:
                total_prefetch_attempts += 1
                if cache.contains(future):
                    continue
                size = sizes[future]
                cache.insert(future, size)
                prefetch_loads += 1
                prefetch_bytes += size

    unique_tokens = len(set(tokens))
    total_load_bytes = demand_bytes + prefetch_bytes
    total = demand_hits + demand_misses
    return {
        "budget_gib": budget_gib,
        "prefetch_window_accesses": window,
        "tokens": unique_tokens,
        "accesses": total,
        "demand_hits": demand_hits,
        "demand_misses": demand_misses,
        "demand_hit_rate": demand_hits / total if total else None,
        "demand_read_bytes": demand_bytes,
        "demand_read_gib": demand_bytes / 1024**3,
        "demand_mb_per_token": demand_bytes / 1_000_000 / max(1, unique_tokens),
        "prefetch_loads": prefetch_loads,
        "prefetch_read_bytes": prefetch_bytes,
        "prefetch_read_gib": prefetch_bytes / 1024**3,
        "total_load_bytes": total_load_bytes,
        "total_load_gib": total_load_bytes / 1024**3,
        "total_load_mb_per_token": total_load_bytes / 1_000_000 / max(1, unique_tokens),
        "prefetch_attempts": total_prefetch_attempts,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--budget-gib", type=float, default=8.0)
    parser.add_argument("--windows", default="0,47,470")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    sizes = load_sizes(args.plan)
    keys, tokens = load_accesses(args.trace, sizes)
    windows = [int(x) for x in args.windows.split(",") if x.strip()]
    results = [simulate(keys, tokens, sizes, args.budget_gib, w) for w in windows]

    out = {
        "plan": str(args.plan),
        "trace": str(args.trace),
        "budget_gib": args.budget_gib,
        "results": results,
        "note": "Oracle lookahead upper bound. This is not a real decoder guarantee.",
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("window  hit    demand GiB  total GiB  demand MB/tok  total MB/tok")
    for row in results:
        print(
            f"{row['prefetch_window_accesses']:>6}  "
            f"{row['demand_hit_rate']:.3f}  "
            f"{row['demand_read_gib']:>10.2f}  "
            f"{row['total_load_gib']:>9.2f}  "
            f"{row['demand_mb_per_token']:>13.1f}  "
            f"{row['total_load_mb_per_token']:>12.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
