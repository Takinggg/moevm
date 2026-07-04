#!/usr/bin/env python3
"""Replay MoE router traces through cache policies.

Accepted trace formats:
- JSONL: {"token": 0, "layer": 0, "experts": [1, 2, ...]}
- CSV/TSV: token,layer,expert1,expert2,...

Each expert key is `(layer, expert_id)`.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import OrderedDict, defaultdict, deque
from pathlib import Path
from typing import Iterable


Record = tuple[int, int, tuple[int, ...]]
Key = tuple[int, int]


def parse_trace(path: Path) -> list[Record]:
    records: list[Record] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("{"):
                obj = json.loads(line)
                token = int(obj.get("token", obj.get("token_idx", 0)))
                layer = int(obj.get("layer", obj.get("layer_idx", 0)))
                experts = tuple(int(x) for x in obj.get("experts", obj.get("expert_ids", [])))
            else:
                parts = [x for x in line.replace("\t", ",").split(",") if x != ""]
                if len(parts) < 3:
                    raise ValueError(f"{path}:{line_no}: expected token,layer,experts")
                token = int(parts[0])
                layer = int(parts[1])
                experts = tuple(int(x) for x in parts[2:])
            records.append((token, layer, experts))
    return records


class Policy:
    def access(self, key: Key) -> bool:
        raise NotImplementedError


class LRU(Policy):
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.od: OrderedDict[Key, None] = OrderedDict()

    def access(self, key: Key) -> bool:
        hit = key in self.od
        if hit:
            self.od.move_to_end(key)
        else:
            self.od[key] = None
            if len(self.od) > self.capacity:
                self.od.popitem(last=False)
        return hit


class LFUDecay(Policy):
    def __init__(self, capacity: int, decay_every: int = 10000, sample: int = 8) -> None:
        self.capacity = capacity
        self.decay_every = decay_every
        self.sample = sample
        self.clock = 0
        self.items: dict[Key, float] = {}
        self.rng = random.Random(1)

    def access(self, key: Key) -> bool:
        self.clock += 1
        if self.decay_every and self.clock % self.decay_every == 0:
            for k in list(self.items):
                self.items[k] *= 0.5
                if self.items[k] < 0.1:
                    del self.items[k]
        hit = key in self.items
        self.items[key] = self.items.get(key, 0.0) + 1.0
        if len(self.items) > self.capacity:
            keys = list(self.items.keys())
            candidates = self.rng.sample(keys, min(self.sample, len(keys)))
            victim = min(candidates, key=lambda k: self.items[k])
            if victim != key or len(self.items) > self.capacity:
                self.items.pop(victim, None)
        return hit


class TwoQ(Policy):
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.a1_cap = max(1, capacity // 4)
        self.am_cap = max(1, capacity - self.a1_cap)
        self.a1: OrderedDict[Key, None] = OrderedDict()
        self.am: OrderedDict[Key, None] = OrderedDict()
        self.ghost: OrderedDict[Key, None] = OrderedDict()
        self.ghost_cap = capacity

    def access(self, key: Key) -> bool:
        if key in self.am:
            self.am.move_to_end(key)
            return True
        if key in self.a1:
            self.a1.pop(key)
            self.am[key] = None
            self._trim()
            return True
        if key in self.ghost:
            self.ghost.pop(key)
            self.am[key] = None
        else:
            self.a1[key] = None
        self._trim()
        return False

    def _trim(self) -> None:
        while len(self.a1) > self.a1_cap:
            k, _ = self.a1.popitem(last=False)
            self.ghost[k] = None
            while len(self.ghost) > self.ghost_cap:
                self.ghost.popitem(last=False)
        while len(self.am) > self.am_cap:
            k, _ = self.am.popitem(last=False)
            self.ghost[k] = None
            while len(self.ghost) > self.ghost_cap:
                self.ghost.popitem(last=False)


class S3FIFO(Policy):
    """Small/main FIFO approximation with second-touch admission."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.small_cap = max(1, capacity // 10)
        self.main_cap = max(1, capacity - self.small_cap)
        self.small: deque[Key] = deque()
        self.main: deque[Key] = deque()
        self.in_small: set[Key] = set()
        self.in_main: set[Key] = set()
        self.ghost: set[Key] = set()
        self.ghost_q: deque[Key] = deque()
        self.ghost_cap = capacity

    def access(self, key: Key) -> bool:
        if key in self.in_main or key in self.in_small:
            if key in self.in_small:
                self.in_small.remove(key)
                self.in_main.add(key)
                self.main.append(key)
            return True
        if key in self.ghost:
            self.in_main.add(key)
            self.main.append(key)
        else:
            self.in_small.add(key)
            self.small.append(key)
        self._trim()
        return False

    def _trim(self) -> None:
        while len(self.in_small) > self.small_cap:
            k = self.small.popleft()
            if k not in self.in_small:
                continue
            self.in_small.remove(k)
            self.ghost.add(k)
            self.ghost_q.append(k)
        while len(self.in_main) > self.main_cap:
            k = self.main.popleft()
            if k in self.in_main:
                self.in_main.remove(k)
        while len(self.ghost_q) > self.ghost_cap:
            old = self.ghost_q.popleft()
            self.ghost.discard(old)


def iter_keys(records: Iterable[Record]) -> Iterable[tuple[int, int, Key]]:
    for token, layer, experts in records:
        for expert in experts:
            yield token, layer, (layer, expert)


def token_in_stats_window(token: int, token_min: int | None, token_max: int | None) -> bool:
    if token_min is not None and token < token_min:
        return False
    if token_max is not None and token > token_max:
        return False
    return True


def flatten_accesses(records: list[Record]) -> list[tuple[int, int, Key]]:
    return list(iter_keys(records))


def simulate_belady(
    records: list[Record],
    capacity: int,
    stats_token_min: int | None = None,
    stats_token_max: int | None = None,
) -> dict:
    accesses = flatten_accesses(records)
    positions: dict[Key, deque[int]] = defaultdict(deque)
    for pos, (_, _, key) in enumerate(accesses):
        positions[key].append(pos)

    cache: set[Key] = set()
    next_use: dict[Key, float] = {}
    hits = misses = 0
    layer_hits: dict[int, int] = defaultdict(int)
    layer_total: dict[int, int] = defaultdict(int)
    inf = float("inf")

    for pos, (token, layer, key) in enumerate(accesses):
        queue = positions[key]
        if queue and queue[0] == pos:
            queue.popleft()
        key_next = float(queue[0]) if queue else inf

        hit = key in cache
        if hit:
            next_use[key] = key_next
        else:
            if len(cache) >= capacity and cache:
                victim = max(cache, key=lambda k: next_use.get(k, inf))
                cache.remove(victim)
                next_use.pop(victim, None)
            cache.add(key)
            next_use[key] = key_next

        if not token_in_stats_window(token, stats_token_min, stats_token_max):
            continue
        if hit:
            hits += 1
            layer_hits[layer] += 1
        else:
            misses += 1
        layer_total[layer] += 1

    total = hits + misses
    worst_layers = sorted(
        (
            {
                "layer": layer,
                "hit_rate": layer_hits[layer] / count if count else 0.0,
                "accesses": count,
            }
            for layer, count in layer_total.items()
        ),
        key=lambda x: x["hit_rate"],
    )[:8]
    return {
        "policy": "belady",
        "capacity_entries": capacity,
        "stats_token_min": stats_token_min,
        "stats_token_max": stats_token_max,
        "hits": hits,
        "misses": misses,
        "hit_rate": hits / total if total else 0.0,
        "worst_layers": worst_layers,
    }


def simulate(
    records: list[Record],
    policy_name: str,
    capacity: int,
    stats_token_min: int | None = None,
    stats_token_max: int | None = None,
) -> dict:
    if policy_name == "belady":
        return simulate_belady(records, capacity, stats_token_min, stats_token_max)
    if policy_name == "lru":
        policy: Policy = LRU(capacity)
    elif policy_name == "2q":
        policy = TwoQ(capacity)
    elif policy_name == "s3fifo":
        policy = S3FIFO(capacity)
    elif policy_name == "lfu-decay":
        policy = LFUDecay(capacity)
    else:
        raise ValueError(policy_name)

    hits = misses = 0
    layer_hits: dict[int, int] = defaultdict(int)
    layer_total: dict[int, int] = defaultdict(int)
    for token, layer, key in iter_keys(records):
        hit = policy.access(key)
        if not token_in_stats_window(token, stats_token_min, stats_token_max):
            continue
        if hit:
            hits += 1
            layer_hits[layer] += 1
        else:
            misses += 1
        layer_total[layer] += 1
    total = hits + misses
    worst_layers = sorted(
        (
            {
                "layer": layer,
                "hit_rate": layer_hits[layer] / count if count else 0.0,
                "accesses": count,
            }
            for layer, count in layer_total.items()
        ),
        key=lambda x: x["hit_rate"],
    )[:8]
    return {
        "policy": policy_name,
        "capacity_entries": capacity,
        "stats_token_min": stats_token_min,
        "stats_token_max": stats_token_max,
        "hits": hits,
        "misses": misses,
        "hit_rate": hits / total if total else 0.0,
        "worst_layers": worst_layers,
    }


def working_set(records: list[Record], windows: list[int], slice_mb: float) -> list[dict]:
    by_token: dict[int, set[Key]] = defaultdict(set)
    for token, layer, experts in records:
        for expert in experts:
            by_token[token].add((layer, expert))
    tokens = sorted(by_token)
    out = []
    for w in windows:
        max_keys = 0
        avg_keys = 0.0
        n = 0
        for i in range(0, len(tokens), max(1, w // 4)):
            chunk = tokens[i : i + w]
            if not chunk:
                continue
            keys: set[Key] = set()
            for tok in chunk:
                keys.update(by_token[tok])
            max_keys = max(max_keys, len(keys))
            avg_keys += len(keys)
            n += 1
        avg_keys = avg_keys / n if n else 0
        out.append(
            {
                "window_tokens": w,
                "avg_unique_keys": avg_keys,
                "max_unique_keys": max_keys,
                "avg_gib": avg_keys * slice_mb / 1024,
                "max_gib": max_keys * slice_mb / 1024,
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", type=Path)
    ap.add_argument("--slice-mb", type=float, required=True)
    ap.add_argument("--budgets", default="20,24,28")
    ap.add_argument("--policies", default="lru,2q,s3fifo,lfu-decay")
    ap.add_argument("--bssd-gbps", type=float, default=5.0)
    ap.add_argument("--compute-ms", type=float, default=14.0)
    ap.add_argument("--stats-token-min", type=int)
    ap.add_argument("--stats-token-max", type=int)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    records = parse_trace(args.trace)
    stats_records = [
        record
        for record in records
        if token_in_stats_window(record[0], args.stats_token_min, args.stats_token_max)
    ]
    budgets = [float(x) for x in args.budgets.split(",") if x.strip()]
    policies = [x.strip().lower() for x in args.policies.split(",") if x.strip()]
    total_accesses = sum(len(r[2]) for r in records)
    stats_accesses = sum(len(r[2]) for r in stats_records)
    stats_unique_tokens = len(set(r[0] for r in stats_records))

    results = []
    for budget in budgets:
        cap = max(1, int((budget * 1024) / args.slice_mb))
        for pol in policies:
            row = simulate(records, pol, cap, args.stats_token_min, args.stats_token_max)
            miss_mb_per_token = (row["misses"] * args.slice_mb) / max(1, stats_unique_tokens)
            io_s = (miss_mb_per_token / 1000) / args.bssd_gbps
            compute_s = args.compute_ms / 1000
            row.update(
                {
                    "cache_gib": budget,
                    "miss_mb_per_token": miss_mb_per_token,
                    "sync_tok_s_est": 1 / (compute_s + io_s),
                    "overlap_tok_s_est": 1 / max(compute_s, io_s),
                }
            )
            results.append(row)

    out = {
        "trace": str(args.trace),
        "records": len(records),
        "unique_tokens": len(set(r[0] for r in records)),
        "total_expert_accesses": total_accesses,
        "stats_records": len(stats_records),
        "stats_unique_tokens": stats_unique_tokens,
        "stats_expert_accesses": stats_accesses,
        "stats_token_min": args.stats_token_min,
        "stats_token_max": args.stats_token_max,
        "slice_mb": args.slice_mb,
        "working_set": working_set(stats_records, [512, 2048, 8192], args.slice_mb),
        "results": results,
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"records: {out['records']}")
    print(f"unique_tokens: {out['unique_tokens']}")
    print(f"expert_accesses: {out['total_expert_accesses']}")
    if args.stats_token_min is not None or args.stats_token_max is not None:
        print(
            f"stats_window: token_min={args.stats_token_min} token_max={args.stats_token_max} "
            f"tokens={stats_unique_tokens} expert_accesses={stats_accesses}"
        )
    print("\nWorking set:")
    for row in out["working_set"]:
        print(
            f"  w={row['window_tokens']}: avg={row['avg_gib']:.2f} GiB, max={row['max_gib']:.2f} GiB"
        )
    print("\nCache policies:")
    for row in results:
        print(
            f"  {row['cache_gib']:>5.1f} GiB {row['policy']:<9} "
            f"h={row['hit_rate']:.3f} miss={row['miss_mb_per_token']:.1f} MB/tok "
            f"sync={row['sync_tok_s_est']:.2f} overlap={row['overlap_tok_s_est']:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
