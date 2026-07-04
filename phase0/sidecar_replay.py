#!/usr/bin/env python3
"""Replay a MoE route trace against a sidecar plan.

This tool measures the I/O shape implied by a trace before a real sidecar file
exists. On cache misses it reads the source GGUF slices listed in the plan.
"""

from __future__ import annotations

import argparse
import collections
import fcntl
import json
import os
import time
from pathlib import Path
from typing import Any


F_NOCACHE = 48  # macOS fcntl flag; harmlessly skipped if unsupported.


def set_nocache(fd: int) -> None:
    try:
        fcntl.fcntl(fd, F_NOCACHE, 1)
    except OSError:
        pass


def load_plan(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {(int(r["layer"]), int(r["expert"])): r for r in data["records"]}


def iter_trace(path: Path, max_tokens: int | None):
    seen_tokens: set[int] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            token = int(row["token"])
            if max_tokens is not None and token not in seen_tokens and len(seen_tokens) >= max_tokens:
                break
            seen_tokens.add(token)
            yield row


class LRU:
    def __init__(self, capacity_bytes: int) -> None:
        self.capacity_bytes = capacity_bytes
        self.bytes = 0
        self.items: collections.OrderedDict[tuple[int, int], int] = collections.OrderedDict()

    def access(self, key: tuple[int, int], size: int) -> bool:
        if key in self.items:
            self.items.move_to_end(key)
            return True
        while self.bytes + size > self.capacity_bytes and self.items:
            _, old_size = self.items.popitem(last=False)
            self.bytes -= old_size
        self.items[key] = size
        self.bytes += size
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--budget-gib", type=float, default=8.0)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--packed", type=Path, help="read one contiguous record from this packed sidecar file on each miss")
    parser.add_argument("--no-read", action="store_true", help="simulate reads without touching disk")
    parser.add_argument("--nocache", action="store_true", help="request F_NOCACHE on source files")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    plan = load_plan(args.plan)
    cache = LRU(int(args.budget_gib * 1024**3))
    fds: dict[str, int] = {}
    packed_fd = None
    if args.packed:
        packed_fd = os.open(args.packed, os.O_RDONLY)
        if args.nocache:
            set_nocache(packed_fd)
    scratch = bytearray(2 * 1024 * 1024)

    tokens: set[int] = set()
    accesses = 0
    hits = 0
    misses = 0
    read_ops = 0
    read_bytes = 0
    missing_plan = 0
    start = time.perf_counter()

    try:
        for row in iter_trace(args.trace, args.max_tokens):
            token = int(row["token"])
            layer = int(row["layer"])
            tokens.add(token)
            for expert in row["experts"]:
                key = (layer, int(expert))
                record = plan.get(key)
                if record is None:
                    missing_plan += 1
                    continue
                accesses += 1
                if cache.access(key, int(record["bytes"])):
                    hits += 1
                    continue
                misses += 1
                if packed_fd is not None:
                    n = int(record["bytes"])
                    if not args.no_read:
                        if n > len(scratch):
                            scratch = bytearray(n)
                        got = os.pread(packed_fd, n, int(record["pack_offset"]))
                        if len(got) != n:
                            raise RuntimeError(f"short packed read for {key}: got {len(got)} expected {n}")
                    read_ops += 1
                    read_bytes += n
                else:
                    for sl in record["slices"]:
                        n = int(sl["bytes"])
                        path = str(sl["source_path"])
                        if path not in fds:
                            fd = os.open(path, os.O_RDONLY)
                            if args.nocache:
                                set_nocache(fd)
                            fds[path] = fd
                        if not args.no_read:
                            if n > len(scratch):
                                scratch = bytearray(n)
                            got = os.pread(fds[path], n, int(sl["source_offset"]))
                            if len(got) != n:
                                raise RuntimeError(f"short read for {key}: got {len(got)} expected {n}")
                        read_ops += 1
                        read_bytes += n
    finally:
        for fd in fds.values():
            os.close(fd)
        if packed_fd is not None:
            os.close(packed_fd)

    elapsed = time.perf_counter() - start
    result = {
        "plan": str(args.plan),
        "trace": str(args.trace),
        "budget_gib": args.budget_gib,
        "max_tokens": args.max_tokens,
        "tokens": len(tokens),
        "accesses": accesses,
        "hits": hits,
        "misses": misses,
        "hit_rate": hits / accesses if accesses else None,
        "missing_plan": missing_plan,
        "read_ops": read_ops,
        "read_bytes": read_bytes,
        "read_gib": read_bytes / 1024**3,
        "elapsed_s": elapsed,
        "read_gbps_decimal": (read_bytes / elapsed / 1e9) if elapsed > 0 else None,
        "tok_s_replay": (len(tokens) / elapsed) if elapsed > 0 else None,
        "no_read": args.no_read,
        "nocache": args.nocache,
        "packed": str(args.packed) if args.packed else None,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
