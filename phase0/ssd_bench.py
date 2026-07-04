#!/usr/bin/env python3
"""Random-read SSD benchmark for MoE expert paging.

Use this on the actual GGUF or sidecar file. It measures the kind of reads a
paged expert runtime would issue: many fixed-size random reads, optionally with
macOS F_NOCACHE to reduce page-cache distortion.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import random
import statistics
import threading
import time
from pathlib import Path

F_NOCACHE = 48  # Darwin fcntl constant.


def worker(path: Path, offsets: list[int], size: int, nocache: bool, stats: list[float], errors: list[str]) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        if nocache:
            try:
                fcntl.fcntl(fd, F_NOCACHE, 1)
            except OSError as exc:
                errors.append(f"F_NOCACHE failed: {exc}")
        for off in offsets:
            t0 = time.perf_counter()
            data = os.pread(fd, size, off)
            dt = time.perf_counter() - t0
            if len(data) != size:
                errors.append(f"short read at {off}: {len(data)} != {size}")
            stats.append(dt)
    finally:
        os.close(fd)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--size-mb", type=float, default=2.0)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--reads", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cached", action="store_true", help="do not request F_NOCACHE")
    args = ap.parse_args()

    file_size = args.path.stat().st_size
    read_size = int(args.size_mb * 1024 * 1024)
    if read_size <= 0 or read_size > file_size:
        raise SystemExit("invalid read size")

    rng = random.Random(args.seed)
    max_off = file_size - read_size
    # Align to 16KiB, common Apple Silicon page size.
    offsets = [(rng.randrange(0, max_off // 16384) * 16384) for _ in range(args.reads)]

    chunks = [offsets[i:: args.threads] for i in range(args.threads)]
    stats: list[float] = []
    errors: list[str] = []
    threads = [
        threading.Thread(target=worker, args=(args.path, chunk, read_size, not args.cached, stats, errors))
        for chunk in chunks
    ]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    total_s = time.perf_counter() - t0
    total_bytes = len(stats) * read_size
    mbps = total_bytes / total_s / 1e6
    lat_ms = [x * 1000 for x in stats]

    print(f"file: {args.path}")
    print(f"file_size_gb: {file_size / 1e9:.2f}")
    print(f"read_size_mb: {read_size / 1e6:.2f}")
    print(f"reads_completed: {len(stats)}")
    print(f"threads: {args.threads}")
    print(f"nocache_requested: {not args.cached}")
    print(f"throughput_mb_s: {mbps:.2f}")
    if lat_ms:
        print(f"latency_ms_p50: {statistics.median(lat_ms):.3f}")
        print(f"latency_ms_p95: {statistics.quantiles(lat_ms, n=100)[94]:.3f}")
        print(f"latency_ms_max: {max(lat_ms):.3f}")
    if errors:
        print("errors:")
        for e in errors[:10]:
            print(f"  {e}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())

