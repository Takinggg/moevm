#!/usr/bin/env python3
"""Pack a MoEVM sidecar payload from a sidecar plan."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import time
from pathlib import Path


F_NOCACHE = 48


def set_nocache(fd: int) -> None:
    try:
        fcntl.fcntl(fd, F_NOCACHE, 1)
    except OSError:
        pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--limit-records", type=int)
    parser.add_argument("--nocache", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1024)
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    records = plan["records"]
    if args.limit_records is not None:
        records = records[: args.limit_records]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    source_fds: dict[str, int] = {}
    written = 0
    start = time.perf_counter()
    hasher = hashlib.sha256()

    try:
        with args.out.open("wb") as out:
            for idx, record in enumerate(records, start=1):
                target = int(record["pack_offset"])
                if written < target:
                    out.write(b"\0" * (target - written))
                    written = target
                elif written > target:
                    raise RuntimeError(f"output offset {written} passed record target {target}")

                record_written = 0
                for sl in record["slices"]:
                    path = str(sl["source_path"])
                    if path not in source_fds:
                        fd = os.open(path, os.O_RDONLY)
                        if args.nocache:
                            set_nocache(fd)
                        source_fds[path] = fd
                    n = int(sl["bytes"])
                    data = os.pread(source_fds[path], n, int(sl["source_offset"]))
                    if len(data) != n:
                        raise RuntimeError(f"short read for {record['layer']}/{record['expert']}: got {len(data)} expected {n}")
                    out.write(data)
                    hasher.update(data)
                    record_written += n
                if record_written != int(record["bytes"]):
                    raise RuntimeError(f"record bytes mismatch: {record_written} vs {record['bytes']}")
                written += record_written

                if args.progress_every and idx % args.progress_every == 0:
                    elapsed = time.perf_counter() - start
                    gib = written / 1024**3
                    print(f"packed {idx}/{len(records)} records, {gib:.2f} GiB, {gib / elapsed:.2f} GiB/s", flush=True)
    finally:
        for fd in source_fds.values():
            os.close(fd)

    elapsed = time.perf_counter() - start
    result = {
        "plan": str(args.plan),
        "out": str(args.out),
        "records": len(records),
        "bytes": written,
        "sha256": hasher.hexdigest(),
        "gib": written / 1024**3,
        "elapsed_s": elapsed,
        "write_gib_s": (written / 1024**3 / elapsed) if elapsed > 0 else None,
        "limited": args.limit_records is not None,
        "nocache": args.nocache,
    }
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
