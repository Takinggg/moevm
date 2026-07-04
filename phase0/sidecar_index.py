#!/usr/bin/env python3
"""Build a compact binary index for a packed MoEVM sidecar.

The JSON sidecar plan is useful for inspection, but a runtime should not parse
that large file on every start. This script converts the plan into a dense
little-endian table indexed by:

    entry = layer * expert_count + expert

Each entry stores the packed sidecar offset and record byte size.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path


MAGIC = b"MOEVMIDX"
VERSION = 1
HEADER = struct.Struct("<8sIIIIQQQQ")
ENTRY = struct.Struct("<QII")


def sidecar_integrity(path: Path) -> tuple[int, tuple[int, int, int]]:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(8 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    digest = h.digest()
    return path.stat().st_size, struct.unpack("<QQQ", digest[:24])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, help="packed .moevm payload to bind by size + sha256 prefix")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    records = plan["records"]
    expert_count = int(plan["expert_count"])
    layer_count = max(int(r["layer"]) for r in records) + 1
    dense_count = layer_count * expert_count
    entries = [(0, 0)] * dense_count
    seen = [False] * dense_count

    for record in records:
        layer = int(record["layer"])
        expert = int(record["expert"])
        idx = layer * expert_count + expert
        if idx < 0 or idx >= dense_count:
            raise SystemExit(f"record key out of range: layer={layer} expert={expert}")
        if seen[idx]:
            raise SystemExit(f"duplicate record key: layer={layer} expert={expert}")
        seen[idx] = True
        entries[idx] = (int(record["pack_offset"]), int(record["bytes"]))

    missing = sum(1 for _, n in entries if n == 0)
    if missing:
        raise SystemExit(f"dense table has {missing} missing entries")

    sidecar_size = 0
    sidecar_hash = (0, 0, 0)
    if args.sidecar:
        sidecar_size, sidecar_hash = sidecar_integrity(args.sidecar)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as f:
        f.write(HEADER.pack(MAGIC, VERSION, dense_count, layer_count, expert_count, sidecar_size, *sidecar_hash))
        for offset, size in entries:
            f.write(ENTRY.pack(offset, size, 0))

    print(f"index: {args.out}")
    print(f"version: {VERSION}")
    print(f"layers: {layer_count}")
    print(f"experts: {expert_count}")
    print(f"entries: {dense_count}")
    print(f"bytes: {args.out.stat().st_size}")
    if args.sidecar:
        print(f"sidecar_size: {sidecar_size}")
        print(f"sidecar_sha256_prefix24: {''.join(word.to_bytes(8, 'little').hex() for word in sidecar_hash)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
