#!/usr/bin/env python3
"""Build a compact component index for MoEVM packed sidecars.

The main `.moevm.idx` maps `(layer, expert)` to a packed record. This companion
index maps the components inside that record: gate, up, and down.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path


MAGIC = b"MOEVMCIX"
HEADER_STRUCT = struct.Struct("<8sIIIIII3Q")
ENTRY_STRUCT = struct.Struct("<IIIIII")
KINDS = ("gate", "up", "down")

assert HEADER_STRUCT.size == 56
assert ENTRY_STRUCT.size == 24


def sidecar_integrity(path: Path) -> tuple[int, tuple[int, int]]:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(8 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    digest = h.digest()
    return path.stat().st_size, struct.unpack("<QQ", digest[:16])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, help="packed .moevm payload to bind by size + sha256 prefix")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    expert_count = int(plan["expert_count"])
    layers = [int(x) for x in plan["layers"]]
    if layers != list(range(len(layers))):
        raise SystemExit("component index v0 requires dense zero-based layers")
    layer_count = len(layers)
    records = plan["records"]
    expected = layer_count * expert_count
    if len(records) != expected:
        raise SystemExit(f"record count mismatch: {len(records)} vs {expected}")
    sidecar_size = 0
    sidecar_hash = (0, 0)
    if args.sidecar:
        sidecar_size, sidecar_hash = sidecar_integrity(args.sidecar)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as out:
        out.write(HEADER_STRUCT.pack(MAGIC, 1, expected, layer_count, expert_count, len(KINDS), 0, sidecar_size, *sidecar_hash))
        for position, record in enumerate(records):
            layer = int(record["layer"])
            expert = int(record["expert"])
            expected_index = layer * expert_count + expert
            if expected_index < 0 or expected_index >= expected:
                raise SystemExit(f"record key out of range: layer={layer} expert={expert}")
            if expected_index != position:
                raise SystemExit(f"record out of dense order at position {position}: layer={layer} expert={expert} expected_index={expected_index}")

            by_kind = {sl["kind"]: sl for sl in record["slices"]}
            values: list[int] = []
            for kind in KINDS:
                if kind not in by_kind:
                    raise SystemExit(f"missing {kind} slice for layer={layer} expert={expert}")
                rel = int(by_kind[kind]["record_relative_offset"])
                n = int(by_kind[kind]["bytes"])
                if rel < 0 or n < 0 or rel > 0xFFFFFFFF or n > 0xFFFFFFFF:
                    raise SystemExit(f"component does not fit uint32: layer={layer} expert={expert} kind={kind}")
                values.extend([rel, n])
            out.write(ENTRY_STRUCT.pack(*values))

    print(
        json.dumps(
            {
                "out": str(args.out),
                "entries": expected,
                "layers": layer_count,
                "experts": expert_count,
                "components": list(KINDS),
                "bytes": args.out.stat().st_size,
                "sidecar_size": sidecar_size if args.sidecar else None,
                "sidecar_sha256_prefix16": "".join(word.to_bytes(8, "little").hex() for word in sidecar_hash) if args.sidecar else None,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
