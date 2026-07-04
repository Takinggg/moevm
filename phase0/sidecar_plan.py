#!/usr/bin/env python3
"""Build a MoEVM sidecar packing plan from GGUF routed expert tensors.

The plan does not copy weights. It maps each `(layer, expert)` record to the
contiguous source slices that must be packed together later.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from gguf_inspect import TensorInfo, parse_ggufs


ROUTED_RE = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")


def align(x: int, n: int) -> int:
    return ((x + n - 1) // n) * n


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def classify_routed_tensor(t: TensorInfo) -> tuple[int, str] | None:
    m = ROUTED_RE.match(t.name)
    if not m:
        return None
    return int(m.group(1)), m.group(2)


def build_plan(paths: list[Path], align_bytes: int, include_hash: bool) -> dict[str, Any]:
    meta, tensors, header = parse_ggufs(paths)
    arch = str(meta.get("general.architecture", "unknown"))

    expert_count = None
    for key, value in meta.items():
        if key.endswith("expert_count") and isinstance(value, int):
            expert_count = value
            break
    if not expert_count:
        raise SystemExit("could not determine expert_count from GGUF metadata")

    data_starts: dict[str, int] = {}
    if "shards" in header:
        for shard in header["shards"]:
            data_starts[shard["path"]] = int(shard["data_start"])
    else:
        data_starts[header["path"]] = int(header["data_start"])

    routed: dict[tuple[int, str], TensorInfo] = {}
    for t in tensors:
        parsed = classify_routed_tensor(t)
        if parsed is None:
            continue
        routed[parsed] = t

    layers = sorted({layer for layer, _ in routed})
    required_kinds = ["gate", "up", "down"]
    missing = []
    for layer in layers:
        for kind in required_kinds:
            if (layer, kind) not in routed:
                missing.append({"layer": layer, "kind": kind})
    if missing:
        raise SystemExit(f"missing routed tensors: {missing[:8]}")

    records = []
    pack_offset = 0
    for layer in layers:
        tensors_for_layer = [routed[(layer, kind)] for kind in required_kinds]
        for expert in range(expert_count):
            pack_offset = align(pack_offset, align_bytes)
            record_offset = pack_offset
            slices = []
            record_bytes = 0
            for kind, tensor in zip(required_kinds, tensors_for_layer):
                if tensor.n_bytes is None:
                    raise SystemExit(f"unknown tensor bytes for {tensor.name}")
                if tensor.n_bytes % expert_count != 0:
                    raise SystemExit(f"tensor {tensor.name} bytes {tensor.n_bytes} not divisible by experts {expert_count}")
                slice_bytes = tensor.n_bytes // expert_count
                source_path = header["paths"][0] if "paths" in header else header["path"]
                # Multi-shard support: full tensor payloads in GGUF split files
                # keep their tensor-info entry in the shard that contains them.
                # TensorInfo currently does not carry path, so for one-file Qwen3
                # this is exact. For multi-shard sidecars, extend TensorInfo with
                # source path before using this script to pack bytes.
                source_data_start = data_starts[source_path]
                source_offset = source_data_start + tensor.offset + expert * slice_bytes
                slices.append(
                    {
                        "kind": kind,
                        "tensor": tensor.name,
                        "ggml_type": tensor.ggml_type_name,
                        "dims": tensor.dims,
                        "source_path": source_path,
                        "source_offset": source_offset,
                        "bytes": slice_bytes,
                        "record_relative_offset": record_bytes,
                    }
                )
                record_bytes += slice_bytes
            records.append(
                {
                    "layer": layer,
                    "expert": expert,
                    "pack_offset": record_offset,
                    "bytes": record_bytes,
                    "slices": slices,
                }
            )
            pack_offset = record_offset + record_bytes

    files = []
    for path in paths:
        item = {"path": str(path), "bytes": path.stat().st_size}
        if include_hash:
            item["sha256"] = file_sha256(path)
        files.append(item)

    return {
        "format": "moevm-sidecar-plan-v0",
        "architecture": arch,
        "files": files,
        "expert_count": expert_count,
        "layer_count_with_routed_tensors": len(layers),
        "layers": layers,
        "record_count": len(records),
        "record_bytes_min": min(r["bytes"] for r in records) if records else 0,
        "record_bytes_max": max(r["bytes"] for r in records) if records else 0,
        "record_bytes_avg": sum(r["bytes"] for r in records) / len(records) if records else 0,
        "pack_alignment": align_bytes,
        "planned_payload_bytes": pack_offset,
        "source_summary": {
            "header": header,
            "metadata_selected": {
                k: v
                for k, v in meta.items()
                if any(s in k.lower() for s in ["architecture", "block_count", "expert", "context"])
            },
        },
        "records": records,
        "note": "This is a plan only. It has not copied or validated sidecar payload bytes.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--align", type=int, default=16 * 1024)
    parser.add_argument("--hash", action="store_true", help="include SHA256 of source GGUF files")
    args = parser.parse_args()

    plan = build_plan(args.gguf, args.align, args.hash)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    print(f"architecture: {plan['architecture']}")
    print(f"records: {plan['record_count']}")
    print(f"layers: {plan['layer_count_with_routed_tensors']}")
    print(f"record bytes avg: {plan['record_bytes_avg']:.0f}")
    print(f"planned payload: {plan['planned_payload_bytes'] / 1024**3:.2f} GiB")
    print(f"out: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
