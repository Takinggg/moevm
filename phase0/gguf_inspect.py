#!/usr/bin/env python3
"""Inspect GGUF metadata and MoE tensor layout.

This parser is deliberately stdlib-only and tolerates truncated GGUF files as
long as the header, metadata, and tensor-info table are present. That makes it
usable with Hugging Face HTTP range downloads.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import re
import struct
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


GGUF_VALUE_TYPES = {
    0: "uint8",
    1: "int8",
    2: "uint16",
    3: "int16",
    4: "uint32",
    5: "int32",
    6: "float32",
    7: "bool",
    8: "string",
    9: "array",
    10: "uint64",
    11: "int64",
    12: "float64",
}

# ggml type id -> (name, block_size, type_size_bytes)
# Values are from ggml common quant layouts. Unknown future types are handled
# by offset deltas when possible, otherwise reported as unknown.
GGML_TYPES = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1),
    25: ("I16", 1, 2),
    26: ("I32", 1, 4),
    27: ("I64", 1, 8),
    28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2),
    31: ("TQ1_0", 256, 54),
    32: ("TQ2_0", 256, 66),
}


class Reader:
    def __init__(self, source: bytes | io.BufferedReader) -> None:
        self.data = source if isinstance(source, bytes) else None
        self.file = None if isinstance(source, bytes) else source
        self.off = 0

    def need(self, n: int) -> None:
        if self.data is not None and self.off + n > len(self.data):
            raise EOFError(f"truncated at byte {self.off}, need {n} more bytes")

    def read(self, n: int) -> bytes:
        if self.data is not None:
            self.need(n)
            out = self.data[self.off : self.off + n]
        else:
            assert self.file is not None
            out = self.file.read(n)
            if len(out) != n:
                raise EOFError(f"truncated at byte {self.off}, need {n} more bytes")
        self.off += len(out)
        return out

    def u32(self) -> int:
        return struct.unpack_from("<I", self.read(4))[0]

    def u64(self) -> int:
        return struct.unpack_from("<Q", self.read(8))[0]

    def i32(self) -> int:
        return struct.unpack_from("<i", self.read(4))[0]

    def i64(self) -> int:
        return struct.unpack_from("<q", self.read(8))[0]

    def f32(self) -> float:
        return struct.unpack_from("<f", self.read(4))[0]

    def f64(self) -> float:
        return struct.unpack_from("<d", self.read(8))[0]

    def bool(self) -> bool:
        return struct.unpack_from("<?", self.read(1))[0]

    def string(self) -> str:
        n = self.u64()
        raw = self.read(n)
        return raw.decode("utf-8", errors="replace")


@dataclass
class TensorInfo:
    name: str
    dims: list[int]
    ggml_type: int
    ggml_type_name: str
    offset: int
    n_elements: int
    n_bytes: int | None
    category: str
    layer: int | None
    source_path: str


def read_value(r: Reader, vtype: int) -> Any:
    if vtype == 0:
        return r.read(1)[0]
    if vtype == 1:
        return struct.unpack_from("<b", r.read(1))[0]
    if vtype == 2:
        return struct.unpack_from("<H", r.read(2))[0]
    if vtype == 3:
        return struct.unpack_from("<h", r.read(2))[0]
    if vtype == 4:
        return r.u32()
    if vtype == 5:
        return r.i32()
    if vtype == 6:
        return r.f32()
    if vtype == 7:
        return r.bool()
    if vtype == 8:
        return r.string()
    if vtype == 10:
        return r.u64()
    if vtype == 11:
        return r.i64()
    if vtype == 12:
        return r.f64()
    if vtype == 9:
        elem_type = r.u32()
        n = r.u64()
        return [read_value(r, elem_type) for _ in range(n)]
    raise ValueError(f"unsupported GGUF value type {vtype}")


def tensor_nbytes(n_elements: int, ggml_type: int) -> int | None:
    spec = GGML_TYPES.get(ggml_type)
    if not spec:
        return None
    _, block, size = spec
    return math.ceil(n_elements / block) * size


def align(x: int, alignment: int) -> int:
    return ((x + alignment - 1) // alignment) * alignment


def layer_from_name(name: str) -> int | None:
    m = re.search(r"(?:^|[.])blk[.](\d+)[.]", name)
    if m:
        return int(m.group(1))
    return None


def categorize(name: str) -> str:
    low = name.lower()
    if "ffn" in low and ("_exps" in low or ".exps" in low or "experts" in low):
        return "routed_expert"
    if "shared" in low or "shexp" in low:
        return "shared_expert"
    if "gate_inp" in low or "router" in low:
        return "router"
    if "attn" in low or "attention" in low or "deltanet" in low:
        return "attention"
    if "token_embd" in low or "embed" in low:
        return "embedding"
    if "output" in low or "lm_head" in low:
        return "output"
    return "other_resident"


def metadata_get_int(meta: dict[str, Any], suffixes: list[str]) -> int | None:
    for suffix in suffixes:
        for key, val in meta.items():
            if not key.endswith(suffix):
                continue
            if isinstance(val, bool):
                continue
            if isinstance(val, int):
                return val
            if isinstance(val, float) and val.is_integer():
                return int(val)
    return None


def parse_gguf(path: Path) -> tuple[dict[str, Any], list[TensorInfo], dict[str, Any]]:
    file_bytes_available = path.stat().st_size
    with path.open("rb") as f:
        r = Reader(f)
        magic = r.read(4)
        if magic != b"GGUF":
            raise ValueError(f"{path} is not GGUF, magic={magic!r}")
        version = r.u32()
        tensor_count = r.u64()
        metadata_count = r.u64()
        meta: dict[str, Any] = {}
        for _ in range(metadata_count):
            key = r.string()
            vtype = r.u32()
            meta[key] = read_value(r, vtype)

        tensors: list[TensorInfo] = []
        for _ in range(tensor_count):
            name = r.string()
            ndims = r.u32()
            dims = [r.u64() for _ in range(ndims)]
            ggml_type = r.u32()
            offset = r.u64()
            n_elements = math.prod(dims) if dims else 0
            type_name = GGML_TYPES.get(ggml_type, (f"TYPE_{ggml_type}", 1, 0))[0]
            tensors.append(
                TensorInfo(
                    name=name,
                    dims=dims,
                    ggml_type=ggml_type,
                    ggml_type_name=type_name,
                    offset=offset,
                    n_elements=n_elements,
                    n_bytes=tensor_nbytes(n_elements, ggml_type),
                    category=categorize(name),
                    layer=layer_from_name(name),
                    source_path=str(path),
                )
            )

    # Some newly-added GGML quant types may not be known by this script yet.
    # For header-only inspection we can still infer tensor payload bytes from
    # offset deltas. Offsets are relative to the GGUF data section.
    by_offset = sorted(tensors, key=lambda t: t.offset)
    for cur, nxt in zip(by_offset, by_offset[1:]):
        if cur.n_bytes is None and nxt.offset > cur.offset:
            cur.n_bytes = nxt.offset - cur.offset

    alignment = int(meta.get("general.alignment", 32))
    data_start = align(r.off, alignment)
    header = {
        "path": str(path),
        "file_bytes_available": file_bytes_available,
        "gguf_version": version,
        "tensor_count": tensor_count,
        "metadata_count": metadata_count,
        "alignment": alignment,
        "tensor_info_end": r.off,
        "data_start": data_start,
        "truncated_data_ok": True,
    }
    return meta, tensors, header


def parse_ggufs(paths: list[Path]) -> tuple[dict[str, Any], list[TensorInfo], dict[str, Any]]:
    metas: list[dict[str, Any]] = []
    tensors: list[TensorInfo] = []
    headers: list[dict[str, Any]] = []
    for path in paths:
        meta, shard_tensors, header = parse_gguf(path)
        metas.append(meta)
        tensors.extend(shard_tensors)
        headers.append(header)

    # Multi-shard GGUF stores full model metadata in the first shard and only
    # minimal split metadata in later shards. Prefer the first shard that
    # declares the architecture, falling back to the first shard.
    canonical_meta = metas[0] if metas else {}
    for meta in metas:
        if "general.architecture" in meta:
            canonical_meta = meta
            break

    header = {
        "paths": [str(path) for path in paths],
        "shard_count": len(paths),
        "file_bytes_available": sum(h["file_bytes_available"] for h in headers),
        "gguf_version": headers[0]["gguf_version"] if headers else None,
        "tensor_count": sum(h["tensor_count"] for h in headers),
        "metadata_count": sum(h["metadata_count"] for h in headers),
        "alignment": headers[0]["alignment"] if headers else None,
        "shards": headers,
        "truncated_data_ok": True,
    }
    return canonical_meta, tensors, header


def summarize(meta: dict[str, Any], tensors: list[TensorInfo], header: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    arch = str(meta.get("general.architecture", "unknown"))
    layer_count = metadata_get_int(meta, ["block_count"]) or (
        max((t.layer for t in tensors if t.layer is not None), default=-1) + 1
    )
    expert_count = metadata_get_int(meta, ["expert_count", "n_expert"])
    expert_used_count = metadata_get_int(meta, ["expert_used_count", "n_expert_used"])
    if expert_count is None:
        # Infer from routed expert tensor dims when one dimension is plausible.
        for t in tensors:
            if t.category == "routed_expert":
                candidates = [d for d in t.dims if 2 <= d <= 4096]
                if candidates:
                    expert_count = max(candidates)
                    break

    by_category: dict[str, int] = {}
    for t in tensors:
        by_category[t.category] = by_category.get(t.category, 0) + (t.n_bytes or 0)

    routed = [t for t in tensors if t.category == "routed_expert"]
    routed_total = sum(t.n_bytes or 0 for t in routed)
    resident_total = sum(t.n_bytes or 0 for t in tensors if t.category != "routed_expert")

    per_layer: dict[int, int] = {}
    per_layer_expert: dict[int, float] = {}
    if expert_count:
        for t in routed:
            if t.layer is None:
                continue
            per_layer[t.layer] = per_layer.get(t.layer, 0) + (t.n_bytes or 0)
        per_layer_expert = {k: v / expert_count for k, v in per_layer.items()}

    avg_slice = (sum(per_layer_expert.values()) / len(per_layer_expert)) if per_layer_expert else None
    min_slice = min(per_layer_expert.values()) if per_layer_expert else None
    max_slice = max(per_layer_expert.values()) if per_layer_expert else None

    active_bytes_per_token = None
    if avg_slice is not None and expert_used_count is not None and layer_count is not None:
        active_bytes_per_token = avg_slice * expert_used_count * layer_count

    # KV estimate. This is intentionally conservative and metadata-driven.
    ctx = args.ctx
    kv_bytes = None
    head_count_kv = metadata_get_int(meta, ["attention.head_count_kv", "attention.head_count"])
    key_len = metadata_get_int(meta, ["attention.key_length"])
    value_len = metadata_get_int(meta, ["attention.value_length"])
    full_interval = metadata_get_int(meta, ["attention.full_attention_interval", "full_attention_interval"])
    kv_layers = layer_count
    if full_interval and full_interval > 1 and layer_count:
        kv_layers = math.ceil(layer_count / full_interval)
    if head_count_kv and key_len and value_len and kv_layers:
        q8_overhead = 34 / 32
        kv_bytes = int(ctx * kv_layers * head_count_kv * (key_len + value_len) * q8_overhead)

    budgets = []
    if routed_total and active_bytes_per_token:
        for b in args.budgets:
            budget_bytes = int(b * 1024**3)
            h_uniform = min(1.0, budget_bytes / routed_total)
            miss_bytes = active_bytes_per_token * (1 - h_uniform)
            io_s = miss_bytes / (args.bssd_gbps * 1e9) if args.bssd_gbps > 0 else None
            compute_s = args.compute_ms / 1000
            sync_tps = 1 / (compute_s + io_s) if io_s is not None else None
            overlap_tps = 1 / max(compute_s, io_s) if io_s is not None else None
            budgets.append(
                {
                    "cache_gib": b,
                    "uniform_hit_floor": h_uniform,
                    "miss_mb_per_token": miss_bytes / 1e6,
                    "sync_tok_s_est": sync_tps,
                    "overlap_tok_s_est": overlap_tps,
                }
            )

    interesting_meta = {
        k: v
        for k, v in meta.items()
        if any(s in k.lower() for s in ["architecture", "block_count", "expert", "attention", "embedding", "context"])
    }

    return {
        "header": header,
        "architecture": arch,
        "metadata_selected": interesting_meta,
        "layer_count": layer_count,
        "expert_count": expert_count,
        "expert_used_count": expert_used_count,
        "tensor_bytes_by_category": by_category,
        "routed_expert_pool_bytes": routed_total,
        "resident_non_routed_bytes": resident_total,
        "slice_bytes_per_layer_expert_avg": avg_slice,
        "slice_bytes_per_layer_expert_min": min_slice,
        "slice_bytes_per_layer_expert_max": max_slice,
        "active_expert_bytes_per_token_all_miss": active_bytes_per_token,
        "kv_cache_q8_bytes_estimate": kv_bytes,
        "cache_budget_projection": budgets,
        "routed_tensor_count": len(routed),
    }


def human_bytes(n: float | int | None) -> str:
    if n is None:
        return "n/a"
    value = float(n)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if abs(value) < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} B"


def print_text(summary: dict[str, Any], tensors: list[TensorInfo], show_tensors: bool) -> None:
    print(f"Architecture: {summary['architecture']}")
    print(f"Layers: {summary['layer_count']}")
    print(f"Experts: {summary['expert_count']}")
    print(f"Experts active/token: {summary['expert_used_count']}")
    print(f"Routed expert pool: {human_bytes(summary['routed_expert_pool_bytes'])}")
    print(f"Resident non-routed estimate: {human_bytes(summary['resident_non_routed_bytes'])}")
    print(f"Avg bytes per (layer, expert): {human_bytes(summary['slice_bytes_per_layer_expert_avg'])}")
    print(f"Min/max bytes per (layer, expert): {human_bytes(summary['slice_bytes_per_layer_expert_min'])} / {human_bytes(summary['slice_bytes_per_layer_expert_max'])}")
    print(f"Active expert bytes/token if all miss: {human_bytes(summary['active_expert_bytes_per_token_all_miss'])}")
    print(f"KV q8 estimate: {human_bytes(summary['kv_cache_q8_bytes_estimate'])}")
    print("\nCache projections:")
    for row in summary["cache_budget_projection"]:
        print(
            f"  {row['cache_gib']:>5.1f} GiB: h_floor={row['uniform_hit_floor']:.3f}, "
            f"miss={row['miss_mb_per_token']:.1f} MB/tok, "
            f"sync={row['sync_tok_s_est']:.2f} tok/s, overlap={row['overlap_tok_s_est']:.2f} tok/s"
        )
    if show_tensors:
        print("\nTensors:")
        for t in tensors:
            print(
                f"{t.name}\tlayer={t.layer}\tcat={t.category}\ttype={t.ggml_type_name}\t"
                f"dims={t.dims}\toffset={t.offset}\tbytes={t.n_bytes}"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf", type=Path, nargs="+")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--tensors", action="store_true")
    parser.add_argument("--ctx", type=int, default=65536)
    parser.add_argument("--budgets", default="20,24,28")
    parser.add_argument("--bssd-gbps", type=float, default=5.0)
    parser.add_argument("--compute-ms", type=float, default=14.0)
    args = parser.parse_args()
    args.budgets = [float(x) for x in args.budgets.split(",") if x.strip()]

    meta, tensors, header = parse_ggufs(args.gguf)
    summary = summarize(meta, tensors, header, args)
    out = {"summary": summary, "tensors": [asdict(t) for t in tensors] if args.tensors else None}
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print_text(summary, tensors, args.tensors)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
