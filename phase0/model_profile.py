#!/usr/bin/env python3
"""Build a capability profile for a GGUF model before enabling MoEVM modules.

The profile is intentionally conservative: it reports what the current MoEVM
tooling can support, what requires a runtime parity gate, and what is blocked by
the model's tensor layout. This keeps gpt-oss/Qwen benchmark assumptions from
silently becoming "all MoE models" assumptions.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from gguf_inspect import TensorInfo, human_bytes, metadata_get_int, parse_ggufs


SPLIT_ROUTED_WEIGHT_RE = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")
FUSED_GATE_UP_WEIGHT_RE = re.compile(r"^blk\.(\d+)\.ffn_(?:gate_up|up_gate)_exps\.weight$")
FUSED_DOWN_WEIGHT_RE = re.compile(r"^blk\.(\d+)\.ffn_down_exps\.weight$")
EXPERT_BIAS_RE = re.compile(r"^blk\.(\d+)\.ffn_.*(?:_exps|_experts)\.bias$")
REQUIRED_SPLIT_COMPONENTS = ("gate", "up", "down")


def sample(values: list[str], limit: int = 12) -> list[str]:
    return values[:limit]


def capability(status: str, reason: str, gates: list[str] | None = None) -> dict[str, Any]:
    return {
        "status": status,
        "reason": reason,
        "gates": gates or [],
    }


def selected_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        k: v
        for k, v in meta.items()
        if any(s in k.lower() for s in ["architecture", "block_count", "expert", "attention", "context"])
    }


def routed_weight_layout(tensors: list[TensorInfo]) -> dict[str, Any]:
    split: dict[int, set[str]] = {}
    fused_gate_up_layers: set[int] = set()
    fused_down_layers: set[int] = set()
    unknown_weights: list[str] = []
    expert_biases: list[str] = []
    routed_weights: list[TensorInfo] = []

    for tensor in tensors:
        name = tensor.name
        if tensor.category != "routed_expert":
            continue
        if name.endswith(".bias") and EXPERT_BIAS_RE.match(name):
            expert_biases.append(name)
            continue
        if not name.endswith(".weight"):
            continue

        routed_weights.append(tensor)
        m = SPLIT_ROUTED_WEIGHT_RE.match(name)
        if m:
            layer = int(m.group(1))
            split.setdefault(layer, set()).add(m.group(2))
            continue

        m = FUSED_GATE_UP_WEIGHT_RE.match(name)
        if m:
            fused_gate_up_layers.add(int(m.group(1)))
            continue

        m = FUSED_DOWN_WEIGHT_RE.match(name)
        if m:
            fused_down_layers.add(int(m.group(1)))
            continue

        unknown_weights.append(name)

    split_layers = sorted(split)
    complete_split_layers = [
        layer
        for layer in split_layers
        if all(component in split[layer] for component in REQUIRED_SPLIT_COMPONENTS)
    ]
    missing_split = [
        {"layer": layer, "missing": [component for component in REQUIRED_SPLIT_COMPONENTS if component not in split[layer]]}
        for layer in split_layers
        if layer not in complete_split_layers
    ]

    fused_layers = sorted(fused_gate_up_layers | fused_down_layers)
    fused_complete_layers = sorted(fused_gate_up_layers & fused_down_layers)

    return {
        "routed_weight_count": len(routed_weights),
        "split_gate_up_down": {
            "layer_count": len(split_layers),
            "complete_layer_count": len(complete_split_layers),
            "layers": split_layers,
            "complete_layers": complete_split_layers,
            "missing": missing_split,
        },
        "fused_gate_up_down": {
            "layer_count": len(fused_layers),
            "complete_layer_count": len(fused_complete_layers),
            "layers": fused_layers,
            "complete_layers": fused_complete_layers,
            "note": "Detected only. Current sidecar/component-index v0 does not pack fused gate_up records.",
        },
        "unknown_routed_weights": {
            "count": len(unknown_weights),
            "examples": sample(sorted(unknown_weights)),
        },
        "expert_bias_tensors": {
            "count": len(expert_biases),
            "examples": sample(sorted(expert_biases)),
            "rule": "Bias ADD_ID nodes must keep their original ids; graph-pool rewrites only routed MUL_MAT_ID weight nodes.",
        },
    }


def infer_dense_zero_based(layers: list[int]) -> bool:
    return layers == list(range(len(layers)))


def build_profile(paths: list[Path], pool_capacity: int) -> dict[str, Any]:
    meta, tensors, header = parse_ggufs(paths)
    arch = str(meta.get("general.architecture", "unknown"))
    layer_count = metadata_get_int(meta, ["block_count"]) or (
        max((tensor.layer for tensor in tensors if tensor.layer is not None), default=-1) + 1
    )
    expert_count = metadata_get_int(meta, ["expert_count", "n_expert"])
    expert_used_count = metadata_get_int(meta, ["expert_used_count", "n_expert_used"])
    layout = routed_weight_layout(tensors)
    expected_split_count = meta.get("split.count")
    expected_tensor_count = meta.get("split.tensors.count")
    input_incomplete = (
        isinstance(expected_split_count, int)
        and expected_split_count > 1
        and int(header.get("shard_count", len(paths))) < expected_split_count
    )

    split_info = layout["split_gate_up_down"]
    split_layers = [int(x) for x in split_info["layers"]]
    complete_split_layers = [int(x) for x in split_info["complete_layers"]]
    fused_count = int(layout["fused_gate_up_down"]["layer_count"])
    unknown_count = int(layout["unknown_routed_weights"]["count"])
    routed_weight_count = int(layout["routed_weight_count"])
    dense_zero_based = infer_dense_zero_based(split_layers)
    split_v0_supported = (
        not input_incomplete
        and
        routed_weight_count > 0
        and split_layers == complete_split_layers
        and fused_count == 0
        and unknown_count == 0
    )

    if input_incomplete:
        sidecar_status = capability(
            "blocked",
            f"Input is incomplete: GGUF metadata declares {expected_split_count} splits but {header.get('shard_count', len(paths))} were provided.",
        )
    elif routed_weight_count == 0:
        sidecar_status = capability("blocked", "No routed expert weight tensors were detected. MoEVM only applies to routed MoE weights.")
    elif split_v0_supported:
        sidecar_status = capability("supported", "All routed expert weights use the split gate/up/down layout supported by sidecar_plan.py.")
    else:
        sidecar_status = capability(
            "blocked",
            "Current sidecar_plan.py supports only split gate/up/down expert weights. Add a layout adapter before packing this model.",
        )

    if not split_v0_supported:
        cindex_status = capability("blocked", "Component index v0 depends on supported split gate/up/down records.")
    elif not dense_zero_based:
        cindex_status = capability("blocked", "Component index v0 requires dense zero-based routed layer ids.")
    else:
        cindex_status = capability("supported", "Routed layers are dense zero-based and each record has gate/up/down components.")

    if cindex_status["status"] == "supported":
        graph_pool_status = capability(
            "runtime_check",
            "Graph-pool can patch MUL_MAT_ID weight nodes for this layout, but each model still needs same-backend logits parity.",
            [
                "Generate sidecar + component index from this exact GGUF set.",
                "Run Metal op-offload baseline vs MoEVM graph-pool; require logits max_abs ~= 0.",
                "Verify ADD_ID/bias nodes are not rewritten if expert bias tensors exist.",
            ],
        )
    else:
        graph_pool_status = capability("blocked", "Graph-pool v0 relies on the same split component index as the sidecar runtime.")

    if graph_pool_status["status"] == "runtime_check":
        cb_eval_status = capability(
            "experimental",
            "cb_eval readback removal is a graph-pool R&D path. It must re-run jitter/fence/logits gates per model.",
            ["Parity vs scheduler graph-pool path.", "Stress run with injected IO delays.", "No runtime_errors and no demand_reads regression."],
        )
    else:
        cb_eval_status = capability("blocked", "Requires graph-pool support first.")

    if expert_used_count and expert_used_count > 0:
        worst_case_ubatch = max(1, pool_capacity // expert_used_count)
        batch_reason = (
            f"With pool_capacity={pool_capacity} and top-k={expert_used_count}, a worst-case ubatch above "
            f"{worst_case_ubatch} can overflow the pool before expert locality/dedup helps."
        )
        batch_status = capability(
            "experimental",
            batch_reason,
            [
                "Compare logits against b=1 ub=1 for this model/backend.",
                "Treat token-match without logits parity as insufficient.",
                "Reject batch settings that trip fallback_k_overflow or safe_run memory gates.",
            ],
        )
    else:
        batch_status = capability("blocked", "Could not determine expert_used_count/top-k from GGUF metadata.")

    transport_status = capability(
        "supported",
        "Sidecar cache/direct-mmap/copy modes are transport-level knobs, but they are benchmark-gated per model and storage layout.",
        ["Run one-shot parity and tok/s A/B before making a transport mode a default for this model."],
    )

    return {
        "format": "moevm-model-profile-v0",
        "inputs": [str(path) for path in paths],
        "header": header,
        "architecture": arch,
        "metadata_selected": selected_metadata(meta),
        "layer_count": layer_count,
        "expert_count": expert_count,
        "expert_used_count": expert_used_count,
        "pool_capacity_assumed": pool_capacity,
        "input_completeness": {
            "complete": not input_incomplete,
            "provided_shards": int(header.get("shard_count", len(paths))),
            "expected_split_count": expected_split_count if isinstance(expected_split_count, int) else None,
            "expected_tensor_count": expected_tensor_count if isinstance(expected_tensor_count, int) else None,
            "parsed_tensor_count": int(header.get("tensor_count", len(tensors))),
        },
        "tensor_layout": layout,
        "capabilities": {
            "sidecar_plan_split_v0": sidecar_status,
            "component_index_v0": cindex_status,
            "graph_pool_v0": graph_pool_status,
            "cb_eval_readback_rnd": cb_eval_status,
            "batch_ubatch_fast_prefill": batch_status,
            "sidecar_transport_modes": transport_status,
        },
        "required_model_gates": [
            "Profile the exact GGUF shard set that will be packed and run; partial headers are only for early sizing.",
            "Build a sidecar and component index only when the profile says the layout is supported.",
            "Use same-backend logits parity for every new model/profile before publishing tok/s.",
            "If expert_bias_tensors.count > 0, verify graph patches never rewrite ADD_ID bias ids.",
            "Run safe_run memory gates before increasing arena, batch, ubatch, or pool capacity.",
        ],
        "note": "A supported profile means the current modules may run after gates pass; it is not a quality or speed claim.",
    }


def markdown(profile: dict[str, Any]) -> str:
    caps = profile["capabilities"]
    layout = profile["tensor_layout"]
    lines = [
        f"# MoEVM Model Profile: {profile['architecture']}",
        "",
        f"- Format: `{profile['format']}`",
        f"- Inputs: {len(profile['inputs'])}",
        f"- Layers: {profile['layer_count']}",
        f"- Experts: {profile['expert_count']}",
        f"- Experts active/token: {profile['expert_used_count']}",
        f"- Assumed pool capacity: {profile['pool_capacity_assumed']}",
        "",
        "## Tensor Layout",
        "",
        f"- Routed expert weight tensors: {layout['routed_weight_count']}",
        f"- Split gate/up/down layers: {layout['split_gate_up_down']['complete_layer_count']} complete / {layout['split_gate_up_down']['layer_count']} detected",
        f"- Fused gate_up/down layers: {layout['fused_gate_up_down']['complete_layer_count']} complete / {layout['fused_gate_up_down']['layer_count']} detected",
        f"- Unknown routed weight tensors: {layout['unknown_routed_weights']['count']}",
        f"- Expert bias tensors: {layout['expert_bias_tensors']['count']}",
        "",
        "## Capability Verdict",
        "",
    ]
    for name, item in caps.items():
        lines.append(f"- `{name}`: **{item['status']}** - {item['reason']}")
    lines.extend(["", "## Gates", ""])
    for gate in profile["required_model_gates"]:
        lines.append(f"- {gate}")
    if layout["unknown_routed_weights"]["examples"]:
        lines.extend(["", "## Unknown Routed Weight Examples", ""])
        for name in layout["unknown_routed_weights"]["examples"]:
            lines.append(f"- `{name}`")
    if layout["expert_bias_tensors"]["examples"]:
        lines.extend(["", "## Expert Bias Examples", ""])
        for name in layout["expert_bias_tensors"]["examples"]:
            lines.append(f"- `{name}`")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf", type=Path, nargs="+")
    parser.add_argument("--pool-capacity", type=int, default=16)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--md", type=Path)
    args = parser.parse_args()

    profile = build_profile(args.gguf, args.pool_capacity)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    if args.md:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text(markdown(profile), encoding="utf-8")

    print(json.dumps({
        "architecture": profile["architecture"],
        "sidecar_plan_split_v0": profile["capabilities"]["sidecar_plan_split_v0"]["status"],
        "component_index_v0": profile["capabilities"]["component_index_v0"]["status"],
        "graph_pool_v0": profile["capabilities"]["graph_pool_v0"]["status"],
        "expert_bias_tensors": profile["tensor_layout"]["expert_bias_tensors"]["count"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
