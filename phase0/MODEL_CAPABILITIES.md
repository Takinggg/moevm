# MoEVM Model Capabilities

MoEVM is not allowed to assume that a model is compatible because Qwen or
gpt-oss worked. Every model must pass a capability profile first.

## Current Production Layout

Supported today:

- routed MoE tensors named `blk.{layer}.ffn_gate_exps.weight`
- routed MoE tensors named `blk.{layer}.ffn_up_exps.weight`
- routed MoE tensors named `blk.{layer}.ffn_down_exps.weight`
- dense zero-based routed layer ids for `sidecar_component_index.py`
- `MUL_MAT_ID` graph-pool patching for the same routed weight tensors

Runtime-gated today:

- expert bias tensors, for example gpt-oss `blk.*.ffn_*_exps.bias`
- batch/ubatch prefill acceleration
- cb_eval readback removal
- sidecar cache/direct-mmap/copy-mode choices

Blocked until an adapter exists:

- fused `ffn_gate_up_exps.weight` expert tensors
- custom expert tensor names
- non-dense layer numbering in the component index
- dense LLMs with no routed expert weights

## Required Flow For A New Model

1. Profile the exact GGUF shard set:

```bash
python3 phase0/model_profile.py \
  model-00001-of-00002.gguf model-00002-of-00002.gguf \
  --json runs/model_profile.json \
  --md runs/model_profile.md
```

2. Build sidecar artifacts only if `sidecar_plan_split_v0` and
   `component_index_v0` are supported.
3. Run same-backend logits parity before publishing speed:
   Metal op-offload baseline vs MoEVM graph-pool.
4. If `expert_bias_tensors.count > 0`, verify graph-pool rewrites only
   `MUL_MAT_ID` weight nodes and never rewrites `ADD_ID` bias ids.
5. Treat every optimization as model-local until it passes the profile and
   parity gates on that model.

## Extension Contract

To add support for a new layout, add it in this order:

1. Extend `model_profile.py` so the layout is detected and named.
2. Extend `sidecar_plan.py` so records map source slices correctly.
3. Extend `sidecar_component_index.py` or introduce a new component-index
   version if the record shape is not `gate/up/down`.
4. Extend `moe-trace.cpp` tensor parsing so graph-pool patches the matching
   `MUL_MAT_ID` weight nodes only.
5. Add a same-backend logits parity run for the new layout before any tok/s
   number is used.

This keeps MoEVM portable across MoE LLMs while avoiding silent correctness
bugs from plausible-but-wrong tensor remaps.
