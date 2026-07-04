# MoEVM Phase 0

Phase 0 answers one question before we write a pager:

> Is this MoE model worth paging on this machine?

Artifacts:

- `gguf_inspect.py`: parse GGUF headers in streaming mode, including truncated HTTP range downloads.
- `ssd_bench.py`: benchmark random reads at expert-slice size with optional macOS `F_NOCACHE`.
- `cache_sim.py`: replay router traces against cache policies.
- `sidecar_plan.py`: map `(layer, expert)` records to source GGUF tensor slices.
- `sidecar_pack.py`: pack those records into a contiguous MoEVM sidecar payload.
- `sidecar_replay.py`: replay traces against source GGUF slices or a packed sidecar.
- `sidecar_index.py`: build a compact runtime index for a packed sidecar.
- `sidecar_component_index.py`: build a compact component index for `gate/up/down` offsets inside each packed record.
- `pager_replay.cpp`: native C++ replay path using the compact index and packed sidecar, with optional bounded async prefetch workers.
- `moevm_runtime.hpp` / `moevm_runtime.cpp`: first reusable C++ runtime API with a real RAM arena, candidate prefetch, selected-expert waits, and resident record views.
- `runtime_smoke.cpp`: small runtime smoke test against a candidate trace.
- `runtime_replay.cpp`: streaming replay through the reusable runtime API, used to compare API overhead with `pager_replay.cpp`.
- `compare_decode_logs.py`: compares generated token IDs and decode/runtime stats between baseline and MoEVM llama-moe-trace logs.
- `prefetch_sim.py`: oracle lookahead simulator for upper-bound prefetch research.
- `safe_run.py`: run risky benchmarks with memory and log guardrails.
- `RUNBOOK.md`: exact commands for this Mac.
- `LLAMA_INTEGRATION.md`: mapped llama.cpp integration boundary for the next phase.
