# MoEVM FAQ

Short answers to the questions people usually ask first.

## Is this just swap?

No. MoEVM does not rely on macOS swapping arbitrary model pages under pressure.
It packs routed expert weights into a sidecar and explicitly fills a bounded
Metal pool with the selected experts needed by the current graph. The runtime
tracks epochs, pinned slots, evictions, and pressure-triggered trims.

## Why not just use a smaller quant?

Smaller quants are the right answer when they meet the quality target and fit in
memory. MoEVM is for the case where the model is a routed MoE and the chosen
weights do not fit as one resident blob. The headline proof is gpt-oss-120b Q4_K_M:
58 GiB of source weights on a 48 GB MacBook with same-backend logits parity.

## Why not MLX?

MLX is strong for resident Apple Silicon inference. MoEVM is testing a different
system boundary: packed expert sidecars, bounded expert residency, graph-pool
rewrites, epoch/pinning correctness, and pressure-aware trimming. A future MLX
port may be interesting, but the current proof is inside a llama.cpp fork
because it exposes the graph and routed expert operations needed for this
experiment.

## Does this destroy the SSD?

The runtime mostly reads from the packed sidecar. Reads do not have the same
wear profile as writes. Packing the sidecar writes the expert pool once; decode
then performs bounded reads and cache hits. The public numbers should report live
decode tok/s and demand-read behavior, not page-cache-contaminated replay tok/s.

## How is this different from KTransformers?

KTransformers is an important reference for hybrid CPU/GPU MoE serving, mainly
in CUDA/Linux-style environments. MoEVM is focused on Apple Silicon laptops:
Metal, unified memory pressure, Darwin `MADV_FREE_REUSABLE`, and a proof point
on a 48 GB MacBook. The comparison is architectural, not a claim that MoEVM is a
drop-in replacement.

## Why is gpt-oss-120b only about 9 tok/s?

The published figure is 8.72 tok/s: the median of five runs at a 4096-token
context with a 990-token prompt
(`runs/gpt_oss_120b_median5_observability_24g_20260705.md`). Earlier single
runs on shorter prompts were faster and are not the reference number.

The bottleneck is not raw model size alone. On gpt-oss, a timestamp diagnostic
showed low GPU busy time and high orchestration/readback cost, and the median
runs had zero demand reads, so disk paging is not the limit. There is still
optimization headroom, but the harder claim is the one measured here: a
larger-than-RAM MoE model decodes with logits parity and no jetsam.

## Why not include every tok/s optimization yet?

The remaining large optimization, ids-readback/orchestration surgery, moves the
epoch fence. That is a correctness-critical path and needs a full parity,
jitter, and soak campaign. It is planned as separate follow-up work rather than
a rushed patch.

## Does this work for dense 1T models?

No. The current design depends on routed MoE sparsity. Dense models do not have a
small active expert set per token, so the same paging trick does not apply.

## Is gpt-oss speculative decoding part of the current path?

No. Qwen DFlash is integrated, but the graph-build runtime already removed much
of the fixed overhead that speculative decoding was amortizing. On Qwen, DFlash
is currently a documented negative or small result, not a headline accelerator.
gpt-oss has a cleaner theoretical profile because it lacks Qwen3Next's recurrent
state tax, but there is no validated drafter in the current path.

## What is upstreamable?

The sparse-mmap loader work is isolated and already represented by an upstream
draft PR, [ggml-org/llama.cpp#25309](https://github.com/ggml-org/llama.cpp/pull/25309).
MoEVM's full graph-pool runtime is more experimental, but the project keeps
separate safety gates, benchmark summaries, and parity harnesses so pieces can
be reviewed independently.
