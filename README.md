# MoEVM

MoEVM is a research/runtime project for virtual memory over Mixture-of-Experts
LLM weights.

The goal is to run self-hosted MoE models beyond physical memory limits by
paging routed expert weights instead of loading the model as a monolithic blob.

Phase 0 is intentionally measurement-first:

1. inspect model metadata without downloading full weights;
2. estimate resident core vs paged expert pool;
3. measure SSD random-read bandwidth at expert-slice granularity;
4. trace routed experts;
5. simulate cache policies before writing a runtime fork.

