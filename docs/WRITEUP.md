# Paging experts: running a 58 GiB MoE model on a 48 GB Mac

MoEVM is an experiment in treating the experts of a Mixture-of-Experts model
like pages of virtual memory. This is the story of what was built, what was
measured, what did not work, and the rules that kept the numbers honest. The
measurements in this repository were taken over about 27 hours on 4–5 July
2026, on one MacBook Pro with an M5 Pro and 48 GB of memory.

## The problem

Mixture-of-Experts models are large but sparse. Qwen3-Coder-Next has 512 routed
experts in each of its 48 layers and uses 10 of them per token; gpt-oss-120b
uses 4. For any given token, most of the file sits idle.

Apple Silicon shares one pool of memory between CPU and GPU, but Metal only
lets a process keep part of it resident. On this machine the recommended
working set is 40.2 GB. Qwen3-Coder-Next at Q4_K_M is a 45.2 GiB file.
gpt-oss-120b at Q4_K_M is 58 GiB, more than the machine's RAM.

llama.cpp can leave the expert tensors on the CPU side
(`-ot ".*ffn_.*_exps.weight=CPU"`) and then either compute them there or copy
them to the GPU for each operation (`--op-offload`). Both work, and neither
remembers that the next token will probably want many of the same experts.
On Qwen3-Coder-Next the op-offload path decoded 6.41 tok/s over 1024 tokens.

Even that baseline needed a fix first. llama.cpp mapped one Metal buffer from
the first to the last GPU tensor in the file, which swept in the CPU-side
experts between them and ran out of Metal memory. Mapping only the ranges each
buffer needs solved it. That loader change is the one piece of this work
proposed upstream, as draft PR
[ggml-org/llama.cpp#25309](https://github.com/ggml-org/llama.cpp/pull/25309).

## The approach

MoEVM is a pipeline with four stages, split between this repository and a
llama.cpp fork.

**Packer.** A capability check reads the GGUF tensor table and refuses expert
layouts it does not understand, so support follows tensor layout rather than
model names. The packer then copies every routed expert into one contiguous
sidecar file indexed by `(layer, expert, component)`. For Qwen3-Coder-Next
that is 24,576 records and 43.69 GiB, written in 42 seconds; for gpt-oss-120b,
4,608 records and 56.74 GiB in 56 seconds. The index records the sidecar's
size and a SHA-256 prefix, so a stale sidecar is refused at load time.

**RAM arena and prefetch.** A C++ runtime keeps a bounded arena of expert
records in RAM, evicts with LRU, and prefetches the candidate experts the
router is likely to pick next. Under macOS memory pressure it returns pages
with `MADV_FREE_REUSABLE` and trims itself, instead of waiting for the system
to kill the process.

**Persistent Metal pool.** In the fork, the routed `MUL_MAT_ID` nodes are
rewritten to read from a pool of Metal buffers that survives across tokens.
The first version hooked into llama.cpp's op-offload copy path. The second,
graph-build version adds a scheduler hook: it reads the selected expert ids
once per layer, fills the pool, patches the graph for that split and restores
it afterwards. Epochs and pinning stop an expert from being evicted while a
graph still reads it, and a test mode injects delays to force those fences.

**Logits parity.** Every claimed result is checked against the same build with
MoEVM off, by dumping the full logits at every decoded position and comparing
them. Matching tokens is a weak test: two runs can pick the same tokens while
their probabilities drift. A bit-identical comparison leaves nothing to argue
about.

Around all of it sits `safe_run.py`. Early gpt-oss experiments on Metal ended
in out-of-memory errors and GPU watchdog failures, so every heavy command now
runs through a wrapper that refuses known-dangerous shapes, stops before memory
pressure turns critical, and insists that the packer bypass the file cache so
writing 57 GiB does not evict everything else.

## The measurements

The headline pair is Qwen3-Coder-Next with MoEVM off and on: the same binary,
prompt, seed, 1536-token context and placement, routing traces off on both
sides, 1024 tokens each. MoEVM off decoded 6.41 tok/s and MoEVM on 19.90 tok/s,
3.1 times faster. The logits were bit-identical at all 1024 positions over the
full 151,936-token vocabulary.

gpt-oss-120b, the model larger than RAM, decoded 8.72 tok/s as the median of
five runs at a 4096-token context with a 990-token prompt. Individual runs
ranged from 7.65 to 10.64 tok/s. All five had zero demand reads, zero runtime
errors and no memory-pressure events, and produced the same output prefix.
Against a same-backend Metal reference, its logits matched exactly over 64
positions. Four parallel sequences on the same process reached 26.81 tok/s in
aggregate, 6.70 each.

The profiling says where the remaining time goes. On Qwen's graph-build path
the GPU was busy for about 53% of decode time. On a gpt-oss diagnostic run,
the GPU was busy for 20.07 s of 71.88 s, while reading expert ids back from the
GPU took 25.79 s. The cache is not the problem: on gpt-oss coding traces with a
28 GiB budget, plain LRU hits 95.10–95.75% of expert accesses, against
97.15–97.49% for the offline-optimal Belady policy. The lever left is
orchestration, not a cleverer cache.

It also matters where MoEVM loses. When a model fits, plain llama.cpp wins
easily: Qwen3-Coder-Next at Q3 (36.3 GB) runs fully on Metal at 49.69 tok/s,
against 11.76–11.95 tok/s through MoEVM. And CPU-only llama.cpp decodes
gpt-oss-120b at 10.85–14.36 tok/s on short contexts. MoEVM's gain is against
the GPU paths, with bounded memory, not against llama.cpp in general.

## What didn't work

**Filling the pool from an evaluation callback.** The first graph-pool
prototype read the selected experts through llama.cpp's per-node evaluation
callback. It produced correct output at 2.07 tok/s: the scheduler broke the
graph into thousands of tiny computes and spent 22.94 s synchronizing the
backend over 50 tokens. The scheduler hook replaced it.

**Caching experts in llama.cpp's copy buffer.** Reusing the scheduler's input
copy tensor as a durable expert cache reached 13.65 tok/s on a short prompt,
then failed its first parity check: the output collapsed into a repeated
token. That buffer is not safe storage across graph splits, and the idea was
dropped.

**Bigger batches.** At `b=4 ub=4`, gpt-oss matched a same-backend reference
running at the same batch size, but did not reproduce the token stream of
`b=1`. It is a throughput mode, not a drop-in replacement. At `b=8` the union
of selected experts overflowed the pool and the run stopped safely.

**Fewer experts.** Routing gpt-oss through 3 experts instead of 4 made it
1.22–1.46 times faster and failed the code-generation quality gate, with or
without renormalizing the router weights. It was not shipped. On Qwen, reducing
to 8 or 6 experts passed its gate but stays opt-in.

**Speculative decoding.** A draft model helped while per-token overhead was
high. Once the graph-build path removed most of that overhead, the remaining
gain on Qwen was small or negative, so it is not part of any headline.

**The 5.2× number.** An early comparison put Qwen at 4.24 → 21.90 tok/s. While
preparing this release, a closer reading of the logs showed that the 4.24
baseline, and a 10.90 tok/s "best hybrid" next to it, had routing traces on,
which read routing data back from the GPU at every token. The 21.90 run had
them off and decoded 50 tokens instead of 92. The comparison was dropped in
favour of the 6.41 → 19.90 pair, which differs only by the MoEVM flags. The
early 10–13 tok/s gpt-oss figures went the same way, replaced by the 8.72
median.

## The discipline

**Gates written before results.** The stability soak's pass criteria were
committed before any final soak ran, with the instruction not to relax them
after seeing the result: zero runtime errors, bit-identical logits snapshots,
less than 10% throughput drift between the first and last hour, a ceiling on
wired memory, no jetsam events.

**Medians, not best runs.** The published gpt-oss number is the median of
five, which is below the best single runs.

**No warm-cache numbers.** Replaying a trace after the file cache is warm
measures the cache, not the system. Only live decode throughput is reported.

**Parity before speed.** A configuration without a parity result does not get
a throughput claim. The raw logs behind every published number are in
[`runs/evidence/`](../runs/evidence/).

The same discipline applies to what is missing. The six-hour soak was never
completed: a pilot of about 30 minutes ran and was stopped on purpose, and its
validation was never finished. Long-run stability is not established, and
everything here was measured on a single machine.

## What's next

The main remaining lever is the round trip that reads the selected expert ids
back from the GPU before each pool fill. Removing it moves the epoch fence, so
it needs a full parity, stress and soak campaign of its own. After that: the
soak itself, and review of the upstream loader PR.
