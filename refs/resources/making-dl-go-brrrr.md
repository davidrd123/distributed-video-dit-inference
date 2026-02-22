# Making Deep Learning Go Brrrr From First Principles

| Field | Value |
|-------|-------|
| Source | https://horace.io/brrr_intro.html |
| Type | blog |
| Topics | 8, 16 |
| Author | Horace He |
| Status | extracted |

## Why it matters

The definitive blog post explaining overhead-bound, memory-bound, and compute-bound regimes. The overhead section explains exactly why Python dispatch matters (~10us per kernel launch) and how operator fusion eliminates it. Also the single best resource for building roofline intuition -- the starting point for any GPU performance reasoning about DiT attention (memory-bandwidth-bound) vs FFN layers (compute-bound).

## Core claims

1. **Claim**: GPU performance bottlenecks fall into exactly three regimes -- compute-bound, memory-bandwidth-bound, and overhead-bound -- and knowing which regime you are in determines which optimizations will actually help.
   **Evidence**: The article's central framework; see [sources/making-dl-go-brrrr/full.md#making-deep-learning-go-brrrr-from-first-principles](../sources/making-dl-go-brrrr/full.md#making-deep-learning-go-brrrr-from-first-principles) (introduction) and the conclusion table mapping regimes to solutions.

2. **Claim**: Non-matmul operations (layer norm, activations, pointwise ops) are only 0.2% of total FLOPS in models like BERT, but achieve 250x-700x fewer FLOPS than matmuls because they are memory-bandwidth-bound, not compute-bound.
   **Evidence**: BERT FLOP breakdown table in [sources/making-dl-go-brrrr/full.md#compute](../sources/making-dl-go-brrrr/full.md#compute).

3. **Claim**: Operator fusion is "the most important optimization in deep learning compilers" -- it eliminates redundant global memory round-trips between sequential operations, cutting memory bandwidth cost proportionally.
   **Evidence**: The `x.cos().cos()` example reducing 4 global memory accesses to 2, achieving a 2x speedup. See [sources/making-dl-go-brrrr/full.md#bandwidth](../sources/making-dl-go-brrrr/full.md#bandwidth).

4. **Claim**: Compute grows faster than memory bandwidth (FLOPS doubling time < bandwidth doubling time), making it structurally harder over time to keep GPUs saturated.
   **Evidence**: ACM 2004 table on doubling times; factory analogy in [sources/making-dl-go-brrrr/full.md#compute](../sources/making-dl-go-brrrr/full.md#compute).

5. **Claim**: PyTorch's asynchronous CUDA execution hides framework overhead as long as the CPU can "run ahead" of GPU kernel execution -- but with many small operators, the GPU starves and overhead dominates.
   **Evidence**: Profiler traces showing GPU gaps vs. fully-utilized GPU in [sources/making-dl-go-brrrr/full.md#overhead](../sources/making-dl-go-brrrr/full.md#overhead).

6. **Claim**: A fused `x.cos().cos()` takes nearly the same time as a single `x.cos()`, because both are memory-bandwidth-bound and the extra compute is free. This means all activation functions cost roughly the same when fused.
   **Evidence**: Discussed in the operator fusion section; see [sources/making-dl-go-brrrr/full.md#bandwidth](../sources/making-dl-go-brrrr/full.md#bandwidth).

7. **Claim**: On an A100, pointwise operations remain memory-bound until ~64 sequential fused operations per element; only beyond that does the operation become compute-bound.
   **Evidence**: Microbenchmark plots (log-log scale) in [sources/making-dl-go-brrrr/full.md#reasoning-about-memory-bandwidth-costs](../sources/making-dl-go-brrrr/full.md#reasoning-about-memory-bandwidth-costs).

## Key insights

- **The roofline model as diagnostic tool**: The ratio of arithmetic intensity (FLOPS per byte transferred) to the hardware's compute-to-bandwidth ratio determines which regime you are in. You can calculate this with a simple formula: `A100 can load 400B numbers in the time it computes 20T ops (at fp32)`, so you need ~50 ops per element to break even.
- **Operator fusion is a bandwidth optimization, not a compute optimization**: Fusion doesn't reduce FLOPS -- it reduces memory traffic by keeping intermediates in SRAM/registers instead of round-tripping through DRAM.
- **Rematerialization can reduce both memory AND runtime**: Because fused recomputation is "free" when memory-bandwidth-bound, activation checkpointing with a fusing compiler can save memory without increasing wall-clock time (or even decrease it). This is counterintuitive -- recomputing is literally faster than saving and reloading.
- **The overhead regime is diagnosed by scaling**: If doubling batch size doesn't proportionally increase runtime, you're overhead-bound. This is a cheap, actionable diagnostic.
- **Tensor Cores create a 16x gap**: A100 does 312 TF with Tensor Cores but only 19.5 TF for non-matmul ops. This makes it critical that the high-FLOP operations (attention QKV projections, FFN linear layers) hit Tensor Cores.
- **The factory/warehouse/shipping analogy**: Compute = factory (SRAM), storage = warehouse (DRAM), bandwidth = shipping. Every kernel launch requires a round-trip to the warehouse. This mental model makes it intuitive why small kernels are wasteful.

## Actionables / gotchas

- **Profile first, optimize second**: Use PyTorch profiler traces to determine whether DiT inference is overhead-bound (gaps in GPU timeline), bandwidth-bound (low achieved FLOPS), or compute-bound (near-peak FLOPS). The regime determines the optimization strategy entirely.
- **DiT attention layers are likely bandwidth-bound**: Attention involves many pointwise and reduction operations (softmax, masking, dropout) between matmuls. Fusing these (e.g., FlashAttention) is the primary lever -- it keeps QK^T intermediates in SRAM rather than writing to DRAM.
- **DiT FFN layers are likely compute-bound**: The large linear projections in FFN blocks are matmul-heavy and should be the regime where you approach peak FLOPS. Ensure these hit Tensor Cores (fp16/bf16, dimensions divisible by 8/16).
- **Use torch.compile / CUDA graphs for overhead**: In distributed video DiT, the number of small operations per step (timestep embedding, conditioning, normalization) can push you into overhead-bound territory. torch.compile traces out the graph; CUDA graphs eliminate per-kernel launch overhead entirely.
- **Operator fusion across normalization + activation**: AdaLN (adaptive layer norm) in DiT involves scale/shift/gate pointwise operations -- these should fuse with adjacent operations. If using eager mode, you're paying separate kernel launches for each.
- **Batch size as overhead diagnostic**: If scaling from 1 to 2 video clips doesn't nearly double inference time, overhead is dominating. This is especially relevant for latency-sensitive single-sample inference.
- **Watch the bandwidth-compute gap across GPU generations**: Moving from A100 to H100 to B100 widens the gap further. Optimizations that were compute-bound on A100 may become bandwidth-bound on newer hardware -- revalidate assumptions when changing hardware.
- **Activation checkpointing + fusion synergy**: For long video sequences where memory is tight, recomputing fused pointwise chains (e.g., GELU + scale) may be strictly better than checkpointing and reloading from DRAM, given a fusing compiler is in the loop.

## Related resources

- [pytorch-cuda-semantics](pytorch-cuda-semantics.md) -- CUDA execution model underlying the performance regimes
- [cuda-graphs-guide](cuda-graphs-guide.md) -- CUDA graphs eliminate the kernel launch overhead described here
- [dynamo-deep-dive](dynamo-deep-dive.md) -- torch.compile addresses overhead-bound regime via tracing
- [dit-paper](dit-paper.md) -- the DiT architecture to apply roofline analysis to
- [ezyang-state-of-compile](ezyang-state-of-compile.md) -- compile as overhead elimination strategy
