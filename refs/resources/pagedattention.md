# Efficient Memory Management for Large Language Model Serving with PagedAttention

| Field | Value |
|-------|-------|
| Source | https://arxiv.org/abs/2309.06180 |
| Type | paper |
| Topics | 22 |
| Authors | Kwon et al. |
| Year | 2023 |
| Status | condensed |

## Why it matters

The canonical reference for *paged* KV-cache management in vLLM. It adapts OS virtual memory ideas (fixed-size blocks, block tables, copy-on-write, preemption) to Transformer attention KV caches to nearly eliminate fragmentation and enable flexible KV sharing.

Even though Scope is *video DiT* (not an LLM server), this paper provides the design vocabulary for our KV-cache lifecycle and eviction/recompute patterns (hard cut, ring-buffer eviction, recompute), and for keeping head-sharded caches consistent across ranks/stages.

## Key sections

- [Memory waste in existing systems](../../sources/pagedattention/full.md#31-memory-management-in-existing-systems) — reserved slots + internal/external fragmentation.
- [PagedAttention algorithm](../../sources/pagedattention/full.md#41-pagedattention) — block-wise attention over non-contiguous KV blocks.
- [KV cache manager](../../sources/pagedattention/full.md#42-kv-cache-manager) — logical blocks, physical blocks, block tables.
- [Scheduling and preemption](../../sources/pagedattention/full.md#45-scheduling-and-preemption) — eviction granularity; swapping vs recomputation.
- [Distributed execution](../../sources/pagedattention/full.md#46-distributed-execution) — head-sharded KV with a shared block table.
- [Ablations](../../sources/pagedattention/full.md#7-ablation-studies) — indirection overhead; block size; swap vs recompute.

## Core claims

1. **Claim**: Prior LLM serving systems store each request’s KV cache in a contiguous tensor and typically pre-allocate space up to a maximum sequence length, which creates large memory waste from (a) *reservation* for future tokens, (b) *internal fragmentation* from over-provisioning, and (c) *external fragmentation* from the allocator; profiling shows only **20.4%–38.2%** of KV-cache memory holds actual token states in baselines, while vLLM reaches **96.3%** token-state utilization.
   **Evidence**: [sources/pagedattention/full.md#1-introduction](../../sources/pagedattention/full.md#1-introduction), [sources/pagedattention/full.md#31-memory-management-in-existing-systems](../../sources/pagedattention/full.md#31-memory-management-in-existing-systems)

2. **Claim**: PagedAttention partitions a sequence’s KV cache into fixed-size **KV blocks** (block size `B` tokens) and reformulates attention as a **block-wise** computation over `(K_j, V_j)` blocks, enabling the KV cache to live in **non-contiguous physical memory** while remaining logically contiguous to the model.
   **Evidence**: [sources/pagedattention/full.md#41-pagedattention](../../sources/pagedattention/full.md#41-pagedattention)

3. **Claim**: vLLM’s KV-cache manager applies a virtual-memory abstraction: each sequence has **logical KV blocks** that grow left→right as tokens are generated; a block engine manages a pool of **physical KV blocks**; and a per-sequence **block table** maps logical→physical blocks (plus “filled positions”), enabling dynamic growth without reserving the full maximum length.
   **Evidence**: [sources/pagedattention/full.md#42-kv-cache-manager](../../sources/pagedattention/full.md#42-kv-cache-manager), [sources/pagedattention/full.md#43-decoding-with-pagedattention-and-vllm](../../sources/pagedattention/full.md#43-decoding-with-pagedattention-and-vllm)

4. **Claim**: vLLM supports KV-cache sharing across sequences (parallel sampling, beam search, shared prefixes) using **reference counts** on physical blocks and a **copy-on-write** mechanism at block granularity, which can avoid repeated large KV-cache copies (often copying only a single block when divergence occurs).
   **Evidence**: [sources/pagedattention/full.md#44-application-to-other-decoding-scenarios](../../sources/pagedattention/full.md#44-application-to-other-decoding-scenarios)

5. **Claim**: vLLM’s design cleanly supports multiple decoding algorithms by expressing them in terms of three KV-cache lifecycle operations—**fork**, **append**, and **free**—and by hiding complex sharing patterns behind the logical→physical mapping layer.
   **Evidence**: [sources/pagedattention/full.md#44-application-to-other-decoding-scenarios](../../sources/pagedattention/full.md#44-application-to-other-decoding-scenarios), [sources/pagedattention/full.md#52-supporting-various-decoding-algorithms](../../sources/pagedattention/full.md#52-supporting-various-decoding-algorithms)

6. **Claim**: Under memory pressure, vLLM uses FCFS scheduling and preempts at the **sequence (or sequence-group) granularity**: it evicts “all or none” of a sequence’s blocks (and gang-schedules related sequences), and can recover evicted state via either **swapping** blocks to CPU memory or **recomputation** of the KV cache.
   **Evidence**: [sources/pagedattention/full.md#45-scheduling-and-preemption](../../sources/pagedattention/full.md#45-scheduling-and-preemption), [sources/pagedattention/full.md#73-comparing-recomputation-and-swapping](../../sources/pagedattention/full.md#73-comparing-recomputation-and-swapping)

7. **Claim**: vLLM supports distributed execution with Megatron-style tensor parallelism by sharding attention heads across workers while keeping a **single shared KV-cache manager** (logical→physical mapping); each iteration, the scheduler broadcasts token inputs and **block tables** so workers can read/write the correct physical blocks for their local head shard.
   **Evidence**: [sources/pagedattention/full.md#46-distributed-execution](../../sources/pagedattention/full.md#46-distributed-execution)

8. **Claim**: Implementing paged KV-cache access efficiently requires kernel work: vLLM introduces fused kernels for **reshape+block write**, **block read fused with attention**, and **batched block copy** for copy-on-write, and adds support for variable sequence lengths.
   **Evidence**: [sources/pagedattention/full.md#51-kernel-level-optimization](../../sources/pagedattention/full.md#51-kernel-level-optimization)

9. **Claim**: Although dynamic block mapping introduces indirection overhead (reported as **20–26%** higher attention-kernel latency vs FasterTransformer), vLLM’s end-to-end performance improves substantially because better KV-cache utilization enables larger effective batches; the paper reports sustained request-rate gains (e.g., **1.7×–2.7×** vs Orca (Oracle), **2.7×–8×** vs Orca (Max), and up to **22×** vs FasterTransformer in tested settings) and summarizes **2–4×** overall throughput improvement over prior systems.
   **Evidence**: [sources/pagedattention/full.md#62-basic-sampling](../../sources/pagedattention/full.md#62-basic-sampling), [sources/pagedattention/full.md#71-kernel-microbenchmark](../../sources/pagedattention/full.md#71-kernel-microbenchmark), [sources/pagedattention/full.md#abstract](../../sources/pagedattention/full.md#abstract), [sources/pagedattention/full.md#10-conclusion](../../sources/pagedattention/full.md#10-conclusion)

10. **Claim**: The **block size** `B` is a first-order tuning knob: smaller blocks reduce internal fragmentation and can increase sharing opportunities but can underutilize the GPU and make swapping expensive (many small transfers), while larger blocks increase fragmentation and reduce sharing; the paper finds **B=16** is a practical default and characterizes when **recomputation vs swapping** wins as `B` changes.
   **Evidence**: [sources/pagedattention/full.md#72-impact-of-block-size](../../sources/pagedattention/full.md#72-impact-of-block-size), [sources/pagedattention/full.md#73-comparing-recomputation-and-swapping](../../sources/pagedattention/full.md#73-comparing-recomputation-and-swapping)

## Key technical details

### Paging model (what “paged KV cache” means)

- **Logical timeline vs physical storage**: tokens fill **logical blocks** left→right; a block table maps each logical block to a **physical block ID** (plus “#filled”); physical blocks are carved out of a pre-allocated pool on GPU (and optionally CPU for swapping).
  **Evidence**: [sources/pagedattention/full.md#42-kv-cache-manager](../../sources/pagedattention/full.md#42-kv-cache-manager)

- **KV block size (`B`)**: each KV block holds K/V for a fixed number of tokens; the last block is often partially filled.
  **Evidence**: [sources/pagedattention/full.md#41-pagedattention](../../sources/pagedattention/full.md#41-pagedattention)

### Block-wise attention (PagedAttention)

- PagedAttention rewrites attention to operate over `(K_j, V_j)` **blocks** and fetches blocks independently during the kernel, so the KV cache need not be contiguous.
  **Evidence**: [sources/pagedattention/full.md#41-pagedattention](../../sources/pagedattention/full.md#41-pagedattention)

### Sharing + copy-on-write (CoW)

- **Reference counts** on physical blocks enable sharing prompt KV across sequences; CoW triggers when a writer needs to modify a shared physical block (allocate new block, copy old block, then write).
  **Evidence**: [sources/pagedattention/full.md#44-application-to-other-decoding-scenarios](../../sources/pagedattention/full.md#44-application-to-other-decoding-scenarios)

- vLLM expresses decoding algorithms via three operations: `fork` (new sequence shares blocks), `append` (advance, allocate blocks as needed), and `free` (release blocks).
  **Evidence**: [sources/pagedattention/full.md#52-supporting-various-decoding-algorithms](../../sources/pagedattention/full.md#52-supporting-various-decoding-algorithms)

### Preemption and recovery

- **Eviction granularity**: “all-or-nothing” eviction per sequence; **sequence groups** (e.g., beam candidates) are preempted/rescheduled together because they may share blocks.
  **Evidence**: [sources/pagedattention/full.md#45-scheduling-and-preemption](../../sources/pagedattention/full.md#45-scheduling-and-preemption)

- **Swap vs recompute**: swapping copies blocks to CPU RAM; recompute regenerates KV by rerunning a prompt-style pass; ablations show recompute is better for small `B` (swap dominated by many small transfers) and competitive for medium `B`.
  **Evidence**: [sources/pagedattention/full.md#45-scheduling-and-preemption](../../sources/pagedattention/full.md#45-scheduling-and-preemption), [sources/pagedattention/full.md#73-comparing-recomputation-and-swapping](../../sources/pagedattention/full.md#73-comparing-recomputation-and-swapping)

### Distributed execution (tensor parallel heads)

- Workers shard attention heads but share the same block table IDs for a request; each worker stores only its local head slice of KV for those physical blocks, and receives the block table each decoding iteration as part of the control message.
  **Evidence**: [sources/pagedattention/full.md#46-distributed-execution](../../sources/pagedattention/full.md#46-distributed-execution)

### Kernel implications / overhead

- PagedAttention adds block-table indirection and variable-length handling; vLLM mitigates overhead with fused kernels (reshape+write, read+attention, batched CoW copies). Microbenchmarks report **20–26%** higher attention-kernel latency vs FasterTransformer, but end-to-end wins due to higher effective batching.
  **Evidence**: [sources/pagedattention/full.md#51-kernel-level-optimization](../../sources/pagedattention/full.md#51-kernel-level-optimization), [sources/pagedattention/full.md#71-kernel-microbenchmark](../../sources/pagedattention/full.md#71-kernel-microbenchmark)

## Actionables / gotchas

- **Use this paper’s *vocabulary*, not its exact implementation**: Scope’s streaming video DiT KV cache is already a fixed-size **ring buffer**; the transferable idea is separating *logical time* (global token indices) from *physical storage* (ring slots / pages) and making eviction/rebuild an explicit lifecycle.
  - Grounding: `refs/implementation-context.md` (row for `pagedattention`), `scope-drd/notes/FA4/h200/tp/explainers/05-kv-cache-head-sharding.md` (“Cache Index Consistency”).

- **Make cache lifecycle events first-class control-plane decisions** (and keep them lockstep across ranks/stages):
  - **Hard cut**: reset cache + bump `cache_epoch` (scene change).
  - **Recompute**: rebuild recent context window when ring buffer fills.
  - **Soft transition**: adjust history influence via a scalar bias.
  - Grounding: `scope-drd/notes/FA4/h200/tp/explainers/05-kv-cache-head-sharding.md` (“Cache Lifecycle Events”), `scope-drd/notes/FA4/h200/tp/explainers/03-broadcast-envelope.md` (header `cache_epoch` semantics).

- **Prefer all-or-nothing eviction semantics for correctness**: PagedAttention’s “evict all blocks of a sequence” and “gang-schedule sequence groups” is the safe mental model for our PP envelopes too—don’t partially evict per-stage/per-rank state if it can cause divergent cache state and downstream mismatched collectives.
  - Evidence (paper): [sources/pagedattention/full.md#45-scheduling-and-preemption](../../sources/pagedattention/full.md#45-scheduling-and-preemption)

- **Recompute is a first-class recovery path in our system**: the PagedAttention paper treats recomputation as a viable alternative to swapping; this matches our existing “re-encode recent context frames” story when the cache fills.
  - Grounding: `scope-drd/notes/FA4/h200/tp/explainers/05-kv-cache-head-sharding.md` (“Recompute”).
  - Evidence (paper tradeoff): [sources/pagedattention/full.md#73-comparing-recomputation-and-swapping](../../sources/pagedattention/full.md#73-comparing-recomputation-and-swapping)

- **Head-sharded KV caches still need shared metadata**: vLLM’s distributed mode matches our TP reality: each rank stores only its head slice, but *must* agree on the logical→physical mapping / indices. Keep mapping decisions deterministic and driven by broadcast inputs (not local heuristics).
  - Evidence (paper): [sources/pagedattention/full.md#46-distributed-execution](../../sources/pagedattention/full.md#46-distributed-execution)
  - Grounding: `scope-drd/notes/FA4/h200/tp/explainers/05-kv-cache-head-sharding.md` (“Cache Index Consistency”).

- **Beware “paging overhead” in compute-bound regimes**: the paper explicitly notes paging helps when the workload is memory-capacity-bound and dynamic, but can degrade performance when the workload is compute-bound due to indirection and non-contiguity. Treat any “paged” KV-cache refactor as a measured optimization, not a default.
  - Evidence: [sources/pagedattention/full.md#8-discussion](../../sources/pagedattention/full.md#8-discussion), [sources/pagedattention/full.md#71-kernel-microbenchmark](../../sources/pagedattention/full.md#71-kernel-microbenchmark)

- **If you want CUDA Graphs, avoid dynamic allocation inside captured regions**: PagedAttention relies on dynamic block allocation and mutable block tables; CUDA graphs generally want stable addresses and shapes. If we ever add a “paged table” to Scope, keep it as a fixed-size tensor updated in-place (metadata updates) and allocate the backing pools ahead of capture.
  - Tie-in cards: `refs/resources/cuda-graphs-guide.md`, `refs/resources/pytorch-cuda-semantics.md`.

- **Block size is “cache page size”**: if we introduce any block/paging abstraction (e.g., per-chunk pages or per-stage cached segments), the `B` trade-offs apply directly: small pages reduce waste but increase bookkeeping / swap overhead; large pages increase fragmentation and reduce sharing. Start from measurement and consider `B=16` as an existence proof, not a requirement.
  - Evidence: [sources/pagedattention/full.md#72-impact-of-block-size](../../sources/pagedattention/full.md#72-impact-of-block-size)

## Related resources

- [dit-paper](dit-paper.md) -- the DiT architecture whose attention KV caches need management
- [streamdiffusionv2](streamdiffusionv2.md) -- sink-token rolling KV cache for video diffusion
- [cuda-graphs-guide](cuda-graphs-guide.md) -- graph capture constraints (stable allocations) relevant to any paged/block-table cache design
- [pipedit](pipedit.md) -- attention co-processing in pipeline-parallel DiT
- [pytorch-cuda-semantics](pytorch-cuda-semantics.md) -- GPU memory management underlying KV cache allocation
