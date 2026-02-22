---
status: draft
---

# Topic 7: GPU memory management — PyTorch caching allocator, fragmentation

The PyTorch caching allocator avoids expensive `cudaMalloc`/`cudaFree` calls by maintaining a pool of allocated blocks. Fragmentation under dynamic workloads (variable sequence lengths, different denoising steps) is the primary operational concern.

## Resources

<!-- Resource IDs from manifest belonging to this topic -->
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| pytorch-cuda-semantics | PyTorch CUDA Semantics | high | condensed |
| pytorch-cuda-module | torch.cuda Module Reference | medium | pending |

## Implementation context

TP head-sharding halves KV-cache memory per rank: per layer ~335 MB → ~167 MB, so K+V across 40 layers is ~26 GB → ~13 GB per rank (TP=2). In the first stable server run (Run 7), both ranks still sat at ~55.2 GiB / 79.6 GiB used, so allocator stability and “don’t allocate per chunk” discipline matter. This memory pressure is a key motivator for v1.1c: generator-only workers are only worth the risk if worker memory drops by ≥10 GB or ≥15% and startup improves materially.

See: `refs/implementation-context.md`, `scope-drd/notes/FA4/h200/tp/explainers/05-kv-cache-head-sharding.md` (cache sizing), `scope-drd/notes/FA4/h200/tp/bringup-run-log.md` (Run 7 nvtop memory), `scope-drd/notes/FA4/h200/tp/v1.1-generator-only-workers.md` (acceptance gates).

## Synthesis

<!-- To be filled during study -->

### Mental model

GPU memory management is mostly “invisible infrastructure” until it becomes the bottleneck. The caching allocator exists to avoid paying `cudaMalloc`/`cudaFree` costs on the hot path, but under real workloads it introduces three classes of problems: (1) **peak memory** (you simply exceed the HBM budget), (2) **fragmentation** (you have enough total free memory but not in the right block sizes), and (3) **allocator stalls** (expensive reclaim/GC paths show up as latency spikes). PyTorch calls out allocator tuning as a *latency* lever, not just a memory lever (see `pytorch-cuda-semantics` claim 5).

For our system, the allocator stressor is PP-style concurrency: we’re trying to keep multiple items in flight (`D_in=2`, `D_out=2` in the PP plan), which means multiple “envelopes” worth of activation/KV/temporary tensors can be resident at once. Even before PP, TP=2 streaming inference had significant headroom pressure (~55.2 GiB / 79.6 GiB used in Run 7 per the topic’s Implementation context). The mental model is: *any per-chunk allocation you accidentally introduce gets multiplied by in-flight depth*, and allocator behavior becomes user-visible.

CUDA graphs add a second layer: capture/replay is sensitive to **address stability**. PyTorch uses graph-private pools to keep allocation addresses stable across replays (see `pytorch-cuda-semantics` claim 8), while the CUDA programming guide describes graph-owned allocations with fixed virtual addresses and strict capture rules (see `cuda-graphs-guide` claim 3 and claim 7). In other words, “graph-safe” implies “allocator-safe.”

### Key concepts

Key vocabulary for debugging “OOM / stutter / fragmentation” issues:

- **Caching allocator**: PyTorch keeps a pool of GPU memory blocks to amortize allocation/free overhead; it can recycle addresses aggressively, which is why stream-ordering and lifetime tracking matter (see `pytorch-cuda-semantics` claim 3).
- **Reserved vs allocated vs active**: “reserved” is what the allocator has asked CUDA for; “allocated/active” is what tensors currently hold. Large gaps can indicate fragmentation or caching behavior; sudden changes often indicate GC/reclaim.
- **Fragmentation**: enough total free memory exists, but it’s split into chunks that don’t satisfy a large allocation request. This tends to show up when sizes vary over time (dynamic shapes, conditional branches) and when there are multiple concurrent in-flight workloads.
- **Allocator knobs (`PYTORCH_ALLOC_CONF`)**: a set of controls that trade fragmentation vs reuse vs latency; `garbage_collection_threshold` exists specifically to avoid expensive global reclaim paths in latency-sensitive workloads (see `pytorch-cuda-semantics` claim 5).
- **Graph-private memory pool**: during CUDA graph capture, allocations are routed to a graph-private pool to preserve pointer/address stability across replay (see `pytorch-cuda-semantics` claim 8).
- **CUDA stream capture constraints**: during capture, many operations are prohibited and invalidation is easy; multi-stream capture requires explicit event-based fork/join and rejoining the origin stream (see `cuda-graphs-guide` claim 4 and claim 5).
- **MemPool / custom allocators**: PyTorch supports scoping allocations via `torch.cuda.MemPool` instead of swapping the allocator globally (see `pytorch-cuda-semantics` claim 6).
- **`ncclMemAlloc` / NVLS path**: one motivation for MemPool is allocating comm buffers in NCCL-friendly ways (NVLink Switch Reductions), but this has sharp compatibility and OOM risks (see `pytorch-cuda-semantics` claim 6 and claim 7).

### Cross-resource agreement / disagreement

Agreement:
- Both PyTorch CUDA semantics and the CUDA graphs guide treat “async GPU execution” as the baseline and make correctness depend on stream/event ordering and object lifetimes. For memory, the shared theme is that *you don’t get correctness “for free” once you leave the default stream / eager launch model* (see `pytorch-cuda-semantics` claim 2–3; `cuda-graphs-guide` claim 4–5).
- Both sources implicitly assume you’re chasing *steady-state performance* by amortizing overhead: allocator pooling amortizes allocation cost; graphs amortize launch setup cost (see `pytorch-cuda-semantics` claim 8; `cuda-graphs-guide` claim 1).

Differences (useful, not contradictory):
- PyTorch focuses on practical allocator controls and the “graph-private pool” integration story, while the CUDA guide focuses on the underlying graph constraints and memory-node semantics (fixed virtual addresses, reuse/remap). When debugging a capture failure or a “works once, breaks on replay” issue, you often need both views (see `pytorch-cuda-semantics` claim 8; `cuda-graphs-guide` claim 7).
- `ncclMemAlloc`/NVLS is presented as an advanced comm optimization path with explicit footguns; it’s not something you reach for during bringup (see `pytorch-cuda-semantics` claim 7).

### Practical checklist

Before PP (baseline hygiene):
- **Eliminate per-chunk allocations in hot paths**: pre-allocate buffers where possible; any “small” allocation multiplied by in-flight depth becomes a cliff under PP.
- **Track allocator headroom explicitly**: monitor reserved/allocated deltas and peaks during a representative run; treat “reserved grows unbounded” as a leak/fragmentation smell.
- **Treat stream usage as part of memory safety**: if you introduce non-default streams for overlap, follow `wait_stream`/`record_stream` patterns or you can recycle an address while a kernel still uses it (see `pytorch-cuda-semantics` claim 2–3).

When enabling PP overlap (`D_in=2`/`D_out=2`):
- **Re-evaluate peak memory under concurrency**: your peak is no longer “one envelope”; it’s “envelopes in flight + KV cache + decode buffers + staging buffers.” Budget explicitly for the additional in-flight resident tensors.
- **Prefer stable shapes when possible**: dynamic sizes are a fragmentation multiplier and also a graph-capture hazard. For PP bringup, “static shapes” is the safest contract.

If/when exploring CUDA graphs:
- **Only capture the steady-state compute region**: keep control-plane logic, conditionals, and “sync-ish” operations outside capture (capture invalidation is easy; see `cuda-graphs-guide` claim 5).
- **Assume address stability is required**: plan around graph-private pools / fixed addresses (see `pytorch-cuda-semantics` claim 8; `cuda-graphs-guide` claim 7).

### Gotchas and failure modes

- **“Enough free memory” but still OOM**: fragmentation can make large allocations fail even when total free memory looks sufficient; PP concurrency increases the variety and lifetime overlap of allocations, which tends to worsen fragmentation.
- **Allocator reuse + streams = heisenbugs**: without correct stream synchronization, the allocator can recycle memory whose previous use is still in flight (see `pytorch-cuda-semantics` claim 3). This can show up as silent corruption, not just crashes.
- **Allocator stalls masquerade as “model jitter”**: global reclaim/GC paths can add tail latency; `garbage_collection_threshold` exists because this is a real issue in server-like workloads (see `pytorch-cuda-semantics` claim 5).
- **Graph capture is brittle**: prohibited operations and missing stream re-joins will invalidate capture (see `cuda-graphs-guide` claim 4–5). If capture fails intermittently, suspect hidden sync calls inside libraries.
- **NVLS / comm-allocator experiments can change the memory budget**: `ncclMemAlloc` can allocate more than requested (alignment) and can fall back when buffers aren’t compatible (see `pytorch-cuda-semantics` claim 7). Treat this as Phase-2+ work only.

### Experiments to run

- **Peak scaling with in-flight depth**: measure peak reserved/allocated at `D_in/D_out = 1/1`, `2/2`, `4/4`. If peak grows faster than expected, you’re allocating per-item or leaking.
- **Fragmentation sensitivity sweep**: run fixed-shape vs intentionally varied-shape workloads and compare reserved−allocated gaps and OOM thresholds; this helps distinguish “true capacity” from fragmentation.
- **Allocator knob A/B**: change one `PYTORCH_ALLOC_CONF` knob at a time (especially `garbage_collection_threshold`) and compare tail latency and peak reserved (see `pytorch-cuda-semantics` claim 5).
- **Stream-safety regression test**: if you add a side stream for overlap, add a test that stresses allocator reuse (allocate/free on one stream while computing on another) and confirm correctness with explicit `record_stream` usage (see `pytorch-cuda-semantics` claim 2–3).
- **CUDA-graph readiness probe**: attempt capture of a minimal steady-state forward with all shapes fixed; if capture fails, use the prohibited-ops list and origin-stream fork/join rules to pinpoint the first invalidating call (see `cuda-graphs-guide` claim 4–5).
