# PyTorch CUDA Semantics

| Field | Value |
|-------|-------|
| Source | https://docs.pytorch.org/docs/stable/notes/cuda.html |
| Type | docs |
| Topics | 5, 6, 7 |
| Status | condensed |

## Why it matters

Comprehensive treatment of streams, events, the caching allocator, and CUDA graphs. The sections on stream synchronization semantics and backward pass stream behavior are especially important for pipeline parallelism, where overlapping compute, communication, and memory transfers is the core challenge.

## Key sections

- [Asynchronous execution](../../sources/pytorch-cuda-semantics/full.md#asynchronous-execution) — what “async CUDA” means in practice; timing correctly; `CUDA_LAUNCH_BLOCKING=1` debugging.
- [CUDA streams](../../sources/pytorch-cuda-semantics/full.md#cuda-streams) — `wait_stream` + `record_stream` rules; why non-default streams are a footgun without explicit sync.
- [Stream semantics of backward passes](../../sources/pytorch-cuda-semantics/full.md#stream-semantics-of-backward-passes) — safe/unsafe patterns when mixing streams around `backward()`.
- [Optimizing memory usage with `PYTORCH_ALLOC_CONF`](../../sources/pytorch-cuda-semantics/full.md#optimizing-memory-usage-with-pytorch_alloc_conf) — allocator knobs that affect fragmentation/latency (e.g., `garbage_collection_threshold`).
- [Mixing different CUDA system allocators in the same program](../../sources/pytorch-cuda-semantics/full.md#mixing-different-cuda-system-allocators-in-the-same-program) — `torch.cuda.MemPool` + custom allocators; `ncclMemAlloc` for NVLS experiments.
- [CUDA Graphs](../../sources/pytorch-cuda-semantics/full.md#cuda-graphs) — capture/replay model, constraints, multi-stream capture patterns, and graph-private memory pools.

## Core claims

1. **Claim**: CUDA ops are asynchronous by default: calls enqueue work on a device; accurate timing requires synchronization (e.g., `torch.cuda.synchronize()` or CUDA events), and `CUDA_LAUNCH_BLOCKING=1` can make GPU errors surface at the call site for debugging.
   **Evidence**: [sources/pytorch-cuda-semantics/full.md#asynchronous-execution](../../sources/pytorch-cuda-semantics/full.md#asynchronous-execution)

2. **Claim**: Operations on the default stream get PyTorch-managed synchronization for common data-movement cases, but when using non-default streams it is the user’s responsibility to add explicit synchronization; `Stream.wait_stream()` and `Tensor.record_stream()` are key primitives for correctness and lifetime safety.
   **Evidence**: [sources/pytorch-cuda-semantics/full.md#cuda-streams](../../sources/pytorch-cuda-semantics/full.md#cuda-streams)

3. **Claim**: Even without an explicit read dependency, stream synchronization can be required because the CUDA caching allocator may recycle memory with pending operations from the “old” use of the same address.
   **Evidence**: [sources/pytorch-cuda-semantics/full.md#cuda-streams](../../sources/pytorch-cuda-semantics/full.md#cuda-streams)

4. **Claim**: Backward CUDA ops run on the same stream(s) as their corresponding forward ops; mixing stream contexts around `backward()` follows normal stream semantics and requires explicit synchronization in the “unsafe” patterns shown in the doc.
   **Evidence**: [sources/pytorch-cuda-semantics/full.md#stream-semantics-of-backward-passes](../../sources/pytorch-cuda-semantics/full.md#stream-semantics-of-backward-passes)

5. **Claim**: `PYTORCH_ALLOC_CONF` exposes allocator controls that trade fragmentation, reuse, and latency; `garbage_collection_threshold` is explicitly motivated as a way to avoid expensive global reclaim paths that are unfavorable for latency-critical server workloads.
   **Evidence**: [sources/pytorch-cuda-semantics/full.md#optimizing-memory-usage-with-pytorch_alloc_conf](../../sources/pytorch-cuda-semantics/full.md#optimizing-memory-usage-with-pytorch_alloc_conf)

6. **Claim**: Instead of swapping the allocator process-wide, PyTorch supports mixing CUDA system allocators via `torch.cuda.MemPool` to scope a region’s allocations while keeping caching benefits; one cited use case is allocating all-reduce output buffers with `ncclMemAlloc` to enable NVLink Switch Reductions (NVLS) and reduce contention between compute and communication kernels.
   **Evidence**: [sources/pytorch-cuda-semantics/full.md#mixing-different-cuda-system-allocators-in-the-same-program](../../sources/pytorch-cuda-semantics/full.md#mixing-different-cuda-system-allocators-in-the-same-program)

7. **Claim**: NVLS-oriented allocator/mempool paths have sharp requirements and risks: buffer compatibility constraints can be broken by dynamic workloads (leading to fallback algorithms), and `ncclMemAlloc` may allocate more memory than requested due to alignment requirements (risking OOM).
   **Evidence**: [sources/pytorch-cuda-semantics/full.md#mixing-different-cuda-system-allocators-in-the-same-program](../../sources/pytorch-cuda-semantics/full.md#mixing-different-cuda-system-allocators-in-the-same-program)

8. **Claim**: CUDA graphs trade dynamic flexibility for reduced CPU overhead: capture records kernels + fixed arguments (including pointer addresses) and replay launches the same work repeatedly; PyTorch’s caching allocator uses a graph-private memory pool during capture to preserve address stability across replays.
   **Evidence**: [sources/pytorch-cuda-semantics/full.md#cuda-graphs](../../sources/pytorch-cuda-semantics/full.md#cuda-graphs), [sources/pytorch-cuda-semantics/full.md#graph-memory-management](../../sources/pytorch-cuda-semantics/full.md#graph-memory-management)

9. **Claim**: Multi-stream capture requires an explicit stream DAG that branches out from and rejoins the initial capturing stream; otherwise the capture is incorrect (example provided in the doc).
   **Evidence**: [sources/pytorch-cuda-semantics/full.md#usage-with-multiple-streams](../../sources/pytorch-cuda-semantics/full.md#usage-with-multiple-streams)

## API surface / configuration

**Core stream + sync primitives:**
- `torch.cuda.synchronize()`, `torch.cuda.Event`
- `torch.cuda.Stream`, `torch.cuda.stream(...)`, `torch.cuda.current_stream()`, `torch.cuda.default_stream(...)`
- `torch.cuda.Stream.wait_stream(...)`, `torch.Tensor.record_stream(...)`

**Allocator + memory knobs:**
- `PYTORCH_ALLOC_CONF=...` (notably `max_split_size_mb`, `garbage_collection_threshold`, `expandable_segments`, …)
- `torch.cuda.memory.CUDAPluggableAllocator`, `torch.cuda.memory.change_current_allocator(...)`
- `torch.cuda.MemPool`, `torch.cuda.use_mem_pool(...)`

**CUDA graphs:**
- `torch.cuda.CUDAGraph`, `torch.cuda.graph(...)`, `torch.cuda.make_graphed_callables(...)`

## Actionables / gotchas

- **Treat “current stream” as part of the distributed contract**: compile-aware collectives only help if the program order and stream semantics match across ranks. If one rank runs an op on a side stream (or hits an implicit sync) while another rank doesn’t, you can induce ordering divergence and hangs. See: `scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md` (Q6); `scope-drd/notes/FA4/h200/tp/bringup-run-log.md` Run 12b (functional collectives fixed compile, 9.6→24.5 FPS).
- **If you introduce non-default streams, do it deliberately**: follow the doc’s `wait_stream` + `record_stream` pattern; lack of synchronization can be wrong even when there’s “no dependency” due to caching allocator reuse.
- **Profile with correct synchronization**: use CUDA events or explicit `torch.cuda.synchronize()` around timers; otherwise you’ll under-measure GPU time and misattribute overlap. Use `CUDA_LAUNCH_BLOCKING=1` for debugging stack traces when a CUDA error shows up “later”.
- **Allocator tuning is a latency lever, not just a memory lever**: `garbage_collection_threshold` is designed to reduce allocator stalls from expensive reclaim paths in server-like workloads; worth considering once correctness is stable and you have tail-latency evidence.
- **Be conservative with NVLS + custom allocator experiments**: `torch.cuda.MemPool` + `ncclMemAlloc` is a powerful tool, but it can change memory footprint and trigger NCCL fallback algorithms if buffers don’t meet requirements; treat it as an opt-in experiment for later phases, not bringup-default.
- **CUDA graphs require stable addresses**: graph-private pools keep addresses stable, but sharing pools across captures can clobber outputs unless you clone them; don’t enable graph-based modes unless your pipeline is graph-safe (static shapes/control flow) and you’ve audited output lifetimes.

## Related resources

- [nccl-user-guide](nccl-user-guide.md) -- NCCL stream semantics, group ordering, and graph-capture caveats
- [cuda-graphs-guide](cuda-graphs-guide.md) -- CUDA graphs detailed in NVIDIA programming guide
- [making-dl-go-brrrr](making-dl-go-brrrr.md) -- overhead-bound vs memory-bound vs compute-bound regimes
- [dynamo-deep-dive](dynamo-deep-dive.md) -- torch.compile interacts with CUDA graphs via reduce-overhead mode
