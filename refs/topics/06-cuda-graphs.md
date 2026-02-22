---
status: draft
---

# Topic 6: CUDA graphs — what they capture, what breaks them, relation to torch.compile

CUDA graphs eliminate kernel launch overhead by recording a sequence of GPU operations and replaying them. They are **critical for inference latency** but impose strict constraints: all tensor addresses must be fixed, control flow cannot change, and memory allocation patterns must be static.

## Resources

<!-- Resource IDs from manifest belonging to this topic -->
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| cuda-graphs-guide | CUDA Programming Guide: CUDA Graphs | high | condensed |
| pytorch-cuda-semantics | PyTorch CUDA Semantics | high | condensed |
| torch-compile-api | torch.compile API | medium | pending |
| nccl-cuda-graphs | Using NCCL with CUDA Graphs | medium | pending |

## Implementation context

Single-GPU optimization on H200 already tried `torch.compile` “reduce-overhead” / CUDAGraph variants, but CUDAGraph capture was reported as **unstable or neutral**. In TP mode, capture becomes higher risk because a mid-capture failure can strand peers at collectives; the v0 roadmap therefore treats “CUDAGraph Trees (non-collective regions only)” as a separate v2.0 milestone and focuses on compile hygiene first (functional collectives + lockstep warmup).

See: `refs/implementation-context.md`, `scope-drd/notes/FA4/h200/tp/streamdiffusion-v2-analysis-opus.md` (What we already tried), `scope-drd/notes/FA4/h200/tp/explainers/07-v0-contract.md` (Roadmap: v2.0 CUDAGraph Trees).

## Synthesis

<!-- To be filled during study -->

### Mental model

CUDA graphs are a **CPU-overhead optimization** for repetitive GPU workloads.

- **Without graphs**: the CPU launches kernels one by one; each launch has overhead (Python dispatch, driver runtime, scheduling), which becomes visible when the GPU work is many small kernels or when you’re already close to “overhead-bound”.
- **With graphs**: you *capture* a fixed sequence of GPU operations once, then *replay* that exact sequence many times with a single (cheap) launch.

The trade is **flexibility → speed**:

- Graph replay assumes the same “program”: same kernel set and ordering, same control flow, and (critically) the same **tensor addresses / pointer arguments**.
- In practice, this pushes you toward “static” execution: fixed shapes, fixed allocation behavior, and avoiding runtime-dependent branching inside the captured region.

This is why CUDA graphs and `torch.compile` interact: `torch.compile` reduces Python overhead by compiling regions into fewer kernels, and “reduce-overhead” style modes often go one step further by wrapping the compiled region in CUDA graph capture/replay to minimize per-iteration launch overhead.

For **pipeline-parallel inference**, the key question is: *can the per-stage forward pass be made graph-capturable?* The usual blockers are the things we already fight for compile stability: dynamic shapes/sequence lengths and dynamic state updates (notably KV-cache slicing/rolling). Our current known graph-break source (KV-cache dynamic slicing) is also a prime suspect for “graph capture breaks”, which is why the roadmap parks “CUDAGraph Trees” for v2.0 after compile hygiene is stable.

### Key concepts

- **Graph definition vs execution**: CUDA graphs are a DAG of GPU work; “definition” builds that DAG, “instantiation” creates an executable graph object, and “execution” launches it cheaply and repeatedly. (`cuda-graphs-guide`)
- **Stream capture (capture vs replay)**: capture brackets stream-based code; during capture, operations are recorded into a capture graph instead of being enqueued/executed, then replay runs the recorded work. (`cuda-graphs-guide`, `pytorch-cuda-semantics`)
- **Stream capture mode + prohibited ops**: many “innocent” operations are illegal during capture (sync/query, some synchronous memcpy patterns, legacy stream interactions). Capture can be invalidated and fail at end-capture. (`cuda-graphs-guide`)
- **Address stability**: replay reuses the same pointer arguments; changing tensor storage addresses breaks correctness/capture. This is the core constraint that dominates everything else. (`cuda-graphs-guide`, `pytorch-cuda-semantics`)
- **Graph-private memory pool**: PyTorch uses a graph-private allocator pool during capture so addresses remain stable across replays, even as the caching allocator would normally recycle memory. (`pytorch-cuda-semantics`)
- **Static vs dynamic shapes**: graphs are happiest when shapes (and thus allocation sizes / kernel selection) do not change; Dynamo is “static by default” and dynamic shapes introduce guards/recompiles/graph breaks. (`dynamo-deep-dive`)
- **`make_graphed_callables`**: a PyTorch API that captures callables into CUDA graphs (typically with static input buffers and strict address/shapes assumptions). (`pytorch-cuda-semantics`)
- **Multi-stream capture DAG**: capturing work across multiple streams is possible, but the dependency graph must be explicit (events) and all participating streams must rejoin the origin stream. This matters for “compute stream + comm/memcpy stream” overlap patterns. (`cuda-graphs-guide`, `pytorch-cuda-semantics`)
- **Graph update vs rebuild**: some changes can be applied via graph update if topology is unchanged, but the rules are strict and many changes force rebuilding/capturing anew. (`cuda-graphs-guide`)

### Cross-resource agreement / disagreement

- **Agreement (fundamentals)**:
  - Both NVIDIA’s guide and PyTorch’s semantics agree that a CUDA graph is a recorded dependency DAG of GPU work, and replay is fast because the driver/runtime work is amortized.
  - Both agree that **address stability** is the hard constraint: if pointers change, the replayed graph is no longer “the same program”.
  - Both describe multi-stream capture as “allowed but structured”: you need an explicit fork/join DAG, not ad-hoc concurrency.
- **Difference (abstraction level)**:
  - `cuda-graphs-guide` is the “driver-level truth”: capture invalidation rules, graph instantiation/update limits, conditional nodes, and memory nodes (addresses can be reused/remapped inside the graph’s lifetime).
  - `pytorch-cuda-semantics` is the “user-facing wrapper”: `torch.cuda.CUDAGraph`, graph-private pools, and `make_graphed_callables`, which translate driver constraints into Python-level patterns.
- **Add-on perspective (why most users meet graphs through compile)**:
  - `dynamo-deep-dive` frames the main practical entry point: most people don’t manually capture graphs; they turn on a compile mode that tries to reduce Python/launch overhead, and graphs are one of the tools used under the hood.
  - It also explains why “graphs don’t just work”: dynamic shapes, graph breaks, and guard-driven recompiles are the exact kinds of variability that make capture/replay brittle.

### Practical checklist

- **Pick a capture unit that can be static**: start with “one stage’s forward pass” (or a smaller submodule), not the entire end-to-end streaming loop.
- **Freeze shapes first**: before thinking about graphs, make input shapes/static metadata stable (sequence length, frame chunk size, batch). If shapes vary, expect recapture/rebuild or outright failure.
- **Eliminate dynamic allocations inside the region**: allocate persistent inputs/outputs/workspaces once; prefer in-place updates and copy new data into preallocated buffers.
- **Assume pointer addresses must remain stable**: graph-private pools help, but only if you keep the captured region’s allocation behavior consistent and manage output lifetimes.
- **Treat multi-stream overlap as a capture contract**: if you capture work on multiple streams (compute + comm), the fork/join must be explicit (events) and streams must rejoin the origin stream before end-capture.
- **Keep collectives out of the first captures**: graph capture + NCCL/collectives multiplies failure blast radius (a failed capture can strand peers). Align with the roadmap: “CUDAGraph Trees” = capture non-collective regions only.
- **Test through the “real” entry point**: try `torch.compile` with a reduce-overhead oriented mode on a fixed-shape microbenchmark, then only graduate to larger regions once capture succeeds and improves latency.
- **Validate output lifetime + pool interactions**: if you share pools across captures or reuse outputs, clone/copy results to a safe buffer before the next replay to avoid aliasing/clobbering.

### Gotchas and failure modes

- **Dynamic shapes / dynamic control flow**: any data-dependent branching, variable-length sequence handling, or per-iteration shape change can prevent capture or force frequent recapture/recompile (often worse than no graphs).
- **KV-cache updates are a capture hazard**: slicing/rolling updates can introduce dynamic indexing and dynamic allocation patterns. In our stack this is already a known compile graph-break source, so it’s a top suspect for graph-capture instability.
- **Hidden sync calls invalidate capture**: sync/query on capturing streams/events, synchronous copies in the wrong context, or a library “helpfully” synchronizing for timing can invalidate capture.
- **Multi-stream capture is easy to get subtly wrong**: forgetting the required fork/join structure (or failing to rejoin the origin stream) can fail capture or record a dependency graph you didn’t intend.
- **Allocator / lifetime issues can become correctness bugs**:
  - Graph replay reuses pointer addresses; if an address no longer points to the intended storage (or has been recycled), you can get silent corruption.
  - Graph-private pools reduce this risk but introduce new ones: output buffers can alias across replays/captures unless you manage lifetimes explicitly.
- **Collectives inside graphs are high-risk**: capture must be identical across ranks (same ops/order). If one rank diverges (shape/branch/exception), you can deadlock at collectives or hang on shutdown.
- **“No win” scenarios are common**: if the workload is dominated by a few large kernels (compute/memory bound), graphs may be neutral; they mainly help when launch overhead/dispatch dominates.

### Experiments to run

- **Single-stage capture smoke test**: choose a single pipeline stage forward pass with fixed shapes; attempt capture/replay and confirm numerically identical outputs across replays.
- **Compile-mode comparison**: benchmark eager vs `torch.compile` vs `torch.compile` with reduce-overhead oriented settings (if available) and measure time/iter and variance (p50/p95).
- **Fixed vs variable sequence length A/B**: run the same stage with a fixed sequence length (expected best-case) and with varying length (expected failure/recapture) to quantify how brittle capture is for our workload.
- **Allocation audit**: instrument/observe allocations during the region (before/after capture) to verify that replay does not allocate. If allocations appear, identify which op is causing them.
- **Multi-stream capture probe**: reproduce a minimal “compute stream + side stream” DAG (e.g., an async memcpy or comm-like op) and verify the required fork/join event pattern works under capture.
- **Distributed safety staging**: only after single-GPU success, test capture on a multi-rank setup with *no collectives inside the captured region* to reduce deadlock risk; then decide whether it’s worth pursuing graph capture for PP stages at all (given the current “unstable/neutral” observation on H200).
