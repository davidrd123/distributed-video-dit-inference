# CUDA Programming Guide: CUDA Graphs

| Field | Value |
|-------|-------|
| Source | https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html |
| Type | docs |
| Topics | 6 |
| Status | condensed |

## Why it matters

CUDA graphs eliminate repeated kernel-launch overhead by defining a DAG of GPU work once and replaying it with low CPU overhead. They pay off when the workload is repetitive with stable topology; when addresses/parameters change you must use graph update (or rebuild), and stream capture has sharp “prohibited operation” rules (sync/status queries, legacy stream interactions, etc.). This guide is the NVIDIA-level ground truth for stream capture, multi-stream dependencies, graph update constraints, graph-owned allocations, and device-launched graphs.

## Key sections

- [Stream capture](../../sources/cuda-graphs-guide/full.md#42212-stream-capture) — capture semantics (work is *not* enqueued), origin-stream rule, and what can/can’t happen during capture.
- [Cross-stream dependencies and events](../../sources/cuda-graphs-guide/full.md#422121-cross-stream-dependencies-and-events) — how to include a side stream in capture and the required re-join back to the origin stream.
- [Prohibited and unhandled operations](../../sources/cuda-graphs-guide/full.md#422122-prohibited-and-unhandled-operations) + [Invalidation](../../sources/cuda-graphs-guide/full.md#422123-invalidation) — the “why did capture fail?” checklist (sync APIs, legacy stream, capture-graph merging mistakes).
- [Graph instantiation](../../sources/cuda-graphs-guide/full.md#4222-graph-instantiation) + [Graph execution](../../sources/cuda-graphs-guide/full.md#4223-graph-execution) — template vs executable graph and launch model.
- [Updating instantiated graphs](../../sources/cuda-graphs-guide/full.md#423-updating-instantiated-graphs) — what you can update in-place (and the ordering invariants it relies on).
- [Conditional graph nodes](../../sources/cuda-graphs-guide/full.md#424-conditional-graph-nodes) — device-evaluated `if/while/switch`, and the tight body-graph restrictions.
- [Graph memory nodes](../../sources/cuda-graphs-guide/full.md#425-graph-memory-nodes) — GPU-ordered allocation/free, fixed virtual addresses, reuse/remap behavior, and `cudaGraphUpload`.
- [Device graph launch](../../sources/cuda-graphs-guide/full.md#426-device-graph-launch) — launching graphs from the GPU (fire-and-forget vs tail vs sibling) and “execution environments”.
- [CUDA user objects](../../sources/cuda-graphs-guide/full.md#428-cuda-user-objects) — lifetime management for resources referenced by asynchronous graph execution.

## Core claims

1. **Claim**: CUDA Graphs separate **definition**, **instantiation**, and **execution**. Instantiation produces an executable graph and amortizes much of the per-launch setup cost, so repeated launches have lower CPU overhead than submitting each operation to streams.
   **Evidence**: [sources/cuda-graphs-guide/full.md#42-cuda-graphs](../../sources/cuda-graphs-guide/full.md#42-cuda-graphs), [sources/cuda-graphs-guide/full.md#422-building-and-running-graphs](../../sources/cuda-graphs-guide/full.md#422-building-and-running-graphs)

2. **Claim**: A CUDA graph is a dependency DAG of node operations; node types include kernels, memcpy/memset, host function calls, CUDA event record/wait, external semaphore signal/wait, conditional nodes, memory nodes, and child graphs.
   **Evidence**: [sources/cuda-graphs-guide/full.md#42111-node-types](../../sources/cuda-graphs-guide/full.md#42111-node-types)

3. **Claim**: Stream capture creates a *capture graph* by bracketing stream-based code with `cudaStreamBeginCapture()` / `cudaStreamEndCapture()`. While in capture mode, work launched into the stream is **not enqueued for execution**; it is appended to the capture graph and returned at end-capture.
   **Evidence**: [sources/cuda-graphs-guide/full.md#42212-stream-capture](../../sources/cuda-graphs-guide/full.md#42212-stream-capture)

4. **Claim**: Multi-stream capture is supported only when cross-stream dependencies are expressed via `cudaEventRecord()` and `cudaStreamWaitEvent()` *within the same capture graph*. All streams participating in the capture graph must be **rejoined to the origin stream** before `cudaStreamEndCapture()`; failing to rejoin causes capture failure.
   **Evidence**: [sources/cuda-graphs-guide/full.md#422121-cross-stream-dependencies-and-events](../../sources/cuda-graphs-guide/full.md#422121-cross-stream-dependencies-and-events)

5. **Claim**: During capture, many operations are invalid: synchronizing/querying a capturing stream or captured event is illegal; using the legacy stream can become invalid; synchronous APIs (e.g., `cudaMemcpy()`) can be invalid; attempting invalid operations **invalidates** the capture graph and `cudaStreamEndCapture()` returns an error and a NULL graph.
   **Evidence**: [sources/cuda-graphs-guide/full.md#422122-prohibited-and-unhandled-operations](../../sources/cuda-graphs-guide/full.md#422122-prohibited-and-unhandled-operations), [sources/cuda-graphs-guide/full.md#422123-invalidation](../../sources/cuda-graphs-guide/full.md#422123-invalidation)

6. **Claim**: Updating an instantiated graph (`cudaGraphExec_t`) is supported when topology is unchanged, but update requires deterministic pairing between “original” and “updating” graphs, including consistent ordering of API calls and dependency arrays; there are explicit limitations on what may be updated per node type.
   **Evidence**: [sources/cuda-graphs-guide/full.md#4231-whole-graph-update](../../sources/cuda-graphs-guide/full.md#4231-whole-graph-update), [sources/cuda-graphs-guide/full.md#4234-graph-update-limitations](../../sources/cuda-graphs-guide/full.md#4234-graph-update-limitations)

7. **Claim**: Conditional nodes evaluate their condition value on-device and can represent `if`, `while`, and `switch` inside graphs, but the conditional **body graph** is restricted (single device; limited node types; no CUDA dynamic parallelism or device graph launch inside the body).
   **Evidence**: [sources/cuda-graphs-guide/full.md#424-conditional-graph-nodes](../../sources/cuda-graphs-guide/full.md#424-conditional-graph-nodes), [sources/cuda-graphs-guide/full.md#4242-conditional-node-body-graph-requirements](../../sources/cuda-graphs-guide/full.md#4242-conditional-node-body-graph-requirements)

8. **Claim**: Graph memory nodes (and captured `cudaMallocAsync`/`cudaFreeAsync`) give graphs GPU-ordered allocation/free semantics with **fixed virtual addresses** over the lifetime of a graph. Addresses can be reused within a graph when lifetimes do not overlap, and physical mappings can be remapped depending on how/where the graph is launched; `cudaGraphUpload()` can move some mapping cost out of first launch if the upload and launch streams match.
   **Evidence**: [sources/cuda-graphs-guide/full.md#4251-introduction](../../sources/cuda-graphs-guide/full.md#4251-introduction), [sources/cuda-graphs-guide/full.md#42522-stream-capture](../../sources/cuda-graphs-guide/full.md#42522-stream-capture), [sources/cuda-graphs-guide/full.md#42531-address-reuse-within-a-graph](../../sources/cuda-graphs-guide/full.md#42531-address-reuse-within-a-graph), [sources/cuda-graphs-guide/full.md#42541-first-launch--cudagraphupload](../../sources/cuda-graphs-guide/full.md#42541-first-launch--cudagraphupload)

9. **Claim**: Device graph launch requires instantiating with `cudaGraphInstantiateFlagDeviceLaunch` and uploading device resources; device launches run in special graph streams (fire-and-forget, tail launch, sibling) and rely on “execution environments” (tail launch provides device-side serial dependency since device-side `cudaStreamSynchronize()` is not available).
   **Evidence**: [sources/cuda-graphs-guide/full.md#426-device-graph-launch](../../sources/cuda-graphs-guide/full.md#426-device-graph-launch), [sources/cuda-graphs-guide/full.md#4262-device-launch](../../sources/cuda-graphs-guide/full.md#4262-device-launch), [sources/cuda-graphs-guide/full.md#426211-graph-execution-environments](../../sources/cuda-graphs-guide/full.md#426211-graph-execution-environments), [sources/cuda-graphs-guide/full.md#42622-tail-launch](../../sources/cuda-graphs-guide/full.md#42622-tail-launch)

10. **Claim**: `cudaGraph_t` objects are not thread-safe; a `cudaGraphExec_t` cannot run concurrently with itself; the stream a graph is launched into provides ordering with other asynchronous work but does not constrain internal parallelism of graph nodes.
   **Evidence**: [sources/cuda-graphs-guide/full.md#427-using-graph-apis](../../sources/cuda-graphs-guide/full.md#427-using-graph-apis)

## API surface / configuration

**Core lifecycle (host-side runtime API):**
- Graph: `cudaGraphCreate`, `cudaGraphDestroy`, `cudaGraphInstantiate`, `cudaGraphInstantiateWithFlags`, `cudaGraphInstantiateWithParams`, `cudaGraphLaunch`, `cudaGraphUpload`, `cudaGraphExecDestroy`
- Update: `cudaGraphExecUpdate`, `cudaGraphNodeSetEnabled`, and the per-node update APIs (e.g., `cudaGraphExecKernelNodeSetParams`, `cudaGraphExecMemcpyNodeSetParams`, …)
- Debugging: `cudaGraphDebugDotPrint`

**Stream capture:**
- Capture: `cudaStreamBeginCapture`, `cudaStreamEndCapture`, `cudaStreamIsCapturing`
- Advanced: `cudaStreamBeginCaptureToGraph`, `cudaStreamGetCaptureInfo`, `cudaStreamUpdateCaptureDependencies`
- Cross-stream deps: `cudaEventRecord`, `cudaStreamWaitEvent` (with `cudaEventWaitExternal` in the edge cases called out by the guide)

**Conditional nodes:**
- Handle: `cudaGraphConditionalHandleCreate`
- Device set: `cudaGraphSetConditional`

**Graph memory nodes / stream-ordered allocations:**
- Node API: `cudaGraphAddNode` with `cudaGraphNodeTypeMemAlloc` / `cudaGraphNodeTypeMemFree`
- Stream-ordered: `cudaMallocAsync`, `cudaFreeAsync` (capturable)
- Footprint: `cudaDeviceGraphMemTrim`
- Relaunch with unfreed allocations: `cudaGraphInstantiateFlagAutoFreeOnLaunch`

**Device graph launch:**
- Instantiate flag: `cudaGraphInstantiateFlagDeviceLaunch` (+ upload flag paths)
- Device launch streams: `cudaStreamGraphFireAndForget`, `cudaStreamGraphTailLaunch`, `cudaStreamGraphFireAndForgetAsSibling`
- Tail self-launch helper: `cudaGetCurrentGraphExec`

**User objects:**
- `cudaUserObjectCreate`, `cudaGraphRetainUserObject`

## Actionables / gotchas

- **Treat CUDA graphs as a Phase-2+ optimization, not a bringup default**: on H200, we already observed “CUDAGraph capture: unstable or neutral” during single-GPU tuning, and our staged plan explicitly treats “CUDAGraph Trees” as a separate high-risk milestone. See: `scope-drd/notes/FA4/h200/tp/feasibility.md` Section 3.4, `scope-drd/notes/FA4/h200/tp/streamdiffusion-v2-analysis-opus.md` (line 37), and `scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md` (Q6 step 4).
- **Don’t graph-capture NCCL (or PG lifecycle) unless you’re willing to own shutdown sharp edges**: CUDA graphs + NCCL has known “hang on teardown” failure modes (e.g., `destroy_process_group()` hangs after CUDA graph capture; PyTorch issue `#115388`). Keep graph capture limited to non-collective regions until this is proven safe for our stack. See: `refs/topics/03-graceful-shutdown.md`, `refs/resources/nccl-user-guide.md`, and [sources/nccl-user-guide/full.md#using-nccl-with-cuda-graphs](../../sources/nccl-user-guide/full.md#using-nccl-with-cuda-graphs).
- **If you do multi-stream overlap, the fork/join must be explicit (events) or capture will fail**: our PP plan relies on non-blocking comms on side streams; capture only works when cross-stream dependencies are expressed via captured events and all participating streams rejoin the origin stream before end-capture. Treat this as a correctness contract, not a perf trick. See: [Cross-stream Dependencies and Events](../../sources/cuda-graphs-guide/full.md#422121-cross-stream-dependencies-and-events) and `scope-drd/notes/FA4/h200/tp/pp-topology-pilot-plan.md` (rank0-out-of-mesh + overlap topology).
- **Rank0-out-of-mesh makes “capture symmetry” easy to violate**: rank0 (Stage 0) should never touch `mesh_pg` collectives, while mesh ranks will. If graphs are introduced, keep capture boundaries role-local (control-plane vs mesh compute) so rank0 doesn’t accidentally participate in a captured collective sequence that only some ranks execute (classic deadlock). See: `scope-drd/notes/FA4/h200/tp/pp-topology-pilot-plan.md` (“Isolation invariant”) and `scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md` (Q6).
- **Capture forbids “innocent” sync calls**: during capture it’s illegal to query/synchronize capturing streams/events or a broader handle that encompasses them; synchronous APIs like `cudaMemcpy()` become invalid in common cases (legacy stream interaction). This bites when a library internally syncs “for timing” or uses the legacy stream. See: [Prohibited and Unhandled Operations](../../sources/cuda-graphs-guide/full.md#422122-prohibited-and-unhandled-operations).
- **Graph memory nodes give stable virtual addresses, but lifetimes are GPU-ordered and addresses can be reused**: don’t stash a pointer and assume it’s globally unique across time; if you read/write outside the allocation’s lifetime you can silently clobber another allocation. Prefer the higher-level “graph-private pool” story in PyTorch, but understand the underlying reuse/remap rules when debugging. See: [Graph Memory Nodes](../../sources/cuda-graphs-guide/full.md#425-graph-memory-nodes) and [Address Reuse within a Graph](../../sources/cuda-graphs-guide/full.md#42531-address-reuse-within-a-graph).
- **Avoid “first launch surprises” by making launch stream stable (or pre-upload)**: mapping for graph allocations can happen at launch and remapping is triggered by changing launch stream / pool trim / relaunch patterns; `cudaGraphUpload()` can front-load mapping cost *if you launch into the same stream*. If we ever graph-capture steady-state denoise loops, keep the launch stream stable across iterations. See: [First Launch / cudaGraphUpload](../../sources/cuda-graphs-guide/full.md#42541-first-launch--cudagraphupload).
- **Conditional nodes exist, but body-graph restrictions are tight**: don’t assume you can “graph” the whole control-plane loop (NOOP/INFER/SHUTDOWN, cache resets, etc.). Use graphs for the steady-state compute region; keep control-plane decisions outside. See: [Conditional Node Body Graph Requirements](../../sources/cuda-graphs-guide/full.md#4242-conditional-node-body-graph-requirements) and `scope-drd/notes/FA4/h200/tp/pp-control-plane-pseudocode.md`.
- **User objects are the escape hatch for async lifetime management**: if any library path uses “create resource, launch async work, destroy via host callback”, capture can break; CUDA user objects let the graph own references and delay destruction until execution completes (but destructor can’t call CUDA). See: [CUDA User Objects](../../sources/cuda-graphs-guide/full.md#428-cuda-user-objects).
- See: `refs/implementation-context.md` (Per-card actionables sharpening guide).

## Related resources

- [pytorch-cuda-semantics](pytorch-cuda-semantics.md) -- PyTorch CUDA graph integration, graph-private memory pools
- [nccl-user-guide](nccl-user-guide.md) -- capturing NCCL operations in CUDA graphs
- [dynamo-deep-dive](dynamo-deep-dive.md) -- torch.compile mode="reduce-overhead" uses CUDA graphs
- [making-dl-go-brrrr](making-dl-go-brrrr.md) -- kernel launch overhead that CUDA graphs eliminate
