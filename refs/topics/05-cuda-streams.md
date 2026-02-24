---
status: draft
---

# Topic 5: CUDA streams — execution/dependency model, events, synchronization, NCCL interaction

Streams are the fundamental concurrency primitive for overlapping compute, communication, and memory transfers. NCCL ops are launched on a CUDA stream and are stream-ordered locally; once you add non-default streams (or cross-thread comms), you must make stream/event ordering explicit to stay correct.

## Resources

<!-- Resource IDs from manifest belonging to this topic -->
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| pytorch-cuda-semantics | PyTorch CUDA Semantics | high | condensed |
| cuda-async-execution | CUDA Programming Guide: Asynchronous Execution | medium | pending |
| leimao-cuda-stream | CUDA Stream | low | pending |

## Implementation context

CUDA streams show up in two load-bearing places: compile-aware collectives (Run 12b) require correct stream ordering when collectives are traced in-graph, and overlap experiments depend on explicit event/stream waits. The parked async-decode-overlap plan proposes launching VAE decode on a separate CUDA stream to hide ~3–10ms/chunk (≈2–8% ceiling) behind the next chunk’s denoise, but warns overlap collapses under semantics-preserving recompute (R0a). StreamDiffusionV2’s design similarly uses a dedicated comm stream plus waits to overlap stage transfers with compute.

See: `refs/implementation-context.md` → Phase 1-2, `scope-drd/notes/FA4/h200/tp/bringup-run-log.md` (Run 12b), `scope-drd/notes/FA4/h200/tp/async-decode-overlap-impl-plan.md` (CUDA stream overlap), `scope-drd/notes/FA4/h200/tp/streamdiffusion-v2-analysis-opus.md` (comm stream overlap).

## Synthesis

<!-- To be filled during study -->

### Mental model

CUDA streams are ordered work queues on a single GPU: operations enqueued on the *same* stream execute in program order; operations on *different* streams may overlap, but only if you explicitly express dependencies. PyTorch defaults to “do everything on the current (often default) stream,” which is safe and simple but implicitly serializes work. (See `pytorch-cuda-semantics` claims 1–2.)

The overlap progression that matters for Scope’s PP bringup is:

1. **Default stream = implicit serialization (safe baseline)**: if all compute, comm, and decode run on the same stream, correctness is easy, but you can’t hide “side work” (VAE decode, p2p transfers) behind the next chunk’s denoise.

2. **Non-default streams = explicit parallelism (explicit sync responsibility)**: once you launch work on a side stream (e.g., rank0 decode stream, comm stream), *you* must add the stream DAG edges (waits/events) and the memory-lifetime edges (allocator safety via `record_stream`). Otherwise you can be wrong even when there is “no direct dependency” because memory can be recycled by the caching allocator while prior work is still pending. (See `pytorch-cuda-semantics` claims 2–3.)

3. **NCCL adds “distributed ordering” on top of stream ordering**: NCCL collectives are launched onto a CUDA stream and return once enqueued; completion is still governed by normal CUDA mechanisms (stream sync / events). Correctness now depends on (a) per-stream dependency correctness on each rank *and* (b) ranks issuing collectives in consistent order. NCCL group semantics can also introduce unexpected serialization when multiple streams are mixed within a single group. (See `nccl-user-guide` claims 1–4.)

In PP0 (rank0-out-of-mesh), the “big overlap” is inter-GPU: rank0 encodes/decodes while the mesh runs the generator. Streams matter for the next layer of optimization: overlapping rank0’s VAE decode on a side stream and overlapping p2p stage-boundary transfers on a dedicated comm stream. (See `refs/implementation-context.md` → Phase 3; and `pytorch-cuda-semantics` Actionables on treating current stream as part of the distributed contract.)

### Key concepts

- **Stream (CUDA / PyTorch)**: ordered execution context on a GPU. PyTorch APIs expose `torch.cuda.Stream()` and “current stream” via `torch.cuda.current_stream()`. Ops follow the current stream unless you enter a `torch.cuda.stream(s)` context. (See `pytorch-cuda-semantics` claim 2.)
- **Default stream**: the implicit baseline stream; keeping everything here is the “safe default,” but it serializes otherwise-parallel work. PyTorch handles some synchronization implicitly on the default stream, but you should not expect that to extend to arbitrary multi-stream setups. (See `pytorch-cuda-semantics` claim 2.)
- **Events and stream waits**: events are the normal way to express “producer stream finished before consumer stream proceeds.” PyTorch’s high-level primitive is `Stream.wait_stream(other_stream)` (establishes that the current stream waits for all prior work on `other_stream`). (See `pytorch-cuda-semantics` claim 2.)
- **`Tensor.record_stream(stream)` (allocator lifetime edge)**: tells the caching allocator that a tensor’s storage is “in use” by `stream` so the allocator won’t recycle the memory until that stream’s work completes. This is necessary even when you *did* add the right compute dependency, because allocator reuse can violate correctness without an explicit lifetime edge. (See `pytorch-cuda-semantics` claim 3.)
- **Backward stream semantics (mostly training)**: backward CUDA ops run on the same stream(s) as their corresponding forward ops; mixing stream contexts around `backward()` can be safe or unsafe depending on whether you inserted the needed synchronization. This matters if we ever train or do gradient-based adapters in a multi-stream environment. (See `pytorch-cuda-semantics` claim 4.)
- **NCCL stream semantics**: NCCL ops are launched on a CUDA stream and execute asynchronously; ordering is stream-ordered locally but must also match across ranks to avoid hangs/incorrectness. (See `nccl-user-guide` claim 1 and claim 4.)
- **NCCL group calls (and multi-stream footgun)**: `ncclGroupStart/End` can aggregate ops and reduce launch overhead, but mixing multiple streams within the same group can force cross-stream dependencies and block all involved streams until the NCCL kernel completes (accidental global sync point). (See `nccl-user-guide` claim 2 and claim 3.)

### Cross-resource agreement / disagreement

**Agreement:**
- “CUDA is async by default”: CPU timing lies unless you synchronize or use events. (See `pytorch-cuda-semantics` claim 1; `nccl-user-guide` claim 1.)
- “Stream order is the core correctness contract”: ops are ordered within a stream; cross-stream correctness requires explicit sync. (See `pytorch-cuda-semantics` claim 2; `nccl-user-guide` claim 1.)

**Different emphasis (complementary):**
- `pytorch-cuda-semantics` is strongest on **single-process correctness hazards**: explicit `wait_stream` edges for data dependencies, and explicit `record_stream` lifetime edges to keep the caching allocator from reusing memory too early. (See `pytorch-cuda-semantics` claims 2–3.)
- `nccl-user-guide` is strongest on **multi-rank correctness hazards**: collective ordering semantics across ranks, group-call ordering rules, and thread-safety constraints. It also highlights a counterintuitive performance trap: mixing multiple streams inside the same NCCL group can serialize streams. (See `nccl-user-guide` claims 2–5.)

**Clarification (common confusion):**
- It’s more precise to think “NCCL ops are *launched on a stream* and are async” than “NCCL runs on its own streams.” You pick the launch stream; stream ordering then defines local dependencies, and you use normal CUDA sync (events/stream sync) to observe completion. (See `nccl-user-guide` claim 1.)

### Practical checklist

This is a bringup-oriented checklist for adding streams safely while pursuing PP overlap (rank0 encode/decode ↔ mesh generator; then comm overlap).

1. **Start with the single-stream baseline** (correctness first): keep compute/comm/decode on the current/default stream, verify correctness and stable ordering across ranks. Then introduce side streams one at a time. (See `pytorch-cuda-semantics` claim 2; `nccl-user-guide` claim 4.)

2. **Name your stream topology explicitly**:
   - `compute_stream`: main generator/denoise work (often current/default stream)
   - `decode_stream` (rank0-only): VAE decode work you want to hide
   - `comm_stream`: p2p stage-boundary transfers / collectives you want to overlap

3. **Add explicit dependency edges with `wait_stream` (or events)**:
   - If `decode_stream` produces frames that will be consumed on `compute_stream`, make the consumer stream wait.
   - If `comm_stream` sends/receives tensors that compute will use, make compute wait before use (and ensure the producer side doesn’t overwrite buffers early). (See `pytorch-cuda-semantics` claim 2.)

4. **Add explicit lifetime edges with `record_stream` on cross-stream tensors**:
   - Any tensor whose storage is produced on one stream and consumed on another must be “recorded” on the consumer stream (or otherwise kept alive) to prevent allocator reuse bugs. This is especially important for staging buffers and send/recv buffers used by async comm. (See `pytorch-cuda-semantics` claim 3.)

5. **Keep NCCL ordering boring and uniform across ranks**:
   - Maintain identical collective call order across ranks/communicators.
   - Avoid issuing ops to the same communicator concurrently from multiple threads.
   - Avoid mixing multiple streams inside a single `ncclGroupStart/End` group unless you explicitly want the cross-stream dependency it introduces. (See `nccl-user-guide` claims 2, 4, 5.)

6. **Profile with correct timing primitives**:
   - Use CUDA events or explicit sync for timing; avoid interpreting wall-clock timings without synchronization.
   - Use `CUDA_LAUNCH_BLOCKING=1` only for debugging (it will destroy overlap). (See `pytorch-cuda-semantics` claim 1.)

7. **Treat “current stream” as part of the distributed contract under compile**:
   - Compile-aware collectives (functional collectives) work when stream ordering is consistent and traceable; introducing side streams inside compiled regions can induce ordering divergence or force graph breaks if not engineered carefully. (See `pytorch-cuda-semantics` Actionables; `refs/implementation-context.md` Phase 1 Run 12b notes.)

8. **PP1 Stage0 overlap pattern (thread + `comm_stream` + CUDA events) is an ordering contract**:
   - In PP1 overlap, `dist.send/recv` correctness depends on the calling thread’s current stream. The safe pattern is: a dedicated comms thread owns *all* `PPControlPlane.send_infer/recv_result` calls and runs them inside `with torch.cuda.stream(comm_stream): ...`.
   - Use explicit ready/done events:
     - main thread produces tensors → `ready_evt.record(main_stream)`
     - comm thread does `comm_stream.wait_event(ready_evt)` → `send/recv` → `done_evt.record(comm_stream)`
     - main thread does `main_stream.wait_event(done_evt)` before touching outputs
   - Add allocator-lifetime edges: after `wait_event(done_evt)`, call `res.latents_out.record_stream(main_stream)` before decoding/consuming.
   - Enforce “single-owner transport”: debug endpoints and smoke tests must not call `PPControlPlane` directly while overlap is active (route through Stage0 API + lock, or disable). See `scope-drd/notes/FA4/h200/tp/explainers/12-pp1-rank0-stage0-state-machine.md` and `deep-research/2026-02-23/pp1-g1c-overlap/reply.md`.

### Gotchas and failure modes

- **Phantom speedups / wrong profiling**: timing without synchronization measures “enqueue time,” not GPU time; it will overstate overlap and understate latency. (See `pytorch-cuda-semantics` claim 1.)
- **Silent overlap killers**: any op that synchronizes the device (explicit sync, certain host transfers, `.item()`-like scalar reads) collapses stream overlap and can create misleading “it didn’t help” results.
- **Use-before-ready bugs**: launching producer/consumer on different streams without a `wait_stream` edge can yield nondeterministic corruption (it may “work” until load changes).
- **Allocator reuse corruption**: even with correct compute ordering, failing to add `record_stream` can allow the caching allocator to recycle memory still in use by another stream. Symptoms range from rare flickers to hard crashes. (See `pytorch-cuda-semantics` claim 3.)
- **Accidental global synchronization via NCCL groups**: mixing multiple streams within one `ncclGroupStart/End` group can block all involved streams until completion (serialization point you didn’t intend). (See `nccl-user-guide` claim 2.)
- **Distributed hangs from ordering divergence**: mismatched collective order across ranks (or multi-threaded communicator misuse) can hang “forever.” Treat ordering as a first-line debug axis. (See `nccl-user-guide` claim 4 and claim 5.)
- **Overlap collapses under semantic coupling**: if later compute depends on decoded pixels (e.g., semantics-preserving recompute paths), `decode_stream` overlap may be limited because the next step must wait for decode anyway. This is why PP bringup starts with recompute disabled (R1) and treats recompute coupling as a design decision. (See `refs/implementation-context.md` → Phase 3.)

### Experiments to run

1. **Stream correctness micro-repro**:
   - Produce a tensor on `stream_a`, consume it on `stream_b` without any waits (expect flakiness), then fix with `wait_stream`.
   - Then add a version that stresses allocator reuse (many allocations/frees) and show why `record_stream` is needed even when dependencies “look right.” (See `pytorch-cuda-semantics` claims 2–3.)

2. **Decode overlap validation (rank0-only)**:
   - Run a controlled A/B where decode runs on the compute stream vs a dedicated decode stream.
   - Verify overlap in a profiler trace (kernels on distinct streams overlapping in time) and quantify the ceiling (single-digit ms per chunk in existing notes). (See `refs/implementation-context.md` and `scope-drd/notes/FA4/h200/tp/async-decode-overlap-impl-plan.md`.)

3. **Comm-stream overlap validation (PP boundary or TP comm)**:
   - Issue p2p transfers / collectives on a comm stream and overlap with compute; verify that correctness holds (wait edges) and that overlap improves the measured critical path.
   - For NCCL, test both “plain calls on one stream” and “grouped calls,” and observe the serialization behavior when mixing streams within a group. (See `nccl-user-guide` claim 2 and claim 3.)

4. **Ordering invariants under compile-aware collectives**:
   - Under `torch.compile`, validate that collectives are traced without graph breaks and that stream ordering stays identical across ranks (the “stream is part of the contract” rule).
   - If experimenting with side streams, add explicit assertions/instrumentation to detect ordering divergence early. (See `refs/implementation-context.md` Run 12b notes; `pytorch-cuda-semantics` Actionables.)
