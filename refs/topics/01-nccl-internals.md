---
status: draft
---

# Topic 1: NCCL internals — ring vs tree algorithms, NVLink topology, GPU-level all-reduce/send/recv

NCCL's algorithm selection (ring, tree, NVLS, CollNet) is governed by message size, topology, and the `NCCL_ALGO` environment variable. Understanding this layer is essential for diagnosing why your pipeline send/recv calls behave differently across NVLink vs PCIe topologies.

## Resources

<!-- Resource IDs from manifest belonging to this topic -->
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| nccl-user-guide | NCCL User Guide | high | condensed |
| scaling-dl-nccl | Scaling Deep Learning Training with NCCL | medium | pending |
| nccl-tuning | Understanding NCCL Tuning to Accelerate GPU-to-GPU Communication | medium | pending |
| nccl-cuda-graphs | Using NCCL with CUDA Graphs | medium | pending |

## Implementation context

This topic directly informs the TP v0 communication budget: **80 large + 80 tiny all-reduces per chunk, ~9ms total** on 2×H200 NVLink. Each large all-reduce (`[1, 2160, 5120]` BF16, ~21 MiB) takes ~0.113ms. Understanding NCCL algorithm selection for this message size and topology is critical for diagnosing collective overhead.

See: `refs/implementation-context.md` → Phase 1: TP v0, `scope-drd/notes/FA4/h200/tp/feasibility.md` Section 1-2.

## Synthesis

<!-- To be filled during study -->

### Mental model

NCCL is the GPU-side communication engine under PyTorch’s distributed ops. When you call `dist.all_reduce(...)` / `dist.broadcast(...)` / `dist.send(...)` / `dist.recv(...)` on a CUDA tensor with the NCCL backend, PyTorch is (roughly) asking NCCL to:

1. Map that op onto a **communicator** (a specific set of ranks/devices),
2. Choose an **algorithm + protocol** based on **message size** and **topology** (NVLink vs PCIe vs NVSwitch, etc.),
3. Enqueue GPU work on a CUDA **stream**, returning to Python once it’s enqueued (not once it’s complete).

For Scope, there are two distinct NCCL “surfaces” to keep straight:

- **TP (inside `mesh_pg`)**: many `all_reduce`s inside the generator. We currently pay **160 all-reduces per chunk** (80 large + 80 tiny). The large hidden-state reductions tend to be **bandwidth-shaped** (NVLink helps a lot); the tiny reductions tend to be **latency/launch-shaped** (many small collectives can dominate even if each is “fast”).
- **PP (between stages, via `world_pg`)**: a small number of **P2P send/recv** transfers for activation-sized tensors between rank0 (Stage 0) and the mesh leader (Stage 1). Whether those are “cheap” depends on whether the link is NVLink or PCIe, and whether your schedule actually overlaps them with compute.

The most important operational mental model: **a communicator is a lockstep distributed program**. Every rank in the communicator must issue *the same NCCL ops in the same order* (and with compatible shapes/dtypes). If they don’t, NCCL doesn’t “throw an exception” — it usually **hangs**.

### Key concepts

- **Process group (PyTorch) vs communicator (NCCL)**: a `ProcessGroupNCCL` is PyTorch’s wrapper; it creates and manages one or more NCCL communicators for a fixed set of ranks/devices.
- **Collectives vs P2P**:
  - **Collectives** (e.g., `all_reduce`) involve *all* ranks in the communicator and are extremely sensitive to call-order determinism.
  - **P2P** (`send`/`recv`) still requires matching order/partnering, but only among the participants.
- **Algorithm selection (ring vs tree vs NVLS)**: NCCL chooses different algorithms based on message size and topology. A useful heuristic:
  - **Large messages** often behave like “bandwidth problems”.
  - **Tiny messages** often behave like “latency + kernel launch overhead problems”.
  NVLS is a topology- and buffer-dependent path and should be treated as an advanced, opt-in optimization.
- **Protocols (`NCCL_PROTO`)**: controls the transport protocol family (e.g., Simple/LL/LL128). This is usually a “debug/tuning” lever; the NCCL guide explicitly warns against enabling LL128 on unsupported platforms (risk: corruption).
- **Debugging + visibility**:
  - `NCCL_DEBUG`, `NCCL_DEBUG_SUBSYS`, `NCCL_DEBUG_FILE` to see what NCCL is doing.
  - `NCCL_ALGO` / `NCCL_PROTO` to constrain choices (prefer narrowing for diagnosis, not “blind tuning”).
- **Group semantics (`ncclGroupStart/End`)**: used to batch multiple NCCL calls (reduce CPU overhead / coordinate multi-GPU-per-thread patterns). Ordering rules still apply, and mixing multiple CUDA streams within one group can serialize/introduce global dependencies.
- **Stream behavior**: NCCL work is enqueued on a CUDA stream and runs asynchronously. Correctness depends on explicit stream/event ordering when you use non-default streams.
- **Allocator interactions**: even if “there is no data dependency,” stream sync may be required because the caching allocator can reuse addresses that still have pending work. `record_stream` is the key lifetime primitive.
- **Topology detection**: NCCL inspects GPU interconnect and chooses transport/algorithm accordingly; debug logs (`INIT`, `TUNING`) are the first place to verify what topology NCCL thinks it has.

### Cross-resource agreement / disagreement

The NCCL User Guide and PyTorch CUDA semantics are largely consistent, but complementary:

- **Agreement (stream semantics)**: both sources emphasize that GPU work is **asynchronous** and correctness is enforced through **CUDA stream ordering** (and explicit synchronization when you step off the default stream). This is why “it hangs” bugs often reduce to “one rank hit a different stream/order”.
- **NCCL User Guide = the NCCL contract + knobs**: it explains communicator/group ordering semantics, thread-safety constraints, CUDA graph capture caveats, and the practical knobs (`NCCL_DEBUG*`, `NCCL_ALGO`, `NCCL_PROTO`, `NCCL_LAUNCH_ORDER_IMPLICIT`) you use when diagnosing hangs or regressions.
- **PyTorch CUDA Semantics = how NCCL interacts with PyTorch**: it fills in the missing “why” for correctness issues: non-default streams require `wait_stream`/`record_stream`, allocator reuse can force synchronization even without explicit read-after-write dependencies, and CUDA graphs introduce address-stability constraints (graph-private memory pools).

The **NVLS story** is the best example of the “two-layer” picture:

- The NCCL guide tells you *when* NVLS may be selected (topology-dependent) and that it’s an algorithm choice.
- The PyTorch CUDA semantics doc explains the **mechanism** PyTorch exposes (`torch.cuda.MemPool` / `ncclMemAlloc`) for experimenting with NVLS-oriented buffer allocation, and why this is sharp-edged (buffer compatibility constraints; potential extra allocation/alignment overhead → OOM risk).

### Practical checklist

1. **Name the communicator for every op**: write down which ranks participate in `world_pg` vs `mesh_pg`, and which ops are allowed on each. In the PP design (rank0-out-of-mesh), *rank0 must never enter `mesh_pg` collectives*.
2. **Turn on NCCL visibility before tuning knobs**:
   - Start with `NCCL_DEBUG=INFO`.
   - Add `NCCL_DEBUG_SUBSYS=INIT,COLL,TUNING` (and `GRAPH` if doing CUDA graphs).
   - Use `NCCL_DEBUG_FILE=...%h.%p` so multi-process logs don’t interleave.
3. **Verify call-order determinism**:
   - All ranks in a communicator must issue collectives in the same order.
   - Eliminate rank-dependent branches inside collective regions (including “one rank skipped decode” or “one rank early-returned” patterns).
   - If you use multiple communicators concurrently, keep the **host launch order deterministic**, and consider `NCCL_LAUNCH_ORDER_IMPLICIT=1` as a guardrail.
4. **Observe algorithm/protocol selection, don’t guess**: use debug logs to see what NCCL picked for the large `[1,2160,5120]` BF16 all-reduce and for the tiny reductions. Only then consider constraining with `NCCL_ALGO`/`NCCL_PROTO` (and treat `NCCL_PROTO` as “disable suspect” rather than “enable random”).
5. **Stream correctness is part of the distributed contract**:
   - If you introduce side streams for overlap, use `wait_stream` and `record_stream` correctly.
   - Avoid mixing multiple CUDA streams inside one NCCL group unless you intend the resulting cross-stream dependency behavior.
6. **CUDA graphs: treat capture/launch as collective**: if you capture NCCL ops, capture/launch must be uniform across participating ranks; mixing captured and non-captured NCCL has explicit support knobs and can hang if misused.

### Gotchas and failure modes

- **Silent hang from ordering mismatch**: the usual failure mode is not “wrong output,” it’s “stuck forever.” The timeout you eventually see is often far removed from the original mistake.
- **Wrong group membership**: issuing a collective on the wrong process group (e.g., `world_pg` vs `mesh_pg`) can hang immediately or (worse) only under certain schedules.
- **Rank0-out-of-mesh footgun**: once PP splits stage0/stage1, rank0 must not participate in mesh collectives; accidental participation tends to look like a deadlock “inside all_reduce”.
- **Multi-communicator interleaving deadlocks**: if different ranks interleave communicator A and communicator B in different orders, NCCL can deadlock. This is the motivation for `NCCL_LAUNCH_ORDER_IMPLICIT` and for keeping launch order deterministic.
- **Thread safety**: issuing ops to the same communicator from multiple threads is not safe; grouped operations must be issued by a single thread.
- **Accidental global synchronization**: mixing multiple CUDA streams in the same `ncclGroupStart/End` can introduce dependencies that collapse intended overlap (or create surprising stalls).
- **Asynchrony hides the true error site**: because ops are enqueued, the stack trace for an error (or the place you notice a hang) is often not the site of the bug. Use CUDA events/synchronization for timing and narrow reproductions.
- **CUDA graphs with NCCL are easy to mis-capture**: capture/launch must be uniform; mixing captured and uncaptured NCCL has explicit support knobs and failure modes.
- **NVLS/mempool experiments can change memory footprint**: `ncclMemAlloc` may allocate more than requested (alignment), and buffer-compatibility constraints can cause algorithm fallback — treat NVLS as a later-phase experiment, not bringup-default.

### Experiments to run

1. **Baseline “what did NCCL pick?” run**: run a representative chunk with `NCCL_DEBUG=INFO` and `NCCL_DEBUG_SUBSYS=INIT,COLL,TUNING` and record:
   - The algorithm/protocol selected for the large all-reduce (~21 MiB) and for the tiny reductions.
   - Any topology/transport lines indicating NVLink vs PCIe.
2. **Microbench TP all-reduces across sizes**: benchmark the two regimes we actually have:
   - Large `[1,2160,5120]` BF16 all-reduce (bandwidth-shaped).
   - A representative “tiny” reduction tensor (latency/launch-shaped).
   Use CUDA events or explicit `torch.cuda.synchronize()` so timing is real.
3. **Microbench PP P2P**: measure send/recv latency and throughput for activation-sized tensors over the link you actually have (NVLink on the 2×H200 box; PCIe on other hosts).
4. **Failure injection: ordering mismatch**: deliberately mis-order two collectives on one rank in a minimal repro and observe the hang behavior + what the NCCL logs show as the last successful op. Repeat with/without `NCCL_LAUNCH_ORDER_IMPLICIT=1` if multiple communicators are involved.
5. **Multi-stream group pitfall demo**: create a minimal program that uses `ncclGroupStart/End` with ops on two different streams and observe the introduced cross-stream dependency/stall (use profiler events to confirm).
6. **CUDA graph capture sanity check** (only if/when graphs are on the roadmap): capture + replay a step that includes NCCL, verifying that all participating ranks capture/launch uniformly; then test whether captured/non-captured mixing is enabled and safe on your NCCL version.
