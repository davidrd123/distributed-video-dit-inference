---
status: draft
---

# Topic 8: Kernel launch overhead — Python-to-GPU dispatch path

Each PyTorch operator call traverses Python -> C++ dispatch -> CUDA kernel launch. At **~10us per launch**, this overhead dominates when running many small operations. `torch.compile` and CUDA graphs are the primary mitigations.

## Resources

<!-- Resource IDs from manifest belonging to this topic -->
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| making-dl-go-brrrr | Making Deep Learning Go Brrrr From First Principles | high | condensed |
| gpu-mode-lecture-6 | GPU MODE Lecture 6: Optimizing Optimizers in PyTorch | low | link_only |
| pytorch-internals-ezyang | PyTorch internals | medium | pending |

## Implementation context

The clearest real-world example of launch/dispatch overhead dominating is TP=2 + compile pre-fix: `torch._dynamo.disable()` around ~160 collectives caused ~160 graph breaks per forward and throughput collapsed to **9.6 FPS** (Runs 8-9b) even after eliminating the `_kv_bias_flash_combine` `Tensor.item()` break. Switching to functional collectives eliminated collective graph breaks and restored fusion (Run 12b: **24.5 FPS**, “single compiled graph per block” behavior).

See: `refs/implementation-context.md` → Phase 1, `scope-drd/notes/FA4/h200/tp/bringup-run-log.md` (Runs 8-12b).

## Synthesis

<!-- To be filled during study -->

### Mental model

Kernel launch overhead is the fixed CPU-side cost you pay every time you ask the GPU to do *anything*: Python bytecode → PyTorch dispatcher → C++ → CUDA runtime/driver → kernel launch. When kernels are “fat” (e.g., GEMMs), this fixed cost is negligible. When kernels are “tiny” (pointwise ops, small reductions, bookkeeping kernels), the fixed cost dominates and you enter the **overhead-bound** regime: the GPU is fast enough, but it spends time idle because the CPU can’t feed it work fast enough.

Three practical levers reduce launch/dispatch overhead:

1. **Fuse ops (fewer kernels)**: `torch.compile` (Dynamo + Inductor) traces Python and generates fused kernels so the GPU does more work per launch.
2. **Replay a captured DAG (amortize launch costs)**: **CUDA graphs** define a fixed sequence/DAG of GPU work once and replay it with much lower per-iteration CPU overhead.
3. **Eliminate Python between ops**: compilation removes Python dispatch between individual operators *as long as* you don’t hit **graph breaks** (which drop you back to CPython between compiled segments).

In Scope’s TP=2 bringup, this was not theoretical: pre-fix, `torch._dynamo.disable()` around collectives caused **~160 graph breaks/forward** and throughput collapsed to **9.6 FPS** (Runs 8–9b). The funcol fix removed that break source, restoring large compiled regions and throughput (**24.5 FPS**, Run 12b). This is “launch overhead” showing up as “Python dispatch between many tiny regions.”

### Key concepts

- **Overhead-bound vs bandwidth-bound vs compute-bound**: the “brrr” framework is the fastest way to decide what matters. If you’re overhead-bound, kernel fusion / graphs dominate; if you’re bandwidth-bound, fuse to reduce memory traffic; if you’re compute-bound, focus on kernel math/Tensor Cores and algorithmic FLOPs.
- **Asynchronous execution hides overhead until it doesn’t**: PyTorch can enqueue work ahead of the GPU, but only if the CPU stays ahead. A long chain of tiny ops causes GPU “bubbles” (idle gaps) even though total FLOPs are small.
- **Graph breaks (Dynamo)**: Dynamo traces Python into FX graphs, but any unsupported/escaped Python path inserts a **graph break**, splitting execution into multiple compiled graphs with CPython between them. Graph breaks are the main “compile succeeded but got slower” footgun.
- **Guards + recompiles**: even without breaks, guard failures can trigger recompilation and churn CPU time (especially if shapes/flags change).
- **Operator/kernel fusion (Inductor)**: fusion reduces both kernel launches (overhead) and DRAM round-trips (bandwidth).
- **CUDA graphs (definition → instantiation → execution)**:
  - **Stream capture** records work into a capture graph (work is *not executed* during capture).
  - **Instantiation** produces an executable graph; **execution** replays it cheaply.
  - Multi-stream capture requires explicit event-based fork/join and all streams must rejoin the origin stream.
  - Many operations are **prohibited** during capture (sync/query, some memcpy patterns, legacy stream interactions), and violations invalidate capture.
- **Address/shape stability**: graphs assume stable topology and (often) stable pointer addresses; PyTorch typically uses graph-private memory pools to keep allocations stable across replays.

### Cross-resource agreement / disagreement

These three resources line up as a coherent stack:

- `making-dl-go-brrrr` explains *why* overhead exists and how to diagnose the overhead-bound regime (GPU bubbles, poor scaling with batch size) and motivates fusion as the primary fix.
- `dynamo-deep-dive` explains *how* `torch.compile` reduces overhead (trace → compile) and why graph breaks/guards can reintroduce overhead (CPython between graphs, recompiles on guard failure). It provides the debug knobs to attribute breaks and recompiles (`TORCH_LOGS=graph_breaks`, `guards`, `recompiles`, `bytecode`).
- `cuda-graphs-guide` explains the “lowest overhead” endgame: a captured DAG replayed with minimal CPU involvement. It also explains why graphs are higher risk: strict capture rules, multi-stream dependency requirements, update limitations, and memory/address lifetime constraints.

The main “disagreement” is really a *trade-off boundary*:

- `torch.compile` tolerates more dynamism than CUDA graphs (it can recompile around shape/guard changes), but that dynamism costs CPU time and can fragment execution if you trigger graph breaks.
- CUDA graphs can deliver lower steady-state overhead, but they require a very stable workload and careful capture boundaries; they are not a default bringup tool, especially once you include collectives or control-plane branches.

### Practical checklist

1. **Confirm you’re actually overhead-bound**:
   - Look for GPU idle gaps (“bubbles”) between kernels in a profiler trace.
   - Try the cheap scaling test: if increasing batch size doesn’t increase runtime proportionally, overhead is dominating.
2. **Eliminate graph breaks first (before “tuning kernels”)**:
   - Run with `TORCH_LOGS=graph_breaks` and count/attribute breaks.
   - Treat increases in `graph_breaks` / `unique_graphs` as performance regressions (this is exactly what bit us in Runs 8–9b).
3. **Keep the hot path “compile-friendly”**:
   - Avoid incidental Python side effects inside the compiled region (`print`, logging, Python callbacks).
   - Avoid `.item()`-style host round-trips and dynamic slicing patterns that force breaks/guards.
4. **Use fusion as the default overhead lever**:
   - Prefer `torch.compile` for steady-state denoise/generator regions; ensure distributed collectives inside compiled regions are traceable (functional collectives fixed this for us).
5. **Treat CUDA graphs as a separate milestone**:
   - Capture only a stable, steady-state region with fixed shapes and a known stream DAG.
   - Avoid capturing NCCL/PG lifecycle until shutdown + capture symmetry is proven in our stack.
6. **In distributed mode, enforce symmetry**:
   - All ranks that participate in a collective must take the same compiled/captured path; divergent graph-break/capture behavior can desynchronize collective order → hang.

### Gotchas and failure modes

- **“Compile made it slower” is usually graph fragmentation**: you successfully compiled *many tiny graphs* and paid Python dispatch between them. Graph breaks are the first thing to check.
- **Overhead vs bandwidth confusion**: fusion helps both, but the *reason* differs. If you’re bandwidth-bound, fusion wins by reducing DRAM traffic; if you’re overhead-bound, fusion wins by reducing launches/dispatch. Know which one you’re in to pick the next optimization.
- **Guard churn looks like overhead**: variable shapes/flags can cause recompiles and wipe out fusion benefits even without obvious graph breaks.
- **CUDA graph capture invalidation**: “innocent” sync/query calls (or legacy-stream interactions inside libraries) can invalidate capture. Multi-stream capture fails unless you explicitly encode fork/join with events and rejoin the origin stream.
- **Graphs don’t like dynamic allocation**: if allocations happen inside the captured region or pointer addresses change across replays, you’ll get capture failures or wrong behavior unless you use graph-private pools / graph memory nodes correctly.
- **Distributed + graphs multiplies risk**: capture/launch must be uniform across participating ranks (including whether an op is captured at all). Mismatches often present as deadlocks/hangs rather than clean errors.

### Experiments to run

1. **Overhead diagnosis sweep**: run a small grid over batch size (and/or chunk count) and record throughput; if performance doesn’t scale with work, you’re overhead-bound.
2. **Kernel-count + bubble measurement**: profile one chunk and record total kernel launches and GPU idle time. Repeat for eager vs `torch.compile` to see whether fusion reduced launches and bubbles.
3. **Graph-break attribution**: run with `TORCH_LOGS=graph_breaks` and record the top break sites; fix the biggest break source first, then re-measure `graph_breaks` and throughput.
4. **“Fragmentation regression” test**: intentionally introduce a known break (e.g., a Python-side print/logging op in the hot path) in a micro-repro and confirm throughput collapses—useful for teaching and for validating your diagnostics.
5. **CUDA graph pilot (single-rank, no collectives)**: capture a steady-state compute-only region with fixed shapes; compare per-iteration CPU time and throughput with and without replay. Only after this is stable should you consider multi-rank capture boundaries.
6. **Distributed symmetry test**: on a minimal 2-rank program, deliberately make one rank take a different compile/capture path and observe the resulting hang; this validates the “symmetry is a correctness requirement” intuition before you debug it in the full system.
