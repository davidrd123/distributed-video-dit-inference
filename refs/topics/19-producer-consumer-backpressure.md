---
status: draft
---

# Topic 19: Producer-consumer with backpressure — bounded channels, ring buffers

In a pipeline-parallel inference system, each stage is a producer for the next stage. Without backpressure, a fast producer can overwhelm a slow consumer, causing OOM or unbounded latency. **Bounded queues** are the simplest correct solution: block the producer when the queue is full.

## Resources

<!-- Resource IDs from manifest belonging to this topic -->
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| backpressure-explained | Backpressure explained — the resisted flow of data through software | low | pending |
| dist-systems-young-bloods | Notes on Distributed Systems for Young Bloods | low | pending |
| warpstream-rejection | Dealing with rejection (in distributed systems) | low | pending |

## Implementation context

The PP overlap design uses **bounded queues** with explicit depth limits: `D_in=2` envelopes in-flight toward mesh, `D_out=2` results queued for decode. The pass gate for PP overlap is `OverlapScore >= 0.30` (30% of smaller stage hidden). Without backpressure, rank0 could send envelopes faster than the mesh can consume them, leading to unbounded memory growth or stale results after hard cuts.

See: `refs/implementation-context.md` → Phase 3, `scope-drd/notes/FA4/h200/tp/pp0-bringup-runbook.md` Phase 2, `pp-control-plane-pseudocode.md`.

Relevant Scope code / notes:
- `scope-drd/scripts/pp_two_rank_pipelined.py` (`max_outstanding` bounded in-flight implementation)
- `scope-drd/notes/FA4/h200/tp/pp-topology-pilot-plan.md` (definitions of `D_in`/`D_out` + flush rules on hard cuts)

## Synthesis

### Mental model

- Pipeline parallelism is a producer-consumer problem at two boundaries: Stage 0 produces envelopes for Stage 1, and Stage 1 produces results for Stage 0.
- Overlap is “assembly line” scheduling: rank0 sends work for chunk `k+1` while it decodes chunk `k`, so the mesh can denoise `k+1` in parallel with decode of `k`.
- This only stays stable if both boundaries are backpressured. Otherwise the faster side will accumulate in-flight chunks until you hit OOM (tensor backlog) or unbounded latency (output falling further behind real time).
- The PP0 design uses two bounded queues with explicit capacities:
- `D_in`: envelopes in-flight toward the mesh.
- `D_out`: results queued and waiting for decode on rank0.
- `D_in=D_out=2` is the minimal “double buffer” that enables overlap at both boundaries; depth 1 can be correct but forces strict alternation on one boundary and collapses steady-state overlap.
- Hard cuts (`reset_cache=True`) are discontinuities. The system must flush both bounded queues, bump `cache_epoch`, and restart fill; bounding makes the stale-work window and flush cost finite.

### Key concepts

- **Bounded queues (`D_in`, `D_out`)** (from `pp-topology-pilot-plan.md`):
- `inflight_to_mesh` with capacity `D_in`: envelopes waiting to be consumed by Stage 1.
- `ready_for_decode` with capacity `D_out`: results waiting to be decoded by Stage 0.
- **Backpressure rules (simple and safe)** (from `pp-topology-pilot-plan.md`):
- Rank0 should not enqueue a new envelope if `ready_for_decode` is full.
- The mesh naturally backpressures because sending results to rank0 can block; the leader must validate before any mesh collective to avoid stranding.
- **Why depth 2 is the minimum for overlap** (double-buffer reasoning from `pp-topology-pilot-plan.md`):
- `D_out=2` lets the mesh produce result `N` while rank0 is decoding `N-1` (decode overlap).
- `D_in=2` lets rank0 pre-stage envelope `N+1` while the mesh works on `N` (fill at the input boundary).
- `D_in=D_out=1` can work, but it forces strict alternation and removes the “both sides stay busy” steady state.
- **Overlap metrics (make the claim falsifiable)** (from `pp-next-steps.md`):
```text
period_k     = tEmit[k] - tEmit[k-1]              # wall time between emitted chunks
stage0_k     = A_time + C_time                    # rank0 encode/build + decode/output
stage1_k     = B_time                             # mesh denoise/generator-only
hidden_k     = max(0, stage0_k + stage1_k - period_k)
OverlapScore = median(hidden_k / min(stage0_k, stage1_k))
pass gate: OverlapScore >= 0.30  (after warmup)
```
- **Overlap proof recipe (no clock sync required)** (from `deep-research/2026-02-22/pp-rank0-out-of-mesh/reply.md`):
  - Rank0 logs monotonic timestamps per chunk: `tA0` (start build), `tA1` (envelope ready after tensor materialization), `tRecv` (result received), `tEmit` (decoded output enqueued), plus queue depths (`len(inflight_to_mesh)`, `len(ready_for_decode)`).
  - Mesh leader reports durations in the result (no synchronized clocks needed): `tB_ms` (Phase B duration) and `t_mesh_idle_ms = max(0, tB0[k] - tB1[k-1])`.
  - Use those to compute:
    - `stage0_k = (tA1[k]-tA0[k]) + (tEmit[k]-tRecv[k])`
    - `stage1_k = tB_ms[k]`
    - `OverlapScore = median(max(0, stage0_k + stage1_k - period_k) / min(stage0_k, stage1_k))`
  - Pass gate is **not just** the score: also require queues stay bounded and `period` trends toward `max(stage0, stage1)` rather than their sum.
- **Period sanity check (no synchronized clocks required)** (from `pp-control-plane-pseudocode.md`):
- With good overlap, observed period approaches `max(Stage0_ms, Stage1_ms)` rather than `Stage0_ms + Stage1_ms`.
- Mesh-leader `t_mesh_idle_ms` should be small; if it grows with decode time, Stage 0 is still serializing the mesh.
- **Hard cuts (`cache_epoch` flush semantics)** (from `pp-topology-pilot-plan.md` and `pp-control-plane-pseudocode.md`):
- Hard cut flushes both queues and increments `cache_epoch`; the next envelope uses `init_cache=True`.
- Envelopes/results are epoch-tagged; rank0 must drop stale-epoch results.
- Control-plane tags derived from `(cache_epoch, call_id, field_id)` prevent cross-chunk message confusion.

#### TP analog (in-flight depth)

`max_outstanding` (used in PP0 pilot code) is the same control knob as “microbatches in flight” in TP/overlap scheduling: it caps outstanding work that is not yet committed to output.
Increasing in-flight depth can improve throughput by reducing bubbles, but it always trades off latency and memory. It also increases the size of the “stale work” window around hard cuts, which is why hard cut semantics and bounded queues must be designed together.

### Cross-resource agreement / disagreement

- The external DS resources in this topic are still `pending`, but the PP design docs already encode the standard consensus: bounded queues are the simplest correct backpressure mechanism for imbalanced pipelines.
- Queue depth is the main policy choice. The PP plan recommends `D_in=D_out=2` as the minimal double buffer; deeper queues can absorb jitter, but in this system they directly increase (1) buffered tensor memory, (2) end-to-end latency, and (3) the amount of stale in-flight work to flush/drop on hard cuts.

### Practical checklist

- **Queue invariants**:
- Maintain `0 <= len(inflight_to_mesh) <= D_in` and `0 <= len(ready_for_decode) <= D_out`.
- Never emit a new envelope when `ready_for_decode` is full (decode backlog is the OOM vector).
- **Epoch invariants (hard cuts)**:
- Every envelope/result includes `cache_epoch`; drop any result whose epoch does not match current.
- Hard cut flushes both queues, increments `cache_epoch`, and restarts fill from an `init_cache=True` envelope.
- **Overlap invariants (steady state)**:
- Rank0 sends envelope `k+1` before decoding result `k` (this is the overlap).
- Mesh leader validates envelope before broadcasting to `mesh_pg` (crash > hang; don’t strand mesh ranks in a collective).
- **What to log (per chunk)**:
- Rank0: monotonic timestamps (`tA0`, `tA1`, `tRecv`, `tEmit`), `period`, queue depths, `cache_epoch`, identifiers (`call_id`, `chunk_index`), and OverlapScore components.
- Mesh leader (returned in result metadata): durations (`tB_ms`) and `t_mesh_idle_ms` (computed leader-local; no clock sync).
- **What to assert (bringup gates)**:
- `validate_before_send()` passes before any bytes are sent and before any mesh collective.
- Overlap gate: `OverlapScore >= 0.30`, and observed period trends toward `max(Stage0, Stage1)` rather than their sum after warmup.
- Safety gate: queue depths never exceed bounds; after a hard cut, no stale-epoch result reaches decode/output.

### Gotchas and failure modes

- Unbounded queues fail by OOM or latency death spiral, not by clean exceptions.
- Bounding only one boundary is insufficient: if `ready_for_decode` is unbounded, decode backlog can still OOM even if `inflight_to_mesh` is bounded.
- Hard cuts without epoch-tagging and queue flush semantics are correctness bugs: stale results can arrive late and be decoded after a reset.
- Depth > 2 can hide scheduling problems: throughput may look better while latency and memory balloon.
- Recompute coupling (R0a) can reintroduce serialization if building envelope `k+1` depends on decoding `k`; overlap must be re-validated when recompute is enabled.
- **Classic “we thought we overlapped but didn’t” signature**: `t_mesh_idle_ms` tracks rank0 decode time and both queues stay near 0. This usually means rank0 is sending `env[k+1]` *after* decoding `res[k]`, serializing the mesh. Fix: send `env[k+1]` before decoding `res[k]` (Stage 0 must behave like a producer even while it is consuming Stage 1 results). See: `deep-research/2026-02-22/pp-rank0-out-of-mesh/reply.md` (“Scheduling bug signature”).

### Experiments to run

- Sweep `D_in/D_out ∈ {1,2,3,4}` and record FPS, period, `OverlapScore`, p50/p95 chunk latency, and peak memory; confirm `1` collapses overlap and `2` is the first setting that sustains it.
- Intentionally remove bounds (or disable the “don’t enqueue when full” rule), inject a slow decode, and observe queue growth and memory/latency blowup; restore bounds and confirm stabilization.
- Inject slow decode (Stage 0 bottleneck) and observe `ready_for_decode` saturating at `D_out`, `t_mesh_idle_ms` rising, and period drifting toward Stage 0 time; confirm `OverlapScore` drops.
- Inject slow denoise (Stage 1 bottleneck) and confirm bounded behavior (no queue growth beyond `D_in/D_out`) with period drifting toward Stage 1 time.
- Trigger a hard cut while queues are non-empty and verify: both queues flush, `cache_epoch` increments, stale results are dropped, and the pipeline refills cleanly from the reset envelope.
