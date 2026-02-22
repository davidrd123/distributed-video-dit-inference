---
status: draft
---

# Topic 24: Video DiT scheduling — denoising steps, causal dependencies, rolling windows

Video DiT scheduling determines how denoising steps are ordered and parallelized across frames. **Rolling window** approaches generate video autoregressively: each window of N frames shares context from the previous window, enabling arbitrarily long generation. **Stream Batch** (from StreamDiffusion) batches denoising steps across time for throughput.

## Resources

<!-- Resource IDs from manifest belonging to this topic -->
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| streamdiffusion | StreamDiffusion: A Pipeline-level Solution for Real-time Interactive Generation | medium | pending |
| streamdiffusionv2 | StreamDiffusionV2 | high | fetched |
| diffusion-video-survey | Diffusion Models for Video Generation | medium | pending |
| pipedit | PipeDiT: Accelerating DiT in Video Generation with Pipelining and Decoupling | high | converted |
| causvid | From Slow Bidirectional to Fast Autoregressive Video Diffusion Models | high | converted |
| dit-paper | Scalable Diffusion Models with Transformers (DiT) | high | condensed |

## Implementation context

This topic covers the scheduling strategy that determines whether PP can be filled. The working system uses 1 denoising step per chunk (causal streaming). StreamDiffusionV2 demonstrates that treating denoising steps as a batch multiplier (Stream Batch) fills the PP pipeline, but only when multiple latents at different noise levels are simultaneously in flight via a rolling window.

For the Scope system, the three paths to B>1 are: (1) multiple concurrent sessions, (2) rolling window of frame chunks in flight, (3) interleaved denoise across frames. Path (1) is a throughput play; paths (2-3) require restructuring the generation loop.

See: `refs/implementation-context.md` → Phase 3, `scope-drd/notes/FA4/h200/tp/streamdiffusion-v2-analysis-opus.md` → "What This Means for Our System".

Relevant Scope code / notes:
- `scope-drd/notes/FA4/h200/tp/explainers/08-streamv2v-mapping.md` (mapping StreamDiffusionV2 contracts/transport → Scope PP0/PP1)
- `scope-drd/src/scope/core/pipelines/krea_realtime_video/pipeline.py` (current per-chunk generator call pattern)
- `scope-drd/src/scope/server/frame_processor.py` (ingress loop, chunking, overrides that will become PP envelope fields)

## Synthesis

### Mental model

- **Scheduling answers one question**: “what independent work exists *right now* that the GPUs can do concurrently without changing semantics?”
- In streaming video diffusion you have three potential concurrency axes:
  1. **Across sessions** (multiple users/streams): true batch; increases throughput, not per-session latency.
  2. **Across chunks in a rolling window** (multiple frame chunks in flight): requires the control plane to tolerate multiple “partially complete” chunks and to manage KV lifecycle/epochs cleanly.
  3. **Across denoising steps / noise levels** (“Stream Batch”): treat denoising micro-steps as the batch dimension; multiple latents at different noise levels are simultaneously in flight.
- **Pipeline parallelism (PP) is not magic**: it only helps when there is enough independent work to keep stages busy.
  - Classic result: with `P=2` stages and `B=1` microbatch, utilization is ~50% because one stage is idle during fill/drain.
  - StreamDiffusionV2’s core point is: *PP + StreamBatch are co-designed*; StreamBatch provides the `B>1` items needed to fill the pipeline. (`scope-drd/notes/FA4/h200/tp/streamdiffusion-v2-analysis-opus.md` “Prerequisite 1: Pipeline fill”.)
- **Scope’s Phase-PP (rank0-out-of-mesh) is a different “fill” story**: even with one denoising step per chunk, you can overlap *phases across successive chunks*:
  - Stage 0 (rank0): A (build envelope) + C (decode/output)
  - Stage 1 (mesh): B (generator lockstep)
  - With bounded queues (`D_in=D_out=2`), you can get a steady-state period of `≈ max(stage0, stage1)` if dependencies allow it. (`scope-drd/notes/FA4/h200/tp/pp-topology-pilot-plan.md`, `scope-drd/notes/FA4/h200/tp/pp0-bringup-runbook.md`.)

### Key concepts

- **Chunk**: the streaming unit we schedule (e.g., 3 frames/chunk in the current system). Chunk boundaries are protocol boundaries: `call_id`, `chunk_index`, `cache_epoch`.
- **Rolling window**: a bounded “in-flight” set of chunks whose state (latents, context, cache indices) is partially complete and must be advanced deterministically.
- **Stream Batch (B)**: the number of independent items concurrently in the pipeline. In StreamDiffusionV2, “items” are denoising micro-steps at different noise levels; in “multi-session” it’s sessions; in phase-PP it’s (chunk k, chunk k+1, …) at different phases.
- **Fill requirement**: a PP schedule only reaches its steady-state throughput when the pipeline is kept full (enough in-flight items). `B=1` tends to collapse to bubbles/serialization. (`scope-drd/notes/FA4/h200/tp/streamdiffusion-v2-analysis-opus.md` and `scope-drd/notes/FA4/h200/tp/streamdiffusion-v2-analysis-5pro.md`.)
- **Backpressure depths (`D_in`, `D_out`)**: explicit bounds on in-flight envelopes/results that determine whether overlap is possible and whether memory stays bounded. For phase-PP, `D_in=D_out=2` is the minimum for “double-buffered” overlap at both stage boundaries. (`scope-drd/notes/FA4/h200/tp/pp-topology-pilot-plan.md`.)
- **Recompute coupling**: some “Phase A” inputs are not purely local; KV-cache recompute can require a context tensor derived from decoded pixels (R0a semantics). If Stage 0 must decode to produce next chunk’s override, the schedule can re-serialize. (`scope-drd/notes/FA4/h200/tp/pp-next-steps.md`, `scope-drd/notes/FA4/h200/tp/pp-topology-pilot-plan.md`, Topic 22.)
- **What “one denoising step per chunk” implies**:
  - It keeps the single-stream loop simple and low-latency.
  - It removes the easiest `B>1` lever (interleaving multiple denoise micro-steps).
  - It makes phase-PP and multi-session the most realistic near-term “fill” options.

### Cross-resource agreement / disagreement

- **Agreement**:
  - StreamDiffusionV2 (and our analysis of it) is consistent with classic PP scheduling theory: PP without fill provides little to no throughput benefit and can even regress due to bubbles. (`scope-drd/notes/FA4/h200/tp/streamdiffusion-v2-analysis-opus.md`.)
  - The PP pilot plan and runbook converge on a practical bringup sequence: prove contract correctness first (PP0/R1), then prove overlap with bounded queues and metrics (OverlapScore), then restore semantic coupling (R0a) and measure how much overlap survives. (`pp-next-steps.md`, `pp0-bringup-runbook.md`.)
- **Nuance / “don’t overgeneralize”**:
  - “PP won’t help single-stream latency” is true for *layer-PP with B=1*, but phase-PP can still hide rank0 decode under mesh compute if dependencies are managed. The correct statement is: **PP only helps if there exists independent work to overlap** (and the schedule exposes it).
  - StreamBatch’s “items in flight” are noise-level micro-steps; phase-PP’s “items in flight” are chunks at different phases. They are not the same mechanism and they stress different contracts (KV epoching, queueing, output ordering).

### Practical checklist

- **1) Decide what you’re optimizing** (explicitly):
  - single-stream FPS at fixed quality and 1-step/chunk, or
  - per-session latency (TTFF/period), or
  - node throughput (multi-session).
- **2) Choose a concurrency axis you can actually support**:
  - If you cannot restructure the generation loop: do **phase-PP** (rank0-out-of-mesh) + bounded queues.
  - If you can accept a larger refactor: do **PP + StreamBatch** (denoise-step interleaving) in the StreamDiffusionV2 style.
  - If product allows multi-session: implement **true batching across sessions** (but serialize any per-mesh collectives per group; Topic 02).
- **3) For phase-PP bringup, follow the safe sequence**:
  - PP0/R1 (no recompute, no TP): prove contract + monotonic IDs + call-count checks.
  - Add bounded queues (`D_in=D_out=2`) and prove overlap via `OverlapScore ≥ 0.30`.
  - Restore recompute via R0a (explicit override tensor) and quantify the overlap loss.
  - Only then attempt PP1 (TP inside mesh) because wrong-group collectives become an instant deadlock surface. (`pp-next-steps.md` Step A5.)
- **4) Instrument overlap and backpressure, not vibes**:
  - Implement the PP0 Phase 2 timestamp set and compute `period/stage0/stage1/OverlapScore`. (`pp0-bringup-runbook.md`.)
  - Track queue depths and stalls; ensure hard cuts flush queues and bump `cache_epoch` (Topic 21/22).
- **5) Keep protocol invariants first-class**:
  - Scheduling changes tend to create “sometimes hang” bugs if they introduce conditional collectives or call-count mismatches. Use explicit phase plans (`expected_generator_calls`) and anti-stranding rules (Topic 20 / operator matrix).

### Gotchas and failure modes

- **Bubble misdiagnosis**: “PP didn’t help” can mean either “B=1 bubbles” or “we accidentally serialized because of a dependency (e.g., recompute override) or backpressure (D_out=1).”
- **Stage imbalance**: if rank0 keeps doing expensive local work (decode/recompute) while mesh is fast, the period becomes stage0-bound and extra GPUs don’t help. Run 10b’s decode+recompute share is the warning label here. (`scope-drd/notes/FA4/h200/tp/bringup-run-log.md` Run 10b.)
- **Unbounded memory via in-flight items**: every increase in “items in flight” increases live tensors (envelopes, results, decode buffers). Without explicit bounds (D_in/D_out), scheduling becomes a memory leak under load. (Topic 19/07.)
- **Output ordering vs concurrency**: if you allow out-of-order completion, you need explicit acceptance rules (epoch + monotonic IDs) and a dedupe policy (Topic 21) or you will emit stale/duplicate frames.
- **Distributed correctness hazards grow with scheduling complexity**:
  - More concurrency means more opportunities for conditional collectives, wrong-group calls, and stranding peers if a sender fails after commitment.
  - Treat every scheduling change as a control-plane change: re-run the operator test matrix.

### Experiments to run

- **Fill requirement demo (phase-PP)**: PP0 with `D_in/D_out=1` vs `2` vs `4`; record `OverlapScore` and whether `period` approaches `max(stage0, stage1)` in steady state. (`pp0-bringup-runbook.md`, `pp-topology-pilot-plan.md`.)
- **R1 → R0a coupling delta**: run overlap metrics with recompute disabled (R1) and enabled (R0a) and quantify the overlap loss attributable to decoded-anchor dependency. (`pp0-bringup-runbook.md` Phase 3; Topic 22.)
- **Multi-session throughput probe**: run two independent sessions concurrently (if supported) and measure per-GPU throughput; verify rank0 broadcast serialization does not deadlock and that scheduling remains bounded. (Tie to Topic 02/03/20 constraints.)
- **StreamBatch feasibility sketch**: prototype a “two items in flight” denoise-step interleave (B=2) and validate that protocol fields (`call_id/chunk_index/cache_epoch`) and KV lifecycle remain deterministic; if not, stop and formalize the state machine first. (StreamDiffusionV2 analysis + Topic 22.)
