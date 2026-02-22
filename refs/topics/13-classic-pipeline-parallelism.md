---
status: draft
---

# Topic 13: Classic PP (GPipe, PipeDream) — micro-batching, 1F1B schedule, bubble fraction

Pipeline parallelism partitions a model across devices by layer. **GPipe** fills the pipeline with micro-batches and synchronizes at the end (high memory, simple). **PipeDream's 1F1B schedule** interleaves one forward and one backward per micro-batch, reducing peak activation memory from O(num_microbatches) to O(num_stages).

## Resources

<!-- Resource IDs from manifest belonging to this topic -->
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| gpipe | GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism | high | condensed |
| pipedream | PipeDream: Generalized Pipeline Parallelism for DNN Training | medium | pending |
| pipedream-2bw | Memory-Efficient Pipeline-Parallel DNN Training (PipeDream-2BW) | high | condensed |
| pp-siboehm | Pipeline-Parallelism: Distributed Training via Model Partitioning | medium | pending |

## Implementation context

PP is the **next topology play** after TP v0. The working design is rank0-out-of-mesh (Stage 0 = encode/decode, Stage 1 = generator-only with TP inside `mesh_pg`). Contracts (`PPEnvelopeV1`/`PPResultV1`) are defined, transport is implemented, smoke test passes. The key insight from StreamDiffusionV2 analysis: PP without batching at B=1 gives only 50% utilization (2 stages). PP becomes compelling when B>1 items are in flight.

Bringup plan is Steps A1-A5 in `scope-drd/notes/FA4/h200/tp/pp-next-steps.md`. Pseudocode reference in `pp-control-plane-pseudocode.md`. Runbook in `pp0-bringup-runbook.md`.

Current PP transport is **PP0**: rank0 ↔ mesh leader point-to-point. Mesh-wide broadcast/repartition is staged work (PP1+), not done end-to-end yet.

See: `refs/implementation-context.md` → Phase 3.

Relevant Scope code:
- `scope-drd/src/scope/core/distributed/pp_contract.py` (`PPEnvelopeV1`/`PPResultV1` schema + validation, `expected_generator_calls` tripwire)
- `scope-drd/src/scope/core/distributed/pp_control.py` (rank0↔leader preflight + send/recv)
- `scope-drd/scripts/pp_two_rank_pipelined.py` (PP0 overlap bringup; `max_outstanding` bounded in-flight)

## Synthesis

<!-- To be filled during study -->

### Mental model

Pipeline parallelism (PP) has two independent problems:
1) **Partitioning**: which layers live on which device (“stages”), and whether those stages are balanced.
2) **Scheduling**: how many items are in flight (“micro-batches”), and in what order each stage executes work.

The progression across the classic papers is: **GPipe → 1F1B → zero-bubble**.

- **GPipe (fill/drain, simple)**: split a batch into micro-batches, run *all forwards* through the pipeline, then *all backwards*, then synchronize/update once per batch. This is easy to reason about, but it requires many micro-batches to amortize “fill/drain” bubble time (see `gpipe` claim 5) and it tends to stash more activation state (see `gpipe` claim 4).
- **1F1B (steady state, bounded stash)**: PipeDream-style schedules interleave work so each stage alternates between a forward and backward in steady state, reducing peak activation stashing vs GPipe-style flushes. PipeDream-2BW is still a training paper, but it gives the cleanest *systems mental model*: pipelining moves the critical path from a **sum over stages** to a **max over stages** (see `pipedream-2bw` claim 7), and it forces you to account for inter-stage communication and bounded in-flight state.
- **Zero-bubble (training ceiling)**: zero-bubble PP shows how much efficiency you can get by increasing scheduling granularity: split “backward” into parts (B and W) and use the weaker-dependency part to fill idle slots (see `zero-bubble-pp` claim 1). It’s training-centric, but the transferable idea is “find useful work that can be moved into bubbles” and “avoid synchronizations that break overlap” (see `zero-bubble-pp` claim 6).

For **inference** (our case), there is no backward or optimizer step. So reframe:
- “Micro-batches” are just **items in flight**. In our PP0 design, those items are “in-flight envelopes/results” (queue depth / `max_outstanding`).
- “1F1B” is best read as the steady-state pattern: once the pipeline is full, every stage should do one unit of useful work per micro-step, and throughput is dominated by the slowest stage (the “max over stages” idea from `pipedream-2bw` claim 7).
- “Zero-bubble” is the long-term ceiling: if we want more utilization than “pipeline full,” we must overlap *something* useful into the slack (e.g., rank0 decode, p2p transfers on side streams, envelope pre-staging), without introducing cross-stage sync points.

### Key concepts

Definitions below are “just enough” vocabulary for bringup/debugging.

- **Stage**: a contiguous partition of the model assigned to one device (or one rank), communicating activations at boundaries (see `gpipe` claim 1).
- **Micro-batch / items in flight**: a unit of work that flows through the pipeline; more micro-batches reduce bubble fraction but can increase latency and memory (see `gpipe` claim 2 and claim 5).
- **Fill/drain**: the pipeline warm-up and tail phases where some stages are idle because the dependency chain hasn’t “filled” the whole pipeline yet (GPipe schedule; see `gpipe` claim 2).
- **Bubble fraction / bubble time**: the idle fraction caused by fill/drain (and imbalance). GPipe explicitly characterizes bubble scaling with stages vs micro-batches (see `gpipe` claim 5).
- **Steady state**: once the pipeline is full, each micro-step produces one output per step (idealized); overall throughput is limited by the slowest stage (the “max over stages” framing in `pipedream-2bw` claim 7).
- **1F1B**: a training schedule that interleaves forward/backward in steady state to avoid flushes and reduce activation stash; for inference, treat it as “don’t leave devices idle if there is independent work to run” (see `pipedream-2bw` claim 1 and claim 3 for context).
- **Activation stash / stashing**: stored intermediate activations needed later (training backward) or intermediate state held because later stages haven’t consumed it yet. GPipe uses re-materialization to trade compute for lower activation memory (see `gpipe` claim 4).
- **Balanced partitioning**: the entire point of pipelining is lost if one stage is much slower; then the “max stage time” dominates steady-state throughput (see `gpipe` claim 7; and the “max over stages” model in `pipedream-2bw` claim 7).
- **Pipeline depth vs width**: depth = number of stages; width = replication of stages (parallel pipelines). PipeDream-2BW formalizes both because replication introduces intra-stage gradient aggregation (training) and interacts with interconnect topology (see `pipedream-2bw` claim 4, claim 8).

### Cross-resource agreement / disagreement

Where the papers agree (and where we should treat it as invariant):
- **Stage balance dominates**: all scheduling tricks assume you’re close to balanced stage times; otherwise the slow stage sets throughput (GPipe shows uniform-layer Transformers scale better than imbalanced models; see `gpipe` claim 7; PipeDream-2BW’s model is max-over-stages; see `pipedream-2bw` claim 7).
- **“More in-flight items” is how you amortize bubbles**: GPipe explicitly states bubble overhead becomes negligible with sufficiently many micro-batches (see `gpipe` claim 5); zero-bubble emphasizes schedules that stay efficient even at smaller microbatch counts (see `zero-bubble-pp` claim 7 and claim 8).

Where they diverge (and why it matters to us):
- **Synchronization vs continuous scheduling**: GPipe is “flushy” (synchronize/update at the end of a mini-batch) while PipeDream-family schedules are designed to avoid that flush, at the cost of more complex scheduling/semantics in training (see `pipedream-2bw` claim 1 and claim 3). For inference, we don’t care about optimizer semantics, but we *do* care about whether a framework introduces implicit global sync points that break overlap.
- **Complexity trade-off**: zero-bubble PP adds significant machinery (auto schedule search, optimizer-sync bypass, rollback) to chase a ceiling (see `zero-bubble-pp` claims 4–6). That complexity is only justified once basic PP topology is stable and you’re truly utilization-limited by bubbles rather than by single-stage bandwidth/compute.

What transfers cleanly to inference vs what does not:
- Transfers: “fill the pipeline,” “max stage time sets throughput,” “avoid sync points that destroy overlap,” and “memory budget vs bubble reduction is a real trade-off” (see `gpipe` claim 5; `pipedream-2bw` claim 7; `zero-bubble-pp` claim 8).
- Mostly training-only: weight stashing/versioning, stale-gradient semantics, and the B/W decomposition itself (see `pipedream-2bw` claim 2; `zero-bubble-pp` claim 1 for what B/W means).

### Practical checklist

Use this as a pre-flight / bringup checklist for PP0 on Wan 2.1 (40 uniform DiT blocks) with rank0-out-of-mesh on H200s.

Before you try to “get speedup,” prove correctness and invariants (PP0 pilot plan A1–A5):
- **Anti-stranding send semantics**: validate/serialize/spec materialization *before* sending any header bytes (Step A1; see `scope-drd/notes/FA4/h200/tp/pp-next-steps.md`).
- **Correctness pilot with no overlap**: run a two-rank pilot with blocking send/recv, monotonic IDs, and timeouts (Step A2; see `scope-drd/notes/FA4/h200/tp/pp-next-steps.md`).
- **Overlap only after correctness**: introduce bounded queues / double buffering (`D_in=2`, `D_out=2`) and prove no hangs (Step A3; see `scope-drd/notes/FA4/h200/tp/pp-next-steps.md`).

Then validate the scheduling assumptions from the papers:
- **Per-stage profiling**: measure Stage 0 vs Stage 1 compute time. If stage0 includes VAE decode/recompute, it may dominate; rebalance before chasing “bubble” improvements (GPipe stage imbalance warning; see `gpipe` claim 7).
- **Queue depth / items in flight**: verify you can maintain >1 item in flight. For 2 stages, `B=1` caps ideal utilization at ~50% (see the bubble discussion in `gpipe` claim 5; and our Implementation context note).
- **p2p boundary cost**: measure activation transfer size/time; deeper splits reduce per-stage compute but keep boundary activation sizes roughly constant, which can push you toward comm-bound (PipeDream-2BW planning trade-offs; see `pipedream-2bw` claim 8).
- **Stash vs recompute**: decide what state must persist across chunks vs what can be recomputed. Even in inference, “stash policies” decide memory headroom vs extra compute, and can introduce coupling that limits overlap.

### Gotchas and failure modes

- **B=1 looks like “PP didn’t help” but it’s expected**: with 2 stages and only one in-flight item, one stage is always idle half the time (GPipe bubble scaling; see `gpipe` claim 5). In our PP0 design, that means you need `D_in/D_out >= 2` (or an equivalent source of multiple items in flight) before expecting throughput gains.
- **Stage imbalance dominates everything**: if Stage 0 (rank0) does encode/decode, VAE, bookkeeping, and control-plane work, it can become the throughput limiter even if Stage 1 is perfectly pipelined (see `gpipe` claim 7; and the max-over-stages framing in `pipedream-2bw` claim 7).
- **p2p becomes the bottleneck as you increase depth**: splitting into more stages reduces per-stage compute but doesn’t necessarily reduce activation boundary size, so comm can dominate (see `pipedream-2bw` claim 8).
- **Non-uniform “extra work” attached to one stage**: embeddings, I/O, VAE decode, or recompute policies can violate the uniform-block assumption (Wan 2.1’s 40 DiT blocks are uniform, but rank0 side work is not). Treat “balance variance” as a first-class requirement, not a later optimization.
- **Over-optimizing schedule too early**: zero-bubble scheduling is a ceiling; it is not the first tool to reach for. If the system is bandwidth-bound per stage, the schedule can’t manufacture headroom; it only reduces idle time (see `zero-bubble-pp` claim 8 for what it optimizes).

### Experiments to run

These experiments are designed to falsify the mental model quickly.

- **Bubble vs in-flight sweep**: run the PP0 pilot with `B=1` vs `B=2` vs `B=4` items in flight (queue depth / `max_outstanding`) and plot throughput. Expect a sharp jump from “half idle” toward steady state as B increases (see `gpipe` claim 5).
- **Per-stage critical path**: measure Stage 0 time vs Stage 1 time per chunk; steady-state throughput should trend toward `1 / max(stage_time)` (the “max over stages” idea; see `pipedream-2bw` claim 7).
- **Partition sweep**: move the stage boundary across the 40 uniform DiT blocks (e.g., 16/24, 20/20, 24/16) and check whether the slow stage moves with it (imbalance diagnosis; see `gpipe` claim 7).
- **Depth sweep at fixed width**: compare 2-stage vs 3-stage vs 4-stage (if/when feasible) and measure whether p2p boundary comm becomes the limiter (trade-offs; see `pipedream-2bw` claim 8).
- **Comm sensitivity**: artificially increase activation transfer cost (e.g., disable overlap or force sync) and observe throughput degradation to confirm the p2p term is on the critical path.
- **“Ceiling” sanity check**: once PP0 is stable, compare “simple steady-state” scheduling vs more aggressive overlap mechanisms (where available) to estimate how much headroom is “bubble” vs “single-stage bandwidth.” Zero-bubble’s reported gains are a useful calibration for what “schedule-only” improvements look like in a mature training system (see `zero-bubble-pp` claim 7).
