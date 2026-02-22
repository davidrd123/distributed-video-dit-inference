---
status: draft
---

# Topic 15: Pipeline scheduling theory — bubble fraction derivation, Little's law connection

The bubble fraction for a P-stage pipeline processing B micro-batches is **(P-1)/(B+P-1)**. This is a direct consequence of pipeline startup and drain latency. **Little's law** (L = lambda * W) provides the framework: to keep P stages busy, you need at least P micro-batches in flight, and throughput approaches the ideal rate as B approaches infinity.

## Resources

<!-- Resource IDs from manifest belonging to this topic -->
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| jax-scaling-book | How to Parallelize a Transformer for Training (JAX Scaling Book) | medium | fetched |
| megatron-trillion-params | Scaling Language Model Training to a Trillion Parameters Using Megatron | medium | fetched |
| megatron-pp-schedules | Megatron-LM Pipeline Parallel Schedules source | medium | fetched |

## Implementation context

Scope already observed the classic pipeline-bubble reality: block-PP can reach **~1.87× throughput** with enough in-flight work, but single-stream per-chunk latency stayed flat (PP doesn’t shorten the critical path). StreamDiffusionV2’s takeaway is the same: PP alone with **2 stages and B=1** has a **50% bubble**, so you need “Stream Batch” / `max_outstanding≥2` to fill. The PP0 plan bakes this into bringup by starting with double-buffer depths `D_in=D_out=2` and an overlap pass gate `OverlapScore ≥ 0.30`.

See: `refs/implementation-context.md`, `scope-drd/notes/FA4/h200/tp/explainers/01-why-two-gpus.md` (PP throughput vs latency), `scope-drd/notes/FA4/h200/tp/streamdiffusion-v2-analysis-opus.md` (PP bubble + Stream Batch), `scope-drd/notes/FA4/h200/tp/pp0-bringup-runbook.md` (overlap gate), `scope-drd/notes/FA4/h200/tp/pp-topology-pilot-plan.md` (queue depths).

Relevant Scope code:
- `scope-drd/scripts/pp_two_rank_pipelined.py` (`max_outstanding` and microbatch/queue depth knobs for PP0)
- `scope-drd/scripts/bench_pp_comm.py` (PP transport cost microbench when validating scheduling assumptions)

## Synthesis

<!-- To be filled during study -->

### Mental model

Topic 13 answers “what do GPipe / PipeDream *do*.” This topic is “how do you reason about the **efficiency bounds** before you implement anything.”

Start with the concurrency requirement (Little’s law):

- **Little’s law**: $L = \\lambda W$ (items-in-flight = throughput × latency).
- For a pipeline with $P$ stages, the end-to-end latency $W$ for a single item is roughly a **sum over stages** (you must traverse each stage in order).
- The best-case steady-state throughput $\\lambda$ is bounded by the **bottleneck stage**: you can’t go faster than one item per bottleneck-stage time.

The key observation is that throughput is only bottleneck-limited if you have enough items in flight to *hide* the pipeline’s fill/drain slack.

In the idealized balanced case (all stage times equal to $t$):

- Makespan for $B$ micro-batches/items is $(B + P - 1) \\cdot t$ (fill $P-1$, then $B$ steady steps).
- Steady-state utilization is $\\frac{B}{B + P - 1}$, so the **bubble fraction** is
  $$
  \\beta = 1 - \\frac{B}{B + P - 1} = \\frac{P - 1}{B + P - 1}.
  $$
  This is the GPipe “bubble” story in one line. See: [gpipe](../resources/gpipe.md).

Two immediate translations for inference PP:

- “Micro-batches” are just **items in flight** (chunks/envelopes/streams). If $P=2$ and $B=1$, you get $\\beta = 1/2$ ⇒ **50% idle time**. That’s why PP0 needs `max_outstanding≥2` / Stream Batch to fill. See: [gpipe](../resources/gpipe.md), [pipedream-2bw](../resources/pipedream-2bw.md), and `refs/implementation-context.md` (Phase 3).
- With enough in-flight items and real overlap, the critical path moves from “sum of stages” to “max of stages”: steady-state period approaches $\\max_i t_i$. PipeDream-2BW’s appendix is the cleanest expression of this “sum → max” shift (and how comm terms enter). See: [pipedream-2bw](../resources/pipedream-2bw.md).

Finally: what’s the ceiling? In training, “zero-bubble” schedules show that if you can break work into finer-grained pieces with weaker dependencies (e.g., split backward into B/W), you can fill bubbles beyond what naive 1F1B achieves, approaching near-zero bubble rate under some memory/sync assumptions. For inference, we don’t have backward passes, but the ceiling-thinking still transfers: you look for “independent-enough” work to overlap with the critical path, and you treat global synchronizations as bubble killers. See: [zero-bubble-pp](../resources/zero-bubble-pp.md).

### Key concepts

- **$P$ / $K$ / depth**: number of pipeline stages/partitions. GPipe uses $K$ partitions; the bubble math is the same. See: [gpipe](../resources/gpipe.md).
- **$B$ / $M$ / micro-batches (items in flight)**: number of independent items simultaneously flowing through the pipeline. Larger $B$ reduces bubbles but typically increases memory (more in-flight activations/state) and increases per-item latency (queueing). See: [gpipe](../resources/gpipe.md), [pipedream-2bw](../resources/pipedream-2bw.md).
- **Fill / drain**: startup and teardown phases that create unavoidable slack; only steady-state overlap can amortize them. See: [gpipe](../resources/gpipe.md).
- **Bubble fraction / utilization**: for balanced stages, bubble fraction $\\beta = (P-1)/(B+P-1)$ and utilization $1-\\beta$. See: [gpipe](../resources/gpipe.md).
- **Throughput vs latency**:
  - Single-item latency is roughly “sum of stages” (plus boundary comm).
  - Steady-state throughput is roughly “1 / max stage time” if the pipeline is filled and overlap is real. See: [pipedream-2bw](../resources/pipedream-2bw.md).
- **Stage imbalance**: if one stage is slower, it becomes the bottleneck and creates bubbles elsewhere no matter how large $B$ is. (Pipedream’s “max” makes this explicit.) See: [pipedream-2bw](../resources/pipedream-2bw.md).
- **Schedule efficiency beyond 1F1B**: zero-bubble work treats scheduling as an optimization problem over dependency and memory constraints and introduces a “bubble rate” metric based on the profiled critical path. See: [zero-bubble-pp](../resources/zero-bubble-pp.md).

### Cross-resource agreement / disagreement

- **Agreement on the first-order law**: you don’t get PP throughput unless you can reach a steady state with enough in-flight items. GPipe expresses this as bubble fraction shrinking with $B$ and gives a heuristic that bubbles are negligible when micro-batches are much larger than stages (e.g., $M \\ge 4K$). See: [gpipe](../resources/gpipe.md).

- **Agreement on the true limit (“max stage wins”)**: PipeDream-2BW formalizes the “sum → max” shift as the defining property of pipelining (and makes comm terms explicit). This matches the mental model we use to validate overlap in PP bringup: the steady-state period should approach `max(Stage0_ms, Stage1_ms)` rather than their sum. See: [pipedream-2bw](../resources/pipedream-2bw.md), `refs/implementation-context.md`.

- **Where zero-bubble differs**: GPipe and PipeDream focus on coarse schedules (fill/drain, 1F1B) where bubbles are largely a consequence of pipeline structure and microbatch count. Zero-bubble reframes scheduling as “find more reorderings by increasing granularity,” backed by automatic schedule search using measured timings and explicit memory constraints. It also highlights fragility to synchronization points. See: [zero-bubble-pp](../resources/zero-bubble-pp.md).

- **Inference translation caveat**: zero-bubble’s specific mechanism (splitting backward into B/W) is training-only, but its transferable idea is general: if you can identify work that is not strictly on the stage-to-stage dependency chain, you can use it to fill bubbles. For inference PP, “independent-enough work” tends to be: transport on a side stream, rank0-side decode/encode, pre-staging envelopes, and other control-plane work (so long as you don’t introduce new global sync points). See: [zero-bubble-pp](../resources/zero-bubble-pp.md).

### Practical checklist

- **Decide what you’re optimizing**: throughput (steady-state FPS) or latency (first chunk). Microbatching improves throughput but can worsen latency due to fill/drain + queueing. See: [gpipe](../resources/gpipe.md).

- **Measure per-stage critical path**:
  - Get per-stage times (including boundary comm) and identify the bottleneck.
  - Your theoretical best steady-state period is approximately $\\max_i t_i$; if your measured period is closer to $\\sum_i t_i$, you don’t have overlap yet. See: [pipedream-2bw](../resources/pipedream-2bw.md).

- **Pick an in-flight depth $B$ that can fill the pipeline**:
  - Minimum to avoid the “$B=1$ gives 1/P utilization” cliff is $B\\ge P$.
  - GPipe’s rule of thumb for “negligible” bubbles is $B$ (microbatches) much larger than $P$ (stages), e.g. $M \\ge 4K$. For $P=2$, that suggests starting at `max_outstanding=2` for bringup, then testing `4` and `8` if memory allows. See: [gpipe](../resources/gpipe.md).

- **Balance stage times first; then optimize schedule**:
  - If you have stage imbalance, no amount of scheduling can beat the bottleneck.
  - Once stage times are balanced, scheduling and overlap work are about making the realized period match the max-stage bound. See: [pipedream-2bw](../resources/pipedream-2bw.md).

- **Treat memory as a hard constraint**:
  - More in-flight items means more boundary state to buffer; your maximum feasible $B$ is a memory decision, not just a scheduling decision. PipeDream’s appendix is a useful checklist for what contributes to footprint. See: [pipedream-2bw](../resources/pipedream-2bw.md).

- **Use “bubble rate” thinking for optimization passes**:
  - Once PP works, use the zero-bubble framing (“what’s idle time relative to the critical-path stage cost?”) to guide where to spend complexity (overlap transport, remove sync points, add buffering). See: [zero-bubble-pp](../resources/zero-bubble-pp.md).

### Gotchas and failure modes

- **Throughput gains without latency gains is expected**: PP changes the steady-state critical path; it does not necessarily shorten the per-item critical path. This is why “PP looked useless at B=1” is not a failure; it’s the bubble math. See: [gpipe](../resources/gpipe.md).

- **Measuring too short a window**: if you measure only a handful of items, fill/drain dominates and you’ll underestimate steady-state throughput. Always measure long enough to observe a stable period. See: [gpipe](../resources/gpipe.md).

- **Stage imbalance dominates everything**: if one stage owns non-uniform work (I/O, conditioning prep, decode) you violate the “balanced stage” assumption and your realized throughput will be far from the theoretical bound. (PipeDream’s max-of-stages makes the bottleneck explicit.) See: [pipedream-2bw](../resources/pipedream-2bw.md).

- **Synchronization points kill zero-bubble-like benefits**: zero-bubble’s optimizer-sync discussion generalizes: any “must wait for everyone” step on the critical path destroys the parallelogram. In inference, the analog is accidental barriers or blocking rank0 dependencies that prevent overlap. See: [zero-bubble-pp](../resources/zero-bubble-pp.md).

- **Bigger $B$ isn’t free**: more in-flight items increase buffering memory, can increase queueing latency, and can amplify long-tail jitter/backpressure effects. Treat $B$ as a controlled knob with observability, not as “more is always better.” See: [pipedream-2bw](../resources/pipedream-2bw.md).

### Experiments to run

1. **Bubble curve sanity check (sweep $B$)**:
   - Vary `max_outstanding` (e.g., 1, 2, 4, 8) and measure utilization / idle time.
   - For balanced $P=2$, your first-order expectation is utilization $\\approx \\frac{B}{B+1}$ and bubble $\\approx \\frac{1}{B+1}$. See: [gpipe](../resources/gpipe.md).

2. **Sum vs max validation**:
   - Measure per-stage times ($t_0, t_1, \\ldots$) and check whether your steady-state period approaches $\\max_i t_i$ when $B$ is large enough.
   - If it stays near $\\sum_i t_i$, overlap isn’t working (or you have hidden sync points). See: [pipedream-2bw](../resources/pipedream-2bw.md).

3. **Stage-balance sensitivity**:
   - Intentionally shift work between stages (different block splits, moving decode/encode ownership) and observe how throughput tracks the bottleneck stage time.
   - Goal: confirm “max stage wins” empirically before investing in schedule complexity. See: [pipedream-2bw](../resources/pipedream-2bw.md).

4. **Memory vs schedule trade-off**:
   - For each $B$, record peak memory and dropped work (backpressure). Plot throughput vs memory to find the knee.
   - Use PipeDream’s appendix checklist to reason about what state is scaling with $B$. See: [pipedream-2bw](../resources/pipedream-2bw.md).

5. **Sync-point hunt**:
   - Add instrumentation to find any “global wait” in the steady state (e.g., stage-wide barriers, “must wait for rank0” steps).
   - Treat each sync point as a candidate bubble source (zero-bubble’s main warning). See: [zero-bubble-pp](../resources/zero-bubble-pp.md).
