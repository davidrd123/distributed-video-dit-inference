# Zero Bubble Pipeline Parallelism

| Field | Value |
|-------|-------|
| Source | https://arxiv.org/abs/2401.10241 |
| Type | paper |
| Topics | 14 |
| Authors | Qi et al. |
| Year | 2024 |
| Status | condensed |

## Why it matters

Establishes the state-of-the-art ceiling for pipeline schedule efficiency in synchronous training: “zero bubble” schedules that approach an upper bound by increasing scheduling granularity. For our inference PP work, the transferable idea is not the B/W split itself (no backward pass), but the general pattern: identify work with weaker dependencies and use it to fill idle time without introducing synchronizations that break overlap.

## Core claims

<!-- Each claim should cite evidence: sources/zero-bubble-pp/full.md#<heading> -->

1. **Claim**: The key scheduling lever is to split the backward computation into **B** (activation/input-gradient) and **W** (parameter-gradient) phases: for a given stage, **W can be scheduled flexibly after its corresponding B**, enabling bubble-filling reorders while keeping the required F→B dependencies across stages.
   **Evidence**: [sources/zero-bubble-pp/full.md#2-handcrafted-pipeline-schedules](../../sources/zero-bubble-pp/full.md#2-handcrafted-pipeline-schedules)

2. **Claim**: The paper provides two illustrative handcrafted schedules: **ZB-H1** reduces bubbles while keeping peak memory within the 1F1B envelope, and **ZB-H2** achieves a zero-bubble “parallelogram” schedule when permitted a higher memory footprint and when optimizer-step synchronizations are removed.
   **Evidence**: [sources/zero-bubble-pp/full.md#21-memory-efficient-schedule](../../sources/zero-bubble-pp/full.md#21-memory-efficient-schedule), [sources/zero-bubble-pp/full.md#22-zero-bubble-schedule](../../sources/zero-bubble-pp/full.md#22-zero-bubble-schedule)

3. **Claim**: The quantitative analysis for transformers argues $T_W < T_F < T_B$ with $T_B + T_W = 2T_F$, and that $W$’s activation memory is smaller than $B$’s, motivating why $W$ is an effective bubble-filler.
   **Evidence**: [sources/zero-bubble-pp/full.md#23-quantitative-analyses](../../sources/zero-bubble-pp/full.md#23-quantitative-analyses)

4. **Claim**: Handcrafted schedules break down in practice because real $T_F/T_B/T_W$ differ and communication time $T_{comm}$ matters; the paper therefore formulates **automatic schedule search** as a function of stages $p$, microbatches $m$, memory limit $M_{limit}$, and profiled timings.
   **Evidence**: [sources/zero-bubble-pp/full.md#3-automatic-pipeline-scheduling](../../sources/zero-bubble-pp/full.md#3-automatic-pipeline-scheduling)

5. **Claim**: The proposed heuristic algorithm (for schedule search) (a) schedules as many warm-up forwards as possible under memory to reduce the bubble before the first backward, then (b) iterates a 1F–1B pattern while inserting W to fill gaps and to recycle memory when hitting the memory limit.
   **Evidence**: [sources/zero-bubble-pp/full.md#31-the-heuristic-algorithm](../../sources/zero-bubble-pp/full.md#31-the-heuristic-algorithm)

6. **Claim**: Optimizer-step synchronizations (e.g., global grad-norm for clipping; NaN/Inf checks) can destroy the zero-bubble parallelogram; the paper proposes a **post-update validation** strategy with partially reduced states flowing stage-to-stage, followed by a backward propagation of the fully reduced global state and an optional rollback if validation fails.
   **Evidence**: [sources/zero-bubble-pp/full.md#4-bypassing-optimizer-synchronizations](../../sources/zero-bubble-pp/full.md#4-bypassing-optimizer-synchronizations), [sources/zero-bubble-pp/full.md#appendix-c-in-place-optimizer-rollback](../../sources/zero-bubble-pp/full.md#appendix-c-in-place-optimizer-rollback)

7. **Claim**: In experiments (GPT-3-like models up to 28.3B on A100s), the best automatically searched schedule (ZB-2p) improves throughput vs 1F1B by up to ~23% under similar memory constraints, and up to ~31% when memory limits are relaxed; it also remains efficient with fewer microbatches than 1F1B.
   **Evidence**: [sources/zero-bubble-pp/full.md#abstract](../../sources/zero-bubble-pp/full.md#abstract), [sources/zero-bubble-pp/full.md#52-main-results](../../sources/zero-bubble-pp/full.md#52-main-results)

8. **Claim**: The automatic schedules can drive bubble rates close to zero (often <1%) by using profiled timings (including $T_{comm}$) rather than assuming equal pass costs; the paper defines and reports “bubble rate” as a schedule-efficiency metric based on the critical-path stage cost.
   **Evidence**: [sources/zero-bubble-pp/full.md#53-efficiency-of-automatic-scheduling](../../sources/zero-bubble-pp/full.md#53-efficiency-of-automatic-scheduling)

9. **Claim**: The paper introduces a memory-efficient “zero bubble” variant (ZB-V) that targets the same peak activation memory as 1F1B under $T_F=T_B=T_W$, balancing memory across workers by structuring the schedule around two chunks and a repetitive 1F–1B–1W steady phase.
   **Evidence**: [sources/zero-bubble-pp/full.md#6-memory-efficient-zero-bubble-schedule](../../sources/zero-bubble-pp/full.md#6-memory-efficient-zero-bubble-schedule), [sources/zero-bubble-pp/full.md#62-schedule-efficiency](../../sources/zero-bubble-pp/full.md#62-schedule-efficiency)

## Key technical details

- **Pass decomposition and dependencies**:
  - $F$: forward
  - $B$: activation/input-gradient backward
  - $W$: parameter-gradient backward
  - Cross-stage dependency: $F$ and $B$ for the same microbatch remain sequentially dependent across stages; within a stage, $W$ can be scheduled after its corresponding $B$.
  **Evidence**: [sources/zero-bubble-pp/full.md#2-handcrafted-pipeline-schedules](../../sources/zero-bubble-pp/full.md#2-handcrafted-pipeline-schedules)

- **Bubble rate definition (schedule efficiency metric)**:
  - Bubble rate is defined as $(cost - m(T_F+T_B+T_W))/cost$, where $cost$ is the largest execution time among stages computed from profiled $T_F,T_B,T_W,T_{comm}$.
  **Evidence**: [sources/zero-bubble-pp/full.md#53-efficiency-of-automatic-scheduling](../../sources/zero-bubble-pp/full.md#53-efficiency-of-automatic-scheduling)

- **Heuristic schedule search inputs**: number of stages $p$, microbatches $m$, activation memory limit $M_{limit}$, and profiled timings $T_F,T_B,T_W,T_{comm}$.
  **Evidence**: [sources/zero-bubble-pp/full.md#3-automatic-pipeline-scheduling](../../sources/zero-bubble-pp/full.md#3-automatic-pipeline-scheduling)

- **Memory–bubble lower bound (Appendix B)**: formalizes how reducing the bubble before the first $B$ can require scheduling more warm-up $F$ passes, increasing the minimum memory limit (via $k$) while decreasing the bubble size ($\\beta$).
  **Evidence**: [sources/zero-bubble-pp/full.md#appendix-b-the-memory-limit-for-automatic-scheduling-algorithm](../../sources/zero-bubble-pp/full.md#appendix-b-the-memory-limit-for-automatic-scheduling-algorithm)

- **Optimizer-step synchronization bypass**:
  - Replace global all-reduce-before-optimizer with a post-validation flow: propagate partially reduced state forward stage-to-stage; propagate fully reduced global state back during the next warm-up; rollback if needed.
  **Evidence**: [sources/zero-bubble-pp/full.md#4-bypassing-optimizer-synchronizations](../../sources/zero-bubble-pp/full.md#4-bypassing-optimizer-synchronizations), [sources/zero-bubble-pp/full.md#appendix-c-in-place-optimizer-rollback](../../sources/zero-bubble-pp/full.md#appendix-c-in-place-optimizer-rollback)

- **ILP formulation (Appendix G)**:
  - Index passes by $(i,j,c)$ for stage $i$, microbatch $j$, and pass kind $c\\in\\{F,B,W\\}$; optimize schedule order variables subject to dependency constraints (adjacent-stage sequencing for $F$/$B$) and memory constraints.
  **Evidence**: [sources/zero-bubble-pp/full.md#appendix-g-ilp-formulation](../../sources/zero-bubble-pp/full.md#appendix-g-ilp-formulation)

## Actionables / gotchas

- **This is a training scheduling paper; don’t cargo-cult the B/W split**: in inference PP there is no backward/optimizer step, so the direct mechanisms (B/W passes, optimizer rollback) are context. Treat this as the “upper bound thinking” reference for how far schedule optimization can go.
- **Transferable idea for inference: bubble-filling via “independent-enough” work**: the key lesson is to identify work that does *not* sit on the strict stage-to-stage critical path and use it to fill idle time (the role played by $W$ in the paper). In our stack, candidates are typically: p2p transfers overlapped on side streams, rank0-side decode work, and pre-staging control-plane envelopes/results. See: `refs/implementation-context.md` (Phase 3 load-bearing resources list that positions `zero-bubble-pp` as a future PP scheduling ceiling).
- **Zero-bubble is fragile to synchronizations**: the paper’s optimizer-sync discussion is the general warning: any “global sync point” on the critical path breaks the parallelogram. For inference, the analog is introducing stage-wide barriers / blocking “must wait for rank0” work that prevents overlap.
- **Use the paper’s “bubble rate” framing to reason about PP tuning**: if you’re changing queue depths / microbatching / stream-batch style concurrency, think in terms of (a) how close you can drive bubbles toward zero under your memory budget and (b) what work is sitting in the max-stage critical path. See: [sources/zero-bubble-pp/full.md#53-efficiency-of-automatic-scheduling](../../sources/zero-bubble-pp/full.md#53-efficiency-of-automatic-scheduling).
- **Memory budget matters; reducing bubbles often consumes more in-flight state**: ZB-2p’s “nearly zero bubble” improvements come with higher memory usage than 1F1B; ZB-V exists specifically to trade schedule complexity for memory. For inference PP, this maps to concrete buffering decisions (how much state you keep in-flight across stage boundaries) and “stash vs recompute” coupling. See: [sources/zero-bubble-pp/full.md#52-main-results](../../sources/zero-bubble-pp/full.md#52-main-results), [sources/zero-bubble-pp/full.md#6-memory-efficient-zero-bubble-schedule](../../sources/zero-bubble-pp/full.md#6-memory-efficient-zero-bubble-schedule).
- **Treat this as out-of-scope for PP0 but in-scope for “Phase 4 optimization”**: the paper is most useful once you have a working PP topology and want to push utilization toward the ceiling by reducing bubbles beyond “just fill the pipeline.” See: `refs/implementation-context.md` (Phase 3 + Phase 4 mentions of `zero-bubble-pp`).

## Related resources

- [gpipe](gpipe.md) -- foundational synchronous PP schedule
- [pipedream-2bw](pipedream-2bw.md) -- the 1F1B schedule that zero-bubble improves upon
- [pytorch-pipelining-api](pytorch-pipelining-api.md) -- ScheduleInterleavedZeroBubble and ScheduleZBVZeroBubble implement this
- [pipedit](pipedit.md) -- pipeline-parallel DiT inference
