---
status: draft
---

# Topic 11: Functional collectives and Dynamo — why in-place breaks tracing, the funcol solution

Standard NCCL collectives (`all_reduce`, `all_gather`) are **in-place and side-effecting** — they mutate tensors and return opaque `Work` objects. This is fundamentally incompatible with functional graph tracing. Functional collectives (`torch.distributed._functional_collectives`) return new tensors and use `AsyncCollectiveTensor` subclasses for deferred synchronization, making them traceable by Dynamo.

## Resources

<!-- Resource IDs from manifest belonging to this topic -->
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| funcol-rfc-93173 | RFC #93173: PT2-Friendly Traceable, Functional Collective Communication APIs | high | condensed |
| funcol-source | _functional_collectives.py source | medium | pending |
| ezyang-state-of-compile | State of torch.compile for training (August 2025) | high | condensed |
| pytorch-issue-138773 | GitHub Issue #138773: functional collectives 67% slower than torch.distributed | medium | pending |

## Implementation context

This topic is the **single most impactful optimization** in the TP v0 bringup. Switching from `torch._dynamo.disable()`'d in-place collectives to functional collectives eliminated ~160 graph breaks per forward and unlocked compile: **9.6 → 24.5 FPS** (Run 12b). The eager-mode trap (Run 12a: funcol unconditionally in eager was 18 FPS, a regression) means the implementation must dispatch between in-place `dist.all_reduce` (eager) and `torch.distributed._functional_collectives.all_reduce` (compile).

See: `refs/implementation-context.md` → Phase 1, `scope-drd/notes/FA4/h200/tp/bringup-run-log.md` Runs 8-12b.

Relevant Scope code:
- `scope-drd/src/scope/core/tensor_parallel/linear.py` (eager `dist.all_reduce` vs compile-time functional collectives in `_maybe_all_reduce()`)
- `scope-drd/src/scope/core/tensor_parallel/rmsnorm.py` (functional-collective reductions for RMSNorm)
- `scope-drd/scripts/tp_compile_repro.py` (regression harness for graph breaks / graph count)

TODO (after Tier 3 cards exist):
- Add citations from `sources/funcol-rfc-93173/full.md` for the exact `all_reduce` + `wait_tensor` call pattern used here.

## Synthesis

<!-- To be filled during study -->

### Mental model

`torch.compile` (Dynamo + backend) wants to turn your Python forward into a clean “**Tensor → Tensor**” graph it can optimize. Classic distributed collectives fight that in two ways:

1. **They’re in-place**: `dist.all_reduce(x)` mutates `x`.
2. **They return non-Tensor objects**: you often get a `Work` handle back.

That combination doesn’t fit Dynamo’s tracing model or the compiler’s IR hygiene goals: you can’t just drop a “mutate `x` and return `Work`” node into an FX graph and keep everything purely functional. The usual “make it work” workaround is to wrap collectives in `torch._dynamo.disable()`/`torch.compiler.disable()` so they run in eager. But that forces a **graph break** at every collective boundary (Dynamo deep-dive calls out graph breaks as the central perf footgun). When you do this inside a Transformer block loop, you fragment execution into **many tiny compiled regions** with CPython in between.

That’s exactly what happened in our TP v0 bringup:

- In-place collectives behind `torch._dynamo.disable()` ⇒ **~160 graph breaks per forward** ⇒ overhead-bound compiled execution ⇒ **~9.6 FPS** (Runs 8–9b).
- Functional collectives (`torch.distributed._functional_collectives`) ⇒ collectives become traceable “Tensor → Tensor” ops ⇒ no collective-induced breaks ⇒ one/few large compiled graphs ⇒ fused/optimized region ⇒ **~24.5 FPS** (Run 12b).

Functional collectives are the “make distributed ops look like normal tensor ops” move: they return new tensors (often an `AsyncCollectiveTensor`/`AsyncTensor` subclass) and use explicit wait semantics (`wait_tensor`) so the compiler can keep the region in a functional IR. See: [funcol-rfc-93173](../resources/funcol-rfc-93173.md), [dynamo-deep-dive](../resources/dynamo-deep-dive.md), [ezyang-state-of-compile](../resources/ezyang-state-of-compile.md).

### Key concepts

- **In-place collective (c10d)**: mutates a tensor and returns a `Work` handle (side effects + non-Tensor return) — hostile to tracing. See: [funcol-rfc-93173](../resources/funcol-rfc-93173.md).
- **Functional collective (funcol)**: returns a *new* tensor value; designed so the traced graph stays “Tensor → Tensor”. The implementation uses an `AsyncCollectiveTensor`/`AsyncTensor` subclass to represent “this tensor exists as a value, but may not be ready yet.” See: [funcol-rfc-93173](../resources/funcol-rfc-93173.md), [ezyang-state-of-compile](../resources/ezyang-state-of-compile.md).
- **Wait-before-access rule**: the RFC’s key semantic move is that the *output tensor is a real tensor*, but **its storage must not be accessed until you wait**; `wait_tensor` returns a new tensor whose storage is safe. Don’t treat `wait_tensor` as “mutating the original tensor to become ready.” See: [funcol-rfc-93173](../resources/funcol-rfc-93173.md).
- **Graph break**: when Dynamo can’t (or is told not to) trace some code, it splits execution into multiple compiled graphs and runs the “uncapturable” region in CPython in between. Many breaks ⇒ many tiny graphs ⇒ overhead-dominated performance. See: [dynamo-deep-dive](../resources/dynamo-deep-dive.md).
- **Guards and recompiles**: Dynamo specializes on shapes/non-Tensor values and installs guards; guard failures trigger recompilation. In distributed settings, recompiles can turn into rank divergence (one rank recompiles while others don’t) and can deadlock around collectives if ranks don’t execute the same compiled program. See: [dynamo-deep-dive](../resources/dynamo-deep-dive.md), [ezyang-state-of-compile](../resources/ezyang-state-of-compile.md).
- **Eager regression trap**: functional collectives can be slower in eager mode due to extra allocations/wrapping overhead. We saw this directly: **Run 12a = 18 FPS** when funcol was used unconditionally in eager, vs **19.5 FPS** baseline. (This is why `_maybe_all_reduce()` dispatches eager vs compile.) See: [funcol-rfc-93173](../resources/funcol-rfc-93173.md), [ezyang-state-of-compile](../resources/ezyang-state-of-compile.md).

### Cross-resource agreement / disagreement

- **Core agreement (problem + solution)**:
  - The RFC defines the core problem as “distributed collectives don’t compose with the PT2 compiler stack” due to non-functional semantics and non-tensor objects (`ProcessGroup`/`Work`) leaking into the trace; the solution is functional, traceable collectives + explicit wait. See: [funcol-rfc-93173](../resources/funcol-rfc-93173.md).
  - Dynamo’s deep-dive explains *why* the workaround (graph breaks) is disastrous for performance: graph breaks split into multiple graphs with CPython between them, turning you into an overhead-bound regime. See: [dynamo-deep-dive](../resources/dynamo-deep-dive.md).
  - Ezyang’s post places functional collectives in the broader compile + distributed story and emphasizes the distributed fragility: rank-divergent compilation (from guards/dynamics) can hang around collectives. See: [ezyang-state-of-compile](../resources/ezyang-state-of-compile.md).

- **Two compiler-compatible routes (Ezyang)**: Ezyang frames **DTensor** (high-level sharded tensor abstraction) and **functional collectives** (low-level explicit escape hatch) as the two main “compiler-friendly distributed” paths. For Scope TP v0, explicit funcol is the more predictable lever; DTensor becomes more appealing when you want higher-level placement semantics (and want compile to erase DTensor overhead). See: [ezyang-state-of-compile](../resources/ezyang-state-of-compile.md).

- **Important limitation (RFC non-goal)**: The RFC explicitly says **multi-stream semantics in Inductor are a non-goal**. That means funcol fixes *traceability* and graph-break fragmentation, but it does **not** by itself give you “PP-style overlap” (comm/compute overlap via multiple CUDA streams). Overlap still requires explicit stream/event discipline (see `pytorch-cuda-semantics` topic/card). See: [funcol-rfc-93173](../resources/funcol-rfc-93173.md).

### Practical checklist

- **Gate funcol by compile mode (mandatory)**:
  - Eager: use in-place `dist.all_reduce` (baseline behavior).
  - Compiled: use `torch.distributed._functional_collectives` in the compiled region.
  - Where: `_maybe_all_reduce()` in `scope-drd/src/scope/core/tensor_parallel/linear.py` (and analogous reductions in RMSNorm).
  - Why: avoid the eager regression trap (Run 12a).

- **Treat funcol as pure “Tensor → Tensor”**:
  - Always assign/return the output tensor from `all_reduce` (and from `wait_tensor` if used).
  - Don’t assume the original tensor was mutated in-place.
  - Follow the wait-before-access semantics from the RFC. See: [funcol-rfc-93173](../resources/funcol-rfc-93173.md).

- **Prove you fixed the right thing (graph-break hygiene)**:
  - Run `scope-drd/scripts/tp_compile_repro.py` with `TORCH_LOGS=graph_breaks` and verify you have **0 collective-induced graph breaks**.
  - If you see breaks, attribute them first (don’t tune kernels while you’re fragmented).
  - Use `TORCH_LOGS=guards,recompiles` if the perf issue smells like specialization churn. See: [dynamo-deep-dive](../resources/dynamo-deep-dive.md).

- **Enforce rank symmetry**:
  - Keep shapes/flags identical across ranks inside compiled regions that execute collectives.
  - Avoid rank-dependent control flow (even “harmless” logging/if-statements) in compiled distributed code. See: [ezyang-state-of-compile](../resources/ezyang-state-of-compile.md).

### Gotchas and failure modes

- **Eager regression trap**: funcol can be slower in eager (extra allocations/wrapping). We measured **18 FPS** in Run 12a when funcol was used unconditionally in eager. Gate it. See: [funcol-rfc-93173](../resources/funcol-rfc-93173.md).

- **Forgetting to consume `wait_tensor`’s return value**: the RFC’s semantics are explicit — `wait` returns a new tensor whose storage is safe. If you ignore the return, you can accidentally keep propagating the pre-wait value and confuse downstream assumptions. See: [funcol-rfc-93173](../resources/funcol-rfc-93173.md).

- **Mixing functional and in-place collectives inside the same compiled region**: one “escaped” in-place collective wrapped in `dynamo.disable()` is enough to reintroduce graph breaks and fracture the graph back into tiny pieces. See: [dynamo-deep-dive](../resources/dynamo-deep-dive.md).

- **Rank-divergent compilation / specialization**: if different ranks take different guard paths or see different shapes, compilation decisions can diverge and you can deadlock around collectives (same symptom class as “different collective order”). This is a distributed stability issue, not just a perf issue. See: [ezyang-state-of-compile](../resources/ezyang-state-of-compile.md), [dynamo-deep-dive](../resources/dynamo-deep-dive.md).

- **Misunderstanding what funcol does not do**: functional collectives make tracing possible and reduce graph breaks; they do not automatically add multi-stream overlap in Inductor (explicitly a non-goal of the RFC). See: [funcol-rfc-93173](../resources/funcol-rfc-93173.md).

### Experiments to run

1. **Reproduce the “graph break cliff”**:
   - Compile + in-place collectives behind `torch._dynamo.disable()` (the pre-funcol behavior).
   - Confirm **~160 graph breaks/forward** and measure throughput (expected order: **~9.6 FPS**, Runs 8–9b).

2. **Confirm funcol fixes the right bottleneck**:
   - Compile + funcol in the same harness.
   - Confirm collective-induced graph breaks go to ~0 and measure throughput (expected order: **~24.5 FPS**, Run 12b).

3. **Quantify the eager regression trap**:
   - Run eager with and without funcol (expected order: funcol-eager can regress; Run 12a was **18 FPS**).
   - Use this as a guardrail test for `_maybe_all_reduce()` gating.

4. **Inspect “one big graph vs many small graphs”**:
   - Use `TORCH_LOGS=graph_breaks` and `TORCH_LOGS=graph_code` to sanity check fragmentation.
   - The goal is not “0 graphs” but “no fragmentation from collectives”; breaks should be attributable to known non-collective issues only. See: [dynamo-deep-dive](../resources/dynamo-deep-dive.md).
