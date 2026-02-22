---
status: draft
---

# Topic 10: Inductor fusion rules — what fuses, verification

TorchInductor's scheduler decides fusion using `score_fusion(node1, node2)`, which scores pairs of operations by **estimated memory traffic savings**. Pointwise-to-pointwise fusion is most common; reduction and template fusions have additional constraints.

## Resources

<!-- Resource IDs from manifest belonging to this topic -->
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| torchinductor-design | TorchInductor: a PyTorch-native Compiler with Define-by-Run IR and Symbolic Shapes | medium | pending |
| inductor-config | TorchInductor config.py | medium | pending |
| pytorch2-asplos | PyTorch 2: Faster Machine Learning Through Dynamic Python (ASPLOS 2024) | medium | pending |
| inductor-fusion-discussion | Inductor scheduler source and fusion discussion | low | pending |

## Implementation context

Run 12b’s performance win depended on Inductor being able to fuse through the TP block when collectives are traceable: functional collectives eliminate graph breaks and enable “one compiled graph per block” behavior. The regression harness treats this as an invariant: `tp_compile_repro.py` expects **mode_C graph_break_count=0** and **graph_count=1**. Any new graph breaks in the collective wrappers tend to push the system back into many tiny graphs and overhead-bound performance (Runs 8-9b).

See: `refs/implementation-context.md` → Phase 1, `scope-drd/notes/FA4/h200/tp/bringup-run-log.md` (Runs 8-12b), `scope-drd/notes/FA4/h200/tp/research-program.md` (compile micro-repro).

## Synthesis

### Mental model

- **Dynamo vs Inductor (separate the failure modes)**:
  - **Dynamo** decides *what becomes a compiled graph* (FX capture) and where **graph breaks** happen.
  - **Inductor** decides *how to lower that graph* and how aggressively to **fuse** operations into fewer kernels.
  - If you have many graph breaks, you’re overhead-bound regardless of Inductor fusion quality (Runs 8–9b). If you have a stable single graph per block, Inductor fusion becomes the next-order lever. (See `scope-drd/notes/FA4/h200/tp/bringup-run-log.md` Runs 8–12b; and `scope-drd/notes/FA4/h200/tp/research-program.md` compile gates.)
- **What fusion buys you** (two different wins):
  1. **Fewer kernel launches** → lower dispatch/launch overhead (Topic 08).
  2. **Fewer round-trips to HBM** → less memory traffic and higher arithmetic intensity (Topic 16/18).
- **What fusion cannot do**:
  - It cannot fuse across graph boundaries (graph breaks) or across true synchronization boundaries (e.g., collective ordering boundaries).
  - It cannot violate correctness constraints around mutation/aliasing; many “it should fuse” expectations fail because there is an in-place update, a view alias, or an ordering dependency.
- **Pragmatic operator framing for Scope**:
  - Our “compile success” is not “some code compiled.” It’s “the hot TP block is one compiled region with no collective-induced breaks.”
  - That condition is necessary for Inductor to even have a chance to fuse the block into a small number of kernels. (Micro-repro invariant: `tp_compile_repro.py` mode C expects `graph_break_count=0` and `graph_count=1`.)

### Key concepts

- **Fusion boundaries** (the things that prevent fusion even inside a single graph):
  - **Mutations / in-place ops**: Inductor must preserve program order for side effects; functional forms are easier to schedule/fuse.
  - **Aliasing / views**: if two tensors share storage or a view is used in a way that introduces overlap, Inductor must be conservative.
  - **Reductions / scans**: reductions can fuse with surrounding pointwise ops only when the producer/consumer pattern is compatible.
  - **Dynamic shapes / symbolic guards**: if shape-dependent control flow causes specialization churn, you may see many variants instead of one stable fused kernel sequence.
- **Pointwise fusion is the baseline**: chains like `add → mul → relu → scale → cast` usually fuse well and are the “first win” you should expect once the region is stable.
- **Template / epilogue fusion**: in the best case, Inductor can fuse pointwise epilogues into GEMMs (and/or fuse small layout transforms), reducing intermediate writes.
- **“One compiled graph” does not mean “one kernel”**: a block can compile as one FX graph but still lower into multiple kernels (e.g., separate GEMMs, attention kernels, reductions). Your goal is to reduce *unnecessary* kernel boundaries (tiny pointwise kernels between large ops) rather than expecting a monolith.
- **Manual fused kernels vs Inductor fusion**: some parts of our stack already use hand-written fused kernels (e.g., Triton fast-paths). These can be faster than generic fusion but come with correctness constraints (see “Triton QKV pack + RMSNorm disabled in TP” below).

### Cross-resource agreement / disagreement

- **Agreement (measured in bringup)**:
  - The huge throughput delta (9.6 → 24.5 FPS) came from removing the graph-break source on collectives, which restored large compiled regions. This matches the “overhead-bound collapse” diagnosis in Topic 08 and the “graph breaks are the perf killer” story in `refs/resources/dynamo-deep-dive.md`. (`scope-drd/notes/FA4/h200/tp/bringup-run-log.md` Runs 8–12b.)
  - The functional-collectives change is also a fusion enabler: by removing `torch._dynamo.disable()` boundaries and avoiding in-place collective side effects in compiled mode, Inductor can schedule/fuse around collectives rather than fragmenting. (`refs/resources/funcol-rfc-93173.md`, `scope-drd/notes/FA4/h200/tp/explainers/04-tp-math.md`.)
- **Nuance**:
  - “Fusion is always good” is false: bigger fused kernels can increase register pressure and reduce occupancy, or increase compile time. The right heuristic is: fuse away tiny pointwise kernels and redundant intermediates; don’t chase “maximum fusion” as a goal.
  - Some optimizations are outside Inductor’s scope because they’re implemented as separate backend kernels (FlashAttention/FA4/CUTLASS). Inductor may still fuse pre/post pointwise around them, but it won’t “fuse attention into a GEMM.”

### Practical checklist

- **Step 0: Confirm you have the right precondition** (no fragmentation):
  - Run `tp_compile_repro.py` and verify the “no breaks, one graph” invariant in the mode that exercises your TP collectives under compile.
  - If `graph_break_count` regresses, you are not debugging Inductor fusion; you are debugging Dynamo/guard/break behavior first. (`scope-drd/notes/FA4/h200/tp/research-program.md`.)
- **Step 1: Verify fusion by *kernel count and bubbles*, not by wishful thinking**:
  - Use a profiler (or even coarse per-op counters) to see whether the compiled region executes as a small number of meaningful kernels vs many tiny pointwise kernels.
  - If you see “lots of tiny kernels,” treat it as a fusion/fragmentation symptom and go look for the boundary (in-place ops, side effects, graph breaks).
- **Step 2: Prefer functional / out-of-place patterns in compiled hot paths**:
  - Eager can use in-place collectives for performance; compiled code should use functional collectives (return-new-tensor) to stay traceable and fusion-friendly. (Run 12a vs 12b; see `refs/implementation-context.md` and `scope-drd/notes/FA4/h200/tp/bringup-run-log.md`.)
- **Step 3: Treat “fused fast paths” as correctness-sensitive in TP**:
  - Disable fused Triton QKV-pack+RMSNorm kernels in TP bringup unless they are explicitly TP-aware (head sharding + distributed RMSNorm). This is a known landmine: the single-GPU fused kernels assume full-dim tensors and are disabled in TP mode for correctness. (`scope-drd/notes/FA4/h200/tp/explainers/04-tp-math.md` “Triton fast-path and TP”, `scope-drd/notes/FA4/h200/tp/explainers/07-v0-contract.md`, and `scope-drd/notes/FA4/h200/tp/feasibility.md` bringup hygiene.)
- **Step 4: When perf changes, classify the cause**:
  - Break/regression only in compile mode → likely fusion/compile boundary issue, not algorithm.
  - Perf regresses with same graph count → suspect worse fusion, worse kernel choice/autotune, or increased register pressure.
  - Perf regresses with more graphs/breaks → you’re back in overhead-bound territory; fix breaks first.

### Gotchas and failure modes

- **Mistaking “compiled” for “fused”**: you can have `graph_count=1` but still run a lot of kernels; the right question is “did we remove *avoidable* kernel boundaries and HBM round-trips?”
- **In-place updates block reordering**: if a tensor is mutated and later read, Inductor must preserve order and may lose fusion opportunities.
- **Hidden host-side work creates fusion illusions**: printing/logging/debug callbacks can induce graph breaks or prevent fusion; treat “debug inside hot path” as a first-class perf hazard. (`refs/resources/dynamo-deep-dive.md`.)
- **TP correctness hazards from fused kernels**: a fused path that assumes full dim or local-only normalization can produce “plausible but wrong” outputs in TP (Franken-model). Keep fused fast paths off until proven TP-correct; rely on digests/fingerprints for detection. (`scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md` Q2/Q6.)
- **Distributed symmetry still dominates**: even “just changing fusion” can change control flow (guards/recompiles) if it introduces dynamic shape dependence; treat rank-divergent compilation/fusion as a correctness risk (NCCL ordering hangs). (`refs/resources/ezyang-state-of-compile.md`, `deep-research/2026-02-22/compile-distributed-hardening/reply.md`.)

### Experiments to run

- **Fusion sanity via kernel count**: capture a profiler trace of a single hot block in eager vs compile mode and compare the number of kernels and idle bubbles; the expected win is “fewer small pointwise kernels and less CPU-side overhead.” (Tie to Runs 8–12b story.)
- **Boundary attribution**: if you see unexpected small kernels between large ops, bisect by:
  - removing in-place ops (temporarily, test-only),
  - eliminating Python-side side effects,
  - simplifying view/reshape chains,
  and see whether kernel count drops.
- **Fast-path correctness A/B**: toggle a fused fast path (e.g., Triton QKV pack) in a TP correctness harness and verify it either (a) matches the reference exactly, or (b) is banned. Don’t accept “looks okay” for TP. (`scope-drd/notes/FA4/h200/tp/explainers/07-v0-contract.md` tripwires; `scope-drd/notes/FA4/h200/tp/feasibility.md`.)
- **Compile regression gate**: keep `tp_compile_repro.py` mode C as the “stop the line” test; if fusion/compile refactors regress graph count/breaks, treat as perf regression until understood. (`scope-drd/notes/FA4/h200/tp/research-program.md`.)
