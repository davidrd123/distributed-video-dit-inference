# Topic 11: Functional collectives and Dynamo — why in-place breaks tracing, the funcol solution

Standard NCCL collectives (`all_reduce`, `all_gather`) are **in-place and side-effecting** — they mutate tensors and return opaque `Work` objects. This is fundamentally incompatible with functional graph tracing. Functional collectives (`torch.distributed._functional_collectives`) return new tensors and use `AsyncCollectiveTensor` subclasses for deferred synchronization, making them traceable by Dynamo.

## Resources

<!-- Resource IDs from manifest belonging to this topic -->
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| funcol-rfc-93173 | RFC #93173: PT2-Friendly Traceable, Functional Collective Communication APIs | high | converted |
| funcol-source | _functional_collectives.py source | medium | pending |
| ezyang-state-of-compile | State of torch.compile for training (August 2025) | high | converted |
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

### Key concepts

### Cross-resource agreement / disagreement

### Practical checklist

### Gotchas and failure modes

### Experiments to run
