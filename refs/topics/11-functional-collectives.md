# Topic 11: Functional collectives and Dynamo — why in-place breaks tracing, the funcol solution

Standard NCCL collectives (`all_reduce`, `all_gather`) are **in-place and side-effecting** — they mutate tensors and return opaque `Work` objects. This is fundamentally incompatible with functional graph tracing. Functional collectives (`torch.distributed._functional_collectives`) return new tensors and use `AsyncCollectiveTensor` subclasses for deferred synchronization, making them traceable by Dynamo.

## Resources

<!-- Resource IDs from manifest belonging to this topic -->
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| funcol-rfc-93173 | RFC #93173: PT2-Friendly Traceable, Functional Collective Communication APIs | high | pending |
| funcol-source | _functional_collectives.py source | medium | pending |
| ezyang-state-of-compile | State of torch.compile for training (August 2025) | high | pending |
| pytorch-issue-138773 | GitHub Issue #138773: functional collectives 67% slower than torch.distributed | medium | pending |

## Synthesis

<!-- To be filled during study -->

### Mental model

### Key concepts

### Cross-resource agreement / disagreement

### Practical checklist

### Gotchas and failure modes

### Experiments to run
