# Topic 2: Deadlock patterns in multi-group distributed code

Deadlocks in multi-group code almost always stem from mismatched collective ordering across ranks or accidentally issuing operations on the wrong process group. `NCCL_DEBUG=INFO` (or `WARN`) is your primary diagnostic tool.

## Resources

<!-- Resource IDs from manifest belonging to this topic -->
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| pytorch-distributed-api | PyTorch Distributed API Reference | medium | pending |
| pytorch-dist-tutorial | Writing Distributed Applications with PyTorch | medium | pending |
| nccl-fault-tolerance | Building Scalable and Fault-Tolerant NCCL Applications | low | pending |

## Synthesis

<!-- To be filled during study -->

### Mental model

### Key concepts

### Cross-resource agreement / disagreement

### Practical checklist

### Gotchas and failure modes

### Experiments to run
