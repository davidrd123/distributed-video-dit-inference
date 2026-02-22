# Topic 18: Bandwidth accounting from first principles

Bandwidth accounting means computing the **theoretical minimum memory traffic** for an operation, then comparing against measured bandwidth to determine utilization. For a matrix multiply C = A x B with dimensions (M, K) x (K, N), minimum traffic is `2(MK + KN + MN)` bytes (read A, read B, write C).

## Resources

<!-- Resource IDs from manifest belonging to this topic -->
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| nvidia-gpu-perf-guide | NVIDIA GPU Performance Background User's Guide | medium | pending |
| modal-memory-bandwidth | What is memory bandwidth? (Modal GPU Glossary) | low | pending |
| modal-roofline | What is the roofline model? (Modal GPU Glossary) | low | pending |
| matmul-shapes | What Shapes Do Matrix Multiplications Like? | medium | pending |

## Synthesis

<!-- To be filled during study -->

### Mental model

### Key concepts

### Cross-resource agreement / disagreement

### Practical checklist

### Gotchas and failure modes

### Experiments to run
