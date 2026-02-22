# Topic 16: Roofline model applied to transformers — arithmetic intensity for attention vs FFN

The roofline model plots achievable FLOPS against **arithmetic intensity** (FLOPS/byte of memory traffic). Attention is typically **memory-bandwidth-bound** (low arithmetic intensity due to reading/writing large KV matrices), while FFN layers are more likely **compute-bound** (high arithmetic intensity from large matrix multiplications). This distinction drives optimization strategy.

## Resources

<!-- Resource IDs from manifest belonging to this topic -->
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| making-dl-go-brrrr | Making Deep Learning Go Brrrr From First Principles | high | pending |
| roofline-paper | Roofline: An Insightful Visual Performance Model for Multicore Architectures | medium | link_only |
| nvidia-gpu-perf-guide | NVIDIA GPU Performance Background User's Guide | medium | pending |
| gpu-mode-lecture-8 | GPU MODE Lecture 8: CUDA Performance Checklist | low | link_only |

## Synthesis

<!-- To be filled during study -->

### Mental model

### Key concepts

### Cross-resource agreement / disagreement

### Practical checklist

### Gotchas and failure modes

### Experiments to run
