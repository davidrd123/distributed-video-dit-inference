# Topic 8: Kernel launch overhead — Python-to-GPU dispatch path

Each PyTorch operator call traverses Python -> C++ dispatch -> CUDA kernel launch. At **~10us per launch**, this overhead dominates when running many small operations. `torch.compile` and CUDA graphs are the primary mitigations.

## Resources

<!-- Resource IDs from manifest belonging to this topic -->
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| making-dl-go-brrrr | Making Deep Learning Go Brrrr From First Principles | high | pending |
| gpu-mode-lecture-6 | GPU MODE Lecture 6: Optimizing Optimizers in PyTorch | low | link_only |
| pytorch-internals-ezyang | PyTorch internals | medium | pending |

## Synthesis

<!-- To be filled during study -->

### Mental model

### Key concepts

### Cross-resource agreement / disagreement

### Practical checklist

### Gotchas and failure modes

### Experiments to run
