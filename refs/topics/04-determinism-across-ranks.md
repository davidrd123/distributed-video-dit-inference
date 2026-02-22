# Topic 4: Determinism across ranks in distributed training/inference

Non-determinism in distributed settings comes from three sources: **CUDA atomicAdd operations** (non-associative floating-point), **cuDNN autotuning** selecting different algorithms per run, and **NCCL reduction order** across ranks.

## Resources

<!-- Resource IDs from manifest belonging to this topic -->
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| pytorch-reproducibility | PyTorch Reproducibility Guide | low | pending |
| pytorch-deterministic-api | torch.use_deterministic_algorithms API | low | pending |
| fp-non-assoc-reproducibility | Impacts of floating-point non-associativity on reproducibility for HPC and deep learning | low | pending |

## Synthesis

<!-- To be filled during study -->

### Mental model

### Key concepts

### Cross-resource agreement / disagreement

### Practical checklist

### Gotchas and failure modes

### Experiments to run
