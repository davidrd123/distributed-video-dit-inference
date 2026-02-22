# Topic 6: CUDA graphs — what they capture, what breaks them, relation to torch.compile

CUDA graphs eliminate kernel launch overhead by recording a sequence of GPU operations and replaying them. They are **critical for inference latency** but impose strict constraints: all tensor addresses must be fixed, control flow cannot change, and memory allocation patterns must be static.

## Resources

<!-- Resource IDs from manifest belonging to this topic -->
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| cuda-graphs-guide | CUDA Programming Guide: CUDA Graphs | high | pending |
| pytorch-cuda-semantics | PyTorch CUDA Semantics | high | pending |
| torch-compile-api | torch.compile API | medium | pending |
| nccl-cuda-graphs | Using NCCL with CUDA Graphs | medium | pending |

## Synthesis

<!-- To be filled during study -->

### Mental model

### Key concepts

### Cross-resource agreement / disagreement

### Practical checklist

### Gotchas and failure modes

### Experiments to run
