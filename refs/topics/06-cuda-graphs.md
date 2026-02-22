# Topic 6: CUDA graphs — what they capture, what breaks them, relation to torch.compile

CUDA graphs eliminate kernel launch overhead by recording a sequence of GPU operations and replaying them. They are **critical for inference latency** but impose strict constraints: all tensor addresses must be fixed, control flow cannot change, and memory allocation patterns must be static.

## Resources

<!-- Resource IDs from manifest belonging to this topic -->
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| cuda-graphs-guide | CUDA Programming Guide: CUDA Graphs | high | converted |
| pytorch-cuda-semantics | PyTorch CUDA Semantics | high | converted |
| torch-compile-api | torch.compile API | medium | pending |
| nccl-cuda-graphs | Using NCCL with CUDA Graphs | medium | pending |

## Implementation context

Single-GPU optimization on H200 already tried `torch.compile` “reduce-overhead” / CUDAGraph variants, but CUDAGraph capture was reported as **unstable or neutral**. In TP mode, capture becomes higher risk because a mid-capture failure can strand peers at collectives; the v0 roadmap therefore treats “CUDAGraph Trees (non-collective regions only)” as a separate v2.0 milestone and focuses on compile hygiene first (functional collectives + lockstep warmup).

See: `refs/implementation-context.md`, `scope-drd/notes/FA4/h200/tp/streamdiffusion-v2-analysis-opus.md` (What we already tried), `scope-drd/notes/FA4/h200/tp/explainers/07-v0-contract.md` (Roadmap: v2.0 CUDAGraph Trees).

## Synthesis

<!-- To be filled during study -->

### Mental model

### Key concepts

### Cross-resource agreement / disagreement

### Practical checklist

### Gotchas and failure modes

### Experiments to run
