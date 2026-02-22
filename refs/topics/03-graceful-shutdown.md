# Topic 3: Graceful shutdown and draining in distributed PyTorch

Graceful shutdown in distributed PyTorch remains **underserved by documentation**. The core API is `destroy_process_group()`, but real-world challenges include ranks exiting at different times, CUDA graph capture preventing clean NCCL communicator destruction, and signal handling under `torchrun`.

## Resources

<!-- Resource IDs from manifest belonging to this topic -->
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| pytorch-distributed-api | PyTorch Distributed API Reference | medium | pending |
| pytorch-issue-115388 | GitHub Issue #115388: destroy_process_group() hangs after CUDA graph capture | medium | pending |
| pytorch-issue-167775 | GitHub Issue #167775: Graceful Ctrl+C handling from torchrun | low | pending |
| kill-pytorch-dist | Kill PyTorch Distributed Training Processes | low | pending |

## Synthesis

<!-- To be filled during study -->

### Mental model

### Key concepts

### Cross-resource agreement / disagreement

### Practical checklist

### Gotchas and failure modes

### Experiments to run
