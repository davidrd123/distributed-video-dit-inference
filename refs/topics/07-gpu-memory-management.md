# Topic 7: GPU memory management — PyTorch caching allocator, fragmentation

The PyTorch caching allocator avoids expensive `cudaMalloc`/`cudaFree` calls by maintaining a pool of allocated blocks. Fragmentation under dynamic workloads (variable sequence lengths, different denoising steps) is the primary operational concern.

## Resources

<!-- Resource IDs from manifest belonging to this topic -->
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| pytorch-cuda-semantics | PyTorch CUDA Semantics | high | pending |
| pytorch-cuda-module | torch.cuda Module Reference | medium | pending |

## Synthesis

<!-- To be filled during study -->

### Mental model

### Key concepts

### Cross-resource agreement / disagreement

### Practical checklist

### Gotchas and failure modes

### Experiments to run
