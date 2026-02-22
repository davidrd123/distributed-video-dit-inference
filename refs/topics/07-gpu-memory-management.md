# Topic 7: GPU memory management — PyTorch caching allocator, fragmentation

The PyTorch caching allocator avoids expensive `cudaMalloc`/`cudaFree` calls by maintaining a pool of allocated blocks. Fragmentation under dynamic workloads (variable sequence lengths, different denoising steps) is the primary operational concern.

## Resources

<!-- Resource IDs from manifest belonging to this topic -->
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| pytorch-cuda-semantics | PyTorch CUDA Semantics | high | converted |
| pytorch-cuda-module | torch.cuda Module Reference | medium | pending |

## Implementation context

TP head-sharding halves KV-cache memory per rank: per layer ~335 MB → ~167 MB, so K+V across 40 layers is ~26 GB → ~13 GB per rank (TP=2). In the first stable server run (Run 7), both ranks still sat at ~55.2 GiB / 79.6 GiB used, so allocator stability and “don’t allocate per chunk” discipline matter. This memory pressure is a key motivator for v1.1c: generator-only workers are only worth the risk if worker memory drops by ≥10 GB or ≥15% and startup improves materially.

See: `refs/implementation-context.md`, `scope-drd/notes/FA4/h200/tp/explainers/05-kv-cache-head-sharding.md` (cache sizing), `scope-drd/notes/FA4/h200/tp/bringup-run-log.md` (Run 7 nvtop memory), `scope-drd/notes/FA4/h200/tp/v1.1-generator-only-workers.md` (acceptance gates).

## Synthesis

<!-- To be filled during study -->

### Mental model

### Key concepts

### Cross-resource agreement / disagreement

### Practical checklist

### Gotchas and failure modes

### Experiments to run
