# Efficient Memory Management for Large Language Model Serving with PagedAttention

| Field | Value |
|-------|-------|
| Source | https://arxiv.org/abs/2309.06180 |
| Type | paper |
| Topics | 22 |
| Authors | Kwon et al. |
| Year | 2023 |
| Status | stub |

## Why it matters

The seminal paper on paged KV cache management. Applies OS virtual memory concepts (block tables, non-contiguous allocation, copy-on-write sharing) to attention KV caches, reducing memory waste from 60--80% to under 4%. Directly relevant to managing temporal KV caches in video DiT streaming inference, where variable-length sequences and rolling windows create fragmentation pressure.

## Core claims

<!-- Each claim should cite evidence: sources/pagedattention/extracted.md#<heading> or page number -->

1. **Claim**: ...
   **Evidence**: *pending extraction*

2. **Claim**: ...
   **Evidence**: *pending extraction*

## Key technical details

<!-- Formulas, algorithms, architecture specifics -->

## Actionables / gotchas

<!-- Implementation implications for distributed video DiT inference -->

## Related resources

- [dit-paper](dit-paper.md) -- the DiT architecture whose attention KV caches need management
- [streamdiffusionv2](streamdiffusionv2.md) -- sink-token rolling KV cache for video diffusion
- [pipedit](pipedit.md) -- attention co-processing in pipeline-parallel DiT
- [pytorch-cuda-semantics](pytorch-cuda-semantics.md) -- GPU memory management underlying KV cache allocation
