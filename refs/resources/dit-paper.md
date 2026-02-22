# Scalable Diffusion Models with Transformers (DiT)

| Field | Value |
|-------|-------|
| Source | https://arxiv.org/abs/2212.09748 |
| Type | paper |
| Topics | 24 |
| Authors | Peebles & Xie |
| Year | 2023 |
| Status | stub |

## Why it matters

The foundational DiT paper. Replaces U-Net with transformers in latent diffusion, demonstrating strong correlation between compute (Gflops) and sample quality. This is the architecture the inference system will be running -- understanding its structure (patchification, adaptive layer norm, conditioning mechanisms) is essential for partitioning it across pipeline stages.

## Core claims

<!-- Each claim should cite evidence: sources/dit-paper/extracted.md#<heading> or page number -->

1. **Claim**: ...
   **Evidence**: *pending extraction*

2. **Claim**: ...
   **Evidence**: *pending extraction*

## Key technical details

<!-- Formulas, algorithms, architecture specifics -->

## Actionables / gotchas

<!-- Implementation implications for distributed video DiT inference -->

## Related resources

- [streamdiffusionv2](streamdiffusionv2.md) -- extends streaming inference to video diffusion models including DiT
- [pipedit](pipedit.md) -- directly addresses pipeline-parallel DiT inference
- [pagedattention](pagedattention.md) -- memory management relevant to DiT's attention KV caches
- [making-dl-go-brrrr](making-dl-go-brrrr.md) -- performance analysis framework for DiT's compute/memory characteristics
