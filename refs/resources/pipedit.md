# PipeDiT: Accelerating DiT in Video Generation with Pipelining and Decoupling

| Field | Value |
|-------|-------|
| Source | https://arxiv.org/abs/2511.12056 |
| Type | paper |
| Topics | 24 |
| Authors | (see paper) |
| Year | 2025 |
| Status | stub |

## Why it matters

Directly addresses pipeline-parallel DiT inference. Introduces PipeSP (pipelined sequence parallelism), DeDiVAE (decoupled diffusion/VAE onto separate GPU groups), and attention co-processing for idle GPU utilization. The most directly applicable paper for the system architecture being built.

## Core claims

<!-- Each claim should cite evidence: sources/pipedit/extracted.md#<heading> or page number -->

1. **Claim**: ...
   **Evidence**: *pending extraction*

2. **Claim**: ...
   **Evidence**: *pending extraction*

## Key technical details

<!-- Formulas, algorithms, architecture specifics -->

## Actionables / gotchas

<!-- Implementation implications for distributed video DiT inference -->

## Related resources

- [dit-paper](dit-paper.md) -- the DiT architecture that PipeDiT accelerates
- [streamdiffusionv2](streamdiffusionv2.md) -- complementary streaming approach for video DiT
- [gpipe](gpipe.md) -- foundational PP schedule
- [pipedream-2bw](pipedream-2bw.md) -- 1F1B schedule that PipeDiT builds on
- [zero-bubble-pp](zero-bubble-pp.md) -- zero-bubble scheduling techniques
- [pytorch-pipelining-api](pytorch-pipelining-api.md) -- PyTorch PP API for implementation
- [pagedattention](pagedattention.md) -- attention memory management
