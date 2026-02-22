# StreamDiffusionV2

| Field | Value |
|-------|-------|
| Source | https://streamdiffusionv2.github.io/, https://github.com/chenfengxu714/StreamDiffusionV2 |
| Type | paper |
| Topics | 24 |
| Authors | Feng et al. |
| Year | 2025 |
| Status | stub |

## Why it matters

The most directly relevant system to the project. Extends streaming inference to video diffusion models with SLO-aware batching, a block scheduler, sink-token rolling KV cache, and motion-aware noise. Achieves 58.28 FPS with a 14B model on 4xH100. Demonstrates that real-time video DiT inference is achievable with the right scheduling and caching strategies.

## Core claims

<!-- Each claim should cite evidence: sources/streamdiffusionv2/extracted.md#<heading> or page number -->

1. **Claim**: ...
   **Evidence**: *pending extraction*

2. **Claim**: ...
   **Evidence**: *pending extraction*

## Key technical details

<!-- Formulas, algorithms, architecture specifics -->

## Actionables / gotchas

<!-- Implementation implications for distributed video DiT inference -->

## Related resources

- [dit-paper](dit-paper.md) -- the DiT architecture that StreamDiffusionV2 runs
- [pipedit](pipedit.md) -- complementary approach using pipelined sequence parallelism
- [pagedattention](pagedattention.md) -- paged memory management for KV caches
- [pytorch-pipelining-api](pytorch-pipelining-api.md) -- PyTorch pipeline schedules
- [making-dl-go-brrrr](making-dl-go-brrrr.md) -- performance analysis framework
