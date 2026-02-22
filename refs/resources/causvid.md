# From Slow Bidirectional to Fast Autoregressive Video Diffusion Models

| Field | Value |
|-------|-------|
| Source | https://arxiv.org/abs/2412.07772 |
| Type | paper |
| Topics | 22, 24 |
| Authors | Yin et al. |
| Year | 2025 |
| Status | stub |

## Why it matters

CausVid is a concrete blueprint for turning a bidirectional video DiT into a **causal/autoregressive** generator so you can use **KV caching** and stream frames with low initial latency. This is directly relevant to Scope’s PP0+recompute/KV-cache lifecycle and to any plan that moves from “full-window denoise” to “rolling-window AR” scheduling.

## Key sections

<!-- Fill with links into sources/causvid/full.md headings. -->

## Core claims

<!-- Each claim must cite Evidence: sources/causvid/full.md#<heading> -->

1. **Claim**: ...
   **Evidence**: *pending extraction*

2. **Claim**: ...
   **Evidence**: *pending extraction*

## Key technical details

<!-- Causal attention mask, distillation recipe, inference loop + KV cache. -->

## Actionables / gotchas

<!-- Implementation implications for distributed video DiT inference (cache cut/recompute, frame-chunking, scheduling). -->

## Related resources

- [streamdiffusionv2](streamdiffusionv2.md) — rolling KV cache + streaming scheduling patterns; references CausVid in codebase.
- [pagedattention](pagedattention.md) — KV cache memory management patterns.
- [dit-paper](dit-paper.md) — baseline DiT architecture that CausVid modifies.
- [pipedit](pipedit.md) — complementary scheduling/partitioning approach for video DiT.
