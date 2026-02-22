# Zero Bubble Pipeline Parallelism

| Field | Value |
|-------|-------|
| Source | https://arxiv.org/abs/2401.10241 |
| Type | paper |
| Topics | 14 |
| Authors | Qi et al. |
| Year | 2024 |
| Status | stub |

## Why it matters

Achieves zero pipeline bubbles by splitting backward into B (input gradient) and W (parameter gradient) phases. The key insight is that W has no data dependency on the next micro-batch, so it can be scheduled to fill what would otherwise be bubble time. Up to 23--31% throughput improvement over 1F1B.

## Core claims

<!-- Each claim should cite evidence: sources/zero-bubble-pp/extracted.md#<heading> or page number -->

1. **Claim**: ...
   **Evidence**: *pending extraction*

2. **Claim**: ...
   **Evidence**: *pending extraction*

## Key technical details

<!-- Formulas, algorithms, architecture specifics -->

## Actionables / gotchas

<!-- Implementation implications for distributed video DiT inference -->

## Related resources

- [gpipe](gpipe.md) -- foundational synchronous PP schedule
- [pipedream-2bw](pipedream-2bw.md) -- the 1F1B schedule that zero-bubble improves upon
- [pytorch-pipelining-api](pytorch-pipelining-api.md) -- ScheduleInterleavedZeroBubble and ScheduleZBVZeroBubble implement this
- [pipedit](pipedit.md) -- pipeline-parallel DiT inference
