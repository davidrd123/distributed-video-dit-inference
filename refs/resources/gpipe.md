# GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism

| Field | Value |
|-------|-------|
| Source | https://arxiv.org/abs/1811.06965 |
| Type | paper |
| Topics | 13 |
| Authors | Huang et al. |
| Year | 2018 |
| Status | stub |

## Why it matters

The foundational synchronous pipeline parallelism paper. Introduces micro-batch splitting and re-materialization (activation checkpointing) for memory efficiency. GPipe fills the pipeline with micro-batches and synchronizes at the end -- simple but high memory. Understanding GPipe is prerequisite to understanding the 1F1B and zero-bubble schedules that modern systems actually implement.

## Core claims

<!-- Each claim should cite evidence: sources/gpipe/extracted.md#<heading> or page number -->

1. **Claim**: ...
   **Evidence**: *pending extraction*

2. **Claim**: ...
   **Evidence**: *pending extraction*

## Key technical details

<!-- Formulas, algorithms, architecture specifics -->

## Actionables / gotchas

<!-- Implementation implications for distributed video DiT inference -->

## Related resources

- [pipedream-2bw](pipedream-2bw.md) -- introduces 1F1B schedule that reduces GPipe's memory overhead
- [zero-bubble-pp](zero-bubble-pp.md) -- eliminates pipeline bubbles by splitting backward into B and W phases
- [pytorch-pipelining-api](pytorch-pipelining-api.md) -- ScheduleGPipe implements this paper's approach
- [pipedit](pipedit.md) -- pipeline-parallel DiT inference building on PP foundations
