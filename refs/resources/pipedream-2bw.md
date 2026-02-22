# Memory-Efficient Pipeline-Parallel DNN Training (PipeDream-2BW)

| Field | Value |
|-------|-------|
| Source | https://arxiv.org/abs/2006.09503 |
| Type | paper |
| Topics | 13 |
| Authors | Narayanan et al. |
| Year | 2020 |
| Status | stub |

## Why it matters

Introduces PipeDream-Flush (synchronous 1F1B) and double-buffered weight updates. The 1F1B schedule from this paper is what most modern systems actually implement -- it interleaves one forward and one backward per micro-batch, reducing peak activation memory from O(num_microbatches) to O(num_stages) compared to GPipe.

## Core claims

<!-- Each claim should cite evidence: sources/pipedream-2bw/extracted.md#<heading> or page number -->

1. **Claim**: ...
   **Evidence**: *pending extraction*

2. **Claim**: ...
   **Evidence**: *pending extraction*

## Key technical details

<!-- Formulas, algorithms, architecture specifics -->

## Actionables / gotchas

<!-- Implementation implications for distributed video DiT inference -->

## Related resources

- [gpipe](gpipe.md) -- foundational synchronous PP that PipeDream-2BW improves upon
- [zero-bubble-pp](zero-bubble-pp.md) -- further improves on 1F1B by splitting backward into B and W
- [pytorch-pipelining-api](pytorch-pipelining-api.md) -- Schedule1F1B implements the 1F1B schedule from this paper
- [pipedit](pipedit.md) -- applies pipeline parallelism to DiT inference
