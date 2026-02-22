# Making Deep Learning Go Brrrr From First Principles

| Field | Value |
|-------|-------|
| Source | https://horace.io/brrr_intro.html |
| Type | blog |
| Topics | 8, 16 |
| Author | Horace He |
| Status | stub |

## Why it matters

The definitive blog post explaining overhead-bound, memory-bound, and compute-bound regimes. The overhead section explains exactly why Python dispatch matters (~10us per kernel launch) and how operator fusion eliminates it. Also the single best resource for building roofline intuition -- the starting point for any GPU performance reasoning about DiT attention (memory-bandwidth-bound) vs FFN layers (compute-bound).

## Core claims

1. **Claim**: ...
   **Evidence**: *pending extraction*

## Key insights

<!-- The distinctive contribution of this piece -->

## Actionables / gotchas

<!-- Implementation implications for distributed video DiT inference -->

## Related resources

- [pytorch-cuda-semantics](pytorch-cuda-semantics.md) -- CUDA execution model underlying the performance regimes
- [cuda-graphs-guide](cuda-graphs-guide.md) -- CUDA graphs eliminate the kernel launch overhead described here
- [dynamo-deep-dive](dynamo-deep-dive.md) -- torch.compile addresses overhead-bound regime via tracing
- [dit-paper](dit-paper.md) -- the DiT architecture to apply roofline analysis to
- [ezyang-state-of-compile](ezyang-state-of-compile.md) -- compile as overhead elimination strategy
