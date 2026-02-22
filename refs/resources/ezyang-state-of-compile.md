# State of torch.compile for training (August 2025)

| Field | Value |
|-------|-------|
| Source | https://blog.ezyang.com/2025/08/state-of-torch-compile-august-2025/ |
| Type | blog |
| Topics | 11, 12 |
| Author | Edward Yang |
| Status | stub |

## Why it matters

The most comprehensive treatment of functional collectives in context. Covers DTensor compilation, SimpleFSDP, async tensor parallelism as a compiler pass, and honest comparison with JAX's approach. Essential for understanding how torch.compile interacts with distributed code and what the current limitations and workarounds are for compiled distributed inference.

## Core claims

1. **Claim**: ...
   **Evidence**: *pending extraction*

## Key insights

<!-- The distinctive contribution of this piece -->

## Actionables / gotchas

<!-- Implementation implications for distributed video DiT inference -->

## Related resources

- [funcol-rfc-93173](funcol-rfc-93173.md) -- the functional collectives RFC that this post contextualizes
- [dynamo-deep-dive](dynamo-deep-dive.md) -- Dynamo tracing internals referenced throughout
- [pytorch-pipelining-api](pytorch-pipelining-api.md) -- pipeline schedules that interact with compilation
- [making-dl-go-brrrr](making-dl-go-brrrr.md) -- performance motivation for compilation
- [cuda-graphs-guide](cuda-graphs-guide.md) -- CUDA graphs as reduce-overhead compile backend
