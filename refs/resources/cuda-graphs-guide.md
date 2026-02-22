# CUDA Programming Guide: CUDA Graphs

| Field | Value |
|-------|-------|
| Source | https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html |
| Type | docs |
| Topics | 6 |
| Status | stub |

## Why it matters

CUDA graphs eliminate kernel launch overhead by recording a sequence of GPU operations and replaying them. They are critical for inference latency but impose strict constraints: all tensor addresses must be fixed, control flow cannot change, and memory allocation patterns must be static. Complete coverage of graph definition vs execution, stream capture, memory allocations within graphs, instantiation, and dependency management.

## Key sections

<!-- Most relevant sections/pages for the project -->

## Core claims

1. **Claim**: ...
   **Evidence**: *pending extraction*

## API surface / configuration

<!-- Key APIs, env vars, config options relevant to the project -->

## Actionables / gotchas

<!-- Implementation implications for distributed video DiT inference -->

## Related resources

- [pytorch-cuda-semantics](pytorch-cuda-semantics.md) -- PyTorch CUDA graph integration, graph-private memory pools
- [nccl-user-guide](nccl-user-guide.md) -- capturing NCCL operations in CUDA graphs
- [dynamo-deep-dive](dynamo-deep-dive.md) -- torch.compile mode="reduce-overhead" uses CUDA graphs
- [making-dl-go-brrrr](making-dl-go-brrrr.md) -- kernel launch overhead that CUDA graphs eliminate
