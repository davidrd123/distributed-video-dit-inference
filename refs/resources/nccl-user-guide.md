# NCCL User Guide

| Field | Value |
|-------|-------|
| Source | https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html, https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html |
| Type | docs |
| Topics | 1 |
| Status | stub |

## Why it matters

Covers communicators, collective operations, CUDA stream semantics, group calls, and algorithm selection. The environment variables page documents `NCCL_DEBUG`, `NCCL_ALGO`, and `NCCL_PROTO` for controlling and debugging algorithm choice -- essential for diagnosing why pipeline send/recv calls behave differently across NVLink vs PCIe topologies.

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

- [pytorch-cuda-semantics](pytorch-cuda-semantics.md) -- CUDA stream semantics interact with NCCL stream behavior
- [cuda-graphs-guide](cuda-graphs-guide.md) -- NCCL operations inside CUDA graphs have specific requirements
- [funcol-rfc-93173](funcol-rfc-93173.md) -- functional collectives wrap NCCL operations for traceability
