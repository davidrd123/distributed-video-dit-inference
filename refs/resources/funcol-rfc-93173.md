# RFC #93173: PT2-Friendly Traceable, Functional Collective Communication APIs

| Field | Value |
|-------|-------|
| Source | https://github.com/pytorch/pytorch/issues/93173 |
| Type | code |
| Topics | 11 |
| Status | stub |

## Why it matters

The foundational design document for functional collectives. Standard NCCL collectives (all_reduce, all_gather) are in-place and side-effecting -- they mutate tensors and return opaque Work objects, which is fundamentally incompatible with functional graph tracing. This RFC defines the functional semantics and the AsyncTensor subclass approach that makes distributed operations traceable by Dynamo.

## Problem statement

<!-- Why existing c10d APIs break tracing -->

## Design decisions

<!-- Functional semantics, AsyncCollectiveTensor subclass approach -->

## Key APIs / interfaces

<!-- Key APIs introduced by this RFC -->

## Actionables / gotchas

<!-- Implementation implications for distributed video DiT inference -->

## Related resources

- [dynamo-deep-dive](dynamo-deep-dive.md) -- the tracing system that functional collectives are designed to work with
- [ezyang-state-of-compile](ezyang-state-of-compile.md) -- comprehensive treatment of functional collectives in context
- [nccl-user-guide](nccl-user-guide.md) -- underlying NCCL operations that funcol wraps
- [pytorch-pipelining-api](pytorch-pipelining-api.md) -- pipeline schedules that use collectives for inter-stage communication
