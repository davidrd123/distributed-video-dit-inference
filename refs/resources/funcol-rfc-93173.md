# RFC #93173: PT2-Friendly Traceable, Functional Collective Communication APIs

| Field | Value |
|-------|-------|
| Source | https://github.com/pytorch/pytorch/issues/93173 |
| Type | code |
| Topics | 11 |
| Status | condensed |

## Why it matters

The foundational design document for functional collectives. Standard NCCL collectives (all_reduce, all_gather) are in-place and side-effecting -- they mutate tensors and return opaque Work objects, which is fundamentally incompatible with functional graph tracing. This RFC defines the functional semantics and the AsyncTensor subclass approach that makes distributed operations traceable by Dynamo.

## Core claims

1. **Claim**: Existing `c10d` collective APIs don’t compose cleanly with the PT2 compiler stack because there are no functional variants, and `ProcessGroup`/`Work` objects pollute the traced IR with non-tensor objects.
   **Evidence**: [sources/funcol-rfc-93173/full.md#traceable-collectives](../../sources/funcol-rfc-93173/full.md#traceable-collectives)

2. **Claim**: The RFC’s goals include: traceable + functional collectives, one API usable in eager and compiled flows, plain data types in the traced API, compile without requiring process group init, support multiple frontends (DTensor/ProcessGroup/etc.), and support autograd for collectives.
   **Evidence**: [sources/funcol-rfc-93173/full.md#traceable-collectives](../../sources/funcol-rfc-93173/full.md#traceable-collectives)

3. **Claim**: Introducing multiple-stream semantics in Inductor is explicitly a non-goal.
   **Evidence**: [sources/funcol-rfc-93173/full.md#traceable-collectives](../../sources/funcol-rfc-93173/full.md#traceable-collectives)

4. **Claim**: The proposed traceable Python API returns an `AsyncTensor` and takes a flexible `GROUP_TYPE` that can represent a list of ranks, `DeviceMesh`, `ProcessGroup`, etc.
   **Evidence**: [sources/funcol-rfc-93173/full.md#new-traceable-collectives-python-api](../../sources/funcol-rfc-93173/full.md#new-traceable-collectives-python-api)

5. **Claim**: The dispatcher op `aten::collective(...) -> Tensor` is the collectives surface intended to be traced into graphs and manipulated by compiler passes; it is functional and asynchronous, supports meta device for traceability, and supports backwards via `derivatives.yaml`.
   **Evidence**: [sources/funcol-rfc-93173/full.md#new-dispatcher-collectives](../../sources/funcol-rfc-93173/full.md#new-dispatcher-collectives)

6. **Claim**: Dispatcher collectives return a “real tensor” value but accessing its data/storage is forbidden until a separate wait op is applied; inspecting metadata like `size()`/`stride()` is allowed pre-wait.
   **Evidence**: [sources/funcol-rfc-93173/full.md#new-dispatcher-collectives](../../sources/funcol-rfc-93173/full.md#new-dispatcher-collectives)

7. **Claim**: Waiting is expressed as `c10d.wait(Tensor) -> Tensor`, and the semantics are that you may only access the storage of the tensor returned from `wait` (do not treat `wait` as mutating the input tensor to become safe).
   **Evidence**: [sources/funcol-rfc-93173/full.md#new-dispatcher-collectives](../../sources/funcol-rfc-93173/full.md#new-dispatcher-collectives)

8. **Claim**: An alternative API based on registering a `ProcessGroup` and passing an opaque ID was considered, but it requires PG initialization and is less interchangeable; it also does not easily represent MPMD collectives.
   **Evidence**: [sources/funcol-rfc-93173/full.md#alternatives](../../sources/funcol-rfc-93173/full.md#alternatives)

## Problem statement

PyTorch’s existing distributed collectives are not designed to be traced as pure “Tensor → Tensor” ops:

- There are no functional variants of the c10d collectives. ([sources/funcol-rfc-93173/full.md#traceable-collectives](../../sources/funcol-rfc-93173/full.md#traceable-collectives))
- `ProcessGroup` and `Work` objects interfere with tracing by injecting non-Tensor objects into the traced IR. ([sources/funcol-rfc-93173/full.md#traceable-collectives](../../sources/funcol-rfc-93173/full.md#traceable-collectives))
- XLA has historically needed workarounds (e.g., custom `ProcessGroup` implementations) to bridge “lazy traced” collectives with the existing `c10d` interface. ([sources/funcol-rfc-93173/full.md#traceable-collectives](../../sources/funcol-rfc-93173/full.md#traceable-collectives))

As a result, Dynamo/AOTAutograd functionalization and decomposition passes can’t reason about collectives cleanly.

## Design decisions

### Goals and non-goals

The RFC’s design goals are explicitly compiler-facing: functional semantics, traceability, and IR cleanliness (plain traced types), while retaining an API usable in eager and compiled modes and across multiple “frontends” (DTensor, ProcessGroup, etc.). ([sources/funcol-rfc-93173/full.md#traceable-collectives](../../sources/funcol-rfc-93173/full.md#traceable-collectives))

The key non-goal is important for PP overlap work: it does not attempt to introduce multi-stream semantics in Inductor. ([sources/funcol-rfc-93173/full.md#traceable-collectives](../../sources/funcol-rfc-93173/full.md#traceable-collectives))

### Two-layer API (Python wrapper + dispatcher op)

1. A traceable Python API using `GROUP_TYPE` and returning `AsyncTensor`. ([sources/funcol-rfc-93173/full.md#new-traceable-collectives-python-api](../../sources/funcol-rfc-93173/full.md#new-traceable-collectives-python-api))
2. A dispatcher-level op (`aten::collective`) that is what gets traced and is amenable to compiler passes. ([sources/funcol-rfc-93173/full.md#new-dispatcher-collectives](../../sources/funcol-rfc-93173/full.md#new-dispatcher-collectives))

### Explicit wait semantics

The RFC draws a hard line between “tensor exists as a value in the IR” and “tensor’s storage is safe to access”:

- Collective outputs are real tensors, but you must not access data/storage until a wait op is applied. ([sources/funcol-rfc-93173/full.md#new-dispatcher-collectives](../../sources/funcol-rfc-93173/full.md#new-dispatcher-collectives))
- `wait` returns a tensor whose storage is safe; do not conceptualize it as mutating its input tensor to make it safe. ([sources/funcol-rfc-93173/full.md#new-dispatcher-collectives](../../sources/funcol-rfc-93173/full.md#new-dispatcher-collectives))

## Key APIs / interfaces

### Traceable Python API (proposed)

```python
def collective(input: Tensor, *, group: GROUP_TYPE) -> AsyncTensor
```

- `GROUP_TYPE` is a union over list-of-ranks, `DeviceMesh`, `ProcessGroup`, etc. ([sources/funcol-rfc-93173/full.md#new-traceable-collectives-python-api](../../sources/funcol-rfc-93173/full.md#new-traceable-collectives-python-api))
- `AsyncTensor` is a Tensor subclass that calls `wait()` automatically when used by another op. ([sources/funcol-rfc-93173/full.md#new-traceable-collectives-python-api](../../sources/funcol-rfc-93173/full.md#new-traceable-collectives-python-api))

### Dispatcher ops (what gets traced)

```text
aten::collective(Tensor, *, str tag, int[] ranks, int stride) -> Tensor
c10d.wait(Tensor) -> Tensor
```

- `aten::collective`: functional + async; intended to be traced into graphs and rewritten by compiler passes. ([sources/funcol-rfc-93173/full.md#new-dispatcher-collectives](../../sources/funcol-rfc-93173/full.md#new-dispatcher-collectives))
- `c10d.wait`: the explicit boundary before storage access is allowed. ([sources/funcol-rfc-93173/full.md#new-dispatcher-collectives](../../sources/funcol-rfc-93173/full.md#new-dispatcher-collectives))

## Actionables / gotchas

- **This RFC explains why Run 12b worked**: in TP=2 + compile, wrapping collectives with `torch._dynamo.disable()` introduced **~160 graph breaks per forward** and throughput collapsed to **9.6 FPS** (Runs 8–9b). Switching to functional collectives (`all_reduce` returning tensors + explicit waits) eliminated collective-induced graph breaks (micro-repro: **0 graph breaks**) and restored steady-state throughput to **~24.5 FPS** (Run 12b). See: `scope-drd/notes/FA4/h200/tp/bringup-run-log.md` (Runs 8–12b).
- **Gate funcol by compile mode (eager regression trap)**: an earlier version that used functional collectives in eager mode regressed throughput (**18.0 FPS** vs **19.5 FPS** baseline) due to extra allocation/wrapping overhead (Run 12a). Keep eager on in-place `dist.all_reduce` and use funcol only under `torch.compile`. See: `scope-drd/notes/FA4/h200/tp/bringup-run-log.md` (Run 12a).
- **Funcol is not in-place**: treat it as “Tensor → Tensor”; always consume the returned tensor, and keep the “wait-before-storage-access” rule in mind when calling ops that will touch data.
- **Don’t expect this RFC to solve overlap by itself**: multi-stream semantics in Inductor are a non-goal here; PP overlap still needs explicit stream/event discipline (tie-in: `refs/resources/pytorch-cuda-semantics.md`, `refs/resources/nccl-user-guide.md`).
- **Keep non-Tensor objects out of compiled distributed regions**: avoid threading `ProcessGroup`/`Work` objects through traced code; stick to “plain types + tensors” surfaces to keep tracing predictable across ranks.
- **MPMD/multi-group designs should prefer explicit group representations**: the “opaque PG ID” alternative is rejected partly due to weak MPMD representation; for PP control-plane vs mesh groups, make group boundaries explicit and deterministic.

## Related resources

- [dynamo-deep-dive](dynamo-deep-dive.md) -- the tracing system that functional collectives are designed to work with
- [ezyang-state-of-compile](ezyang-state-of-compile.md) -- comprehensive treatment of functional collectives in context
- [nccl-user-guide](nccl-user-guide.md) -- underlying NCCL operations that funcol wraps
- [pytorch-cuda-semantics](pytorch-cuda-semantics.md) -- stream ordering + allocator semantics; funcol doesn't introduce multi-stream semantics
- [pytorch-pipelining-api](pytorch-pipelining-api.md) -- pipeline schedules that use collectives for inter-stage communication
- [making-dl-go-brrrr](making-dl-go-brrrr.md) -- overhead-bound regime that funcol addresses by eliminating graph breaks
