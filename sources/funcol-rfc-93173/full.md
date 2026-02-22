---
title: "[RFC] PT2-Friendly Traceable, Functional Collective Communication APIs"
source_url: https://github.com/pytorch/pytorch/issues/93173
fetch_date: 2026-02-22
source_type: rfc
author: wconstab (Will Constable)
---

# [RFC] PT2-Friendly Traceable, Functional Collective Communication APIs

**Issue**: pytorch/pytorch#93173
**Author**: [wconstab](https://github.com/wconstab) (Will Constable)
**Created**: 2023-01-27
**Closed**: 2025-07-12 (by [ezyang](https://github.com/ezyang))
**State**: closed (completed)
**Labels**: oncall: distributed, feature, triaged, oncall: pt2, module: ProxyTensor, module: pt2-dispatcher, no-scrub
**Reactions**: 19 heart, 18 rocket

---

## Traceable Collectives!

Collective APIs (e.g. all_reduce, all_gather, ...) are used in distributed PyTorch programs, but do not compose cleanly with compilers.

Specifically, torchDynamo and the AotAutograd pipeline for decompositions and functionalization do not work with the existing c10d collective APIs
* there are not functional variants of these collectives
* ProcessGroup and Work objects interfere with graph tracing and pollute the IR with non-tensor objects

XLA also currently has to implement some workarounds, to marry the XLA collective ops via lazy tensor tracing with the existing PyTorch / C10D side. They have to use a custom ProcessGroup implementation and swizzle PTD PG creation functions.

**Goals**
1) provide collectives that are **traceable** with the PT2 stack and XLA stack
2) provide **functional** collectives, which are easier for IR transformations to reason about
3) support **eager and compiled** flows with the same API
4) use **plain data types** in the traced API
5) allow tracing/compilation **without requiring process group init**
6) **support different frontends** (DTensors, ProcessGroups, etc)
7) support **autograd** for collective ops
8) clean up c10d python bindings and dispatcher registrations

**Non-goals**

1) Introduce multiple stream semantics in inductor

[Figure: Architecture diagram showing the relationship between Python API layer (DTensor, ProcessGroup), traceable collectives layer, and dispatcher ops layer. Shows how functional collectives sit between the user-facing APIs and the traced graph IR.]

## New traceable collectives python API

```
def collective(input:Tensor, *, group: GROUP_TYPE) -> AsyncTensor
```

`GROUP_TYPE` is a Union over List<rank>, DeviceMesh, ProcessGroup, etc. It allows flexible usage by different frontends.

`AsyncTensor` is a Tensor subclass that calls `wait()` automatically when the tensor is used by another op.

## New Dispatcher Collectives

```
aten::collective(Tensor, *, str tag, int[] ranks, int stride) -> Tensor`
```

These are the ops that actually get traced into a graph and can be manipulated by compiler passes.

The collective ops are functional, but compilers may be able to convert them to inplace. They are asynchronous.

These ops support meta device (for traceability), and support backwards via derivatives.yaml.

The semantics of these ops are that they return a real tensor, but you aren't allowed to access its data or storage.

```
c10d.wait(Tensor) -> Tensor
```

`wait()` must be called on the output of any collective before its underlying data or storage is accessed.
* It is valid to peek at the size() or stride() (or probably other metadata) of a tensor returned from a collective, but not its data.
* wait() is the only way to make an output from collectives safe to use by other non collective ops
* we are considering whether wait(collective(collective)) can be implemented safely, but by default we assume it is not

The semantics of wait are that you must only access the storage of the tensor returned from wait. You can't think of wait as mutating its input tensor and making it safe to use.

## Alternatives

The following style of API has also been considered. Its main disadvantage is in requiring a user to first initialize a processgroup, but it is also opaque and not easily interchangeable with lists of ranks or DTensors. It doesn't allow us to easily represent MPMD collectives.

```
pg = init_process_group()
pg_id = dist.register_process_group(pg)
collective(tensor, pg_id)
```

## Detailed Proposal

See [Traceable Collectives Design](https://docs.google.com/document/d/1Jqa68gvuVeFWZJFOiukmb58jAaUEET1GVMkd1GOMRT4/edit)

cc @H-Huang @awgu @wanchaol @fegin @fduwjj @wz337 @d4l3k @chauhang @penguinwu @zou3519 @bdhirsh @mrshenli @pritamdamania87 @zhaojuanmao @satgera @rohan-varma @gqchen @aazzolini @osalpekar @jiayisuse @kwen2501 @XilunWu @ezyang @msaroufim @anijain2305 @soumith @ngimel
