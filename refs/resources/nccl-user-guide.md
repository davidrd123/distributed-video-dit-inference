# NCCL User Guide

| Field | Value |
|-------|-------|
| Source | https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html, https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html |
| Type | docs |
| Topics | 1 |
| Status | condensed |

## Why it matters

Covers communicators, collective operations, CUDA stream semantics, group calls, and algorithm selection. The environment variables page documents `NCCL_DEBUG`, `NCCL_ALGO`, and `NCCL_PROTO` for controlling and debugging algorithm choice -- essential for diagnosing why pipeline send/recv calls behave differently across NVLink vs PCIe topologies.

## Key sections

- [CUDA Stream Semantics](../../sources/nccl-user-guide/full.md#cuda-stream-semantics) — what “async NCCL” actually means (enqueue vs complete) and the gotchas around groups + multi-stream usage.
- [Group Calls](../../sources/nccl-user-guide/full.md#group-calls) — `ncclGroupStart/End` for multi-GPU-per-thread, aggregation, and ordering semantics.
- [Thread Safety](../../sources/nccl-user-guide/full.md#thread-safety) — what is and isn’t safe with multiple threads/communicators.
- [Using NCCL with CUDA Graphs](../../sources/nccl-user-guide/full.md#using-nccl-with-cuda-graphs) — capture/launch rules and caveats; see also env vars `NCCL_GRAPH_REGISTER` and `NCCL_GRAPH_MIXING_SUPPORT`.
- [Error handling and communicator abort](../../sources/nccl-user-guide/full.md#error-handling-and-communicator-abort) — async error polling + abort patterns to avoid “hang forever”.
- [Environment Variables](../../sources/nccl-user-guide/full.md#environment-variables) — debug (`NCCL_DEBUG*`), algorithm/protocol selection (`NCCL_ALGO`, `NCCL_PROTO`), multi-communicator ordering (`NCCL_LAUNCH_ORDER_IMPLICIT`).

## Core claims

1. **Claim**: NCCL collectives are launched onto a specific CUDA stream; NCCL returns once the work is enqueued, and the collective executes asynchronously on the device (completion is managed with normal CUDA mechanisms like stream sync or events).
   **Evidence**: [sources/nccl-user-guide/full.md#cuda-stream-semantics](../../sources/nccl-user-guide/full.md#cuda-stream-semantics)

2. **Claim**: Using multiple streams within the same `ncclGroupStart/End` enforces dependencies across all involved streams and blocks them until the NCCL kernel completes, effectively creating a global synchronization point between those streams.
   **Evidence**: [sources/nccl-user-guide/full.md#mixing-multiple-streams-within-the-same-ncclgroupstartend-group](../../sources/nccl-user-guide/full.md#mixing-multiple-streams-within-the-same-ncclgroupstartend-group)

3. **Claim**: Group calls (`ncclGroupStart/End`) exist for multi-GPU-per-thread management, operation aggregation (reduce launch overhead), and grouping P2P patterns; when called inside a group, stream ops may return before enqueue and CUDA stream sync calls must be performed only after `ncclGroupEnd` returns.
   **Evidence**: [sources/nccl-user-guide/full.md#group-calls](../../sources/nccl-user-guide/full.md#group-calls)

4. **Claim**: NCCL group operations require a consistent issuing order across ranks/GPUs/communicators; changing the order can lead to incorrect results or hangs.
   **Evidence**: [sources/nccl-user-guide/full.md#group-operation-ordering-semantics](../../sources/nccl-user-guide/full.md#group-operation-ordering-semantics)

5. **Claim**: NCCL primitives are generally not thread-safe; it is not allowed to issue operations to a single communicator in parallel from multiple threads, and grouped operations must be issued by a single thread (serialization is required).
   **Evidence**: [sources/nccl-user-guide/full.md#thread-safety](../../sources/nccl-user-guide/full.md#thread-safety)

6. **Claim**: NCCL operations can be captured by CUDA Graphs (since NCCL 2.9) with CUDA ≥ 11.3; graph capture/launch involving NCCL is collective and must be uniform across all ranks participating in the captured operations.
   **Evidence**: [sources/nccl-user-guide/full.md#using-nccl-with-cuda-graphs](../../sources/nccl-user-guide/full.md#using-nccl-with-cuda-graphs)

7. **Claim**: NCCL provides extensive environment-variable control for debugging and tuning: `NCCL_DEBUG`/`NCCL_DEBUG_FILE`/`NCCL_DEBUG_SUBSYS` for logging, `NCCL_ALGO` for constraining algorithms (with per-collective selectors), and `NCCL_PROTO` for constraining protocols (discouraged except to disable a suspected-buggy protocol; enabling LL128 on unsupported platforms can corrupt data).
   **Evidence**: [sources/nccl-user-guide/full.md#nccl_debug](../../sources/nccl-user-guide/full.md#nccl_debug), [sources/nccl-user-guide/full.md#nccl_debug_subsys](../../sources/nccl-user-guide/full.md#nccl_debug_subsys), [sources/nccl-user-guide/full.md#nccl_algo](../../sources/nccl-user-guide/full.md#nccl_algo), [sources/nccl-user-guide/full.md#nccl_proto](../../sources/nccl-user-guide/full.md#nccl_proto)

8. **Claim**: When using multiple communicators per device, NCCL can implicitly order operations from different communicators using host program order (`NCCL_LAUNCH_ORDER_IMPLICIT`, since 2.26) to prevent deadlocks; this relies on deterministic host-side launch order.
   **Evidence**: [sources/nccl-user-guide/full.md#using-multiple-nccl-communicators-concurrently](../../sources/nccl-user-guide/full.md#using-multiple-nccl-communicators-concurrently), [sources/nccl-user-guide/full.md#nccl_launch_order_implicit](../../sources/nccl-user-guide/full.md#nccl_launch_order_implicit)

## API surface / configuration

**Common API concepts (C-level NCCL):**
- Communicator lifecycle: `ncclCommInitRank{,Config}`, `ncclCommFinalize`, `ncclCommDestroy`, `ncclCommAbort`, `ncclCommGetAsyncError`
- Collectives / P2P: `ncclAllReduce` (+ friends), `ncclSend` / `ncclRecv`
- Grouping: `ncclGroupStart` / `ncclGroupEnd`

**High-leverage env vars (debug + correctness/perf triage):**
- Logging: `NCCL_DEBUG`, `NCCL_DEBUG_SUBSYS`, `NCCL_DEBUG_FILE`
- Algo/proto selection: `NCCL_ALGO`, `NCCL_PROTO`
- Multi-communicator ordering: `NCCL_LAUNCH_ORDER_IMPLICIT`, `NCCL_LAUNCH_RACE_FATAL`
- CUDA graphs mixing: `NCCL_GRAPH_MIXING_SUPPORT`, `NCCL_GRAPH_REGISTER`
- Topology/transport knobs (situational): `NCCL_P2P_DISABLE`, `NCCL_P2P_LEVEL`, `NCCL_SHM_DISABLE`, `NCCL_IB_DISABLE`, `NCCL_SOCKET_IFNAME`

## Actionables / gotchas

- **Use measured comm costs to sanity-check TP/PP designs**: on 2×H200 NVLink, we measured BF16 all-reduce on `[1,2160,5120]` (~21.1 MiB) at **~0.113 ms p50**; this suggests “big hidden-state” collectives can be sub-millisecond and TP viability is dominated by (a) how many collectives you pay per forward and (b) launch/graph-break overhead. See: `scope-drd/notes/FA4/h200/tp/feasibility.md` Section 1.
- **Debug “why did this hang?” with ordering + thread-safety first**: mismatched collective order across ranks (including across multiple communicators) can hang; NCCL is not thread-safe and grouped operations must be issued by one thread. See: [Group Operation Ordering Semantics](../../sources/nccl-user-guide/full.md#group-operation-ordering-semantics), [Thread Safety](../../sources/nccl-user-guide/full.md#thread-safety), and the multi-communicator note + `NCCL_LAUNCH_ORDER_IMPLICIT`.
- **Avoid accidental global sync points**: mixing multiple CUDA streams inside a single NCCL group forces dependencies and blocks all streams until completion; treat it as a serialization point unless you explicitly want that behavior. See: [Mixing Multiple Streams within the same ncclGroupStart/End() group](../../sources/nccl-user-guide/full.md#mixing-multiple-streams-within-the-same-ncclgroupstartend-group).
- **Prefer instrumentation over forcing knobs**: use `NCCL_DEBUG=INFO` with `NCCL_DEBUG_SUBSYS=...` (notably `TUNING`, `COLL`, `GRAPH`) to observe what NCCL is doing before setting `NCCL_ALGO`/`NCCL_PROTO`. If you *do* touch `NCCL_PROTO`, follow the guide’s warning: use it to **disable** a suspected buggy protocol, and don’t enable LL128 on unsupported platforms (risk: corruption). See: [NCCL_PROTO](../../sources/nccl-user-guide/full.md#nccl_proto).
- **CUDA Graph capture is a collective contract**: capture/launch of NCCL ops must be uniform across ranks participating in the communicator; mixing captured and non-captured ops has explicit support controlled by `NCCL_GRAPH_MIXING_SUPPORT` (disabling support is motivated by observed hangs when multiple ranks launch graphs from the same thread). See: [Using NCCL with CUDA Graphs](../../sources/nccl-user-guide/full.md#using-nccl-with-cuda-graphs), [NCCL_GRAPH_MIXING_SUPPORT](../../sources/nccl-user-guide/full.md#nccl_graph_mixing_support).
- **Make debug logs actually usable**: `NCCL_DEBUG_FILE` supports a `filename.%h.%p` format (hostname + PID), overwrites existing files, and does not accept `~` in paths. See: [NCCL_DEBUG_FILE](../../sources/nccl-user-guide/full.md#nccl_debug_file).

## Related resources

- [pytorch-cuda-semantics](pytorch-cuda-semantics.md) -- CUDA stream semantics interact with NCCL stream behavior
- [cuda-graphs-guide](cuda-graphs-guide.md) -- NCCL operations inside CUDA graphs have specific requirements
- [funcol-rfc-93173](funcol-rfc-93173.md) -- functional collectives wrap NCCL operations for traceability
- [making-dl-go-brrrr](making-dl-go-brrrr.md) -- overhead model; helps reason about when launch overhead dominates vs comm/compute
