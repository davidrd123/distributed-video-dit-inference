# PyTorch Distributed API (torch.distributed)

| Field | Value |
|-------|-------|
| Source | https://docs.pytorch.org/docs/stable/distributed.html |
| Type | docs |
| Topics | 2, 3 |
| Status | stub |

## Why it matters

This is the primary API reference for all inter-rank communication in the pipeline-parallel inference system. Every `dist.send()`, `dist.recv()`, `dist.broadcast()`, process group creation, and timeout configuration in the PP control plane traces back to semantics defined here. Understanding the blocking/async distinction and process group lifecycle is essential for correctness.

## Key sections

- [Backends](../../sources/pytorch-distributed-api/full.md#backends) -- four built-in backends (gloo, mpi, nccl, xccl) and their per-op/per-device capability matrix; backend selection guidance.
- [Copy Engine Collectives](../../sources/pytorch-distributed-api/full.md#copy-engine-collectives) -- NCCL 2.28+ optimization that offloads data movement to DMA engines, freeing SMs for compute overlap.
- [Initialization](../../sources/pytorch-distributed-api/full.md#initialization) -- `init_process_group()` and `init_device_mesh()` requirements; timeout defaults; `device_id` for eager NCCL init.
- [Shutdown](../../sources/pytorch-distributed-api/full.md#shutdown) -- `destroy_process_group()` lifecycle; why omitting it causes hangs on exit with NCCL.
- [Groups](../../sources/pytorch-distributed-api/full.md#groups) -- `new_group()` for sub-world process groups; rank translation utilities.
- [DeviceMesh](../../sources/pytorch-distributed-api/full.md#devicemesh) -- higher-level N-d process group management abstraction.
- [Point-to-point communication](../../sources/pytorch-distributed-api/full.md#point-to-point-communication) -- `send()`, `recv()`, `isend()`, `irecv()`, `batch_isend_irecv()`, `send_object_list()`, `recv_object_list()`.
- [Synchronous and asynchronous collective operations](../../sources/pytorch-distributed-api/full.md#synchronous-and-asynchronous-collective-operations) -- `async_op` flag semantics; `Work.wait()` behavior for CPU vs CUDA; explicit stream sync example.
- [Collective functions](../../sources/pytorch-distributed-api/full.md#collective-functions) -- `broadcast()`, `all_reduce()`, `all_gather()`, `barrier()`, `monitored_barrier()`, etc.
- [Object collectives](../../sources/pytorch-distributed-api/full.md#object-collectives) -- pickle-based Python object transport; performance/memory warnings.
- [Distributed Key-Value Store](../../sources/pytorch-distributed-api/full.md#distributed-key-value-store) -- `TCPStore`, `FileStore`, `HashStore` for out-of-band coordination.
- [Debugging torch.distributed applications](../../sources/pytorch-distributed-api/full.md#debugging-torch-distributed-applications) -- `monitored_barrier`, `TORCH_DISTRIBUTED_DEBUG`, debug HTTP server, `torch.distributed.breakpoint`.
- [Logging](../../sources/pytorch-distributed-api/full.md#logging) -- log level matrix (`TORCH_CPP_LOG_LEVEL` x `TORCH_DISTRIBUTED_DEBUG`); custom exception types (`DistError`, `DistBackendError`, `DistNetworkError`, `DistStoreError`).

## Core claims

1. **Claim**: `dist.send()` is documented as sending "a tensor synchronously" and `dist.recv()` as receiving "a tensor synchronously." However, for CUDA operations this means the call blocks the CPU thread until the operation is enqueued, not until the GPU transfer completes -- the GPU work is still asynchronous per CUDA semantics.
   **Evidence**: [sources/pytorch-distributed-api/full.md#point-to-point-communication](../../sources/pytorch-distributed-api/full.md#point-to-point-communication), [sources/pytorch-distributed-api/full.md#synchronous-and-asynchronous-collective-operations](../../sources/pytorch-distributed-api/full.md#synchronous-and-asynchronous-collective-operations)

2. **Claim**: `dist.isend()` and `dist.irecv()` return a `Work` object supporting `is_completed()` and `wait()`. For CUDA collectives, `wait()` blocks the currently active CUDA stream until the operation completes but does not block the CPU. The doc warns: "Modifying `tensor` before the request completes causes undefined behavior."
   **Evidence**: [sources/pytorch-distributed-api/full.md#point-to-point-communication](../../sources/pytorch-distributed-api/full.md#point-to-point-communication)

3. **Claim**: `init_process_group()` blocks until all processes have joined. The default timeout is **10 minutes for NCCL** and **30 minutes for other backends**. After timeout, collectives are "aborted asynchronously and the process will crash" because continued execution after a failed async NCCL op risks corrupted data.
   **Evidence**: [sources/pytorch-distributed-api/full.md#initialization](../../sources/pytorch-distributed-api/full.md#initialization)

4. **Claim**: `new_group()` requires that **all processes in the main group** enter the function, even non-members. Groups must be created in the same order across all processes. When using multiple NCCL process groups, the user must ensure globally consistent execution order of collectives across ranks.
   **Evidence**: [sources/pytorch-distributed-api/full.md#groups](../../sources/pytorch-distributed-api/full.md#groups)

5. **Claim**: `destroy_process_group()` must be called by all ranks within the timeout duration. Omitting it causes hangs on exit because `ProcessGroupNCCL`'s destructor calls `ncclCommAbort`, which must be called collectively, and Python GC ordering is non-deterministic.
   **Evidence**: [sources/pytorch-distributed-api/full.md#shutdown](../../sources/pytorch-distributed-api/full.md#shutdown)

6. **Claim**: `barrier()` with NCCL is implemented as an `all_reduce` of a 1-element tensor. `ProcessGroupNCCL` now blocks the CPU thread until the barrier collective completes. Device selection for the internal tensor follows a 4-step fallback: (1) `device_ids` arg, (2) `init_process_group` device, (3) first device used with this PG, (4) global rank mod local device count.
   **Evidence**: [sources/pytorch-distributed-api/full.md#collective-functions](../../sources/pytorch-distributed-api/full.md#collective-functions)

7. **Claim**: `monitored_barrier()` is a host-side barrier using send/recv that can report which rank(s) failed to respond within the timeout. It is only supported with the **Gloo backend** and has a performance impact -- the doc recommends it only for debugging or scenarios requiring full host-side synchronization.
   **Evidence**: [sources/pytorch-distributed-api/full.md#collective-functions](../../sources/pytorch-distributed-api/full.md#collective-functions), [sources/pytorch-distributed-api/full.md#monitored-barrier](../../sources/pytorch-distributed-api/full.md#monitored-barrier)

8. **Claim**: For synchronous collective operations (default `async_op=False`), the function return guarantees the collective is "performed" but for CUDA this does NOT guarantee the CUDA operation completed -- only that it is enqueued. Using the output on a different CUDA stream requires explicit `wait_stream()` synchronization.
   **Evidence**: [sources/pytorch-distributed-api/full.md#synchronous-and-asynchronous-collective-operations](../../sources/pytorch-distributed-api/full.md#synchronous-and-asynchronous-collective-operations)

9. **Claim**: `batch_isend_irecv()` processes a list of `P2POp` objects asynchronously. When used with NCCL, users must call `torch.cuda.set_device` first. If this is the first collective on the group, **all ranks** must participate; subsequent batched P2P operations allow subset participation.
   **Evidence**: [sources/pytorch-distributed-api/full.md#initialization](../../sources/pytorch-distributed-api/full.md#initialization)

10. **Claim**: Object collectives (`send_object_list`, `broadcast_object_list`, etc.) use pickle internally, incur 2 collective operations per call (size then data), and have serious performance/scalability limitations: asymmetric pickle/unpickle time, inefficient tensor transport (CPU-sync + device-to-host copy), and unexpected tensor device placement after unpickling.
    **Evidence**: [sources/pytorch-distributed-api/full.md#object-collectives](../../sources/pytorch-distributed-api/full.md#object-collectives)

11. **Claim**: `TORCH_DISTRIBUTED_DEBUG=DETAIL` wraps all process groups with a consistency-checking layer that inserts a `monitored_barrier()` before each collective and validates tensor shapes across ranks. This turns silent hangs from shape mismatches into explicit error messages.
    **Evidence**: [sources/pytorch-distributed-api/full.md#torch-distributed-debug](../../sources/pytorch-distributed-api/full.md#torch-distributed-debug)

## API surface / configuration

**Initialization & lifecycle:**
- `torch.distributed.init_process_group(backend, init_method, timeout, world_size, rank, store, device_id)`
- `torch.distributed.destroy_process_group(group)`
- `torch.distributed.is_initialized()`
- `torch.distributed.is_torchelastic_launched()`

**Process groups:**
- `torch.distributed.new_group(ranks, timeout, backend, use_local_synchronization, device_id)`
- `torch.distributed.get_group_rank(group, global_rank)` / `get_global_rank(group, group_rank)`
- `torch.distributed.get_process_group_ranks(group)`
- `torch.distributed.device_mesh.init_device_mesh(device_type, mesh_shape, mesh_dim_names)`
- `DeviceMesh.get_group(mesh_dim)`, `DeviceMesh.get_local_rank(mesh_dim)`

**Point-to-point (PP-critical):**
- `dist.send(tensor, dst, group)` -- synchronous (CPU-blocking, GPU-async)
- `dist.recv(tensor, src, group)` -- synchronous (CPU-blocking, GPU-async)
- `dist.isend(tensor, dst, group)` -> `Work` -- asynchronous
- `dist.irecv(tensor, src, group)` -> `Work` -- asynchronous
- `dist.batch_isend_irecv(p2p_op_list)` -> `list[Work]`
- `dist.send_object_list(object_list, dst)` / `dist.recv_object_list(object_list, src)` -- pickle-based

**Collectives:**
- `dist.broadcast(tensor, src, group, async_op)`
- `dist.all_reduce(tensor, op, group, async_op)`
- `dist.all_gather(tensor_list, tensor, group, async_op)`
- `dist.barrier(group, async_op)`
- `dist.monitored_barrier(group, timeout, wait_all_ranks)` -- Gloo only

**Async work handle:**
- `Work.is_completed()`, `Work.wait()`, `Work.get_future()`

**Key-value store:**
- `TCPStore`, `FileStore`, `HashStore`, `PrefixStore`

**Debugging:**
- `TORCH_DISTRIBUTED_DEBUG` (OFF / INFO / DETAIL)
- `TORCH_CPP_LOG_LEVEL` (ERROR / WARNING / INFO)
- `torch.distributed.monitored_barrier()`
- `torch.distributed.breakpoint(rank)`
- `torch.distributed.debug.start_debug_server(port)`

## Actionables / gotchas

- **Set short timeouts during bringup**: The default NCCL timeout of 10 minutes means a hung rank blocks all other ranks for 10 minutes before crashing. During PP bringup, use `SCOPE_DIST_TIMEOUT_S=60` (or pass `timeout=timedelta(seconds=60)` to `init_process_group()`). Production can use longer values. See `refs/implementation-feedback.md` section 2e and section 3a.

- **`dist.send()` is CPU-blocking but GPU-async**: The call returns after enqueuing the NCCL kernel, not after the transfer completes. This means rank0 can proceed to other CPU work after `send()` returns, but the tensor buffer must not be reused until the GPU transfer finishes. For PP0 where we do sequential send-then-compute, this is fine. For overlap at Step A3, switch to `isend()`/`irecv()` and manage `Work.wait()` explicitly.

- **`isend()`/`irecv()` for overlap (Step A3)**: When moving from synchronous to async p2p, the tensor must not be modified until `Work.wait()` returns (undefined behavior per the doc). This means the PP envelope tensors need dedicated send/recv buffers that are not aliases of compute tensors. The `Work.wait()` call blocks the current CUDA stream, not the CPU -- so a side-stream pattern with `wait_stream()` is the right structure for overlapping transport with compute.

- **p2p + broadcast interaction pattern**: PPControlPlane uses `dist.send()`/`dist.recv()` for rank0-to-leader p2p communication, then the leader uses `dist.broadcast()` inside `mesh_pg` to fan out to the mesh. These are on different process groups (`world_pg` for p2p, `mesh_pg` for broadcast). The `new_group()` creation order constraint (all ranks must participate, same order everywhere) means both groups must be created during initialization before any p2p or broadcast calls. See `refs/implementation-feedback.md` section 3a.

- **`new_group()` is a collective**: Even ranks that will NOT be members of the new group must call `new_group()`. Failing to do so causes a hang. For our topology (rank0 outside mesh, ranks 1..N in mesh), rank0 must still participate in creating `mesh_pg` even though it gets `GroupMember.NON_GROUP_MEMBER` back.

- **`destroy_process_group()` prevents exit hangs**: Always call it at shutdown. Without it, `ncclCommAbort` ordering is non-deterministic across ranks, causing hangs. Our `PPAction.SHUTDOWN` protocol should ensure all ranks reach `destroy_process_group()` before exiting.

- **`barrier()` is an all_reduce under the hood (NCCL)**: It allocates a 1-element tensor and blocks the CPU thread. In production, avoid barriers in the hot path -- they serialize all ranks and defeat the purpose of PP overlap. Use barriers only during initialization and debugging. Prefer `monitored_barrier()` (requires a Gloo process group) during bringup because it reports WHICH rank failed, rather than just timing out.

- **`TORCH_DISTRIBUTED_DEBUG=DETAIL` for shape mismatch debugging**: This inserts `monitored_barrier()` + shape validation before every collective, turning silent hangs into informative crash messages. Use during bringup (with `TORCH_CPP_LOG_LEVEL=INFO`), disable in production due to performance overhead.

- **Object collectives (`send_object_list`) are slow but convenient for metadata**: They pickle, perform 2 collectives per call, and force CPU sync for GPU tensors. Fine for small metadata (envelope headers, configs) during initialization. Never use for tensor data in the hot path -- use `dist.send()` with pre-allocated tensors instead.

- **`batch_isend_irecv()` first-call constraint**: The first batched p2p call on a group requires all group ranks to participate. Subsequent calls allow subsets. If your PP topology starts p2p before all ranks are ready, this will hang. Initialize with a barrier or ensure the first `batch_isend_irecv` is a coordinated step.

## Related resources

- [nccl-user-guide](nccl-user-guide.md) -- NCCL-specific semantics for group calls, stream ordering, timeout behavior; complements the PyTorch API surface documented here
- [pytorch-cuda-semantics](pytorch-cuda-semantics.md) -- stream synchronization, `wait_stream()`/`record_stream()` patterns needed when using async distributed ops on non-default streams
- [funcol-rfc-93173](funcol-rfc-93173.md) -- functional collectives (`all_reduce` returning a new tensor) that integrate with `torch.compile`; alternative to the in-place collectives documented here
- [pytorch-pipelining-api](pytorch-pipelining-api.md) -- PyTorch's higher-level pipeline parallelism API built on top of these primitives
- [cuda-graphs-guide](cuda-graphs-guide.md) -- CUDA graph capture constraints relevant when combining distributed ops with graph replay
