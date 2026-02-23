Below is a straight “PP1 topology + recompute collectives” audit keyed to your Q1–Q6. I’m going to be annoyingly explicit about what must be true to avoid hangs.

---

## Q1) VAE thread safety on rank0 under A4b

### Verdict

* **VAE decode concurrent with NCCL p2p on the same GPU:** **YES** (safe), **if** you use stream handoff correctly and don’t call `torch.distributed` from the VAE thread.
* **WanVAE streaming encode/decode concurrent from multiple threads:** **NO** (not thread-safe), because the model mutates shared cache state.

### What’s actually safe here

In your PP0 overlap model (`scope-drd/scripts/pp0_pilot.py`, comms thread uses `comm_stream`), you’re doing the right *shape* of thing:

* NCCL send/recv happens inside a **non-default CUDA stream** via `with torch.cuda.stream(comm_stream): ... pp.send_infer(); pp.recv_result(); ...`.
* The main thread can run VAE decode on the default stream concurrently.

This matches the “NCCL ops are enqueued on a CUDA stream; async w.r.t host” mental model in your NCCL notes and the PyTorch stream semantics notes. (Your repo references: `refs/resources/nccl-user-guide.md` + `refs/resources/pytorch-cuda-semantics.md`.)

### The real landmine: WanVAE is stateful

`WanVAEWrapper.decode_to_pixel(..., use_cache=True)` calls `self.model.stream_decode(...)` (see `scope-drd/src/scope/core/pipelines/wan2_1/vae/wan.py`). That bottoms out in `WanVAE_.stream_decode` which mutates:

* `self.first_batch`
* `_feat_map`, `_conv_idx`
* plus “clear_cache_decode” on first batch
  (see `scope-drd/src/scope/core/pipelines/wan2_1/vae/modules/vae.py`, `stream_decode`, `clear_cache_decode`, `first_batch` changes).

That means:

* **One thread calling `decode_to_pixel(use_cache=True)` is fine.**
* **Two threads calling encode/decode streaming paths is not fine.**
* Even “encode in one thread while decode in another” is not safe if either uses the streaming cache (`use_cache=True`).

### Stream handoff correctness (you need both ordering and lifetime)

You already use CUDA events for ordering in `pp0_pilot.py`:

* comms thread records `ready_evt` on `comm_stream` after `pp.recv_result()`
* main thread should wait on that before touching `latents_out`

**Do this on the consumer (main thread) before decode:**

```py
if recv_item.ready_evt is not None:
    torch.cuda.current_stream().wait_event(recv_item.ready_evt)

# Lifetime safety: tell the allocator this tensor will be used on the current stream.
result.latents_out.record_stream(torch.cuda.current_stream())
```

Why the `record_stream` matters: PyTorch’s caching allocator can reuse memory “too early” when tensors cross streams unless you explicitly record the usage stream. Waiting on an event gives *ordering*, `record_stream` gives *lifetime safety*. (This is exactly the warning class in `refs/resources/pytorch-cuda-semantics.md` about non-default streams + allocator reuse.)

### Allocator contention

* **Correctness risk:** low.
* **Latency risk:** non-zero. Both threads can allocate (recv buffers in `pp_control._recv_meta_and_tensors` allocate fresh tensors; VAE allocates temporaries). PyTorch allocator uses locks; you can see CPU-side contention in profiles.

If this shows up:

* preallocate + reuse recv buffers (cache by tensor spec) inside `PPControlPlane.recv_result()` / `_recv_meta_and_tensors` (`scope-drd/src/scope/core/distributed/pp_control.py`)
* keep VAE decode allocations stable (channels-last settings already exist in `wan.py`)

### `torch.no_grad()` interaction

**No effect across threads.** Grad mode is thread-local in PyTorch. So main thread in `no_grad()` doesn’t “infect” comms thread.

---

## Q2) Process group creation recipe for PP1 `mesh_pg`

### Verdicts (direct answers)

* **Does rank0 need to call `dist.new_group(ranks=[1..N])`?** **YES.** Treat `new_group` as **collective across the default group**; if rank0 skips it, some other rank will block/hang creating the communicator.
* **Can ranks call `new_group` at different times?** **Technically they can**, but the early callers will block until the late callers arrive. In practice: **call it during init, once, in a fixed order**.
* **Can rank0 do p2p on world while mesh uses mesh_pg?** **YES.** That’s the whole point: separate communicators.
* **Should PP p2p use a dedicated `pp_pg=[0, leader]`?** **Recommended** for PP1 bringup (not mandatory). It makes “who participates” explicit and reduces accidental “world group” misuse.
* **Destroy semantics:** don’t rely on teardown for correctness. If you do destroy, do it on **all ranks** and in **reverse creation order** to avoid weird partial-teardown hangs.
* **Reuse:** create once at startup and reuse for the process lifetime.

### Concrete recipe

Put this right after `dist.init_process_group(...)` in `scope-drd/src/scope/core/distributed/runtime.py:init_distributed_if_needed()` (or a helper it calls). Every rank executes it:

```py
# world ranks: 0..N
mesh_ranks = [1, 2, ..., N]
leader = 1

# Optional but recommended: isolate PP p2p
pp_pg = dist.new_group(ranks=[0, leader])

# TP mesh group
mesh_pg = dist.new_group(ranks=mesh_ranks)
```

**Ordering constraint:** if you create multiple groups, every rank must call `new_group` in the **same sequence** (pp_pg first, then mesh_pg, or vice versa). Don’t “if rank in group: create it” – that’s a deadlock recipe.

### Root rank gotcha (group-rank vs global-rank)

For `dist.broadcast(..., src=...)`, `src` is **rank within that group**.

If `mesh_pg` ranks are `[1..N]`, then:

* **global rank 1 == mesh group-rank 0**
* so mesh leader uses `src=0` when broadcasting on `mesh_pg`

This matters for every “leader → workers” broadcast protocol you’re about to write.

---

## Q3) Concurrent NCCL communicators: PP p2p + TP collectives

### Two separate questions are getting mixed

1. **Can NCCL execute ops from different communicators concurrently on one GPU?**
   → **Yes in principle.** You can overlap kernels if they’re enqueued on different streams and there’s bandwidth/SM headroom.

2. **Is it safe to do that from multiple Python threads calling `torch.distributed`?**
   → **Usually no for bringup.** NCCL’s own guidance is “don’t issue ops to the same communicator from multiple threads” (your `refs/resources/nccl-user-guide.md` summary includes this). PyTorch’s ProcessGroupNCCL adds its own complexity. The safe posture is: **single thread issues all `torch.distributed` calls per process**.

### The PP1 architecture implication

* On **rank0**: you can have a comms thread calling `dist.send/recv` while main thread runs VAE decode. That’s fine because main thread doesn’t call `dist`.
* On **mesh ranks (including leader)**: the model forward includes TP all-reduces (see `scope-drd/src/scope/core/tensor_parallel/{linear.py,rmsnorm.py}`), which are `dist.all_reduce(...)` today. If you also run PP recv/broadcast in another thread, you have multiple threads calling `dist` in the same process.

That’s where you get deadlocks that feel “non-deterministic”.

### Recommended concurrency matrix (PP1 bringup posture)

Interpret “concurrent” as “same process, same GPU”.

| PP op (leader)                    | TP op (leader)               | Safe to overlap? | Recommendation                                                                   |
| --------------------------------- | ---------------------------- | ---------------: | -------------------------------------------------------------------------------- |
| `dist.recv` on world/pp_pg        | `dist.all_reduce` on mesh_pg |    maybe$_{60%}$ | **Don’t overlap**. Serialize: recv envelope → broadcast → forward → send result. |
| `dist.send` result on world/pp_pg | `dist.all_reduce` on mesh_pg |    maybe$_{60%}$ | Same: serialize.                                                                 |
| rank0 `dist.send/recv`            | mesh `dist.all_reduce`       |      yes$_{95%}$ | Different processes, no issue. Overlap is “free”.                                |

So the answer to “can rank1 receive next envelope while also in TP all-reduce” is: **you can try, but you’re buying yourself a debugging pit. Don’t do it for A5 bringup.**

### What Megatron/DeepSpeed effectively do (pattern-level)

They:

* create separate process groups for TP/PP/DP
* overlap compute and comm mostly via **CUDA streams**, not multiple Python threads issuing collectives willy-nilly
* keep collective scheduling deterministic

So: **copy the “streams not threads” idea**.

### Crash > hang settings

Your repo already pushes the right direction:

* shorter `SCOPE_DIST_TIMEOUT_S` (runtime uses it in `dist.init_process_group(timeout=...)` in `runtime.py`)
* watchdogs that `os._exit(...)` on stale headers (see `scope-drd/src/scope/server/tp_worker.py`)

For PP1, mirror that:

* leader + workers need a “no control message seen for X seconds” watchdog
* keep `NCCL_DEBUG=INFO` handy and use `NCCL_DEBUG_FILE=...%h.%p` if logs get noisy (your NCCL notes mention this)

If you decide to attempt multi-communicator overlap later, read your own note about `NCCL_LAUNCH_ORDER_IMPLICIT` (NCCL 2.26+) first. But I would not make PP1 correctness depend on it.

---

## Q4) Recompute-collective coupling across PP boundary

### Verdict

* **As written, recompute gating is not safely “plan-driven”.**
  Right now `RecomputeKVCacheBlock` makes an independent decision based on `SCOPE_KV_CACHE_RECOMPUTE_EVERY` and state (`scope-drd/src/scope/core/pipelines/krea_realtime_video/blocks/recompute_kv_cache.py`). That is exactly the “conditional collectives” deadlock class in your deadlock taxonomy (`refs/topics/02-deadlock-patterns.md`).

### The concrete deadlock you should care about

If recompute causes an extra generator forward, then it causes **extra TP all-reduces**.

If *any* mesh rank takes a different branch around recompute, you hang inside the next all-reduce. That’s the scary case.

### Are env vars identical across ranks under torchrun?

* **Not a guarantee of the universe**, but in practice torchrun spawns ranks with the same env.
* More importantly: you already defend this with `_assert_tp_env_parity` in `scope-drd/src/scope/core/distributed/runtime.py`, and it includes **`SCOPE_KV_CACHE_RECOMPUTE_EVERY`** in `_TP_ENV_PARITY_KEYS`. Good.

So: “different env var per rank” should crash at startup, not hang.

### But you still have a worse problem: the block ignores the envelope plan

You *already* added plan fields to the PP boundary:

* `PPEnvelopeV1.do_kv_recompute`
* `PPEnvelopeV1.expected_generator_calls`
* `PPEnvelopeV1.context_frames` (required if recompute)
  (`scope-drd/src/scope/core/distributed/pp_contract.py`)

Yet the recompute block currently:

* does **not** check `do_kv_recompute`
* looks for `context_frames_override`, **not** `context_frames`
  (`RecomputeKVCacheBlock.__call__` reads `context_frames_override`)

So PP1 will either:

* silently do recompute when the envelope said “skip” (call-count mismatch, or hang if ranks diverge), or
* fall back to the VAE re-encode path when override is missing (which is illegal if mesh ranks don’t have VAE), or
* both.

### Recommended defense (do this before A5)

**Make recompute decision a single source of truth: the envelope bit.**

In `scope-drd/src/scope/core/pipelines/krea_realtime_video/blocks/recompute_kv_cache.py`:

1. Gate recompute strictly:

```py
do_kv_recompute = bool(getattr(block_state, "do_kv_recompute", True))
if not do_kv_recompute:
    self.set_block_state(state, block_state)
    return components, state
```

2. Accept the PP contract field as an override (fix the naming mismatch):

```py
override = getattr(block_state, "context_frames_override", None)
if not isinstance(override, torch.Tensor):
    override = getattr(block_state, "context_frames", None)
```

3. Treat `SCOPE_KV_CACHE_RECOMPUTE_EVERY` as a **rank0-side planner knob only**, not a mesh-side branch. If you keep it in the block, it’s a second “planner” that can drift.

This aligns with your own operator test matrix OM-10 (override required) and OM-05 (conditional collectives). (`distributed-video-dit-inference/refs/operator-test-matrix.md`.)

### Likelihood of “some mesh ranks recompute, others don’t”

If the only inputs are (env var, current_start_frame, config), and those are consistent, then divergence is unlikely$_{10%}$. But you don’t need it to be likely; you need it to be **impossible by construction**.

---

## Q5) Shutdown and error handling during active mesh collectives

### Verdict

* **Leader cannot “interrupt” a TP collective.** If workers are mid-forward, the leader is too. Your shutdown protocol must be **between-chunks**, not “anytime”.
* Therefore the Stage0 rule must be: **stop enqueuing, drain in-flight, then send SHUTDOWN.**

This is consistent with your Topic 3 posture: crash-only during bringup, explicit SHUTDOWN action, watchdogs/timeouts to avoid 300s waits. (`refs/topics/03-graceful-shutdown.md`.)

### PP1 shutdown protocol sketch (practical)

#### Rank0 (Stage0)

* Stop producing new envelopes.
* Wait for the last result you already sent (drain).
* Send `PPAction.SHUTDOWN` (`PPControlPlane.send_shutdown()`).
* Join comm thread, then exit.

This is already how your PP0 pilot avoids interleaving (`pp0_pilot.py` comment: “send_shutdown only AFTER comms_thread.join()”).

If you try to send SHUTDOWN “while leader is computing”, you risk blocking in `dist.send` because the leader isn’t posting a recv. Don’t do that.

#### Leader (global rank 1)

Single-threaded loop:

1. `header, env = PPControlPlane.recv_next()` (world/pp_pg)
2. if `SHUTDOWN`: broadcast `SHUTDOWN` to mesh workers (mesh_pg), then exit loop
3. if `INFER`: validate, broadcast, run forward, send result
4. if validation fails: broadcast `ERROR`, send error status to rank0, exit

#### Mesh workers (global rank 2..N)

Loop:

1. receive action broadcast from leader on `mesh_pg`
2. if `SHUTDOWN` or `ERROR`: exit
3. if `INFER`: receive envelope payload broadcast, run forward, loop

#### Rank0 crash

* leader blocks in `recv_next()` until `SCOPE_DIST_TIMEOUT_S`, unless you add a watchdog thread like TP worker.
* workers block in mesh broadcast waiting for leader; watchdog is the escape hatch. (OM-13 “orphan shutdown” is exactly this.)

So for PP1: add **leader watchdog** + **worker watchdog** similar to `scope-drd/src/scope/server/tp_worker.py`’s pattern.

---

## Q6) Leader validate-before-broadcast under PP1

### Verdict

Use **(a) broadcast action first** (1 int64), then broadcast envelope only for INFER.

But: you must obey anti-stranding: the leader must **preflight everything that could throw before broadcasting INFER**.

### Concrete leader protocol (what to implement)

You want the exact same structure you already use in TP v0 and PP0:

* preflight before commitment point
* deterministic tensor_specs
* no throw-after-header

Implement on the leader something analogous to:

* `TPControlPlane.broadcast_infer` preflight (pickle meta, dtype checks) in `scope-drd/src/scope/core/distributed/control.py`
* `PPControlPlane.send_infer` preflight in `scope-drd/src/scope/core/distributed/pp_control.py`

#### Mesh broadcast message framing (recommended)

On leader (mesh group-rank 0, global rank 1):

1. **Preflight (no mesh collectives yet)**

   * validate envelope (`PPEnvelopeV1.validate_before_send()` is a good start, but also validate anything PP1-specific)
   * build `payload_meta = {"meta": env.metadata_dict(), "tensor_specs": env.tensor_specs()}`
   * `pickle.dumps(payload_meta)` must succeed
   * ensure all tensors exist, dtypes supported, shapes match spec, and are contiguous on-device

2. **Commitment point**

   * broadcast `action=INFER` on `mesh_pg` (a tiny int64 tensor)

3. broadcast `payload_meta` (object broadcast or byte broadcast)

4. broadcast tensors in spec order (all on `mesh_pg`)

5. run forward

If preflight fails:

* broadcast `action=ERROR` (so workers don’t hang waiting for payload)
* send an ERROR result/status back to rank0
* exit

### Rank0 unstick mechanism

Right now, if the leader exits early and rank0 keeps calling `dist.send`, rank0 can hang.

The lowest-complexity fix is: **guarantee that for every `send_infer`, the leader always sends back exactly one “result”, even if it’s an error.**

That implies changing the PP result contract:

* In `scope-drd/src/scope/core/distributed/pp_contract.py:PPResultV1`, add:

  * `ok: bool = True`
  * `error_message: str = ""`
  * optionally `action: int` (INFER/ERROR/SHUTDOWN)

Then:

* `PPResultV1.validate_before_send()` only requires `latents_out` when `ok=True`.
* `PPControlPlane.send_result()` can send an error result with `tensor_specs=[]` (no tensors).
* `PPControlPlane.recv_result()` returns a result that rank0 can interpret without hanging.

That’s strictly better than “timeouts only”, because it turns “mysterious hang” into “explicit error”.

Also: consider adding a `_broadcast_lock` equivalent to `PPControlPlane` (like TP has in `TPControlPlane._broadcast_lock`). You already rely on “don’t interleave shutdown with infer” at the caller; enforce it in the transport too.

---

## Red flags (things that will break under PP1 if unchanged)

These are not subtle.

1. **TP collectives currently use the default/world group.**

   * `scope-drd/src/scope/core/tensor_parallel/linear.py:_maybe_all_reduce` calls `dist.all_reduce(tensor, ...)` with no group.
   * `scope-drd/src/scope/core/tensor_parallel/rmsnorm.py:_all_reduce_sum` same.
     In PP1, rank0 is not in the mesh. Any default-group collective inside the mesh forward is an instant deadlock. This is OM-08.

2. **Functional collectives use `ranks=range(world_size)`**
   In compile mode you do:

   * `funcol.all_reduce(..., _tp_group_ranks(runtime.world_size))`
     That includes rank0 if `world_size = mesh+1`. Also deadlock.

3. **`runtime.init_distributed_if_needed()` enforces `WORLD_SIZE == SCOPE_TENSOR_PARALLEL` when TP>0**
   PP1 needs `WORLD_SIZE == mesh_tp + 1`. This check must change if `SCOPE_PP_ENABLED=1`.

4. **Recompute plan is currently not driven by the PP envelope**

   * `PPEnvelopeV1.do_kv_recompute` exists, but `RecomputeKVCacheBlock` doesn’t consult it.
   * Envelope provides `context_frames`; block looks for `context_frames_override`.
     This is a correctness break *before* it’s a performance issue.

5. **PP transport has no single-flight guard**
   `TPControlPlane` has `_broadcast_lock` and even `try_shutdown()` to avoid interleaving.
   `PPControlPlane` does not. You’re currently relying on “caller discipline” (join comm thread before shutdown). That’s fragile in the real server.

---

## Priority ranking (if you only fix 3 things before A5)

1. **Process-group topology + “wrong group” enforcement (OM-08).**
   Make mesh_pg real, store it somewhere (runtime), and require every TP collective to use it. Fix both eager and compile paths (`tensor_parallel/{linear,rmsnorm}.py`). Also fix any stray broadcasts (`prepare_video_latents.py` style hazards).

2. **Make recompute deterministic and plan-driven across the PP boundary (OM-05, OM-10).**
   Wire `do_kv_recompute` into `RecomputeKVCacheBlock`, and accept `context_frames` as the override (or rename the contract and bump version). Remove env-var-driven branching from the mesh path.

3. **Leader protocol: validate-before-broadcast + ERROR propagation + watchdogs (OM-07, OM-13).**
   Action-first broadcast on mesh_pg. On error, broadcast ERROR to workers and send an error result to rank0. Add watchdogs so crashes don’t become 300s waits.

Everything else (performance knobs like communicator resource sharing) can wait until you’ve got “crash > hang” nailed.

---

If you want, I can also sketch the exact PP1 leader/worker loop pseudocode in the style of your TP worker (`scope-drd/src/scope/server/tp_worker.py`), but the core point is: **single thread does all dist ops on mesh ranks; streams handle overlap; plan bits are the only branching authority.**
