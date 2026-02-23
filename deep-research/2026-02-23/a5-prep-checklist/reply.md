Got it. I’m going to “address” `pp1-a5-prep-checklist.md` the way I’d want it if I were about to implement PP1: concrete script skeletons, what TPControlPlane actually guarantees (and what it doesn’t), and what that implies for the PP1 leader protocol + operator tests.

I’m focusing on: **P‑1**, **P‑2b**, and what that forces for **P‑3/P‑4**. I’m not going to assume a single “correct” architecture; I’ll lay out the viable branches and what each costs.

---

## P‑1: empirical `new_group` test (implementation-ready)

### Why P‑1 matters (in one sentence)

PP1 lives or dies on “can one process participate in **two communicators** (rank0↔leader p2p + leader↔mesh collectives) without weird deadlocks/ordering hazards,” and the only non-theoretical check you can do cheaply is an empirical `new_group` suite.

### Reality check about the checklist’s Step 1/3 on 2×GPU

Your checklist already calls it out, but I’ll be explicit: with 2 ranks you **cannot** create a multi-rank mesh that excludes rank0, so you can’t test “rank0 sends while mesh all_reduces” in a faithful way. What you *can* test:

* **Non-member handle semantics**: what does `new_group(ranks=[1])` return on rank0?
* **Group creation/destroy cost**.
* **Multiple communicators with same membership** (two distinct `new_group([0,1])`) and whether you can have **overlapping async ops** across them without deadlock.

The last bullet is not identical to PP1, but it’s the closest you can get with 2 ranks and it’s still useful: PP1 leader is a “communicator junction.”

### Script: `scripts/pp1_new_group_test.py`

This is designed to:

* run under `SCOPE_PP_ENABLED=1 SCOPE_TENSOR_PARALLEL=0` so your runtime init doesn’t trip TP world-size checks,
* avoid python threads calling `dist.*` concurrently (since you already treat that as risky elsewhere),
* use **async ops** (`async_op=True`, `isend/irecv`) to create real overlap without threads,
* output a single JSON report on rank0 (via a CUDA-safe UTF-8 all_gather).

Copy-paste version:

```py
#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
import torch.distributed as dist

from scope.core.distributed import (
    init_distributed_if_needed,
    destroy_distributed_if_needed,
    get_distributed_runtime,
)

# -------------------------
# Helpers (CUDA-safe gather)
# -------------------------

def _all_gather_utf8(payload: str, *, device: torch.device) -> list[str]:
    b = payload.encode("utf-8", errors="replace")
    n = int(len(b))
    n_t = torch.tensor([n], device=device, dtype=torch.int64)
    sizes = [torch.empty_like(n_t) for _ in range(int(dist.get_world_size()))]
    dist.all_gather(sizes, n_t)
    lens = [int(x.item()) for x in sizes]
    max_len = max(lens) if lens else n

    data = torch.zeros((max_len,), device=device, dtype=torch.uint8)
    if n:
        data[:n] = torch.tensor(list(b), device=device, dtype=torch.uint8)
    gathered = [torch.empty_like(data) for _ in range(int(dist.get_world_size()))]
    dist.all_gather(gathered, data)

    out: list[str] = []
    for i, t in enumerate(gathered):
        ln = int(lens[i]) if i < len(lens) else 0
        out.append(bytes(t[:ln].cpu().tolist()).decode("utf-8", errors="replace"))
    return out


@dataclass
class Subtest:
    name: str
    ok: bool
    ms: float
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def _run(name: str, fn) -> Subtest:
    t0 = time.perf_counter()
    try:
        details = fn() or {}
        t1 = time.perf_counter()
        return Subtest(name=name, ok=True, ms=(t1 - t0) * 1000.0, details=dict(details), error=None)
    except Exception as e:
        t1 = time.perf_counter()
        return Subtest(name=name, ok=False, ms=(t1 - t0) * 1000.0, details={}, error=f"{type(e).__name__}: {e}")


def _env_true(name: str) -> bool:
    return (os.getenv(name, "") or "").strip().lower() in {"1", "true", "yes", "on"}


def main() -> int:
    # Force PP-only runtime so WORLD_SIZE can be 2 with TP disabled.
    if int(os.getenv("SCOPE_TENSOR_PARALLEL", "0") or "0") != 0:
        raise RuntimeError("This test expects SCOPE_TENSOR_PARALLEL=0")
    if not _env_true("SCOPE_PP_ENABLED"):
        raise RuntimeError("This test expects SCOPE_PP_ENABLED=1")

    rt = init_distributed_if_needed()
    if not rt.initialized:
        raise RuntimeError("Expected distributed runtime to initialize.")
    if int(rt.world_size) != 2:
        raise RuntimeError(f"Expected world_size=2 for this script, got {rt.world_size}")

    device = torch.device(rt.device)
    rank = int(rt.rank)

    results: list[Subtest] = []

    # ------------
    # P-1.1: new_group([1]) non-member semantics
    # ------------
    # IMPORTANT: new_group is collective; all ranks must call it in the same order.
    def _mk_nonmember_group():
        t0 = time.perf_counter()
        pg = dist.new_group(ranks=[1])
        t1 = time.perf_counter()
        # Non-members may see GroupMember.NON_GROUP_MEMBER, or a ProcessGroup with no membership.
        return {
            "create_ms": (t1 - t0) * 1000.0,
            "pg_type": str(type(pg)),
            "pg_repr": repr(pg),
            "is_non_member_sentinel": bool(pg == dist.GroupMember.NON_GROUP_MEMBER),
        }

    results.append(_run("new_group_nonmember_handle", _mk_nonmember_group))

    # Re-create it so we have a local var (we don’t trust the details dict for logic).
    pg_nonmember = dist.new_group(ranks=[1])

    # ------------
    # P-1.2: Two distinct 2-rank groups (A and B) for overlap tests
    # ------------
    def _mk_two_groups():
        t0 = time.perf_counter()
        pg_a = dist.new_group(ranks=[0, 1])
        t1 = time.perf_counter()
        pg_b = dist.new_group(ranks=[0, 1])
        t2 = time.perf_counter()
        return {
            "pg_a_type": str(type(pg_a)),
            "pg_b_type": str(type(pg_b)),
            "pg_a_id": id(pg_a),
            "pg_b_id": id(pg_b),
            "same_object": bool(pg_a is pg_b),
            "create_a_ms": (t1 - t0) * 1000.0,
            "create_b_ms": (t2 - t1) * 1000.0,
        }

    results.append(_run("new_group_two_distinct_full_groups", _mk_two_groups))

    pg_a = dist.new_group(ranks=[0, 1])
    pg_b = dist.new_group(ranks=[0, 1])

    # ------------
    # P-1.3: World p2p while rank1 does a mesh-only op (single-rank group)
    # (Not a real NCCL concurrency test; just API/mechanics.)
    # ------------
    def _world_send_recv_vs_mesh_bcast_singleton():
        # rank1 does a broadcast on singleton group; should not involve rank0 at all.
        # Meanwhile rank0<->rank1 do a world-group isend/irecv.
        t = torch.arange(1024, device=device, dtype=torch.int64)
        if rank == 0:
            w = dist.isend(t, dst=1)  # world
            w.wait()
        else:
            # Do singleton broadcast first (no-op-ish), then recv world payload.
            u = torch.zeros_like(t)
            if pg_nonmember != dist.GroupMember.NON_GROUP_MEMBER:
                dist.broadcast(t, src=1, group=pg_nonmember)
            w = dist.irecv(u, src=0)  # world
            w.wait()
            if not torch.equal(u.cpu(), torch.arange(1024, dtype=torch.int64)):
                raise RuntimeError("world recv mismatch")
        return {}

    results.append(_run("world_p2p_while_singleton_group_op", _world_send_recv_vs_mesh_bcast_singleton))

    # ------------
    # P-1.4: Overlap test without threads:
    # async all_reduce on pg_a concurrently with isend/irecv on pg_b
    # ------------
    def _overlap_allreduce_vs_p2p_two_groups():
        # Make payload non-trivial so ops aren’t instantly done.
        n = 4 * 1024 * 1024  # 4M elems; float32 -> 16MB
        x = torch.ones((n,), device=device, dtype=torch.float32) * float(rank + 1)
        y = torch.empty((n,), device=device, dtype=torch.float32)

        # 1) kick async all_reduce on group A
        w1 = dist.all_reduce(x, op=dist.ReduceOp.SUM, group=pg_a, async_op=True)

        # 2) kick async p2p on group B
        if rank == 0:
            y.copy_(x)
            w2 = dist.isend(y, dst=1, group=pg_b)
        else:
            w2 = dist.irecv(y, src=0, group=pg_b)

        # 3) wait both
        w2.wait()
        w1.wait()

        if rank == 1:
            # Expect sender’s original values (rank0 had x=1’s before all_reduce finishes).
            # This isn’t a correctness test for ordering; it’s just a “did we deadlock / corrupt?”
            if not torch.isfinite(y).all():
                raise RuntimeError("non-finite recv payload")

        return {"n_elems": n, "bytes_per_tensor": int(n * 4)}

    results.append(_run("overlap_allreduce_pgA_vs_p2p_pgB_async", _overlap_allreduce_vs_p2p_two_groups))

    # ------------
    # P-1.5: destroy_process_group semantics
    # ------------
    def _destroy_groups():
        out: dict[str, Any] = {}
        # Nonmember group destroy behavior is what we care about.
        try:
            dist.destroy_process_group(pg_nonmember)
            out["destroy_nonmember_ok"] = True
        except Exception as e:
            out["destroy_nonmember_ok"] = False
            out["destroy_nonmember_err"] = f"{type(e).__name__}: {e}"

        # Full groups should be destroyable on both ranks.
        dist.destroy_process_group(pg_a)
        dist.destroy_process_group(pg_b)
        out["destroy_full_groups_ok"] = True
        return out

    results.append(_run("destroy_process_groups", _destroy_groups))

    # Final barrier so rank0 doesn’t exit early.
    dist.barrier()

    # Gather per-rank JSON via CUDA-safe all_gather.
    payload = json.dumps(
        {"rank": rank, "world_size": int(rt.world_size), "device": str(rt.device), "results": [asdict(r) for r in results]},
        sort_keys=True,
    )
    gathered = _all_gather_utf8(payload, device=device)

    if rank == 0:
        per_rank = [json.loads(s) for s in gathered]
        report = {
            "meta": {
                "world_size": int(rt.world_size),
                "backend": str(rt.backend),
                "device_rank0": str(rt.device),
            },
            "per_rank": per_rank,
        }
        print(json.dumps(report, indent=2, sort_keys=True))

    destroy_distributed_if_needed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

**Run:**

```bash
SCOPE_PP_ENABLED=1 SCOPE_TENSOR_PARALLEL=0 \
uv run torchrun --nproc_per_node=2 scripts/pp1_new_group_test.py | tee outputs/pp1_new_group_test.json
```

### What to look for in the JSON (interpretation guide)

* `new_group_nonmember_handle.is_non_member_sentinel`:

  * If `true`, your “non-member handle” is `GroupMember.NON_GROUP_MEMBER`. That means any future code must **not** call collectives on that handle from non-member ranks (should treat it like `None`).
* `destroy_process_groups.destroy_nonmember_ok`:

  * If destroying a non-member group throws, you have to ensure only members destroy, or guard it.
* `overlap_allreduce_pgA_vs_p2p_pgB_async`:

  * If this hangs or errors, **don’t** try to do PP p2p and mesh collectives concurrently in separate streams/works without additional discipline.
  * If it passes, you still haven’t proven PP1’s partial-overlap case, but you at least know two communicators can have in-flight ops simultaneously.

---

## P‑2b: TPControlPlane deep read (answers you can code against)

You wanted five specific questions. Here they are, with the non-handwavy bits.

### 1) What `_broadcast_lock` protects

It serializes the entire “commitment point” sequence for rank0 broadcasts:

* increments (`_call_id`, `_control_epoch`, maybe `_cache_epoch`)
* broadcasts header
* broadcasts payload meta
* broadcasts tensors
* optional input-digest check

Meaning: it prevents **interleaving** of two broadcast envelopes if someone calls `broadcast_infer()` concurrently (different threads or reentrant code). It’s not about distributed safety, it’s about local sequencing.

### 2) Broadcast sequence + wire format

For **INFER**:

1. `_exchange_header(header)`

   * `header_tensor = [call_id, chunk_index, control_epoch, cache_epoch, action]` on leader
   * `dist.broadcast(header_tensor, src=0)` on default group
2. `dist.broadcast_object_list([payload_meta], src=0)`

   * payload_meta is `{"kwargs": kwargs_obj, "tensor_specs": tensor_specs}`
3. For each `spec` in `tensor_specs`:

   * worker allocates `torch.empty(shape, dtype)`
   * `dist.broadcast(t, src=0)`

Workers know what to expect because:

* header’s `action` tells them whether to expect a payload,
* payload meta includes tensor_specs (dtype + shape), so they can allocate exactly.

### 3) Worker blocking and timeout behavior

`TPControlPlane.recv_next()` is a pure blocking sequence of `dist.broadcast*` calls. There is **no** per-call timeout at this layer.

The *only* hang hygiene here is external:

* process group timeout (`SCOPE_DIST_TIMEOUT_S` in runtime init)
* the worker watchdog in `tp_worker.py` (`SCOPE_TP_WORKER_WATCHDOG_S`) which exits the process if no header arrives for a while.

So: if the leader never broadcasts a header, workers sit in `_exchange_header(None)` until:

* the process group timeout triggers, or
* watchdog kills them.

### 4) Error propagation (what happens if leader throws)

There is no “error broadcast.” The only control-plane actions are `NOOP/INFER/SHUTDOWN`.

The key protection is the **anti-stranding preflight** inside `broadcast_infer()`:

* split tensors/meta
* materialize tensors
* `pickle.dumps(payload_meta)`
* validate each tensor spec’s dtype/shape

Only after those succeed does it send the INFER header. That’s exactly the “commitment point” discipline you want.

But: if the leader throws **after** `_exchange_header` (e.g., during `dist.broadcast_object_list`, CUDA OOM during tensor broadcast, etc.), workers will block mid-sequence. There’s no graceful recovery except timeouts/watchdog.

### 5) kwargs → pipeline state flow (how overrides propagate)

On each rank, worker dispatch calls `locked_pipeline(**call_params)` or `tp_worker_infer(**call_params)`.

In `KreaRealtimeVideoPipeline._generate`, it does:

```py
for k, v in kwargs.items():
    self.state.set(k, v)
```

So any broadcast kwargs become state. That’s why keys like `context_frames_override`, `conditioning_embeds_override`, `video_latents_override`, etc. are “transported” implicitly by broadcast_infer. No special casing needed as long as the key names match what blocks read.

---

## What P‑2b implies for PP1 (P‑3/P‑4), without picking one path

Your checklist lists three options (a/b/c). Here’s the honest shape of each, given what TPControlPlane actually is.

### Option A: “PP envelope → TP broadcast kwargs” (reuse TP wire format)

**Idea:** PP1 leader receives `PPEnvelopeV1`, converts to the kwargs that mesh workers expect, then runs a TP-style broadcast to mesh ranks.

**What you have to change / add:**

* TPControlPlane is hard-coded to:

  * leader is **global rank 0**
  * group is **default world**
* For PP1, you need:

  * leader = **global rank 1**
  * group = **mesh_pg (ranks [1,2,…])**
* So you need either:

  1. a *new* `TPControlPlaneMesh(group, leader_rank)` implementation, or
  2. refactor TPControlPlane to accept `group` + `leader_global_rank` (defaulting to world/0).

**Why this option is attractive:**

* You get the existing anti-stranding discipline “for free.”
* You can keep worker receive logic basically identical (just instantiate with group).

**Where it bites you later:**

* TP collectives are currently global-world-size based. In PP1, TP inside mesh needs to operate on mesh group, not world.
* That is A5’s big blocker anyway, but you’ll run into it fast.

### Option B: “New PP-specific broadcast on mesh_pg” (workers are PP-aware)

**Idea:** leader broadcasts the raw `PPEnvelopeV1` (metadata + tensors) over mesh_pg, using the PP contract as the transport schema, not TP kwargs.

**What you have to build:**

* A mesh-local “envelope broadcast” that looks like PPControlPlane + TPControlPlane hybrid:

  * preflight meta pickling + tensor specs before broadcasting an INFER commitment
  * broadcast meta bytes + tensor specs + tensors to mesh group
* Mesh workers implement `recv_envelope()` and then call a stage1 runner.

**Why this option is attractive:**

* It respects your “stage boundary contract is the only thing that crosses” rule.
* Workers can be simpler (they only know PPEnvelope, not frame-processor kwargs soup).
* It sets you up to prune blocks cleanly (mesh doesn’t have to pretend it’s running the same pipeline path as TP v0).

**Why it’s more work now:**

* You’re duplicating a broadcast protocol you already have in TPControlPlane.
* You need new worker main loop (PP-aware) anyway.

### Option C: “Transparent passthrough via existing tp_worker_infer” (workers don’t know PP exists)

This is basically Option A *plus* “don’t change worker code at all.” It’s only viable if:

* mesh leader can act like “rank0” of a TP run inside mesh_pg, and
* the call_params needed by `tp_worker_infer` can be derived cleanly from `PPEnvelopeV1`.

Given current TPControlPlane hard-coding, this requires the same group/leader refactor as Option A.

**Important nit:** in your current PP0 pilot, the `tp_worker_infer` worker mode is not obviously using `env.latents_in` at all (it’s building call_params from env fields like conditioning + denoising steps, but not passing latents). That’s fine for bringup but it means “Option C works” is not the same as “Option C runs the correct PP boundary tensors.” Keep that distinction clear.

---

## P‑3: leader state machine sketch, conditional on the option you pick

Instead of one diagram, here are two “happy path” sketches (A/C vs B). Error path is where OM‑07 matters.

### If you reuse TP broadcast semantics (Option A/C)

Leader loop (global rank 1):

```
while True:
  header, env = pp.recv_next()    # rank0 → leader p2p
  if header.action == SHUTDOWN:
      tp.broadcast_shutdown(mesh_pg)   # terminal broadcast to mesh workers
      break
  if header.action != INFER:
      tp.broadcast_noop(mesh_pg)
      continue

  # leader-side validate BEFORE mesh broadcast commitment
  validate(env)

  call_params = envelope_to_call_params(env)
  tp.broadcast_infer(call_params, chunk_index=env.chunk_index)  # mesh_pg leader=rank1

  # leader also runs Phase B as a TP participant
  latents_out = pipeline.tp_worker_infer(**call_params)  # or stage1 runner

  result = build_pp_result(...)
  pp.send_result(result)     # leader → rank0 p2p
```

Mesh non-leader loop (global rank ≥2):

* identical to TP worker loop except it uses mesh control plane / group
* `recv_next()` gets header+payload, then runs tp_worker_infer.

### If you use PPEnvelope broadcast (Option B)

Leader:

```
while True:
  header, env = pp.recv_next()  # rank0 → leader p2p
  if header.action != INFER:
      broadcast_terminal_or_noop(...)
      if shutdown: break
      continue

  validate(env)                 # must be before broadcast
  mesh_bcast_env(env)           # mesh_pg

  latents_out = stage1_runner(env)   # pure Phase-B runner
  pp.send_result(result)
```

Mesh workers:

```
while True:
  env = mesh_recv_env()
  if env.action == SHUTDOWN: break
  stage1_runner(env)
```

---

## P‑4: scaffolding OM‑07 / OM‑08 / OM‑13 (what you can write *now*)

Your checklist puts these in `tests/test_pp1_operator_matrix.py`, but the repo’s “real” distributed validation lives in `scripts/operator_matrix/` under `run_operator_matrix.py`. I’d scaffold as **torchrun scripts**, not pytest, because:

* you need real multi-rank behavior,
* pytest doesn’t own process-group lifecycle well.

That said: you can still write pytest *unit* scaffolds (monkeypatching `dist`) for logic, but the hang-failure modes won’t be covered.

### OM‑07: leader safety (no broadcast if leader rejects)

Scaffold idea that works today:

* Add (or plan to add) a leader-side check that rank0 validation doesn’t enforce.
* Easiest candidate: `pp_disable_vace` is intended as a gate but `PPEnvelopeV1.validate_before_send()` doesn’t currently assert it’s `True`.

So the test can:

1. rank0 constructs an envelope that passes rank0 validation but should fail leader validation:

   * set `pp_disable_vace=False` (or set an unknown `pp_envelope_version` if you stop validating version on sender later).
2. rank0 sends it.
3. leader should:

   * **not** start mesh broadcast of an INFER payload,
   * instead broadcast terminal `SHUTDOWN` or `ERROR` to mesh workers,
   * then exit cleanly.

**Scaffold without committing to SHUTDOWN vs ERROR:**

* Accept either terminal action for now, but require “non-leaders do not hang.”

### OM‑08: wrong-group collective (fail before calling NCCL)

This one is impossible to “really” implement until you have explicit-group plumbing in your TP layer, because right now most collectives implicitly use the default group (and funcol uses global rank lists).

So scaffolding should look like:

* “When TP wrappers take a `group`/`group_ranks`, assert the passed group matches mesh_pg; if not, raise before calling dist.”

For now you can stub a test that:

* creates a fake “expected group id”
* calls wrapper with wrong group
* asserts it raises with a message containing `group_used vs expected`.

### OM‑13: orphan shutdown (rank0 crash hygiene)

Today, hang hygiene is mostly:

* `SCOPE_DIST_TIMEOUT_S`
* worker watchdog `SCOPE_TP_WORKER_WATCHDOG_S`

PP1 will need an additional mechanism:

* leader times out waiting for rank0 p2p receive and then broadcasts terminal action to mesh workers.

**Scaffold:**

* launch PP1-ish topology under CPU/gloo (so you can do 3 ranks on your dev machine)
* have rank0 exit abruptly (os._exit) after a few NOOPs
* assert:

  * leader notices within watchdog/timeout window,
  * non-leaders exit (via leader broadcast or their own watchdog), not hang for 300s.

---

## Two concrete “next edits” I’d do after P‑1 + P‑2b, before A5 starts

Not asked, but this is what will save you time:

1. **Write down the required refactor surface for “TPControlPlane but with group+leader”:**

   * `_broadcast_object`, `_broadcast_tensor`, `_exchange_header`, `_maybe_check_input_digest`, and every `dist.*` call need `group=...` and `src=leader_global_rank`.
   * You also need `group_ranks` for funcol collectives and digest all_gathers.

2. **Add a PP1-only env-parity strategy (role-aware), or you’ll deadlock by configuration drift:**

   * current parity check assumes identical env across ranks, including `SCOPE_TENSOR_PARALLEL`. That can’t hold in PP1 if rank0’s role differs.
   * Prep work here is just: decide what must be equal across *all* ranks vs equal within *mesh* ranks.

---

If you want one thing to do immediately: copy in the P‑1 script above and run it on the 2×H100 box. It gives you actual facts about `new_group` handles + destroy behavior + “two communicators with overlap async ops” without touching PP1 proper.

Also: take 10 seconds, unclench your jaw, drop your shoulders. You’ll debug faster.
