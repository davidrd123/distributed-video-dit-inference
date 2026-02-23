## 1) Proposal review (Q1): 3-role dispatch + server surfaces

### ✅ The 3-role split is right

Your proposed roles match what the code already “wants”:

* **rank0 (non-mesh):** HTTP/WebRTC + “Stage0” responsibilities; talks to leader via **PPControlPlane**.
* **rank1 (mesh_rank=0):** mesh leader; receives PP envelopes and relays into TP mesh via **TPControlPlane**.
* **rank≥2 (mesh_rank>0):** TP workers; unchanged `TPControlPlane.recv_next()` loop.

This aligns with the already-landed runtime/control design:

* `DistributedRuntime` already differentiates **rank** vs **mesh_rank** and exposes `is_mesh_member` + `mesh_global_ranks`.
* `TPControlPlane` already routes broadcasts on `mesh_pg` and enforces leader role via `mesh_rank==0`.
* So the “mesh leader is global rank 1 under PP1” is already first-class in `TPControlPlane._resolve_mesh()`.

### ✅ app.py main dispatch change is necessary and correct

File: `scope-drd/src/scope/server/app.py` around **main() lines 4383–4484**.

Current behavior: all `rank!=0` go to `run_tp_worker_forever()`. Under PP1 this would incorrectly route **rank1 (mesh leader)** into the worker loop, and it will eventually explode because `TPControlPlane.recv_next()` rejects `mesh_rank==0` (see `TPControlPlane.recv_next()`).

Your proposed branch is correct:

* if `pp_enabled and mesh_rank==0`: run PP1 mesh leader loop
* else: run TP worker loop

**Nitpick:** do not hardcode `leader_rank=1` in multiple places. You already have `runtime.mesh_global_ranks[0]` in PP1 mode (it’ll be `1`). Use that so topology generalizes.

### ✅ New module `server/pp1_mesh_leader.py` is the clean separation

Option A (new file) is the right call. The PP1 mesh leader has a different failure envelope and different responsibilities (PP recv + TP broadcast + PP result). Keeping it out of `tp_worker.py` keeps invariants obvious.

### ❗ Major correction: “frame_processor already computes all envelope fields” is false

This is the biggest misconception in the proposal text.

`FrameProcessor.process_chunk()` currently computes **call_params**, not the PP boundary tensors. In the pure-TP server today, **the pipeline** computes or owns most of what PPEnvelopeV1 requires:

PPEnvelopeV1 requires (INFER):

* `conditioning_embeds` tensor (required)
* `current_denoising_step_list` tensor (required)
* `context_frames` tensor when recompute (required)
* `latents_in` tensor (required by validation)

But in `frame_processor.py`:

* conditioning override is **best-effort** and only built under a TP gating branch (see TP callsite around **lines 4320–4455**). It relies on `locked_pipeline.components.text_encoder`, which rank0 won’t have under PP1 unless you load it separately.
* context frames override is also **best-effort** and relies on `locked_pipeline.state` + `locked_pipeline.components.vae` (**lines 1–260** show it explicitly).
* `latents_in` is not something frame_processor currently materializes at all.

So PP1 server integration is not “just swapping broadcast_infer() for pp.send_infer()”. It’s a **stage split**: you must define what runs on rank0 vs mesh.

---

## 2) frame_processor recommendation (Q2): PP envelope path with concrete pseudocode

### What rank0 must own under PP1

Under PP1, rank0 cannot call the pipeline, so anything you need for:

* constructing the envelope inputs (conditioning + context frames + latents seed/init)
* decoding outputs for WebRTC
  must exist on rank0.

Given your current PPEnvelopeV1 schema, rank0 must (at minimum) own:

* **Text conditioning**: text encoder + blending logic (or a cached embedding bank).
* **VAE decode** (and likely encode-to-latent if you want correct recompute semantics later).
* **A small Stage0 state**: cache_epoch, last conditioning embeds, previous latents_out for context, etc.

### Minimal viable PP1 path (first working server)

Goal: get streaming working with *no VACE*, *no transitions*, *no fancy modes*, and keep behavior close to pilot.

**Key design choice:** keep mesh doing **generator-only** (`tp_plan="v1_generator_only"`) like pilot; rank0 does decode.

#### Pseudocode sketch inside `FrameProcessor.process_chunk()`

File: `scope-drd/src/scope/server/frame_processor.py` near TP callsite **(lines ~4320–4455)**.

```py
rt = get_distributed_runtime()
_pp1 = rt.initialized and rt.pp_enabled and rt.rank == 0
_tp0 = rt.initialized and rt.is_mesh_member and rt.mesh_rank == 0  # pure TP leader

if _pp1:
    # Stage0 components must be available on rank0:
    # - vae (decode; maybe encode_to_latent later)
    # - text encoder / embedding blender (or cached conditioning)
    stage0 = self._pp1_stage0  # new helper object
    pp = self._pp1_pp_control  # PPControlPlane(leader_rank=rt.mesh_global_ranks[0])

    # 1) derive envelope fields from current server params + stage0 state
    env = stage0.build_envelope(
        chunk_index=self.chunk_index,
        params=self.parameters,
        # plus any per-chunk things you already compute (height/width, seed, etc)
        video_input=video_input,  # maybe unsupported initially
    )

    # 2) send/recv
    pp.send_infer(env)
    res = pp.recv_result()  # should be guarded by watchdog (see Q5)

    if not res.ok:
        # decide policy: fatal vs reset+retry (see Q5)
        raise RuntimeError(res.error_message)

    latents_out = res.latents_out

    # 3) Stage0 decode (ideally async)
    frames = stage0.decode_latents(latents_out)

    # 4) push frames to output queue exactly like existing path
    self._emit_frames(frames)

else:
    # existing TP v0 path
    if _tp_runtime_enabled:
        call_params = get_tp_control_plane().broadcast_infer(call_params, chunk_index=self.chunk_index)
    output = locked_pipeline(**call_params)
```

### What PPResultV1 is missing for rank0 state machine parity

Right now, rank0 uses pipeline state after the call for at least:

* transition completion (`_transition_active`)
* conditioning embeds capture for k-SAE live (`conditioning_embeds_after_call`)
* maybe other internal counters

Under PP1, rank0 has no pipeline state, so you either:

1. **Disable these features under PP1** initially (honest and easy), or
2. Extend `PPResultV1` to return the few booleans / small tensors you need.

For minimal PP1 server bringup, I’d explicitly gate off:

* transitions
* k-SAE live
* VACE + control maps (already contract-gated via `pp_disable_vace=True`)

…and treat enabling them as Phase 2 work.

### Context frames under PP1: what to do right now

You already proved the synthetic approach works in PP1 pilot R0a/R0b: build recompute context from previous `latents_out` only.

For initial server integration, I’d do the same:

* store `prev_latents_out` in rank0 Stage0 state
* when `do_kv_recompute=True`, build `context_frames` from `prev_latents_out` (exactly like pilot)

This satisfies the hard guard in `RecomputeKVCacheBlock`:

* Under PP, it **refuses** to call `get_context_frames()` and requires `context_frames_override` (see `recompute_kv_cache.py`: it raises if PP enabled and override missing).

Later, if you want closer parity, rank0 must maintain a decoded buffer and occasionally re-encode the first frame (matching the non-PP path in `get_context_frames()`), which implies VAE encode too. But that’s Phase 2.

---

## 3) Pipeline load protocol (Q3): exact startup sequence across ranks

Here’s a sequence that matches your current code structure and avoids deadlocks.

### Current behavior summary

* rank0 loads pipeline in `app.py` lifespan (**lines ~200–539**) in TP mode.
* ranks>0 load pipeline in `tp_worker.py` (`_load_pipeline_if_requested`) and then `all_reduce(load_ok)` and `barrier()`.

### PP1 requirement

* rank0 must **not** load the pipeline (generator heavy).
* rank0 still participates in `all_reduce` + default-group `barrier()`.

### Proposed PP1 startup protocol

#### Rank0 (HTTP server process)

File: `scope-drd/src/scope/server/app.py` lifespan pipeline load section (**~lines 200–539**)

1. `init_distributed_if_needed()`
2. `pipeline_manager = PipelineManager()` (you may still want a Stage0 manager, but not the full pipeline)
3. If `tp_enabled` (tensor_parallel>0):

   * If `runtime.is_mesh_member` is False (PP1 rank0):

     * **skip** `pipeline_manager.load_pipeline()`
     * set `ok=True, load_exc=None`
   * participate in `all_reduce(load_ok_t)` on the default group
   * participate in `barrier()` on the default group
4. **Skip** `_maybe_tp_lockstep_warmup()` under PP1 (it will crash today, see below).
5. Start uvicorn.

#### Mesh ranks (ranks 1..N)

* rank1 (mesh leader) and rank≥2 (mesh workers) load the pipeline in their respective loops (`run_pp1_mesh_leader_forever()` and `run_tp_worker_forever()`) using the same `_load_pipeline_if_requested()` logic as tp_worker does today.
* they do the same `all_reduce(load_ok)` and default-group `barrier()`.

### Two critical gating fixes (otherwise PP1 crashes at startup)

These are non-negotiable:

1. `_maybe_tp_lockstep_warmup()` must not run on PP1 rank0
   File: `scope-drd/src/scope/server/app.py`, function `_maybe_tp_lockstep_warmup()` and its call site in lifespan.
   Reason: it calls `get_tp_control_plane().broadcast_infer(...)`, which requires `mesh_rank==0`. In PP1 rank0 has `mesh_rank==-1`, so it will throw.

2. TP heartbeat must not run on PP1 rank0
   File: `scope-drd/src/scope/server/app.py` heartbeat task creation (**~line 433 in your proposal**).
   Reason: it calls `control.broadcast_noop()` which again requires `mesh_rank==0`. PP1 rank0 is not mesh leader.

### Warmup decision (Q1/Q3 overlap)

Given Krea constructor skips warmup when `pp_enabled` or `tp_degree>0` (see pipeline snippet **lines 362–385**), you need some warmup strategy.

My recommendation:

* **Mesh leader runs warmup autonomously** before entering the PP recv loop.
* Do it via TPControlPlane broadcasts so workers compile consistently.
* No need for rank0 involvement.

If you want rank0 to block until warmup is done, do a default-group `barrier()` after warmup too (rank0 waits). If you want health to come up earlier, do not block; instead surface “warming_up=true” in `/health` from rank0 until mesh leader sends a one-time ready signal (PP NOOP + result, or a small “READY” PP action, see red flags section).

---

## 4) Rank0 VAE recommendation (Q4): pick (a), but do it with overlap

### Pick (a): rank0 loads only VAE (plus text encoder if you keep current PPEnvelope schema)

Option (a) is the only one that scales sanely with bandwidth and keeps the mesh focused on generator throughput.

Option (b) (mesh decodes and sends pixels) is viable as a temporary hack, but it:

* increases per-chunk transfer size
* couples mesh to output format/resolution
* wastes the rank0 GPU if it’s otherwise idle

Option (c) is extra moving parts with no payoff yet.

### Latency and overlap

From your measured PP1 times on 3×A100:

* R1 median chunk period ~282ms (no recompute)
* R0a recompute chunks ~514ms
  (from bringup log runs 19/24/25)

If rank0 decode takes, say, O(50–100ms) per chunk, you can hide most of it by pipelining:

* mesh is on GPUs 1..N
* rank0 decode is on GPU0
* overlap decode for chunk k while mesh runs chunk k+1

So the right shape is:

* a background decode worker/thread on rank0
* the PP send/recv loop feeds it latents_out
* WebRTC output reads decoded frames from a queue

You already allude to this as “A3 overlap pattern”. Yes. Do it. It’s the easiest free win you have.

### GPU allocation reality check (4×A100)

With 4 GPUs total and PP1 topology:

* rank0 uses GPU0 (Stage0 VAE + text encoder)
* mesh uses GPUs1..(tp_degree)
  If tp_degree=2, GPU3 is idle.

You suggested TP=3 as an alternative (rank0 VAE + TP=3 on GPUs1–3). With `num_heads=40`, TP=3 is almost certainly invalid unless you’ve implemented a non-head-even sharding scheme. Your TP baselines confirm head sharding is the scheme (TP=2, TP=4 are clean). So:

* **TP=3 is a no-go** unless you change how you shard attention.
* PP1 on 4 GPUs probably means **TP=2 + 1 idle GPU** for now.

If you really want to use all 4 GPUs for generator, PP1-in-one-torchrun-job is the wrong packaging. You’d need rank0 outside the torchrun GPU assignment (separate process/job) or world_size=5.

---

## 5) Error handling spec (Q5): what to do in a torchrun server

### First: be honest about torchrun failure domains

If any rank hard-crashes, most torchrun setups will take the whole job down. So “mesh errors should not crash HTTP server” is only achievable if:

* you catch errors on mesh ranks and keep the processes alive, or
* you accept full job restart managed by a supervisor (systemd, k8s, etc).

Given your current worker loop (`tp_worker.py`) raises on pipeline exception and exits, the realistic production policy is:

* **Fail fast on distributed errors, restart the job.**

You can still keep *the HTTP server codepath* from exploding on a single bad chunk, but if the mesh is broken the job is broken.

### Recommended policies

#### (A) PPResult ok=False

* Treat as fatal for the inference subsystem.
* rank0 should:

  * log `error_message`
  * attempt `pp.try_shutdown()` (best effort) so the mesh doesn’t hang
  * then either:

    * `os._exit(2)` (clean fail-fast; supervisor restarts), or
    * set a “degraded mode” flag and stop generation loops (health goes red), if you’re running without a supervisor and prefer manual recovery.

Given you already designed OM tests to prefer “crash > hang”, I’d keep that rule.

#### (B) recv hang / mesh deadlock

`PPControlPlane.recv_result()` is blocking and has no timeout. PyTorch `dist.recv` won’t give you a clean timeout without switching to async recv or a helper thread.

The pragmatic move (you already used it in tp_worker watchdog) is:

* add a **rank0 watchdog thread** that tracks “time since last PP result”
* if it exceeds `SCOPE_PP_WATCHDOG_S` (new env), log and `os._exit(2)`

**Timeout value:**
Based on your data:

* steady chunks are 0.28–0.55s
* worst warm/cold chunk in pilot is ~3.5s

So:

* `SCOPE_PP_WATCHDOG_S = 10s` is safe and catches real hangs quickly.

#### (C) Retry vs skip for streaming

Retrying a failed chunk is usually wrong if the failure is on mesh execution. You’ll likely retry into the same broken state. If you want a “soft” recovery attempt:

* on first failure: force a hard cut (`init_cache=True`, `reset_kv_cache=True`, `reset_crossattn_cache=True`) and try **one** restart chunk
* if that fails: fail-fast

But do not silently keep going; you’ll drift into non-debuggable state.

### Shutdown propagation (must-fix)

File: `scope-drd/src/scope/server/app.py` lifespan shutdown tail (**~lines 485+ in proposal; in slice it’s near the end of lifespan**)

Current: rank0 calls `get_tp_control_plane().shutdown()` if distributed and rank0.
Under PP1 that will throw (rank0 is not mesh leader).

Correct behavior under PP1:

* rank0 sends PP shutdown to leader: `PPControlPlane.send_shutdown()`
* leader relays: `TPControlPlane.shutdown()` inside mesh leader process.

Also: add a best-effort `try_shutdown` path in the exception handlers (you already did this pattern in pilot leader loop).

---

## 6) Test plan (Q6): incremental checklist

### Stage 0: startup-only, no WebRTC

1. Launch PP1 server under torchrun:

   * `SCOPE_PP_ENABLED=1`
   * `SCOPE_TENSOR_PARALLEL=2`
   * `WORLD_SIZE=3`
   * `PIPELINE=krea-realtime-video`
2. Expected:

   * rank0 starts uvicorn and `/health` returns 200
   * rank1 runs mesh leader loop, loads pipeline
   * rank2 runs tp worker loop, loads pipeline
   * all ranks pass load_ok all_reduce + barrier

**What this validates:** dispatch + lifecycle + load synchronization.

### Stage 1: PP NOOP plumbing sanity

Add a trivial PP action path:

* rank0 sends `PPAction.NOOP`
* leader receives and does nothing (or optionally replies with a small PPResultV1 “ok” ping if you add a PPAction for that)

This validates PP channel isn’t wedged before you debug video.

### Stage 2: “latents-only” fake output

Before full WebRTC:

* in frame_processor, run one PP infer per chunk and just confirm `latents_out` arrives and matches expected shape
* do not decode; just measure throughput and confirm call ordering

This isolates PP1 inference path from VAE decode and WebRTC.

### Stage 3: rank0 decode + WebRTC

* enable VAE-only on rank0
* decode `latents_out` and feed existing output queue path
* verify stream is visible and stable

### Stage 4: recompute

* enable `do_kv_recompute` scheduling (match pilot R0b, recompute every 5)
* verify recompute chunks are slower, non-recompute chunks match baseline
* confirm no `RecomputeKVCacheBlock` error about missing `context_frames_override`

### Stage 5: shutdown + idle keepalive

* leave server idle longer than worker watchdog threshold (if set)
* verify mesh does not die (requires mesh leader TP heartbeat)
* Ctrl-C rank0: verify PP shutdown propagates and torchrun exits cleanly

### OM tests reuse

Your existing OM PP1 tests (om_07 leader safety, om_13 orphan shutdown) are still relevant. Add one server-mode-specific OM test later:

* “server startup with PP1 does not call TPControlPlane from rank0” (basically asserts you gated warmup/heartbeat/shutdown correctly).

---

## 7) Red flags (stuff that will bite you)

1. **PPEnvelopeV1 requires `latents_in` but pilot doesn’t forward it into call_params.**
   In `pp1_pilot.py`, rank0 sets `latents_in=...` but `_pp1_unpack_to_call_params()` never uses it.
   That means you’re shipping a tensor per chunk that currently does nothing. Either:

* remove `latents_in` from the contract (make optional), or
* actually plumb it into stage1 so the contract matches reality.

2. **app.py warmup/heartbeat/shutdown will crash under PP1 unless gated.**
   All three call TPControlPlane methods from rank0, and TPControlPlane explicitly requires `mesh_rank==0`. PP1 rank0 has `mesh_rank==-1`.

Concrete sites:

* `_maybe_tp_lockstep_warmup()` call in lifespan
* heartbeat task creation block (rank0)
* shutdown block at end of lifespan

3. **frame_processor currently depends on `locked_pipeline` for prepare(), conditioning override, context override, transitions.**
   If rank0 doesn’t load a pipeline, you need a Stage0 replacement for:

* input requirements (`prepare()`) or a new way to size chunks
* text encoder path
* VAE decode path
* transition bookkeeping (or disable)

4. **Mesh worker watchdog will kill idle servers unless mesh leader provides a TP heartbeat.**
   Today the heartbeat is on rank0 only. Under PP1, rank0 must not broadcast NOOP into TP. So the mesh leader must do it.

5. **Barrier group semantics:** your `barrier()` helper uses the default process group always.
   That’s fine for “all ranks ready”, but if you try to use barrier for “mesh ready” you’ll stall rank0 too. If you need mesh-only synchronization, you need an explicit `dist.barrier(group=get_mesh_pg())` call site (or a helper).

---

## 8) Implementation ordering + top-3 priorities

If you only do three things before the first PP1 server-mode run (startup + health):

### Priority 1: Process-role dispatch + new mesh leader module

* File: `scope-drd/src/scope/server/app.py` main() **(~4383–4484)**
* Add PP1 branch:

  * `pp_enabled and mesh_rank==0` → `run_pp1_mesh_leader_forever()`
  * else → `run_tp_worker_forever()`
* Create `scope-drd/src/scope/server/pp1_mesh_leader.py` by lifting pilot leader loop.

### Priority 2: Lifespan gating for PP1 (load, warmup, heartbeat, shutdown)

* File: `scope-drd/src/scope/server/app.py` lifespan **(~200–539)**
* Gate pipeline load on `runtime.is_mesh_member`
* Skip `_maybe_tp_lockstep_warmup()` when `runtime.pp_enabled`
* Skip TP heartbeat when `runtime.pp_enabled`
* Replace shutdown with PP shutdown when `runtime.pp_enabled`

This prevents immediate PP1 crashes.

### Priority 3: Minimal frame_processor PP1 path that at least round-trips latents

* File: `scope-drd/src/scope/server/frame_processor.py` TP callsite **(~4320–4455)**
* Add `_pp_enabled` gating and swap TP broadcast path for PP send/recv.
* Initially you can skip decode and just validate the PP loop is functional.
* Then add rank0 VAE decode as the next increment.

---

If you want one small “be present” nudge: look away from the screen for 10 seconds, drop your shoulders, and unclench your jaw.
