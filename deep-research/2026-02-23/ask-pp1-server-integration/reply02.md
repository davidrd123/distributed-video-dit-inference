## 1. Proposal review

### What’s already right (and matches your proposal)

* **Three-role dispatch is correct**:

  * `app.py:main()` routes **rank>0** into either:

    * **PP1 mesh leader** when `pp_enabled and mesh_rank==0` → `run_pp1_mesh_leader_forever()`
    * **TP worker** otherwise → `run_tp_worker_forever()`
      This matches the “rank0 HTTP + rank1 bridge + rank2..N workers” split in `pp1-server-integration.md`.
    * Evidence: `scope/server/app.py` main dispatch block (≈4414+ in your slice).

* **Startup collectives are (mostly) aligned across ranks**:

  * Rank0 lifespan does `all_reduce(load_ok)` + `barrier()` on the **default group** even under PP1, while skipping pipeline load when `not is_mesh_member`.
  * Mesh leader and workers do the same via `_load_pipeline_if_requested()` (tp_worker), which also does `all_reduce(load_ok)` + `barrier()`.
  * This is the key “don’t deadlock by changing collective order” requirement, and you’ve preserved it.

* **Mesh leader loop is the right shape**:

  * Receives PP envelope → maps to `call_params` → TP broadcast → runs `tp_worker_infer()` → sends `PPResultV1`.
  * This is lifted straight from `scripts/pp1_pilot.py`, so it’s anchored in working evidence (PP Run 19/24/25).

* **Recompute semantics are now PP-safe**:

  * `RecomputeKVCacheBlock` refuses to fall back to `get_context_frames()` under PP (`SCOPE_PP_ENABLED=1`), forcing `context_frames_override` to be provided by rank0. That’s a hard “no silent divergence” guard.
  * Also, it honors an explicit “don’t recompute” decision (if `do_kv_recompute is False`) before env-var logic.

### What’s still missing / inconsistent with “production server”

These are the gaps that will bite you first:

1. **FrameProcessor is not PP-aware yet**
   Right now it still treats “rank0” as “TP leader + local pipeline runner”. Under PP1, rank0 has **no pipeline**, so:

   * `pipeline_manager.locked_pipeline()` will fail.
   * all `locked_pipeline.state` reads are wrong.
   * the inference path must become “PPEnvelope send/recv”, not `broadcast_infer()`.

2. **Clean shutdown is currently a deadlock trap**

   * `pp1_mesh_leader.py` ends with `barrier()` then `destroy_distributed_if_needed()`.
   * `tp_worker.py` does **not** barrier on exit; rank0 lifespan also does **not** barrier on exit.
   * Result: mesh leader can hang forever at shutdown waiting for a barrier nobody will hit.

   This is a design-level mismatch, not an implementation nit.

3. **TP plan enforcement likely breaks PP1 server unless you set env**

   * Mesh leader sends payload `tp_plan="v1_generator_only"` (in `_pp1_unpack_to_call_params`).
   * TP workers enforce `plan == env SCOPE_TP_PLAN`… but they default `env_plan` to `"v0_full_pipeline"` even if you didn’t set anything.
     That means: unless you run with `SCOPE_TP_PLAN=v1_generator_only`, workers can throw “plan mismatch”.

   Pilot avoided this because it didn’t use `tp_worker.py` for mesh workers; server does.

4. **PP transport is not server-safe yet**

   * `PPControlPlane.recv_result()` blocks forever (no timeout / cancellation) and will hang the HTTP server thread if mesh dies.
   * Also: **multiple PPControlPlane instances on rank0 is unsafe** (see robustness section).

---

## 2. FrameProcessor recommendation

You want the smallest surgical change that:

* keeps **all the existing chunk-state logic** on rank0,
* but swaps out the **“execute pipeline”** step with **“execute PP call”**.

### A. Introduce a single “mode switch” at the top of `process_chunk()`

Today you compute:

* `_tp_runtime_enabled = runtime.initialized and runtime.rank == 0` (≈line 3168 slice)

That’s wrong under PP1 because rank0 is still rank0. Replace the mental model with:

* `_pp_runtime_enabled = runtime.initialized and runtime.pp_enabled and runtime.rank == 0`
* `_tp_runtime_enabled = runtime.initialized and runtime.is_mesh_member and runtime.mesh_rank == 0 and not runtime.pp_enabled`

  * i.e. “TP leader exists only in pure-TP topology”

Then enforce:

* If `_pp_runtime_enabled`: **never** touch `locked_pipeline`.
* If `_tp_runtime_enabled`: current behavior.

### B. Create an explicit “call_params → PPEnvelopeV1” boundary

You already have the opposite mapping in two places:

* `scripts/pp1_pilot.py:_pp1_unpack_to_call_params`
* `server/pp1_mesh_leader.py:_pp1_unpack_to_call_params`

For maintainability, you want **one canonical mapping module** (even if you keep two wrappers). Otherwise you will drift.

**Design rule**: treat PPEnvelope as a *stage boundary contract*, not “just another payload”:

* rank0 must preflight everything (shape, dtype, required tensors) before sending.
* leader must reject unknown versions immediately.

### C. Bringup-safe PP1 behavior (Phase G0)

In `frame_processor.py` inference section (≈4200–4425 slice), replace:

* `call_params = control.broadcast_infer(...)`
* `output = locked_pipeline(**call_params)`
* state reads

with:

* build an envelope with:

  * `conditioning_embeds`: placeholder zero tensor (as in pilot)
  * `latents_in`: random noise tensor with correct shape/dtype
  * `current_denoising_step_list`: from existing denoise list logic (or a fixed list)
  * `context_frames`: **synthetic** from previous `latents_out` when recompute is scheduled (pilot’s `_build_synthetic_context_frames`)
  * `pp_disable_vace=True`
* send via PPControlPlane
* recv result and store `latents_out` (don’t decode yet)
* do not read `locked_pipeline.state` (return safe defaults)

This matches your proposal’s “bringup: synthetic context frames, skip WebRTC decode”.

### D. Minimal usable server (Phase G1)

Once the PP path is wired, the next “minimum viable correctness” for a real server is:

* **rank0 decodes** `latents_out` → frames (WebRTC needs pixels)
* prompt updates work

This implies rank0 must own at least:

* VAE decode (mandatory)
* a way to build conditioning embeds from prompts (either rank0 text encoder, or push prompts to mesh and compute there)

Given your current PPEnvelopeV1 requires `conditioning_embeds`, the clean plan is:

* rank0 loads text encoder + embedding blender (or equivalent) and builds embeds locally.

If you don’t want that VRAM hit on rank0, then you should admit PPEnvelopeV1 is the wrong boundary for Phase G1 and pick Option 2 (send call_params to leader) sooner.

### E. Hard gates you should add in FrameProcessor under PP1 (Phase G0/G1)

Until G2, the PP contract does not cover VACE/V2V. So in PP1 mode:

* If `call_params` contains any of:

  * `video`
  * `vace_input_frames`, `vace_input_masks`, `vace_ref_images`
  * `transition`
* then do one of:

  * **reject** the update (clear error to user)
  * or **force-disable** by stripping keys and logging

Don’t “partially support” it. That’s how you get silent wrong output.

---

## 3. Startup/load protocol

### What you should treat as the protocol (so you don’t regress it later)

#### Rank0 (HTTP server)

1. `init_distributed_if_needed()`
2. **skip pipeline load** if PP1 + non-mesh member
3. participate in `all_reduce(load_ok)` (default group)
4. participate in `barrier()` (default group)
5. start FastAPI/uvicorn
6. inference requests drive PP envelopes

This is already what `lifespan()` is doing.

#### Mesh ranks (rank1..N)

1. `init_distributed_if_needed()`
2. load pipeline
3. participate in `all_reduce(load_ok)` + `barrier()`
4. leader enters PP loop and relays into TP; workers enter TP recv loop

### Two protocol tweaks I recommend

1. **Make TP plan explicit for PP1**

* Either:

  * run PP1 server with `SCOPE_TP_PLAN=v1_generator_only` on all ranks, OR
  * change worker “plan mismatch” enforcement to only enforce when env var is explicitly set (not when it fell back to a default).

Right now, you’re relying on out-of-band convention. That’s fragile.

2. **Decide where warmup lives**

* `krea_realtime_video/pipeline.py` skips warmup whenever `tp_degree>0 or pp_enabled`, so **someone else must warm up**.
* In pure TP you do `_maybe_tp_lockstep_warmup()` on rank0.
* In PP1, rank0 can’t, so the warmup must move to:

  * **mesh leader**, after the startup barrier, before serving real requests.

If you don’t do this, your first user-visible inference will look like PP Run 24 chunk0/1 (multi-second) and then suddenly settle (good for bringup, bad for a server).

---

## 4. Rank0 stage0/VAE recommendation

There’s a misconception embedded in “rank0 has no pipeline”:

* Correct statement: rank0 has **no generator**.
* Production statement: rank0 still needs a **stage0 pipeline** (at least VAE decode, likely also text encoder).

### Recommended staged ownership

**G0 bringup**

* rank0: PP controller only (no VAE), accept “blank output” or no WebRTC frames
* mesh: generator-only via `tp_worker_infer`
* context frames: synthetic from prev latents_out (pilot pattern)

**G1 usable streaming**

* rank0 owns:

  * **VAE decode** for WebRTC frames
  * **text encoder** for prompt-to-conditioning (since PPEnvelopeV1 requires embeds)
* mesh owns:

  * generator-only
* context frames: still synthetic (avoid re-encode complexity)

**G2 feature parity**

* rank0 additionally owns:

  * decode → re-encode path for “real” KV recompute context frames (to match `get_context_frames()` semantics)
  * any transition/conditioning state that currently depends on `locked_pipeline.state`
* mesh leader returns additional metadata in `PPResultV1` (see below) if you still need post-call state visibility.

### GPU budgeting reality

With torchrun’s “one process ↔ one GPU” default, PP1 effectively consumes:

* 1 GPU for rank0 stage0 work
* (N−1) GPUs for TP mesh

That is a deliberate trade: you’re buying “server owns stage0 + low-latency I/O” at the cost of one GPU from generator sharding. If you want “all GPUs for generator + stage0 too”, you need a different placement story (multi-process per GPU, or separate node), which your current runtime does not support.

---

## 5. Robustness model

Here’s the failure surface you should design against.

### A. PP send atomicity and concurrency (this is the big one)

`PPControlPlane.send_infer()` sends:

1. header tensor
2. bytes meta payload
3. one `dist.send()` per tensor spec

That is a stream protocol. If **two threads** send interleaved messages, the receiver will desynchronize and likely hang.

Right now you can accidentally do that because:

* `PPControlPlane` has a per-instance `_send_lock`
* but rank0 code already creates **separate instances** in different places (e.g. lifespan shutdown constructs a new PPControlPlane)

**Design requirement**: rank0 must have exactly **one PPControlPlane instance** (or one global lock) used by:

* inference path (send_infer)
* keepalive path (send_noop)
* shutdown path (send_shutdown / try_shutdown)

Otherwise you have a “rare, catastrophic protocol corruption” class of bugs.

### B. recv_result must not block the HTTP server forever

`PPControlPlane.recv_result()` is currently an unbounded blocking receive. That’s okay for a pilot, not for a server.

Your stated plan (and I agree):

* Bringup: rely on NCCL timeout (`SCOPE_DIST_TIMEOUT_S`) and accept crash
* Production: implement `recv_result(timeout_s=...)` and return `PPResultV1(ok=False, error_message="timeout")`

But be clear about semantics:

* if rank0 times out, the mesh may still be computing; you need a policy for the next request:

  * either “send SHUTDOWN, reset session”
  * or “drop outstanding work, resync with cache reset”

### C. Keepalive: don’t let TP workers watchdog-exit during idle

Under PP1:

* rank0 heartbeat is disabled (good, because rank0 can’t broadcast TP)
* mesh leader currently ignores `PPAction.NOOP`
* mesh workers can have `SCOPE_TP_WORKER_WATCHDOG_S>0` and will `os._exit(2)` if they don’t see TP headers

Therefore you need one of:

* **Option A (recommended)**: rank0 sends periodic `PPAction.NOOP`; mesh leader translates that into `tp.broadcast_noop()`.
  This keeps leader single-threaded (no need for leader-side polling timers while blocked in recv).

* Option B: leader runs its own periodic `broadcast_noop()` in a background thread and/or uses non-blocking PP receive. More moving parts.

For bringup you can also set watchdog to 0, but that’s not a solution.

### D. Shutdown protocol must not block forever

As noted earlier, the mesh leader’s final `barrier()` is mismatched with other ranks. Pick a shutdown philosophy:

* **Server-first**: rank0 should always be able to exit promptly even if mesh is wedged.
  → Avoid barriers on shutdown; best-effort shutdown + timeouts + process exit.

That’s what most “production servers” do. Barriers are for test harnesses.

### E. Post-call pipeline state dependency

FrameProcessor currently reads `locked_pipeline.state` after call to drive features (transitions, conditioning embeds persistence). Under PP1, the call happened on mesh.

Bringup fix: gate all those reads off under PP1 (return defaults).
Production fix: extend `PPResultV1` to return whichever post-call state rank0 truly needs.

Be disciplined here: if rank0 needs only 2 booleans and one tensor, add exactly those fields. Don’t turn PPResult into a dump of pipeline state.

---

## 6. Test plan

Anchor to your proven evidence (PP Run 19/24/25, OM Run 22) and add server-specific tests in increasing realism.

### Phase T0: regression guardrails (cheap, fast)

* Run existing Operator Matrix tests you already have (OM Run 22 list).
* Add one new OM-style test specifically for **PPControlPlane send atomicity**:

  * create two PPControlPlane instances on rank0 and attempt concurrent sends
  * expected: test should fail fast (or prevent it by design)

### Phase T1: PP1 server boot + single chunk (no WebRTC)

Command shape (conceptually):

* `SCOPE_PP_ENABLED=1`
* `SCOPE_TENSOR_PARALLEL=2` → `WORLD_SIZE=3`
* `PIPELINE=krea-realtime-video`
* **explicit**: `SCOPE_TP_PLAN=v1_generator_only` (until you fix plan enforcement)
* `torchrun --nproc_per_node=3 -m scope.server.app`

Assertions:

* rank0 serves `/health`
* mesh ranks load pipeline, barrier completes (you should see the startup logs)
* a synthetic “one chunk” request produces:

  * mesh leader logs `PPAction.INFER`
  * rank0 receives `PPResultV1(ok=True, latents_out != None)`

### Phase T2: recompute correctness (server path)

Mimic PP Run 24/25 logic inside server requests:

* schedule `do_kv_recompute` True/False across chunks
* verify:

  * recompute chunks show stage1_ms jump similar to Run 24 (~+230ms over baseline)
  * non-recompute chunks match baseline

Even if you don’t wire observed_generator_calls yet, stage time deltas should line up with the run-log evidence.

### Phase T3: idle robustness (watchdog + keepalive)

* Set `SCOPE_TP_WORKER_WATCHDOG_S=5`
* Let server idle for > 15s
* Assert workers do not exit
* This forces you to implement NOOP relay or disable watchdog under PP1.

### Phase T4: WebRTC decode (rank0 VAE-only)

* Load VAE-only on rank0 and decode latents_out to frames
* Verify WebRTC stream shows non-empty frames

### Phase T5: failure injection

* Kill mesh worker process mid-run
* rank0 should:

  * not hang forever in recv_result
  * surface an error and recover/reset

---

## 7. Red flags

These are the “don’t proceed until addressed” items:

1. **Shutdown barrier mismatch**
   `pp1_mesh_leader.py` barriers on exit, other ranks don’t. This will hang on shutdown.

2. **PPControlPlane instance multiplicity on rank0**
   Separate instances can interleave sends and corrupt the PP stream protocol. Catastrophic class.

3. **TP plan mismatch in PP1 server**
   Mesh leader sends `tp_plan="v1_generator_only"`, TP workers enforce env default `"v0_full_pipeline"`. This likely crashes unless you set env carefully.

4. **recv_result can hang the HTTP server forever**
   Needs timeout semantics.

5. **No keepalive path under PP1**
   If watchdog is enabled, workers will exit during idle. If watchdog is disabled, you still want liveness signals for operational sanity.

6. **observed_generator_calls is currently always 0 in server PPResult**
   You lose the strongest correctness tripwire you had in pilot runs (call-count parity). That’s how recompute bugs sneak in.

---

## 8. Top-3 implementation priorities

1. **FrameProcessor PP path (bringup-safe)**

   * Add `_pp_runtime_enabled` gating.
   * Replace rank0 local pipeline call with PPEnvelope send/recv.
   * Gate out all `locked_pipeline.state` reads under PP1.

2. **Transport safety: single PPControlPlane + recv_result timeout**

   * Enforce single-sender semantics on rank0 (singleton PP control plane shared by inference/keepalive/shutdown).
   * Add result timeout behavior so HTTP can’t wedge forever.

3. **Liveness + shutdown semantics**

   * Fix the shutdown deadlock (remove/align barrier on exit).
   * Add PP NOOP relay → TP broadcast_noop (or disable worker watchdog under PP1 until keepalive exists).

---

Unrelated but I did see your IMS exploration notes in the workspace; not needed for this PP1 review.
