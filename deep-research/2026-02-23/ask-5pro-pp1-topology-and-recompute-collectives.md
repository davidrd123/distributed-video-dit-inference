# 5 Pro Deep Research Request — PP1 process group topology, recompute-collective coupling, VAE thread safety

Date: 2026-02-23
Status: Ready to run (copy/paste into repo prompt)

## Objective

We've completed **PP0 Steps A3** (overlap with bounded queues) and are mid-flight on **A4a** (recompute scheduling validation). Next is **A4b** (correct recompute with real VAE decode on rank0) and then **A5** (PP1 — rank0 out-of-mesh with a TP mesh on ranks 1..N).

A4a's questions are mostly empirical (run it, check counts). But A4b and A5 have **systems-level questions about NCCL multi-communicator concurrency, process group topology, and collective-level failure modes** that are hard to test incrementally and dangerous to get wrong.

Goal output: a **topology and safety audit** covering PP1 process group creation, concurrent communicator usage, recompute-triggered collective coupling across the PP boundary, and VAE thread safety for A4b. Concrete yes/no answers, code patterns, and deadlock scenarios.

## Repo prompt pack (include these files)

### A4/A5 design (what we're planning)

- `scope-drd/notes/FA4/h200/tp/pp0-a4-recompute-codex-instructions.md` (A4 spec — covers the recompute data dependency, synthetic vs real context_frames, prediction logic)
- `scope-drd/notes/FA4/h200/tp/pp0-a3-overlap-codex-instructions.md` (A3 spec — the overlap/threading foundation A4/A5 build on)
- `scope-drd/notes/FA4/h200/tp/pp0-a3-overlap-acceptance.md` (acceptance criteria O-01, O-02, O-03)
- `scope-drd/notes/FA4/h200/tp/pp-next-steps.md` (A4/A5 roadmap context — step sequencing, acceptance gates)
- `scope-drd/notes/FA4/h200/tp/explainers/09-pp-topology.md` (PP0→PP1 mental model — Stage 0 vs Stage 1 roles, contract boundary)
- `scope-drd/notes/FA4/h200/tp/explainers/10-pp0-a3-overlap-threading.md` (comm_stream event DAG, threading invariants, NCCL thread safety rule, bounded queue deadlock avoidance)

### Current implementation

- `scope-drd/scripts/pp0_pilot.py` (PP0 pilot with overlap, comms thread, D sweep — the A3 implementation)
- `scope-drd/src/scope/core/distributed/pp_control.py` (PPControlPlane — blocking p2p send/recv on world_pg)
- `scope-drd/src/scope/core/distributed/pp_contract.py` (PPEnvelopeV1, PPResultV1 — envelope carries context_frames)

### Pipeline code (recompute + VAE)

- `scope-drd/src/scope/core/pipelines/krea_realtime_video/blocks/recompute_kv_cache.py` (the recompute block — reads env var, decides skip logic, has override path + VAE fallback)
- `scope-drd/src/scope/core/pipelines/krea_realtime_video/blocks/prepare_context_frames.py` (buffer maintenance — context_frame_buffer, decoded_frame_buffer)
- `scope-drd/src/scope/core/pipelines/krea_realtime_video/pipeline.py` (tp_worker_infer, _generate, block execution — kwargs → state → blocks)
- `scope-drd/src/scope/core/pipelines/wan2_1/vae/wan.py` (WanVAEWrapper — encode_to_latent, decode_to_pixel, cache semantics)
- `scope-drd/src/scope/core/pipelines/wan2_1/vae/modules/vae.py` (WanVAE_ internals — stream_encode/stream_decode caches, first_batch flag, feat_cache)

### TP control plane (for PP1 mesh context)

- `scope-drd/src/scope/core/distributed/control.py` (TPControlPlane — broadcast protocol, _broadcast_lock, mesh collectives)
- `scope-drd/src/scope/core/distributed/runtime.py` (DistributedRuntime — rank, world_size, device assignment)

### Library resources (physics)

- `refs/resources/nccl-user-guide.md` (multi-communicator concurrency, thread safety)
- `refs/resources/pytorch-cuda-semantics.md` (allocator, streams, reference counting)
- `refs/topics/05-cuda-streams.md` (stream ordering, record_stream, multi-stream safety)
- `refs/topics/02-deadlock-patterns.md` (FM taxonomy — especially FM-01 membership, FM-03 conditional collectives)
- `refs/topics/03-graceful-shutdown.md` (drain vs abort, watchdog, terminal broadcast)
- `refs/topics/19-producer-consumer-backpressure.md` (bounded queues, overlap proof)
- `refs/topics/15-pipeline-scheduling-theory.md` (bubble fraction, PP+TP interaction)

### Operator test matrix (acceptance tests that constrain PP1)

- `refs/operator-test-matrix.md` (OM-07 leader safety, OM-08 wrong-group collective, OM-13 orphan shutdown — these are the PP1-specific tests)

### Prior 5 Pro history (calibration)

- `scope-drd/notes/FA4/h200/tp/5pro/13-pp-execution-ready/response.md` (PP readiness review — overlap metrics, shutdown, process group sketch)
- `scope-drd/notes/FA4/h200/tp/5pro/10-v11-correctness-deadlock-audit/response.md` (deadlock taxonomy — FM patterns)
- `scope-drd/deep-research/2026-02-22/pp-rank0-out-of-mesh/reply.md` (prior deep research on PP rank0-out-of-mesh topology)

## Background: the PP topology

**PP0 (current, 2 ranks):**
- rank0 = Stage 0 (control plane, envelope builder, decode/output). No model weights, no pipeline.
- rank1 = Stage 1 (full pipeline: generator + VAE + text encoder). Runs `tp_worker_infer()`.
- Communication: `dist.send`/`dist.recv` on default process group (world_pg). No TP collectives.

**PP1 (future, 3+ ranks):**
- rank0 = Stage 0 (control plane + VAE decode/encode for A4b). Still out-of-mesh.
- ranks 1..N = Stage 1 (TP mesh). Generator is sharded via TP across mesh ranks. Mesh ranks do NOT have VAE (memory savings).
- Communication:
  - rank0 ↔ mesh leader (rank1): `dist.send`/`dist.recv` on world_pg (PP p2p)
  - mesh leader → mesh workers: `dist.broadcast` on mesh_pg (TP collectives)
  - within generator forward: TP all-reduce on mesh_pg
- `mesh_pg` = `torch.distributed.new_group(ranks=[1, ..., N])`

**The overlap threading model (from A3):**
- rank0 main thread: builds envelopes, decodes results, computes metrics
- rank0 comms thread: blocking `send_infer()` / `recv_result()` on world_pg
- rank0 main thread calls `send_shutdown()` only AFTER `comms_thread.join()`
- Under A4b: rank0 main thread also runs VAE decode on GPU (concurrent with comms thread doing NCCL on world_pg)

## Questions to answer

### Q1) VAE thread safety on rank0 under A4b

Under A4b, rank0's main thread runs VAE decode (`components.vae.decode_to_pixel(latents_out)`) while the comms thread does `dist.send`/`dist.recv` on the same GPU. The VAE decode uses GPU compute on the default CUDA stream; NCCL send/recv uses the NCCL internal stream.

Questions:
- Is a `torch.nn.Module.forward()` call (specifically `Decoder3d`, a Conv3d-heavy network) safe to run on the default stream while NCCL operations run on the NCCL stream concurrently? Or does PyTorch/CUDA serialize them?
- The VAE `encode_to_latent()` and `decode_to_pixel()` methods may use internal state (e.g., cached padding, buffers). Are these safe for concurrent use from the main thread while the comms thread holds references to previously-encoded tensors?
- If the comms thread runs on a separate `torch.cuda.Stream` (as our A3 implementation does — see `comm_stream` in pp0_pilot.py), does this provide sufficient isolation from the main thread's default-stream VAE operations?
- For correctness with the CUDA caching allocator: if rank0 receives `latents_out` on a non-default stream, is `stream.wait_event(...)` sufficient before VAE decode, or do we also need `latents_out.record_stream(torch.cuda.current_stream())` (or similar) to prevent use-after-free / premature reuse?
- Is there any risk of CUDA allocator contention between the main thread (allocating for VAE decode) and the comms thread (allocating receive buffers for `dist.recv`)?
- What about `torch.no_grad()` context — the recompute block uses it (line 165 of recompute_kv_cache.py). If the main thread is in `torch.no_grad()` for VAE decode, does that affect the comms thread?

Deliverable: **Safety verdict** — can rank0 safely run VAE decode concurrently with NCCL p2p send/recv? If yes, what precautions are needed? If no, what's the alternative (serialize, use separate streams with explicit sync, etc.)?

### Q2) Process group creation for PP1 mesh_pg

Under PP1, we need a process group containing only the mesh ranks (1..N) for TP collectives. Rank0 is excluded from this group.

Questions:
- `torch.distributed.new_group(ranks=[1, 2, ..., N])` — must rank0 also call `new_group` even though it's not in the group? (PyTorch docs say `new_group` is collective across all ranks in the default group.) What happens if rank0 skips the call?
- Can `new_group` be called at different points in rank0's execution vs mesh ranks' execution? Or must all ranks call it at the same "logical point"?
- After `mesh_pg` is created, can rank0 continue using `dist.send`/`dist.recv` on the default group while mesh ranks use `mesh_pg` for TP collectives? Are these on separate NCCL communicators?
- Should PP p2p use a dedicated process group (e.g., `pp_pg = new_group(ranks=[0, leader_rank])`) instead of `world_pg` to reduce communicator overlap/pressure and make participation explicit? If yes: ordering constraints vs `mesh_pg`, and does it materially change concurrency behavior?
- Is there a performance cost to having rank0 participate in `new_group` creation for a group it doesn't belong to?
- How does `destroy_process_group(mesh_pg)` work? Must rank0 also call destroy, or just the member ranks?
- Can we create `mesh_pg` once at init and reuse it for the entire session? Or does it need to be recreated if ranks join/leave (not applicable for us, but good to know)?

Deliverable: **Process group creation recipe** — exact call sequence for rank0 and mesh ranks, with ordering constraints and cleanup.

### Q3) Concurrent NCCL communicators: PP p2p + TP collectives

This is the core PP1 concurrency question. During steady-state operation:

- rank0's comms thread does `dist.send(tensor, dst=1)` on world_pg (PP envelope to mesh leader)
- simultaneously, ranks 1..N are executing a generator forward pass that includes TP all-reduce on mesh_pg

Questions:
- Can NCCL operations on different communicators (world_pg vs mesh_pg) run concurrently on the same GPU? Or does NCCL serialize across communicators?
- If they can run concurrently: do they use separate NCCL streams? Can this cause resource contention (GPU SM occupancy, NVLink bandwidth, PCIe bandwidth)?
- If they serialize: what's the ordering? Does a `dist.send` on world_pg block until an in-flight all-reduce on mesh_pg completes?
- Does the answer change depending on whether `NCCL_COMM_SPLIT_SHARE_RESOURCES` is set?
- What about the converse: can rank1 (mesh leader) be receiving a PP envelope from rank0 (`dist.recv` on world_pg) while simultaneously participating in a TP all-reduce on mesh_pg? This is the mesh leader's dual role — it participates in both PP and TP communication.
- Is it safe in PyTorch/NCCL to have multiple threads in one process concurrently call `torch.distributed` ops on different process groups/communicators (e.g., one thread in PP p2p recv/send while another thread is inside TP collectives)? If not, what’s the recommended architecture (single-thread phased comms, a global dist lock, explicit stream/handle discipline, etc.)?
- In Megatron-LM and DeepSpeed, how is this handled? Do they use separate NCCL communicators for PP and TP? Do they ever run PP and TP communication concurrently, or do they carefully phase them?
- For crash>hang bringup: which NCCL / torch.distributed settings are recommended specifically for multi-communicator setups (async error handling, blocking wait, timeouts, debug knobs) so an orphan rank doesn’t sit for minutes?

Deliverable: **Concurrency matrix** — for each pair of (PP operation on world_pg) × (TP operation on mesh_pg), whether they can run concurrently, and any constraints. Include Megatron/DeepSpeed patterns.

### Q4) Recompute-collective coupling across the PP boundary (deadlock analysis)

This is the scariest question. Under PP1 with recompute enabled:

1. Rank0 decides whether chunk k needs recompute (via `_predict_do_kv_recompute`)
2. Rank0 sets `do_kv_recompute` and `expected_generator_calls` in the envelope
3. Mesh leader receives envelope, broadcasts to mesh workers via mesh_pg
4. All mesh ranks run `RecomputeKVCacheBlock`, which reads `SCOPE_KV_CACHE_RECOMPUTE_EVERY` from os.getenv and **independently decides** whether to recompute
5. If the block decides to recompute, it runs a generator forward pass → which triggers TP all-reduce on mesh_pg (all mesh ranks must participate)
6. Then the denoise pass runs → more TP all-reduces

**The deadlock scenario:** If rank0's prediction says "skip recompute" (`expected_generator_calls = 1`) but the block on mesh says "do recompute" (block runs generator twice), then `expected_generator_calls` mismatches → assertion fires. This is the benign case.

**The worse scenario:** What if the prediction matches on some mesh ranks but not others? Under torchrun, all ranks share the same env vars, so `SCOPE_KV_CACHE_RECOMPUTE_EVERY` should be identical. But `current_start_frame` is derived from the envelope, so it should be consistent too. **Is there any way mesh ranks could disagree on whether to recompute?** If one rank recomputes (enters TP all-reduce) and another doesn't (skips to denoise), the TP all-reduce hangs.

Questions:
- Under torchrun, are environment variables guaranteed identical across all ranks? (Process-level, not thread-level.)
- Is `current_start_frame` guaranteed to be identical across all mesh ranks? (It comes from the envelope, which is broadcast from leader → workers.)
- Are there any other inputs to the block's skip logic that could differ across ranks? (`components.config.num_frame_per_block`, `components.config.kv_cache_num_frames` — are these guaranteed identical after TP-sharded loading?)
- In production frameworks, how is the "conditional compute" problem handled? (e.g., Megatron's activation recomputation — does it ever conditionally skip recompute based on runtime state?)
- Should we eliminate conditional recompute decision-making inside the mesh entirely under PP1 by making recompute an explicit per-chunk plan bit (leader broadcasts `do_kv_recompute`, block obeys it), rather than having each mesh rank consult `os.getenv("SCOPE_KV_CACHE_RECOMPUTE_EVERY")`?
- If we keep env-var-driven skipping: what is the best defense to prevent “some ranks skip, others recompute” (hard hang)? E.g., broadcast the computed decision and assert equality on every rank before entering any TP collective.
- Related (OM-08): what’s the recommended way to enforce “all TP collectives use `mesh_pg`” and fail fast before NCCL if any codepath accidentally uses the default/world group?
- What's the recommended defense? Options:
  - (a) Leader broadcasts the recompute decision (not just the envelope) so all mesh ranks use the same decision
  - (b) Ban conditional recompute under PP1 (always recompute or never recompute)
  - (c) Add a parity check: mesh leader computes the prediction, compares with envelope's `do_kv_recompute`, crashes if mismatch

Deliverable: **Deadlock risk assessment** — enumerate all ways mesh ranks could disagree on recompute gating, rate the likelihood, and recommend a defense.

### Q5) Shutdown and error handling during active mesh collectives

Under PP1, shutdown is more complex than PP0 because the mesh has internal collectives:

**Scenario A:** Rank0 sends SHUTDOWN. Mesh leader receives it. But mesh workers are mid-generator-forward, executing TP all-reduces on mesh_pg. The leader can't just exit — the workers are blocked in collectives expecting the leader to participate.

**Scenario B:** Rank0 sends SHUTDOWN while the mesh is between chunks (idle). Leader broadcasts SHUTDOWN to workers. Workers exit cleanly. This is the easy case.

**Scenario C:** Mesh leader encounters an error validating the envelope (OM-07 from operator test matrix). It must NOT start the generator forward (which would enter TP collectives). Instead, it must broadcast a terminal action (SHUTDOWN or ERROR) to workers so they exit cleanly.

Questions:
- In scenario A: how does the leader safely propagate SHUTDOWN to workers who are mid-collective? Must it wait for the current generator forward to complete before processing SHUTDOWN? Or can it queue SHUTDOWN and process it between generator calls?
- If the leader waits for the current forward to complete: the mesh processes one more chunk's worth of TP collectives. Is there a risk that rank0 has already moved on (sent the next envelope, or exited) and the leader's `recv_next()` call hangs?
- For terminal broadcast (scenario C): what action code should the leader broadcast? Our current `PPAction` enum has `SHUTDOWN`, `INFER`, `NOOP`. Should we add `ERROR`? What should workers do when they receive `ERROR` — exit immediately, or clean up first?
- In Megatron-LM: how does the PP stage boundary interact with TP collective shutdown? Does Megatron have a "flush the current micro-batch" protocol?
- What's the watchdog situation? If rank0 dies while the mesh is mid-forward, the leader eventually times out on `recv_next()`. But workers are in TP collectives with the leader — the leader is participating in those collectives (it's mid-forward too). How does the NCCL timeout propagate through the mesh?

Deliverable: **Shutdown protocol sketch** for PP1 — covering normal SHUTDOWN, error during validation, and rank0 crash. Include the leader's state machine (receiving from rank0 vs participating in mesh collectives vs broadcasting to workers).

### Q6) Leader validate-before-broadcast under PP1

Our operator test matrix (OM-07) requires: "Leader does NOT start payload broadcast [to workers] if envelope validation fails; broadcasts terminal action (SHUTDOWN/ERROR) so non-leaders exit cleanly."

Under PP1, the leader receives an envelope from rank0 via p2p recv, validates it, then broadcasts to mesh workers via mesh_pg. The broadcast is a collective — all mesh workers are blocked waiting for it.

Questions:
- What should the broadcast payload look like? Options:
  - (a) Broadcast the raw action code first (INFER/SHUTDOWN/ERROR), then broadcast the full envelope only for INFER. Workers check the action code and exit on non-INFER.
  - (b) Broadcast the full envelope always, with an error flag. Workers check the flag.
  - (c) Use a separate "control" broadcast (1 int64) before the data broadcast.
- If the leader broadcasts ERROR: what information should it include? Just the action code, or also the error message?
- Is there a standard pattern in Megatron/DeepSpeed for this "leader validates, then broadcasts or aborts" protocol?
- If the leader broadcasts ERROR and workers exit, but rank0 doesn't know about the error (it's still sending envelopes): rank0's `dist.send` to the leader will block because the leader is no longer calling `dist.recv`. This is the "orphan rank0" problem. **Recommend a concrete unstick mechanism:** ACK/NACK channel (leader sends error status back to rank0 before exiting), heartbeat protocol, or timeout policy with specific values. Which approach gives the best crash>hang behavior without adding protocol complexity? If timeout: what value, and how does rank0 distinguish "leader is slow" from "leader exited"?

Deliverable: **Leader protocol specification** — exact sequence of broadcasts for INFER (success), SHUTDOWN (clean exit), and ERROR (validation failure). Include how rank0 learns about mesh-side errors, with a concrete "rank0 unstick" mechanism recommendation.

## Output format

Return:
- VAE thread safety verdict (Q1)
- Process group creation recipe (Q2)
- Concurrent communicator matrix (Q3)
- Recompute-collective deadlock assessment (Q4)
- PP1 shutdown protocol sketch (Q5)
- Leader validate-before-broadcast protocol (Q6)
- Any "red flags" — things in our current PP0 implementation or A4/A5 plans that will break under PP1
- **Priority ranking** — if we can only address 3 of these before starting A5 implementation, which 3 matter most?

Make the output actionable: if something needs to change in existing code or planned designs, say exactly what and where.
