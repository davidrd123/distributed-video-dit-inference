# 5 Pro Deep Research Request — PP1 server integration design review

Date: 2026-02-23
Status: Ready to run (copy/paste into repo prompt)

## Objective

We've completed PP1 protocol bringup (R1, R0a, R0b all PASS on 4×A100 via
`scripts/pp1_pilot.py`). The next milestone is **server integration** — making
PP1 work as a production HTTP/WebRTC server instead of a standalone pilot
script.

We have a written proposal (`scope-drd/notes/FA4/h200/tp/proposals/pp1-server-integration.md`)
covering the 6 server-side locations that need changes. The proposal identifies
`frame_processor.py` as the main design surface.

Goal output: a **design review** of the proposal, with concrete recommendations
for the open questions, risk assessment for the frame_processor change, and
any blind spots we're missing.

## Repo prompt pack (include these files)

### The proposal (primary input)

- `scope-drd/notes/FA4/h200/tp/proposals/pp1-server-integration.md` (the full proposal — read this first)

### Server code being changed

- `scope-drd/src/scope/server/app.py` (main entrypoint — startup dispatch, lifespan, pipeline load, heartbeat, shutdown)
- `scope-drd/src/scope/server/tp_worker.py` (TP worker loop — loads pipeline, recv_next, run pipeline)
- `scope-drd/src/scope/server/frame_processor.py` (the big one — `_generate()` drives inference, TP broadcasts, KV cache state, recompute scheduling)

### PP1 pilot (proven reference implementation)

- `scope-drd/scripts/pp1_pilot.py` (3-rank PP1 harness — rank0 loop, mesh leader loop, mesh worker loop. R1/R0a/R0b all PASS.)

### Distributed runtime (already landed)

- `scope-drd/src/scope/core/distributed/runtime.py` (DistributedRuntime with mesh_rank, mesh_size, is_mesh_member, mesh_global_ranks, get_mesh_pg())
- `scope-drd/src/scope/core/distributed/control.py` (TPControlPlane — already parameterized with mesh_pg)
- `scope-drd/src/scope/core/distributed/pp_control.py` (PPControlPlane — blocking p2p send/recv)
- `scope-drd/src/scope/core/distributed/pp_contract.py` (PPEnvelopeV1, PPResultV1)

### Pipeline code (for frame_processor context)

- `scope-drd/src/scope/core/pipelines/krea_realtime_video/pipeline.py` (tp_worker_infer, _generate)
- `scope-drd/src/scope/core/pipelines/krea_realtime_video/blocks/recompute_kv_cache.py` (recompute block — reads do_kv_recompute from call_params)

### Run log (timing data for Q4/Q5 analysis)

- `scope-drd/notes/FA4/h200/tp/bringup-run-log.md` (Runs 19-25 — exact timing data for R1/R0a/R0b, plus TP=1/2/4 baselines)

### Launch scripts (for Q6 testing context)

- `scope-drd/scripts/run_daydream_a100_tp.sh` (current TP launch script — no PP1 support yet, shows what env vars need extending)

### Prior 5Pro context

- `scope-drd/deep-research/2026-02-23/pp1-topology-and-recompute/reply.md` (Q1-Q6 answers — VAE safety, new_group, concurrency, recompute coupling, shutdown, leader protocol)

## Background: where we are

**What's PASSED (all on 4×A100-SXM4-80GB):**

| Run | What | Result |
|-----|------|--------|
| 19 | PP1 R1 happy-path (no recompute), 20 chunks | PASS, median 282ms |
| 20 | A100 single-GPU baseline | PASS, ~6 FPS |
| 21 | A100 TP=2 visual quality | PASS, ~8.2 FPS |
| 22 | OM tests (5/5 including om_07 leader safety, om_13 orphan shutdown) | PASS |
| 23 | A100 TP=4 visual quality (all 4 GPUs) | PASS, ~10.0 FPS |
| 24 | PP1 R0a (recompute every chunk), 10 chunks | PASS, median 514ms |
| 25 | PP1 R0b (recompute every 5), 20 chunks | PASS, skip logic correct |

**PP1 topology (proven in pilot):**
- rank0: out-of-mesh, no pipeline, builds PPEnvelopeV1, sends to rank1
- rank1 (mesh_rank=0): mesh leader — PP recv → unpack → TP broadcast → pipeline.tp_worker_infer() → PP send result
- rank2 (mesh_rank=1): mesh worker — TPControlPlane.recv_next() loop → pipeline.tp_worker_infer()

**What's landed in code:**
- `DistributedRuntime` has `pp_enabled`, `mesh_rank`, `mesh_size`, `is_mesh_member`, `mesh_global_ranks`
- `get_mesh_pg()` returns the TP mesh process group (or None for pure TP / rank0)
- All TP collectives and TPControlPlane broadcasts already use `mesh_pg`
- PP contract (PPEnvelopeV1/PPResultV1) and PP control plane are stable
- `_pp1_unpack_to_call_params()` mapping is proven in the pilot

**What's NOT landed:**
- Server code (`app.py`, `tp_worker.py`, `frame_processor.py`) has zero PP1 awareness
- No `pp1_mesh_leader.py` server module yet

## Questions to answer

### Q1) Proposal review: is the 3-role dispatch correct?

The proposal adds a branch in `app.py` main():
```python
if distributed_runtime.pp_enabled and distributed_runtime.mesh_rank == 0:
    run_pp1_mesh_leader_forever()  # new module
else:
    run_tp_worker_forever()        # existing
```

Questions:
- Is there any server-side state that the mesh leader needs from `app.py` lifespan that `run_tp_worker_forever()` currently gets from being in the same process? (e.g., pipeline_manager instance, config, signal handlers)
- The mesh leader loads its own pipeline via `_load_pipeline_if_requested()` (same as tp_worker). Is there any ordering constraint with rank0's lifespan startup?
- Under pure TP, rank0 enters the HTTP server path. Under PP1, rank0 also enters the HTTP server path (it still serves HTTP). Is there any TP-specific initialization in the HTTP path that would break when `is_mesh_member=False`? (e.g., TPControlPlane creation, broadcast calls during warmup)

### Q2) frame_processor.py: PP envelope construction

The proposal says frame_processor's `_generate()` should build PP envelopes instead of calling `tp.broadcast_infer()`. The mapping is:

```python
# Current (pure TP):
call_params = tp.broadcast_infer(call_params, chunk_index=...)
latents_out = pipeline.tp_worker_infer(**call_params)

# Proposed (PP1):
env = _build_pp_envelope_from_state(...)  # new helper
pp.send_infer(env)
result = pp.recv_result()
latents_out = result.latents_out
```

Questions:
- The frame_processor maintains significant state between chunks (KV cache state, recompute counters, context_frame_buffer, decoded_frame_buffer, transition blending, VACE conditioning). Under PP1, this state lives on rank0 (which doesn't have a pipeline). Is there any state that's only available on the pipeline side (mesh ranks) that rank0 would need to build correct envelopes?
- `PPResultV1` returns `latents_out`, `observed_generator_calls`, `mesh_current_start_frame`. Is this sufficient for rank0 to maintain its state machine, or does it need more data back from the mesh (e.g., updated KV cache metadata, block_state fields)?
- The current frame_processor calls `pipeline(**call_params)` or `pipeline.tp_worker_infer(**call_params)` and gets back output that it processes further (VAE decode, frame buffer updates, WebRTC send). Under PP1, rank0 doesn't have a pipeline. Should rank0:
  - (a) Get `latents_out` from PPResultV1 and do its own VAE decode (requires rank0 to load a VAE — like A4b's `Rank0Buffers` pattern)
  - (b) Get decoded frames directly from the mesh (PPResultV1 would carry decoded pixel output — larger tensor transfer)
  - (c) Get `latents_out` and send it to WebRTC as-is (not possible — WebRTC needs pixel frames)
- How should rank0 handle the `context_frames_override` construction? Under pure TP, the frame_processor builds it from local buffers. Under PP1, rank0 needs to build it from `latents_out` returned by the mesh (synthetic path, already proven in R0a/R0b) or from its own VAE decode (A4b-grade). The proposal doesn't address this explicitly — is the synthetic path sufficient for initial server integration?

### Q3) Lifespan pipeline load under PP1

rank0 must participate in the `all_reduce(load_ok)` collective (it's on the default group). But rank0 doesn't load a pipeline. The proposal says rank0 contributes `1` ("always ok").

Questions:
- Is there any pipeline-load side effect on rank0 that other ranks depend on? (e.g., shared filesystem state, model cache, tokenizer init)
- The `init_tp_shard_fingerprint_baseline()` and `barrier()` after load — rank0 must skip the fingerprint (mesh-only) but participate in the barrier (all ranks). Is the barrier on the default group or mesh_pg? If default group, rank0 participates. If mesh_pg, rank0 must skip. Check which it is.
- `_maybe_tp_lockstep_warmup()` runs on rank0 and broadcasts warmup calls to workers. Under PP1, rank0 can't do this. Should the mesh leader do autonomous warmup before entering the PP recv loop? If so, rank0 needs to know warmup is done before sending the first inference envelope. How is this synchronized?

### Q4) The "rank0 VAE" question

This is the elephant in the room for production PP1. Under the pilot, rank0 is
lightweight (no pipeline, no VAE). But in production:

1. rank0 receives `latents_out` from the mesh via PPResultV1
2. rank0 must decode `latents_out` to pixel frames for WebRTC output
3. rank0 must build `context_frames_override` for the next chunk's recompute

Both require VAE access on rank0. Options:

- **(a) rank0 loads only the VAE** (not the full pipeline). ~1-2GB VRAM. Mirrors the A4b `Rank0Buffers` pattern from pp0_pilot.py. rank0 does decode+re-encode locally.
- **(b) Mesh decodes and sends pixels** in PPResultV1. Avoids VAE on rank0 but increases tensor transfer size (pixel frames are much larger than latents). Also couples mesh to output resolution.
- **(c) Defer VAE to a separate process/GPU.** Over-engineered for now.

Questions:
- Is (a) the right approach for initial server integration? The pilot already proves it works (A4b on PP0). What are the VRAM and latency implications of rank0 VAE on A100?
- If (a): should rank0's VAE decode happen synchronously (blocking the PP send/recv loop) or asynchronously (overlap decode with next chunk's mesh inference)? The A3 overlap pattern from pp0_pilot.py does this with a comms thread — should the server do the same?
- Under PP1 with 4 GPUs: rank0=VAE-only, ranks 1-2=TP mesh (generator), GPU3 idle. Is there a better GPU allocation? Note: **TP=4 is now proven** (Run 23, ~10 FPS on 4×A100, visually correct). So an alternative is rank0=VAE on GPU0, ranks 1-3=TP=3 mesh. But TP=3 (non-power-of-2) may cause head-sharding issues (40 heads / 3 = not even). Assess whether TP=3 is viable or if we should stick with TP=2 + idle GPU.

### Q5) Error propagation in the server context

The pilot has simple error handling (mesh leader catches exceptions, sends
`ok=False` in PPResultV1, rank0 raises). In the server, we need:

- Mesh errors should not crash the HTTP server (rank0). Instead, rank0 should
  log the error, skip the frame, and continue accepting WebRTC traffic.
- rank0 HTTP server shutdown (Ctrl-C, SIGTERM) must propagate SHUTDOWN to
  the mesh cleanly.
- If the mesh hangs (NCCL timeout), rank0 must detect this and not block
  the HTTP server forever.

Questions:
- Should `pp.recv_result()` have a timeout on rank0? If so, what's the right
  value relative to mesh inference time (~282ms for R1, ~514ms for R0a)?
- If rank0 gets `ok=False` from PPResultV1, should it retry the chunk or
  skip? What's the right recovery behavior for a streaming server?
- How should rank0 handle the case where mesh leader dies mid-forward? The
  `pp.recv_result()` call would block forever without a timeout.

### Q6) Testing strategy

Questions:
- Can we test the server integration incrementally (e.g., test startup/shutdown
  without WebRTC, then add WebRTC)?
- What's the minimum viable test: `curl localhost:8000/health` returns 200
  with all 3 ranks running?
- Should we add a PP1-specific health endpoint that reports mesh status?
- Can we reuse the existing OM tests (om_07, om_13) for server-mode PP1,
  or do we need new server-specific operator tests?

## Output format

Return:
1. **Proposal review** — assessment of each proposed change (Q1). Confirm, revise, or reject.
2. **frame_processor recommendation** (Q2) — concrete approach for the PP envelope path, with pseudocode.
3. **Pipeline load protocol** (Q3) — exact sequence for PP1 startup across all 3 ranks.
4. **rank0 VAE recommendation** (Q4) — which option, with VRAM/latency analysis.
5. **Error handling spec** (Q5) — timeout values, recovery behavior, shutdown sequence.
6. **Test plan** (Q6) — ordered checklist from smoke test to full WebRTC validation.
7. **Red flags** — anything in the proposal or current code that will break or cause subtle bugs.
8. **Implementation ordering** — if we can only do 3 things before attempting the first server-mode PP1 run, which 3?

Make the output actionable: if something needs to change, say exactly what file, what line, and what the change should be.
