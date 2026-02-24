# 5 Pro Deep Research Request — PP1 G1c: overlap (rank0 decode ∥ mesh inference)

Date: 2026-02-24
Status: Ready to implement; requesting overlap safety audit

## Update (2026-02-24)

Overlap implementation has now been prototyped in `scope-drd` and is streaming video end-to-end, but we’d like a second pass:

- **Code review request (implementation vs plan + safety audit):** `scope-drd/notes/FA4/h200/tp/proposals/g1c-impl-plan/ask-g1c-code-review.md`
- **Companion output-path request (VAE decode + frame pacing + queue/backpressure):** `scope-drd/notes/FA4/h200/tp/proposals/g1c-impl-plan/ask-pp1-output-architecture.md`
- **SM120-specific symptom report (repeat ~3 frames + WebRTC churn):** `scope-drd/notes/FA4/h200/tp/5pro/15-pp1-sm120-webrtc-repeat3frames/ask.md`

## Update (2026-02-24 later) — temporal coherence (“slideshow”)

We identified that the “chunk restarts / slideshow” symptom is largely due to **KV-cache recompute being disabled by default under PP1**:

- Non-PP pipeline default: `RecomputeKVCacheBlock` runs every chunk (`SCOPE_KV_CACHE_RECOMPUTE_EVERY` defaults to `1` in the pipeline).
- PP1 Stage0 default: `PP1Stage0._should_recompute()` returns `False` when `SCOPE_KV_CACHE_RECOMPUTE_EVERY` is unset, which sets `do_kv_recompute=False` in the PP envelope and causes the mesh to skip recompute.

Immediate workaround to validate:
- launch PP1 with `SCOPE_KV_CACHE_RECOMPUTE_EVERY=1` so rank0 supplies `context_frames` (currently synthetic from `prev_latents_out`) and the mesh recomputes KV cache each chunk.

Open question (for review):
- What is the correct “proper” context-frame construction under PP1 (decoded-first-frame anchor + re-encode path) so temporal coherence matches non-PP, and how should this interact with overlap/backpressure?

## Objective

We’ve shipped PP1 server mode through **G1b** (dynamic prompts from rank0) and are now planning
**Phase G1c: overlap** rank0 VAE decode of chunk N with mesh inference of chunk N+1 to hide
decode latency behind the mesh’s stage1 time.

We have a concrete plan in:
- `scope-drd/notes/FA4/h200/tp/proposals/g1c-implementation-plan.md` (latest draft; include file contents in prompt pack)

Before we code it, we want a safety review focused on:
1) PyTorch/NCCL stream semantics for p2p send/recv  
2) Threading constraints for `torch.distributed` p2p  
3) `record_stream()` correctness for cross-stream tensor lifetime  
4) Hard-cut + shutdown behavior (crash-only rules)  
5) How to keep debug endpoints overlap-safe

Relevant prior art (already reviewed once for PP0):
- `deep-research/2026-02-22/pp-overlap-threading-and-queues/reply.md` (stream-ordering + crash-only shutdown notes)

Note: the plan doc should be treated as the source of truth (PP1 server-mode specifics), and this ask
is to confirm we’re not missing a subtle NCCL/thread/stream hazard when moving the same pattern into PP1.

## Repo prompt pack (include these files)

### The plan doc (primary input)

- `scope-drd/notes/FA4/h200/tp/proposals/g1c-implementation-plan.md` (read first)

### Server code that will change

- `scope-drd/src/scope/server/pp1_stage0.py` (new comms thread + overlap-safe `infer_latents(...)`)
- `scope-drd/src/scope/server/frame_processor.py` (PP1 path delegates to Stage0; no private overlap flags)
- `scope-drd/src/scope/server/app.py` (lifespan start/stop overlap; debug smoke endpoint becomes overlap-safe)

### Proven reference implementation

- `scope-drd/scripts/pp0_pilot.py` (A3 overlap segment: comms thread + streams + events + `record_stream()`)

### Distributed primitives / contract

- `scope-drd/src/scope/core/distributed/pp_control.py` (PPControlPlane p2p send/recv)
- `scope-drd/src/scope/core/distributed/pp_contract.py` (PPEnvelopeV1 / PPResultV1)

## Current state

PP1 server in **G1b** is stable:
- Rank0: HTTP/WebRTC + Stage0 stack (text encoder + VAE decode)
- Rank1/2: TP mesh (TP=2)

We’ve already observed a real “PP protocol corruption on kill” failure mode in PP1 bringup
(`_pickle.UnpicklingError` after killing rank0 mid-stream). Overlap increases the chance of
rank0 being terminated while comms are in flight, so we’re taking a hard line on crash-only
rules (no post-timeout PP/`dist` calls).

Rank0 inference path is synchronous today:
```
build_envelope → pp.send_infer → pp.recv_result → decode_latents → emit
```

## Proposed overlap design (depth=1, comms thread)

High-level idea: keep *exactly one* PP envelope in flight at a time; while the mesh computes
chunk N+1, rank0 decodes/emits chunk N.

Mechanics:
- Stage0 owns a comms thread + dedicated CUDA stream.
- Main thread builds envelope and records a `ready_evt` on the current stream.
- Comms thread waits on `ready_evt`, then:
  - `pp.send_infer(env)`
  - `res = pp.recv_result()` (blocking while mesh runs)
  - record `done_evt` on comm stream
  - enqueue `(res, done_evt)` to the main thread
- Main thread:
  - waits on `done_evt` on its current stream
  - calls `res.latents_out.record_stream(current_stream)`
  - returns `latents_out` to the existing VAE decode/emit path

Invariants / guardrails:
- **Comms thread owns all PP send/recv** on rank0. No other call sites may call
  `stage0.pp.send_infer/recv_result` directly.
- Queue depth = 1 (send_q/recv_q maxsize=1).
- Overlap gated behind `SCOPE_PP1_OVERLAP=1` (default off).
- Hard cut (`init_cache=True`): drain/discard in-flight result, `reset()` local state, restart bootstrap.
- Shutdown: `stop_overlap()` sends sentinel and joins.
  - If join times out: **crash-only** (do not touch PP/`dist` after timeout).

## Questions

### Q1) Stream semantics for `dist.send/recv` on CUDA tensors

- For NCCL backend: do `torch.distributed.send/recv` enqueue work on the *current* CUDA stream
  of the calling thread?
- Is waiting on `ready_evt` + recording `done_evt` sufficient to guarantee correct ordering for
  both send and recv?

### Q2) Memory safety across streams (`record_stream`)

- Is `res.latents_out.record_stream(main_stream)` the right/complete pattern to prevent
  allocator reuse when the tensor was received on a comm stream?
- Any additional steps required to make this safe without adding global synchronizations?

### Q3) Thread safety of `torch.distributed` p2p in a background thread

- Is it safe to call `dist.send/recv` from a Python background thread while the main thread
  is running CUDA kernels/launching on the same GPU?
- Any known NCCL/PyTorch gotchas with multi-threaded point-to-point, even if **only one**
  thread uses distributed primitives on that rank?

### Q4) Hard cuts with one in-flight envelope

- Is “always drain/discard the in-flight result, then reset + submit hard-cut envelope” the
  best practice for depth=1 overlap?
- Would you recommend an explicit control message (RESET) instead, or is that unnecessary
  complexity for this phase?

### Q5) Shutdown and failure behavior

- If the comms thread can’t join (hung `recv_result`), is the crash-only rule correct?
- Preferred termination behavior: raise and let the process die, `os._exit(1)`, `dist.abort()`,
  or something else to avoid stuck ranks / protocol corruption?

### Q6) Debug endpoint / smoke test safety

- `/api/v1/debug/pp1-smoke` currently sends a direct PP request for quick testing.
  Under overlap, it must route through Stage0’s overlap-safe API (or be disabled).
- Any recommended guardrails so debug calls can’t corrupt the comms protocol while streaming?

### Q7) Instrumentation

What minimal instrumentation would you add to catch overlap bugs early?
- timeouts / invariants (`_has_inflight`, queue occupancy)
- detection of “PP send without recv” protocol drift
- watchdog/keepalive best practices

### Q8) Process group + timeout isolation (optional, but important if you think it’s load-bearing)

Today `PPControlPlane` uses the default process group. Would you recommend creating a dedicated PP
process group with its own timeout/debug settings (and passing `group=` to send/recv) to reduce
the chance of PP overlap interactions with other collectives, and to make hang recovery less risky?

## Output format

Please reply with:
1) A review of the plan’s correctness + any missing synchronization steps.
2) Clear do/don’t guidance for p2p send/recv + streams + threads.
3) Recommended changes to shutdown + hard-cut handling (if any).
4) Any red flags that would make this unsafe to roll out.
