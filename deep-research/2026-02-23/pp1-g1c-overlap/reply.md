According to a document from **2026-02-24**, your **G1c overlap plan (decode chunk N ∥ mesh inference chunk N+1)** is *basically correct*, and it ports the right “A3” ideas from `pp0_pilot.py`: **single comms thread**, **dedicated comm stream**, **ready/done CUDA events**, **bounded queues**, **`record_stream()`**, and **crash-only on join timeout**.

The remaining risk isn’t “threads are scary” in the abstract. It’s a small set of concrete footguns:

1. **cross-thread CUDA “current stream” mismatches** (ordering bug)
2. **accidental second caller of PP send/recv** (protocol corruption)
3. **shutdown/hard-cut paths that touch `dist` after the comms thread might be wedged** (hang/corruption)
4. **debug endpoints that bypass the overlap API** (same as #2)

Below I’ll answer your Q1–Q8 with do/don’t guidance, then list red flags and minimal instrumentation.

---

## What’s implemented today vs what G1c proposes

### Today (PP1 rank0 synchronous)

`frame_processor.py` does:

* `stage0.build_envelope(call_params)`
* `stage0.pp.send_infer(env)`
* `stage0.pp.recv_result()`
* `stage0.process_result(res)`
* `stage0.decode_latents(res.latents_out)` (if VAE loaded)

…and `/api/v1/debug/pp1-smoke` **also directly** calls `stage0.pp.send_infer/recv_result` (bypassing frame_processor entirely). That’s safe *only because there is no comms thread today*.

### G1c (proposed)

Move the entire overlap state machine into `PP1Stage0`:

* comms thread owns *all* PP transport
* main thread builds envelopes and decodes
* **depth=1**: exactly one envelope in flight
* bootstrap: submit first, then block for first result
* steady-state: when result N arrives, submit N+1 immediately, then decode N

That is the right architecture boundary: FrameProcessor becomes a dumb caller of `stage0.infer_latents()`.

---

## Q1) Stream semantics for `dist.send/recv` on CUDA tensors

### What to assume (NCCL)

For CUDA/NCCL, **the calling thread’s “current stream” matters**. ProcessGroupNCCL sets up dependencies relative to *the thread-local current stream* when it enqueues NCCL work. If you call `dist.send/recv` from a different thread without explicit stream edges, you can get a real ordering bug (NCCL reads inputs before they’re produced, or main consumes outputs before they’re written).

### Is your ready/done event scheme sufficient?

**Yes, if (and only if) all dist ops happen inside `with torch.cuda.stream(comm_stream):` in the comms thread**, and you do:

* main thread: build GPU tensors → `ready_evt.record(main_stream)`
* comm thread: `comm_stream.wait_event(ready_evt)` → `send_infer` → `recv_result` → `done_evt.record(comm_stream)`
* main thread: `main_stream.wait_event(done_evt)` before touching `res.latents_out`

That creates the explicit DAG you want:

`main_produce → ready_evt → comm_stream → NCCL stream(s) → comm_stream → done_evt → main_consume`

**Do**

* Keep the `with torch.cuda.stream(comm_stream):` around *both* send and recv.
* Record `ready_evt` *after* any GPU work that produces envelope tensors (notably UMT5 encode).

**Don’t**

* Ever call `pp.send_infer` in main thread “just this once”.
* Assume `dist.send/recv` CPU-blocks and therefore “the GIL doesn’t matter”. It’s mostly stream semantics.

My confidence this is correct is high$_{85%}$, because it matches your PP0 A3 pattern and matches how NCCL PGs are designed to be used.

---

## Q2) Memory safety across streams (`record_stream`)

### Is `res.latents_out.record_stream(main_stream)` the right pattern?

**Yes.** The ordering (`wait_event`) is about *correctness of values*.
`record_stream()` is about *allocator lifetime*. They solve different problems.

Even if you wait on `done_evt`, the caching allocator doesn’t automatically know that the main stream will keep using memory allocated/filled on the comm stream. `record_stream(main_stream)` tells it “don’t recycle this storage until main_stream is done with it.”

### Any extra steps?

A couple of nitpicks:

* Only call `record_stream` after you’ve done `main_stream.wait_event(done_evt)`. (Otherwise you can attach the wrong lifetime edge.)
* Guard it: only if `res.latents_out` is a CUDA tensor.

**Do**

* `wait_event(done_evt)` → `record_stream(main_stream)` → decode/clone/postprocess.

**Don’t**

* Treat `record_stream` as ordering. It isn’t.

One subtle performance note (not correctness): your current `process_result()` does `detach().clone()` of `latents_out`. That adds a GPU copy on rank0 before you can submit N+1. It’s safe; it just reduces overlap efficiency. You can revisit later.

---

## Q3) Thread safety of `torch.distributed` p2p in a background thread

### Is it safe?

**Yes, with the constraint you already stated:** only one thread on that rank uses the PP process group for p2p operations during steady state.

PyTorch/NCCL p2p from a background Python thread is a standard trick. The problems show up when:

* two threads call into the *same communicator/process group* concurrently, or
* you mix thread-local streams without explicit edges, or
* you call into dist from signal/exception paths after partial failure.

Your design avoids (most of) this.

**Do**

* Enforce “PP transport is owned by comms thread” as a hard invariant.
* Add a single Stage0-level lock so that `infer_latents()` cannot run concurrently from two call sites (frame loop vs debug endpoint). This is an easy-to-miss corruption vector.

**Don’t**

* Let `/api/v1/debug/pp1-smoke` call `stage0.pp.*` directly when overlap is active. That’s a protocol-corruption bug, not a “maybe”.

Confidence: safe$_{75%}$, because PyTorch doesn’t *guarantee* thread safety broadly, but “single dist thread + explicit stream edges” is the known-good subset you’re using.

---

## Q4) Hard cuts with one in-flight envelope

### Is “drain/discard in-flight result, then reset + submit hard-cut envelope” best practice?

For depth=1: **yes**. It’s the simplest protocol-safe thing.

The key invariant is: **you can’t skip a recv**. If you’ve sent an envelope, you must receive its result (even if you throw it away), or the next message boundary is ambiguous and you risk the kind of pickle corruption you’ve already seen when killing rank0 mid-stream.

### Should you add a RESET control message instead?

Not for G1c. It’s extra surface area, and you’d still need “receive what you already sent” semantics.

**Do**

* On `init_cache=True` while `_has_inflight`:

  * block on `get_result()` (with the same watchdog/timeout policy)
  * discard result (don’t decode)
  * `reset()` local state
  * bootstrap submit of the hard-cut envelope

**Don’t**

* Reset local sequencing state while an old result is still inbound. That’s how you get “chunk_index drift” and hard-to-debug misalignment.

One extra guard I’d add: include `cache_epoch` checks in the overlap path. You already carry it in the contract; use it to reject stale results after reset.

---

## Q5) Shutdown and failure behavior

### If the comms thread can’t join (hung in `recv_result`), is crash-only correct?

**Yes.** If join times out, you’re in the “receiver might be waiting on NCCL / rank1 might be dead / stream might be wedged” regime. Touching `dist` from the main thread after that is how you get interleaving and corruption.

The plan’s rule is correct:

* `stop_overlap()` must **not** call PP/`dist` after a join timeout.

### Preferred termination behavior?

For “comms thread didn’t exit” and “comms thread threw”, I prefer:

* log a clear error
* **`os._exit(2)`** (or equivalent hard stop)

Rationale: raising may get caught by FastAPI/uvicorn plumbing and leave the process limping while still holding NCCL resources. A hard exit is the cleanest “crash-only” behavior.

If you want nicer cluster behavior, you can also recommend operator env knobs:

* shorter `SCOPE_DIST_TIMEOUT_S` for faster error
* `NCCL_ASYNC_ERROR_HANDLING=1` / `NCCL_BLOCKING_WAIT=1` style settings (depends on your environment)

But the core rule stands: **if join timed out, do not attempt “graceful PP shutdown”.**

---

## Q6) Debug endpoint / smoke test safety

### Current state

`/api/v1/debug/pp1-smoke` currently does direct `stage0.pp.send_infer()` / `recv_result()`. That will become **actively dangerous** once overlap is enabled, because it can interleave with the comm thread (even if `_send_lock` serializes sends, it does not serialize the full send↔recv protocol).

### Recommended guardrails

Pick one and be strict:

**Option A (simplest, safest):**

* If overlap is active, `/pp1-smoke` returns 409 “disabled during overlap”.

**Option B (usable):**

* Route `/pp1-smoke` through `stage0.infer_latents(...)` and put a Stage0-level lock so that smoke can’t run concurrently with the realtime loop.

**Do**

* Make it impossible for any endpoint to call `stage0.pp.*` directly while overlap is started.

**Don’t**

* Rely on “users won’t click smoke while streaming”. They will.

---

## Q7) Instrumentation: minimal set that catches real bugs early

Here’s the smallest instrumentation that actually detects overlap bugs (not vanity metrics):

### Invariants (assertions)

* `res.ok` must be True else crash (you already do this)
* `res.chunk_index == expected_chunk_index` (you should add this to overlap path)
* `res.cache_epoch == current cache_epoch` (reject stale after reset)
* `_has_inflight` consistency:

  * `_has_inflight=True` ⇒ you must eventually receive exactly one result before submitting another
* queue occupancy:

  * with depth=1, `send_q.qsize()` and `recv_q.qsize()` should never exceed 1; log if they do

### Timing / watchdogs

* comm thread “last progress monotonic”

  * updated after each successful `recv_result` completion
* main thread “time since last recv_q item”

  * if exceeds e.g. `max(5×median_stage1_ms, 5s)` you are hung; crash-only

### Protocol drift detection

* Track monotonic `call_id` on rank0 side; if you ever observe non-monotonic or gaps (beyond expected), treat as corruption.

### One correctness canary (optional)

* After waiting on `done_evt`, read one scalar from `latents_out` (e.g. `.flatten()[0].float().item()`) under an env flag. It forces sync and can catch “forgot to wait_event” mistakes fast. Only enable in debug; it adds latency.

---

## Q8) Dedicated PP process group + timeout isolation

### Should you do it?

I’d recommend it as **P1 hardening**, not strictly required for G1c correctness.

Why it helps:

* Lets you set a **shorter timeout** for PP p2p than the default group (which also covers startup collectives / barriers).
* Reduces the blast radius if something else ever touches the default group on rank0 (future features).
* Keeps PP traffic on a smaller communicator (ranks {0,1}) which is conceptually cleaner.

Costs:

* You must create it with `dist.new_group([0, leader])` in a way that **all ranks participate** (same as your `mesh_pg` creation) to avoid deadlocks.
* You must thread `group=` through every PP send/recv call.

Net: worth it if you’re about to rely on overlap in production, because timeout/isolation becomes operationally important.

---

## Missing steps / red flags before you ship overlap

These are the “don’t ignore” items:

1. **Stage0 must serialize callers.**
   Even if you route smoke through `infer_latents`, if it can run concurrently with the realtime loop you can still violate your own depth=1 invariants. Add a lock or disable smoke.

2. **FrameProcessor must stop touching `reset()` directly.**
   Today it does `if init_cache: stage0.reset()` before the synchronous PP send/recv. Under overlap, reset needs to be mediated by the overlap state machine (drain/discard first). Your plan already says this; treat it as a hard requirement.

3. **Never touch PP/`dist` after a comm-thread join timeout.**
   Not even for “send shutdown”. Crash-only means crash.

4. **Be precise about events:**

   * `ready_evt` must be recorded after *all* envelope GPU producers (UMT5 encode).
   * `done_evt` must be recorded after `recv_result` returns, inside the comm stream context.
   * main must wait `done_evt` before touching result tensors.

If you meet those, I don’t see a lurking NCCL/stream gotcha that makes G1c unsafe.

---

## TL;DR do/don’t checklist

**DO**

* One comm thread owns PP send/recv.
* Wrap PP send+recv in `with torch.cuda.stream(comm_stream)`.
* Use `ready_evt` / `done_evt` and `wait_event` exactly as in pp0 A3.
* `record_stream(main_stream)` on `latents_out` after `wait_event(done_evt)`.
* Drain/discard in-flight result on hard cut before reset.
* Crash-only on join timeout; prefer `os._exit`.

**DON’T**

* Let any debug endpoint call `stage0.pp.send_infer/recv_result` while overlap is active.
* Call any dist op from main thread while comm thread might be inside dist.
* Assume `record_stream` provides ordering (it doesn’t).
* Allow concurrent `infer_latents()` callers.

If you want, I can rewrite your ask doc’s Q1–Q8 into a “ship gate” rubric (pass/fail checks) so you can run G1c like an operator-matrix test rather than vibes.

And yeah: quick posture check. Relax your shoulders and unclench your jaw for 10 seconds.
