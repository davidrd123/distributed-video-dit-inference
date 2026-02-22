# External patterns and resources pack for TP v1.1 + PP operator manuals

You already have the local invariants nailed (lockstep collectives, crash > hang, anti-stranding, cache_epoch, explicit phase plans). What you want from “outside” is not more background, it’s battle-tested protocol patterns you can turn into: **contract bullets + tripwires + break-it tests**.

Below is a response shaped to match `deep-research/2026-02-22/ask-5pro-external-patterns-and-resources.md`.

---

## 1) Prioritized outside sources to import

P0 means: if you add only these, the operator-manual layer improves immediately.

### P0 sources

* **PyTorch Distributed docs (ProcessGroup, collectives, send/recv, timeouts)**
  Load-bearing because PP0/PP1 correctness is “matching operations on the right group,” and PyTorch’s semantics differ from folks’ MPI intuitions in just enough ways to hurt.
* **NVIDIA NCCL documentation (stream ordering, communicator lifecycle, error handling)**
  Load-bearing because most “mysterious hangs” reduce to “collective ordering + stream ordering + lifetime.” Your manuals should cite one canonical source for “what NCCL actually guarantees.”
* **Over-decomposition and latency hiding in distributed/GPU systems (Dally et al.)**
  Load-bearing because your PP overlap story is basically “limited in-flight work hides bubbles,” and this gives you canonical language + mental models for why `max_outstanding` works and where it stops working.
* **Crash-only software (Candea/Fox)**
  Load-bearing because your posture already is crash-only in practice (restart `torchrun` is the recovery). This paper gives you a crisp justification and standard patterns (fail fast, don’t attempt heroic recovery in-process).
* **GPipe + PipeDream (pipeline bubbles, microbatch schedules)**
  Load-bearing because it’s the canonical “bubble math” for stage overlap, and it maps cleanly onto `D_in/D_out` and “expected_generator_calls” as a schedule contract.

### P1 sources

* **MPI communicator + tag discipline (MPI standard tutorials like LLNL / Open MPI docs)**
  Load-bearing as an analogy: communicator == process group, tag == (cache_epoch, call_id, field_id). Helps prevent “cross-chunk confusion” and wrong-group deadlocks.
* **Kafka idempotent producer and exactly-once semantics docs**
  Load-bearing because they turn “dedupe + epochs” into a clear operator story you can copy for PP: monotonic IDs, epoch fences, drop stale, retry policy.
* **Flink checkpointing and exactly-once writeups**
  Load-bearing because it cleanly separates compute idempotency vs side-effect idempotency, which is exactly the trap you must avoid when you eventually flirt with retries.

### P2 sources

* **Kubernetes liveness/readiness probes**
  Load-bearing for watchdog/heartbeat patterns and how to make failure detection operational.
* **TensorRT-LLM / vLLM KV cache docs**
  Load-bearing as “how production serving systems reason about KV cache ownership and lifecycle,” even if their structure differs from your video DiT cache.

---

## 2) Category-by-category: sources + copy-paste operator checklists + break-it tests

Each category below has: (a) 2–5 sources to pull, (b) a mini checklist you can paste into the matching `refs/topics/*.md`, (c) 2–3 break-it tests that prove crash > hang.

### A) GPU distributed protocol patterns

#### Candidate sources

* MPI message matching, tags, and communicators (LLNL/Open MPI docs)
* Protocol Buffers language guide (schema evolution patterns)
* Designing Data-Intensive Applications: encoding + schema evolution chapters
* PyTorch distributed send/recv + collectives semantics (as the “actual transport contract”)

#### Mini operator checklist for `refs/topics/20-message-framing-versioning.md`

* **Header is a commitment:** if you send an `INFER` header, the peer must be able to receive the entire payload without needing local inference.
* **Preflight before header:** validate schema, serialize meta, build tensor_specs, validate dtypes, and materialize tensors before emitting `INFER`.
* **Explicit versioning:** `*_envelope_version` is required; unknown version → fail fast before any collective.
* **Deterministic meta bytes:** canonical serialization for digesting and for “diffable” bug reports (stable key order, explicit defaults).
* **Manifest discipline:** receiver allocates strictly from `tensor_specs` (no shape inference, no hidden tensors in meta).
* **Top-level tensors only:** nested tensors in meta are forbidden and must fail fast.
* **Monotonic identifiers:** `call_id` monotonic across all actions; `chunk_index` monotonic across `INFER`s; `cache_epoch` fences cache worlds.
* **Action-driven receive loop:** `NOOP`/`SHUTDOWN` have no payload; do not allocate/receive tensors on those actions.

#### Break-it tests

* **Meta serialization failure:** inject a non-serializable object into meta; assert sender fails before header and receiver does not block.
* **Unsupported dtype:** add a `torch.bool` or fp8 tensor to specs; assert sender fails before header (or casts preflight) and receiver does not hang mid-recv.
* **Nested tensor in meta:** hide a tensor inside a dict; assert sender rejects it (no “falls through pickle”).

---

### B) Multi-communicator and multi-process-group correctness

#### Candidate sources

* PyTorch ProcessGroup and collective APIs
* NVIDIA NCCL communicator lifecycle and collective semantics
* MPI communicator split and message matching (for analogy and “order discipline”)

#### Mini operator checklist for `refs/topics/02-deadlock-patterns.md`

* **Group creation must be deterministic:** all ranks call `new_group` in the same order with the same rank lists, or you are building footguns.
* **Never mix group scopes:** rank0 must never call a `mesh_pg` collective; mesh ranks must never call TP collectives on `world_pg`.
* **Use group-rank semantics consistently:** for collectives, `src` is the rank within the group, not global rank (write this explicitly in the operator manual).
* **One wrapper to rule them all:** all collectives go through a small wrapper that requires `group=` and logs `(call_id, group_name, collective_name, tensor_shape)`.
* **No conditional collectives without parity:** any flag that gates a collective must be parity-checked (env or broadcast plan) and logged at init.
* **Per-chunk phase plan is mandatory:** broadcast `expected_generator_calls` + recompute flags + denoise steps; assert observed count matches.
* **Leader preflight in PP:** mesh leader must validate envelope and allocate tensors before any `mesh_pg` collective or broadcast.

#### Break-it tests

* **Wrong group collective:** intentionally call one mesh collective on `world_pg`; confirm it deadlocks instantly, then add a guard that fails before calling it.
* **Plan mismatch:** force rank0 to expect `N` generator calls but mesh does `N±1`; assert you crash via plan mismatch before hanging in a later collective.
* **Group creation order mismatch:** reorder `new_group` calls on one rank in a harness; ensure you fail at startup via sanity checks and never enter inference.

---

### C) Failure handling and shutdown semantics

#### Candidate sources

* Crash-only software (Candea/Fox)
* NCCL fault tolerance writeups and troubleshooting docs (NVIDIA)
* PyTorch distributed teardown caveats and real-world issues (destroy_process_group + torchrun signal handling)
* Kubernetes liveness/readiness probes patterns

#### Mini operator checklist for `refs/topics/03-graceful-shutdown.md`

* **Shutdown is part of the protocol:** include `*_Action.SHUTDOWN` and treat it as a first-class state transition.
* **Drain vs abort:** define two paths explicitly:

  * Drain: stop producing, finish in-flight, then send SHUTDOWN.
  * Abort: on invariant violation or suspected hang, crash fast (don’t attempt “best effort cleanup”).
* **Heartbeat and watchdog are paired:** watchdog based on “time since last header” needs either heartbeat NOOPs or a threshold that exceeds idle gaps.
* **Short bringup timeouts:** set process-group timeout low during bringup; long defaults hide bugs.
* **Never rely on teardown success:** `destroy_process_group()` can hang in some cases; you still need watchdog hard exits.
* **Log for hang localization:** always log last `(call_id, chunk_index, action, group, phase)` on each rank.

#### Break-it tests

* **Kill rank0 while worker is blocked in recv:** worker must exit quickly via watchdog/timeout, not after long defaults.
* **Crash after header:** force sender exception after emitting `INFER` header; ensure anti-stranding prevents it (sender must not emit header until preflight passes).
* **SIGTERM torchrun:** send SIGTERM to parent; verify both ranks exit and no rank remains pinned until NCCL timeout.

---

### D) Idempotency and replay primitives for streaming pipelines

#### Candidate sources

* Kafka idempotent producer + transactions documentation (IDs, epochs, fencing)
* DDIA: stream processing semantics and exactly-once vs at-least-once
* Flink checkpointing and state consistency docs
* “Dealing with rejection” style DS writeups (client retry contracts)

#### Mini operator checklist for `refs/topics/21-idempotency-and-replay.md`

* **Separate compute-idempotent from side-effect-idempotent:** generator forward may be repeatable; cache mutation and output emission are not.
* **Monotonic acceptance rule:** rank0 accepts only `(cache_epoch == current) ∧ (call_id == expected_next)` unless you deliberately implement a reorder buffer.
* **Epoch fences are mandatory:** hard cut increments `cache_epoch` and flushes bounded queues; stale results are dropped on sight.
* **Dedupe policy is explicit:** define what happens on duplicate envelope/result:

  * Drop duplicates (safe baseline).
  * Or return cached result (only if Stage 1 has a stable result cache keyed by IDs).
* **Retry policy is bounded:** retries allowed only within the same `cache_epoch`, and only for a limited wall-clock window.
* **No “implicit replay”:** never recompute a chunk with changed cache state; if you can’t guarantee same starting state, crash and restart.

#### Break-it tests

* **Duplicate envelope:** send same `(call_id, cache_epoch)` twice; Stage 1 must drop or return cached, and rank0 must emit output once.
* **Delayed result after hard cut:** delay result, then hard cut; rank0 must drop by epoch mismatch.
* **Out-of-order call_id:** send call_id regression; receiver must crash immediately.

---

### E) Backpressure and overlap scheduling

#### Candidate sources

* GPipe (pipeline microbatching, bubble)
* PipeDream (pipeline scheduling and staleness constraints)
* Dally et al. on over-decomposition and latency hiding
* Triton Inference Server docs on dynamic batching and queueing (optional)

#### Mini operator checklist for `refs/topics/19-producer-consumer-backpressure.md`

* **Two bounded queues, both enforced:** `D_in` bounds envelopes; `D_out` bounds decode backlog. Bounding only one is fake safety.
* **Double-buffer minimum:** `D_in=D_out=2` is the smallest depth that allows steady overlap on both boundaries.
* **Backpressure rule:** if `ready_for_decode` is full, Stage 0 must stop sending new envelopes.
* **Period math, not vibes:** steady-state period should approach `max(Stage0, Stage1)` if overlap works.
* **OverlapScore is a gate:** compute it from wall-clock emits; require a pass threshold (your current ≥0.30 is fine).
* **Hard cut flush is atomic:** flush both queues, bump epoch, restart fill.
* **Couplings are tracked explicitly:** when recompute is enabled (R0a), re-measure overlap and record how much of Stage 1 is serialized.

#### Break-it tests

* **Disable backpressure:** remove the “don’t enqueue when full” rule; inject slow decode; watch queue grow and memory blow up. Reinstate and confirm stabilization.
* **Depth sweep:** run `D ∈ {1,2,3,4}` and confirm `1` collapses overlap and `2` is first stable overlap.
* **Forced stage imbalance:** slow Stage 1 and confirm Stage 0 blocks (bounded), not accumulates.

---

### F) KV-cache lifecycle as a state machine and how to decouple it

#### Candidate sources

* PagedAttention paper (vLLM KV management model) and vLLM internals writeups
* TensorRT-LLM KV cache docs (paged KV, cache ownership)
* Megatron-LM / DeepSpeed inference caching discussions (if any)
* Any serving system that explicitly models cache epochs/fencing

#### Mini operator checklist for `refs/topics/22-kv-cache-management.md`

* **Treat KV cache as protocol state:** cache reset, recompute, eviction, and advance must be explicit in the envelope/plan.
* **Cache epoch is the fence:** hard cut increments epoch; no mixing results across epochs.
* **Recompute is a scheduled generator call:** include it in `expected_generator_calls`; no “optional recompute” decided locally on mesh/worker.
* **Anchor-frame dependency is explicit:** if recompute needs decoded-anchor semantics, then `context_frames_override` is required input, never a fallback.
* **No out-of-band cache mutation:** any RPC or helper that mutates caches outside the control plane is a Franken-model generator.
* **State parity hooks exist:** optional digest of cache indices or `current_start_frame` parity check across ranks/stages.
* **Decoupling experiments are gated:** R0 (latent-only anchor) must be treated as a semantic change with quality gates; R0a is the semantics-preserving baseline.

#### Break-it tests

* **Omit recompute override:** schedule recompute but omit `context_frames_override`; must fail pre-send or pre-collective, never “best effort fallback.”
* **Epoch mismatch accept attempt:** deliver a stale result after hard cut; rank0 must refuse to decode.
* **Out-of-band reset:** trigger a rank0-only cache reset in a test; drift tripwires must fire quickly.

---

### G) Compiler and distributed interactions

#### Candidate sources

* PyTorch `torch.compile` docs and compiler FAQ for distributed
* PyTorch functional collectives RFC and examples (already in your repo sources, but still citeable)
* TorchInductor design docs and Dynamo internals writeups

#### Mini operator checklist for `refs/topics` touching compile and `refs/topics/04-determinism-across-ranks.md`

* **Compile parity is required:** either all ranks compile the same regions with the same flags, or none do.
* **Collectives in compiled regions must be traceable:** use functional collectives to avoid graph breaks and keep ordering identical across ranks.
* **Warmup is lockstep:** compiled graphs must be warmed up in a rank-synchronized way (rank0-driven plan), or you will get divergent specializations.
* **Shape specialization is a contract:** dynamic shapes that diverge across ranks are a deadlock risk.
* **Graph-break budget is tracked:** record `graph_breaks` and `unique_graphs`; regressions fail CI for operator-manual readiness.
* **No conditional compile-time collectives:** anything that changes the set/order of collectives across ranks is a hang vector.

#### Break-it tests

* **Compile mismatch:** enable compile on rank0 only; must fail early via parity handshake, not via later hang.
* **Graph-break divergence:** introduce a rank-only graph break near a collective; ensure it fails in warmup, not in production.
* **Shape skew:** feed slightly different shapes to two ranks; ensure your envelope validation catches it before forward.

---

## 3) Terminology corrections to bake into the manuals

These are the phrasing landmines that cause wrong implementations.

* **CUDA async semantics:** “async” usually means “enqueued on a stream.” CPU time is not GPU completion time. If you measure or depend on completion, you need an explicit sync (event, stream sync, or a dependent op). (This belongs in the CUDA async execution resource card.)
* **NCCL stream semantics:** NCCL collectives are launched on a CUDA stream and are stream-ordered locally; correctness also requires a cross-rank ordering agreement about which collective happens when.
* **Process group rank vs global rank:** for collectives, `src` is the rank within the group. Make the “mesh leader is group-rank 0” convention explicit and never mix it with global rank.
* **“Determinism” scope:** for TP/PP inference, the goal is not bitwise reproducibility; it’s “either lockstep or crash with a diff.” (Your input digest + shard fingerprint posture matches this.)
* **“Exactly once” language:** don’t promise exactly-once unless you have durable checkpoints of side effects. What you can promise early is “drop stale, bound retries, dedupe by ID+epoch.”

---

## 4) Integration plan that matches your packaging standard

Concrete mapping to your operator-manual targets:

* `refs/topics/20-message-framing-versioning.md`
  Add: preflight-before-header, deterministic meta bytes, manifest discipline, and the 3 break-it tests from Category A.
* `refs/topics/02-deadlock-patterns.md`
  Add: multi-group rules, group-rank semantics, wrapper requirement, and wrong-group break-it test.
* `refs/topics/03-graceful-shutdown.md`
  Add: crash-only framing, drain vs abort, heartbeat+watchdog pairing, and kill tests.
* `refs/topics/19-producer-consumer-backpressure.md`
  Add: GPipe/PipeDream terminology for bubbles + microbatches, and make `D=2` “double buffer” rationale the headline.
* `refs/topics/21-idempotency-and-replay.md`
  Add: Kafka/Flink-inspired dedupe and epoch fencing patterns; keep it explicitly “drop stale” not “exactly once.”
* `refs/topics/22-kv-cache-management.md`
  Add: cache lifecycle as a protocol state machine; make recompute override requirement non-negotiable.
* `refs/topics/04-determinism-across-ranks.md`
  Add: compile parity + shape specialization parity + “drift detection not bitwise” phrasing; attach compile break-it tests.

Also, do the two pending conversions as first-class resource cards:

* `refs/resources/pytorch-distributed-api.md` (canonical cite for groups, collectives, send/recv, teardown)
* `refs/resources/cuda-async-execution.md` (canonical cite for stream enqueue semantics, timing gotchas)

---

If you want this to be maximally “pasteable,” I can also output each category’s checklist as a standalone block labeled with the exact target filename, so you can drop it into `refs/topics/*` with minimal editing.
