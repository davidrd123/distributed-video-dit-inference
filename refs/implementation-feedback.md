# Implementation Feedback — Gaps Found During PP0 Bringup

> **Source**: Scope PP bringup, Steps A0–A2 (2026-02-19 through 2026-02-22).
> Machine: 2x H100 80GB HBM3, NVLink. Branch: `worktree-tp-runtime-v0`.
>
> This file is written from the implementation side back to the reference library.
> It documents gaps, missing coverage, and "things we wish we'd had a page for"
> discovered while actually writing and debugging PP code against the operator manuals.

Last updated: 2026-02-22

---

## 1. Streaming inference PP vs training PP — the conceptual gap

**The problem**: Topics 13–15 and resources `gpipe`, `pipedream-2bw`, `zero-bubble-pp` are all
training-oriented. Our use case is structurally different in ways that matter for every design decision:

| Training PP | Streaming inference PP (our case) |
|-------------|-----------------------------------|
| Multiple micro-batches fill the bubble | Single micro-batch (B=1), one chunk in flight |
| Backward pass exists → F/B/W scheduling | No backward → no 1F1B, no zero-bubble scheduling |
| Activation checkpointing trades memory for compute | KV cache recompute trades freshness for overlap |
| Pipeline bubble = `(P-1)/M` fraction | Pipeline bubble = `max(0, stage0 - stage1)` or vice versa (simple 2-stage) |
| Weight updates require synchronization | Weights are frozen; no gradient sync |
| Batch size is a tuning knob | Batch size is always 1 (real-time SLO) |
| Micro-batch ID identifies work | `call_id` + `cache_epoch` identify work + staleness |

**What would help**: A synthesis (could be a new topic or a section in topic 15) that explicitly
maps training PP concepts to streaming inference PP. The mapping is:

- "Micro-batch" → "chunk" (but there's only ever one in flight per stage at PP0)
- "Bubble fraction" → "stage imbalance" (simple subtraction, not `(P-1)/M`)
- "Activation checkpointing" → "KV cache recompute" (but recompute is optional and couples stages)
- "1F1B schedule" → not applicable until D>1 overlap, and even then it's "send while computing next"
- "Gradient accumulation" → no equivalent
- The key scheduling question isn't "how to fill bubbles" but "how to overlap Stage 0 (encode+decode) with Stage 1 (denoise)" — which is closer to async pipeline than micro-batch scheduling

**Reference**: StreamDiffusionV2 is the closest but frames it as "Stream Batch" without connecting
back to the training PP literature. PipeDiT addresses DiT-specific PP but focuses on sequence
parallelism rather than the stage-split model we're implementing.

> **Library response (2026-02-22):** Topic 15 (`refs/topics/15-pipeline-scheduling-theory.md`) now covers this mapping explicitly. The "Mental model" section derives the bubble formula and immediately translates to inference: "Micro-batches are just **items in flight**" and "if P=2 and B=1, you get β = 1/2 ⇒ 50% idle time." The "Inference translation caveat" in Cross-resource agreement maps zero-bubble's B/W splitting to inference-side overlap candidates (transport on side stream, decode/encode, pre-staging envelopes). The "Throughput vs latency" key concept addresses the "sum → max" shift. The specific mapping table requested here (micro-batch→chunk, bubble fraction→stage imbalance, activation checkpointing→KV recompute, etc.) would be a valuable addition to Topic 15's mental model section — flagging for next revision.

---

## 2. Multi-rank debugging patterns — tribal knowledge gap

**The problem**: No topic covers the practical mechanics of debugging distributed code. This is
the single biggest time sink during bringup and it's pure tribal knowledge.

**Specific patterns we needed and had to figure out:**

### 2a. Reading torchrun crash output
- When rank1 crashes, torchrun kills rank0 too, and the output interleaves both ranks' stderr.
  The actual root cause is often buried 50+ lines above the `torch.distributed.elastic` summary.
- Pattern: search backward from "RuntimeError" or "NCCL error" for the rank that crashed *first*.
  The other rank's error is usually "connection reset" or "NCCL timeout" — a symptom, not the cause.

### 2b. Device mismatches in multi-GPU code
- `torch.cuda.current_device()` returns 0 on all ranks unless `torch.cuda.set_device(local_rank)`
  was called. This is not an NCCL problem — it's a CUDA context problem. We hit this in
  `sinusoidal_embedding_1d()` which used `device=torch.cuda.current_device()` as a default arg.
- Rule: never use `torch.cuda.current_device()` in model code. Use `tensor.device` from an input.
- Corollary: `torch.zeros(..., device="cuda")` also goes to device 0. Always be explicit.

### 2c. Silent warmup failures
- Pipeline constructors that run warmup inference silently break in multi-rank setups because
  warmup runs before the distributed coordination loop starts.
- We hit this twice: first with TP (already gated), then with PP (had to add a gate).
- Pattern: any "do work in `__init__`" is suspect in distributed code. Constructor should only
  allocate; first real inference should be driven by the coordination loop.

### 2d. Envelope/message debugging
- When a p2p send/recv fails, you need to know what was *in* the envelope. Our preflight
  pattern (validate → pickle → verify tensor specs → THEN send) was designed for this, but
  the general principle is: **log the message contents before the commitment point, not after**.
- Without this, a recv-side shape mismatch gives you "expected [1, 2160, 20, 128] got
  [2160, 40, 128]" with no context about what the sender thought it was sending.

### 2e. Distinguishing hangs from crashes
- A hang (all ranks blocked in NCCL) looks different from a crash (one rank exits, others timeout).
- `NCCL_DEBUG=INFO` helps but produces enormous output.
- Practical: set `NCCL_TIMEOUT` (or `SCOPE_DIST_TIMEOUT_S`) low during bringup (30-60s) so hangs
  surface quickly. Production can use longer timeouts.

**What would help**: A new topic (maybe `25-multi-rank-debugging.md`) with a practical checklist.
Not theory — just "when you see X, check Y first." The operator test matrix (OM-01 through OM-13)
tests for specific failure modes, but there's no guide for *diagnosing* an unexpected failure you
didn't plan for.

---

## 3. Resource gaps — specific cards that would have helped

### 3a. `pytorch-distributed-api` — HIGH priority for PP

Status in manifest: `pending`. This is the most-used reference during PP bringup and we don't
have a condensed card for it.

**What we needed from it:**
- `dist.send()` / `dist.recv()` — exact semantics, blocking behavior, tensor requirements
- `dist.isend()` / `dist.irecv()` — for overlap (Step A3)
- Process group creation and lifetime — `new_group()` with explicit ranks
- Timeout semantics — default 10 min for NCCL is way too long for bringup
- `barrier()` semantics — when it's safe to use (and when it masks bugs)

**Suggested actionables for the card:**
- Cite our bringup timeout choice (60s via `SCOPE_DIST_TIMEOUT_S`)
- Note that `dist.send()` is blocking on the CPU side but async on GPU
- Document the p2p + broadcast interaction pattern (PPControlPlane uses p2p for rank0↔leader,
  then leader uses broadcast inside mesh_pg)

### 3b. `cuda-async-execution` — MEDIUM priority (needed at A3)

Status: `pending`. Will become critical when we add overlap (async send/recv on dedicated streams).

**What we'll need:**
- Stream ordering guarantees across different streams
- Event record/wait pattern for cross-stream dependencies
- How NCCL operations interact with non-default streams

### 3c. Missing from manifest entirely

**`torchrun` / `torch.distributed.elastic` reference**: Not in the manifest at all. We use
torchrun for every multi-rank launch. Understanding its process management (how it spawns workers,
how it handles failures, what `--nproc_per_node` actually does to CUDA device assignment) would
prevent several classes of bug.

**Wan 2.1 paper** (arxiv 2503.20314): Listed in reading-guide.md section 5 as a candidate but
not in the manifest. It's the target model — understanding its architecture (causal vs bidirectional,
temporal attention patterns, how the 40-block structure maps to our pipeline) matters for every
PP partitioning decision.

---

## 4. Topic-level gaps and sharpening suggestions

### 4a. Topic 03 (Graceful shutdown) — needs PP shutdown protocol

The current synthesis covers general principles but doesn't address PP-specific shutdown:
- How does rank0 signal SHUTDOWN to the mesh? (Answer: PPAction.SHUTDOWN envelope)
- What happens if rank0 crashes without sending SHUTDOWN? (Answer: mesh timeout → watchdog)
- What's the ordering between "drain in-flight results" and "send SHUTDOWN"?
- The operator test matrix has OM-13 (orphan shutdown) but the topic doesn't walk through
  the PP shutdown sequence.

> **Library response (2026-02-22):** Topic 03 (`refs/topics/03-graceful-shutdown.md`) now covers all four points. The "Mental model" section distinguishes **drain + exit** (normal) vs **abort** (crash > hang). "Key concepts" covers `PPAction.SHUTDOWN` as the sentinel, heartbeat NOOPs for liveness detection, watchdog `os._exit(2)` for breaking blocked NCCL, and bringup timeouts (`SCOPE_DIST_TIMEOUT_S=60`). The "Drain vs abort" concept explicitly addresses the ordering question. The crash-only framing answers "what if rank0 crashes" — mesh timeout + watchdog is the recovery path.

### 4b. Topic 19 (Backpressure) — needs concrete PP queue design

The synthesis discusses backpressure in general terms. For PP0 A3 (overlap), we need:
- Concrete bounded queue design: `D_in` (envelopes buffered on mesh side) and `D_out`
  (results buffered on rank0 side)
- What happens when `D_out` is full and mesh produces a new result? (Block? Drop? Which?)
- How `D_in`/`D_out` interact with hard cuts (epoch change flushes queues)
- The O-01/O-02/O-03 tests in the operator matrix reference this but the topic doesn't
  have the design spelled out

> **Library response (2026-02-22):** Topic 19 (`refs/topics/19-producer-consumer-backpressure.md`) now has the concrete PP queue design. "Mental model" covers `D_in`/`D_out` with `D_in=D_out=2` as the minimal double buffer. "Key concepts" § "Bounded queues" defines both queues with capacity semantics, § "Backpressure rules" addresses the "D_out full" question (rank0 blocks enqueueing; mesh blocks on result send), and § "Why depth 2" derives the minimum. Hard-cut flushing is covered in the mental model: "Hard cuts are discontinuities. The system must flush both bounded queues, bump cache_epoch, and restart fill."

### 4c. Topic 20 (Message framing) — strongest topic, minor gap

This is the best topic in the library for PP implementation. The preflight-before-commit
pattern directly informed our Step A1 work. One gap: it doesn't cover **result framing**
(mesh → rank0 direction). Currently PPResultV1 is simpler than PPEnvelopeV1, but as we
add overlap, the result path needs the same anti-stranding treatment.

> **Library response (2026-02-22):** Acknowledged — Topic 20 covers the forward path (rank0→mesh) comprehensively but is lighter on result framing (mesh→rank0). The "Key concepts" section does reference `PPResultV1` versioning and `cache_epoch` filtering for stale results, but doesn't walk through the anti-stranding protocol for the result direction. Flagging for next revision when overlap (Step A3) makes result framing load-bearing.

### 4d. Topic 22 (KV cache management) — needs recompute coupling for PP

The synthesis covers KV cache lifecycle well for single-rank and TP cases. PP adds a
coupling that isn't addressed:
- Recompute requires `context_frames_override` from rank0's decoded frame buffer
- This creates a data dependency: mesh can't recompute until rank0 decodes → breaks overlap
- The R1 → R0a → R0 progression in our bringup plan is specifically about managing this coupling
- OM-10 tests for the missing-override case but the topic doesn't explain the PP recompute
  architecture

> **Library response (2026-02-22):** Topic 22 (`refs/topics/22-kv-cache-management.md`) now covers this coupling in detail. "Key concepts" § "The recompute coupling problem" explains why generator-only workers need `context_frames_override` and cites the failure-modes explainer + PP topology plan. "Practical checklist" item 3 walks through the R1→R0a→R0 progression explicitly: start with recompute disabled (R1), restore via rank0-provided override (R0a), and treat latent-only anchor (R0) as a gated quality-risk experiment. The `cache_epoch` + queue flush coupling with hard cuts is checklist item 4.

---

## 5. Operator test matrix — implementation status

From actual bringup, here's which OM tests we've *effectively* validated vs which are still
theoretical:

| Test | Status | Notes |
|------|--------|-------|
| OM-01 | Validated | Preflight catches unserializable meta before send (Step A1) |
| OM-02 | Validated | Preflight catches bad dtype before send (Step A1) |
| OM-03 | Not yet tested | Need plan-based dispatch (v1.1c) to test |
| OM-04 | Partially | Env parity check exists but doesn't cover all keys yet |
| OM-05 | Not yet tested | No conditional collectives in current code path |
| OM-06 | Not applicable yet | No ad-hoc broadcasts in PP0 (no TP inside mesh) |
| OM-07 | Not yet tested | Need PP1 (leader + non-leader) to test |
| OM-08 | Not yet tested | Need PP1 with mesh_pg to test |
| OM-09 | Not yet tested | Need overlap (D_out>1) to test |
| OM-10 | Not yet tested | Recompute disabled in PP0 bringup |
| OM-11 | Partially | We caught a real drift bug (sinusoidal_embedding device) but not via digest |
| OM-12 | Not yet tested | No weight fingerprinting yet |
| OM-13 | Informally | torchrun handles this via process group timeout, but no explicit watchdog |

**Observation**: The test matrix is well-designed but skews toward PP1 and v1.1c scenarios.
For PP0 bringup, the most valuable tests would be:
- A "basic round-trip" smoke test (send envelope → recv → compute → send result → recv result)
- A "shape handshake" test (rank1 tells rank0 its tensor shapes before first inference)
- A "call_id monotonicity" test (verify call_id never goes backward)

These are less about fault injection and more about "does the happy path work at all" — which
is where we've been spending all our time.

---

## 6. What's working well (keep doing this)

- **Topic 20 (message framing)** directly influenced the preflight-before-commit pattern in Step A1.
  The "commitment point" concept is the single most useful idea from the library for PP implementation.

- **Topic 02 (deadlock patterns)** — the three root causes (ordering, membership, stranding) are
  the right mental model. Every hang we've debugged falls into one of these.

- **`implementation-context.md`** — having the measurement → resource mapping is exactly right.
  When we hit the sinusoidal_embedding bug, knowing that "device mismatch" maps to Topic 04
  (determinism) helped frame the fix correctly.

- **`streamdiffusionv2` resource card** — the PP comm cost comparison (2-5ms PP vs 60ms TP-like
  at 2 GPUs) gave us confidence that the PP topology is worth pursuing. Having a reference number
  prevented "is PP even fast enough?" anxiety.

- **Reading guide phase structure** — "read what you need when you need it" is exactly right.
  We haven't needed topics 10, 16, 17, 18 yet and correctly skipped them.
