# Explainers → Reference Library (Backmap)

This document is a “reverse index” for the `scope-drd/notes/FA4/h200/tp/explainers/` series: start from the system we actually built (and its invariants/failure modes), then jump to the durable **topic syntheses** (`refs/topics/`) and **resource cards** (`refs/resources/`) that justify or explain those invariants.

Use this when:
- You’re debugging a TP/PP failure and want the right *conceptual* reference quickly (not a web search).
- You’re adding/updating an explainer and want to ensure the reference library stays aligned with production reality.

## One-screen mental model (what the explainers assume)

TP v0 is “BSP per chunk”: a control-plane broadcast once per chunk, then a data-plane generator call on every rank with many collectives inside. The top-level invariant shows up repeatedly:

> All ranks must call the same collectives, in the same order, with compatible tensors.

If that invariant breaks: **NCCL hang**.  
If it “holds” but ranks compute on different weights/inputs: **Franken-model**.

See: `../scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md`.

## Backmap by explainer

### 01 — Why Two GPUs

Link: `../scope-drd/notes/FA4/h200/tp/explainers/01-why-two-gpus.md`

What it’s doing:
- Frames the objective (real-time streaming latency) and why TP vs PP vs DP differ.

Invariants it leans on:
- “PP doesn’t help single-item latency unless you can fill it.”
- “Bandwidth-bound work doesn’t necessarily speed up with TP unless you change the memory traffic story.”

Jump into the library:
- Topics: `topics/13-classic-pipeline-parallelism.md`, `topics/15-pipeline-scheduling-theory.md`, `topics/16-roofline-model.md`, `topics/18-bandwidth-accounting.md`
- Resource cards: `resources/gpipe.md`, `resources/pipedream-2bw.md`, `resources/streamdiffusionv2.md`, `resources/making-dl-go-brrrr.md`

### 02 — Rank0 / worker architecture

Link: `../scope-drd/notes/FA4/h200/tp/explainers/02-rank0-and-workers.md`

What it’s doing:
- Explains asymmetric roles (rank0 serves + orchestrates, worker blocks on recv + runs pipeline) while keeping data-plane lockstep safe.

Invariants it leans on:
- Deterministic “one pipeline call per chunk” discipline on workers.
- Shutdown has to be explicit (avoid orphaned blocked ranks).

Jump into the library:
- Topics: `topics/03-graceful-shutdown.md`, `topics/04-determinism-across-ranks.md`, `topics/19-producer-consumer-backpressure.md`, `topics/21-idempotency-and-replay.md`
- Resource cards: `resources/nccl-user-guide.md`, `resources/pytorch-cuda-semantics.md`

### 03 — Broadcast envelope

Link: `../scope-drd/notes/FA4/h200/tp/explainers/03-broadcast-envelope.md`

What it’s doing:
- Makes the control plane concrete: what’s in `call_params`, what must be broadcast, and how “header/meta/tensors” framing prevents divergence.

Invariants it leans on:
- “Same inputs to the generator across ranks” is a correctness contract, not an optimization.
- Message framing must be explicit (versioned schema + fail-fast validation).

Jump into the library:
- Topics: `topics/20-message-framing-versioning.md`, `topics/04-determinism-across-ranks.md`, `topics/21-idempotency-and-replay.md`
- Resource cards: (protocol discipline) `resources/nccl-user-guide.md` (ordering), plus the PP contract/scheduling spine in `implementation-context.md` (Phase 3)

### 04 — TP math (layer sharding)

Link: `../scope-drd/notes/FA4/h200/tp/explainers/04-tp-math.md`

What it’s doing:
- Explains col/row-parallel linears, where the all-reduces happen, and why this creates many synchronization points.

Invariants it leans on:
- Collective ordering is lexical/program-order, so “same code path” is mandatory.
- Stream semantics matter once you try overlap/compile.

Jump into the library:
- Topics: `topics/01-nccl-internals.md`, `topics/02-deadlock-patterns.md`, `topics/05-cuda-streams.md`, `topics/11-functional-collectives.md`, `topics/12-compile-distributed-interaction.md`
- Resource cards: `resources/nccl-user-guide.md`, `resources/pytorch-cuda-semantics.md`, `resources/funcol-rfc-93173.md`, `resources/dynamo-deep-dive.md`, `resources/ezyang-state-of-compile.md`

### 05 — KV-cache head sharding

Link: `../scope-drd/notes/FA4/h200/tp/explainers/05-kv-cache-head-sharding.md`

What it’s doing:
- Explains why KV can be sharded by heads in TP, why cache contents aren’t broadcast, and why lifecycle must stay identical across ranks.

Invariants it leans on:
- Cache *indices and epochs* are part of the distributed contract (state machine, not just tensors).
- Memory pressure is an always-on constraint; allocator behavior becomes user-visible under concurrency.

Jump into the library:
- Topics: `topics/22-kv-cache-management.md`, `topics/07-gpu-memory-management.md`, `topics/04-determinism-across-ranks.md`, `topics/18-bandwidth-accounting.md`
- Resource cards: `resources/pytorch-cuda-semantics.md`, `resources/pagedattention.md` (when reasoning about cache lifecycle/memory patterns)

### 06 — Failure modes (Seven Questions)

Link: `../scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md`

What it’s doing:
- Collapses failure space into two catastrophic classes (hang vs drift) and maps them to concrete guardrails (Q1–Q7).

Invariants it leans on:
- Drift detection (digest/fingerprint) is necessary because NCCL won’t tell you you’re wrong.
- Distributed compile is fragile when ranks diverge on breaks/guards.

Jump into the library:
- Topics: `topics/02-deadlock-patterns.md`, `topics/03-graceful-shutdown.md`, `topics/04-determinism-across-ranks.md`, `topics/09-dynamo-tracing.md`, `topics/11-functional-collectives.md`
- Resource cards: `resources/nccl-user-guide.md`, `resources/dynamo-deep-dive.md`, `resources/ezyang-state-of-compile.md`, `resources/funcol-rfc-93173.md`

### 07 — The v0 contract

Link: `../scope-drd/notes/FA4/h200/tp/explainers/07-v0-contract.md`

What it’s doing:
- Makes the “correctness/diagnosability before performance” rule explicit and turns it into checkable gates.

Invariants it leans on:
- Role symmetry is less important than invariant visibility (env parity, tripwires, watchdogs).

Jump into the library:
- Topics: `topics/03-graceful-shutdown.md`, `topics/04-determinism-across-ranks.md`, `topics/21-idempotency-and-replay.md`
- Resource cards: `resources/nccl-user-guide.md` (hang mechanics + ordering constraints), `resources/dynamo-deep-dive.md` / `resources/ezyang-state-of-compile.md` (compile divergence class)

### 08 — StreamV2V mapping (what to borrow, what not to port)

Link: `../scope-drd/notes/FA4/h200/tp/explainers/08-streamv2v-mapping.md`

What it’s doing:
- Separates “transport/scheduling patterns worth stealing” from “topology-specific design that doesn’t port 1:1.”

Invariants it leans on:
- Distinguish phase-PP (rank0-out-of-mesh control boundary) vs block-PP (mesh-internal layer pipeline).
- Bounded in-flight (`max_outstanding`) and buffer pools are often required for sustained streaming.

Jump into the library:
- Topics: `topics/13-classic-pipeline-parallelism.md`, `topics/15-pipeline-scheduling-theory.md`, `topics/19-producer-consumer-backpressure.md`, `topics/05-cuda-streams.md`
- Resource cards: `resources/streamdiffusionv2.md`, `resources/pytorch-pipelining-api.md`, `resources/gpipe.md`, `resources/zero-bubble-pp.md`

## Cross-cut map: Seven Questions → library pointers

This mirrors the “Seven Questions” in `../scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md` and gives a quick “where do I read about this class of bug” index.

- **Q1 (pipeline reload → drift)**: determinism + idempotency at protocol boundaries → `topics/04-determinism-across-ranks.md`, `topics/21-idempotency-and-replay.md`
- **Q2 (drift detection)**: parity checks and “prove sameness” instrumentation → `topics/04-determinism-across-ranks.md`
- **Q3 (LoRA / weight mutation)**: “same weights” contract + invariant enforcement → `topics/04-determinism-across-ranks.md`
- **Q4 (KV cache lifecycle)**: cache-as-state-machine + reset/recompute/evict invariants → `topics/22-kv-cache-management.md`
- **Q5 (workers run full pipeline)**: stash vs recompute coupling and what state crosses a boundary → `topics/22-kv-cache-management.md`, `topics/23-vae-latency-chunking.md`
- **Q6 (compile divergence)**: tracing/guards/breaks + why funcol exists → `topics/09-dynamo-tracing.md`, `topics/11-functional-collectives.md`, `topics/12-compile-distributed-interaction.md`
- **Q7 (shutdown/orphans)**: teardown semantics + watchdog strategies → `topics/03-graceful-shutdown.md`

## Maintenance note (keeping explainers and the library aligned)

If an explainer introduces a new *invariant* (“must be identical across ranks” / “must be stable across chunks”), it should usually be grounded in one of:
- a topic synthesis (`refs/topics/...`) that states the invariant plainly, and/or
- a resource card (`refs/resources/...`) that makes the “why” citable.

If you’re unsure where a new invariant belongs, start by updating the relevant topic synthesis (human-facing), then sharpen or add a resource card only if the invariant depends on an external source (NCCL semantics, CUDA stream rules, Dynamo tracing behavior, etc.).

