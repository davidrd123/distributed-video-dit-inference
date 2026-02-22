# 5 Pro Deep Research Request — External patterns/resources to strengthen TP/PP operator manuals

Date: 2026-02-22  
Status: Ready to run (copy/paste into repo prompt)

## Objective

We have strong **project-local** TP/PP notes in `scope-drd/notes/FA4/h200/tp/` and good “physics” cards (NCCL/CUDA/Dynamo), but we’re missing the **operator-manual layer**: short, standardized checklists + failure-mode rules that make v1.1/PP implementation and review fast.

This request asks you to propose **categories of outside material** (papers/docs/blogs/system design writeups) that would add leverage beyond our current notes, and to recommend specific sources/patterns to pull in.

The goal is not more background; it’s to import **battle-tested protocol patterns** and **review checklists** that reduce hang/drift risk.

## Current state (what works, what doesn’t)

### What works today

- The library provides “physics + failure-mode backing” for:
  - NCCL ordering/thread safety/stream semantics (`refs/resources/nccl-user-guide.md`)
  - CUDA stream + allocator lifetime correctness (`refs/resources/pytorch-cuda-semantics.md`)
  - torch.compile + collectives survivability (functional collectives + Dynamo internals: `refs/resources/funcol-rfc-93173.md`, `refs/resources/dynamo-deep-dive.md`)
- `scope-drd/notes/FA4/h200/tp/` contains ground-truth protocols and audits:
  - TP broadcast envelope (`.../explainers/03-broadcast-envelope.md`)
  - Failure taxonomy (hang vs Franken-model) (`.../explainers/06-failure-modes.md`)
  - v1.1 generator-only workers scaffold (`.../v1.1-generator-only-workers.md`)
  - PP rank0-out-of-mesh plan and runbook (`.../pp-topology-pilot-plan.md`, `.../pp0-bringup-runbook.md`, `.../pp-next-steps.md`)
  - 5 Pro deadlock audit for v1.1c (`.../5pro/10-v11-correctness-deadlock-audit/response.md`)

### What’s missing / weak

- The library’s “protocol/pattern” **topic syntheses** need to become enforceable operator manuals:
  - control-plane contract/anti-stranding/versioning,
  - multi-group deadlock avoidance (`world_pg` vs `mesh_pg`, conditional collectives),
  - bounded queues + overlap metrics,
  - KV-cache lifecycle as a lockstep state machine.
- Two high-leverage canonical basics are fetched but not yet converted/condensed:
  - `pytorch-distributed-api`
  - `cuda-async-execution`

## Repo prompt pack (include these files)

### Scope notes (ground truth)

- `scope-drd/notes/FA4/h200/tp/v1.1-generator-only-workers.md`
- `scope-drd/notes/FA4/h200/tp/explainers/03-broadcast-envelope.md`
- `scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md`
- `scope-drd/notes/FA4/h200/tp/5pro/10-v11-correctness-deadlock-audit/response.md`
- `scope-drd/notes/FA4/h200/tp/pp-next-steps.md`
- `scope-drd/notes/FA4/h200/tp/pp-topology-pilot-plan.md`

### Library packaging context

- `refs/tp-pp-operator-manual-packaging.md`
- `refs/v1.1-generator-only-workers-crosswalk.md`

### Operator-manual topics we’re trying to “ship”

- `refs/topics/20-message-framing-versioning.md`
- `refs/topics/02-deadlock-patterns.md`
- `refs/topics/03-graceful-shutdown.md`
- `refs/topics/19-producer-consumer-backpressure.md`
- `refs/topics/21-idempotency-and-replay.md`
- `refs/topics/04-determinism-across-ranks.md`
- `refs/topics/22-kv-cache-management.md` *(may still be stub; treat as a must-fill gap if missing)*

## What we want from the outside (categories + candidate sources)

For each category below:
1) recommend 2–5 specific sources (papers/docs/blogs),  
2) extract the *operator-relevant* rules/checklists, and  
3) suggest how to integrate them here (resource card vs topic synthesis claim).

### A) GPU distributed “protocol” patterns (anti-stranding, versioning, manifests)

We want sources that talk about:
- header/payload design,
- schema evolution/versioning,
- deterministic serialization,
- “preflight before sending header” / avoiding partial-protocol deadlocks.

### B) Multi-communicator / multi-process-group correctness (hierarchies, meshes, MPMD)

We want sources that cover:
- subgroup creation ordering rules,
- safe use of multiple NCCL communicators,
- pitfalls when mixing groups/streams,
- deterministic collective ordering across groups.

(This directly affects PP1+ where `mesh_pg` collectives must never involve rank0.)

### C) Failure handling + shutdown semantics (crash-only design, watchdogs, timeouts)

We want sources that provide:
- practical patterns for “crash > hang,”
- heartbeat/watchdog design,
- teardown hazards (including CUDA graphs / communicator destruction),
- “what to log so you can debug a hang.”

### D) Idempotency + replay primitives for streaming pipelines

We want patterns for:
- monotonic IDs and epochs (e.g., `cache_epoch`) to drop stale in-flight work,
- dedupe rules for at-least-once retry,
- bounded reordering buffers,
- exactly-once vs at-least-once tradeoffs in real-time systems.

### E) Backpressure + overlap scheduling (bounded queues, double buffering, metrics)

We want sources that help with:
- choosing queue depths (double buffering as the minimal overlap depth),
- stage imbalance management,
- proving overlap without synchronized clocks (period ≈ max vs sum),
- “microbatches in flight” / `max_outstanding` as the throughput lever.

### F) KV-cache lifecycle as a state machine (and how to decouple it)

We want sources/patterns for:
- cache reset/recompute/evict invariants,
- per-request cache ownership and lifecycle contracts,
- how other systems avoid “decoded-anchor” coupling (if applicable),
- how to structure recompute so it doesn’t collapse overlap.

### G) Compiler + distributed interactions (torch.compile, graph breaks, functional collectives)

We want sources with:
- concrete “do/don’t” rules for compiled distributed regions,
- multi-rank compilation parity requirements,
- how to design APIs so tracing stays functional (Tensor → Tensor),
- tests that catch graph divergence early.

## Deliverables (make it operational)

1) **A prioritized list of outside sources** by category, with 1–2 sentence “why it’s load-bearing for us.”
2) For each category, a **mini operator checklist** (5–12 bullets) that we can paste into the relevant `refs/topics/*` file.
3) 2–3 **break-it tests** per category (minimal, high-value) that confirm “crash > hang” or “drop stale work safely.”
4) Any **terminology corrections** (e.g., stream semantics wording) that prevent common misunderstandings.

## Non-goals

- Don’t propose a brand-new architecture; we want near-term guardrails for v1.1/PP bringup.
- Don’t require adopting heavy frameworks unless there’s a clear operator-manual payoff.

