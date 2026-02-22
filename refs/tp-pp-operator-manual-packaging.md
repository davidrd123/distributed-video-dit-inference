# TP/PP “Operator Manual” Packaging — Marching Orders + Prompt Pack

Date: 2026-02-22  
Owner: (shared)  
Status: Active

## Goal

Turn the **project-local** TP/PP notes in `scope-drd/notes/FA4/h200/tp/` (explainers, bringup runbooks, 5Pro audits) into **short, enforceable, test-driven operator manuals** in this library (`refs/topics/*`). This makes v1.1 + PP implementation/review faster and less error-prone, and prepares a compact “repo prompt pack” to send to 5 Pro on the web.

## Why this matters (TP maintainer POV)

The library already helps as “physics + failure-mode backing” (NCCL/CUDA/Dynamo), but v1.1/PP work mostly needs **operator-manual packaging**:

- Implement/review the control-plane contract without stranding peers (anti-hang).
- Avoid multi-group deadlocks (world vs mesh PG).
- Prove overlap/backpressure with metrics, not vibes.
- Keep KV-cache lifecycle deterministic/lockstep (TP coupling root cause).

## The 4 operator surfaces (day-to-day hot spots)

1) **Control-plane contract patterns**  
   Anti-stranding, versioning, monotonic IDs, `cache_epoch`, manifest discipline (`tensor_specs`), deterministic meta serialization.

2) **Multi-group deadlock avoidance (TP + PP)**  
   `world_pg` vs `mesh_pg`, conditional collectives, per-chunk “phase plan” (call-count), leader preflight before `mesh_pg` collectives.

3) **Backpressure + overlap mechanics (PP)**  
   Bounded queues (`D_in/D_out`), double-buffer depth rationale, overlap metrics (`OverlapScore`), recompute coupling impact.

4) **KV-cache lifecycle (TP + PP coupling)**  
   Cache reset/recompute/advance is a lockstep state machine. It’s also the reason v0 workers run the full pipeline and the main semantic coupling that can collapse overlap.

## Operator-manual topic standard (opinionated + test-driven)

For the operator-surface topics below, standardize the `## Synthesis` section to include the same substructure (even if headings are short):

- **Contract**: must-haves + forbidden patterns
- **Tripwires**: fail-fast assertions + where they live (rank0 preflight, mesh leader preflight, worker-side checks, env parity)
- **Break-it tests**: 2–4 intentional violations that prove “crash > hang”
- **Instrumentation**: required log fields/counters + pass/fail gates

Wording correction that prevents real confusion:
- Avoid “NCCL runs on its own streams.” Use: **“NCCL ops are launched on a CUDA stream and are stream-ordered locally; plus there is a cross-rank ordering contract.”**

## Status snapshot (this repo)

Topic docs (`refs/topics/*.md`) current counts:
- `draft`: 18
- `stub`: 6

Remaining `stub` topic files:
- `refs/topics/10-inductor-fusion-rules.md`
- `refs/topics/17-reading-profiler-traces.md`
- `refs/topics/18-bandwidth-accounting.md`
- `refs/topics/22-kv-cache-management.md`
- `refs/topics/23-vae-latency-chunking.md`
- `refs/topics/24-video-dit-scheduling.md`

High-leverage “authoritative basics” (fetched but not converted/condensed yet):
- `sources/pytorch-distributed-api/raw/page.html`
- `sources/cuda-async-execution/raw/page.html`

## Work plan (assignable units)

Each unit is designed to be done by an agent independently with minimal merge conflicts.
Rule: **one agent edits one file per unit** unless explicitly coordinating.

### A) Operator-manual topics (ship checklists)

| Unit | File | Scope | Status | Owner | Notes |
|---|---|---|---|---|---|
| A1 | `refs/topics/20-message-framing-versioning.md` | control-plane contract | draft | — | Ensure “preflight before header” + manifest discipline + break-it tests |
| A2 | `refs/topics/02-deadlock-patterns.md` | multi-group deadlocks | draft | — | Ensure explicit `world_pg` vs `mesh_pg` rules + conditional-collective policy |
| A3 | `refs/topics/19-producer-consumer-backpressure.md` | queues/overlap | draft | — | Ensure `D_in/D_out=2` rationale + OverlapScore + recompute coupling callout |
| A4 | `refs/topics/03-graceful-shutdown.md` | crash > hang | draft | — | Ensure shutdown protocol + watchdog/heartbeat + anti-stranding tie-in |
| A5 | `refs/topics/04-determinism-across-ranks.md` | drift detection | draft | — | Needs operator-manual standardization (contract + tripwires + break-it tests + instrumentation) |
| A6 | `refs/topics/21-idempotency-and-replay.md` | epochs/flush/drop | draft | — | Ensure cache_epoch + monotonic IDs + bounded reordering policy |
| A7 | `refs/topics/22-kv-cache-management.md` | KV lifecycle | stub | — | Must become first-class operator manual (TP lockstep + PP recompute coupling) |
| A8 | `refs/topics/24-video-dit-scheduling.md` | PP scheduling | stub | — | Should connect bubble math + `max_outstanding` + PP overlap gating |

### B) Convert + condense “authoritative basics” (so topics can cite canon)

| Unit | Resource | Tier 2 | Tier 3 | Status | Owner | Notes |
|---|---|---|---|---|---|---|
| B1 | `pytorch-distributed-api` | `sources/pytorch-distributed-api/full.md` | `refs/resources/pytorch-distributed-api.md` | pending | — | Needed for deadlock/shutdown/process group lifecycle citations |
| B2 | `cuda-async-execution` | `sources/cuda-async-execution/full.md` | `refs/resources/cuda-async-execution.md` | pending | — | Needed for correct “async means enqueue” phrasing + timing/sync pitfalls |

### C) Crosswalk + prompt packaging

| Unit | File | Status | Owner | Notes |
|---|---|---|---|---|
| C1 | (new) crosswalk doc | todo | — | Map `scope-drd/notes/FA4/h200/tp/v1.1-generator-only-workers.md` sections → operator-manual topics/claims |
| C2 | (optional) prompt pack text | todo | — | A copy/paste-able prompt that references the “pack” file list below |

## “Repo prompt pack” (for 5 Pro on the web)

### Intent

Give 5 Pro enough context to:
- spot missing operator rules / failure modes,
- propose additional patterns/resources worth adding,
- and review the v1.1/PP contracts for deadlock/drift/stranding hazards,
without having to read the entire repos.

### Pack contents (recommended v0)

Include these files (both repos), in roughly this order:

**Scope local notes (ground truth for current design)**
- `scope-drd/notes/FA4/h200/tp/v1.1-generator-only-workers.md`
- `scope-drd/notes/FA4/h200/tp/explainers/03-broadcast-envelope.md`
- `scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md`
- `scope-drd/notes/FA4/h200/tp/explainers/07-v0-contract.md`
- `scope-drd/notes/FA4/h200/tp/5pro/10-v11-correctness-deadlock-audit/response.md`
- `scope-drd/notes/FA4/h200/tp/pp-next-steps.md`
- `scope-drd/notes/FA4/h200/tp/pp-topology-pilot-plan.md`
- `scope-drd/notes/FA4/h200/tp/pp0-bringup-runbook.md`

**Library operator-manual topics (what we want to “ship” as checklists)**
- `refs/topics/20-message-framing-versioning.md`
- `refs/topics/02-deadlock-patterns.md`
- `refs/topics/19-producer-consumer-backpressure.md`
- `refs/topics/03-graceful-shutdown.md`
- `refs/topics/04-determinism-across-ranks.md`
- `refs/topics/22-kv-cache-management.md` *(once drafted)*

**Library physics cards (minimal set; keep short)**
- `refs/resources/nccl-user-guide.md`
- `refs/resources/pytorch-cuda-semantics.md`
- `refs/resources/funcol-rfc-93173.md`
- `refs/resources/dynamo-deep-dive.md`

### Suggested 5 Pro ask (copy/paste)

1) Audit the v1.1 “generator-only workers” envelope contract: required fields, versioning, plan semantics, and preflight-before-header rules. What is missing that could cause hangs/Franken-model?
2) Audit PP0/PP1 bringup: multi-group process-group hazards (`world_pg` vs `mesh_pg`), conditional collectives, leader preflight, bounded queues and overlap metrics. Where are the deadlocks hiding?
3) Propose **additional operator-manual rules** (short checklists) and **break-it tests** we should standardize in the library topics.
4) Call out any place our mental model wording is misleading (streams/order/async semantics) and suggest corrected phrasing.

## Progress log (fill in as we go)

Update this table as units complete (keep it short; links only):

| Date | Unit | Change | Owner | Notes |
|---|---|---|---|---|
| 2026-02-22 | — | Initial plan doc created | codex | Snapshot + assignments |
| 2026-02-22 | C1 | Crosswalk first pass | codex | `refs/v1.1-generator-only-workers-crosswalk.md` |
