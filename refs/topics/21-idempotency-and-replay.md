---
status: draft
---

# Topic 21: Idempotency and replay

For fault tolerance in a streaming video pipeline, operations should be idempotent — re-executing a denoising step or VAE decode with the same inputs produces the same output. Combined with **replay from checkpointed state**, this gives you exactly-once semantics without distributed transactions.

## Resources

<!-- Resource IDs from manifest belonging to this topic -->
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| idempotency-dist | What is Idempotency in Distributed Systems? | low | pending |
| exactly-once | Exactly Once in Distributed Systems | low | pending |

## Implementation context

In TP v0, “replay” is intentionally coarse: pipeline reload and snapshot restore are blocked, so the safe recovery path for both hangs and Franken-models is restarting the `torchrun` job (v0 contract). In PP bringup, we reintroduce limited replay/idempotency via metadata: `call_id` must be monotonic, and `cache_epoch` increments on hard cuts so rank0 can drop stale results and flush bounded queues deterministically. These are the primitives needed before attempting any richer at-least-once / retry semantics.

See: `refs/implementation-context.md`, `scope-drd/notes/FA4/h200/tp/explainers/07-v0-contract.md` (no reload/snapshot), `scope-drd/notes/FA4/h200/tp/pp-next-steps.md` (cache_epoch filtering), `scope-drd/notes/FA4/h200/tp/pp-topology-pilot-plan.md` (hard cut flush rules).

Relevant Scope code:
- `scope-drd/src/scope/core/distributed/control.py` (TP headers: monotonic `call_id`, epoch fields, action framing)
- `scope-drd/src/scope/core/distributed/pp_contract.py` (`call_id`, `chunk_index`, `cache_epoch` as ordering/replay primitives)
- `scope-drd/scripts/pp_two_rank_pipelined.py` (queue flush/drop behavior that uses `cache_epoch` during bringup)

## Synthesis

<!-- To be filled during study -->

### Mental model

Idempotency + replay is how you make a streaming pipeline robust to “do it again” events: retries, duplicates, delayed messages, and restarts. The core idea is to separate:

- **Compute that can be safely re-run** (idempotent given the same inputs + state), from
- **Side effects that must not happen twice** (mutating caches, emitting outputs, advancing global counters).

In Scope, we intentionally handle this differently for TP v0 vs PP:

- **TP v0: no fine-grained replay; restart is the recovery.**
  - The TP model is SPMD lockstep: both ranks must execute the same code path and collectives in the same order, or you get an NCCL hang; and if they execute the same collectives but with different weights/inputs, you can get a Franken-model (silent corruption). (`scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md`)
  - Because the cost of “retry but with wrong state” is catastrophic, v0 locks down state mutation: pipeline reload is disabled, snapshot restore is disabled, and the safest recovery path is to restart the `torchrun` job. (`scope-drd/notes/FA4/h200/tp/explainers/07-v0-contract.md`)

- **PP bringup: limited replay primitives (ordering + epoch filters), not full checkpoint replay.**
  - Once you introduce stage boundaries (rank0 ↔ mesh) and bounded queues, you *will* observe “late” and “duplicate” phenomena in practice (timeouts, retries, overlap). Rather than implement full replay-from-checkpoint, PP starts with “first safe step” primitives: **monotonic IDs** (`call_id`, `chunk_index`) plus an **epoch** (`cache_epoch`) that increments on hard cuts so stale in-flight work can be dropped deterministically. (`scope-drd/notes/FA4/h200/tp/pp-topology-pilot-plan.md`, `scope-drd/notes/FA4/h200/tp/pp0-bringup-runbook.md`)
  - The goal is not “exactly once” in the distributed-transactions sense; it’s “don’t deadlock, don’t silently corrupt, and be able to drop stale/duplicate work safely” as a foundation for future retries.

### Key concepts

- **Idempotency of compute vs idempotency of side effects**
  - *Compute-idempotent* (ideal): running Stage 1’s generator twice with the same inputs and the same starting model/cache state yields the same `latents_out`.
  - *Side-effect-idempotent* (harder): repeating the operation does **not** double-apply state changes:
    - KV-cache mutation (append/evict/reset)
    - decoded-frame-buffer updates
    - output emission (don’t enqueue the same chunk twice)
  - TP v0 largely avoids needing side-effect idempotency by forbidding replay and forcing lockstep; PP reintroduces a minimal subset via metadata filters. (`scope-drd/notes/FA4/h200/tp/explainers/07-v0-contract.md`, `scope-drd/notes/FA4/h200/tp/pp-topology-pilot-plan.md`)

- **Exactly-once vs at-least-once across a stage boundary**
  - *Exactly-once* is expensive: you typically need durable checkpoints and/or transactional “apply” semantics for side effects.
  - *At-least-once + dedupe* is the pragmatic baseline: a sender may retry, so the receiver must detect duplicates and either (a) ignore them or (b) return the previously computed result.
  - For PP0/PP1 bringup, we aim for “at-least-once safe” **at the interface level** (envelopes/results carry IDs/epochs so rank0 can ignore duplicates/stale), while still preferring “crash > hang” if invariants are violated. (`scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md`, `scope-drd/notes/FA4/h200/tp/pp0-bringup-runbook.md`)

- **Monotonic IDs are the first safe step**
  - `call_id`: monotonic per session/call stream. TP v0 already treats out-of-order as corruption and rejects it. (`scope-drd/notes/FA4/h200/tp/explainers/07-v0-contract.md`)
  - `chunk_index` / `chunk_id`: monotonic per chunk stream; PP0 bringup explicitly checks monotonicity on both sides. (`scope-drd/notes/FA4/h200/tp/pp0-bringup-runbook.md`)
  - With monotonic IDs, receivers can implement a simple rule: **accept only the next expected ID; drop anything older; log anything unexpected.**

- **Epoch-based invalidation (`cache_epoch`)**
  - A hard cut is a semantic reset of caches; any in-flight work from the “old world” becomes invalid.
  - PP introduces `cache_epoch` to label which cache world a message/result belongs to, and increments it on hard cuts so rank0 can safely drop stale results without guessing. (`scope-drd/notes/FA4/h200/tp/pp-topology-pilot-plan.md`)

- **Bounded queues make drops/flushes deterministic**
  - PP0 defines two bounded queues (to-mesh and ready-for-decode) and a rule: a hard cut flushes both queues and increments `cache_epoch`. (`scope-drd/notes/FA4/h200/tp/pp-topology-pilot-plan.md`)
  - The PP0 runbook treats queue bounds + per-chunk invariants as the correctness surface before any “retry” sophistication. (`scope-drd/notes/FA4/h200/tp/pp0-bringup-runbook.md`)

### Cross-resource agreement / disagreement

- **TP docs agree replay is a liability at v0**: the failure-modes doc frames the two catastrophic outcomes (hang vs Franken-model), and the v0 contract locks down reload/snapshot precisely because “replay-like” state changes are hard to make safe under lockstep collectives. (`scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md`, `scope-drd/notes/FA4/h200/tp/explainers/07-v0-contract.md`)
- **PP docs agree on minimal replay primitives**: the PP topology plan and PP0 runbook both lean on monotonic IDs + `cache_epoch` + bounded queues as the minimum viable mechanism to drop stale/duplicate work and keep overlap bringup diagnosable. (`scope-drd/notes/FA4/h200/tp/pp-topology-pilot-plan.md`, `scope-drd/notes/FA4/h200/tp/pp0-bringup-runbook.md`)
- **Tension (intentional)**: distributed systems literature often pushes “at-least-once + idempotent handler” early; Scope TP v0 deliberately does *not* implement fine-grained retry, because the side effects (collectives + caches + weights) make silent corruption too easy. PP reintroduces only the parts that are safe without checkpointing (drop-by-ID/epoch), and still treats restart as the last-resort recovery.

### Practical checklist

**Goal for PP bringup**: make it safe for rank0 to ignore duplicates/stale work without deadlocking or emitting corrupted output.

- **Define the interface as a replay boundary**:
  - Stage 0 → Stage 1: envelope is the only authority on “what to run” (don’t let mesh infer reset decisions).
  - Stage 1 → Stage 0: result is accepted only if its IDs/epoch match the current expected stream.
  See: `scope-drd/notes/FA4/h200/tp/pp-topology-pilot-plan.md`.

- **Envelope must include (at minimum)**:
  - Monotonic identifiers: `call_id`, `chunk_index`/`chunk_id`
  - Cache epoch: `cache_epoch`
  - Explicit reset decisions: `init_cache` / reset flags for cache state
  - “Do the same amount of work” guard: `expected_generator_calls`
  - Shape/geometry fields needed to validate compatibility
  See: `scope-drd/notes/FA4/h200/tp/pp-topology-pilot-plan.md`, `scope-drd/notes/FA4/h200/tp/pp0-bringup-runbook.md`.

- **Result must include (at minimum)**:
  - The same identifiers + epoch: `call_id`, `chunk_index`/`chunk_id`, `cache_epoch`
  - A single authoritative payload: `latents_out`
  - Optional bringup tripwires: digest/fingerprint to detect corruption early
  See: `scope-drd/notes/FA4/h200/tp/pp-topology-pilot-plan.md`.

- **Receiver-side acceptance rules (rank0)**:
  - Maintain `expected_call_id` and `expected_chunk_index` (strictly increasing).
  - Drop and log:
    - `result.cache_epoch != current_cache_epoch` (stale or future)
    - `result.call_id < expected_call_id` (duplicate/late)
    - `result.chunk_index < expected_chunk_index` (duplicate/late)
  - Treat “ahead-of-time” results (`call_id > expected`) as a bringup error unless you intentionally add a reorder buffer (bounded by `D_out`).
  - On hard cut: flush queues and increment `cache_epoch` (so late results become self-identifying as stale).
  See: `scope-drd/notes/FA4/h200/tp/pp-topology-pilot-plan.md`, `scope-drd/notes/FA4/h200/tp/pp0-bringup-runbook.md`.

- **Mesh-side guardrails (Stage 1)**:
  - Reject out-of-order envelopes (TP v0 already uses “reject out-of-order call_ids” as an ordering corruption tripwire). (`scope-drd/notes/FA4/h200/tp/explainers/07-v0-contract.md`)
  - Do not apply cache mutations twice for the same `call_id`/`chunk_index` unless you explicitly implement “return cached result” semantics.

### Gotchas and failure modes

- **Duplicate envelope is not automatically safe**: even if compute is deterministic, Stage 1 cache state advances; replaying without guarding side effects can double-advance cache and silently corrupt downstream outputs.
- **Delayed results after hard cut are toxic unless filtered**: a result computed under the old cache state can arrive after reset; if decoded/emitted, it produces a Franken-model-like “looks plausible but wrong” stream. `cache_epoch` exists to make this droppable. (`scope-drd/notes/FA4/h200/tp/pp-topology-pilot-plan.md`)
- **Out-of-order acceptance is a hidden state machine**: buffering/reordering results is a correctness feature, not a convenience. If you don’t explicitly implement it, prefer to log + drop (or crash) rather than accidentally decode in the wrong order.
- **Hangs are worse than crashes**: if one side blocks forever (recv/send mismatch), you’ll strand GPUs until timeout. The v0 posture (“crash loudly with context, restart for recovery”) still applies; PP should preserve that posture while adding the minimal drop/flush semantics. (`scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md`, `scope-drd/notes/FA4/h200/tp/explainers/07-v0-contract.md`)
- **Bounded queues require explicit backpressure**: if Stage 0 keeps enqueuing without respecting `D_out`, you’ve built “unbounded replay backlog” accidentally. The PP plan makes queue depth a contract for this reason. (`scope-drd/notes/FA4/h200/tp/pp-topology-pilot-plan.md`)

### Experiments to run

- **Duplicate envelope**: send the same envelope twice (same `call_id`, `chunk_index`, `cache_epoch`).
  - Expected: receiver logs “duplicate/out-of-order” and drops; no deadlock; monotonicity checks still hold. (`scope-drd/notes/FA4/h200/tp/explainers/07-v0-contract.md`, `scope-drd/notes/FA4/h200/tp/pp0-bringup-runbook.md`)

- **Delayed result after hard cut**: artificially delay a Stage 1 result, then trigger a hard cut (flush queues, increment `cache_epoch`) before the delayed result arrives.
  - Expected: rank0 drops stale result by epoch mismatch, logs it, and continues; no decode/output emission from the stale epoch. (`scope-drd/notes/FA4/h200/tp/pp-topology-pilot-plan.md`)

- **Result reordering**: intentionally reorder two results (swap send order) in a test harness.
  - Expected: rank0 refuses to emit out-of-order output; either drops the unexpected result (and logs) or triggers an explicit bringup failure path. In either case: no deadlock and no silent corruption. (`scope-drd/notes/FA4/h200/tp/pp0-bringup-runbook.md`, `scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md`)

- **Retry after timeout (at-least-once simulation)**: simulate a lost message by timing out waiting for a result and resending the same envelope.
  - Expected: Stage 1 handles the duplicate safely (drop or return cached result); rank0 emits at most one output for that `call_id`/`chunk_index`; logs show the retry and the dedupe. (`scope-drd/notes/FA4/h200/tp/pp-topology-pilot-plan.md`)
