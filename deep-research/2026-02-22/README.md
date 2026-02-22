# Deep Research — 2026-02-22 (TP v1.1 + PP + compile operator manuals)

This folder contains:
- one Deep Research writeup (`phase1-instrumentation.md`), and
- five “ask” prompts intended to be run against a strong web model / 5 Pro, producing **copy-pastable operator-manual output** (schemas, checklists, tripwires, break-it tests).

The missing value we’re targeting is *not* more NCCL/CUDA/Dynamo background — it’s **operator-manual packaging**: short, standardized, test-driven checklists that let you implement/review v1.1 + PP changes without rereading 5 long notes.

## Recommended run order (lowest risk → highest leverage)

1. `ask-5pro-tp-v11-envelope-contract.md`
2. `ask-5pro-pp-rank0-out-of-mesh.md`
3. `ask-5pro-compile-distributed-hardening.md`
4. `ask-5pro-kv-cache-lifecycle-and-decoupling.md`
5. `ask-5pro-external-patterns-and-resources.md` (nice-to-have enrichment; can be deferred)

## Where the outputs should land (paste targets)

Treat the outputs as “patches” to these **topic operator manuals** (not resource cards):

- TP v1.1 envelope contract → `refs/topics/20-message-framing-versioning.md`, `refs/topics/02-deadlock-patterns.md`, `refs/topics/04-determinism-across-ranks.md`, `refs/topics/22-kv-cache-management.md`, `refs/topics/23-vae-latency-chunking.md`
- PP rank0-out-of-mesh bringup → `refs/topics/19-producer-consumer-backpressure.md`, `refs/topics/20-message-framing-versioning.md`, `refs/topics/02-deadlock-patterns.md`, `refs/topics/21-idempotency-and-replay.md`, `refs/topics/03-graceful-shutdown.md`
- compile+distributed hardening → `refs/topics/09-dynamo-tracing.md`, `refs/topics/11-functional-collectives.md`, `refs/topics/12-compile-distributed-interaction.md`, `refs/topics/04-determinism-across-ranks.md`, `refs/topics/02-deadlock-patterns.md`
- KV-cache lifecycle + decoupling → `refs/topics/22-kv-cache-management.md` (primary), `refs/topics/21-idempotency-and-replay.md`, `refs/topics/04-determinism-across-ranks.md`

Crosswalk companions (optional, but useful to keep in sync):
- `refs/v1.1-generator-only-workers-crosswalk.md` (make it a one-page “ship checklist”)
- `refs/explainers-to-reference-library.md` (keep mappings accurate as topics evolve)

## Quality bar (how to judge whether the output is “useful”)

An answer is only “good” if it is **Scope-shaped** and operational:

- **Specific fields and invariants**, not generic advice:
  - `tp_plan`, `tp_envelope_version`
  - `expected_generator_calls`, recompute decision bits
  - `call_id`, `chunk_index`, `cache_epoch`
  - `D_in` / `D_out` and queue-depth implications
  - “validate/pickle/spec everything *before* header/bcast” (anti-stranding)
- **Break-it tests**: each ask should return 4–10 intentional violations with expected failure signatures (“crash quickly; no multi-minute hang”).
- **“Where to assert”** is explicit (rank0 preflight vs mesh leader vs worker).
- **Anchored to our notes** (file paths + run numbers) and consistent with the existing 5 Pro audit history in `scope-drd/notes/FA4/h200/tp/5pro/`.
- **Delta-first**: if a point is already covered in prior 5 Pro threads (especially #10 and #13), don’t re-derive it — either (a) refine it into a reusable checklist/break-it test, or (b) explicitly state “unchanged” and move on.

## Prior art / continuity (why this works)

We already have a long interaction history with 5 Pro here:
- `scope-drd/notes/FA4/h200/tp/5pro/10-v11-correctness-deadlock-audit/`
- `scope-drd/notes/FA4/h200/tp/5pro/12-v11-risk-ranked-execution-plan/`
- `scope-drd/notes/FA4/h200/tp/5pro/13-pp-execution-ready/`
- `scope-drd/notes/FA4/h200/tp/5pro/07-graph-break-audit/`

Use those as “tone and rigor” calibration: P0/P1 blockers, concrete schemas, and testable guardrails.

## Notes

- This repo is intentionally **not** asking Deep Research to propose a brand-new architecture; it’s asking for operator-manual scaffolding that makes safe iteration faster.
- If an output suggests adding new external sources, capture them as candidates (later) rather than immediately modifying `refs/manifest.yaml` during an active multi-agent sprint.
