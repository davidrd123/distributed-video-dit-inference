# 5 Pro Deep Research Request — TP v1.1 Generator-only Workers (Envelope Contract Audit)

Date: 2026-02-22  
Status: Ready to run (copy/paste into repo prompt)

## Objective

Audit the **TP v1.1 “generator-only workers”** plan as a **protocol/contract problem**:
- required fields (schema),
- versioning/rollout strategy,
- preflight-before-header (“anti-stranding”) rules,
- lockstep call-count plan,
- drift detection / determinism posture,
- and concrete break-it tests.

Goal output: a **ship checklist** + a **minimal envelope schema** that prevents hangs and Franken-models.

## Repo prompt pack (include these files)

### Scope notes (ground truth)

- `scope-drd/notes/FA4/h200/tp/v1.1-generator-only-workers.md`
- `scope-drd/notes/FA4/h200/tp/explainers/03-broadcast-envelope.md`
- `scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md`
- `scope-drd/notes/FA4/h200/tp/explainers/07-v0-contract.md`
- `scope-drd/notes/FA4/h200/tp/5pro/10-v11-correctness-deadlock-audit/response.md`
- `scope-drd/notes/FA4/h200/tp/bringup-run-log.md` (only the v1.1 matrix + memory notes are needed)

### Prior 5 Pro history (calibration; treat as baseline)

These are “what 5 Pro already decided” in the v1.1 thread. Your output should explicitly note whether it is:
- reiterating an existing recommendation,
- refining it with a clearer schema/test,
- or overriding it (and why).

- `scope-drd/notes/FA4/h200/tp/5pro/08-v11-generator-only-workers/response.md`
- `scope-drd/notes/FA4/h200/tp/5pro/10-v11-correctness-deadlock-audit/response.md`
- `scope-drd/notes/FA4/h200/tp/5pro/11-v11-perf-roi-stop-go/response.md`
- `scope-drd/notes/FA4/h200/tp/5pro/12-v11-risk-ranked-execution-plan/response.md`

### Library operator manuals (the “ship checklists” we want to align to)

- `refs/v1.1-generator-only-workers-crosswalk.md`
- `refs/topics/20-message-framing-versioning.md`
- `refs/topics/02-deadlock-patterns.md`
- `refs/topics/03-graceful-shutdown.md`
- `refs/topics/04-determinism-across-ranks.md`
- `refs/topics/21-idempotency-and-replay.md`
- `refs/topics/23-vae-latency-chunking.md`
- `refs/topics/22-kv-cache-management.md` *(may still be stub; treat as “must write” if missing)*

### Minimal physics cards (only if needed for backing)

- `refs/resources/nccl-user-guide.md`
- `refs/resources/pytorch-cuda-semantics.md`
- `refs/resources/funcol-rfc-93173.md`
- `refs/resources/dynamo-deep-dive.md`

## Questions to answer (be opinionated; prioritize P0 correctness)

### 0) Delta vs prior 5 Pro v1.1 thread (08–12)

Summarize (briefly) what is unchanged vs changed relative to the 5 Pro history above.

Deliverable: a “delta list” with:
- `unchanged` (still true),
- `refined` (same idea, sharper schema/tripwire/test),
- `overridden` (what changed and why).

### 1) Envelope schema: what is required, when?

Propose a minimal **`tp_envelope_version=1`** schema for `tp_plan=v1_generator_only`:
- Required scalars/flags every chunk (e.g., `call_id`, `chunk_index`, `cache_epoch`, geometry, reset decisions, recompute decision, expected call count, seed policy).
- Required tensors every chunk (bringup-safe defaults), and which ones can be “only on update” later (with an epoch mechanism).
- Explicit forbidden patterns (e.g., ad-hoc in-block broadcasts; nested tensors in meta).

Deliverable: a table of fields:
- `field`, `type`, `required_when`, `why`, `fail-fast check location`.

### 2) Anti-stranding: what must be preflighted before sending the INFER header?

Define the exact “preflight-before-header” ordering for TP broadcast:
- picklability / stable-json encoding,
- dtype support mapping,
- tensor spec ordering determinism,
- tensor materialization (device/contiguous/cast),
- and where exceptions must be caught to guarantee “crash > hang”.

Deliverable: a short deterministic send/recv protocol checklist.

### 3) Lockstep: how do we prevent generator-call count mismatch?

Define a “phase plan” that rank0 broadcasts and workers must follow:
- `tp_do_kv_recompute`,
- `tp_num_denoise_steps`,
- `tp_expected_generator_calls`,
- cache reset bits (`tp_reset_*`).

Deliverable: the exact plan bits + where to assert them (rank0 preflight, worker runner, optional per-call all_gather).

### 4) Determinism posture: what do we pin vs detect?

We’re not aiming for bitwise determinism; we want **fast detection** of divergence:
- What must be env-parity-checked vs carried in-envelope?
- What should always be digest-checked in bringup? (and what sampling is safe later)
- RNG policy: broadcast latents/noise vs lockstep `torch.Generator` (bringup recommendation).

Deliverable: a recommended “bringup mode” and “later optimize mode” list.

### 5) Override lifetime / stale reuse hazards

Identify the highest-risk “stale override reuse” cases (override tensors persist in PipelineState across chunks). Recommend:
- clear-after-read behavior inside blocks,
- explicit `None` fields in envelope when not used,
- and tests.

Deliverable: top 3 hazards + fixes.

## Required break-it tests (2–4, minimal but high value)

Specify tests (or harness-level experiments) that intentionally violate the contract and prove:
- it **crashes** quickly,
- it does **not** hang for minutes,
- and it provides actionable logs.

Use the failure-mode taxonomy naming from the deadlock audit (FM-01/FM-02/…) where applicable, so tests map cleanly to known risks.

At minimum include:
1) meta serialization failure (anti-stranding),
2) dtype unsupported discovered late,
3) call-count mismatch,
4) per-rank flag mismatch gating a collective.

## Output format (so we can operationalize it)

Return:
- **P0 blockers** (must fix before v1.1c),
- **P1 recommendations**,
- a proposed **envelope schema v1** (table),
- a **tripwire checklist** (where/when to assert),
- and **break-it tests** (with expected failure signatures).

Make the output copy-pastable into the topic operator manuals (especially `refs/topics/20-message-framing-versioning.md`, `refs/topics/02-deadlock-patterns.md`, `refs/topics/04-determinism-across-ranks.md`, and `refs/topics/22-kv-cache-management.md`).
