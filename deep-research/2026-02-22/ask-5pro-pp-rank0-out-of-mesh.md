# 5 Pro Deep Research Request — PP (Rank0-out-of-mesh) bringup: contracts, deadlocks, overlap proof

Date: 2026-02-22  
Status: Ready to run (copy/paste into repo prompt)

## Objective

Audit and strengthen the **PP pilot design** where:
- **rank0** does Phase A (envelope materialization) + Phase C (decode/output),
- **mesh ranks 1..N** do Phase B (generator-only) with TP collectives **inside `mesh_pg`**,
- and overlap is achieved via bounded queues/double buffering.

We want: a crisp **ship checklist**, a minimal PP envelope/result schema, and “break-it tests” that prove **crash > hang**.

## Repo prompt pack (include these files)

### Scope notes (ground truth)

- `scope-drd/notes/FA4/h200/tp/pp-topology-pilot-plan.md`
- `scope-drd/notes/FA4/h200/tp/pp-control-plane-pseudocode.md`
- `scope-drd/notes/FA4/h200/tp/pp-next-steps.md`
- `scope-drd/notes/FA4/h200/tp/pp0-bringup-runbook.md`
- `scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md`
- (optional) `scope-drd/notes/FA4/h200/tp/v1.1-generator-only-workers.md` (only for shared “plan + envelope” vocabulary)

### Prior 5 Pro history (calibration; treat as baseline)

- `scope-drd/notes/FA4/h200/tp/5pro/13-pp-execution-ready/response.md`
- (shared v1.1 deadlock taxonomy) `scope-drd/notes/FA4/h200/tp/5pro/10-v11-correctness-deadlock-audit/response.md`

### Library operator manuals (target packaging)

- `refs/topics/20-message-framing-versioning.md`
- `refs/topics/02-deadlock-patterns.md`
- `refs/topics/19-producer-consumer-backpressure.md`
- `refs/topics/03-graceful-shutdown.md`
- `refs/topics/21-idempotency-and-replay.md`

## Questions to answer (prioritize P0 correctness)

### 0) Delta vs prior 5 Pro PP readiness review (13)

Summarize (briefly) what is unchanged vs changed relative to the 5 Pro PP readiness thread.

Deliverable: a “delta list” with:
- `unchanged` (still true),
- `refined` (same idea, sharper schema/tripwire/test),
- `overridden` (what changed and why).

### 1) Minimal PP contract (schema) — what must be explicit?

Propose minimal `PPEnvelopeV1` / `PPResultV1` fields for PP0/PP1 bringup:
- identifiers: `call_id`, `chunk_index`, `cache_epoch`, version fields,
- phase plan: `do_kv_recompute`, `num_denoise_steps`, `expected_generator_calls`,
- explicit reset bits (rank0 decides; mesh must not infer),
- geometry fields required to avoid stale mask/shape state.

Deliverable: a schema table with:
- `field`, `type`, `required_when`, `why`, `where validated`.

### 2) Multi-group deadlocks: world_pg vs mesh_pg hazards

Enumerate the “instant deadlock” pitfalls for PP1+ (mesh TP degree > 1):
- wrong group handle used for a collective,
- rank0 accidentally participates in `mesh_pg`,
- per-rank conditional collectives,
- leader broadcasting before preflight,
- “partial protocol” stranding mid-message.

Deliverable: a **group usage rulebook**:
- what must run on `world_pg`,
- what must run on `mesh_pg`,
- what must never cross,
- and the minimal env-parity keys / startup asserts.

### 3) Anti-stranding protocol ordering (sender and leader)

Define the exact preflight ordering so:
- rank0 never strands the leader mid-envelope,
- and the leader never strands mesh ranks mid-`mesh_pg` broadcast.

Deliverable: a deterministic send/recv checklist for:
- rank0 → leader p2p,
- leader → mesh bcast,
- leader → rank0 result send.

### 4) Backpressure + overlap: prove it with metrics

Review the current OverlapScore definition and propose:
- minimal required instrumentation fields,
- where to log them (rank0 vs leader),
- and pass/fail gates that avoid clock-sync requirements.

Deliverable: “overlap proof recipe”:
- expected period relation (`≈ max(Stage0, Stage1)` vs `sum`),
- what `t_mesh_idle_ms` should look like,
- and what queue-depth signatures imply scheduling bugs.

### 5) Recompute coupling: how it changes overlap claims

Given the decoded-anchor coupling (R0a) described in the PP plan:
- identify where overlap collapses,
- recommend the bringup sequencing (R1 → R0a → R0),
- and specify contract fields required to keep Phase B call-count deterministic under each regime.

Deliverable: a short decision table: `{R1, R0a, R0} → contract requirements + expected overlap impact`.

## Required break-it tests (minimal but high-value)

Specify 4–6 intentional violations that must:
- crash quickly (no minutes-long hangs),
- produce actionable logs (call_id/epoch/group),
- and avoid stranding any mesh rank in a collective.

At minimum include:
1) wrong process group used for one collective (PP1+),
2) leader throws after receiving header but before bcast (anti-stranding),
3) rank0 throws after sending header but before tensors (anti-stranding),
4) call-count mismatch (`expected_generator_calls ± 1`),
5) stale-epoch result after hard cut (must drop by `cache_epoch` mismatch),
6) conditional collective behind a per-rank flag mismatch.

## Output format

Return:
- **P0 blockers**, **P1 recommendations**,
- minimal PP schema (table),
- group usage rulebook,
- overlap proof recipe (metrics + gates),
- break-it tests with expected failure signatures.

Use the failure-mode taxonomy naming (FM-01/FM-02/…) where applicable, so tests map cleanly to known risks.

Make the output copy-pastable into the topic operator manuals (especially `refs/topics/19-producer-consumer-backpressure.md`, `refs/topics/20-message-framing-versioning.md`, and `refs/topics/02-deadlock-patterns.md`).
