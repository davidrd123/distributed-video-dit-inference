# 5 Pro Deep Research Request — torch.compile + distributed hardening (parity, graph breaks, tests)

Date: 2026-02-22  
Status: Ready to run (copy/paste into repo prompt)

## Objective

We have a working TP=2 + compile path (functional collectives removed ~160 graph breaks and restored ~24.5 FPS historically). Now we want a “ship checklist” for **compiled distributed regions**:
- what invariants must hold across ranks,
- what to pin in env parity vs carry in-envelope,
- what tests catch divergence early,
- and what patterns to avoid (graph breaks, guard churn, conditional collectives).

Goal output: an operator-manual checklist + a minimal regression test set we can run daily.

## Repo prompt pack (include these files)

### Scope notes (ground truth)

- `scope-drd/notes/FA4/h200/tp/session-state.md` (compile posture)
- `scope-drd/notes/FA4/h200/tp/runtime-checklist.md` (locked decisions + compile flags)
- `scope-drd/notes/FA4/h200/tp/research-program.md` (regression checks)
- `scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md` (Q6 compile divergence)
- `scope-drd/notes/FA4/h200/tp/bringup-run-log.md` (Run 8–12b story: graph breaks → fix)

### Library resources (physics)

- `refs/resources/funcol-rfc-93173.md` (functional collectives semantics)
- `refs/resources/dynamo-deep-dive.md` (guards, graph breaks)
- `refs/resources/ezyang-state-of-compile.md` (practical compile framing)

### Library topics (target operator manual)

- `refs/topics/09-dynamo-tracing.md`
- `refs/topics/11-functional-collectives.md`
- `refs/topics/12-compile-distributed-interaction.md`
- `refs/topics/02-deadlock-patterns.md`
- `refs/topics/04-determinism-across-ranks.md`

## Questions to answer

### 1) Parity invariants: what must match across ranks?

List the rank-parity invariants required for compiled distributed execution:
- shapes/dtypes/control flags,
- compile mode and compiler settings,
- backend selection (attention/KV-bias backend),
- optional features that change graph structure.

Deliverable: a “parity checklist” divided into:
- must be **env-parity-checked** at init,
- must be **broadcast in-envelope** each call,
- can be “best effort” later.

### 2) Graph breaks and guard churn: operational triage

Provide a triage flow:
- how to distinguish “graph breaks” vs “recompiles,”
- which `TORCH_LOGS` to enable first,
- what counts as an acceptable steady state (unique graphs / break count),
- and when to stop optimizing and instead change the program structure.

Deliverable: a short “diagnose compile regression” checklist.

### 3) Functional collectives boundaries: do/don’t rules

We want explicit rules for:
- eager vs compiled collective implementations,
- where `c10d.wait` belongs,
- what can’t appear in compiled regions (Work objects, ProcessGroups, Python side effects),
- and how to avoid introducing conditional collectives via debug features.

Deliverable: a “compiled distributed region” contract (for code review).

### 4) Warmup and capture: what to standardize?

Rank lockstep warmup exists and is required to avoid startup divergence.

Recommend:
- what warmup must do (fill caches, compile steady-state),
- how to ensure warmup calls are excluded from digest checks but still enforce lockstep,
- how CUDA graphs interact with NCCL and what to avoid if we experiment later.

Deliverable: a “warmup protocol” and “CUDA graph caveats” shortlist.

### 5) Minimal regression suite (“break-it tests”)

Specify a minimal daily suite that catches distributed compile regressions:
- graph-break count invariants (e.g., “mode_C graph_break_count=0” for micro-repro),
- parity-key mismatch must fail fast (not hang),
- deliberate divergence tests (shape change, flag mismatch) must either recompile deterministically or crash with a crisp error.

Deliverable: 6–10 tests/experiments (short), grouped by:
- parity,
- graph breaks,
- recompiles,
- deadlocks/hang hygiene.

## Output format

Return:
- P0 ship checklist,
- parity invariants table,
- triage flow,
- do/don’t rules for compiled distributed regions,
- warmup protocol,
- regression suite list (with expected failure signatures).

