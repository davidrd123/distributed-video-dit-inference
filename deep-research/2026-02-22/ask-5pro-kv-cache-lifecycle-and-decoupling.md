# 5 Pro Deep Research Request — KV-cache lifecycle as a lockstep state machine (TP + PP coupling)

Date: 2026-02-22  
Status: Ready to run (copy/paste into repo prompt)

## Objective

Make KV-cache management “operator-manual simple” for TP and for PP follow-ons:
- define the cache lifecycle as an explicit **state machine** (reset/recompute/advance/evict),
- identify the highest-risk divergence points that cause hangs or Franken-models,
- and recommend decoupling patterns that avoid collapsing overlap (decoded-anchor recompute coupling).

Goal output: a **ship checklist** + a small set of **contract fields + tests** that guarantee deterministic behavior.

## Repo prompt pack (include these files)

### Scope notes (ground truth)

- `scope-drd/notes/FA4/h200/tp/explainers/05-kv-cache-head-sharding.md`
- `scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md` (Q4 + cache drift framing)
- `scope-drd/notes/FA4/h200/tp/v1.1-generator-only-workers.md` (why workers run full pipeline; recompute overrides)
- `scope-drd/notes/FA4/h200/tp/pp-topology-pilot-plan.md` (R0/R0a/R1 recompute coupling section)
- `scope-drd/notes/FA4/h200/tp/pp0-bringup-runbook.md` (PP0/R0a step)
- `scope-drd/notes/FA4/h200/tp/bringup-run-log.md` (Run 11 recompute frequency experiment)

### Library (target operator manual)

- `refs/topics/22-kv-cache-management.md` *(may still be stub; treat as the output target)*
- `refs/topics/04-determinism-across-ranks.md` (drift detection)
- `refs/topics/20-message-framing-versioning.md` (contract fields/epochs/IDs)
- `refs/topics/21-idempotency-and-replay.md` (epochs + drop semantics)

### Minimal physics cards (optional)

- `refs/resources/pagedattention.md` (memory-management patterns)
- `refs/resources/streamdiffusionv2.md` (streaming cache patterns; stage imbalance)
- `refs/resources/pytorch-cuda-semantics.md` (stream/allocator gotchas if overlap touches cache buffers)

## Questions to answer

### 1) State machine: define the lifecycle precisely

Write a minimal state machine for the cache lifecycle across chunks:
- inputs that drive transitions (`init_cache`, hard cut, prompt update, recompute schedule),
- state variables that must remain identical across ranks (`current_start_frame`, cache epochs, cache reset flags),
- and what “advance” means (what indices move, where writes land).

Deliverable: a diagram/table of states + transitions + invariants.

### 2) Divergence hazards (hang vs Franken-model)

List the top divergence hazards and classify each as:
- likely **NCCL hang** (call-count/collective order mismatch),
- likely **Franken-model** (collectives line up but cached content differs),
- or likely **quality drift only**.

Deliverable: top 8–12 hazards with “tripwire to catch it” suggestions.

### 3) Contract fields that must be explicit (TP v1.1 and PP)

Recommend the minimal contract bits that must be broadcast/declared to keep cache behavior deterministic:
- reset decisions (`tp_reset_kv_cache`, `tp_reset_crossattn_cache`, etc.),
- recompute decision per chunk (`do_kv_recompute`, `expected_generator_calls`),
- `cache_epoch` semantics,
- required override tensors (e.g. `context_frames_override` under R0a).

Deliverable: a table of fields: `field`, `why`, `fail-fast check`.

### 4) Decoupling recompute from decoded-anchor dependency

Our current steady-state coupling is: recompute wants a context tensor that is derived from `decoded_frame_buffer` (decoded pixels) via VAE re-encode (decoded-anchor).

Evaluate and recommend:
- R1 (disable recompute for bringup),
- R0a (rank0 provides semantics-preserving `context_frames_override`),
- R0 (semantic change: latent-only anchor).

Deliverable:
- a bringup sequencing recommendation,
- expected overlap impact for each,
- and explicit “quality validation must run” triggers for semantic-change paths.

### 5) Break-it tests (cache semantics)

Specify the smallest tests that prove:
- cache reset decisions don’t drift,
- recompute call-count doesn’t drift,
- stale-epoch results get dropped (PP),
- and “skip recompute” doesn’t silently turn into mismatched plan.

Deliverable: 4–6 tests with expected failure signatures and logs.

## Output format

Return:
- state machine (states/transitions/invariants),
- P0 contract requirements,
- tripwire checklist (what to assert where),
- recommended recompute decoupling path (R1→R0a→R0 or alternative),
- and break-it tests.

