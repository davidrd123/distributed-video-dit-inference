# Study Agenda — Socratic Walkthroughs

> **Purpose**: Bridge the gap between "the library has operator manuals" and
> "I have the mental models in my head." Each session is a Socratic walkthrough
> of one topic, grounded in the drafted synthesis + real PP0 implementation
> code and bugs. The goal is internalization, not coverage.

Status: active
Created: 2026-02-22
Updated: 2026-02-22 (grounding refreshed against PP0 bringup A0–A2)

---

## Key grounding resources (across all sessions)

These are the "reality anchors" — the study sessions constantly refer back to them:

- **PP0 implementation**: `scope-drd/src/scope/core/distributed/pp_contract.py` (PPEnvelopeV1/PPResultV1), `pp_control.py` (PPControlPlane + anti-stranding preflight)
- **PP0 pilot harness**: `scope-drd/scripts/pp0_pilot.py` (3 worker modes, A2 gate validation)
- **Operator matrix tests**: `scope-drd/scripts/operator_matrix/om_01_*.py`, `om_02_*.py` (validated), `refs/operator-test-matrix.md` (full matrix)
- **Implementation feedback**: `refs/implementation-feedback.md` — gaps found during actual PP0 bringup (tribal knowledge, resource gaps, topic sharpening needs)
- **Bringup run log**: `scope-drd/notes/FA4/h200/tp/bringup-run-log.md`

## How each session works

1. **Pre-read** (~10 min): Skim the listed topic file's Mental Model + Key Concepts sections. Don't memorize — just get the shape.
2. **Socratic walkthrough** (~20-30 min): I ask you to explain the core invariant back to me. We work through "what breaks and why" together, connecting to real PP0 code and bugs.
3. **Grounding check**: We look at real scope-drd code, a real bug, or a real measurement that makes the concept concrete.
4. **Checkpoint**: What's clear, what's still fuzzy, what changes how you think about the next implementation step.
5. **Approve or revise**: If the topic synthesis holds up under your scrutiny, mark it `approved`. If it's missing something (the implementation feedback doc flags specific gaps), we fix it.

---

## Session sequence

Ordered by dependency — each session builds on the previous ones.

### Session 1: Deadlock patterns (Topic 02)
**Why first**: This is the foundational failure model. Everything else (framing, cache lifecycle, shutdown) is ultimately about preventing one of the three deadlock root causes.

- **Pre-read**: `refs/topics/02-deadlock-patterns.md` — Mental model + Key concepts
- **Grounding (design)**: `scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md` (the Seven Questions)
- **Grounding (code)**: `scope-drd/scripts/operator_matrix/om_01_*.py` — OM-01 test proves "crash > hang" for unpicklable meta. Walk through how the preflight in `pp_control.py` prevents stranding.
- **Grounding (real bug)**: `refs/implementation-feedback.md` §2b — the `sinusoidal_embedding_1d` device mismatch. Not a deadlock, but shows how ranks silently diverge when assumptions break.
- **Key question**: Can you name the three root causes of hangs and explain why "crash > hang" follows from them?
- **OM tests to review**: OM-01 (validated), OM-04 (partial), OM-05 (not yet — conditional collectives)
- **Status**: pending

### Session 2: Functional collectives + compile (Topics 11 + 12)
**Why second**: Your single biggest perf breakthrough (9.6 → 24.5 FPS) came from funcol. Understanding *why* it worked — and what constraints it imposes — is the physics behind the "no in-place collective" rules in the operator manuals.

- **Pre-read**: `refs/topics/11-functional-collectives.md` — Mental model; `refs/resources/funcol-rfc-93173.md` — Core claims 1-3
- **Grounding (measurement)**: `scope-drd/notes/FA4/h200/tp/bringup-run-log.md` — Run 10 → Run 12b transition (9.6 → 24.5 FPS)
- **Grounding (code)**: Topic 12 now has "effective-backend parity and symmetric compile-health gate" — how does the compile health gate prevent rank-asymmetric compilation?
- **Key question**: Why did switching to functional collectives fix compile? What did the old c10d ops do that broke tracing?
- **OM tests to review**: C-01 through C-04 (compile regression tests in operator matrix)
- **Status**: pending

### Session 3: Message framing + envelope contract (Topic 20)
**Why third**: This is the boundary you're *already building* (PPEnvelopeV1 is implemented). The deadlock mental model from Session 1 + the "what can be in a compiled graph" intuition from Session 2 both feed directly into envelope design choices.

- **Pre-read**: `refs/topics/20-message-framing-versioning.md` — Mental model + Practical checklist
- **Grounding (TP side)**: `scope-drd/notes/FA4/h200/tp/explainers/03-broadcast-envelope.md`
- **Grounding (PP side — real code)**: `scope-drd/src/scope/core/distributed/pp_contract.py` — walk through the actual PPEnvelopeV1 fields, `validate_before_send()`, and `tensor_specs()`. Then `pp_control.py` `_preflight_meta_and_tensors()` — this is the anti-stranding rule implemented.
- **Known gap** (`refs/implementation-feedback.md` §4c): result framing (mesh→rank0) lacks the same anti-stranding treatment as envelope framing. Worth discussing during the session.
- **Key question**: What's the anti-stranding rule, and why does "validate before header" prevent the worst class of PP deadlock?
- **OM tests to review**: OM-01 (validated), OM-02 (validated), OM-03 (plan dispatch — not yet), OM-06/07 (PP1)
- **Status**: pending

### Session 4: KV cache as distributed state machine (Topic 22)
**Why fourth**: The coupling between cache lifecycle and distributed correctness is the main thing v1.1 is trying to reduce. This session connects the deadlock model (Session 1) and the envelope contract (Session 3) to the actual state that lives on each GPU.

- **Pre-read**: `refs/topics/22-kv-cache-management.md` — Mental model + Key concepts (now has expanded lifecycle state machine with provenance citations)
- **Grounding (design)**: `scope-drd/notes/FA4/h200/tp/explainers/05-kv-cache-head-sharding.md`
- **Grounding (real bug)**: `scope-drd` commit `94bd266` — KV cache head-split used `world_size` instead of `tp_degree`. Subtle: works at TP=2 on 2 GPUs, breaks when PP adds a third rank.
- **Known gap** (`refs/implementation-feedback.md` §4d): PP recompute coupling — `context_frames_override` creates a data dependency between stages that breaks overlap. The R1→R0a→R0 progression manages this. Topic 22 doesn't explain this yet.
- **Key question**: Why is cache epoch part of the distributed contract (not just a local optimization)?
- **OM tests to review**: OM-09 (epoch fence), OM-10 (missing override), OM-11 (drift — partially validated via sinusoidal_embedding bug)
- **Status**: pending

### Session 5: Backpressure + shutdown (Topics 19 + 03)
**Why fifth**: With the contract boundary (Session 3) and state machine (Session 4) in place, this session covers what happens at the edges — queue depth, flow control, and tearing everything down without stranding ranks. PP0 A3 (overlap) is the next implementation phase, so this session is directly pre-read for building it.

- **Pre-read**: `refs/topics/19-producer-consumer-backpressure.md` — Mental model; `refs/topics/03-graceful-shutdown.md` — Practical checklist
- **Grounding (design)**: `scope-drd/notes/FA4/h200/tp/pp-next-steps.md` (D_IN/D_OUT queue design), `scope-drd/notes/FA4/h200/tp/pp0-a3-overlap-acceptance.md` (O-01/O-02/O-03 acceptance criteria)
- **Grounding (code)**: `scope-drd/scripts/pp0_pilot.py` — the current rank0 loop is strict send→recv serial. `--simulate-decode-ms` scaffolding is in place but the overlap scheduler (bounded queues, double-buffering) is design-only. Walk through what needs to change.
- **Known gaps** (`refs/implementation-feedback.md` §4a, §4b): Topic 03 needs PP shutdown protocol (SHUTDOWN envelope, rank0 crash without it). Topic 19 needs concrete D_in/D_out design + epoch-flush interaction.
- **Key question**: What's the difference between backpressure as "flow control" and backpressure as "correctness" in your PP design?
- **OM tests to review**: O-01 (D sweep), O-02 (overlap signature), O-03 (recompute coupling delta), OM-13 (orphan shutdown)
- **Status**: pending

### Session 6: Determinism + drift detection (Topic 04)
**Why last**: Drift is the sneaky failure mode — everything "works" but outputs are wrong. This session ties together all the invariants from Sessions 1-5 and asks: how do you *prove* ranks are still in agreement?

- **Pre-read**: `refs/topics/04-determinism-across-ranks.md` — Mental model + Experiments to run
- **Grounding (design)**: `scope-drd/notes/FA4/h200/tp/5pro/10-v11-correctness-deadlock-audit/response.md` (FM-10 through FM-13)
- **Grounding (real bug)**: The `sinusoidal_embedding_1d` device mismatch — `torch.cuda.current_device()` returns 0 on all ranks without `set_device()`. Both ranks "compute" but on different devices, producing wrong outputs. Caught by accident, not by a tripwire. (`refs/implementation-feedback.md` §2b)
- **Grounding (code)**: PPResultV1 already has `output_digest_i64` field — how would you use this to detect drift? What would a meaningful digest look like?
- **Key question**: What's the difference between "hang" and "Franken-model," and which is harder to detect?
- **OM tests to review**: OM-11 (drift digest — partial), OM-12 (weight fingerprint — not yet)
- **Status**: pending

---

## After the walkthroughs

Once all 6 sessions are done, you'll have internalized the mental models behind the operator manuals. At that point:
- The 5Pro deep-research outputs (all 5 received, 2 integrated) become *refinements* to things you already understand
- Topic approval passes will go fast because you'll catch what's wrong — the implementation feedback doc (§4) flags specific gaps per topic
- The remaining OM tests (OM-03 through OM-13) become implementation tasks you can spec and review confidently
- PP0 A3 overlap design (which is next) maps directly to Sessions 3+5

## Bonus: training-PP vs inference-PP mental model

The implementation feedback (`refs/implementation-feedback.md` §1) identified a conceptual gap: Topics 13-15 and the PP papers are training-oriented, but our use case is structurally different. Key mapping:

| Training PP | Streaming inference PP (our case) |
|---|---|
| Multiple micro-batches fill the bubble | Single chunk in flight (B=1) |
| Backward pass → F/B/W scheduling | No backward → no 1F1B, no zero-bubble |
| Activation checkpointing trades memory for compute | KV cache recompute trades freshness for overlap |
| Bubble fraction = `(P-1)/M` | Stage imbalance = `max(0, stage0 - stage1)` |

This isn't a separate session — it's context to keep in mind when the PP papers come up during Sessions 1 and 5.

---

## Topics NOT in this sequence (and why)

All 25 topics are now drafted. These are valuable but not on the critical path for v1.1 + PP understanding:

| Topic | Why deferred | When it becomes relevant |
|---|---|---|
| 01 (NCCL internals) | Physics reference — consult when debugging | When you need to tune NCCL_ALGO/PROTO or diagnose a transport-level failure |
| 05 (CUDA streams) | Important for overlap work | **Candidate for promotion**: A3 overlap needs stream reasoning. Consider adding as Session 5a pre-read if overlap bringup starts before we reach Session 5. |
| 06 (CUDA graphs) | Graph capture + distributed has sharp edges | PP1 + compile, not PP0 |
| 09 (Dynamo tracing) | Covered enough in Session 2 via funcol | If graph breaks bite during compile hardening |
| 10 (Inductor fusion) | Now drafted; useful for perf analysis | When interpreting Triton kernel fusion decisions |
| 13-15 (PP papers) | Resource cards exist; see training-vs-inference mapping above | Skim as needed; the mapping table above prevents misapplication |
| 16-18 (perf analysis) | All now drafted; measurement topics | When profiling A3 overlap or diagnosing bandwidth bottlenecks |
| 23 (VAE latency) | Becomes relevant at VAE overlap gate | When DeDiVAE or streaming VAE decode enters scope |
| 24 (video DiT scheduling) | Now drafted; covers StreamV2V mapping | When block-PP or B>1 scheduling enters scope |
