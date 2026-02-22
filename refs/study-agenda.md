# Study Agenda — Socratic Walkthroughs

> **Purpose**: Bridge the gap between "the library has operator manuals" and
> "I have the mental models in my head." Each session is a Socratic walkthrough
> of one topic, grounded in the drafted synthesis + your actual scope-drd
> measurements and design notes. The goal is internalization, not coverage.

Status: active
Created: 2026-02-22

---

## How each session works

1. **Pre-read** (~10 min): Skim the listed topic file's Mental Model + Key Concepts sections. Don't memorize — just get the shape.
2. **Socratic walkthrough** (~20-30 min): I ask you to explain the core invariant back to me. We work through "what breaks and why" together, connecting to your actual architecture.
3. **Grounding check**: We look at a specific scope-drd note or measurement that makes the concept concrete (not abstract).
4. **Checkpoint**: What's clear, what's still fuzzy, what changes how you think about the next implementation step.
5. **Approve or revise**: If the topic synthesis holds up under your scrutiny, mark it `approved`. If it's missing something you noticed, we fix it.

---

## Session sequence

Ordered by dependency — each session builds on the previous ones.

### Session 1: Deadlock patterns (Topic 02)
**Why first**: This is the foundational failure model. Everything else (framing, cache lifecycle, shutdown) is ultimately about preventing one of the three deadlock root causes.

- **Pre-read**: `refs/topics/02-deadlock-patterns.md` — Mental model + Key concepts
- **Grounding**: `scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md` (the Seven Questions)
- **Key question**: Can you name the three root causes of hangs and explain why "crash > hang" follows from them?
- **Status**: pending

### Session 2: Functional collectives + compile (Topics 11 + 12)
**Why second**: Your single biggest perf breakthrough (9.6 → 24.5 FPS) came from funcol. Understanding *why* it worked — and what constraints it imposes — is the physics behind the "no in-place collective" rules in the operator manuals.

- **Pre-read**: `refs/topics/11-functional-collectives.md` — Mental model; `refs/resources/funcol-rfc-93173.md` — Core claims 1-3
- **Grounding**: `scope-drd/notes/FA4/h200/tp/bringup-run-log.md` — Run 10 → Run 12b transition
- **Key question**: Why did switching to functional collectives fix compile? What did the old c10d ops do that broke tracing?
- **Status**: pending

### Session 3: Message framing + envelope contract (Topic 20)
**Why third**: This is the boundary you're designing (PPEnvelopeV1). The deadlock mental model from Session 1 + the "what can be in a compiled graph" intuition from Session 2 both feed directly into envelope design choices.

- **Pre-read**: `refs/topics/20-message-framing-versioning.md` — Mental model + Practical checklist
- **Grounding**: `scope-drd/notes/FA4/h200/tp/explainers/03-broadcast-envelope.md` (current TP envelope shape)
- **Key question**: What's the anti-stranding rule, and why does "validate before header" prevent the worst class of PP deadlock?
- **Status**: pending

### Session 4: KV cache as distributed state machine (Topic 22)
**Why fourth**: The coupling between cache lifecycle and distributed correctness is the main thing v1.1 is trying to reduce. This session connects the deadlock model (Session 1) and the envelope contract (Session 3) to the actual state that lives on each GPU.

- **Pre-read**: `refs/topics/22-kv-cache-management.md` — Mental model + Key concepts
- **Grounding**: `scope-drd/notes/FA4/h200/tp/explainers/05-kv-cache-head-sharding.md`
- **Key question**: Why is cache epoch part of the distributed contract (not just a local optimization)?
- **Status**: pending

### Session 5: Backpressure + shutdown (Topics 19 + 03)
**Why fifth**: With the contract boundary (Session 3) and state machine (Session 4) in place, this session covers what happens at the edges — queue depth, flow control, and tearing everything down without stranding ranks.

- **Pre-read**: `refs/topics/19-producer-consumer-backpressure.md` — Mental model; `refs/topics/03-graceful-shutdown.md` — Practical checklist
- **Grounding**: `scope-drd/notes/FA4/h200/tp/pp-next-steps.md` (D_IN/D_OUT queue design, A1/A2 timeouts)
- **Key question**: What's the difference between backpressure as "flow control" and backpressure as "correctness" in your PP design?
- **Status**: pending

### Session 6: Determinism + drift detection (Topic 04)
**Why last**: Drift is the sneaky failure mode — everything "works" but outputs are wrong. This session ties together all the invariants from Sessions 1-5 and asks: how do you *prove* ranks are still in agreement?

- **Pre-read**: `refs/topics/04-determinism-across-ranks.md` — Mental model + Experiments to run
- **Grounding**: `scope-drd/notes/FA4/h200/tp/5pro/10-v11-correctness-deadlock-audit/response.md` (FM-10 through FM-13)
- **Key question**: What's the difference between "hang" and "Franken-model," and which is harder to detect?
- **Status**: pending

---

## After the walkthroughs

Once all 6 sessions are done, you'll have internalized the mental models behind the operator manuals. At that point:
- The 5Pro deep-research outputs will land as *refinements* to things you already understand, not new information to absorb
- The "3 gaps" (contract tables, compiled-distributed checklist, break-it suite) become formatting exercises you can direct confidently
- Topic approval passes will go fast because you'll catch what's wrong, not just accept what's written

---

## Topics NOT in this sequence (and why)

These are valuable but not on the critical path for v1.1 + PP understanding:

| Topic | Why deferred |
|---|---|
| 01 (NCCL internals) | Physics reference — consult when debugging, not for mental model building |
| 05 (CUDA streams) | Important for overlap work, but PP0 doesn't need deep stream reasoning yet |
| 06 (CUDA graphs) | Same — becomes critical at PP1 + compile, not PP0 |
| 09 (Dynamo tracing) | Covered enough in Session 2 via funcol; deep dive later if graph breaks bite |
| 13-15 (PP papers) | The resource cards exist; skim as needed during PP0 bringup |
| 16-18 (perf analysis) | Measurement topics — use when profiling, not for conceptual grounding |
| 23 (VAE latency) | Becomes relevant when you hit the VAE overlap gate, not before |
| 24 (video DiT scheduling) | StreamDiffusionV2 card exists; revisit when block-PP enters scope |
