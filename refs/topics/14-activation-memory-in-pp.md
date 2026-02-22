---
status: draft
---

# Topic 14: Activation memory in PP

Peak activation memory in PP depends on the schedule. GPipe stores activations for all micro-batches; 1F1B limits in-flight activations to `num_stages`. **Activation checkpointing** trades compute for memory by recomputing activations during backward.

## Resources

<!-- Resource IDs from manifest belonging to this topic -->
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| megatron-ptdp | Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM | medium | pending |
| zero-bubble-pp | Zero Bubble Pipeline Parallelism | high | condensed |
| pytorch-pipelining-api | PyTorch torch.distributed.pipelining API | high | condensed |

## Implementation context

PP0 bringup starts in R1 (recompute disabled) because the semantics-preserving recompute path is coupled to decoded pixels: steady-state `get_context_frames()` uses `decoded_frame_buffer[:, :1]` and a VAE re-encode. In the R0a plan, rank0 supplies `context_frames_override` in the envelope; the tensor is small (~0.26 MB/chunk at 320×576) but it creates a timing dependency that can constrain overlap. The PP pilot therefore treats `expected_generator_calls = (do_kv_recompute?1:0)+num_denoise_steps` as a deadlock tripwire on every chunk.

See: `refs/implementation-context.md` → Phase 2, `scope-drd/notes/FA4/h200/tp/pp-topology-pilot-plan.md` (recompute coupling + sizes), `scope-drd/notes/FA4/h200/tp/pp-next-steps.md` (expected_generator_calls), `scope-drd/notes/FA4/h200/tp/pp0-bringup-runbook.md` (per-chunk checklist).

Relevant Scope code:
- `scope-drd/src/scope/core/distributed/pp_contract.py` (`context_frames`, `do_kv_recompute`, `expected_generator_calls`)
- `scope-drd/src/scope/core/pipelines/krea_realtime_video/blocks/recompute_kv_cache.py` (why recompute forces decoded-frame dependency)

## Synthesis

<!-- To be filled during study -->

### Mental model

Pipeline parallelism (PP) is a deliberate trade: you add **concurrency** (more work in flight) to reduce idle “bubble” time, but you pay in **memory** because each in-flight item needs some state to remain live until downstream stages consume it.

- **Training framing (papers)**:
  - GPipe’s baseline is “stash activations for all micro-batches, then do backward,” which drives activation memory up with the micro-batch count; its mitigation is **re-materialization**: store only boundary activations and recompute inside each partition, with the paper’s peak-activation scaling dropping from $O(N\\times L)$ to $O\\big(N + \\frac{L}{K}\\times\\frac{N}{M}\\big)$. (See `gpipe` claim 4.)
  - PipeDream-family schedules (1F1B / flush variants) aim to keep a **steady state** without having to stash as many outstanding activations; PipeDream-2BW explicitly frames the benefit as bounding peak activation stash by **pipeline depth** rather than by “number of microbatches over which gradients are accumulated.” (See `pipedream-2bw` claim 5.)
  - Zero-bubble schedules push utilization toward a ceiling by increasing scheduling granularity; the key lesson for memory is that “less bubble” often requires **more concurrent in-flight state** unless you use a memory-aware variant (e.g., ZB-H1 or ZB-V designed to stay within the 1F1B memory envelope). (See `zero-bubble-pp` claim 2 and claim 9.)

- **Inference reframing (our case)**: there is no backward pass, so most “activation stash” disappears. What remains is “state that must persist while other work runs,” typically:
  1) **boundary activations / p2p tensors** for each in-flight item, and
  2) **persistent model state across time**, especially the temporal **KV cache** (plus any rolling-window buffers),
  3) **queue depth** (e.g., envelopes/results) that makes PP worthwhile in the first place.

For Scope’s PP0 design (Wan 2.1, 40 DiT blocks; rank0-out-of-mesh; `D_in=D_out=2` pilot), “activation memory” is less about backprop stashes and more about the concrete tensors you keep alive across the Stage 0 ↔ Stage 1 boundary (plus the KV-cache / context-window state that lives on Stage 1). (See `refs/implementation-context.md` → Phase 3.)

### Key concepts

- **Activation stash** (training term): activations saved during forward that must survive until backward consumes them. In PP, the schedule determines how many microbatches’ worth are simultaneously live. (See `pipedream-2bw` claim 5.)
- **Boundary activations**: the tensors communicated across stage boundaries (`send`/`recv`). These contribute to memory on both sides of the boundary while in flight, even in inference. (See `gpipe` claim 6; `pipedream-2bw` key technical details.)
- **Re-materialization / activation checkpointing**: store fewer activations and recompute them later, trading extra compute for lower peak memory. GPipe formalizes this as “store boundary activations, recompute partition internals,” and PipeDream-2BW models recompute overhead as a constant multiplier (paper suggests $c^{extra}=4/3$). (See `gpipe` claim 4; `pipedream-2bw` claim 6.)
- **Peak vs steady-state memory**: “peak” often occurs near warm-up/steady-state transition, when the pipeline is full *and* queues/buffers are at their maximum depth; steady-state can look stable while still having high-water peaks during fill/drain. (See `gpipe` claim 5; `zero-bubble-pp` claim 5.)
- **Pipeline depth effect**: deeper pipelines reduce per-stage compute and per-stage weight memory, but they typically increase (a) the number of stage boundaries and (b) the amount of concurrently live state needed to keep utilization high under your schedule/memory limit. (See `pipedream-2bw` claim 8; `zero-bubble-pp` claim 4.)
- **Stash vs recompute trade-off (inference translation)**: in inference, “checkpointing” shows up as deciding whether to **buffer** some intermediate representation (latents, decoded frames, context frames) or **recompute/rehydrate** it later (e.g., recompute-KV-cache paths), which can create coupling that reduces overlap if the recompute depends on Stage 0 outputs. (See `pipedream-2bw` claim 6; Scope PP Phase 3 notes on recompute coupling.)

### Cross-resource agreement / disagreement

**Agreement (transferable):**
- Activation memory is a first-order constraint on PP feasibility and schedule choice; modern papers treat memory limits as an explicit parameter in scheduling/placement decisions. (See `gpipe` claim 4; `pipedream-2bw` key technical details → Memory model; `zero-bubble-pp` claim 4.)
- The schedule changes what is simultaneously “in flight,” so you can’t talk about activation memory without specifying the schedule. (See `gpipe` claim 2; `pipedream-2bw` claim 5; `zero-bubble-pp` claim 1.)

**Disagreement / emphasis:**
- **GPipe** makes memory manageable primarily via **re-materialization** (store boundary activations, recompute partition internals) and uses micro-batching to amortize bubbles; it’s simple but can be memory-heavy without recompute. (See `gpipe` claim 4 and claim 5.)
- **PipeDream-2BW (PipeDream-Flush / 1F1B family)** reduces peak stash pressure via **steady-state scheduling**: peak activation stash is bounded by pipeline depth rather than the micro-batch count, and the paper provides closed-form throughput/memory models to reason about feasibility. (See `pipedream-2bw` claim 5 and key technical details.)
- **Zero-bubble PP** increases scheduling granularity (split backward into B/W) to fill bubbles; the core activation-memory lesson is that chasing “near-zero bubble” can consume more memory unless you use a memory-aware schedule (ZB-H1, ZB-V) that stays within the 1F1B envelope. (See `zero-bubble-pp` claim 2 and claim 9.)

**What’s training-only vs what transfers to inference:**
- Mostly training-only: gradient accumulation semantics, weight-versioning (2BW), and the literal B/W backward decomposition. (See `pipedream-2bw` claim 1–2; `zero-bubble-pp` claim 1.)
- Transfers cleanly: bounding in-flight state, treating memory as a scheduling constraint, and using “stash vs recompute” as a deliberate lever. These map directly to inference PP buffering decisions (queue depth, boundary tensors, KV-cache recompute policies). (See `pipedream-2bw` claim 5–6; `zero-bubble-pp` Actionables.)

### Practical checklist

Use this checklist before dialing up PP depth or in-flight queue depth on H200-class GPUs (80GB) for Wan 2.1 inference.

1. **Inventory the “must-live” tensors per in-flight item**:
   - For classic block-PP: boundary activation tensor(s) at the split (often `[B, tokens, hidden]` for transformers).
   - For Scope PP0: `PPEnvelopeV1`/`PPResultV1` tensors (e.g., `latents_in`, `conditioning_embeds`, optional `context_frames` for recompute). (See `refs/implementation-context.md` → Phase 3.)

2. **Pick target in-flight depth first, then check memory**:
   - More in-flight items improves utilization but increases memory; for two stages, `B=1` leaves one stage idle roughly half the time, so the pilot plan starts with double buffering (`D_in=D_out=2`). (See `gpipe` claim 7 and claim 5; `pipedream-2bw` Actionables.)

3. **Compute a conservative peak-memory estimate** (per device):
   - `weights + KV_cache + runtime_workspaces + (in_flight_boundary_tensors × multiplicity)`.
   - Remember multiplicity includes: sender + receiver residency, plus any extra copies from contiguity/padding.
   - If you use recompute instead of stash, account for compute overhead (rule-of-thumb $c^{extra}\\approx 4/3$). (See `pipedream-2bw` claim 6.)

4. **Decide stash-vs-recompute at each boundary explicitly**:
   - If a value is needed later (e.g., context frames for KV-cache recompute), decide whether to buffer it, recompute it, or change semantics to remove a dependency chain that would serialize stages. (See `pipedream-2bw` claim 6; Scope PP Phase 3 recompute coupling note.)

5. **Validate stage balance before increasing depth**:
   - PP throughput trends toward `1 / max(stage_time)`; if one stage carries “extra work” (decode, I/O, synchronization), increasing micro-batches may just increase memory without improving throughput. (See `gpipe` claim 7; `pipedream-2bw` claim 7.)

### Gotchas and failure modes

- **“No backward” ≠ “no activation memory”**: inference still allocates memory for (a) boundary tensors per in-flight item and (b) persistent KV cache / rolling-window state; queue depth multiplies boundary memory even if per-item compute is small.
- **OOM from too many in-flight envelopes/results**: if `D_in`/`D_out` (or `max_outstanding`) grows without a hard cap, you can OOM purely from buffering boundary tensors. (Scope PP0 explicitly uses bounded queues; see Phase 3 notes.)
- **KV cache dominates the budget**: activation stash can be small while KV cache is huge; treating “activation memory” as the only PP limiter is a common misread for streaming inference. (Connects to Topic 22 KV-cache management; see `pagedattention` / `streamdiffusionv2` when available.)
- **p2p tensors cost memory on both ends**: a boundary activation is live on the sender until the send completes and on the receiver until consumed; double-buffering multiplies this again. (See `pipedream-2bw` key technical details → comm terms; `gpipe` claim 6.)
- **Deeper ≠ always better**: deeper PP reduces per-stage compute but keeps boundary sizes similar, so you can become comm- or memory-bound quickly; then schedule tweaks (even “zero-bubble”) can’t rescue throughput. (See `pipedream-2bw` claim 8; `zero-bubble-pp` claim 4.)

### Experiments to run

These experiments are meant to quickly find the high-water mark and confirm whether you’re memory- or bubble-limited.

1. **In-flight sweep (B=1 vs 2 vs 4)**:
   - Measure throughput and peak memory as you vary queue depth / in-flight items.
   - Expect utilization to improve with B, but peak memory to rise roughly linearly with buffered boundary tensors. (See `gpipe` claim 5; `pipedream-2bw` claim 5.)

2. **Peak-memory timeline capture**:
   - Record a memory timeline during warm-up → steady state → hard cut reset; identify where the high-water mark occurs (often at “pipeline full”). (See `zero-bubble-pp` claim 5 on warm-up under memory limits.)

3. **Boundary-tensor accounting test**:
   - Instrument exact byte sizes of each boundary tensor (and count how many copies exist concurrently) to validate your peak-memory model; include sender+receiver and double buffering.

4. **Stash vs recompute A/B**:
   - For any persistent intermediate state (e.g., context frames for recompute), test buffering vs recomputing; track both memory headroom and throughput impact, using $c^{extra}$ as a starting expectation but validating empirically. (See `pipedream-2bw` claim 6.)

5. **Schedule sensitivity (when applicable)**:
   - If/when using `torch.distributed.pipelining`, compare schedule variants (GPipe vs 1F1B vs zero-bubble implementations) under the same memory budget to see whether you’re bubble-limited or single-stage-limited. (See `pytorch-pipelining-api` Core claims; `zero-bubble-pp` claim 7.)
