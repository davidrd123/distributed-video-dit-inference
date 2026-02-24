---
status: draft
---

# Topic 23: VAE latency and chunking for video

The VAE decoder is often the **latency bottleneck** in video generation pipelines. 3D VAEs compress both spatially and temporally (typical compression: 8x8x4), but decoding back to pixel space is expensive. **Tiled decoding** splits the latent spatially, **temporal chunking** with causal convolution caching enables frame-by-frame streaming decode.

## Resources

<!-- Resource IDs from manifest belonging to this topic -->
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| cogvideox-vae | AutoencoderKLCogVideoX (Diffusers) | medium | pending |
| lightx2v-vae | VAE System and Video Encoding (LightX2V) | medium | pending |
| seedance | Seedance 1.0 | medium | pending |
| improved-video-vae | Improved Video VAE for Latent Video Diffusion Model | low | link_only |

## Implementation context

Block profiling shows VAE decode is already a material slice of wall time in TP v0: **107ms/chunk (16.5%)** at 320×576 (Run 10b), and decode+recompute totals **33%** of measured GPU time. This motivates two parallel threads: v1.1 “generator-only workers” (avoid duplicated decode on worker ranks), and (if TTFF/latency is a priority) StreamDiffusionV2’s “Stream-VAE” idea (chunked 3D conv with cached features; reported ~30% of pipeline time). The parked async-decode-overlap plan estimates only a ~2–8% ceiling (~0.5 FPS) unless recompute is rare.

Update (PP1 / 2026-02-24): in PP1 server mode on SM120, rank0 VAE decode measured ~172ms/chunk and consumed nearly the entire “idle bubble,” so the decode∥mesh overlap lever (Approach E) was mechanically correct but delivered ~0% FPS gain. This reaffirms the operator rule: overlap is only valuable once decode is small enough (or the semantic coupling is changed); otherwise, decode optimization is the first throughput lever.

See: `refs/implementation-context.md` → Phase 2, `scope-drd/notes/FA4/h200/tp/bringup-run-log.md` (Run 10b), `scope-drd/notes/FA4/h200/tp/v1.1-generator-only-workers.md` (v1.1 rationale), `scope-drd/notes/FA4/h200/tp/streamdiffusion-v2-analysis-opus.md` (Stream-VAE), `scope-drd/notes/FA4/h200/tp/async-decode-overlap-scoping.md` (gain ceiling).
See also: `scope-drd/notes/FA4/h200/tp/landscape.md` (PP1 scoreboard and “overlap E is ~0 on SM120 because decode fills bubble”).

Relevant Scope code:
- `scope-drd/src/scope/core/pipelines/krea_realtime_video/modules/vae.py` and `scope-drd/src/scope/core/pipelines/krea_realtime_video/blocks/prepare_context_frames.py` (where decode/encode and context assembly happen today)
- `scope-drd/src/scope/server/frame_processor.py` (frame chunking / decoded frame buffer lifecycle that couples to recompute)

## Synthesis

### Mental model

Treat each chunk as three phases:

- **Phase A (rank0-only):** materialize/broadcast the generator inputs (“envelope”).
- **Phase B (TP lockstep):** generator + denoise work that includes collectives and must execute in identical order on all ranks.
- **Phase C (rank0-only):** VAE decode + postprocess/output.

Run 10b block profiling makes the sizing concrete: at 320×576, `decode=107ms/chunk (16.5%)` and `recompute_kv_cache=104ms/chunk (16.0%)`, so decode+recompute is **33%** of measured GPU time. (See `scope-drd/notes/FA4/h200/tp/bringup-run-log.md` Run 10b.)

Treat “VAE latency work” as three different levers with different win conditions:

- **Stop doing decode on worker ranks** (structural): reduces duplicated work and is a prerequisite for PP-style specialization.
- **Overlap rank0 decode with the next chunk’s denoise** (throughput win): capped by how much decode can fit under denoise, and often blocked by recompute semantics.
- **Make decode cheaper** (algorithmic/implementation): chunked 3D-conv caching, tiling; can affect TTFF and/or steady-state.

Decision playbook: do not guess. Measure decode at compile-ON target settings and only pursue overlap if decode is “big enough” and the recompute coupling is addressed. (See `scope-drd/notes/FA4/h200/tp/async-decode-overlap-scoping.md` and `scope-drd/notes/FA4/h200/tp/async-decode-overlap-impl-plan.md`.)

### Key concepts

- **Ground truth numbers (compile OFF, TP v0):** `denoise=435ms (66.8%)`, `decode=107ms (16.5%)`, `recompute_kv_cache=104ms (16.0%)` → `decode+recompute=211ms (33%)`. (See `scope-drd/notes/FA4/h200/tp/bringup-run-log.md` Run 10b.)
- **Decode is local and non-collective:** skipping decode on workers is safe for TP lockstep because decode does not participate in collectives, but the state-update portions around it cannot be skipped freely. (See `scope-drd/notes/FA4/h200/tp/async-decode-overlap-scoping.md`.)
- **The coupling that breaks naive overlap is decoded-anchor recompute:**
  - `prepare_context_frames` consumes decoded pixels to keep `decoded_frame_buffer` moving, and steady-state recompute semantics re-encode a decoded anchor frame to build `context_frames_override` for the next chunk.
  - Rank0 builds `context_frames_override` before starting the next chunk’s TP lockstep call, so deferring decode can gate the next chunk under R0a semantics. (See `scope-drd/notes/FA4/h200/tp/async-decode-overlap-scoping.md` and `scope-drd/notes/FA4/h200/tp/async-decode-overlap-impl-plan.md`.)
- **Lever 1: generator-only workers (stop doing decode on workers):**
  - v1.1’s Phase A/B/C split: rank0 precomputes non-TP work, all ranks run Phase B, rank0 alone runs Phase C decode; workers should skip VAE/text-enc duties under the v1 plan. (See `scope-drd/notes/FA4/h200/tp/v1.1-generator-only-workers.md`.)
  - Primary benefits: worker memory/startup and enabling future PP/topology changes; not automatically an FPS win if rank0 remains the bottleneck. (See `scope-drd/notes/FA4/h200/tp/v1.1-generator-only-workers.md`.)
- **Lever 2: async decode overlap (rank0 side-stream):**
  - Mechanism: launch VAE decode on a separate CUDA stream, immediately start next chunk’s Phase B, and drain the pending decode via a CUDA event; clone/detach latents to avoid races. (See `scope-drd/notes/FA4/h200/tp/async-decode-overlap-impl-plan.md`.)
  - Gate: the scoping doc recommends a go/no-go based on measured decode time at compile-ON settings (decode ≥ ~6ms/call) and calls out that R0a semantics tend to collapse overlap unless recompute is disabled/rare or semantics change is accepted. (See `scope-drd/notes/FA4/h200/tp/async-decode-overlap-scoping.md`.)
  - Ceiling estimate: with 3 frames per chunk at ~23 FPS, call period ≈130ms; hiding ~3–10ms yields only ~2–8% (~0.5 FPS best case). (See `scope-drd/notes/FA4/h200/tp/async-decode-overlap-scoping.md`.)
- **Lever 3: make decode cheaper (Stream-VAE, chunking, tiling):**
  - StreamDiffusionV2’s Stream-VAE: decode short temporal chunks and cache intermediate 3D conv features; their analysis note reports VAE as ~30% of runtime and highlights edge-rank VAE duties as a major imbalance driver. (See `scope-drd/notes/FA4/h200/tp/streamdiffusion-v2-analysis-opus.md`.)
  - Spatial tiling reduces peak memory and can reduce latency, but increases kernel count and can shift you into overhead-bound behavior; treat tiling as a measured trade-off, not a default.
- **CUDA stream + allocator correctness (required for overlap):** if you introduce side-stream decode, you must explicitly synchronize (`wait_stream`, `record_stream`, events) because the caching allocator can create hidden dependencies via memory reuse even when “there’s no data dependency”. (See `refs/resources/pytorch-cuda-semantics.md`.)

### Cross-resource agreement / disagreement

- **Agreement (Scope ↔ StreamDiffusionV2):** both treat VAE as a structurally important contributor and a source of edge-stage imbalance if not isolated or load-balanced (scope-drd/notes/FA4/h200/tp/streamdiffusion-v2-analysis-opus.md Figure 13; scope-drd/notes/FA4/h200/tp/v1.1-generator-only-workers.md).
- **Agreement (Scope overlap plan ↔ PyTorch semantics):** overlap designs are easy to get subtly wrong without explicit stream/event and allocator-lifetime discipline (refs/resources/pytorch-cuda-semantics.md; scope-drd/notes/FA4/h200/tp/async-decode-overlap-impl-plan.md).
- **Context sensitivity / apparent disagreement:** StreamDiffusionV2 reports VAE as ~30% of runtime, while our async-decode scoping expects decode to be single-digit ms per chunk under TP+compile and therefore has only a ~2–8% throughput ceiling for overlap (scope-drd/notes/FA4/h200/tp/streamdiffusion-v2-analysis-opus.md; scope-drd/notes/FA4/h200/tp/async-decode-overlap-scoping.md). Treat the “VAE share” as workload-, implementation-, and setting-dependent.
- **Operational conclusion for Scope:** prioritize structural changes (workers skip decode; Phase A/B/C contract) and only pursue async overlap behind the measured gate (scope-drd/notes/FA4/h200/tp/v1.1-generator-only-workers.md; scope-drd/notes/FA4/h200/tp/async-decode-overlap-scoping.md).

### Practical checklist

- **Measure at the right settings:** Run 10b is compile-OFF; async overlap is explicitly gated on decode time measured at compile-ON target settings (scope-drd/notes/FA4/h200/tp/bringup-run-log.md Run 10b; scope-drd/notes/FA4/h200/tp/async-decode-overlap-scoping.md).
- **Don’t change TP lockstep call counts:** any VAE optimization must not perturb Phase B generator call ordering across ranks. Worker-side “skip decode” is safe; skipping state-update blocks is not (scope-drd/notes/FA4/h200/tp/async-decode-overlap-scoping.md; scope-drd/notes/FA4/h200/tp/v1.1-generator-only-workers.md).
- **Start with structural wins:** enforce the v1 “generator-only workers” contract (complete envelope; crash-before-broadcast; fail-fast) so workers can skip decode and other rank0-only duties safely (scope-drd/notes/FA4/h200/tp/v1.1-generator-only-workers.md).
- **Async decode overlap correctness checklist (if enabled):**
  - Preconditions: workers must skip decode via the v1 plan, and v1.1a context override must be enabled; follow the documented flag and requirements (`SCOPE_TP_ASYNC_DECODE_OVERLAP=1`, requires `SCOPE_TP_V11_CONTEXT_FRAMES_OVERRIDE=1`) (scope-drd/notes/FA4/h200/tp/async-decode-overlap-impl-plan.md).
  - Keep chunk 0 and hard cuts synchronous to seed/reset buffers (scope-drd/notes/FA4/h200/tp/async-decode-overlap-impl-plan.md).
  - Launch decode on a separate CUDA stream, clone/detach latents, and use an event to drain without blocking the main stream (scope-drd/notes/FA4/h200/tp/async-decode-overlap-impl-plan.md).
  - Stream correctness: use `wait_stream`/`record_stream` patterns and assume allocator reuse can force sync even without explicit dependencies (refs/resources/pytorch-cuda-semantics.md).
- **Recompute coupling gate:** if you require semantics-preserving decoded-anchor recompute every chunk (R0a, default `SCOPE_KV_CACHE_RECOMPUTE_EVERY=1`), expect overlap to collapse unless you introduce targeted waits; if you remove waits, you are accepting a semantic change and must validate quality (scope-drd/notes/FA4/h200/tp/async-decode-overlap-scoping.md; scope-drd/notes/FA4/h200/tp/async-decode-overlap-impl-plan.md).
- **Allocator + peak memory checks (especially with overlap + buffering):**
  - Overlap increases live tensors (pending decoded output, cloned latents, queued outputs); track peak memory and fragmentation on long runs (refs/resources/pytorch-cuda-semantics.md).
  - If also experimenting with PP-style in-flight buffering (`D_in/D_out`), re-check memory peaks because multiple envelopes/results can be simultaneously resident (scope-drd/notes/FA4/h200/tp/pp-topology-pilot-plan.md).
- **Don’t “optimize” recompute by skipping it:** `SCOPE_KV_CACHE_RECOMPUTE_EVERY=2` produced visible quality glitches and no net FPS gain; treat recompute frequency as quality-sensitive (scope-drd/notes/FA4/h200/tp/bringup-run-log.md Run 11).

### Gotchas and failure modes

- **Hang vs silent quality drift:**
  - Hangs: distributed contract violations (mismatched Phase B generator-call counts or collective order). Any VAE change that perturbs Phase B across ranks is a deadlock risk; enforce crash-before-broadcast and explicit phase plans (scope-drd/notes/FA4/h200/tp/v1.1-generator-only-workers.md).
  - Silent drift: semantic changes in context construction (decoded-anchor staleness, anchor lag, stale buffers). If you relax R0a to recover overlap, explicitly validate output quality (scope-drd/notes/FA4/h200/tp/async-decode-overlap-scoping.md; scope-drd/notes/FA4/h200/tp/async-decode-overlap-impl-plan.md).
- **Async timing lies:** profiling without CUDA events or with unnecessary `synchronize()` can destroy overlap or misattribute time; use events for timing and keep the measurement method consistent (refs/resources/pytorch-cuda-semantics.md; scope-drd/notes/FA4/h200/tp/bringup-run-log.md Run 10b).
- **Side-stream contention:** overlap only helps if decode does not meaningfully steal resources from denoise; measure actual contention and net period change (scope-drd/notes/FA4/h200/tp/async-decode-overlap-impl-plan.md).
- **Reset edge cases:** hard cuts (`init_cache=True`) must flush/synchronize any pending decode and reset buffers; otherwise stale outputs can leak into the next epoch (scope-drd/notes/FA4/h200/tp/async-decode-overlap-impl-plan.md).

Triage flow (what to measure/log first):
- If FPS regresses: re-run block profiling at the same settings and check whether denoise slowed (contention) or decode grew (work increase) (scope-drd/notes/FA4/h200/tp/bringup-run-log.md Run 10b; scope-drd/notes/FA4/h200/tp/async-decode-overlap-scoping.md).
- If output glitches appear: compare semantics-preserving (R0a with waits) vs overlap-friendly (R1/R0) runs; suspect stale decoded anchors / buffer advancement first (scope-drd/notes/FA4/h200/tp/async-decode-overlap-scoping.md).
- If hangs occur: suspect Phase B divergence (call-count mismatch) rather than decode itself; validate fail-fast “crash > hang” preflight checks and parity keys (scope-drd/notes/FA4/h200/tp/v1.1-generator-only-workers.md).

### Experiments to run

- **Baseline block profile at target settings:** run block profiling with TP+compile ON at production resolution/chunking and compare to Run 10b’s compile-OFF profile to decide if decode is still material (scope-drd/notes/FA4/h200/tp/bringup-run-log.md Run 10b; scope-drd/notes/FA4/h200/tp/async-decode-overlap-scoping.md).
- **Skip-decode-on-workers sanity (structural test):** enable the v1 generator-only plan and confirm:
  - Phase B call counts/collective ordering are unchanged.
  - Worker memory/startup improves.
  - Baseline behavior is unchanged when flags are off (scope-drd/notes/FA4/h200/tp/v1.1-generator-only-workers.md).
- **Async decode overlap A/B (show the coupling):**
  - Run with `SCOPE_TP_ASYNC_DECODE_OVERLAP=1` under R0a semantics (recompute every chunk) and observe forced waits / overlap collapse.
  - Run with recompute disabled or rare (R1-style) and observe whether the period improves by the predicted small ceiling (scope-drd/notes/FA4/h200/tp/async-decode-overlap-scoping.md; scope-drd/notes/FA4/h200/tp/async-decode-overlap-impl-plan.md).
- **Resolution sweep:** sweep `height×width` and measure how decode ms and decode % scale with pixel count, and whether decode becomes dominant at higher resolutions (scope-drd/notes/FA4/h200/tp/bringup-run-log.md Run 10b).
- **Memory/fragmentation stress:** run 100+ chunks with overlap enabled and log peak memory/allocator behavior; repeat with increased “in-flight” buffering (if experimenting with PP queues) to catch fragmentation or clones pushing you into OOM (refs/resources/pytorch-cuda-semantics.md; scope-drd/notes/FA4/h200/tp/pp-topology-pilot-plan.md).
- **Make decode cheaper (separate from overlap):** prototype tiling or chunked decode changes and measure TTFF and steady FPS separately; validate boundary artifacts and temporal consistency (scope-drd/notes/FA4/h200/tp/streamdiffusion-v2-analysis-opus.md).
