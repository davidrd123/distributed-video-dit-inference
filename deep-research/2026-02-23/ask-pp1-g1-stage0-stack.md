# 5 Pro Deep Research Request — PP1 G1: rank0 stage0 stack design

Date: 2026-02-23
Status: Ready to run

## Objective

PP1 server integration is mechanically complete through Phase G0b (first
server-mode INFER round-trip, blank frames). The next milestone is **Phase G1:
usable streaming** — real WebRTC pixels from rank0, with dynamic prompt updates.

G1 requires rank0 to own a "stage0 stack" (VAE decode + text encoder at minimum).
Today rank0 has nothing — it sends synthetic zeros and the mesh encodes a fixed
prompt from an env var.

Goal output: a **concrete execution plan** for the G1 stage0 stack, covering
which components to load, how to structure the decode loop, and which prompt
routing path to take. We have proven reference patterns in `pp0_pilot.py` (A3
overlap, A4b Rank0Buffers) — we want to know how to adapt them for the server.

## Repo prompt pack

### Stage0 code (current state — G0b level)

- `scope-drd/src/scope/server/pp1_stage0.py` (PP1Stage0 class: shape handshake, envelope builder, result processing, G1 stubs for VAE + text encoder)
- `scope-drd/src/scope/server/pp1_mesh_leader.py` (mesh leader loop: envelope unpack, TP broadcast, result send. Currently uses `SCOPE_PP1_PROMPT` env var for mesh-side text encoding)

### Server integration points

- `scope-drd/src/scope/server/app.py` (lifespan: Stage0 init after shape handshake. Shutdown uses `stage0.pp.send_shutdown()`)
- `scope-drd/src/scope/server/frame_processor.py` (the PP1 path: `_pp1_stage0_enabled` gate → `stage0.build_envelope()` → `stage0.pp.send_infer()` → `stage0.pp.recv_result()` → blank frames output)

### Proven reference patterns (pp0_pilot.py)

- `scope-drd/scripts/pp0_pilot.py` — contains both patterns we want to adapt:
  - **A4b Rank0Buffers** (lines 592–762): VAE decode + re-encode on rank0, sliding context frame buffer, `_build_real_context_frames()` with VAE re-encode path
  - **A3 overlap** (lines 1016–1229): `_rank0_overlap_segment()` with comms thread, bounded queues (`send_q`/`recv_q`), CUDA stream separation, `torch.cuda.Event` synchronization. Decode happens on main thread while comms thread does PP send/recv on a separate CUDA stream.

### Pipeline component loading (for Q1)

- `scope-drd/src/scope/core/pipelines/krea_realtime_video/pipeline.py` (how VAE, text encoder, generator are loaded — lines 76-332. Components registered via `ComponentsManager`)
- `scope-drd/src/scope/core/pipelines/wan2_1/vae/wan.py` (`WanVAEWrapper` — `encode_to_latent()`, `decode_to_pixel()`, `clear_cache()`. Constructor takes `model_dir`, `model_name`, `vae_path`)
- `scope-drd/src/scope/core/pipelines/wan2_1/components.py` (`WanTextEncoderWrapper` — uses umt5_xxl, takes `model_name`, `text_encoder_path`, `tokenizer_path`)
- `scope-drd/src/scope/core/pipelines/wan2_1/blocks/text_conditioning.py` (TextConditioningBlock — the `tp_v11_conditioning_override` semantics. When True + override tensor present: skip local text encoder, use override directly. When False: run local text encoder from prompts string.)
- `scope-drd/src/scope/core/pipelines/wan2_1/blocks/embedding_blending.py` (EmbeddingBlendingBlock — handles prompt transitions, spatial blending, `conditioning_embeds_override` consumption)

### PP contract (for context)

- `scope-drd/src/scope/core/distributed/pp_contract.py` (PPEnvelopeV1 fields include `conditioning_embeds` tensor)

### Pipeline config

- `scope-drd/src/scope/core/pipelines/krea_realtime_video/model.yaml` (num_frame_per_block=3, kv_cache_num_frames=3, vae_spatial_downsample_factor=8)
- `scope-drd/src/scope/core/pipelines/schema.py` (`KreaRealtimeVideoConfig` — height=320, width=576 defaults)

## Background: where we are

**G0b is mechanically complete.** The server boots with 3 ranks, rank0 sends
PP envelopes, mesh leader unpacks and runs `tp_worker_infer`, rank0 receives
`PPResultV1` with real `latents_out`. Output is blank frames (no VAE decode).

**Current G0b prompt routing:** mesh leader hardcodes `tp_v11_conditioning_override=False`
and passes `prompts=SCOPE_PP1_PROMPT` (env var, default "a cinematic scene").
The mesh's own text encoder runs locally. This works but prompts can't change
at runtime.

**Proven patterns from pp0_pilot.py:**
- A4b: rank0 loads `WanVAEWrapper`, decodes `latents_out` to pixels, re-encodes
  first decoded frame for `context_frames_override`. Uses sliding buffers.
- A3: comms thread + bounded queues overlap PP send/recv with rank0 VAE decode.
  Uses separate CUDA streams + events for synchronization. Measured ~30% latency
  hiding on 2×H100.

**Machine:** 4×A100-SXM4-80GB. PP1 = rank0 (GPU0) + TP=2 mesh (GPU1, GPU2). GPU3 idle.

## Questions

### Q1) VAE-only loading on rank0: how and how much VRAM?

Components are loaded individually in `pipeline.py` (lines 76-332). Each has its
own wrapper class (`WanVAEWrapper`, `WanTextEncoderWrapper`, `WanDiffusionWrapper`).

Questions:
- Can we instantiate `WanVAEWrapper` standalone on rank0 without loading the full
  pipeline? The constructor takes `model_dir`/`model_name`/`vae_path` — are there
  any hidden dependencies on other components or pipeline state?
- What's the expected VRAM footprint for VAE-only on A100 in bf16? (The generator
  is ~28GB; VAE should be much smaller. But streaming decode/encode may allocate
  intermediate buffers.)
- `WanVAEWrapper` has streaming methods (`stream_encode`, `stream_decode`) and a
  cache (`clear_cache()`). For PP1 rank0, should we use streaming mode or the
  simpler `decode_to_pixel()`/`encode_to_latent()` methods?
- Should rank0 also load the text encoder at the same time (for Q2), or is it
  better to keep them as separate loading phases?

### Q2) Prompt routing: rank0 text encoder vs mesh-side encoding

Two paths for getting conditioning_embeds into the mesh:

**(a) Rank0 loads text encoder, builds conditioning_embeds, sends in PPEnvelopeV1.**
- Pros: PPEnvelopeV1 already has `conditioning_embeds` field. Mesh sets
  `tp_v11_conditioning_override=True` and uses the tensor directly (skips its own
  text encoder). Prompt changes are immediate — rank0 encodes on prompt update.
- Cons: rank0 needs ~3-5GB more VRAM for umt5_xxl. Adds latency to prompt updates
  (text encoding time). Must replicate embedding blending logic for transitions.
- This is the PPEnvelopeV1-native path.

**(b) Extend envelope to carry prompt string, let mesh encode.**
- Pros: rank0 stays lightweight. Mesh already has text encoder loaded. No
  embedding blending duplication.
- Cons: Requires contract change (PPEnvelopeV1 doesn't carry strings today).
  Mesh leader must run TextConditioningBlock, which means it needs pipeline state
  access beyond `tp_worker_infer()`. Prompt-to-conditioning latency happens on
  the critical mesh path (blocks inference).

**(c) Hybrid: mesh encodes on first prompt, rank0 caches conditioning_embeds
from PPResultV1 metadata, sends cached version on subsequent chunks.**
- Requires extending PPResultV1 to return `conditioning_embeds` after encoding.

Questions:
- Which path do you recommend for G1? The key constraint is: prompt changes must
  propagate to the mesh within 1-2 chunks, and we cannot add multi-second latency
  to the mesh critical path.
- If (a): how should we handle the EmbeddingBlendingBlock? For G1, transitions
  are disabled (`pp_disable_vace=True`, no `transition` in call_params). Is it
  safe to just call `text_encoder.encode(prompts)` directly without the blending
  block? What would we lose?
- If (b): what's the minimum contract change to PPEnvelopeV1? Just add a `prompts`
  string field? How does the mesh leader route it into the text conditioning block?
- For G2 (transitions, embedding blending): does the choice here constrain us, or
  can we switch paths later without major rework?

### Q3) Decode ∥ mesh inference: adapting A3 overlap for the server

The pp0_pilot A3 pattern uses a comms thread + bounded queues + CUDA stream
separation. In the server, `frame_processor.py` drives inference from a
`_generate()` loop that runs in a thread (not asyncio). The PP1 path is currently
synchronous: build envelope → send → recv → emit blank frames.

Questions:
- Should the server overlap pattern mirror A3 exactly (comms thread + bounded
  queues), or is there a simpler model? The server processes one chunk at a time
  (no multi-chunk prefetch like the pilot), so the overlap is just "decode chunk
  N-1 while mesh computes chunk N."
- Where does the overlap boundary live? Options:
  - (a) Inside `frame_processor._generate()`: after recv_result, immediately
    queue next send, then decode previous result. Requires restructuring the
    generate loop to be 1-chunk-ahead.
  - (b) PP1Stage0 owns the overlap internally: `stage0.submit_envelope()` is
    non-blocking, `stage0.get_result()` returns the previous chunk's result.
    frame_processor sees a simple API.
  - (c) Separate decode thread: frame_processor sends/recvs synchronously,
    but VAE decode runs in a background thread and posts frames to the output
    queue.
- What's the expected latency budget? Mesh inference is ~282ms (TP=2). If VAE
  decode is <282ms, it's fully hidden. What's VAE decode time for 3 frames at
  320×576 on A100?
- The A3 pattern uses `torch.cuda.Event` for stream synchronization. Is this
  necessary in the server, or can we use simpler thread synchronization (the
  comms thread and decode thread are on the same GPU)?

### Q4) `pipeline.prepare()` replacement for PP1 rank0

`FrameProcessor` calls `pipeline.prepare(**params)` to get a `Requirements`
object with `input_size` (chunk frame count). This gates video input buffering.
Under PP1, rank0 has no pipeline.

Today's PP1 path skips `prepare()` entirely (pipeline=None → requirements=None →
video input block skipped). This works for G0b (text-only, no video input).

For G1, rank0 still won't have video input (VACE disabled, no V2V). But the
`prepare()` call also has side effects in the pipeline (mode resolution,
parameter validation). Questions:
- For G1 (text-only, no video): is it safe to keep skipping `prepare()` entirely?
  Are there any side effects we need to replicate?
- For G2 (video/VACE): will we need a full `prepare()` equivalent? If so, should
  we build a lightweight `Stage0Requirements` now (just returns `input_size=3`
  from shapes), or defer until G2?
- Is there any `prepare()` logic that affects `call_params` values (height, width,
  denoising schedule) that rank0 should be computing instead of hardcoding?

## Output format

Return:
1. **VAE loading plan** (Q1) — exact code path, VRAM estimate, streaming vs batch mode recommendation.
2. **Prompt routing recommendation** (Q2) — pick (a), (b), or (c) with justification. Pseudocode for the chosen path.
3. **Overlap architecture** (Q3) — pick an option, describe the threading model, estimate decode latency.
4. **prepare() plan** (Q4) — what to do for G1, what to plan for G2.
5. **Implementation ordering** — which of Q1-Q4 to do first, and what can be deferred.
6. **Red flags** — anything in the current G0b code or the proposed G1 approach that will cause problems.

Be concrete: file paths, line numbers, class names, estimated VRAM numbers.
