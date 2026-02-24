# 5 Pro Deep Research Request — Anchor-only decode, VAE temporal constraints, and remaining PP1 levers

Date: 2026-02-24
Status: Ready to run

## Context (what happened since the last ask)

We've completed a comprehensive performance exploration of PP1 (phase parallelism)
on 8×SM120 GPUs. Every easy lever has been pulled. Here's the scorecard:

| Run | Config | FPS | Verdict |
|-----|--------|-----|---------|
| 30 | PP1+TP=4 (baseline, overlap off) | 11.2 | C2 context parity confirmed |
| 31 | PP1+TP=4, Approach E overlap | 11.2 | Correct but ~0 gain (VAE decode=172ms fills 185ms bubble) |
| 32 | PP1+TP=4, SM120 CuTe FA kernel | 11.2 | Kernel works, segment-combine too small |
| 33 | Pure TP=8 (no PP) | 10.0 | **TP scaling collapsed** — PP1 wins with fewer GPUs |
| 34 | PP1+TP=4, **compile** | **12.1** | 9% mesh speedup (885→805ms) — ceiling without arch change |
| 35 | PP1+TP=5 | N/A | **Impossible** — ffn_dim=13824 not divisible by 5 |
| 36 | **Single GPU** (compile, no PP, no TP) | ~11.0 | PP1+TP=4+compile only 10% faster on 5× GPUs |
| 37 | PP1+TP=4, compile, **R0 latent anchor** | **14.8** | 22% gain! But **quality glitchy** — rejected |

**The binding constraint is clear**: the decode→re-encode anchor dependency serializes
the pipeline. Period = sum(stage0, stage1) = 805 + 185 = 990ms, not max = 805ms.

**R0 (latent-only anchor) proved the thesis**: removing decode from the critical
path gives 14.8 FPS (period ≈ stage1). But quality is unacceptable — the VAE
reconstruction acts as a regularizer that grounds the KV cache in pixel-space.
Latent-space drift causes visible artifacts.

**Next: anchor-only decode** — decode only T=1 latent frame (~57ms estimated) for
the re-encode anchor, submit envelope, then full display decode overlapped with mesh.
Projected ~14.9 FPS with strict quality parity. This is the only untested lever that
could meaningfully change FPS.

## What we need from you (three research threads)

### Thread 1: VAE temporal convolution constraints — can we decode T=1?

Our VAE is Wan2.1's WanVAE (3D video VAE with temporal convolutions).
The standard call is:

```python
# Full chunk in our realtime config: 3 latent frames → 12 pixel frames steady-state
# (chunk 0 is 9 pixel frames). latents_out: [1, 3, 16, H_lat, W_lat]
pixels = vae.decode(latents)
```

We want to do:
```python
# Anchor-only: 1 latent frame → 4 pixel frames
anchor_pixels = vae.decode(latents[:, :1])
```

**Questions:**
1. Do Wan2.1's temporal convolutions (CausalConv3d / temporal upsampling) handle T=1
   input correctly? Or do they require minimum T>1 due to padding/kernel size assumptions?

2. If T=1 decode produces 4 pixel frames, is `pixel_frames[-1]` (the last of 4)
   semantically equivalent to the *boundary* pixel frame used by the full streaming decode?
   (In the reference pipeline, the anchor comes from `decoded_frame_buffer[:, :1]` after
   sliding to max=9 frames, so it tends to be a boundary pixel frame.)

3. Are there known workarounds if T=1 is unsupported? (E.g., decode T=2 and discard,
   or pad with zeros, or use the encoder's temporal receptive field to determine
   minimum T.)

4. **Search broadly**: Other video VAEs with temporal convolutions (Open-Sora,
   CogVideoX, LTX-Video, SVD) — how do they handle partial-temporal decode?
   Is there a standard pattern for "decode minimum temporal footprint"?

5. What is the **temporal receptive field** of Wan2.1's decoder? If each temporal
   conv has kernel_size=3 with causal padding, and there are N temporal layers,
   the minimum T for stable output is bounded. What's that bound?

6. **Streaming cache interaction:** our display decode uses streaming mode (`decode_to_pixel(..., use_cache=True)`
   → `stream_decode()`), which preserves internal conv caches across calls. If we run an extra
   anchor-only decode “in the middle”, how do we avoid corrupting the streaming cache?
   - Is `use_cache=False` for the anchor decode acceptable (batch decode), or does it change the
     anchor pixel enough to matter?
   - Is there a known pattern for “decode with explicit cache” (shadow cache) like our encoder’s
     `encode_to_latent_with_cache` helper?

7. **Which pixel frame is the correct anchor?** After `_update_decoded_buffer()`, the
   re-encode uses `decoded_frame_buffer[:, :1]`. With steady-state 12 decoded frames
   and a buffer max of 9, that anchor is the oldest of the retained 9 — NOT “frame 0
   of the current chunk.” Derive the exact mapping: which latent frame(s) must be
   decoded to produce that specific pixel, and can it be produced without full-chunk
   decode? (See `PrepareContextFramesBlock` in the prompt pack for the sliding-buffer
   semantics.)

8. **T=1 output length under streaming decode**: Our `decode_to_pixel()` calls
   `stream_decode()` which has `first_batch` semantics (chunk 0 returns 9 frames,
   steady-state returns 12 for T=3 input). What does T=1 return in `first_batch=True`
   vs `first_batch=False`? Is the output length even well-defined for T=1 under
   streaming decode?

### Thread 2: Why does torch.compile give 9% on SM120 vs 53% on H200?

We measured:
- H200 (SM90): TP=2 compile took 16→24.5 FPS (**53% gain**)
- SM120 (RTX PRO 6000 Blackwell Server Edition): PP1+TP=4 compile took 11.2→12.1 FPS (**9% gain**)

Known factors:
- SM120 uses SM80-era MMA (compute_capability 12.0 but Blackwell MMA instructions)
- Graph break at `_kv_bias_flash_combine` (`int(current_block_start)` → `Tensor.item()`)
- KV-bias flash attention falls back from CuTe to Triton under dynamo
  (`__dlpack__` on FakeTensor incompatible with CuTe DSL)
- SM120 may be more memory-bound at our tile sizes

**Note**: The comparison is not perfectly apples-to-apples — H200 result was TP=2 only
(no PP), SM120 is PP1+TP=4 (different topology, different attention backend constraints).
The question is whether the gap is explained by topology differences or by fundamental
SM120 compile effectiveness.

**Questions:**
1. What determines compile speedup variance across GPU architectures?
   Specifically: what makes Inductor fusions more effective on SM90 vs SM120?

2. Is the KV-bias graph break the dominant loss? If we fixed the `Tensor.item()` call
   (replaced with symbolic tracing or moved outside the compiled region), what's the
   expected additional gain?

3. Are there published benchmarks or analysis of torch.compile effectiveness across
   SM80/SM89/SM90/SM100 architectures for DiT-class models? What FPS gains do others
   report?

4. Inductor's cost model for kernel fusion — does it account for SM120's specific
   memory hierarchy and compute characteristics? Or is SM120 too new for the cost
   model to be tuned?

5. **Practical diagnostic**: We can run `TORCH_LOGS=graph_breaks` on SM120. What
   should we look for? Is there a methodology to compare "fusion opportunity" between
   two GPU architectures for the same model? Minimal diagnostic: run single-rank
   `profile_krea_pipeline_blocks.py --compile` with same KV-bias path and compare
   op mix / fusion between SM90 and SM120.

6. **CuTe↔dynamo compatibility pattern**: We lose the KV-bias flash attention path
   under compile because CuTe DSL uses `__dlpack__` which is incompatible with
   dynamo's FakeTensor tracing. What's the recommended pattern?
   - `torch._dynamo.disable` on the CuTe call site (preserves graph elsewhere)?
   - Register a custom op with a fake impl (`torch.library.custom_op`)?
   - Accept the graph break and Triton fallback?
   What net speedup should we expect on SM120 from fixing this vs accepting fallback?

### Thread 3: What else exists for hiding decode latency in pipeline-parallel video inference?

We've exhausted our reference library (CausVid, StreamDiffusionV2, PipeDiT,
Zero-Bubble-PP) and found no prior art for "T=1 anchor-only decode." We need
a broader search.

**Questions:**
1. Has anyone published work on **partial VAE decode** in the critical path of
   video generation pipelines? (Not tiling — temporal partitioning.)

2. Are there systems that use a **cheaper proxy for the VAE round trip**?
   E.g., a small learned network that approximates encode(decode(x)) without
   the full VAE, specifically for maintaining temporal coherence in rolling-window
   video generation.

3. **Latent-space anchoring without pixel round trip**: Our R0 test showed quality
   degradation from using raw latents as context. Is there published analysis of
   latent-space drift in rolling-window diffusion? Are there techniques (EMA on
   latents, periodic correction, learned projection) that make latent-only anchoring
   viable?

4. **Asynchronous VAE decode patterns**: Are there systems that decouple VAE decode
   from the generation loop entirely? E.g., a separate process/GPU doing decode
   with a queue, where the generator never waits for decode completion.

5. **Multi-frame latent caching**: Instead of sliding-window with re-encode, are
   there video DiT systems that maintain a latent cache directly (no decode→re-encode
   round trip) while preserving temporal coherence? What anchoring mechanism do they use?

6. **WanVAE-specific**: Any analysis of Wan2.1's VAE decode performance characteristics?
   Temporal upsampling cost vs spatial? Which layers dominate the 172ms?

7. **Hybrid anchor schedule (R0 + periodic correction)**: Our R0 test (latent-only
   anchor) gave 14.8 FPS but drifted visually. What if we use latent-only anchor
   most chunks and do a real decode→re-encode every N chunks as periodic correction?
   - How to pick N? (Drift rate vs correction cost trade-off.)
   - What cheap drift detectors/metrics could gate when correction is needed?
     (L2 norm divergence, LPIPS on decoded frames, latent histogram shift, etc.)
   - Has anyone published on periodic correction schedules for rolling-window
     diffusion with KV cache?

### Thread 4 (optional): Throughput levers beyond single-stream FPS (multi-stream packing)

PP1’s biggest practical advantage on this 8×SM120 box is that **TP=4 uses 5 GPUs** (rank0 + 4 mesh),
leaving **3 idle GPUs**. If single-stream gains plateau, our next lever is B>1 / multi-stream packing.

Questions:
1. What’s the best “first ship” way to use the idle GPUs?
   - Two independent processes (each TP=2) vs one rank0 serving multiple meshes.
2. If sharing rank0 across streams, what should be shared vs isolated (VAE decode, text encoder, session state)?
3. What scheduling/backpressure invariants should we enforce so multiple streams don’t self-DDOS (queue latency budgets)?
4. Any prior art in realtime video generation systems for multi-stream packing with a shared decode stage?

## Repo prompt pack (include these files)

### Our coordination docs (primary input)
- `scope-drd/notes/FA4/h200/tp/proposals/r0-latent-anchor/design.md`
- `scope-drd/notes/FA4/h200/tp/proposals/r0-latent-anchor/ask-5pro-anchor-decode.md` (companion: detailed anchor-decode questions with ground-truth semantics)

### Current state / big picture
- `scope-drd/notes/FA4/h200/tp/landscape.md`
- `scope-drd/notes/FA4/h200/tp/strategic-outlook.md`

### Code paths (VAE + context anchoring)
- `scope-drd/src/scope/server/pp1_stage0.py` (lines 859-895: decode_latents; 1068-1214: context buffers + _build_real_context_frames)
- `scope-drd/src/scope/core/pipelines/wan2_1/vae/wan.py` (WanVAE class — decode/encode methods)
- `scope-drd/src/scope/core/pipelines/wan2_1/vae/modules/vae.py` (decoder internals — temporal convolutions)
- `scope-drd/src/scope/core/pipelines/krea_realtime_video/blocks/recompute_kv_cache.py` (mesh-side context consumption, lines 264-357)
- `scope-drd/src/scope/core/pipelines/krea_realtime_video/blocks/prepare_context_frames.py` (sliding-buffer semantics, decoded_frame_buffer size formula)

### Compile paths
- `scope-drd/src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py` (lines 545-582: _kv_bias_flash_combine — graph break source)
- `scope-drd/src/scope/core/tensor_parallel/linear.py` (funcol switching)

### Reference library (already digested, for cross-reference)
- `refs/topics/23-vae-latency-chunking.md`
- `refs/topics/15-pipeline-scheduling-theory.md`
- `refs/topics/19-producer-consumer-backpressure.md`
- `refs/topics/09-dynamo-tracing.md`
- `refs/topics/10-inductor-fusion-rules.md`
- `refs/resources/causvid.md`
- `refs/resources/streamdiffusionv2.md`
- `refs/resources/pipedit.md`
- `refs/resources/zero-bubble-pp.md`

### Prior 5 Pro threads (for calibration — don't repeat, extend)
- `deep-research/2026-02-22/ask-5pro-compile-distributed-hardening.md` (compile thread)
- `deep-research/2026-02-23/ask-pp1-g1c-overlap.md` (overlap thread)
- `deep-research/2026-02-22/ask-5pro-pp-overlap-threading-and-queues.md` (overlap primitives)

## What "good" looks like

### Thread 1 (VAE T=1):
- A clear YES/NO/CONDITIONAL on T=1 decode feasibility for Wan2.1
- If NO: the minimum T and why, plus workaround strategies
- If YES: confirmation of semantic equivalence (T=1 output vs first-T-of-full-decode)
- **Equivalence check recipe**: Given a fixed latent tensor and defined decoder-cache
  state, how to test (numerically) that `anchor_pixel_full == anchor_pixel_T1` — which
  indices, what cache mode, what tolerance is acceptable
- External precedent from other video VAEs

### Thread 2 (compile):
- An explanation grounded in Inductor internals for why SM120 gains are lower
- Whether the graph break at `_kv_bias_flash_combine` is the dominant factor
- A diagnostic plan we can run to quantify lost fusion opportunity
- Whether we should spend time fixing the graph break or accept 9%

### Thread 3 (decode hiding):
- Any prior art for partial/temporal VAE decode in pipeline-parallel video generation
- Cheaper alternatives to the full decode→re-encode round trip
- Whether latent-space anchoring can be made viable (techniques, not just theory)
- Systems that decouple decode from the generation loop entirely

**Explicitly not asking for**: general PP scheduling theory (we have it), CausVid/PipeDiT/StreamDiffusionV2 summaries (already digested), or broad compile optimization guides. We need targeted answers to specific questions we can't answer from our existing library.
