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
| 36 | PP1+TP=4, single GPU baseline | ~11.0 | PP1+TP=4+compile only 10% faster on 5× GPUs |
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
# Full chunk: 9 latent frames → 33 pixel frames (4:1 temporal upsample - 1 overlap)
pixels = vae.decode(latents)  # latents: [1, 9, 16, H_lat, W_lat]
```

We want to do:
```python
# Anchor-only: 1 latent frame → 4 pixel frames
anchor_pixels = vae.decode(latents[:, :1])  # [1, 1, 16, H_lat, W_lat]
```

**Questions:**
1. Do Wan2.1's temporal convolutions (CausalConv3d / temporal upsampling) handle T=1
   input correctly? Or do they require minimum T>1 due to padding/kernel size assumptions?

2. If T=1 decode produces 4 pixel frames, is `pixel_frames[-1]` (the last of 4)
   semantically equivalent to the first 4 pixels from a full T=9 decode? Or do later
   temporal convolution layers cause cross-frame bleeding that makes T=1 output differ?

3. Are there known workarounds if T=1 is unsupported? (E.g., decode T=2 and discard,
   or pad with zeros, or use the encoder's temporal receptive field to determine
   minimum T.)

4. **Search broadly**: Other video VAEs with temporal convolutions (Open-Sora,
   CogVideoX, LTX-Video, SVD) — how do they handle partial-temporal decode?
   Is there a standard pattern for "decode minimum temporal footprint"?

5. What is the **temporal receptive field** of Wan2.1's decoder? If each temporal
   conv has kernel_size=3 with causal padding, and there are N temporal layers,
   the minimum T for stable output is bounded. What's that bound?

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
   two GPU architectures for the same model?

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

## Repo prompt pack (include these files)

### Our coordination doc (primary input)
- `scope-drd/notes/FA4/h200/tp/proposals/r0-latent-anchor/design.md`

### Current state / big picture
- `scope-drd/notes/FA4/h200/tp/landscape.md`
- `scope-drd/notes/FA4/h200/tp/strategic-outlook.md`

### Code paths (VAE + context anchoring)
- `scope-drd/src/scope/server/pp1_stage0.py` (lines 859-895: decode_latents; 1068-1214: context buffers + _build_real_context_frames)
- `scope-drd/src/scope/core/pipelines/wan2_1/vae/wan.py` (WanVAE class — decode/encode methods)
- `scope-drd/src/scope/core/pipelines/wan2_1/vae/modules/vae.py` (decoder internals — temporal convolutions)
- `scope-drd/src/scope/core/pipelines/krea_realtime_video/blocks/recompute_kv_cache.py` (mesh-side context consumption, lines 264-357)

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
