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

See: `refs/implementation-context.md` → Phase 2, `scope-drd/notes/FA4/h200/tp/bringup-run-log.md` (Run 10b), `scope-drd/notes/FA4/h200/tp/v1.1-generator-only-workers.md` (v1.1 rationale), `scope-drd/notes/FA4/h200/tp/streamdiffusion-v2-analysis-opus.md` (Stream-VAE), `scope-drd/notes/FA4/h200/tp/async-decode-overlap-scoping.md` (gain ceiling).

Relevant Scope code:
- `scope-drd/src/scope/core/pipelines/krea_realtime_video/modules/vae.py` and `scope-drd/src/scope/core/pipelines/krea_realtime_video/blocks/prepare_context_frames.py` (where decode/encode and context assembly happen today)
- `scope-drd/src/scope/server/frame_processor.py` (frame chunking / decoded frame buffer lifecycle that couples to recompute)

## Synthesis

<!-- To be filled during study -->

### Mental model

### Key concepts

### Cross-resource agreement / disagreement

### Practical checklist

### Gotchas and failure modes

### Experiments to run
