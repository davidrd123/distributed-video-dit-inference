# Topic 24: Video DiT scheduling — denoising steps, causal dependencies, rolling windows

Video DiT scheduling determines how denoising steps are ordered and parallelized across frames. **Rolling window** approaches generate video autoregressively: each window of N frames shares context from the previous window, enabling arbitrarily long generation. **Stream Batch** (from StreamDiffusion) batches denoising steps across time for throughput.

## Resources

<!-- Resource IDs from manifest belonging to this topic -->
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| streamdiffusion | StreamDiffusion: A Pipeline-level Solution for Real-time Interactive Generation | medium | pending |
| streamdiffusionv2 | StreamDiffusionV2 | high | fetched |
| diffusion-video-survey | Diffusion Models for Video Generation | medium | pending |
| pipedit | PipeDiT: Accelerating DiT in Video Generation with Pipelining and Decoupling | high | converted |
| dit-paper | Scalable Diffusion Models with Transformers (DiT) | high | condensed |

## Implementation context

This topic covers the scheduling strategy that determines whether PP can be filled. The working system uses 1 denoising step per chunk (causal streaming). StreamDiffusionV2 demonstrates that treating denoising steps as a batch multiplier (Stream Batch) fills the PP pipeline, but only when multiple latents at different noise levels are simultaneously in flight via a rolling window.

For the Scope system, the three paths to B>1 are: (1) multiple concurrent sessions, (2) rolling window of frame chunks in flight, (3) interleaved denoise across frames. Path (1) is a throughput play; paths (2-3) require restructuring the generation loop.

See: `refs/implementation-context.md` → Phase 3, `scope-drd/notes/FA4/h200/tp/streamdiffusion-v2-analysis-opus.md` → "What This Means for Our System".

Relevant Scope code / notes:
- `scope-drd/notes/FA4/h200/tp/explainers/08-streamv2v-mapping.md` (mapping StreamDiffusionV2 contracts/transport → Scope PP0/PP1)
- `scope-drd/src/scope/core/pipelines/krea_realtime_video/pipeline.py` (current per-chunk generator call pattern)
- `scope-drd/src/scope/server/frame_processor.py` (ingress loop, chunking, overrides that will become PP envelope fields)

## Synthesis

<!-- To be filled during study -->

### Mental model

### Key concepts

### Cross-resource agreement / disagreement

### Practical checklist

### Gotchas and failure modes

### Experiments to run
