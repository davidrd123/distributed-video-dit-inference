# Topic 24: Video DiT scheduling — denoising steps, causal dependencies, rolling windows

Video DiT scheduling determines how denoising steps are ordered and parallelized across frames. **Rolling window** approaches generate video autoregressively: each window of N frames shares context from the previous window, enabling arbitrarily long generation. **Stream Batch** (from StreamDiffusion) batches denoising steps across time for throughput.

## Resources

<!-- Resource IDs from manifest belonging to this topic -->
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| streamdiffusion | StreamDiffusion: A Pipeline-level Solution for Real-time Interactive Generation | medium | pending |
| streamdiffusionv2 | StreamDiffusionV2 | high | pending |
| diffusion-video-survey | Diffusion Models for Video Generation | medium | pending |
| pipedit | PipeDiT: Accelerating DiT in Video Generation with Pipelining and Decoupling | high | pending |
| dit-paper | Scalable Diffusion Models with Transformers (DiT) | high | pending |

## Synthesis

<!-- To be filled during study -->

### Mental model

### Key concepts

### Cross-resource agreement / disagreement

### Practical checklist

### Gotchas and failure modes

### Experiments to run
