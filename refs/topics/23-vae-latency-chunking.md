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

## Synthesis

<!-- To be filled during study -->

### Mental model

### Key concepts

### Cross-resource agreement / disagreement

### Practical checklist

### Gotchas and failure modes

### Experiments to run
