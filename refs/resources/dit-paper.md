# Scalable Diffusion Models with Transformers (DiT)

| Field | Value |
|-------|-------|
| Source | https://arxiv.org/abs/2212.09748 |
| Type | paper |
| Topics | 24 |
| Authors | William Peebles, Saining Xie |
| Year | 2023 (ICCV) |
| Status | extracted |

## Why it matters

The foundational DiT paper. Replaces U-Net with transformers in latent diffusion, demonstrating strong correlation between compute (Gflops) and sample quality. This is the architecture the inference system will be running -- understanding its structure (patchification, adaptive layer norm, conditioning mechanisms) is essential for partitioning it across pipeline stages.

## Core claims

1. **Claim**: The U-Net inductive bias is *not* crucial to diffusion model performance — standard transformers can directly replace the U-Net backbone with no architectural compromise.
   **Evidence**: [sources/dit-paper/full.md#1-introduction](../sources/dit-paper/full.md#1-introduction) — "We show that the U-Net inductive bias is *not* crucial to the performance of diffusion models, and they can be readily replaced with standard designs such as transformers."

2. **Claim**: Model Gflops, not parameter count, is the critical scaling variable. DiT configs with similar Gflops but different parameter counts achieve similar FID (-0.93 correlation between log Gflops and FID-50K).
   **Evidence**: [sources/dit-paper/full.md#5-experiments](../sources/dit-paper/full.md#5-experiments) — "different DiT configs obtain similar FID values when their total Gflops are similar (e.g., DiT-S/2 and DiT-B/4). We find a strong negative correlation between model Gflops and FID-50K." Figure 8: correlation -0.93.

3. **Claim**: adaLN-Zero conditioning is the best block design — lower FID than cross-attention and in-context conditioning while being the most compute-efficient. Zero-initialization (each block starts as identity function) is key.
   **Evidence**: [sources/dit-paper/full.md#5-experiments](../sources/dit-paper/full.md#5-experiments) — "The adaLN-Zero block yields lower FID than both cross-attention and in-context conditioning while being the most compute-efficient. At 400K training iterations, the FID achieved with the adaLN-Zero model is nearly half that of the in-context model."

4. **Claim**: Decreasing patch size (increasing token count) improves quality as much as increasing model size, even though parameter count stays essentially the same.
   **Evidence**: [sources/dit-paper/full.md#5-experiments](../sources/dit-paper/full.md#5-experiments) — "we again observe considerable FID improvements throughout training by simply scaling the number of tokens processed by DiT, holding parameters approximately fixed." Figure 6 (bottom).

5. **Claim**: Scaling sampling compute cannot compensate for a lack of model compute. A smaller model with 5x more sampling steps still underperforms a larger model.
   **Evidence**: [sources/dit-paper/full.md#52-scaling-model-vs-sampling-compute](../sources/dit-paper/full.md#52-scaling-model-vs-sampling-compute) — "DiT-L/2 uses 80.7 Tflops to sample each image; XL/2 uses 5× less compute... Nonetheless, XL/2 has the better FID-10K (23.7 vs 25.9)."

6. **Claim**: DiT-XL/2 achieves SOTA FID of 2.27 on class-conditional ImageNet 256×256 at 118.6 Gflops — substantially more compute-efficient than U-Net alternatives (ADM: 1120 Gflops, ADM-U: 742 Gflops).
   **Evidence**: Table 2 and Table 6 in [sources/dit-paper/full.md#51-state-of-the-art-diffusion-models](../sources/dit-paper/full.md#51-state-of-the-art-diffusion-models).

7. **Claim**: Training is remarkably stable — no learning rate warmup, no regularization, no loss spikes. Identical hyperparameters across all model sizes and patch sizes.
   **Evidence**: [sources/dit-paper/full.md#4-experimental-setup](../sources/dit-paper/full.md#4-experimental-setup) — "We did not find learning rate warmup nor regularization necessary... training was highly stable across all model configs and we did not observe any loss spikes."

## Key technical details

### Architecture

- **Input pipeline**: Image (256×256×3) → VAE encoder (downsample 8×) → latent (32×32×4) → patchify (patch size p) → sequence of T = (I/p)² tokens of dimension d → sine-cosine positional embeddings
- **Patch sizes explored**: p = 2, 4, 8. Halving p quadruples T and at least quadruples Gflops. p has no meaningful impact on parameter count.
- **adaLN-Zero block**: LayerNorm with scale/shift (γ, β) regressed from sum of timestep + class embeddings via MLP with SiLU. Additionally regresses scaling parameters α applied before residual connections. MLP initialized to output zero → each block initializes as identity.
- **Conditioning input**: timestep t embedded via 256-dim frequency embedding → 2-layer MLP with SiLU. Class label embedded similarly. Sum of both fed to adaLN layers.
- **Decoder**: Final adaptive layer norm → linear projection to p×p×2C per token → rearrange to spatial layout → noise prediction + diagonal covariance prediction.
- **Nonlinearity**: GELU (tanh approximation) in core transformer.

### Model configs

| Model | Layers | Hidden dim | Heads | Gflops (p=4) | Params |
|-------|--------|-----------|-------|-------------|--------|
| DiT-S | 12 | 384 | 6 | 1.4 | 33M |
| DiT-B | 12 | 768 | 12 | 5.6 | 130M |
| DiT-L | 24 | 1024 | 16 | 19.7 | 458M |
| DiT-XL | 28 | 1152 | 16 | 29.1 | 675M |

DiT-XL/2 at 256×256: 118.6 Gflops, 675M params.
DiT-XL/2 at 512×512: 524.6 Gflops, 675M params (1024 tokens).

### Diffusion setup

- Operates in VAE latent space (Stable Diffusion VAE, 84M params, frozen)
- t_max = 1000, linear variance schedule 1×10⁻⁴ to 2×10⁻²
- Trains εθ with L_simple (MSE), trains Σθ with full D_KL
- EMA decay 0.9999
- Classifier-free guidance: dropout conditioning during training, guidance scale s > 1 at inference
- Best results use ft-EMA VAE decoder (fine-tuned from Stable Diffusion)

### Classifier-free guidance detail

Guidance applied to first 3 of 4 latent channels only. Three-channel guidance with scale (1+x) ≈ four-channel guidance with scale (1+¾x). Scale of 1.5 → FID 2.27.

## Actionables / gotchas

- **Pipeline partitioning**: DiT is a uniform stack of N identical blocks (no U-Net skip connections, no resolution changes mid-network). This makes it naturally amenable to pipeline parallelism — partition by blocks, each stage gets a contiguous slice. No complex skip-connection routing needed.
- **Patchify is cheap but critical**: The patchify layer and positional embeddings are trivial compute but determine the token sequence length that dominates all downstream Gflops. For video DiT, the temporal dimension further multiplies T — this is where Gflops explode.
- **adaLN conditioning couples timestep to every block**: The conditioning MLP produces per-block scale/shift/gate parameters. In pipeline-parallel inference, the timestep/class embeddings must be available at every stage, not just the first. Either broadcast the embeddings or compute the adaLN MLP per-stage.
- **Zero-initialization matters**: adaLN-Zero's identity initialization halved FID vs. vanilla adaLN. If modifying the architecture (e.g., adding temporal attention for video), maintaining zero-init on new residual paths is likely important.
- **Gflops scale quadratically with token count**: Halving patch size → 4× tokens → at least 4× Gflops (more due to quadratic attention). For video with temporal tokens, this is the dominant cost driver. Sequence parallelism or attention optimization (FlashAttention) becomes critical.
- **VAE decoder is swappable**: Different decoder weights yield comparable results without retraining the diffusion model. For video inference, this means VAE decode optimization (tiling, chunking) is independent of the diffusion backbone.
- **Sampling compute can't substitute for model compute**: Don't try to save Gflops by using a smaller DiT with more denoising steps. The quality ceiling is set by model size.
- **No skip connections simplifies activation memory**: Unlike U-Net where activations from early layers must be kept for skip connections, DiT blocks are purely sequential. For pipeline parallelism, this means activation memory per stage is proportional to the number of blocks in that stage, with no cross-stage activation dependencies beyond the pipeline send/recv.

## Related resources

- [streamdiffusionv2](streamdiffusionv2.md) — extends streaming inference to video diffusion models including DiT
- [pipedit](pipedit.md) — directly addresses pipeline-parallel DiT inference
- [pagedattention](pagedattention.md) — memory management relevant to DiT's attention KV caches
- [making-dl-go-brrrr](making-dl-go-brrrr.md) — performance analysis framework for DiT's compute/memory characteristics
- [gpipe](gpipe.md) — foundational pipeline parallelism; DiT's uniform block structure maps cleanly to PP stages
- [zero-bubble-pp](zero-bubble-pp.md) — advanced PP scheduling applicable to DiT's sequential block structure
