---
status: stub
---

# Topic 16: Roofline model applied to transformers — arithmetic intensity for attention vs FFN

The roofline model plots achievable FLOPS against **arithmetic intensity** (FLOPS/byte of memory traffic). Attention is typically **memory-bandwidth-bound** (low arithmetic intensity due to reading/writing large KV matrices), while FFN layers are more likely **compute-bound** (high arithmetic intensity from large matrix multiplications). This distinction drives optimization strategy.

## Resources

<!-- Resource IDs from manifest belonging to this topic -->
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| making-dl-go-brrrr | Making Deep Learning Go Brrrr From First Principles | high | condensed |
| roofline-paper | Roofline: An Insightful Visual Performance Model for Multicore Architectures | medium | link_only |
| nvidia-gpu-perf-guide | NVIDIA GPU Performance Background User's Guide | medium | pending |
| gpu-mode-lecture-8 | GPU MODE Lecture 8: CUDA Performance Checklist | low | link_only |

## Implementation context

The roofline model explains why the system is **memory-bandwidth-bound on H200** (~4.8 TB/s HBM) and why B200/B300 (~8 TB/s) are proportionally faster (~33 FPS vs ~20 FPS, tracking the bandwidth ratio ~0.60). It also explains why TP doesn't give 2x speedup: sharding reduces compute per device, but total bytes loaded across devices is unchanged. The StreamDiffusionV2 roofline analysis (Figure 4, Appendix A) shows that Stream Batch increases effective batch size, sliding the operating point rightward toward the roofline knee.

Block profiling (Run 10b): decode + recompute = 33% of wall time. These blocks are not TP-sharded, making them pure "overhead-bound" work on worker ranks.

See: `refs/implementation-context.md` → Phase 2, `scope-drd/notes/FA4/h200/tp/streamdiffusion-v2-analysis-opus.md`.

Relevant Scope artifacts:
- `scope-drd/scripts/bench_nccl_collectives.py` (collective microbenchmarks used to ground bandwidth claims)
- `scope-drd/scripts/gemm_shape_census.py` (token/matmul shape census for roofline-style reasoning)

## Synthesis

<!-- To be filled during study -->

### Mental model

### Key concepts

### Cross-resource agreement / disagreement

### Practical checklist

### Gotchas and failure modes

### Experiments to run
