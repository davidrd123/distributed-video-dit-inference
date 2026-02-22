---
status: draft
---

# Topic 16: Roofline model applied to transformers — arithmetic intensity for attention vs FFN

The roofline model plots achievable FLOPS against **arithmetic intensity** (FLOPS/byte of memory traffic). Attention is typically **memory-bandwidth-bound** (low arithmetic intensity due to reading/writing large KV matrices), while FFN layers are more likely **compute-bound** (high arithmetic intensity from large matrix multiplications). This distinction drives optimization strategy.

## Resources

<!-- Resource IDs from manifest belonging to this topic -->
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| making-dl-go-brrrr | Making Deep Learning Go Brrrr From First Principles | high | condensed |
| roofline-paper | Roofline: An Insightful Visual Performance Model for Multicore Architectures | medium | link_only |
| nvidia-gpu-perf-guide | NVIDIA GPU Performance Background User's Guide | medium | fetched |
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

- **Roofline**: a kernel’s best-case throughput is limited by `min(peak_compute, arithmetic_intensity * peak_bandwidth)`, where arithmetic intensity is `FLOPs / bytes_moved_to_HBM`.
- **Three regimes (add the missing third axis)**: in practice you often hit a third limiter, *launch/dispatch overhead* (the “overhead-bound” regime), where the GPU is idle between many tiny kernels/subgraphs. Roofline explains compute-vs-bandwidth; `making-dl-go-brrrr` adds overhead as the third “why isn’t it brrr-ing?” bucket.
- **Transformer mapping**:
- Attention’s *matmuls* can be compute-heavy, but the *end-to-end attention block* is often bandwidth-limited because naive implementations materialize and repeatedly read/write large intermediates; FlashAttention-style fusion is primarily about cutting HBM traffic (raising effective arithmetic intensity), not reducing FLOPs.
- FFN (large GEMMs) is the canonical compute-bound case: if it’s slow, you’re usually failing to reach Tensor Core roofs (dtype/layout/dim divisibility/launch configuration).
- **Scope grounding**:
- H200 has very high HBM bandwidth (~4.8 TB/s), and observed single-GPU FPS tracking HBM across generations (H200 ~20 FPS vs B200/B300 ~33–34 FPS) is consistent with a bandwidth-limited steady state for large parts of the pipeline.
- The 9.6 → 24.5 FPS jump when removing ~160 graph breaks (Runs 8–9b → 12b) is the signature of an overhead-bound regime: reducing dispatch fragmentation mattered more than “math” optimizations.
- Block profiling (Run 10b) shows decode + recompute_kv_cache are ~33% of wall time, making them a first-order throughput limiter unless overlapped/hidden or their memory traffic reduced.

### Key concepts

- **Arithmetic intensity (AI)**: `AI = FLOPs / bytes_to_HBM`. Low AI → bandwidth-bound; high AI → compute-bound.
- **Ridge point (“knee”)**: `AI_knee = peak_FLOPs / peak_bandwidth`. If `AI < AI_knee`, the kernel rides the bandwidth roof; if `AI > AI_knee`, it hits the compute roof.
- **Bandwidth-bound optimization strategy**: reduce bytes moved, not FLOPs.
- Fuse pointwise chains (LN/SiLU/scale/shift/gates) so intermediates stay in registers/SRAM instead of round-tripping to HBM (`making-dl-go-brrrr`).
- Avoid materializing large intermediates (e.g., attention score matrices); algorithmic changes that change memory traffic can move you rightward/upward on the roofline plot even if FLOPs stay similar.
- **Compute-bound optimization strategy**: increase achieved FLOPs.
- Hit Tensor Cores (bf16/fp16/fp8 as appropriate), ensure dims align well, and keep GEMMs large enough to amortize overhead.
- **Overhead-bound strategy**: reduce launches/subgraphs.
- Eliminate graph breaks, maximize fusion/compilation, and (later) consider CUDA Graphs; the Runs 8–9b outcome is the concrete “don’t ignore overhead” example in this codebase.
- **Token count drives both compute and communication surfaces**: DiT reports Gflops scaling strongly with token count, and attention includes quadratic terms in tokens; for our canonical `S=2160` token setting, modest token increases quickly dominate both compute and TP collective payloads.

### Cross-resource agreement / disagreement

- **Agreement**:
- `making-dl-go-brrrr`’s three-regime framework (overhead/bandwidth/compute bound) matches Scope’s empirical behavior: graph breaks caused an overhead-bound collapse (Run 9b), while decode+recompute being ~33% of wall time makes “bytes moved” work unavoidable unless re-architected/overlapped.
- `dit-paper`’s “Gflops scales with tokens (including quadratic attention terms)” explains why resolution/token choices dominate end-to-end cost, and why attention implementations that reduce memory traffic (not just FLOPs) matter for roofline position.
- **Nuance / potential disagreement**:
- “Attention is bandwidth-bound, FFN is compute-bound” is a useful first approximation, but attention is a mix: QKV/out-projection GEMMs can be compute-bound while the softmax/reduction/materialization path is bandwidth/overhead sensitive; FlashAttention shifts the balance by changing what touches HBM.
- “Decode + recompute is bandwidth-bound” is a hypothesis supported by the HBM-tracking FPS observation and the general character of decode-style kernels, but the correct diagnosis should be confirmed with bandwidth counters / scaling tests (roofline is a diagnostic, not a vibe check).
- **Roofline alone is incomplete for distributed inference**: NVLink/NCCL comm and Python/dispatch overhead add roofs of their own; you need a “multi-roofline” mental model (HBM roof, NVLink roof, and dispatch overhead).

### Practical checklist

- **Choose a unit of analysis**: per-kernel (best), per-layer (OK), per-block/stage (useful for PP decisions).
- **Measure first**:
- For overhead-bound: look for GPU idle gaps and many short kernels/subgraphs; track graph breaks (Runs 8–9b).
- For bandwidth-bound: check achieved HBM throughput and whether runtime scales with bytes (dtype/resolution/sequence length).
- For compute-bound: check achieved Tensor Core utilization / achieved FLOPs.
- **Compute a back-of-envelope AI**: estimate FLOPs from shapes (e.g., GEMMs from `gemm_shape_census.py`) and bytes from tensor reads/writes; compare against `AI_knee`.
- **Apply the right lever**:
- Overhead-bound: remove graph breaks (functional collectives), increase fusion/compile coverage, reduce Python dispatch in the critical loop.
- Bandwidth-bound: fuse (reduce intermediate writes), use FlashAttention-style kernels, avoid redundant format conversions/copies, reduce KV / activation traffic, and for VAE decode consider tiling/chunking strategies that reduce total bytes moved (but watch overhead).
- Compute-bound: ensure tensor-core-friendly dtypes/layouts and large GEMM shapes; avoid small matmuls that fall back to non-TC paths.
- **Distributed-specific**:
- Don’t expect TP/PP to speed up work that is not sharded: if decode/recompute runs identically on every worker rank, it is pure throughput overhead regardless of whether it is compute- or bandwidth-bound.
- If end-to-end FPS tracks HBM bandwidth across GPU generations, prioritize bandwidth reducers over adding compute parallelism.

### Gotchas and failure modes

- **Confusing “overhead-bound” (dispatch) with “overhead” (redundant work)**: Runs 8–9b are overhead-bound in the dispatch sense; decode/recompute duplicated on workers is “system overhead” even if the kernels are bandwidth-bound.
- **AI is easy to miscompute**: counting FLOPs is straightforward; counting bytes isn’t (caches, read-for-ownership, transposes, materialized intermediates). Treat AI estimates as order-of-magnitude tools.
- **Fusion tradeoffs**: fusion reduces HBM traffic but can raise register pressure, reduce occupancy, or make compilation fragile; performance can regress if the fused kernel becomes too complex.
- **Distributed roofs exist**: for TP, collective bandwidth/latency can dominate even if per-GPU compute looks fine; roofline reasoning must include NVLink/NCCL “bytes per step”.
- **Token scaling changes the diagnosis**: increasing tokens can push some kernels toward compute-bound while making others worse; always re-evaluate when changing resolution/temporal length/patching.
- **Tiling decode can move you into overhead-bound**: smaller tiles reduce working-set size but increase kernel count and dispatch; validate with a profiler.

### Experiments to run

- **Overhead-bound confirmation (compile path)**: reproduce the Runs 8–9b pattern intentionally (graph breaks on collectives) and confirm that reducing graph breaks increases FPS more than kernel-level math tweaks.
- **Bandwidth-bound confirmation (decode/recompute)**: vary bytes (dtype where possible, resolution, number of frames) and test whether decode and recompute_kv_cache time scales roughly with tensor size; inspect achieved HBM throughput in a profiler.
- **Attention vs FFN roofline placement**: use a GEMM shape census (QKV/FFN projections) to estimate AI and compare measured kernel time vs expected compute/bandwidth ceilings; check whether attention kernels benefit most from memory-traffic reducers (FlashAttention settings) while FFN benefits most from TC utilization.
- **System-level “33% slice” mitigation**: test the v1.1 “generator-only workers” idea (broadcast minimal tensors so workers skip decode) and validate whether the end-to-end FPS improvement matches the removed decode+recompute share.
- **Token-scaling sanity check**: increase/decrease token count (spatial or temporal) and confirm whether attention cost rises superlinearly (quadratic term), consistent with DiT’s token-driven Gflops story.
