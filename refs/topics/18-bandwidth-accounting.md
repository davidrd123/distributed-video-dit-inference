---
status: stub
---

# Topic 18: Bandwidth accounting from first principles

Bandwidth accounting means computing the **theoretical minimum memory traffic** for an operation, then comparing against measured bandwidth to determine utilization. For a matrix multiply C = A x B with dimensions (M, K) x (K, N), minimum traffic is `2(MK + KN + MN)` bytes (read A, read B, write C).

## Resources

<!-- Resource IDs from manifest belonging to this topic -->
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| nvidia-gpu-perf-guide | NVIDIA GPU Performance Background User's Guide | medium | pending |
| modal-memory-bandwidth | What is memory bandwidth? (Modal GPU Glossary) | low | pending |
| modal-roofline | What is the roofline model? (Modal GPU Glossary) | low | pending |
| matmul-shapes | What Shapes Do Matrix Multiplications Like? | medium | pending |

## Implementation context

Bandwidth accounting explains two observed ceilings: (1) single-stream FPS scales with HBM bandwidth across GPUs (B200/B300 ~8 TB/s ≈ 33–34 FPS vs H200 ~4.8 TB/s ≈ 20 FPS; ratio ~0.58–0.60), and (2) TP collectives are fast enough on NVLink to be plausible (BF16 all-reduce of a ~21 MiB `[1,2160,5120]` tensor is **~0.113ms**, ~195 GB/s per rank). These numbers anchor whether a proposed optimization is compute-bound vs bandwidth/overhead-bound.

See: `refs/implementation-context.md` → System summary + Phase 1, `scope-drd/notes/FA4/h200/tp/streamdiffusion-v2-analysis-opus.md` (H200 bandwidth-bound evidence), `scope-drd/notes/FA4/h200/tp/feasibility.md` (Section 1: collective bandwidth).

Relevant Scope artifacts:
- `scope-drd/scripts/bench_nccl_collectives.py` (collective latency/bandwidth measurements)
- `scope-drd/scripts/bench_gemm_shapes.py` / `scope-drd/scripts/gemm_shape_census.py` (compute-vs-bytes reasoning for common matmuls)

## Synthesis

<!-- To be filled during study -->

### Mental model

### Key concepts

### Cross-resource agreement / disagreement

### Practical checklist

### Gotchas and failure modes

### Experiments to run
