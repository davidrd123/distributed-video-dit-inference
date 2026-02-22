---
status: stub
---

# Topic 17: Reading Nsight / torch.profiler traces

Profiling is the primary tool for understanding where time is spent in distributed GPU workloads. PyTorch's built-in profiler integrates with TensorBoard for visualization, while NVIDIA Nsight Compute provides kernel-level analysis including roofline positioning.

## Resources

<!-- Resource IDs from manifest belonging to this topic -->
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| pytorch-profiler-tensorboard | PyTorch Profiler with TensorBoard | medium | pending |
| profiling-torch-compile | Profiling torch.compile performance | medium | pending |
| gpu-mode-lecture-1 | GPU MODE Lecture 1: How to Profile CUDA Kernels in PyTorch | low | link_only |
| nsight-roofline | NVIDIA Nsight Compute: Roofline Analysis | low | pending |

## Implementation context

Block profiling (Run 10b) is the main measured attribution we have: **denoise 435ms (66.8%)**, **decode 107ms (16.5%)**, **recompute_kv_cache 104ms (16.0%)** at TP=2, compile off. That profile also notes the measurement pitfall: per-block `synchronize()` perturbs overlap and inflates times, so use it for attribution rather than absolute throughput. For PP0/PP1, overlap is validated via derived metrics (`period`, `stage0`, `stage1`, `OverlapScore ≥ 0.30`) instead of relying on cross-rank trace alignment.

See: `refs/implementation-context.md` → Phase 2, `scope-drd/notes/FA4/h200/tp/bringup-run-log.md` (Run 10b), `scope-drd/notes/FA4/h200/tp/pp0-bringup-runbook.md` (Phase 2 overlap metrics).

## Synthesis

<!-- To be filled during study -->

### Mental model

### Key concepts

### Cross-resource agreement / disagreement

### Practical checklist

### Gotchas and failure modes

### Experiments to run
