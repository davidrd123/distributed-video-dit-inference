# Topic 8: Kernel launch overhead — Python-to-GPU dispatch path

Each PyTorch operator call traverses Python -> C++ dispatch -> CUDA kernel launch. At **~10us per launch**, this overhead dominates when running many small operations. `torch.compile` and CUDA graphs are the primary mitigations.

## Resources

<!-- Resource IDs from manifest belonging to this topic -->
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| making-dl-go-brrrr | Making Deep Learning Go Brrrr From First Principles | high | condensed |
| gpu-mode-lecture-6 | GPU MODE Lecture 6: Optimizing Optimizers in PyTorch | low | link_only |
| pytorch-internals-ezyang | PyTorch internals | medium | pending |

## Implementation context

The clearest real-world example of launch/dispatch overhead dominating is TP=2 + compile pre-fix: `torch._dynamo.disable()` around ~160 collectives caused ~160 graph breaks per forward and throughput collapsed to **9.6 FPS** (Runs 8-9b) even after eliminating the `_kv_bias_flash_combine` `Tensor.item()` break. Switching to functional collectives eliminated collective graph breaks and restored fusion (Run 12b: **24.5 FPS**, “single compiled graph per block” behavior).

See: `refs/implementation-context.md` → Phase 1, `scope-drd/notes/FA4/h200/tp/bringup-run-log.md` (Runs 8-12b).

## Synthesis

<!-- To be filled during study -->

### Mental model

### Key concepts

### Cross-resource agreement / disagreement

### Practical checklist

### Gotchas and failure modes

### Experiments to run
