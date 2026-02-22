# Topic 1: NCCL internals — ring vs tree algorithms, NVLink topology, GPU-level all-reduce/send/recv

NCCL's algorithm selection (ring, tree, NVLS, CollNet) is governed by message size, topology, and the `NCCL_ALGO` environment variable. Understanding this layer is essential for diagnosing why your pipeline send/recv calls behave differently across NVLink vs PCIe topologies.

## Resources

<!-- Resource IDs from manifest belonging to this topic -->
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| nccl-user-guide | NCCL User Guide | high | converted |
| scaling-dl-nccl | Scaling Deep Learning Training with NCCL | medium | pending |
| nccl-tuning | Understanding NCCL Tuning to Accelerate GPU-to-GPU Communication | medium | pending |
| nccl-cuda-graphs | Using NCCL with CUDA Graphs | medium | pending |

## Implementation context

This topic directly informs the TP v0 communication budget: **80 large + 80 tiny all-reduces per chunk, ~9ms total** on 2×H200 NVLink. Each large all-reduce (`[1, 2160, 5120]` BF16, ~21 MiB) takes ~0.113ms. Understanding NCCL algorithm selection for this message size and topology is critical for diagnosing collective overhead.

See: `refs/implementation-context.md` → Phase 1: TP v0, `scope-drd/notes/FA4/h200/tp/feasibility.md` Section 1-2.

## Synthesis

<!-- To be filled during study -->

### Mental model

### Key concepts

### Cross-resource agreement / disagreement

### Practical checklist

### Gotchas and failure modes

### Experiments to run
