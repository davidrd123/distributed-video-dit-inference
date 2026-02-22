# Topic 1: NCCL internals — ring vs tree algorithms, NVLink topology, GPU-level all-reduce/send/recv

NCCL's algorithm selection (ring, tree, NVLS, CollNet) is governed by message size, topology, and the `NCCL_ALGO` environment variable. Understanding this layer is essential for diagnosing why your pipeline send/recv calls behave differently across NVLink vs PCIe topologies.

## Resources

<!-- Resource IDs from manifest belonging to this topic -->
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| nccl-user-guide | NCCL User Guide | high | pending |
| scaling-dl-nccl | Scaling Deep Learning Training with NCCL | medium | pending |
| nccl-tuning | Understanding NCCL Tuning to Accelerate GPU-to-GPU Communication | medium | pending |
| nccl-cuda-graphs | Using NCCL with CUDA Graphs | medium | pending |

## Synthesis

<!-- To be filled during study -->

### Mental model

### Key concepts

### Cross-resource agreement / disagreement

### Practical checklist

### Gotchas and failure modes

### Experiments to run
