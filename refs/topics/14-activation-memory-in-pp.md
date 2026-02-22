# Topic 14: Activation memory in PP

Peak activation memory in PP depends on the schedule. GPipe stores activations for all micro-batches; 1F1B limits in-flight activations to `num_stages`. **Activation checkpointing** trades compute for memory by recomputing activations during backward.

## Resources

<!-- Resource IDs from manifest belonging to this topic -->
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| megatron-ptdp | Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM | medium | pending |
| zero-bubble-pp | Zero Bubble Pipeline Parallelism | high | pending |
| pytorch-pipelining-api | PyTorch torch.distributed.pipelining API | high | pending |

## Synthesis

<!-- To be filled during study -->

### Mental model

### Key concepts

### Cross-resource agreement / disagreement

### Practical checklist

### Gotchas and failure modes

### Experiments to run
