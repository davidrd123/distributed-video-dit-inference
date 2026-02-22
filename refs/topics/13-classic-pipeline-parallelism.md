# Topic 13: Classic PP (GPipe, PipeDream) — micro-batching, 1F1B schedule, bubble fraction

Pipeline parallelism partitions a model across devices by layer. **GPipe** fills the pipeline with micro-batches and synchronizes at the end (high memory, simple). **PipeDream's 1F1B schedule** interleaves one forward and one backward per micro-batch, reducing peak activation memory from O(num_microbatches) to O(num_stages).

## Resources

<!-- Resource IDs from manifest belonging to this topic -->
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| gpipe | GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism | high | pending |
| pipedream | PipeDream: Generalized Pipeline Parallelism for DNN Training | medium | pending |
| pipedream-2bw | Memory-Efficient Pipeline-Parallel DNN Training (PipeDream-2BW) | high | pending |
| pp-siboehm | Pipeline-Parallelism: Distributed Training via Model Partitioning | medium | pending |

## Synthesis

<!-- To be filled during study -->

### Mental model

### Key concepts

### Cross-resource agreement / disagreement

### Practical checklist

### Gotchas and failure modes

### Experiments to run
