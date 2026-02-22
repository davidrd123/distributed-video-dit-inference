# Topic 15: Pipeline scheduling theory — bubble fraction derivation, Little's law connection

The bubble fraction for a P-stage pipeline processing B micro-batches is **(P-1)/(B+P-1)**. This is a direct consequence of pipeline startup and drain latency. **Little's law** (L = lambda * W) provides the framework: to keep P stages busy, you need at least P micro-batches in flight, and throughput approaches the ideal rate as B approaches infinity.

## Resources

<!-- Resource IDs from manifest belonging to this topic -->
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| jax-scaling-book | How to Parallelize a Transformer for Training (JAX Scaling Book) | medium | pending |
| megatron-trillion-params | Scaling Language Model Training to a Trillion Parameters Using Megatron | medium | pending |
| megatron-pp-schedules | Megatron-LM Pipeline Parallel Schedules source | medium | pending |

## Synthesis

<!-- To be filled during study -->

### Mental model

### Key concepts

### Cross-resource agreement / disagreement

### Practical checklist

### Gotchas and failure modes

### Experiments to run
