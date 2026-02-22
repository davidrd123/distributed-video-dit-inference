# Topic 21: Idempotency and replay

For fault tolerance in a streaming video pipeline, operations should be idempotent — re-executing a denoising step or VAE decode with the same inputs produces the same output. Combined with **replay from checkpointed state**, this gives you exactly-once semantics without distributed transactions.

## Resources

<!-- Resource IDs from manifest belonging to this topic -->
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| idempotency-dist | What is Idempotency in Distributed Systems? | low | pending |
| exactly-once | Exactly Once in Distributed Systems | low | pending |

## Synthesis

<!-- To be filled during study -->

### Mental model

### Key concepts

### Cross-resource agreement / disagreement

### Practical checklist

### Gotchas and failure modes

### Experiments to run
