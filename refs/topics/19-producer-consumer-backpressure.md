# Topic 19: Producer-consumer with backpressure — bounded channels, ring buffers

In a pipeline-parallel inference system, each stage is a producer for the next stage. Without backpressure, a fast producer can overwhelm a slow consumer, causing OOM or unbounded latency. **Bounded queues** are the simplest correct solution: block the producer when the queue is full.

## Resources

<!-- Resource IDs from manifest belonging to this topic -->
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| backpressure-explained | Backpressure explained — the resisted flow of data through software | low | pending |
| dist-systems-young-bloods | Notes on Distributed Systems for Young Bloods | low | pending |
| warpstream-rejection | Dealing with rejection (in distributed systems) | low | pending |

## Synthesis

<!-- To be filled during study -->

### Mental model

### Key concepts

### Cross-resource agreement / disagreement

### Practical checklist

### Gotchas and failure modes

### Experiments to run
