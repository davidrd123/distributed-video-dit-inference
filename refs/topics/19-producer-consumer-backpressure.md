# Topic 19: Producer-consumer with backpressure — bounded channels, ring buffers

In a pipeline-parallel inference system, each stage is a producer for the next stage. Without backpressure, a fast producer can overwhelm a slow consumer, causing OOM or unbounded latency. **Bounded queues** are the simplest correct solution: block the producer when the queue is full.

## Resources

<!-- Resource IDs from manifest belonging to this topic -->
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| backpressure-explained | Backpressure explained — the resisted flow of data through software | low | pending |
| dist-systems-young-bloods | Notes on Distributed Systems for Young Bloods | low | pending |
| warpstream-rejection | Dealing with rejection (in distributed systems) | low | pending |

## Implementation context

The PP overlap design uses **bounded queues** with explicit depth limits: `D_in=2` envelopes in-flight toward mesh, `D_out=2` results queued for decode. The pass gate for PP overlap is `OverlapScore >= 0.30` (30% of smaller stage hidden). Without backpressure, rank0 could send envelopes faster than the mesh can consume them, leading to unbounded memory growth or stale results after hard cuts.

See: `refs/implementation-context.md` → Phase 3, `scope-drd/notes/FA4/h200/tp/pp0-bringup-runbook.md` Phase 2, `pp-control-plane-pseudocode.md`.

## Synthesis

<!-- To be filled during study -->

### Mental model

### Key concepts

### Cross-resource agreement / disagreement

### Practical checklist

### Gotchas and failure modes

### Experiments to run
