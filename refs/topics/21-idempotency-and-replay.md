---
status: stub
---

# Topic 21: Idempotency and replay

For fault tolerance in a streaming video pipeline, operations should be idempotent — re-executing a denoising step or VAE decode with the same inputs produces the same output. Combined with **replay from checkpointed state**, this gives you exactly-once semantics without distributed transactions.

## Resources

<!-- Resource IDs from manifest belonging to this topic -->
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| idempotency-dist | What is Idempotency in Distributed Systems? | low | pending |
| exactly-once | Exactly Once in Distributed Systems | low | pending |

## Implementation context

In TP v0, “replay” is intentionally coarse: pipeline reload and snapshot restore are blocked, so the safe recovery path for both hangs and Franken-models is restarting the `torchrun` job (v0 contract). In PP bringup, we reintroduce limited replay/idempotency via metadata: `call_id` must be monotonic, and `cache_epoch` increments on hard cuts so rank0 can drop stale results and flush bounded queues deterministically. These are the primitives needed before attempting any richer at-least-once / retry semantics.

See: `refs/implementation-context.md`, `scope-drd/notes/FA4/h200/tp/explainers/07-v0-contract.md` (no reload/snapshot), `scope-drd/notes/FA4/h200/tp/pp-next-steps.md` (cache_epoch filtering), `scope-drd/notes/FA4/h200/tp/pp-topology-pilot-plan.md` (hard cut flush rules).

Relevant Scope code:
- `scope-drd/src/scope/core/distributed/control.py` (TP headers: monotonic `call_id`, epoch fields, action framing)
- `scope-drd/src/scope/core/distributed/pp_contract.py` (`call_id`, `chunk_index`, `cache_epoch` as ordering/replay primitives)
- `scope-drd/scripts/pp_two_rank_pipelined.py` (queue flush/drop behavior that uses `cache_epoch` during bringup)

## Synthesis

<!-- To be filled during study -->

### Mental model

### Key concepts

### Cross-resource agreement / disagreement

### Practical checklist

### Gotchas and failure modes

### Experiments to run
