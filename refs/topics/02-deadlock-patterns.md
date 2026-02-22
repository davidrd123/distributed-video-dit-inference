# Topic 2: Deadlock patterns in multi-group distributed code

Deadlocks in multi-group code almost always stem from mismatched collective ordering across ranks or accidentally issuing operations on the wrong process group. `NCCL_DEBUG=INFO` (or `WARN`) is your primary diagnostic tool.

## Resources

<!-- Resource IDs from manifest belonging to this topic -->
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| pytorch-distributed-api | PyTorch Distributed API Reference | medium | pending |
| pytorch-dist-tutorial | Writing Distributed Applications with PyTorch | medium | pending |
| nccl-fault-tolerance | Building Scalable and Fault-Tolerant NCCL Applications | low | pending |

## Implementation context

TP v0 is BSP lockstep; if ranks diverge around a collective you get an **NCCL hang** with no error until the default ~300s timeout. v0 mitigates this by preferring crash-over-hang: `TPAction.SHUTDOWN`, optional heartbeat NOOPs (`SCOPE_TP_HEARTBEAT_S`), and a worker watchdog (`SCOPE_TP_WORKER_WATCHDOG_S`, e.g. 30s). The PP plan applies the same principle: validate/pickle/spec on the mesh leader *before* any `mesh_pg` collective, and treat “wrong process group” as an instant deadlock risk.

See: `refs/implementation-context.md`, `scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md` (NCCL hang, Q7), `scope-drd/notes/FA4/h200/tp/pp-next-steps.md` (Step A5).

## Synthesis

<!-- To be filled during study -->

### Mental model

### Key concepts

### Cross-resource agreement / disagreement

### Practical checklist

### Gotchas and failure modes

### Experiments to run
