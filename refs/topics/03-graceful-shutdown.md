# Topic 3: Graceful shutdown and draining in distributed PyTorch

Graceful shutdown in distributed PyTorch remains **underserved by documentation**. The core API is `destroy_process_group()`, but real-world challenges include ranks exiting at different times, CUDA graph capture preventing clean NCCL communicator destruction, and signal handling under `torchrun`.

## Resources

<!-- Resource IDs from manifest belonging to this topic -->
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| pytorch-distributed-api | PyTorch Distributed API Reference | medium | pending |
| pytorch-issue-115388 | GitHub Issue #115388: destroy_process_group() hangs after CUDA graph capture | medium | pending |
| pytorch-issue-167775 | GitHub Issue #167775: Graceful Ctrl+C handling from torchrun | low | pending |
| kill-pytorch-dist | Kill PyTorch Distributed Training Processes | low | pending |

## Implementation context

TP v0 needed an explicit shutdown protocol because orphaned workers blocked in `recv_next()` can sit for up to the default **300s NCCL timeout** after rank0 exits. The implemented pattern is a versioned header with `*_Action.SHUTDOWN`, plus optional heartbeat NOOPs and a watchdog (`SCOPE_TP_WORKER_WATCHDOG_S`, e.g. 30s) that `os._exit(2)`s to break out of a blocked NCCL call. PP0 reuses the same idea (`PPAction.SHUTDOWN`) and recommends short `SCOPE_DIST_TIMEOUT_S=60` during bringup.

See: `refs/implementation-context.md`, `scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md` (Q7), `scope-drd/notes/FA4/h200/tp/pp0-bringup-runbook.md` (Phase 0-1 env + shutdown).

## Synthesis

<!-- To be filled during study -->

### Mental model

### Key concepts

### Cross-resource agreement / disagreement

### Practical checklist

### Gotchas and failure modes

### Experiments to run
