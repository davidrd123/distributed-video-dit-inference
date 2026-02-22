---
status: draft
---

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

### Mental model

Treat shutdown as part of the **distributed protocol**, not a cleanup detail.

In NCCL-backed PyTorch, if a rank is blocked inside a collective (or a paired `send/recv`) and its peer disappears, you often don’t get an immediate error — you get a **silent hang** until the process-group/NCCL timeout fires (often ~300s by default). So “graceful shutdown” has two distinct goals:

1. **Normal path (drain + exit):** stop producing new work, let in-flight work finish, then explicitly tell peers to exit their receive loops (`*_Action.SHUTDOWN`).
2. **Abnormal path (crash > hang):** when the normal path is impossible (peer crashed mid-op), prefer fast termination via **short timeouts + watchdogs** over waiting minutes for NCCL timeouts.

Scope’s control planes (TP v0 and PP0) use the same core pattern: a small, versioned header/envelope with an explicit `action` (`NOOP`/`INFER`/`SHUTDOWN`) that drives a state machine on the receiver. If the receiver always knows “am I about to receive a payload or not?”, then draining and shutdown become deterministic.

This is intentionally **crash-only** in bringup: the recovery path is “kill and restart `torchrun`,” not “try to unwind a partially-dead distributed program from inside Python.” That’s not pessimism — it’s the pragmatic response to how easily a single wedged rank can pin GPUs for minutes. (See: `deep-research/2026-02-22/external-patterns-and-resources/reply.md` → crash-only framing; and `scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md` Q7.)

### Key concepts

- **Explicit shutdown action (protocol-level):** `TPAction.SHUTDOWN` / `PPAction.SHUTDOWN` is the clean exit path. The worker treats it as a sentinel: break the loop, then attempt teardown (`destroy_process_group()`), but don’t rely on teardown being reliable under failure.
- **Heartbeat `NOOP`s (liveness vs idleness):** optional periodic `NOOP` headers during idle periods (`SCOPE_TP_HEARTBEAT_S>0`). This gives workers a monotonically updating “last seen leader” signal so they can distinguish “rank0 is idle” from “rank0 is dead.”
- **Watchdog hard-exit (`os._exit`) (breaking blocked NCCL):** a worker watchdog checks `time_since_last_header` and, if it exceeds a threshold (e.g. `SCOPE_TP_WORKER_WATCHDOG_S=30`), calls `os._exit(2)`. Rationale: a thread blocked in NCCL/PyTorch comms often can’t be interrupted cleanly; `os._exit` is the blunt instrument that guarantees the process dies instead of pinning the GPU for minutes.
- **Drain vs abort (make it explicit):**
  - **Drain** means “stop accepting new work, finish in-flight work, then send `SHUTDOWN` and exit.”
  - **Abort** means “we cannot drain safely (peer died / comm wedged), so exit quickly via timeout/watchdog and let the supervisor restart.”
  If you don’t name this distinction, systems accidentally do “half-drain” behavior that strands peers.
- **Bringup timeouts (fail fast):** during bringup, set shorter distributed timeouts (e.g. `SCOPE_DIST_TIMEOUT_S=60`) so “peer died” failures surface quickly instead of waiting out long defaults.
- **Anti-stranding invariants (preflight before header):** never send a header that commits the peer to waiting for more bytes unless you have already preflighted the entire message. Concretely:
  - Validate the meta/object payload is picklable and “tensor-free”.
  - Validate tensor specs (dtype support, shapes) and materialize/normalize tensors (device, contiguity) *before* emitting the `INFER` header.
  - Prefer “crash before broadcast” to “best-effort fallback after header” — post-header exceptions are how you strand a worker mid-message.

### Cross-resource agreement / disagreement

- **Agreement (PyTorch + local notes):** distributed execution is a lockstep program: once you enter a communicator’s receive/collective region, you’re relying on peers to do the matching operation. If a rank exits early, someone else can block until timeout.
- **Gap (official docs):** PyTorch exposes teardown primitives (`destroy_process_group`) but largely doesn’t provide a complete “graceful shutdown + draining” recipe for multi-rank applications under failure.
- **Local implementation stance:** Scope explicitly designs for “crash > hang” in bringup (short timeouts, watchdogs) and makes shutdown a first-class control-plane action (`*_Action.SHUTDOWN`) rather than relying on implicit teardown behavior.

### Practical checklist

1. **Define an explicit shutdown message in the contract** (`*_Action.SHUTDOWN`) and ensure it is handled as a first-class state transition on receivers (break loop → teardown).
2. **Make `NOOP` heartbeats and watchdogs an intentional pair**:
   - If you enable a watchdog, either enable `NOOP` heartbeat or set the watchdog threshold high enough that expected idle won’t trigger it.
   - Ensure “last header time” updates on *every* header (INFER and NOOP), not just “work”.
3. **Decide drain vs abort per shutdown path**:
   - Normal operator shutdown should drain with a deadline (finish in-flight work up to some bound), then send `SHUTDOWN`.
   - Any “peer disappeared / comm wedged” condition should flip to abort (timeout + watchdog + restart), not “wait for cleanup to succeed.”
4. **Preflight before sending any header/preamble** (“don’t strand the peer mid-message”):
   - Sender: validate/pickle/spec every field before emitting an `INFER` header.
   - Receiver: allocate based on transmitted specs (not on local assumptions).
   - Policy: treat preflight failures as fatal during bringup (crash and restart the `torchrun` job).
5. **Use shorter distributed timeouts during bringup** (e.g. `SCOPE_DIST_TIMEOUT_S=60`) so peer-loss manifests as a quick error instead of a multi-minute stall.
6. **Implement explicit draining semantics where queues exist (PP overlap)**:
   - Stop enqueueing new work first.
   - Ensure any bounded queues have a sentinel/close mechanism so blocked producers/consumers wake up on shutdown.
   - Only exit rank0 after in-flight envelopes have been accounted for (results received or explicitly abandoned by policy).
7. **Have a last-resort kill path on workers** (`os._exit`) for cases where the receiver is blocked in comms and cooperative shutdown cannot run.

### Gotchas and failure modes

- **Orphaned worker pinned for timeout:** the canonical failure is rank0 crashing and the worker blocking in `recv_next()` waiting for a broadcast that will never come. Without watchdog/short timeouts, the worker can sit for the full default timeout (often ~300s), holding GPU memory.
- **Post-header exception strands peers:** if rank0 sends an `INFER` header and then throws before sending the object meta / tensors, the worker has “committed” to receiving a payload and may block deep in comms. This is exactly what “preflight before header” is meant to prevent.
- **Watchdog without heartbeat kills healthy idle:** a watchdog that measures “time since last header” will fire during normal idle gaps unless heartbeats are enabled or thresholds are tuned.
- **`os._exit` is intentionally harsh:** it skips Python cleanup and can leave logs/IO buffers unflushed, but it reliably terminates a process that might otherwise be unkillable from within Python due to blocked NCCL ops.
- **`destroy_process_group()` is best-effort, not a guarantee:** teardown can hang even when the workload “ran fine,” and CUDA graph capture can introduce additional teardown sharp edges. Treat teardown as “attempted cleanup” and ensure your watchdog/timeout story still bounds time-to-exit. (See: resource stub `pytorch-issue-115388` in this topic; and `scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md` Q7 for the bringup posture.)
- **Teardown can hang even when metrics look perfect:** successful throughput/latency metrics do not guarantee clean shutdown. The TP bringup run log notes that a matrix runner sends SIGTERM shortly after writing JSON outputs specifically to avoid **intermittent teardown hangs**; the resulting SIGTERM noise from `torchrun` does not invalidate the measured metrics, it just highlights that teardown is its own failure surface.

### Experiments to run

1. **Kill rank0 mid-recv (peer disappears while other rank is blocking):** start TP/PP, ensure the non-leader is in a blocking receive, then `kill -9` rank0. Expect: worker exits via watchdog (`os._exit`) quickly (tens of seconds), not after a multi-minute default timeout.
2. **Kill worker mid-send / mid-collective:** terminate the worker while it is in a send/broadcast/collective region. Expect: rank0 errors out within `SCOPE_DIST_TIMEOUT_S` (bringup) and does not hang indefinitely.
3. **SIGTERM the `torchrun` job:** send SIGTERM to the parent `torchrun` and confirm both ranks exit promptly. Expect: shutdown action path runs when possible; otherwise watchdog/timeout prevents long hangs.
4. **SIGINT/Ctrl+C handling under `torchrun`:** run a short harness loop and press Ctrl+C. Expect: rank0 emits `SHUTDOWN` (best-effort), workers exit receive loops, and the whole job exits promptly (no minutes-long “waiting for NCCL timeout”).
4. **Timeout regression check:** repeat (1)–(3) with “bringup” timeouts vs longer defaults and measure wall-clock time-to-exit. Expect: short timeouts + watchdog cap failure duration; long defaults can leave you waiting minutes.
5. **PP drain correctness on shutdown:** run PP with bounded queues (`SCOPE_PP_D_IN`/`SCOPE_PP_D_OUT` > 1), initiate shutdown while work is in-flight, and verify the `SHUTDOWN` path drains/flushes queues (no stuck producers/consumers) and both processes exit without hanging.
