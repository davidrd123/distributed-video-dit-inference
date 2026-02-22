---
status: draft
---

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

Most distributed “deadlocks” in our stack are not subtle: they’re **one rank waiting for a communication partner that never arrives**. There are three root causes that account for almost all hangs:

1. **Ordering mismatch**: ranks call the same collective APIs, but in a different order (or one rank calls an extra collective). This is the classic TP lockstep failure: one rank arrives at `all_reduce`, the other never does. See: `scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md` (“NCCL Hang”, invariant).
2. **Membership mismatch (wrong process group)**: ranks call collectives in the same *order*, but on different *groups*. This is the PP1 “instant deadlock” failure mode: `mesh_pg` collectives must be executed by mesh ranks only, and rank0 must never participate. See: `scope-drd/notes/FA4/h200/tp/pp-next-steps.md` (Step A5 warning).
3. **Stranding (partial send/recv / half-protocol)**: one side sends part of a message (or just a header) and then throws before completing the rest; the peer blocks forever waiting for the next recv/broadcast. This is a control-plane deadlock, not a data-plane collective deadlock. See: `scope-drd/notes/FA4/h200/tp/pp-next-steps.md` (Step A1 anti-stranding) and `scope-drd/notes/FA4/h200/tp/5pro/10-v11-correctness-deadlock-audit/response.md` (FM-04).

Two design principles fall directly out of this:
- **Crash > hang**: fail-fast before entering collectives (or before sending a header that commits the peer to blocking). This is explicit in TP v0 (heartbeat + watchdog) and in PP plans (preflight before send). See: `scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md` (Q7), `scope-drd/notes/FA4/h200/tp/pp-next-steps.md` (A1/A2 timeouts).
- **Make the “plan” explicit**: the easiest way to get deadlocks is “optional behavior” that only some ranks take. Solve this by broadcasting a per-chunk plan and asserting it was followed, rather than letting each rank infer behavior locally. See: `scope-drd/notes/FA4/h200/tp/5pro/10-v11-correctness-deadlock-audit/response.md` (FM-01, FM-02, FM-03).

### Key concepts

Terms that matter for diagnosing deadlocks in TP + PP:

- **Collective ordering invariant**: “same collectives, same order, compatible tensors” across ranks in the participating group. This is the whole game for TP lockstep. See: `scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md` (“The Invariant”).
- **Process group / membership**: a collective is defined over a specific group. If any rank calls the collective on the wrong group (or calls it while a non-member rank calls a different collective), you can deadlock instantly. See: `scope-drd/notes/FA4/h200/tp/pp-next-steps.md` (Step A5 process groups).
- **NCCL “async” is stream enqueue + ordering contract**: NCCL operations are launched onto a CUDA stream (asynchronous enqueue); correctness and liveness still require a cross-rank agreement on *which* collective happens *when* (order + membership). Treat “async” as “not completed yet”, not “safe to reorder.” See: `refs/resources/nccl-user-guide.md` (CUDA stream semantics; ordering semantics) and `refs/resources/pytorch-cuda-semantics.md` (stream ordering + lifetime).
- **Group-rank vs global-rank discipline**: for group collectives, “source rank” (e.g., `src` for broadcasts) is the **rank within the process group**, not the global rank. Adopt an explicit convention (e.g., `mesh_pg` ranks `{1..N}` with global-rank 1 as group-rank 0) and never mix these namespaces in logs/assumptions. This is a common root cause of “wrong root” or “wrong group” hangs in PP1+ bringup.
- **Plan/role mismatch**: rank0 and workers must agree on which codepath is active (v0 full pipeline vs v1 generator-only vs PP0 vs PP1). A mismatch changes the collective sequence/call count. See: `scope-drd/notes/FA4/h200/tp/5pro/10-v11-correctness-deadlock-audit/response.md` (FM-01).
- **Call-count mismatch**: even if ordering is “the same,” different *counts* per chunk will hang (e.g., one rank runs `N` generator forwards and the other runs `N±1`). See: `scope-drd/notes/FA4/h200/tp/5pro/10-v11-correctness-deadlock-audit/response.md` (FM-02).
- **Conditional collectives**: any “if flag: do collective” is a deadlock trap unless the flag is parity-checked and identical across all ranks. Preferred pattern is “always execute the collective; optionally NOOP its payload.” See: `scope-drd/notes/FA4/h200/tp/5pro/10-v11-correctness-deadlock-audit/response.md` (FM-03).
- **Anti-stranding preflight**: validate/pickle/spec everything that might throw *before* sending any header/bytes that will cause the peer to block. See: `scope-drd/notes/FA4/h200/tp/pp-next-steps.md` (Step A1), `scope-drd/notes/FA4/h200/tp/5pro/10-v11-correctness-deadlock-audit/response.md` (FM-04).
- **Tensor manifest discipline**: all cross-rank tensors must travel via the explicit `tensor_specs` / manifest; ad-hoc “in-block broadcast with shape inference” is a correctness and deadlock footgun. See: `scope-drd/notes/FA4/h200/tp/5pro/10-v11-correctness-deadlock-audit/response.md` (FM-08).

### Cross-resource agreement / disagreement

Where the working notes are consistent:
- TP and PP bringup both reduce to the same operational rule: **make the distributed plan explicit, then execute it deterministically**. TP does this via per-chunk broadcast + lockstep generator collectives; PP does this via explicit stage-boundary contracts and send/recv discipline. See: `scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md`, `scope-drd/notes/FA4/h200/tp/pp-next-steps.md`.
- “Crash > hang” is the shared posture: heartbeats/watchdogs/timeouts exist because “waiting for the NCCL timeout” is not acceptable in an interactive system. See: `scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md` (Q7), `scope-drd/notes/FA4/h200/tp/pp-next-steps.md` (A2 watchdog/timeout guidance).

What’s different between TP and PP (and why you can’t copy patterns blindly):
- TP v0 is **BSP lockstep**: one control-plane broadcast, then both ranks run the same forward. The primary deadlock risk is divergence in control flow around in-generator collectives.
- PP introduces **multi-group topology + role isolation**: rank0-out-of-mesh means some ranks must *never* execute mesh collectives, and group creation + usage must be globally consistent. Wrong group is an “instant deadlock” class in PP1. See: `scope-drd/notes/FA4/h200/tp/pp-next-steps.md` (A5).

### Practical checklist

This is the “deadlock playbook” to apply any time you touch TP/PP distributed code.

**1) Enforce the invariant up front**
- Make “same collectives, same order” an explicit contract for every participating group (TP world, PP mesh). See: `scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md` (“The Invariant”).
- Any flag that can change collective behavior must be **env parity checked** (or transmitted in the per-chunk plan and asserted). See: `scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md` (env parity mention) and `scope-drd/notes/FA4/h200/tp/5pro/10-v11-correctness-deadlock-audit/response.md` (FM-03).

**2) Make a per-chunk phase plan (and verify it)**
- Broadcast and assert a **plan** that nails down call counts and optional phases:
  - `expected_generator_calls` (belt + suspenders)
  - recompute decisions (e.g., `do_kv_recompute`, `tp_do_kv_recompute`)
  - denoise step count / schedule
  See: `scope-drd/notes/FA4/h200/tp/pp-next-steps.md` (A2 validation gate), `scope-drd/notes/FA4/h200/tp/5pro/10-v11-correctness-deadlock-audit/response.md` (FM-02).

**3) Validate before you commit the peer to blocking**
- **Anti-stranding rule**: validate/pickle/spec everything *before* sending an INFER header or entering a broadcast loop. This includes:
  - `pickle.dumps(payload_meta)` (picklability)
  - tensor-spec dtype mapping preflight (no “throw after header”)
  - deterministic ordering of tensor specs
  See: `scope-drd/notes/FA4/h200/tp/pp-next-steps.md` (A1), `scope-drd/notes/FA4/h200/tp/5pro/10-v11-correctness-deadlock-audit/response.md` (FM-04, FM-07).

**4) Ban ad-hoc collectives inside blocks**
- No “in-block broadcasts” with worker-side shape inference. All tensors that cross ranks must go through the explicit tensor manifest (`tensor_specs`) so both sides allocate identically and fail-fast before any broadcast. See: `scope-drd/notes/FA4/h200/tp/5pro/10-v11-correctness-deadlock-audit/response.md` (FM-08).

**5) Multi-group rulebook (PP1+)**
- Treat “wrong group” as an **instant deadlock** class. The only safe posture is a strict rulebook + hard asserts/wrappers. See: `scope-drd/notes/FA4/h200/tp/pp-next-steps.md` (A5), `deep-research/2026-02-22/pp-rank0-out-of-mesh/reply.md`.
- Minimal conventions (copy/paste strict):
  - `world_pg`: all ranks. Used for startup + rank0↔leader p2p envelope/result traffic.
  - `mesh_pg`: ranks `{1..mesh_tp}` only. Used for leader→mesh broadcast and **all TP collectives** inside Phase B.
  - Convention: global rank 1 is mesh leader and is **mesh group-rank 0** (`src=0, group=mesh_pg`).
- Hard rules:
  1. Rank0 must never call a collective on `mesh_pg`.
  2. Mesh Phase B must never call a collective on `world_pg` (or the default group).
  3. Every TP collective wrapper must take an explicit `group=`; default-group collectives are forbidden in PP mode.
  4. No conditional collectives behind per-rank flags unless the flag is parity-checked (or carried in the per-chunk plan and asserted).

**6) Leader preflight + terminal actions (PP anti-stranding)**
- Leader must fully receive + validate the envelope **before** initiating any `mesh_pg` broadcast. If invalid, leader must broadcast a terminal action (e.g., `SHUTDOWN`/`ERROR`) rather than throwing and disappearing (which strands non-leaders in a collective). See: `deep-research/2026-02-22/pp-rank0-out-of-mesh/reply.md`.
- Same idea on rank0: **preflight-before-header** (picklability, dtype mapping, deterministic spec ordering, tensor materialization) must complete before sending an `INFER` commitment that makes peers block. See: `deep-research/2026-02-22/tp-v11-envelope-contract/reply.md`.

**7) Prevent cross-thread collective interleaving**
- “Same collectives, same order” is not just cross-rank: if one rank issues collectives from multiple threads, you can interleave `INFER` with `SHUTDOWN` and create a deadlock even when every other rank is correct.
- Rule: any codepath that touches distributed comms (TP broadcast, shutdown, heartbeat, PP leader broadcast) must share a single lock / single-flight guard. See: `scope-drd/notes/FA4/h200/tp/5pro/03-code-review/response-v1.md` (shutdown interleaving hazard).

**8) Timeouts and exit paths**
- Ensure there is a clean `SHUTDOWN` path and a watchdog/timeout story so you don’t wait the default NCCL timeout (~300s) to learn you deadlocked. See: `scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md` (Q7), `scope-drd/notes/FA4/h200/tp/pp-next-steps.md` (A2 `SCOPE_DIST_TIMEOUT_S`).

### Gotchas and failure modes

**Hang vs Franken-model (don’t misdiagnose)**
- **NCCL hang**: GPUs go idle, processes alive, last log line near a collective/broadcast; you’ll often get no error until timeout. Root cause: ordering/membership/call-count mismatch. See: `scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md` (“NCCL Hang”).
- **Franken-model**: system continues running but outputs drift/blur; collectives “line up” but ranks used different inputs/weights/state. Root cause: plan mismatch that didn’t change collective ordering (e.g., pipeline reload, cache reset divergence). See: `scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md` (“Franken-Model”), `scope-drd/notes/FA4/h200/tp/5pro/10-v11-correctness-deadlock-audit/response.md` (FM-10–FM-13).

**Short triage flow**
1. If it looks hung: check last log line → identify the collective (broadcast/all_reduce/send/recv) → suspect ordering/group mismatch first.
2. If it runs but looks wrong: suspect plan/state divergence (weights, inputs, cache epoch/reset, recompute decisions) and turn on drift tripwires (digest/fingerprint) before you chase performance.
3. In both cases: prefer to reproduce with fail-fast enabled (timeouts, watchdogs) rather than waiting for NCCL’s default timeout. See: `scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md` (triage checklist).

**Common “how did we get here” traps (from the audit)**
- Plan/role mismatch between rank0 and worker (FM-01) ⇒ hang.
- Generator call-count mismatch (FM-02) ⇒ hang.
- Conditional collectives added behind a flag not parity-checked (FM-03) ⇒ hang.
- Throw-after-header (partial broadcast) (FM-04) ⇒ worker stranded in recv ⇒ hang.
- Ad-hoc in-block broadcast / worker shape inference mismatch (FM-08) ⇒ hang/error.

### Experiments to run

Run these intentionally-broken experiments to verify your tripwires catch deadlocks quickly (crash > hang):

1. **Call-count mismatch**: force one rank to run `expected_generator_calls ± 1` (e.g., skip recompute once) and confirm you fail-fast before hanging (plan + assertion should catch). See: `scope-drd/notes/FA4/h200/tp/5pro/10-v11-correctness-deadlock-audit/response.md` (FM-02).
2. **Wrong process group**: in PP1 harness, intentionally issue one TP collective on `world_pg` instead of `mesh_pg` and confirm it deadlocks immediately (then add/verify guardrails that prevent this class). See: `scope-drd/notes/FA4/h200/tp/pp-next-steps.md` (A5).
3. **Conditional collective injection**: gate an `all_gather` behind a per-rank flag (without env parity) and confirm it hangs; then add the flag to parity keys and confirm it becomes deterministic (or refactor to “always collective, NOOP payload”). See: `scope-drd/notes/FA4/h200/tp/5pro/10-v11-correctness-deadlock-audit/response.md` (FM-03).
4. **Anti-stranding regression**: make `pickle.dumps(payload_meta)` fail (inject a non-picklable object) and confirm no header/bytes were sent and the peer does not block forever (preflight ordering). See: `scope-drd/notes/FA4/h200/tp/pp-next-steps.md` (A1), `scope-drd/notes/FA4/h200/tp/5pro/10-v11-correctness-deadlock-audit/response.md` (FM-04).
5. **Ad-hoc broadcast ban**: reintroduce a worker-side inferred-shape broadcast and confirm it is rejected by tests or lint rules; the only allowed transport should be via tensor manifest (`tensor_specs`). See: `scope-drd/notes/FA4/h200/tp/5pro/10-v11-correctness-deadlock-audit/response.md` (FM-08).
6. **Leader validation failure (no mesh stranding)**: corrupt an envelope (bad version or missing required tensor) and confirm the leader does **not** start broadcasting payload; it broadcasts a terminal action so non-leaders exit cleanly. See: `deep-research/2026-02-22/pp-rank0-out-of-mesh/reply.md` (break-it test 2).
7. **Wrong-group wrapper guard (PP1+)**: intentionally use `group=world_pg` for one Phase-B TP collective and confirm the wrapper detects mismatch and raises *before calling NCCL*, logs IDs + group, and triggers a clean mesh shutdown (no multi-minute hang). See: `deep-research/2026-02-22/pp-rank0-out-of-mesh/reply.md` (break-it test 1).
