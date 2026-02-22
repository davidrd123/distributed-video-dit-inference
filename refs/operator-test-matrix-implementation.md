---
status: draft
---

# Operator Test Matrix — Implementation Guide (scope-drd)

This is the “how” companion to `refs/operator-test-matrix.md` (the “what”). It maps each matrix row to a **runnable harness** in `scope-drd/` and calls out any **test hooks / code fixes** needed to make the test meaningful.

Goal: an H200-side agent can pull both repos and implement a correctness-first operator gate without re-deriving the plan from chat history.

## Where this lives (in scope-drd)

Recommended layout:

- `scope-drd/scripts/operator_matrix/` — one small harness per failure surface (`om_*.py`)
- `scope-drd/scripts/run_operator_matrix.py` — a tiny runner that:
  - shells out to `uv run torchrun ...` (or `python -m torch.distributed.run`)
  - enforces **per-test timeouts**
  - prints a short pass/fail summary

This should feel like `scope-drd/scripts/tp_v11_matrix.py`, but “correctness-first”: every failure mode must **crash fast** or **cleanly shutdown peers**, never hang.

## Tranche split (staging)

Implement in increasing-risk tranches:

1. **Tranche 0: CPU / `gloo` protocol + contract tests**
   - No heavy pipeline load.
   - Proves anti-stranding ordering, env parity, schema validation, dedupe/epoch filtering.
   - Expected to run on any dev box.

2. **Tranche 1: GPU / `nccl` runtime tests**
   - Uses real CUDA tensors and NCCL collectives.
   - Proves “crash > hang” under realistic transport.

3. **Tranche 2: PP0 overlap + epoch fencing**
   - Requires PP0 machinery (rank0↔rank1 p2p) plus bounded queues (`D_in/D_out`) and acceptance rules (`cache_epoch`/monotonic IDs).
   - Proves overlap/backpressure and “drop stale results” behavior in the simplest PP topology.

4. **Tranche 3: PP1 (rank0-out-of-mesh + `mesh_pg`)**
   - Requires PP1 machinery (leader broadcast + terminal action + wrong-group guards).
   - Add once PP0 is rock solid.

## Test-only hook conventions (non-negotiable)

- All test-only behavior must be gated behind `SCOPE_TEST_*` env vars that are **env-parity-checked** at init.
  - No per-rank test flags; that creates conditional collectives.
- Any hook that could strand a peer must either:
  - fail **before the commitment point** (before sending/broadcasting a header / before entering a broadcast loop), or
  - ensure a **terminal action** (`SHUTDOWN` or `ERROR`) is sent/broadcast so peers exit cleanly.
- Runner policy: if a test fails on rank0, it should still attempt a best-effort `SHUTDOWN` (or kill the torchrun job) so “test failure” doesn’t become “GPU wedged for 300s.”

## Mapping table (matrix row → harness/hook)

Legend:
- **Script** paths are in `scope-drd/` (not this repo).
- “Hook needed” means you’ll likely add a small, parity-gated test-only branch in runtime code.

| Matrix ID | Tranche | Script (proposed) | Exercises | Hook / fix needed | Notes |
|---|---:|---|---|---|---|
| OM-01 meta unserializable | 0 | `scripts/operator_matrix/om_tp_preflight_meta.py` | TP control-plane preflight ordering | (optional) PP needs meta injection hook | TP can construct an unserializable meta easily; PP contract objects may be “too clean” without a test-only escape hatch. |
| OM-02 unsupported dtype | 0 | `scripts/operator_matrix/om_tp_preflight_dtype.py` | dtype/spec preflight before commitment | **PP fix**: sender-side dtype preflight in `pp_control.py` | This is a real correctness gap if PP can send a header then fail on dtype after commit. |
| OM-03 plan mismatch | 0 | `scripts/operator_matrix/om_tp_v11_plan_mismatch.py` | phase-plan / `expected_generator_calls` consistency | maybe a v1.1-only harness entrypoint | Should fail pre-bcast or pre-Phase-B (never hang). |
| OM-04 plan parity | 0 | `scripts/operator_matrix/om_env_parity_mismatch.py` | env parity check at init | none | Intentionally set parity keys differently per rank and assert init fails. |
| OM-05 conditional collectives | 0 | `scripts/operator_matrix/om_env_parity_collective_gate.py` | “collective-gating flags must be parity-checked” | none | This is really a policy enforcement test: parity mismatch must fail before any collective. |
| OM-06 ad-hoc broadcasts | 0 | (lint-ish) `scripts/operator_matrix/om_tp_v11_manifest_only.py` | “all tensors must flow via tensor_specs manifest” | none (or static check) | This can start as a “contract test” rather than runtime-only. |
| OM-10 missing recompute override | 0 | `scripts/operator_matrix/om_pp_recompute_override_required.py` and `om_tp_v11_recompute_override_required.py` | “no fallback” + override-required semantics | may need v1.1 harness hook | Keep TP v1.1 and PP paths distinct; failure should be pre-commit if possible. |
| OM-11 input digest mismatch | 0/1 | `scripts/operator_matrix/om_tp_input_digest_mismatch.py` | drift tripwire (Franken-model prevention) | **Hook**: parity-gated tensor mutation at a defined point | Preferred: mutate on receiver side after receive/before digest to reduce risk of breaking sender framing. |
| OM-12 weight fingerprint drift | 0 | `scripts/operator_matrix/om_tp_shard_fingerprint_drift.py` | shard fingerprint tripwire | none (may need a minimal model fixture) | Keep it lightweight; avoid full pipeline load. |
| OM-13 orphan shutdown | 1 | `scripts/operator_matrix/om_orphan_watchdog_exit.py` | watchdog / heartbeat / timeout behavior | none (if watchdog exists) | Kill rank0 mid-loop and verify workers exit within watchdog window. |
| C-01 compile parity | 0/1 | `scripts/operator_matrix/c_compile_parity.py` | compile enabled on one rank only | none | Should fail at startup before any collective program diverges. |
| C-02 graph-break gate | 0/1 | reuse `scripts/tp_compile_repro.py` | compile regression invariant | none | Keep this as the canonical gate; don’t duplicate. |
| C-03 rank-symmetric compile health | 0/1 | `scripts/operator_matrix/c_rank_symmetric_compile_health.py` | counters gathered after warmup | none (if counters already logged) | If not present, add a small gather-and-assert function at end of warmup. |
| C-04 effective backend parity | 0 | `scripts/operator_matrix/c_effective_backend_parity.py` | “auto resolved backend must match across ranks” | none | Needs a way to force divergent resolution in test-only mode. |
| O-01 D sweep | 2 | `scripts/operator_matrix/o_pp0_d_sweep.py` | overlap/buffer depth effects | PP0 overlap instrumentation | This becomes meaningful once PP0 overlap is implemented. |
| O-02 overlap signature | 2 | `scripts/operator_matrix/o_pp0_overlap_signature.py` | `t_mesh_idle_ms` and OverlapScore | PP0 metric plumbing | Requires the timestamping described in `pp0-bringup-runbook.md`. |
| O-03 recompute coupling delta | 2 | `scripts/operator_matrix/o_pp0_recompute_coupling_delta.py` | R1 vs R0a overlap delta | PP0 R0a implemented | Captures the “does recompute re-serialize us?” question. |
| OM-09 epoch fence / stale drop | 2 | `scripts/operator_matrix/om_pp0_epoch_fence_drop.py` | drop stale results after hard cut | PP0 queueing + acceptance rules | This is PP0: it requires `cache_epoch` filtering and `D_out>1` to create the “late result arrives after epoch bump” scenario. |
| OM-07 PP leader safety | 3 | (future) `scripts/operator_matrix/pp1_leader_terminal_bcast.py` | leader validate-before-bcast + terminal action | PP1 required | Requires mesh leader broadcast + terminal action path. |
| OM-08 wrong group collective | 3 | (future) `scripts/operator_matrix/pp1_wrong_group_guard.py` | wrong-group detection before NCCL call | PP1 required | Requires `mesh_pg` vs `world_pg` plumbing and explicit group handles in wrappers. |

## Dependency order (recommended)

1. Runner skeleton (`run_operator_matrix.py`) + per-test timeout policy.
2. Env parity harnesses (OM-04/OM-05) — establishes the “no conditional collectives” discipline.
3. Sender-side preflight ordering (OM-01/OM-02) — proves “anti-stranding” on both TP and PP.
4. Plan/call-count mismatch tests (OM-03/OM-10).
5. Drift tripwires (OM-11/OM-12).
6. Orphan/shutdown hygiene (OM-13).
7. PP0 epoch/overlap tests (OM-09 + O-*) only after PP0 queueing/metrics exist.

## Notes (what not to overbuild)

- Don’t implement a heavy DSL for the runner; keep it as “list of test invocations + timeouts + summary.”
- Prefer **CPU/gloo** versions of tests first. The operator value is “it crashes fast,” not “it’s the fastest GPU test suite.”
- Keep the test hooks small and explicit; every hook is a future footgun unless it is parity-gated and isolated.
