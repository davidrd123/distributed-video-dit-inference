# Operator Test Matrix — TP v1.1 + PP0/PP1 + compile hardening

Purpose: a shared, test-driven “definition of done” for changes that touch **distributed control planes**, **multi-process-group execution**, **KV-cache lifecycle**, and **compiled distributed regions**. Every test here is designed to prove **crash > hang** and to produce actionable logs (`call_id`, `chunk_index`, `cache_epoch`, group).

Primary operator-manual references:
- `refs/topics/20-message-framing-versioning.md` (framing + anti-stranding)
- `refs/topics/02-deadlock-patterns.md` (ordering/membership/call-count)
- `refs/topics/19-producer-consumer-backpressure.md` (queues + overlap proof)
- `refs/topics/22-kv-cache-management.md` (lifecycle contract + R1/R0a/R0)
- `refs/topics/03-graceful-shutdown.md` (watchdogs/timeouts + drain vs abort)
- `refs/topics/12-compile-distributed-interaction.md` (compile parity + regression suite)

Deep Research outputs these tests are distilled from:
- `deep-research/2026-02-22/tp-v11-envelope-contract/reply.md`
- `deep-research/2026-02-22/pp-rank0-out-of-mesh/reply.md`
- `deep-research/2026-02-22/compile-distributed-hardening/reply.md`
- `deep-research/2026-02-22/kv-cache-lifecycle/reply.md`

## Conventions

- **Commitment point**: any action that causes the peer to block waiting for more bytes/collectives (e.g., broadcasting an `INFER` header, starting a `mesh_pg` broadcast loop, entering Phase B collectives). Log the IDs at every commitment point.
- **Anti-stranding rule**: anything that can throw must happen *before* the commitment point.
- **FM mapping**: uses the v1.1 deadlock audit taxonomy where possible (FM-01/FM-02/FM-03/FM-04/FM-07/FM-08/…); PP-only cases add “PP-FM-*” labels.

## Matrix (P0 correctness tests)

Each row includes: the smallest setup, how to break it, and what “good failure” looks like.

| ID | Surface | FM | Setup | Fault injection | Expected fast-fail signature | Where to assert / log | References |
|---|---|---|---|---|---|---|---|
| OM-01 | TP/PP framing | FM-04 | TP broadcast or PP p2p send | Make meta unserializable (e.g., lambda) so `pickle.dumps(meta)` fails | Sender fails **before** sending/bcasting `INFER` header; peer does not block waiting for payload | Rank0 preflight; log `call_id/chunk_index/cache_epoch` and failing key path | `refs/topics/20-message-framing-versioning.md`, `refs/topics/02-deadlock-patterns.md` |
| OM-02 | TP/PP framing | FM-07 | TP broadcast or PP p2p send | Include a tensor spec with unsupported dtype (e.g., `torch.bool`) | Sender fails **before** header; no worker/leader stranding in receive loop | Rank0 preflight dtype-map; log spec key + dtype | `refs/topics/20-message-framing-versioning.md` |
| OM-03 | TP v1.1 plan | FM-02 | `tp_plan=v1_generator_only` | Make phase plan inconsistent (e.g., `tp_num_denoise_steps != len(denoising_step_list)` or wrong `tp_expected_generator_calls`) | Rank0 rejects envelope pre-bcast, or worker rejects pre-Phase-B; **no hang** | Rank0 preflight + worker preflight; log expected vs computed | `refs/topics/20-message-framing-versioning.md` (worked example), `refs/topics/02-deadlock-patterns.md` |
| OM-04 | TP/PP plan parity | FM-01 | Any distributed mode | Mismatch `tp_plan` / PP enabled across ranks | Env-parity / startup handshake fails **before** first inference | Init parity check; print diff of key/value by rank | `refs/topics/02-deadlock-patterns.md`, `refs/topics/12-compile-distributed-interaction.md` |
| OM-05 | Conditional collectives | FM-03 | TP=2 or PP1 mesh | Gate a debug collective behind per-rank flag mismatch | Fail at env parity (preferred) or crash before first gated collective; **no intermittent hang** | Env parity keys; if runtime: crash before collective boundary | `refs/topics/02-deadlock-patterns.md` |
| OM-06 | Ad-hoc broadcasts | FM-08 | TP v1.1 bringup | Reintroduce an in-block broadcast with worker-side shape inference | Fail-fast (lint/test) or crash pre-collective; never “sometimes hangs” | Code review rule + runtime assert in v1 mode | `refs/topics/02-deadlock-patterns.md`, `refs/topics/20-message-framing-versioning.md` |
| OM-07 | PP leader safety | PP-FM-LEADER-THROW | PP1 (rank0 + leader + ≥1 non-leader) | Corrupt envelope so leader rejects it after recv | Leader does **not** start payload broadcast; broadcasts terminal action (`SHUTDOWN`/`ERROR`) so non-leaders exit cleanly | Leader “validate-before-bcast” + terminal broadcast path; log IDs + reason | `refs/topics/20-message-framing-versioning.md` |
| OM-08 | Wrong group collective | PP-FM-WRONG-GROUP | PP1 with `mesh_pg` + TP>1 inside mesh | Intentionally call one TP collective on `world_pg` instead of `mesh_pg` | Wrapper detects mismatch and crashes **before calling NCCL**; leader coordinates clean exit | Collective wrappers must take explicit group; log `group_used` vs expected | `refs/topics/02-deadlock-patterns.md` (group discipline), `deep-research/2026-02-22/pp-rank0-out-of-mesh/reply.md` |
| OM-09 | KV lifecycle (epoch fence) | FM-10 / replay | PP0 with `D_out=2` | Delay a result; trigger hard cut (epoch++) before delayed result arrives | Rank0 drops stale result by `cache_epoch` mismatch; never decodes/emits it | Rank0 acceptance rules; log dropped IDs + epochs | `refs/topics/21-idempotency-and-replay.md`, `refs/topics/22-kv-cache-management.md`, `refs/topics/19-producer-consumer-backpressure.md` |
| OM-10 | KV recompute override | FM-06 | TP v1.1c or PP0/PP1 with recompute enabled | Set `do_kv_recompute=True` but omit `context_frames_override/context_frames` | Fail pre-send or pre-Phase-B; **no fallback** to VAE path on mesh/worker | Rank0 preflight; leader validate-before-bcast; log “override required” | `refs/topics/22-kv-cache-management.md` |
| OM-11 | Drift tripwire | Franken-model | TP mode | Perturb one broadcast tensor on rank0 only (test-only) | Crash with crisp digest mismatch diff; no “it looks worse but keeps running” | Input digest parity (bringup); log mismatch summary | `refs/topics/04-determinism-across-ranks.md`, `refs/topics/22-kv-cache-management.md` |
| OM-12 | Weight divergence tripwire | Franken-model | TP mode | Mutate a weight shard on one rank (test-only) | Crash on shard fingerprint mismatch | Shard fingerprint check; log both fingerprints | `refs/topics/04-determinism-across-ranks.md` |
| OM-13 | Orphan shutdown | Hang hygiene | TP/PP worker roles | Kill rank0 abruptly mid-idle/mid-loop | Worker exits within watchdog window; no 300s wait | Heartbeat + watchdog; log last header time | `refs/topics/03-graceful-shutdown.md` |

## Matrix (compile + distributed regression tests)

These protect the “TP=2+compile baseline” from regressing into graph fragmentation or rank-asymmetric compilation.

| ID | Surface | Setup | Fault injection | Expected | References |
|---|---|---|---|---|---|
| C-01 | Compile parity | TP=2 + compile | Enable compile on one rank only | Startup compile parity handshake fails (no hang) | `refs/topics/12-compile-distributed-interaction.md` |
| C-02 | Graph-break gate | `scope-drd/scripts/tp_compile_repro.py` | Introduce `torch._dynamo.disable()` around collectives | Mode C fails (break count > 0 / graph explosion), treated as regression | `refs/topics/12-compile-distributed-interaction.md` |
| C-03 | Rank-symmetric compile health | TP=2 warmup | Induce rank-only graph break (e.g., debug print on one rank) | Post-warmup gather of `graph_breaks/unique_graphs` differs → crash early | `refs/topics/12-compile-distributed-interaction.md` |
| C-04 | Effective backend parity | TP=2 | Force `auto` to resolve differently per rank (test-only) | Startup “effective backend” assert fails | `refs/topics/12-compile-distributed-interaction.md`, `refs/topics/04-determinism-across-ranks.md` |

## Matrix (overlap/backpressure proof)

These validate that PP overlap is real, bounded, and doesn’t silently serialize.

| ID | Surface | Setup | What to measure | Expected | References |
|---|---|---|---|---|---|
| O-01 | D sweep | PP0 | Sweep `D_in/D_out ∈ {1,2,3,4}` | `D=1` collapses overlap; `D=2` is first stable overlap; memory/latency rise with D | `refs/topics/19-producer-consumer-backpressure.md` |
| O-02 | Overlap signature | PP0 overlap | `OverlapScore`, `t_mesh_idle_ms`, queue depths | Healthy: `period≈max(Stage0,Stage1)`, small `t_mesh_idle_ms`; Bug: `t_mesh_idle_ms` tracks decode and queues ~0 | `refs/topics/19-producer-consumer-backpressure.md` |
| O-03 | Recompute coupling delta | PP0/PP1 | Compare R1 vs R0a | OverlapScore drops when R0a introduces dependency; quantify and record | `refs/topics/22-kv-cache-management.md`, `deep-research/2026-02-22/pp-rank0-out-of-mesh/reply.md` |

## Notes / gaps (what likely needs explicit test hooks)

- Some tests (wrong-group collective; rank-only graph break injection; weight mutation) require explicit test-only hooks or harness modes. Keep them gated behind parity-checked env vars (so tests don’t create conditional collectives).
- If you add an explicit `ERROR` action for PP leader terminal broadcasts, bump envelope version and treat it as a protocol change (Topic 20 rules apply).

