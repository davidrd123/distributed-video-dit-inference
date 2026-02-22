# KV cache lifecycle operator manual

Scope: streaming video DiT KV-cache in **TP v0/v1.1** and **PP rank0-out-of-mesh**. This is “operator manual” style: explicit state machine, deterministic contract fields, and break-it tests that fail fast.

---

## Delta vs prior 5Pro threads 10, 12, 13

### Unchanged

* **Two catastrophe classes still dominate**: NCCL hang (collective order mismatch) vs Franken-model (collectives align but inputs/weights/state differ). (See `scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md`.)
* **Lockstep remains the only correctness story for TP**: KV contents are never broadcast; consistency comes from identical `call_params` and identical execution order. (`05-kv-cache-head-sharding.md`.)
* **“Crash > hang” remains policy**: validate everything before sending any header or entering any collective. (`5pro/10-v11-correctness-deadlock-audit/response.md`, `refs/topics/20-message-framing-versioning.md`.)
* **Recompute coupling is real**: steady-state recompute wants an anchor derived from `decoded_frame_buffer` via VAE re-encode, which is why “generator-only workers” and PP need an override tensor (R0a). (`05`, `06`, `v1.1-generator-only-workers.md`, `pp-topology-pilot-plan.md`.)
* **Recompute frequency is not a free knob**: `SCOPE_KV_CACHE_RECOMPUTE_EVERY=2` degraded quality and did not improve net chunk time (Run 11). (`bringup-run-log.md`.)

### Refined

* The cache lifecycle is now stated as an explicit per-chunk **state machine with phases**, and the protocol fields are grouped by “what can cause a hang” vs “what can cause Franken-model.”
* Naming drift is pinned with a **canonical mapping** between TP v1.1 and PP contracts (call_id, epoch, recompute override tensor).
* Added a small set of **state-digest** style checks (scalars only) that catch lifecycle divergence earlier than “watch output quality.”

### Overridden

* Treat “decoded-anchor recompute” as a **first-class protocol dependency** in PP and v1.1c. Earlier it could read like a pipeline-internal detail. It is not. If you do not contract it, you will eventually ship silent drift.

---

## Mental model

The KV cache is the generator’s **hidden state across chunks**. In Scope it is:

* A fixed-size **ring buffer over tokens** (frames expand to `frame_seq_length` tokens).
* **Head-sharded** under TP: each rank stores only its head slice, so you cannot “sync caches” by copying (it is ~13 GB per rank at TP=2 for the full model). (`05-kv-cache-head-sharding.md`.)
* Correctness-critical because divergence is usually **silent** (Franken-model): you keep running and the video degrades.

Operational rule: **cache lifecycle is part of the distributed protocol**, not an internal optimization.

---

## Cache lifecycle state machine

### State variables

These are the variables that define KV-cache lifecycle across chunks. Some are “derived” but still worth checking.

| State variable                                    | Meaning                                               |          Owner in TP |                  Owner in PP | Must match across TP ranks / mesh ranks |
| ------------------------------------------------- | ----------------------------------------------------- | -------------------: | ---------------------------: | --------------------------------------: |
| `cache_epoch`                                     | hard-cut generation counter                           |      rank0 + workers |                 rank0 + mesh |                                   ✅ yes |
| `call_id`                                         | monotonic message id                                  |      rank0 + workers |                 rank0 + mesh |                                   ✅ yes |
| `chunk_index`                                     | monotonic chunk id                                    |      rank0 + workers |                 rank0 + mesh |                                   ✅ yes |
| `current_start_frame`                             | where this chunk starts in stream                     |      rank0 + workers | rank0 decides, mesh executes |                                   ✅ yes |
| `init_cache`                                      | “hard cut” flag                                       |            broadcast |                  in envelope |                                   ✅ yes |
| `reset_kv_cache`                                  | explicit KV reset decision                            |            broadcast |                  in envelope |                                   ✅ yes |
| `reset_crossattn_cache`                           | explicit cross-attn reset decision                    |            broadcast |                  in envelope |                                   ✅ yes |
| `do_kv_recompute`                                 | whether recompute generator call runs                 |            broadcast |                  in envelope |                                   ✅ yes |
| `num_denoise_steps`                               | denoise steps this chunk                              |            broadcast |                  in envelope |                                   ✅ yes |
| `expected_generator_calls`                        | `(do_kv_recompute?1:0)+num_denoise_steps`             |            broadcast |                  in envelope |                                   ✅ yes |
| `kv_cache_attention_bias`                         | soft forgetting scalar                                |            broadcast |                  in envelope |                                   ✅ yes |
| `denoising_step_list`                             | actual timestep schedule                              | broadcast or derived |      in envelope (preferred) |                                   ✅ yes |
| `global_end_index`, `local_end_index` (per layer) | ring-buffer indices                                   |              derived |                      derived |                         ✅ yes (derived) |
| `first_context_frame`, `context_frame_buffer`     | latent context used by recompute                      |       pipeline state |              mesh-side state |                       ✅ mesh ranks only |
| `decoded_frame_buffer`                            | decoded pixels used for steady-state recompute anchor |       pipeline state |                   rank0-only |    ❌ not shared, but must feed override |

Two notes:

* `global_end_index/local_end_index` are not broadcast, but should be deterministically derived from `current_start_frame` and token counts. (`05-kv-cache-head-sharding.md`.)
* In PP, the decoded buffer is intentionally rank0-only, so the only safe way to use it is to produce an explicit override tensor for the mesh (R0a).

---

### Per-chunk phases

This phase split is the cleanest way to reason about lifecycle.

```
Phase A  (rank0 control plane)
  - decide lifecycle actions: init/reset/recompute/steps/bias
  - materialize generator inputs: latents_in, conditioning_embeds, step_list
  - if recompute scheduled and using R0a: materialize context_frames_override
  - preflight: validate contract + dtype/specs BEFORE sending header

Phase B  (generator lockstep)
  - apply reset decisions
  - optional recompute call (uses context frames)
  - denoise loop calls (N steps)
  - update KV indices and context buffers
  - advance current_start_frame deterministically

Phase C  (rank0 post)
  - decode latents_out
  - update decoded_frame_buffer
  - (R0a) VAE re-encode anchor for next chunk’s override
```

TP v0 runs A/B/C on every rank (wasteful but simple).
TP v1.1 moves A and C to rank0 but keeps B lockstep across all ranks.
PP moves A and C to rank0, and B to mesh only.

---

### State transitions

Treat these as the only lifecycle transitions that exist. Everything else is a bug.

| Event           | Trigger                                       | Transition                                                          | Required invariant                                                |
| --------------- | --------------------------------------------- | ------------------------------------------------------------------- | ----------------------------------------------------------------- |
| Hard cut        | `init_cache=True` (or explicit reset request) | `cache_epoch += 1`, KV and cross-attn caches cleared, indices reset | All participants observe same `init_cache` at same boundary       |
| Soft transition | `kv_cache_attention_bias` changes             | attention weighting changes, cache still advances                   | bias must be identical across ranks                               |
| Normal advance  | `do_kv_recompute=False`                       | KV append for new tokens, indices advance                           | `current_start_frame` and token counts identical                  |
| Recompute       | `do_kv_recompute=True`                        | extra generator call that rewrites cache slice, then denoise        | all ranks must execute the recompute call, with identical context |
| Evict/roll      | ring would overflow                           | internal roll/evict adjusts `local_end_index`                       | eviction decision must be identical (derived)                     |

---

## Divergence hazards

Top hazards, classified by likely failure. “Tripwire” means: how to make it crash loudly instead of rotting.

| Hazard                                         | Failure class          | Typical cause                                                                | Tripwire                                                                       | Where to assert                           |
| ---------------------------------------------- | ---------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ----------------------------------------- |
| Plan mismatch                                  | NCCL hang              | one rank runs v0, other runs v1 generator-only                               | env parity on `SCOPE_TP_PLAN` / `SCOPE_PP_ENABLED`                             | init (`runtime.py`)                       |
| Generator-call count mismatch                  | NCCL hang              | recompute scheduled on one side only, step count differs                     | broadcast `expected_generator_calls`; count observed calls and assert          | rank0 preflight + worker/mesh runner      |
| Post-header exception strands peer             | NCCL hang              | sender broadcasts header then fails building specs/tensors                   | “preflight before header”: pickle/meta/spec/dtype checks                       | sender control plane                      |
| KV reset decision drift                        | Franken-model          | one side resets KV (hard cut) but other doesn’t                              | explicit `reset_kv_cache` + `cache_epoch` rule; forbid out-of-band resets      | rank0 decision + worker/mesh validation   |
| Cross-attn cache reset drift                   | Franken-model          | `conditioning_embeds_updated` inferred differently when workers skip blender | broadcast explicit reset bit or epoch                                          | rank0 preflight + setup-caches block      |
| Transition state drift                         | Franken-model          | worker `_transition_active` differs if it didn’t run blending                | do not infer; broadcast final reset decisions                                  | rank0 preflight                           |
| `current_start_frame` drift                    | Franken-model          | mesh advances start frame differently, or rank0 sends wrong scalar           | broadcast scalar every chunk; optional all_gather check every N                | worker/mesh runner                        |
| Missing recompute context override in v1.1c/PP | Franken-model or crash | worker falls back to VAE re-encode (if available) or uses wrong context      | if `do_kv_recompute`, require `context_frames_override` and disallow fallback  | rank0 preflight, mesh validate-before-run |
| Backend mismatch (fa4 vs flash)                | Franken-model          | auto backend differs per rank                                                | pin backend, add to env parity, log effective backend                          | init + startup report                     |
| Recompute frequency tweak                      | quality drift          | `EVERY=2` etc                                                                | treat as an experiment with explicit quality gates; do not ship as “perf flag” | operator policy + regression test         |
| Out-of-band cache mutation                     | Franken-model          | rank0 endpoint clears cache without broadcast                                | audit and delete such endpoints, or make them broadcast actions                | code audit + tests                        |
| Non-deterministic meta encoding breaks digest  | false positive trips   | JSON key order etc                                                           | canonical serialization for digests                                            | digest implementation                     |

The ones that cause hangs are the only ones that matter at 2am. The rest will silently ruin output if you do not have digests/fingerprints.

---

## Minimal deterministic contract

This is the smallest set of fields that make KV lifecycle deterministic in TP v1.1 and PP. Everything else is optional optimization.

### Canonical field mapping

Naming drift exists. Fix it by committing to a mapping:

* TP v1.1 plan fields: `tp_plan`, `tp_envelope_version`, `tp_do_kv_recompute`, `tp_num_denoise_steps`, `tp_expected_generator_calls`, `tp_reset_kv_cache`, `tp_reset_crossattn_cache`.
* PP fields: `pp_envelope_version`, `do_kv_recompute`, `num_denoise_steps`, `expected_generator_calls`, `reset_kv_cache`, `reset_crossattn_cache`.
* Override tensor:

  * TP: `context_frames_override`
  * PP plan sometimes calls it `context_frames`
  * Canonical: treat them as the same semantic thing, name mismatch only.

### P0 fields table

| Field                               |         TP v1.1 |            PP | Why it exists                         | Fail-fast check                                                   |
| ----------------------------------- | --------------: | ------------: | ------------------------------------- | ----------------------------------------------------------------- |
| `plan_id` (`tp_plan` / PP enabled)  |               ✅ |             ✅ | prevents divergent codepaths          | env parity + per-message assert                                   |
| `envelope_version`                  |               ✅ |             ✅ | schema mismatch must crash            | reject unknown version before any collective                      |
| `call_id`                           |               ✅ |             ✅ | ordering and replay boundary          | monotonic, crash if decreases                                     |
| `chunk_index`                       |               ✅ |             ✅ | output ordering                       | monotonic for INFER                                               |
| `cache_epoch`                       |               ✅ |             ✅ | hard-cut invalidation                 | increments iff KV reset boundary occurs; drop stale results in PP |
| `init_cache`                        |               ✅ |             ✅ | explicit hard cut                     | required on first chunk of epoch                                  |
| `reset_kv_cache`                    |               ✅ |             ✅ | stop relying on per-rank inference    | must be identical across participants                             |
| `reset_crossattn_cache`             |               ✅ |             ✅ | same as above                         | must be identical                                                 |
| `current_start_frame`               |               ✅ |             ✅ | cache indexing depends on it          | broadcast scalar; worker uses it as source of truth               |
| `do_kv_recompute`                   |               ✅ |             ✅ | recompute adds a generator call       | must be explicit                                                  |
| `num_denoise_steps`                 |               ✅ |             ✅ | controls call count                   | must be explicit                                                  |
| `expected_generator_calls`          |               ✅ |             ✅ | prevents hangs                        | worker counts calls and asserts                                   |
| `kv_cache_attention_bias`           |               ✅ |             ✅ | soft forgetting must match            | broadcast scalar every chunk                                      |
| `denoising_step_list` or equivalent | ✅ (recommended) | ✅ (preferred) | scheduler drift can change call graph | treat as required input, not inferred                             |
| `height`, `width`                   | ✅ (recommended) |             ✅ | shape sanity for context/latents      | validate shapes match geometry                                    |

### Required tensors

Bringup-safe default is “broadcast them every chunk” until stable.

| Tensor                           | Required when                               | Why                                         | Validation                                      |
| -------------------------------- | ------------------------------------------- | ------------------------------------------- | ----------------------------------------------- |
| `latents_in`                     | always                                      | generator inputs must be identical          | dtype/shape exact match                         |
| `conditioning_embeds` / override | always in bringup                           | avoid worker text encoder divergence        | dtype/shape; optionally epoch-based later       |
| `denoising_step_list`            | always (if used)                            | prevents scheduler divergence               | dtype/shape; length matches `num_denoise_steps` |
| `context_frames_override`        | whenever `do_kv_recompute=True` in v1.1c/PP | avoids decoded-anchor coupling and fallback | must be present, and mesh must not fall back    |

### Env parity keys that must include cache semantics

If a knob can change control flow or call count, it must be parity-checked.

Minimum set:

* `SCOPE_KV_CACHE_RECOMPUTE_EVERY` (call-count gating)
* `SCOPE_KV_BIAS_BACKEND` (backend parity)
* `SCOPE_TP_PLAN` (role/plan parity)
* `SCOPE_PP_ENABLED` (topology parity)
* compile gating flags that change collective behavior (`SCOPE_TP_ALLOW_COMPILE`, etc.)

This is straight from the deadlock audit logic: any per-rank mismatch that changes “how many generator calls happen” is a hang trap.

---

## Recompute decoupling options

This is the decoded-anchor dependency, stated plainly:

* In steady state, recompute wants an anchor derived from `decoded_frame_buffer[:, :1]` and VAE re-encode. (`pp-topology-pilot-plan.md`, `05-kv-cache-head-sharding.md`.)
* If the worker or mesh does not decode, you must either:

  * supply an override tensor (R0a), or
  * change semantics (R0), or
  * disable recompute (R1) temporarily.

### Option table

| Option | What you do                              |                  Semantic risk |                    Overlap impact | When to use                         |
| ------ | ---------------------------------------- | -----------------------------: | --------------------------------: | ----------------------------------- |
| R1     | disable recompute (`EVERY=999999`)       |         high (quality changes) |                      best overlap | PP0 bringup, contract testing       |
| R0a    | rank0 supplies `context_frames_override` | low (matches current behavior) | medium (adds boundary dependency) | “real” PP runs, v1.1c correctness   |
| R0     | latent-only anchor on mesh               |        unknown (must validate) |                      best overlap | later experiment if R0a bottlenecks |

### Recommendation

1. **PP0 bringup**: start with **R1**. You are proving control-plane and queueing, not video quality. (`pp0-bringup-runbook.md`.)
2. Restore correctness: move to **R0a**. Make `context_frames_override` required whenever `do_kv_recompute=True`. Mesh must never take the VAE fallback. This is the “semantic match” path.
3. Only if overlap is measurably hurt by the R0a dependency, consider **R0** as an experiment:

   * define a quality gate before writing any more code
   * compare against baseline outputs (or at least subjective regression criteria)
   * assume it will drift unless proven otherwise

Subjective expectation: R0a is the right default$_{80%}$; R0 might be viable but only after you have hard evidence that the R0a dependency is a real bottleneck.

### Why you should not “just lower recompute frequency”

Run-log evidence says it is not a win: in Run 10b, decode+recompute was ~33% of measured GPU time per chunk (decode ~107 ms, recompute ~104 ms). In Run 11, skipping recompute every other chunk made output glitchy and did not improve net chunk time because denoise got slower. So: treat recompute scheduling as correctness, not a perf dial. (`bringup-run-log.md`.)

---

## Tripwire checklist

This is the “where to assert” list that keeps you out of hung NCCL calls.

### Rank0 preflight checks before sending any header

Do these before any broadcast/send that commits the receiver.

* Validate schema: version, plan, required fields present
* Validate phase plan: `expected_generator_calls` computed and consistent
* Validate tensors: dtype supported, shapes match geometry, contiguous if required
* Validate meta serialization: picklable or stable JSON, no nested tensors in meta
* Validate env parity already passed (plan, backend, recompute knobs)
* Only after all of that: send header → meta/specs → tensors

### Mesh leader preflight before any mesh collective

In PP, the leader must not strand the mesh.

* Receive full envelope p2p
* Validate envelope fully
* Only then broadcast within mesh_pg

### Worker or mesh runner checks at Phase B entry

* Assert `plan_id` matches role
* Assert `call_id`, `chunk_index`, `cache_epoch` monotonicity rules
* Assert `current_start_frame` matches expected stream progression
* Assert `expected_generator_calls` equals observed calls before returning

### Periodic lightweight parity checks

Cheap and effective:

* Scalar all_gather every N chunks (N=64 or 128):

  * `cache_epoch`, `current_start_frame`, `do_kv_recompute`, `num_denoise_steps`, `kv_cache_attention_bias`
* Optional input digest on envelope+tensors during bringup (`SCOPE_TP_INPUT_DIGEST=1`)
* Weight shard fingerprint baseline and periodic recheck to detect Franken-model

---

## Break-it tests

These are intentionally adversarial. Each one should have a crisp failure signature and should not require waiting 300 seconds.

### Test 1: Plan mismatch

**Break it**: set `SCOPE_TP_PLAN` (or PP enable) differently across ranks.
**Expected**: fail at init via env parity. No collective hang.

### Test 2: Generator-call count mismatch

**Break it**: force `do_kv_recompute=True` on rank0 but false on worker/mesh, or set `SCOPE_KV_CACHE_RECOMPUTE_EVERY` differently across ranks (in a sandbox).
**Expected**: fail-fast before Phase B with “expected_generator_calls mismatch” (not a hang).
**Required wiring**: broadcast `expected_generator_calls` and assert observed call count.

### Test 3: Missing recompute override

**Break it**: schedule recompute but omit `context_frames_override`.
**Expected**:

* rank0: `validate_before_send()` fails pre-header, or
* mesh: leader rejects envelope before broadcast/collectives.
  No fallback to VAE path on mesh.

### Test 4: Cache reset drift via out-of-band mutation

**Break it**: trigger a cache reset on rank0 only (simulate a bad REST endpoint).
**Expected**: fail by a state digest mismatch (preferred) or input digest mismatch soon after. If neither triggers, that is a bug in your tripwires.

### Test 5: Current-start-frame drift

**Break it**: perturb `current_start_frame` on one side for a single chunk.
**Expected**: scalar all_gather check catches it (or worker asserts local progression mismatches envelope).

### Test 6: PP stale result after hard cut

**Break it**: delay a mesh result, then issue a hard cut so `cache_epoch` increments, then deliver the delayed result.
**Expected**: rank0 drops it by epoch mismatch and does not decode or emit it. (`pp0-bringup-runbook.md` acceptance rules.)

Optional extra (worth having):

* **Backend mismatch**: force `flash` vs `fa4`. Expected: env parity failure at init. If it runs, you are one step away from silent drift.

---

## Deltas to refs topic docs

### Patch for `refs/topics/21-idempotency-and-replay.md`

Add one explicit rule: **PP stage boundary is a replay boundary**.

* Accept results only if `(cache_epoch, call_id, chunk_index)` match expectations.
* On hard cut: increment `cache_epoch`, flush queues, drop stale results.
* Do not attempt “retry” semantics until you have side-effect idempotency for cache updates.

### Patch for `refs/topics/04-determinism-across-ranks.md`

Add a short subsection: **cache lifecycle determinism is not bitwise determinism**.

* Primary goal: prevent hang and Franken-model via explicit contracts and tripwires.
* Recommended bringup set:

  * input digest on envelope+tensors
  * scalar state digest every N chunks
  * shard fingerprint baseline and periodic recheck
* Backend pinning (`SCOPE_KV_BIAS_BACKEND`) is a correctness contract, not a perf setting.

If you want one more “belt + suspenders” improvement: implement a tiny “KV lifecycle digest” that hashes only the scalars and a few cache index values (`global_end_index/local_end_index` from one representative layer) and all_gathers it every 128 chunks. Cheap, and it catches the class of bugs where execution stays lockstep but the cache state silently diverges.
