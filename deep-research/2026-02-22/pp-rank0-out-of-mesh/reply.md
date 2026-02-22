## PP Rank0-out-of-mesh bringup audit (P0 correctness first)

### 0) Delta vs prior 5Pro PP readiness review (#13)

**Unchanged (still true):**

* Topology goal stays the same: Stage 0 on rank0 does A (materialize inputs) + C (decode/output), Stage 1 on ranks 1..N does B (generator-only), with TP collectives only inside `mesh_pg`.
* “Crash > hang” is still the posture: validate before any collective; short timeouts + watchdogs; restart is the recovery during bringup.
* The stage boundary still needs a boring, versioned contract and a deterministic per-chunk phase plan including `expected_generator_calls`.
* Overlap still comes from double buffering with bounded queues, with `D_in=D_out=2` as the first setting that can sustain overlap.
* Recompute sequencing still should be R1 → R0a → (optional) R0.

**Refined (same idea, sharper / now implemented):**

* Contract naming and monotonicity are now concrete:

  * `PPAction = {NOOP, INFER, SHUTDOWN}`
  * `call_id` is globally monotonic across *all* actions
  * `chunk_index` is monotonic for `INFER` actions
  * `pp_disable_vace` exists as a bringup gate
* `PPEnvelopeV1.validate_before_send()` is action-aware (NOOP/SHUTDOWN skip tensor checks).
* `PPResultV1` includes bringup debug fields (`observed_generator_calls`, `mesh_current_start_frame`, `output_digest_i64`).
* `PrepareContextFramesBlock` split is done (decoded buffer update can be optional), which is necessary for “mesh updates latent buffers without decode”.

**Overridden (what changed and why):**

* Older docs/pseudocode that mention `chunk_id` are outdated: the contract now uses `(call_id, chunk_index)` and that needs to propagate into tags, logs, and acceptance rules.
* Naming drift is real: planning docs talk about `context_frames_override`, but the live PP envelope tensor is named `context_frames`. Treat this as a pure naming mismatch: envelope `context_frames` should feed pipeline `context_frames_override`. Pick one canonical mapping and enforce it.
* “Header-first send” is now explicitly flagged as dangerous: anti-stranding requires preflight (validate + serialize + spec + dtype checks) before sending any header that commits the peer to blocking.

---

## P0 blockers (ship-stoppers for PP0/PP1 bringup)

1. **Anti-stranding must be enforced in the PP transport layer (rank0 and leader).**
   No header (or any “payload follows” commitment) until:

   * `validate_before_send()` passes,
   * metadata is serialized (pickle or stable JSON),
   * `tensor_specs` are materialized,
   * every dtype in specs is supported,
   * tensors are materialized/cast/contiguous.

2. **Leader must not strand mesh ranks in `mesh_pg` collectives.**
   Leader must fully receive + validate the envelope before initiating any `mesh_pg` broadcast. If the envelope is invalid, leader must broadcast a terminal action (SHUTDOWN or ERROR) rather than throwing and disappearing.

3. **PP1+ group correctness must be enforced by code, not hope.**
   Any TP collective on the wrong group is an instant deadlock. You need a “group usage rulebook” plus hard asserts/wrappers so “wrong group” becomes a fast crash, not a 300s NCCL timeout.

4. **Per-chunk call-count determinism is mandatory.**
   `expected_generator_calls` must be:

   * computed on rank0 and transmitted,
   * enforced on the mesh runner (assert while executing),
   * reported back (`observed_generator_calls`) and checked on rank0.
     Anything else is how you get a hang (PP1) or silent corruption (PP0).

5. **Hard cut semantics must flush queues and filter by `cache_epoch`.**
   Rank0 must drop stale results with mismatched `cache_epoch` and must not decode them. Flush `D_in` and `D_out` on hard cuts. This is your minimum replay safety.

6. **`pp_disable_vace` must be enforced as “True” until VACE has a stage contract.**
   Otherwise you will “work” while being wrong (silent drift) or wedge Stage 1 due to missing tensors.

---

## P1 recommendations (high leverage, but you can stage them)

* Add explicit dtype/shape checks to contract validation (or transport preflight) for:

  * `conditioning_embeds` dtype BF16, expected rank/shape class
  * `latents_in/out` dtype BF16, shape matches decode expectations
  * `current_denoising_step_list` dtype int64, length matches `num_denoise_steps`
  * `context_frames` dtype BF16, expected kv-cache frame count
* Add “group asserts” wrappers so collectives fail fast with logs that include:
  `call_id`, `chunk_index`, `cache_epoch`, `group=world_pg|mesh_pg`, `rank`, `mesh_rank`.
* Always return Stage 1 timings in `PPResultV1` (durations, not absolute timestamps) so overlap proof does not need clock sync.
* Watchdogs:

  * mesh non-leader watchdog: “time since last broadcast header”
  * rank0 watchdog: “time since last result received”
* Add a startup handshake that all-gathers:

  * PP enabled bit, mesh TP degree, compile enabled bit, funcol availability, any collective-gated debug flags

---

# 1) Minimal PP contract schema (PP0/PP1 bringup)

Below is the minimal schema you should treat as required for “P0 correctness” bringup. It matches the live `pp_contract.py` fields, plus a few “where validated” requirements that you should enforce in the transport/runner.

## PPEnvelopeV1 (Stage 0 → mesh leader → mesh broadcast)

### Metadata fields

| field                      | type     |    required when | why                                              | where validated                                                            |
| -------------------------- | -------- | ---------------: | ------------------------------------------------ | -------------------------------------------------------------------------- |
| `pp_envelope_version`      | int      |           always | schema dispatch, fast reject                     | rank0 `validate_before_send`; leader pre-bcast                             |
| `action`                   | int enum |           always | framing: does payload follow?                    | rank0 `validate_before_send`; leader loop                                  |
| `call_id`                  | int      |           always | global ordering, dedupe, replay boundary         | rank0 monotonic; leader monotonic; rank0 on result                         |
| `chunk_index`              | int      |   `action=INFER` | output ordering and assertions                   | rank0 monotonic; leader monotonic; rank0 on result                         |
| `cache_epoch`              | int      |           always | hard cut invalidation and stale drop             | rank0 increments; rank0 drops mismatched results                           |
| `height`, `width`          | int      |          `INFER` | prevents stale geometry/mask state on mesh       | rank0 validate; leader validate; mesh asserts                              |
| `current_start_frame`      | int      |          `INFER` | cache indexing consistency                       | rank0 sets; mesh uses; leader reports `mesh_current_start_frame`           |
| `do_kv_recompute`          | bool     |          `INFER` | call-count determinism                           | rank0 sets; mesh follows (must not infer)                                  |
| `num_denoise_steps`        | int      |          `INFER` | call-count determinism and step list length      | rank0 validate; mesh asserts                                               |
| `expected_generator_calls` | int      |          `INFER` | deadlock tripwire                                | rank0 validate; mesh asserts; rank0 compares to `observed_generator_calls` |
| `init_cache`               | bool     |          `INFER` | explicit reset semantics (no inference on mesh)  | rank0 sets; mesh follows                                                   |
| `reset_kv_cache`           | bool     |          `INFER` | prevents cache divergence                        | rank0 sets; mesh follows; debug asserts                                    |
| `reset_crossattn_cache`    | bool     |          `INFER` | prevents cross-attn drift                        | rank0 sets; mesh follows; debug asserts                                    |
| `base_seed`                | int      |          `INFER` | determinism across retries / replay              | rank0 sets; mesh uses                                                      |
| `kv_cache_attention_bias`  | float    |          `INFER` | generator config parity                          | rank0 sets; mesh uses                                                      |
| `noise_scale`              | float    |          `INFER` | informational unless latents already noise-mixed | rank0 sets; mesh uses if applicable                                        |
| `pp_disable_vace`          | bool     | always (bringup) | forbid unsupported stage mode                    | rank0 validate hard True; leader validate hard True                        |

### Tensor fields

| field                         | dtype |          required when | why                                                                                 | where validated                                           |
| ----------------------------- | ----- | ---------------------: | ----------------------------------------------------------------------------------- | --------------------------------------------------------- |
| `conditioning_embeds`         | BF16  |                `INFER` | mesh cannot text-encode                                                             | rank0 validate; leader validate; mesh asserts dtype/shape |
| `latents_in`                  | BF16  |                `INFER` | mesh cannot VAE encode                                                              | rank0 validate; leader validate; mesh asserts dtype/shape |
| `current_denoising_step_list` | int64 |                `INFER` | mesh must not infer scheduler decisions                                             | rank0 validate; mesh asserts len==`num_denoise_steps`     |
| `context_frames`              | BF16  | `do_kv_recompute=True` | semantic-preserving recompute override (maps to pipeline `context_frames_override`) | rank0 validate; mesh requires override path only          |

**Naming note (resolve early):**
Envelope uses `context_frames`. Pipeline block expects `context_frames_override`. Decide on an unambiguous mapping (example: `env.context_frames → call_params["context_frames_override"]`) and assert that the mesh never takes the decoded-pixels fallback path when recompute is scheduled.

## PPResultV1 (mesh leader → Stage 0)

### Metadata fields

| field                      | type | required when | why                          | where validated                        |
| -------------------------- | ---- | ------------: | ---------------------------- | -------------------------------------- |
| `pp_result_version`        | int  |        always | schema dispatch              | leader validate; rank0 validate        |
| `call_id`                  | int  |        always | ordering, dedupe             | rank0 acceptance rules                 |
| `chunk_index`              | int  |        always | output ordering              | rank0 acceptance rules                 |
| `cache_epoch`              | int  |        always | stale drop after hard cut    | rank0 drop mismatched                  |
| `observed_generator_calls` | int  |       bringup | deadlock/call-count tripwire | rank0 compares with envelope plan      |
| `mesh_current_start_frame` | int  |       bringup | cache progress sanity        | rank0 logs/compares                    |
| `output_digest_i64`        | int  |      optional | corruption tripwire          | rank0 logs; PP1 can all-gather on mesh |

### Tensor fields

| field         | dtype |     required when | why          | where validated                               |
| ------------- | ----- | ----------------: | ------------ | --------------------------------------------- |
| `latents_out` | BF16  | always for result | decode input | leader validate; rank0 validates shape+finite |

Rank0 acceptance rule (P0):

* Drop any result where `cache_epoch != current_cache_epoch`.
* Enforce `call_id` monotonic; enforce `chunk_index` monotonic.
* Reject or crash if `observed_generator_calls != expected_generator_calls` for that chunk.

---

# 2) Multi-group deadlock rulebook (world_pg vs mesh_pg)

This is the PP1+ “how not to wedge the cluster” rulebook.

## Process groups

* `world_pg`: all ranks. Used for:

  * rank0 ↔ leader p2p envelope/result transport
  * startup barriers (optional)
  * shutdown coordination (optional)

* `mesh_pg`: ranks `{1..mesh_tp}` only. Used for:

  * leader → mesh broadcasts (envelope distribution)
  * all TP collectives inside Phase B (all_reduce, all_gather, funcol, etc)
  * any mesh-wide digests/fingerprints

**Convention:** `mesh_pg = dist.new_group(ranks=[1,2,...,mesh_tp])` so the mesh leader (global rank 1) is mesh group rank 0. All mesh broadcasts use `src=0, group=mesh_pg`.

## Rules (copy-paste strict)

1. **Rank0 must never call a collective on `mesh_pg`.**
   Make this a startup assert. Easiest check: on rank0, any attempt to use `dist.get_rank(mesh_pg)` should throw. If it does not, you misbuilt groups.

2. **Mesh Phase B must never call a collective on `world_pg` (or default group).**
   The only allowed comm in Phase B is `mesh_pg`. If you need logging, do it after Phase B.

3. **Every TP collective wrapper must take an explicit `group=`.**
   Default-group collectives are forbidden in PP mode. In PP1, using the default group is the most likely “instant deadlock” bug.

4. **No conditional collectives behind per-rank flags.**
   If a flag gates an all_gather/fingerprint/digest, it must be:

   * env-parity checked at init, or
   * transmitted in the envelope plan and identical across mesh ranks.
     Otherwise: FM-03 hang.

5. **Leader validation before mesh broadcast is mandatory.**
   Leader must not start broadcasting meta/specs/tensors until the entire envelope is received and validated. If invalid, broadcast SHUTDOWN/ERROR.

## Minimum env parity keys (PP)

At minimum, parity-check across all ranks:

* `SCOPE_PP_ENABLED`
* “mesh TP degree” (whatever env var defines it, currently `SCOPE_TENSOR_PARALLEL` is used as mesh size when PP enabled)
* `SCOPE_DIST_BACKEND`, `SCOPE_DIST_TIMEOUT_S`
* any flags that cause collectives:

  * input/output digest flags
  * fingerprint flags
  * compile enable flags and any compile mode knobs (compile mismatches can shift collective behavior)
* recompute schedule knobs, if any remain outside the envelope (ideally none)

Startup handshake asserts (P0):

* all ranks agree PP is enabled
* all ranks agree mesh size
* mesh ranks agree they are in mesh and their mesh ranks are unique and contiguous
* rank0 is not in mesh
* compile parity bit (all_gather) is identical across mesh ranks

---

# 3) Anti-stranding protocol ordering (sender and leader)

Goal: no one blocks forever waiting for “the rest of a message”.

## Rank0 → leader p2p (envelope send)

**Sender preflight (must complete before header):**

1. `env.validate_before_send()`
2. `meta = env.metadata_dict()`
3. `meta_bytes = pickle.dumps(meta)` (or stable JSON bytes)
4. `tensor_specs = env.tensor_specs()`
5. For each spec:

   * validate dtype name is supported by your dtype map
   * validate shape is sane (non-negative, expected rank)
6. `tensors = [env.tensor_dict()[spec["key"]] for spec in tensor_specs]`
7. Normalize tensors:

   * correct device
   * correct dtype (cast if needed)
   * contiguous (or assert)

**Only after 1–7 succeed:**
8. Send header (action, call_id, chunk_index, cache_epoch, n_tensors, version)
9. Send `(meta_bytes, tensor_specs)`
10. Send tensors in spec order, with tags derived from `(cache_epoch, call_id, field_id)` if you use tags.

**If any step 1–7 fails:** nothing was sent. Peer cannot be stranded. That is the whole point.

## Leader receive (envelope)

1. Recv header. If `action != INFER`, handle NOOP/SHUTDOWN and do not expect payload.
2. Recv `(meta_bytes, tensor_specs)` and deserialize.
3. Allocate tensors from specs.
4. Recv tensors in spec order.
5. Reconstruct `env = PPEnvelopeV1.from_metadata_and_tensors(meta, tensors)`.
6. Validate again (leader-side):

   * schema version
   * required tensors present
   * `pp_disable_vace=True`
   * monotonic `call_id` and `chunk_index` (for INFER)
   * `expected_generator_calls` matches computed
7. Only now is it legal to enter any `mesh_pg` collective.

## Leader → mesh broadcast (envelope distribution)

The broadcast is also a “commitment”. Apply the same rule: validate before entering.

Leader:

1. Preflight already done (above).
2. Broadcast a small broadcast-header first (action, call_id, chunk_index, cache_epoch, n_specs).
3. Broadcast meta/specs (bytes or object list).
4. Broadcast tensors in spec order.

Non-leaders:

1. Enter broadcast to receive the broadcast-header.
2. Receive meta/specs.
3. Allocate tensors.
4. Receive tensors.

**Invalid envelope policy:**
If leader decides the envelope is invalid after recv/preflight, leader must still broadcast a terminal action (SHUTDOWN or ERROR) so non-leaders are not stranded in broadcast waiting for a leader that never arrives.

## Leader → rank0 p2p (result send)

Mirror the envelope rules:

Preflight (before header):

1. `result.validate_before_send()`
2. `meta_bytes = serialize(result.metadata_dict())`
3. `tensor_specs = result.tensor_specs()`
4. `tensors = [result.latents_out]` (plus any others later)

Then send:
5. header
6. meta/specs
7. tensors

Rank0 receive:

* drop if `cache_epoch` mismatch
* enforce monotonic IDs
* validate tensor shape and finite
* compare `observed_generator_calls` vs `expected_generator_calls`

---

# 4) Backpressure + overlap proof recipe (no clock sync required)

## Required instrumentation (minimal)

Rank0 per chunk:

* IDs: `call_id`, `chunk_index`, `cache_epoch`
* Timestamps (monotonic clock):

  * `tA0`: start build envelope
  * `tA1`: envelope ready (after tensors materialized)
  * `tSend`: envelope sent (optional)
  * `tRecv`: result received
  * `tEmit`: decoded output enqueued
* Queue depths:

  * `len(inflight_to_mesh)` (≤ `D_in`)
  * `len(ready_for_decode)` (≤ `D_out`)

Mesh leader per chunk (reported back in `PPResultV1` as durations):

* `tB_ms = tB1 - tB0` where:

  * `tB0`: Phase B start
  * `tB1`: Phase B end
* `t_mesh_idle_ms = max(0, tB0[k] - tB1[k-1])` (leader-local)

Because these are durations computed on the same host, rank0 can use them without clock sync.

## OverlapScore (bringup gate)

After warmup (skip first 10 chunks):

* `period_k = tEmit[k] - tEmit[k-1]`
* `stage0_k = (tA1[k] - tA0[k]) + (tEmit[k] - tRecv[k])`  (A + C wall time)
* `stage1_k = tB_ms[k] / 1000`

Then:

* `hidden_k = max(0, stage0_k + stage1_k - period_k)`
* `overlap_ratio_k = hidden_k / max(1e-6, min(stage0_k, stage1_k))`
* `OverlapScore = median(overlap_ratio_k)`

**Pass gate (PP0 overlap claim):**

* `OverlapScore ≥ 0.30`
* and queues remain bounded: `max(inflight_to_mesh) ≤ D_in`, `max(ready_for_decode) ≤ D_out`
* and median period moves toward `max(median(stage0), median(stage1))` rather than their sum

## Interpreting `t_mesh_idle_ms` and queue signatures

If overlap is working, the mesh should rarely idle waiting for rank0.

* Healthy overlap when Stage 1 is bottleneck:

  * `t_mesh_idle_ms` small
  * `inflight_to_mesh` often near `D_in`
  * `ready_for_decode` stays low
  * `period ≈ stage1`

* Healthy overlap when Stage 0 is bottleneck:

  * `ready_for_decode` hits `D_out` often
  * mesh begins to idle (`t_mesh_idle_ms` increases)
  * `period ≈ stage0`

Scheduling bug signature (common):

* `t_mesh_idle_ms` tracks rank0 decode time and both queues stay near 0.
  This usually means rank0 is sending envelope `k+1` after decode of `k`, killing overlap. Fix: send `env[k+1]` before decode of `res[k]`.

---

# 5) Recompute coupling decision table (R1 / R0a / R0)

| regime                        | recompute behavior                                            | contract requirements                                                                                                        | call-count rule                                                             | overlap impact                                                                                                                                 |
| ----------------------------- | ------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| **R1** (bringup)              | recompute disabled                                            | `do_kv_recompute=False`; omit `context_frames`; enforce mesh does not infer recompute from local env                         | `expected_generator_calls = num_denoise_steps`                              | best overlap (no Stage 0 → Stage 1 dependency)                                                                                                 |
| **R0a** (semantic-preserving) | recompute enabled; rank0 provides anchor via VAE re-encode    | `do_kv_recompute=True`; require `context_frames` tensor for every chunk that recomputes; mesh must not take decoded fallback | `expected_generator_calls = num_denoise_steps + 1`                          | overlap constrained at chunk boundary (rank0 must produce `context_frames` before mesh starts recompute), but denoise can still overlap decode |
| **R0** (semantic change)      | recompute anchor derived from mesh latents (no VAE roundtrip) | drop `context_frames` requirement; add an explicit “latent-anchor mode” bit so semantics are not accidental                  | still `+1` calls if recompute remains a generator pass, or change algorithm | best overlap, but requires quality validation and drift tripwires                                                                              |

Bringup sequencing recommendation (still correct):

1. PP0/R1: prove contract, queues, overlap
2. PP0/R0a: restore semantics, re-measure OverlapScore delta
3. Optional R0 experiment only if R0a overlap hit is too large and quality is acceptable

---

# Required break-it tests (4–6), mapped to FM taxonomy

Each test should (a) fail fast, (b) log call_id/chunk_index/cache_epoch/group, and (c) not leave ranks pinned for minutes.

### Test 1: Wrong process group used for one collective (PP1+)

* **FM mapping:** FM-22 (new, PP-specific): wrong group collective
* **Setup:** PP1 (3 ranks), `mesh_pg={1,2}`, TP degree 2
* **Inject:** in one TP collective wrapper, intentionally use `group=world_pg` when `SCOPE_PP_TEST_WRONG_GROUP=1`
* **Expected:** wrapper detects mismatch and raises *before calling NCCL*, logs:

  * `call_id`, `chunk_index`, `group_used`, `expected_group=mesh_pg`
* **No stranding requirement:** leader catches exception, broadcasts `PPAction.SHUTDOWN` to mesh, then exits; rank0 times out quickly and exits.

### Test 2: Leader receives envelope header/meta, then fails before mesh broadcast

* **FM mapping:** FM-23 (PP-specific): leader pre-bcast failure handling
* **Setup:** PP1 (or PP0 with a single mesh rank)
* **Inject:** corrupt envelope (bad version, missing required tensor) so leader fails validation after recv
* **Expected:** leader does **not** start broadcasting payload; instead it broadcasts a terminal action (SHUTDOWN/ERROR) so non-leaders exit cleanly. Logs include failure reason plus IDs.
* **No stranding requirement:** non-leaders see terminal action and exit loop without waiting for missing tensors.

### Test 3: Rank0 throws after “commitment point” (anti-stranding regression)

* **FM mapping:** FM-04 (throw-after-header strands peer) + FM-07 (unsupported dtype discovered too late)
* **Setup:** PP0 contract smoke
* **Inject:** add a tensor with unsupported dtype or inject an unserializable object into meta
* **Expected:** send path fails during preflight (pickle/spec/dtype checks) and **no header is sent**. Leader never logs “header received” for that `call_id`.
* **Pass condition:** peer does not block in recv; test ends immediately with a controlled exception on rank0.

### Test 4: Generator call-count mismatch

* **FM mapping:** FM-02
* **Setup:** PP0 or PP1
* **Inject:** on one mesh rank only, enable a local flag that would change the call count (skip recompute once, or change denoise loop length)
* **Required guard:** before Phase B, mesh ranks must all_gather a scalar “planned_call_count” and assert equality with `env.expected_generator_calls`
* **Expected:** mismatch triggers a fast crash before entering TP collectives (PP1) with log including:

  * `expected_generator_calls`, `planned_call_count`, `rank`
* **No stranding requirement:** failure happens before any generator collective.

### Test 5: Stale-epoch result after hard cut must be dropped

* **FM mapping:** FM-10 (cache reset semantics) + replay safety (`cache_epoch` filtering)
* **Setup:** PP0 overlap with `D_out=2`
* **Inject:** delay sending result for chunk k on mesh; trigger hard cut on rank0 (increment `cache_epoch`, flush queues) before delayed result arrives
* **Expected:** delayed result arrives with old `cache_epoch`, rank0 drops it and does not decode. Logs include:

  * `dropped_result_epoch`, `current_epoch`, `call_id`, `chunk_index`
* **Pass condition:** output stream continues in new epoch without decoding stale work.

### Test 6: Conditional collective behind per-rank flag mismatch

* **FM mapping:** FM-03
* **Setup:** PP1
* **Inject:** enable a debug collective (digest all_gather) on only one mesh rank via env var
* **Expected:** env parity handshake fails at startup (before any runtime loop), with explicit report of mismatched key.
* **Pass condition:** no runtime hang, immediate exit with a clear env mismatch message.

---

If you want one extra cheap guardrail that pays for itself: add a log line at every “commitment point” (header send, mesh broadcast start, first TP collective) that includes `call_id/chunk_index/cache_epoch`. When something wedges, that single line tells you which protocol boundary you stranded.

And yep, quick physical interrupt: look away from the screen, drop your shoulders, unclench your jaw.
