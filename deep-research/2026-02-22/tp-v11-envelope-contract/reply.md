## P0 blockers

### 1) Delta vs prior 5 Pro thread (08/10/11/12)

**Unchanged**

* **Plan must be explicit**: keep `TPAction.INFER` and select worker behavior via payload `tp_plan` (plus a version int). Do not infer from “which tensors are present.”
* **Worker must not call `__call__` in v1**: dispatch to `pipeline.tp_worker_infer(...)` for `tp_plan=v1_generator_only` (to avoid decode/text paths and mode-transition side effects).
* **Recompute coupling is real**: if recompute can happen, workers need a `context_frames_override`-style tensor. No VAE fallback in v1.
* **Crash > hang posture** stays the rule: validate before any collective that commits the peer to blocking, keep watchdogs/heartbeats available, prefer process exit over limping.
* **No nested tensors in meta** (top-level tensor extraction only) remains non-negotiable.

**Refined**

* **“Protocol first” got sharper**: preflight is now an exact ordered checklist (picklability, dtype mapping, deterministic spec ordering, tensor materialization) that must complete *before* the INFER header is broadcast.
* **Lockstep call-count is now a first-class field**: `tp_do_kv_recompute`, `tp_num_denoise_steps`, `tp_expected_generator_calls`, plus reset bits must be explicit and cross-checked on both sides.
* **Cache lifecycle is elevated**: v1 requires explicit `tp_reset_kv_cache` and `tp_reset_crossattn_cache` (do not rely on worker-local transition inference).
* **Conditioning transport becomes two-mode**: bringup-safe (broadcast each chunk) vs later optimize (epoch-based updates) is now the clean split.

**Overridden**

* **Conditioning “broadcast only on update chunks” is no longer acceptable for v1 bringup.** In v1, workers cannot encode. Update-only without an explicit epoch contract invites stale conditioning and silent corruption. Update-only becomes a later optimize mode only (with `conditioning_epoch` and reset semantics nailed down).
* **“This will improve single-stream FPS” expectation is dropped.** In current topology rank0 still gates wall time; v1.1c is justified by worker VRAM/startup/topology, not immediate FPS.

### P0 ship blockers for `tp_plan=v1_generator_only`

* **Versioned v1 envelope**: require `tp_plan` and `tp_envelope_version==1` in every INFER payload; reject unknown versions before any generator work.
* **Strict schema validation (rank0 + worker)**: missing any required tensor/flag is a fatal error *before* broadcast (rank0) or *before* generator entry (worker). No best-effort fallbacks in v1.
* **Preflight-before-header (anti-stranding)** in `TPControlPlane.broadcast_infer`: ensure nothing in meta/spec/dtype/materialization can throw after sending the INFER header.
* **Ban ad-hoc in-block broadcasts for v1**: all cross-rank tensors must go through control-plane `tensor_specs`. No worker-side shape inference broadcasts.
* **Explicit lockstep phase plan**: worker must execute exactly the plan, including recompute decision and denoise step count. Any mismatch must crash before collectives.
* **Explicit cache reset bits**: broadcast `tp_reset_kv_cache` and `tp_reset_crossattn_cache` and enforce cache epoch semantics against them.
* **Plan-aware warmup**: `_maybe_tp_lockstep_warmup()` must not call full pipeline on generator-only workers; either drive warmup through v1 entrypoint or disable warmup under v1 plan.

## P1 recommendations

* **Deterministic ordering for `tensor_specs`** (sort by `(key, index)`), so digests and diffs are readable and stable.
* **Expand dtype support or enforce casting**: preflight `_dtype_from_name` on every spec, and either add missing dtypes (at least `bool`, `int32` if they can appear) or cast to allowed dtypes at envelope build time.
* **Bringup-only generator output parity**: after each generator forward in v1 mode, compute a small output digest and `all_gather` it across ranks (gated behind an env var that is parity-checked).
* **Compile parity handshake**: after init, `all_gather` a “compile enabled” bit and fail if ranks differ.
* **Clear-after-read for override tensors** in consuming blocks and/or in the v1 runner, plus require explicit `None` fields so omission cannot reuse stale state.
* **Rank0 watchdog**: detect “broadcast started but end-to-end did not complete” quickly so rank0 exits instead of hanging on a dead worker.

## Minimal envelope schema v1 (`tp_envelope_version=1`)

> Table includes header fields (fixed) plus payload fields (in `call_params`).
> “Fail-fast check location” names the *first* place it should be asserted.

| field                         | type                                 | required_when                                          | why                                                  | fail-fast check location                                          |
| ----------------------------- | ------------------------------------ | ------------------------------------------------------ | ---------------------------------------------------- | ----------------------------------------------------------------- |
| `header.action`               | int64 enum `{NOOP, INFER, SHUTDOWN}` | always                                                 | receiver control flow                                | `control.py::recv_next`                                           |
| `header.call_id`              | int64                                | always                                                 | ordering + debug localization                        | `tp_worker.py` (monotonicity)                                     |
| `header.chunk_index`          | int64                                | `action==INFER`                                        | plan bookkeeping + logs                              | worker preflight (optional monotonic check)                       |
| `header.cache_epoch`          | int64                                | always                                                 | hard-cut / cache-world separation                    | `control.py::broadcast_infer` (bump rule) + worker asserts        |
| `header.control_epoch`        | int64                                | always                                                 | sequencing sanity                                    | `control.py` + optional worker asserts                            |
| `tp_plan`                     | str                                  | `action==INFER`                                        | dispatch (`v0_full_pipeline` vs `v1_generator_only`) | `tp_worker.py` (env vs payload mismatch) + worker schema validate |
| `tp_envelope_version`         | int                                  | `tp_plan=="v1_generator_only"`                         | schema dispatch + forward compat                     | rank0 preflight (`frame_processor.py`) + worker validate          |
| `height`                      | int                                  | `tp_plan=="v1_generator_only"`                         | geometry-dependent cache/mask setup                  | rank0 preflight + worker validate                                 |
| `width`                       | int                                  | `tp_plan=="v1_generator_only"`                         | geometry-dependent cache/mask setup                  | rank0 preflight + worker validate                                 |
| `current_start_frame`         | int                                  | `tp_plan=="v1_generator_only"`                         | KV cache indexing and recompute offsets              | rank0 preflight + worker validate                                 |
| `init_cache`                  | bool                                 | `tp_plan=="v1_generator_only"`                         | hard cut semantics                                   | rank0 preflight + worker validate                                 |
| `manage_cache`                | bool                                 | `tp_plan=="v1_generator_only"`                         | must not diverge across ranks                        | rank0 preflight + worker validate                                 |
| `tp_reset_kv_cache`           | bool                                 | `tp_plan=="v1_generator_only"`                         | make cache reset explicit (no inference)             | rank0 preflight + worker validate                                 |
| `tp_reset_crossattn_cache`    | bool                                 | `tp_plan=="v1_generator_only"`                         | avoid stale cross-attn cache on prompt updates       | rank0 preflight + worker validate                                 |
| `conditioning_embeds_updated` | bool                                 | `tp_plan=="v1_generator_only"`                         | cross-attn cache lifecycle control                   | rank0 preflight + worker validate                                 |
| `kv_cache_attention_bias`     | float                                | `tp_plan=="v1_generator_only"`                         | soft transition scalar must match                    | rank0 preflight + worker validate                                 |
| `tp_do_kv_recompute`          | bool                                 | `tp_plan=="v1_generator_only"`                         | lockstep call-count plan                             | rank0 preflight + worker validate                                 |
| `tp_num_denoise_steps`        | int                                  | `tp_plan=="v1_generator_only"`                         | lockstep call-count plan                             | rank0 preflight + worker validate                                 |
| `tp_expected_generator_calls` | int                                  | `tp_plan=="v1_generator_only"`                         | belt+suspenders against FM-02                        | rank0 preflight + worker validate                                 |
| `denoising_step_list`         | Tensor[int64]                        | `tp_plan=="v1_generator_only"`                         | identical timesteps across ranks                     | rank0 preflight + worker validate (len match)                     |
| `latents`                     | Tensor[bf16/fp16]                    | `tp_plan=="v1_generator_only"`                         | identical generator input (avoid per-rank RNG)       | rank0 preflight + worker validate                                 |
| `conditioning_embeds`         | Tensor[bf16/fp16]                    | `tp_plan=="v1_generator_only"` (bringup-safe)          | workers cannot encode; prevents stale conditioning   | rank0 preflight + worker validate                                 |
| `context_frames_override`     | Tensor[bf16/fp16] or None            | `tp_do_kv_recompute==True`                             | recompute anchor without VAE/decode buffers          | rank0 preflight + worker validate                                 |
| `base_seed`                   | int                                  | `tp_plan=="v1_generator_only"` (bringup: diagnostic)   | deterministic audit trail, future RNG modes          | rank0 preflight + worker validate                                 |
| `video`                       | Tensor[...]                          | must be **absent** when `tp_plan=="v1_generator_only"` | v1 workers should never receive pixel frames         | rank0 preflight (strip keys)                                      |

Notes for “later optimize mode” (not schema v1 bringup):

* Add `conditioning_epoch: int` and make `conditioning_embeds` required only when `epoch` increments.
* If you ever switch to lockstep RNG, make that an explicit `tp_rng_policy` plus version bump.

## Tripwire checklist

### 3) Deterministic preflight-before-header protocol (anti-stranding)

**Rank0 sender (`control.py::broadcast_infer`, before broadcasting INFER header)**

1. **Schema validate (v1)**: if `tp_plan=="v1_generator_only"`, check required fields and “forbidden keys absent” (at least strip `video`).
2. **Split tensors deterministically**:

   * extract only top-level tensors (reject nested tensors)
   * build `tensor_specs` list
   * sort `tensor_specs` by `(key, index)` for determinism
3. **Preflight meta serialization**:

   * build `payload_meta = {"kwargs": kwargs_obj, "tensor_specs": tensor_specs}`
   * `pickle.dumps(payload_meta)` must succeed (or stable JSON encode if you choose to enforce JSON-ish meta)
4. **Preflight dtype support**:

   * for each `spec["dtype"]`, call `_dtype_from_name(spec["dtype"])` and fail if unsupported
5. **Materialize tensors**:

   * move to broadcast device
   * cast to allowed dtype set (if needed)
   * `.contiguous()` (and any layout normalization you require)
6. Only now: **broadcast header**.
7. Broadcast `payload_meta` (`broadcast_object_list`).
8. Broadcast tensors in `tensor_specs` order.
9. Optional bringup: input digest parity check (must be parity-keyed env, and ideally “always collective, maybe NOOP payload”).

**Receiver (`control.py::recv_next` + worker dispatch, before generator entry)**

1. Receive header, reject unknown action/version quickly.
2. If `INFER`: receive `payload_meta`.
3. Validate `tp_plan` and `tp_envelope_version` before allocating large tensors.
4. Allocate tensors strictly from `tensor_specs` and receive them in order.
5. Reconstruct call_params.
6. Worker-side schema validate v1 again (fast, defensive).
7. Only then enter `tp_worker_infer` generator collectives.

**Last-resort rule**

* If anything throws after the header broadcast (should be rare after preflight), treat the run as poisoned and exit process (fail-fast), rather than attempting to “recover and continue.”

### 4) Lockstep phase / call-count plan and assertion points

**Plan bits (must be present in v1 payload)**

* `tp_do_kv_recompute: bool`
* `tp_num_denoise_steps: int`
* `tp_expected_generator_calls: int`
* reset bits:

  * `tp_reset_kv_cache: bool`
  * `tp_reset_crossattn_cache: bool`

**Canonical consistency checks**

* `tp_num_denoise_steps == len(denoising_step_list)`
* `tp_expected_generator_calls == (1 if tp_do_kv_recompute else 0) + tp_num_denoise_steps`
  (If you add phases later, bump envelope version and extend the formula.)

**Where to assert**

* **Rank0 preflight (`frame_processor.py`, before calling `broadcast_infer`)**

  * compute the plan bits once and insert them into `call_params`
  * assert consistency checks above
  * assert `context_frames_override` present iff `tp_do_kv_recompute`
* **Worker preflight (`tp_worker.py` or inside `tp_worker_infer`, before first generator call)**

  * recompute expected call count from plan bits and assert it equals `tp_expected_generator_calls`
  * assert required tensors exist and shapes are compatible with `(height,width,current_start_frame)`
* **Worker runtime (inside `tp_worker_infer`)**

  * maintain `observed_generator_calls` and assert equals `tp_expected_generator_calls` before returning
* Optional bringup: `all_gather(observed_generator_calls)` every N chunks (parity-keyed env)

### 5) Determinism posture (bringup mode vs later optimize mode)

**Bringup mode (recommended defaults for first “v1 is safe” runs)**

* **Broadcast full generator inputs every chunk**:

  * require `conditioning_embeds` every chunk
  * require `latents` every chunk
  * require `context_frames_override` whenever recompute will run
* **Pin backend choices across ranks** (env parity): especially KV-bias backend (no `auto`).
* **Enable drift tripwires**:

  * `SCOPE_TP_INPUT_DIGEST=1` (every chunk)
  * bringup-only generator output digest parity (post-forward `all_gather`)
  * shard fingerprint checks at init and optionally on `init_cache`
* **Keep RNG boring**:

  * do not rely on lockstep RNG consumption for correctness
  * treat randomness as data (carried in `latents`) during bringup

**Later optimize mode (only after stable v1 bringup)**

* Conditioning bandwidth reduction:

  * add `conditioning_epoch`
  * broadcast `conditioning_embeds` only when epoch increments
  * keep `tp_reset_crossattn_cache` tied to epoch changes
* Relax digests:

  * input digest every N chunks + sampled bytes
  * output digest parity off by default
* Consider lockstep RNG only if you can prove “same RNG calls, same order” across ranks and phases (I would keep broadcasting latents unless bandwidth is actually a bottleneck).

### 6) Top 3 stale override reuse hazards + fixes

1. **FM-15: `context_frames_override` persists in state**

* Hazard: recompute block consumes override but does not clear it; a later chunk that omits the key can reuse stale frames silently.
* Fix:

  * in `recompute_kv_cache.py`: clear `context_frames_override` after read (always, even on branches)
  * in v1 envelope: always include `context_frames_override=None` when not used (explicitly present key)
  * add worker-side assert: if `tp_do_kv_recompute==False`, then `context_frames_override is None`

2. **FM-15: `video_latents_override` (or equivalent) persists**

* Hazard: prepare-latents block can reuse stale latents if key omitted on later chunk.
* Fix:

  * clear override after consumption in `prepare_video_latents.py` (always)
  * in v1 plan: stop using in-block broadcasts and prefer `latents` as the direct generator input, so there is no “latent override stored in state” at all

3. **FM-06/FM-17: “missing conditioning tensor” falls back to stale cached conditioning**

* Hazard: in v1 generator-only workers, any “conditioning only on update hints” scheme can silently reuse stale conditioning.
* Fix:

  * bringup: require `conditioning_embeds` every chunk (schema v1)
  * optimize later: introduce `conditioning_epoch` and assert epoch monotonicity and cache reset semantics across ranks

## Break-it tests (mapped to FM IDs)

> Goal for every test: crashes quickly with a crisp log, no multi-minute hang.

1. **FM-04: Meta serialization failure (anti-stranding)**

* Inject into payload meta something non-picklable (example: a lambda) so `pickle.dumps(payload_meta)` would fail.
* Expected signature:

  * rank0 throws **before** broadcasting INFER header
  * workers are not stranded in a half-received payload (they remain waiting for the next header)
  * log contains: `call_id`, `chunk_index`, and the failing meta path/key

2. **FM-07: Unsupported dtype discovered late**

* Add a tensor field with an unsupported dtype for the broadcast mapping (example: `torch.bool`), and ensure it would appear in `tensor_specs`.
* Expected signature (after implementing preflight):

  * rank0 fails **before** header broadcast with “unsupported dtype” including the field name
  * no worker hang
* Regression signal (if preflight is missing):

  * rank0 sends header then throws mid-broadcast, worker blocks waiting for tensors (watchdog/NCCL timeout later)

3. **FM-02: Call-count mismatch**

* Create a self-inconsistent phase plan:

  * example: `tp_num_denoise_steps=5` but `denoising_step_list` length is 4, or `tp_expected_generator_calls` does not match the formula.
* Expected signature:

  * rank0 preflight rejects envelope before broadcast, or worker rejects immediately after receive
  * error message shows both values (expected vs observed), includes `call_id/chunk_index`
  * no deadlock because mismatch is caught before entering generator collectives

4. **FM-03: Per-rank flag mismatch gating a collective**

* Launch with an env var that gates a collective enabled on only one rank (example: enable `SCOPE_TP_INPUT_DIGEST` or a new debug all_gather flag on rank0 only).
* Expected signature:

  * env parity check fails at init (or at least before first INFER), with explicit key diff
  * no hang in the first collective
* Regression signal:

  * intermittent hang when the gated collective triggers on only one rank
