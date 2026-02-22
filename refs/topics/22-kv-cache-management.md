---
status: draft
---

# Topic 22: KV cache management in streaming inference

KV caches store the key/value projections from previous tokens to avoid recomputation during autoregressive generation. In video DiT streaming, this extends to **temporal KV caches** across denoising steps and frames. PagedAttention (from vLLM) introduced OS-style paged memory management, reducing waste from **60-80% to under 4%**.

## Resources

<!-- Resource IDs from manifest belonging to this topic -->
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| pagedattention | Efficient Memory Management for Large Language Model Serving with PagedAttention | high | pending |
| causvid | From Slow Bidirectional to Fast Autoregressive Video Diffusion Models | high | converted |
| continuous-batching | Continuous Batching from First Principles | medium | pending |
| vllm-anatomy | Inside vLLM: Anatomy of a High-Throughput LLM Inference System | medium | pending |
| vllm-distributed | vLLM Distributed Inference Blog Post | medium | pending |

## Implementation context

The Scope KV-cache is a **fixed-size ring buffer** (~32K tokens, head-sharded in TP mode) with eviction and recompute. The cache lifecycle (hard cut, recompute, soft transition via `kv_cache_attention_bias`) is a major coupling point: recompute depends on `decoded_frame_buffer`, which is why v0 workers run the full pipeline. The `cache_epoch` counter in `PPEnvelopeV1` tracks hard-cut generations. The `SCOPE_KV_CACHE_RECOMPUTE_EVERY=2` experiment (Run 11) was a dead end — quality degradation with no net FPS gain.

See: `refs/implementation-context.md` → Phase 2-3, `scope-drd/notes/FA4/h200/tp/explainers/05-kv-cache-head-sharding.md`.

Relevant Scope code:
- `scope-drd/src/scope/core/distributed/pp_contract.py` (`cache_epoch`, `do_kv_recompute`, explicit reset flags)
- `scope-drd/src/scope/core/pipelines/krea_realtime_video/blocks/recompute_kv_cache.py` (recompute path + coupling to decoded frames)
- `scope-drd/src/scope/server/frame_processor.py` (hard cuts / soft transitions, `kv_cache_attention_bias` override lifecycle)

## Synthesis

### Mental model

In streaming video inference, the KV cache is the generator’s **hidden state across chunks**. It is simultaneously:

- a **performance feature** (reuse past K/V instead of recomputing), and
- a **correctness contract** (if cache state diverges, everything after can look “plausible” but be wrong).

Scope’s KV cache is a fixed-size **ring buffer** over *tokens* (frames expand to `frame_seq_length` tokens). Each chunk appends new entries and attends over the active window; when it fills, it evicts old entries and may run a **recompute** pass to rebuild the relevant cache slice. (`scope-drd/notes/FA4/h200/tp/explainers/05-kv-cache-head-sharding.md`)

The key operational takeaway is: **KV cache management is not a local optimization knob**. It’s part of the distributed protocol:

- In **TP v0/v1.1**, correctness comes from lockstep execution: both ranks receive identical `call_params`, update cache indices deterministically, and never broadcast cache contents. (`scope-drd/notes/FA4/h200/tp/explainers/05-kv-cache-head-sharding.md`, `scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md`)
- In **PP bringup**, the generator KV cache lives on the mesh side, but rank0 still controls cache lifecycle via the envelope: hard cuts must become epoch-bounded (`cache_epoch`), in-flight work must be droppable, and recompute coupling must be handled explicitly (e.g., `context_frames_override`). (`scope-drd/notes/FA4/h200/tp/pp-topology-pilot-plan.md`, `scope-drd/notes/FA4/h200/tp/pp-next-steps.md`)

### Key concepts

- **Head-sharded KV cache (TP correctness + memory)**
  - The cache is allocated per layer as `[batch, max_tokens, heads, head_dim]` and sharded by heads in TP mode (e.g., 40 → 20 heads per rank at TP=2). Cache contents are never exchanged; only the post-attention projection needs an all_reduce. (`scope-drd/notes/FA4/h200/tp/explainers/05-kv-cache-head-sharding.md`)
  - Practical implication: the cache is *huge* even when sharded (order-of-10GB per rank at TP=2 for the full model), so “copy/migrate the cache” is not a viable baseline mechanism. (`scope-drd/notes/FA4/h200/tp/explainers/05-kv-cache-head-sharding.md`)

- **Ring-buffer indexing: `global_end_index` vs `local_end_index`**
  - `global_end_index` is the monotonic position in the global token timeline; `local_end_index` maps that into the fixed-size ring after evictions/rolls. The write window is derived deterministically from `call_params` (start offset + token count), so ranks stay consistent without extra coordination. (`scope-drd/notes/FA4/h200/tp/explainers/05-kv-cache-head-sharding.md`)

- **Cache lifecycle events are correctness-critical**
  - **Hard cut** (`init_cache=True`): resets cache state; must happen on all participating ranks/stages at the same logical boundary. (`scope-drd/notes/FA4/h200/tp/explainers/05-kv-cache-head-sharding.md`, `scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md`)
  - **Recompute**: when the cache fills, the system re-encodes recent context and rewrites the cache; this triggers another generator pass and therefore participates in the same lockstep/collective invariants as normal denoise. (`scope-drd/notes/FA4/h200/tp/explainers/05-kv-cache-head-sharding.md`, `scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md`)
  - **Soft transition / forgetting** (`kv_cache_attention_bias`): a scalar bias applied in attention; must be identical across ranks or you induce drift. (`scope-drd/notes/FA4/h200/tp/explainers/05-kv-cache-head-sharding.md`)

- **The recompute coupling problem (why “generator-only workers” is non-trivial)**
  - In steady state, the recompute anchor frame is derived from `decoded_frame_buffer` via a VAE re-encode path. If a worker/mesh stage does not perform decode (or cannot VAE-encode), it must be provided an override tensor (`context_frames_override` / `context_frames` in the PP envelope), or recompute semantics break. (`scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md`, `scope-drd/notes/FA4/h200/tp/pp-topology-pilot-plan.md`)
  - This is why TP v0 kept the full pipeline on workers, and why PP plans explicitly stage recompute as “disable first (R1), then restore via rank0-provided override (R0a).” (`scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md`, `scope-drd/notes/FA4/h200/tp/pp-next-steps.md`, `scope-drd/notes/FA4/h200/tp/pp0-bringup-runbook.md`)

### Cross-resource agreement / disagreement

- **External resources emphasize memory efficiency; Scope emphasizes lifecycle correctness first**
  - PagedAttention / vLLM-style systems focus on reducing KV memory waste from variable sequence lengths via paged allocation and block managers.
  - Scope’s immediate KV pain is different: cache **lifecycle** (hard cuts, eviction, recompute) and **distributed coupling** (TP lockstep; PP stage boundary + recompute anchor). Head-sharding + deterministic indices already solve “don’t broadcast cache” efficiently; the next frontier is making lifecycle safe under PP. (`scope-drd/notes/FA4/h200/tp/explainers/05-kv-cache-head-sharding.md`, `scope-drd/notes/FA4/h200/tp/pp-topology-pilot-plan.md`)

- **Internal docs align on the central coupling**
  - The failure-modes explainer calls out KV lifecycle divergence and recompute coupling as a primary Franken-model risk. (`scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md`)
  - The PP topology plan and next-steps doc treat recompute as the “decide early” coupling and explicitly propose R1 → R0a sequencing, plus `cache_epoch`-based invalidation on hard cuts. (`scope-drd/notes/FA4/h200/tp/pp-topology-pilot-plan.md`, `scope-drd/notes/FA4/h200/tp/pp-next-steps.md`)

- **Measured performance supports the PP motivation**
  - Block profiling shows `decode` + `recompute_kv_cache` are ~33% of measured GPU time per chunk in TP mode, motivating a topology where rank0 owns decode and mesh owns generator. (`scope-drd/notes/FA4/h200/tp/bringup-run-log.md`)
  - “Reduce recompute frequency” (`SCOPE_KV_CACHE_RECOMPUTE_EVERY=2`) was empirically a dead end: quality loss with no net FPS gain. (`scope-drd/notes/FA4/h200/tp/bringup-run-log.md`)

### Practical checklist

**Goal**: keep cache lifecycle deterministic and auditable as you move from TP v0 → TP v1.1 → PP0/PP1.

1. **Make cache lifecycle explicit in the control-plane plan**
   - Drive all cache events from per-chunk params/envelopes: `init_cache`, `reset_kv_cache`, `reset_crossattn_cache`, `do_kv_recompute`, `kv_cache_attention_bias`, and `expected_generator_calls`. (`scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md`, `scope-drd/notes/FA4/h200/tp/pp-topology-pilot-plan.md`)
   - Assert `observed_generator_calls == expected_generator_calls` so “recompute happened on one side but not the other” can’t silently slip through. (`scope-drd/notes/FA4/h200/tp/pp0-bringup-runbook.md`)

2. **Pin backend selection across ranks**
   - Do not use per-rank `auto` backend selection for KV-bias attention; pin `SCOPE_KV_BIAS_BACKEND` (e.g., `fa4` vs `flash`) and include it in env parity checks. (`scope-drd/notes/FA4/h200/tp/explainers/05-kv-cache-head-sharding.md`)

3. **Treat recompute as a first-class distributed feature (not an afterthought)**
   - For PP0 bringup, start in **R1** (disable recompute) to prove contracts/queues first: `SCOPE_KV_CACHE_RECOMPUTE_EVERY=999999`. (`scope-drd/notes/FA4/h200/tp/pp-next-steps.md`, `scope-drd/notes/FA4/h200/tp/pp0-bringup-runbook.md`)
   - To restore correctness semantics, move to **R0a**: rank0 must provide `context_frames_override` / `context_frames` when `do_kv_recompute=True`; mesh must use the override and never fall back to VAE re-encode. (`scope-drd/notes/FA4/h200/tp/pp-next-steps.md`, `scope-drd/notes/FA4/h200/tp/pp0-bringup-runbook.md`)
   - Only as a later experiment, consider **R0** (semantic change): mesh uses a latent-only anchor (no decoded-anchor dependency). Treat this as an explicit quality-risk trade and gate it behind a validation run; don’t silently “accidentally do R0” while chasing overlap. (`scope-drd/notes/FA4/h200/tp/pp-topology-pilot-plan.md`)

4. **In PP, couple hard cuts to epoching + queue flush**
   - On hard cut, flush bounded queues and increment `cache_epoch` so stale in-flight results are self-identifying and droppable. (`scope-drd/notes/FA4/h200/tp/pp-topology-pilot-plan.md`)

5. **Audit for out-of-band cache mutation**
   - Any cache reset or lifecycle mutation that bypasses the broadcast/envelope (e.g., rank0-only side effects from a server endpoint) is a direct Franken-model risk. (`scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md`)

#### Minimal deterministic contract (TP v1.1 + PP)

The “operator manual” framing: KV-cache lifecycle must be an explicit distributed **contract**. If it isn’t, you will either (a) hang (call-count mismatch) or (b) drift silently (Franken-model).

**Canonical field mapping (naming drift is not allowed):**
- TP v1.1: `tp_plan`, `tp_envelope_version`, `tp_do_kv_recompute`, `tp_num_denoise_steps`, `tp_expected_generator_calls`, `tp_reset_kv_cache`, `tp_reset_crossattn_cache`.
- PP: `pp_envelope_version`, `do_kv_recompute`, `num_denoise_steps`, `expected_generator_calls`, `reset_kv_cache`, `reset_crossattn_cache`.
- Recompute context tensor:
  - TP: `context_frames_override`
  - PP: sometimes called `context_frames`
  - **Canonical semantics**: same tensor, required whenever recompute runs.

**P0 fields (minimum)**

| Field | Why | Fail-fast check |
|---|---|---|
| `plan_id` (`tp_plan` / PP enable) | prevents divergent codepaths | env parity + per-message assert |
| `*_envelope_version` | schema mismatch must crash | reject unknown version pre-collective |
| `call_id` | ordering + replay boundary | monotonic; crash if decreases |
| `chunk_index` | output ordering | monotonic for `INFER` |
| `cache_epoch` | hard-cut invalidation | bump on reset boundary; PP drops stale results |
| `init_cache` | explicit hard cut | required on first chunk of epoch |
| `reset_kv_cache` | stop per-rank inference | must be identical across participants |
| `reset_crossattn_cache` | same as above | must be identical |
| `current_start_frame` | cache indexing depends on it | broadcast scalar; treat as source of truth |
| `do_kv_recompute` | recompute adds a generator call | must be explicit |
| `num_denoise_steps` | controls call count | must be explicit |
| `expected_generator_calls` | prevents hangs | worker counts calls and asserts |
| `kv_cache_attention_bias` | soft forgetting must match | broadcast scalar every chunk |
| `denoising_step_list` (or equivalent) | scheduler drift can change call graph | treat as required input, not inferred |
| `height`, `width` | shape sanity for context/latents | validate tensors match geometry |

#### Cache lifecycle sketch (state machine)

Treat the lifecycle as a small state machine driven by the per-chunk plan:

- **RESET (hard cut)**: `init_cache=True` (and explicit reset bits) starts a new epoch (`cache_epoch += 1` in PP). All indices are reinitialized.
- **STEADY**: each chunk appends tokens, advances indices deterministically from `current_start_frame` and token counts.
- **RECOMPUTE** (when scheduled/required): an extra generator call that rebuilds the cache slice for the active window. This must be explicit (`do_kv_recompute=True`) and must consume `context_frames_override` in generator-only workers / PP (no fallback).
- **ADVANCE**: after recompute + denoise, the cache advances and the chunk commits; call-count must match `expected_generator_calls`.

Operational note: in TP v0/v1.1, state-machine alignment comes from lockstep execution; in PP it comes from an explicit envelope plus drop-by-epoch semantics at the stage boundary. (`scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md`, `scope-drd/notes/FA4/h200/tp/pp-topology-pilot-plan.md`)

**Required tensors (bringup-safe default: broadcast every chunk until stable)**

| Tensor | Required when | Why | Validation |
|---|---|---|---|
| `latents_in` | always | generator inputs must be identical | dtype/shape exact match |
| `conditioning_embeds` (or override) | always in bringup | avoid worker text-encoder divergence | dtype/shape; later can be epoch-based |
| `denoising_step_list` | always (if used) | prevents scheduler divergence | dtype/shape; length matches `num_denoise_steps` |
| `context_frames_override` | whenever `do_kv_recompute=True` (v1.1c/PP) | avoids decoded-anchor coupling + fallback | must be present; mesh must not fall back |

**Env parity keys that must include cache semantics**
- `SCOPE_KV_CACHE_RECOMPUTE_EVERY` (call-count gating)
- `SCOPE_KV_BIAS_BACKEND` (backend parity)
- `SCOPE_TP_PLAN` (role/plan parity)
- `SCOPE_PP_ENABLED` (topology parity)
- Compile gating flags that change collective behavior (`SCOPE_TP_ALLOW_COMPILE`, etc.)

#### Failure modes: what hangs vs what silently drifts

The hazards that cause hangs are the ones that matter at 2am; the rest will silently ruin output if you don’t have digests/fingerprints.

| Hazard | Failure class | Typical cause | Tripwire | Where to assert |
|---|---|---|---|---|
| Plan mismatch | NCCL hang | one rank runs v0, other runs v1/PP | env parity on `SCOPE_TP_PLAN` / `SCOPE_PP_ENABLED` | init (`runtime.py`) |
| Generator-call count mismatch | NCCL hang | recompute scheduled on one side only, step count differs | broadcast `expected_generator_calls`; assert observed call count | rank0 preflight + worker/mesh runner |
| Post-header exception strands peer | NCCL hang | sender emits header then fails building specs/tensors | preflight-before-header (pickle/meta/spec/dtype) | sender control plane |
| KV reset decision drift | Franken-model | one side resets KV but other doesn’t | explicit `reset_kv_cache` + `cache_epoch` rules | rank0 decision + worker/mesh validation |
| Cross-attn cache reset drift | Franken-model | `conditioning_embeds_updated` inferred differently | broadcast explicit reset bit/epoch | rank0 preflight + cache-setup path |
| Transition state drift | Franken-model | worker didn’t run blending so `_transition_active` differs | don’t infer; broadcast final reset decisions | rank0 preflight |
| `current_start_frame` drift | Franken-model | mesh advances differently or rank0 sends wrong scalar | broadcast scalar; optional scalar all_gather every N | worker/mesh runner |
| Missing recompute override | Franken-model / crash | worker/mesh falls back to VAE path or wrong context | require `context_frames_override` when `do_kv_recompute` | rank0 preflight; mesh validate-before-run |
| Backend mismatch (FA4 vs flash) | Franken-model | auto backend differs per rank | pin + parity-check effective backend | init + startup report |
| Non-deterministic meta encoding | false positive trips | JSON key order, float formatting | canonical meta serialization | digest implementation |

#### Tripwire checklist (where to assert)

**Rank0 preflight (before sending any header that commits the receiver):**
- Validate schema: version, plan, required fields present.
- Validate phase plan: `expected_generator_calls` computed and consistent.
- Validate tensors: dtype supported; shapes match geometry; contiguous if required.
- Validate meta serialization: picklable or stable JSON; no nested tensors in meta.
- Only then send header → meta/specs → tensors.

**Mesh leader preflight (PP, before any `mesh_pg` collective):**
- Receive full envelope p2p.
- Validate envelope fully.
- Only then broadcast within `mesh_pg`.

**Worker / mesh runner checks (Phase B entry and exit):**
- Assert `plan_id` matches role.
- Assert `call_id`, `chunk_index`, `cache_epoch` monotonicity rules.
- Assert `current_start_frame` matches expected stream progression.
- Assert `expected_generator_calls` equals observed calls before returning.

**Periodic lightweight parity checks (cheap and effective):**
- Scalar all_gather every N chunks (N=64 or 128): `cache_epoch`, `current_start_frame`, `do_kv_recompute`, `num_denoise_steps`, `kv_cache_attention_bias`.
- Optional input digest on envelope + tensors during bringup (`SCOPE_TP_INPUT_DIGEST=1`).
- Shard fingerprint baseline and periodic recheck to detect Franken-model.

**Optional: KV lifecycle digest (scalars/indices only)**
- Every N chunks (N=128 is a reasonable starting point), compute a small hash over:
  - envelope scalars: `cache_epoch`, `call_id`, `chunk_index`, `current_start_frame`, `do_kv_recompute`, `num_denoise_steps`, `kv_cache_attention_bias`
  - plus 1–2 representative cache index scalars (e.g., `global_end_index`, `local_end_index`) from a fixed layer chosen deterministically
- All-gather the hash across TP ranks / mesh ranks and crash if it differs.

This is cheaper than full tensor digests and catches “lockstep but state drifted” bugs earlier than subjective output inspection. (Motivated by `scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md` Franken-model class.)

#### Break-it tests (minimal, high value)

1. **Plan mismatch**: set `SCOPE_TP_PLAN` (or PP enable) differently across ranks → must fail at init via env parity (no hang).
2. **Generator-call count mismatch**: force `do_kv_recompute=True` on rank0 but false on worker/mesh → must fail-fast via `expected_generator_calls` assert (not hang).
3. **Missing recompute override**: schedule recompute but omit `context_frames_override` → rank0 preflight fails pre-header (or leader rejects pre-broadcast); no mesh fallback to VAE path.
4. **Cache reset drift**: trigger cache reset on rank0 only (simulated bad endpoint) → must trip state digest or input digest quickly.
5. **`current_start_frame` drift**: perturb start frame on one side for one chunk → must trip scalar all_gather or runner asserts.
6. **PP stale result after hard cut**: delay a mesh result, then hard cut (`cache_epoch` increments), then deliver delayed result → rank0 must drop by epoch mismatch and never emit it.

### Gotchas and failure modes

- **“Just skip decode on workers/mesh” breaks recompute**
  - Without `decoded_frame_buffer`, steady-state recompute either crashes or uses different inputs → cache divergence → silent corruption. This is the core blocker for generator-only workers unless you supply override tensors. (`scope-drd/notes/FA4/h200/tp/explainers/05-kv-cache-head-sharding.md`, `scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md`, `scope-drd/notes/FA4/h200/tp/pp-topology-pilot-plan.md`)

- **Hard cuts without epoching are dangerous under PP**
  - If you reset cache state but accept late results from the previous cache world, you effectively splice two different histories into one output stream. PP uses `cache_epoch` specifically to avoid this class. (`scope-drd/notes/FA4/h200/tp/pp-topology-pilot-plan.md`)

- **Backend mismatch can look like “mysterious quality drift”**
  - Different attention backends (FP8 FA4 vs FP16 flash) can introduce small numeric differences that compound across chunks; treat backend choice as a contract, not a suggestion. (`scope-drd/notes/FA4/h200/tp/explainers/05-kv-cache-head-sharding.md`)

- **Recompute frequency is not a free knob**
  - `SCOPE_KV_CACHE_RECOMPUTE_EVERY=2` caused visible quality glitches and did not improve end-to-end FPS in measured runs; don’t treat it as a safe perf lever without quality validation. (`scope-drd/notes/FA4/h200/tp/bringup-run-log.md`)

- **Mesh “generator-only” requires careful block refactors**
  - Some cache-setup paths currently assume VAE presence (e.g., `SetupCachesBlock` clearing VAE cache). In true generator-only mesh, these must be made optional or moved to rank0. (`scope-drd/notes/FA4/h200/tp/pp-topology-pilot-plan.md`)

### Experiments to run

1. **PP0/R0a recompute override enforcement**
   - Enable recompute (`SCOPE_KV_CACHE_RECOMPUTE_EVERY=1`) and intentionally omit `context_frames` when `do_kv_recompute=True`.
   - Expected: envelope validation fails pre-send or mesh refuses to proceed; logs show “override required” and no silent fallback to VAE re-encode. (`scope-drd/notes/FA4/h200/tp/pp-next-steps.md`, `scope-drd/notes/FA4/h200/tp/pp0-bringup-runbook.md`)

2. **Hard cut + late result (epoch drop)**
   - Trigger a hard cut while an older chunk is in flight and force its result to arrive late.
   - Expected: rank0 drops by `cache_epoch` mismatch and does not decode/emit it. (`scope-drd/notes/FA4/h200/tp/pp-topology-pilot-plan.md`)

3. **Backend mismatch tripwire**
   - Intentionally run mismatched KV-bias backends across ranks (e.g., `flash` vs `fa4`) and confirm drift detectors (digest/fingerprint) or explicit env-parity checks catch it early. (`scope-drd/notes/FA4/h200/tp/explainers/05-kv-cache-head-sharding.md`, `scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md`)

4. **Recompute frequency regression test**
   - Re-run the “every=2” experiment and verify the observed behavior (quality loss, no net FPS gain) to prevent future “quick win” regressions. (`scope-drd/notes/FA4/h200/tp/bringup-run-log.md`)
