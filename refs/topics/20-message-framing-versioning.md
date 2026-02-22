---
status: draft
---

# Topic 20: Message framing and versioning

When pipeline stages communicate over network boundaries (multi-node inference), you need message framing (length-prefixed or delimited) and versioning for forward/backward compatibility as the system evolves.

## Resources

<!-- Resource IDs from manifest belonging to this topic -->
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| message-framing | Message Framing | low | pending |
| framing-textbook | Framing (Computer Networks: A Systems Approach) | low | pending |
| protobuf-guide | Protocol Buffers Language Guide (proto3) | low | pending |

## Implementation context

Both TP and PP bringup converged on the same pattern: a tiny fixed header (`call_id`, `chunk_index`, epochs, `action`) that determines whether a payload follows, plus a versioned payload manifest (`tensor_specs`) that lets receivers allocate tensors safely. TP v0 uses a 5×int64 header (~40 bytes) and `TPAction.{NOOP,INFER,SHUTDOWN}`; PP contracts similarly use `PPAction` plus `pp_envelope_version=1` and now a globally monotonic `call_id` (replacing the earlier `chunk_id` field). The key guardrail is “validate/pickle/spec everything before sending the header” to avoid stranding the receiver mid-message.

See: `refs/implementation-context.md` → Phase 2, `scope-drd/notes/FA4/h200/tp/explainers/03-broadcast-envelope.md` (TPControlHeader + tensor_specs), `scope-drd/notes/FA4/h200/tp/pp-next-steps.md` (anti-stranding Step A1 + contract changes).

Relevant Scope code:
- `scope-drd/src/scope/core/distributed/control.py` (TPControlHeader + tensor_specs framing)
- `scope-drd/src/scope/core/distributed/pp_contract.py` (`PPEnvelopeV1`/`PPResultV1` versioned schema)
- `scope-drd/src/scope/core/distributed/pp_control.py` (PP preflight + header-first send pattern)

## Synthesis

### Mental model

Distributed inference control planes are “tiny messages that prevent huge hangs.”

You have two distinct problems:

1. **Framing**: the receiver must know *exactly* what bytes/tensors to expect next (and in what order), so it can allocate safely and avoid reading past boundaries or blocking forever.
2. **Versioning**: both sides must agree on the schema. If they don’t, the failure mode should be a fast crash *before* any collectives, not a silent mismatch that turns into a deadlock later.

The Scope pattern (used in TP today and being hardened for PP) is:

- **Header-first**: a tiny fixed header that is always received in full and determines whether a payload follows (`NOOP/INFER/SHUTDOWN`) and how to interpret it (version + monotonic IDs).
- **Manifested payload**: metadata is serialized deterministically (small, CPU-friendly), and tensors are described by a **versioned `tensor_specs` manifest** so receivers can allocate and receive tensors without “shape inference.”

This is not “networking trivia.” The header is a **commitment**: if you send an `INFER` header, the receiver will block expecting meta/specs/tensors. Therefore the top correctness rule (from both TP and PP bringup) is:

> **Validate + serialize + materialize specs BEFORE sending the header** (crash > hang).

TP example: `scope-drd/notes/FA4/h200/tp/explainers/03-broadcast-envelope.md` describes a 5×int64 `TPControlHeader` broadcast that gates whether payload follows. PP example: `scope-drd/notes/FA4/h200/tp/pp-next-steps.md` Step A1 calls out “anti-stranding” as the first hardening task for PP send semantics.

### Key concepts

- **Version fields (`tp_envelope_version` / `pp_envelope_version`)**:
  - Every message must carry an explicit version integer (and ideally a plan/role id) so the receiver can reject unknown schemas immediately.
  - In practice, version belongs in both (a) the payload meta (for schema dispatch) and (b) the “things you assert at init” (env parity / startup handshake).

- **Action header (`NOOP/INFER/SHUTDOWN`)**:
  - `INFER` means “payload follows.”
  - `NOOP` means “no payload; loop back immediately.”
  - `SHUTDOWN` means “no payload; exit cleanly.”
  This is how you avoid allocating/receiving tensors when nothing should be sent.
  - **Policy**: validate/serialize/spec everything *before* emitting the “payload follows” commitment. This applies equally to:
    - rank0 sending a p2p PP envelope header to the leader, and
    - the leader starting any `mesh_pg` broadcast loop (leader is a “sender” to the mesh). See: `deep-research/2026-02-22/pp-rank0-out-of-mesh/reply.md` (“Anti-stranding protocol ordering”).

- **Monotonic identifiers**:
  - **`call_id`**: globally monotonic across *all actions* (TP and PP are converging on this). Used for ordering checks and to prevent “stale work” from being mistaken as current work.
  - **`chunk_index`**: monotonic for `INFER` actions only (video chunk counter). Useful for output ordering and “did we skip a chunk?” asserts.
  - **`cache_epoch`**: increments on hard cut / `init_cache=True`; used to invalidate stale in-flight results and to derive message tags.
  TP adds `control_epoch` as a second monotonic sequence number; the important property is “receiver can detect out-of-order and crash fast.”

- **Deterministic meta serialization (stable JSON)**:
  - Meta should be “plain data”: ints/floats/bools/strings/lists/dicts with stable key ordering.
  - Serialize it deterministically (e.g., stable JSON with sorted keys) so digests/logging are meaningful and you don’t get accidental nondeterminism across ranks.
  - Whether the transport is `broadcast_object_list` (TP) or p2p `send` (PP), **pre-serialize** to bytes so failures happen *before* the header.

- **`tensor_specs` manifest**:
  - A list describing each tensor field (`key`, `shape`, `dtype`, and any indexing for list entries) in a deterministic order.
  - Receiver allocates tensors from specs (no ad-hoc “infer shape locally”), then receives tensors in exactly that order.
  - Deterministic ordering matters for debuggability and for “update” style refactors later.

- **Ban nested tensors in meta**:
  - TP v0 explicitly fails fast if tensors appear inside nested objects (to avoid “tensor falls through pickled object path” and device/lifetime footguns). See `scope-drd/notes/FA4/h200/tp/explainers/03-broadcast-envelope.md`.
  - Apply the same rule to PP: meta is CPU-only; tensors are transported only via the manifest.

- **Anti-stranding (preflight before header)**:
  - The most important ordering rule from both PP Step A1 and the TP deadlock audit is: **compute everything that could throw before sending the header**.
  - “Everything that could throw” includes: schema validation, meta serialization, dtype support checks, spec generation, and tensor materialization/casting/contiguity.

- **Leader validation before mesh collectives (PP)**:
  - In PP, the mesh leader receives p2p from rank0 and then broadcasts within `mesh_pg`. If the leader throws after entering a mesh collective, other mesh ranks strand.
  - Therefore the leader must validate *before any* mesh collective. See `scope-drd/notes/FA4/h200/tp/pp-control-plane-pseudocode.md`.

- **Tagging to prevent cross-chunk confusion**:
  - For p2p (`send`/`recv`) or interleaving multiple fields, derive tags from `(cache_epoch, call_id, field_id)` so stale or out-of-order fields can’t be misinterpreted as current work. See `scope-drd/notes/FA4/h200/tp/pp-control-plane-pseudocode.md`.

### Cross-resource agreement / disagreement

- **TP and PP agree on the core shape**: both converge on a tiny action header + versioned payload manifest (meta + `tensor_specs` + tensors). TP already does this with `TPControlHeader` and top-level tensor extraction; PP is explicitly hardening the same shape (anti-stranding Step A1; versioned `PPEnvelopeV1`/`PPResultV1` contracts).
  - TP reference: `scope-drd/notes/FA4/h200/tp/explainers/03-broadcast-envelope.md`.
  - PP reference: `scope-drd/notes/FA4/h200/tp/pp-next-steps.md` and `scope-drd/notes/FA4/h200/tp/pp-control-plane-pseudocode.md`.

- **“Crash > hang” is consistent across docs**:
  - PP planning docs treat fail-fast as a bringup guardrail.
  - The TP v1.1 deadlock audit makes it explicit: preflight before header, validate before collectives, and prefer watchdog timeouts over silent hangs. See `scope-drd/notes/FA4/h200/tp/5pro/10-v11-correctness-deadlock-audit/response.md`.

- **Where TP and PP differ is transport shape, not principles**:
  - TP uses collectives (`dist.broadcast`, `broadcast_object_list`) which already provide message boundaries, but still has the “post-header exception strands workers” hazard if preflight is missing.
  - PP uses p2p to the leader plus a mesh broadcast, which makes “validate before any mesh collective” and correct tag derivation more critical.

- **Stable JSON vs pickle is a trade**:
  - Current TP/PP patterns often use Python object broadcast / pickle internally.
  - Stable JSON is stricter (fewer types), but gives deterministic bytes and clearer forward/back-compat boundaries. The key requirement either way is **pre-serialization before header** so failures can’t strand receivers mid-message.

### Practical checklist

**Header schema (TP + PP)**

1. **Include explicit version fields**: `tp_envelope_version` / `pp_envelope_version` in meta (and ideally in the header for early reject). Reject unknown versions immediately (crash > hang).
2. **Action is the first discriminant**: `NOOP/INFER/SHUTDOWN` must be in the fixed header. Receiver behavior:
   - `NOOP`: no payload; loop.
   - `SHUTDOWN`: no payload; exit cleanly.
   - `INFER`: payload must follow.
3. **Enforce monotonic IDs**:
   - `call_id` strictly monotonic across all actions.
   - `chunk_index` monotonic for `INFER` only (can reset on hard cut if that’s your contract, but then `cache_epoch` must also bump).
   - `cache_epoch` must match between envelope and result; drop stale results after hard cut (`cache_epoch` filtering).

**Payload schema**

4. **Meta must be deterministic and CPU-only**:
   - Define `metadata_dict()` to contain only JSON-serializable primitives (no tensors, no device objects, no callables).
   - Serialize meta deterministically (stable JSON) and treat it as bytes over the wire.
5. **Ban nested tensors**:
   - Extract tensors only from top-level fields; fail fast if a tensor appears in nested meta structures (TP v0 already does this).
6. **`tensor_specs` is the manifest**:
   - Generate specs deterministically (stable key order; stable ordering for list-valued tensors).
   - Receiver allocates tensors strictly from specs; no “infer my own shapes.”

**Anti-stranding ordering (most important)**

7. **Validate/pickle/spec BEFORE sending the header** (sender-side preflight):
   - `validate_before_send()` (schema + required fields by action/version).
   - Serialize meta bytes (catch pickle/JSON errors).
   - Build `tensor_specs` and preflight dtype support (catch unsupported dtype before header; see FM-07 in the deadlock audit).
   - Materialize/normalize tensors (device, dtype casting, contiguity) and gather them in spec order.
   - **Only then** send header → meta/specs → tensors.
   This is Step A1 for PP (`scope-drd/notes/FA4/h200/tp/pp-next-steps.md`) and also the fix for TP FM-04 (`scope-drd/notes/FA4/h200/tp/5pro/10-v11-correctness-deadlock-audit/response.md`).

**Receiver-side rules**

8. **Header check before any blocking receives**:
   - Validate version/action.
   - Enforce monotonic `call_id` / `chunk_index` rules; crash if violated (don’t continue into collectives).
9. **Allocate from specs, then receive**:
   - Receive meta/specs first, allocate tensors, then receive tensors in the declared order.
10. **Validate before entering generator collectives**:
   - TP worker: validate reconstructed `call_params` before calling the pipeline.
   - PP leader: validate envelope *before* broadcasting within `mesh_pg` to avoid stranding mesh ranks (`scope-drd/notes/FA4/h200/tp/pp-control-plane-pseudocode.md`).
11. **Use tags for p2p**:
   - Derive tags from `(cache_epoch, call_id, field_id)` so stale messages cannot be misinterpreted after a hard cut or retry.
12. **Bringup timeouts**:
   - Set low-ish distributed timeouts (`SCOPE_DIST_TIMEOUT_S`) and watchdogs during bringup. If something goes wrong, you want an early crash, not a 300s mystery hang.

#### Worked example: TP v1.1 generator-only envelope (`tp_plan=v1_generator_only`, `tp_envelope_version=1`)

This is the “operator manual” version of the TP v1.1 envelope contract: a strict schema + preflight ordering that prevents hangs and Franken-models when workers stop running text/VAE/decode. See: `deep-research/2026-02-22/tp-v11-envelope-contract/reply.md`.

**P0 ship blockers (v1 plan)**

- **Plan must be explicit**: select behavior via `tp_plan` + `tp_envelope_version`, never by “which tensors are present”.
- **Strict schema validation** on both sender (rank0) and receiver (worker) with “no best-effort fallback” under v1.
- **Preflight-before-header** must guarantee nothing can throw after the `INFER` header is broadcast (picklability, dtype map, deterministic `tensor_specs`, tensor materialization).
- **Ban ad-hoc in-block broadcasts** for v1: all cross-rank tensors travel via control-plane `tensor_specs` (no worker-side shape inference).
- **Explicit lockstep call-count plan**: `tp_do_kv_recompute`, `tp_num_denoise_steps`, `tp_expected_generator_calls` + reset bits must be present and cross-checked.
- **Plan-aware warmup**: warmup must use the v1 entrypoint (or be disabled) so generator-only workers don’t accidentally run the full pipeline during warmup.

**Minimal payload schema v1 (key fields)**

Header fields (`action`, `call_id`, `chunk_index`, `cache_epoch`) are still required; this table is the **payload meta + tensors** that become mandatory in v1.

| Field | Type | Required when | Why | Fail-fast check location |
|---|---|---|---|---|
| `tp_plan` | `str` | always (`INFER`) | dispatch + plan parity | rank0 preflight; worker preflight |
| `tp_envelope_version` | `int` | `tp_plan=v1_generator_only` | schema dispatch | rank0 preflight; worker preflight |
| `height`, `width` | `int` | `tp_plan=v1_generator_only` | shape sanity across ranks | rank0 preflight; worker preflight |
| `current_start_frame` | `int` | `tp_plan=v1_generator_only` | cache indexing / determinism | rank0 preflight; worker preflight |
| `tp_reset_kv_cache` | `bool` | `tp_plan=v1_generator_only` | cache reset must be explicit | rank0 preflight; worker validate |
| `tp_reset_crossattn_cache` | `bool` | `tp_plan=v1_generator_only` | cache reset must be explicit | rank0 preflight; worker validate |
| `kv_cache_attention_bias` | `float` | `tp_plan=v1_generator_only` | soft transition must match | rank0 preflight; worker validate |
| `tp_do_kv_recompute` | `bool` | `tp_plan=v1_generator_only` | adds generator call | rank0 preflight; worker validate |
| `tp_num_denoise_steps` | `int` | `tp_plan=v1_generator_only` | call-count plan | rank0 preflight; worker validate |
| `tp_expected_generator_calls` | `int` | `tp_plan=v1_generator_only` | belt+suspenders vs FM-02 | rank0 preflight; worker validate |
| `denoising_step_list` | `Tensor[int64]` | `tp_plan=v1_generator_only` | identical timesteps | rank0 preflight; worker validate |
| `latents` | `Tensor[bf16/fp16]` | `tp_plan=v1_generator_only` | identical generator input (no per-rank RNG) | rank0 preflight; worker validate |
| `conditioning_embeds` (or override tensor) | `Tensor[bf16/fp16]` | v1 bringup: every chunk | workers cannot encode; avoid stale conditioning | rank0 preflight; worker validate |
| `context_frames_override` | `Tensor[bf16/fp16] \| None` | `tp_do_kv_recompute=True` | recompute anchor w/out VAE fallback | rank0 preflight; worker validate |
| `video` | **forbidden** | `tp_plan=v1_generator_only` | workers must never receive pixel frames | rank0 preflight (strip) |

**Lockstep call-count checks (must hold on both sides)**

- `tp_num_denoise_steps == len(denoising_step_list)`
- `tp_expected_generator_calls == (1 if tp_do_kv_recompute else 0) + tp_num_denoise_steps`
- Worker must maintain `observed_generator_calls` and assert `observed == tp_expected_generator_calls` before returning from Phase B.

**Stale override reuse hazards (FM-15/FM-06/FM-17)**

- `context_frames_override` must be **cleared after read** and explicitly set to `None` when absent; omission must never reuse stale state.
- Prefer `latents` as a direct generator input over “latents override stored in state,” to avoid stale reuse.
- In v1 bringup, require conditioning every chunk; “only on update chunks” is unsafe without an explicit epoch contract (`conditioning_epoch`) and reset semantics.

**Last-resort rule (anti-stranding)**

If something still throws *after* the `INFER` header broadcast, treat the run as poisoned and fail-fast (exit) rather than attempting to continue; continuing risks stranded peers.

### Gotchas and failure modes

- **Stranding receivers (the canonical failure)**:
  - Sender sends `INFER` header, then throws before sending meta/specs/tensors ⇒ receiver blocks forever in the next receive/broadcast step.
  - This is why “preflight before header” is a hard requirement (TP FM-04; PP Step A1).

- **Unsupported dtype discovered too late**:
  - If dtype mapping fails during the receive loop (after header/meta), you strand the peer. The deadlock audit calls this out explicitly (FM-07).
  - Fix: dtype preflight during sender-side preflight; cast tensors to supported dtypes at envelope build time.

- **Non-deterministic payload ordering**:
  - If `tensor_specs` order can change across runs (or differs across ranks), debugging becomes impossible and “graph update” style future work becomes brittle.
  - Fix: deterministic ordering (sort keys; stable list indices).

- **Nested tensors in meta**:
  - If a tensor slips into meta/object serialization, it can land on the wrong device, bypass intended validation/digest checks, or balloon CPU memory.
  - Fix: fail-fast nested tensor detection (TP already enforces this; apply to PP too).

- **Version drift / plan mismatch**:
  - If one side expects a different schema/plan (e.g., generator-only vs full pipeline), you can hit collectives in different orders and hang.
  - Fix: version fields + plan id in every message and env parity checks at init; reject mismatches before any collectives.

- **Leader “throws and disappears” can strand the mesh** (PP multi-rank):
  - If the leader enters a `mesh_pg` broadcast and then throws (or exits) before completing the broadcast, non-leader mesh ranks can block indefinitely waiting for the rest of the payload.
  - Policy: leader must fully receive + validate the envelope *before* starting any `mesh_pg` broadcast; if invalid, the leader should broadcast a **terminal action** (`SHUTDOWN` or an explicit `ERROR` action/version) so non-leaders can exit cleanly rather than strand. See: `deep-research/2026-02-22/pp-rank0-out-of-mesh/reply.md` (“Invalid envelope policy”).

- **Make “commitment points” visible in logs**:
  - Log `(call_id, chunk_index, cache_epoch, action, version)` at each commitment point: header send, mesh broadcast start, and the first collective boundary in Phase B. When something wedges, these log lines localize which boundary stranded peers. See: `deep-research/2026-02-22/pp-rank0-out-of-mesh/reply.md` (“commitment point” note).

- **Out-of-order headers**:
  - If `call_id` goes backwards, you’re in undefined state. Continuing risks using stale tensors or misordered collective entry.
  - Fix: assert monotonicity and crash immediately (TP worker already does this per the broadcast envelope doc).

- **Hard cut without epoching**:
  - If you reset caches but don’t bump `cache_epoch` (and incorporate it into tags/filtering), stale in-flight results can be accepted as current work.
  - Fix: bump `cache_epoch` on hard cut, flush queues, and drop results where `cache_epoch` mismatches.

### Experiments to run

1. **Forced meta serialization failure (anti-stranding test)**:
   - Inject an unserializable value into meta (e.g., a lambda or a CUDA stream object) and attempt to send `INFER`.
   - Expected: sender throws **before sending the header**; receiver is not stranded mid-message (it should not have received an `INFER` header at all).

2. **Unsupported dtype preflight**:
   - Add a tensor field with an unsupported dtype (e.g., `torch.bool`, `torch.float64`, fp8) to the payload.
   - Expected: sender preflight fails before header (or envelope builder casts to a supported dtype); receiver does not hang waiting for tensors that will never arrive.
   - Regression signal (if preflight is missing): sender throws after header/meta and the receiver blocks forever (this reproduces TP FM-07).

3. **Out-of-order header (monotonicity tripwire)**:
   - Force `call_id` to decrease for one message (or replay an older header).
   - Expected: receiver detects monotonicity violation and **crashes immediately** (crash > hang), before any generator execution or collectives.

4. **Nested-tensor-in-meta rejection**:
   - Place a tensor inside a nested dict/list in the “meta” path (not the tensor manifest path).
   - Expected: sender fails preflight (or receiver rejects meta) before any payload send/collective, preventing wrong-device tensors and avoiding stranding.
