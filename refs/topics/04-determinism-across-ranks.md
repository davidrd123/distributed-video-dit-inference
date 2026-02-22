---
status: draft
---

# Topic 4: Determinism across ranks in distributed training/inference

Non-determinism in distributed settings comes from three sources: **CUDA atomicAdd operations** (non-associative floating-point), **cuDNN autotuning** selecting different algorithms per run, and **NCCL reduction order** across ranks.

## Resources

<!-- Resource IDs from manifest belonging to this topic -->
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| pytorch-reproducibility | PyTorch Reproducibility Guide | low | pending |
| pytorch-deterministic-api | torch.use_deterministic_algorithms API | low | pending |
| fp-non-assoc-reproducibility | Impacts of floating-point non-associativity on reproducibility for HPC and deep learning | low | pending |

## Implementation context

TP v0 treats cross-rank determinism as “detect drift and crash,” not bitwise reproducibility: optional per-call input digests (`SCOPE_TP_INPUT_DIGEST=1`) catch envelope/tensor mismatches, and shard fingerprints catch weight divergence (Franken-model). A concrete bringup bug: the initial fingerprint baseline was identical on both ranks (Run 3: `[6137181582272122429, 6137181582272122429]`), which was fixed so fingerprints diverge as expected (`[-505995101758162675, 4154699338113877488]`). Backend selection is also pinned via env parity (don’t let ranks auto-pick `flash` vs `fa4`).

See: `refs/implementation-context.md`, `scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md` (Q2), `scope-drd/notes/FA4/h200/tp/bringup-run-log.md` (Run 3 + Run 8 shard fingerprint notes), `scope-drd/notes/FA4/h200/tp/runtime-checklist.md` (Locked v0 decisions: determinism + backend pinning).

## Synthesis

### Mental model

“Determinism across ranks” for TP/PP is not “bitwise identical forever.” It’s the set of guardrails that prevent (or rapidly diagnose) the two catastrophic outcomes from `scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md`:

- **NCCL hang (control-flow divergence):** ranks hit collectives in a different order or not at all → both processes stall until timeout.
- **Franken-model (data/weight divergence):** ranks stay in lockstep on collectives, but compute from different weights/inputs → silent corruption.

The operational goal is: **either (a) stay lockstep, or (b) fail fast with a crisp diff** before you waste minutes on a hang or ship subtly-wrong video.

Practical approach (v0 TP, and reusable for PP):

- Treat the **control envelope** (`call_id`, `chunk_index`, cache/recompute flags, action) as the “contract” for what code should run.
- Treat the **payload** (tensors + meta kwargs) as the “bytes that must match.”
- Add **tripwires**:
  - **Input digest**: per-call parity check on envelope + payload (`SCOPE_TP_INPUT_DIGEST=1`).
  - **Shard fingerprint**: weight-sample hash baseline (and optional periodic re-check) to catch “Franken-model.”
- Pin or validate **environment-selected behavior** (attention / KV-bias backend, compile flags, RNG policy) so ranks don’t “helpfully” choose different kernels.

### Key concepts

1. **Numeric non-associativity (expected, usually non-fatal)**
   - Floating-point reduction order and atomic updates can produce *small* numeric differences across runs/hardware.
   - These differences matter for *reproducibility*, but they are typically not what causes **distributed deadlocks**.
   - In TP/PP, you usually tolerate “epsilon drift” as long as it does not change shapes, control decisions, or collective order.

2. **Control-flow divergence (fatal)**
   - Any per-rank difference that changes which branch executes, how many times a block runs, or when a collective is called can produce an **NCCL hang**.
   - Examples: one rank takes recompute while the other doesn’t; one rank thinks `init_cache=True`; one rank compiles a different graph and introduces a different collective boundary.
   - This is why the v0 invariant is framed in terms of **collective order**, not floating-point equality:
     > both ranks must call the same collectives, in the same order, with compatible tensors.

3. **Deterministic meta serialization (make “what we digest” stable)**
   - “Same semantics” isn’t enough if your *encoding* differs: if rank0 serializes kwargs/envelope differently than rank1, an input-digest check becomes noisy (false mismatches) or brittle.
   - Canonicalization rules that matter in practice:
     - stable dict key order (e.g., `sort_keys=True` for JSON),
     - stable separators/whitespace,
     - explicit handling for floats/NaNs, enums, and optional fields,
     - fixed tensor broadcast order + explicit manifest (name/shape/dtype).
   - Bringup evidence: the run log includes a dedicated fix for **deterministic digest JSON** (`8690b8d3`: `Input digest JSON deterministic`).

4. **Env parity for backend selection (FA4 vs flash isn’t “just perf”)**
   - Backend selection can diverge across ranks due to env vars *or* due to “ambient” availability (e.g., CUTLASS present on one rank but not the other).
   - v0 takes this seriously: `scope-drd/notes/FA4/h200/tp/runtime-checklist.md` locks “KV-bias backend must be identical across ranks” and treats this as an env-parity requirement.
   - For bringup/debug: prefer **pinning** (`SCOPE_KV_BIAS_BACKEND=flash` or `fa4`) over `auto`, and **log the effective backend** at startup and per-call.

5. **RNG contracts: broadcast inputs vs lockstep generators**
   - **Broadcast randomness (robust, more traffic):** rank0 samples noise/stochastic tensors and broadcasts them; input digest verifies the bytes. This survives “extra RNG calls” on one rank because the randomness is *data*, not implicit state.
   - **Lockstep RNG (leaner, brittle):** ranks independently generate randomness from the same seed/state and must consume RNG in the exact same sequence. v0 policy is to keep the existing per-chunk `torch.Generator` lockstep (`PrepareLatentsBlock` / `PrepareVideoLatentsBlock`), which works only if control flow is identical.
   - If you choose lockstep RNG, treat “extra `torch.rand(...)` somewhere” as a correctness bug, not a harmless difference.

### Cross-resource agreement / disagreement

Pending deep reading of the low-priority reproducibility resources listed above, the working alignment looks like this:

- **Agreement:** classic sources of nondeterminism (floating non-associativity, algorithm selection/autotuning, reduction order) are real and can explain run-to-run numeric differences.
- **Key difference for TP/PP inference:** the dominant risk isn’t “my last decimal changed,” it’s **deadlock** (collectives out of order) or **silent corruption** (weights/inputs drift). Those failure modes require *distributed integrity checks* (digest/fingerprint) and *control-plane contracts* (call envelopes), which are generally outside typical “reproducibility” guidance.

### Practical checklist

Use this as a “bringup/debug mode” checklist; relax knobs only once you’ve earned confidence.

**A. Invariants to enforce**

- **Monotonic control envelope:** `call_id` is strictly increasing; `chunk_index` is consistent; actions (`INFER/NOOP/SHUTDOWN`) are agreed. Log and assert these per call.
- **Deterministic meta encoding:** any serialized envelope/kwargs used for comparison/digests must be canonical (sorted keys, stable formatting, explicit defaults).
- **Fixed tensor broadcast schema:** explicit tensor manifest + deterministic ordering; avoid “implicit” Python container ordering for which tensors are sent/received.
- **Backend parity:** pin/validate kernel/backend choices across ranks (don’t allow one rank to “auto” into a different backend).
- **RNG contract is explicit:** either broadcast randomness tensors, or lockstep generator state derived from broadcast meta (seed, step index, etc.). Don’t mix.

**A2. Cache lifecycle determinism is not bitwise determinism**

In TP/PP streaming inference, the highest-leverage “determinism” is usually not float-level reproducibility; it’s **KV-cache lifecycle integrity** (reset/recompute/advance). The operator goal is: prevent hangs and Franken-models via explicit contracts + tripwires.

- Make cache lifecycle explicit in the per-call contract: `init_cache`, reset bits, `do_kv_recompute`, `expected_generator_calls`, `cache_epoch`, `current_start_frame`, `kv_cache_attention_bias`, and (ideally) `denoising_step_list`.
- Add a cheap **scalar state digest** every N chunks (N=64/128): all_gather and compare `cache_epoch`, `current_start_frame`, `do_kv_recompute`, `num_denoise_steps`, `kv_cache_attention_bias`.
- Treat backend pinning (e.g., `SCOPE_KV_BIAS_BACKEND`) as a **correctness contract**, not a perf setting.
- Optional “belt + suspenders”: a tiny “KV lifecycle digest” that hashes only scalars + a couple representative cache index values and all_gathers it every 128 chunks.

**B. Input digest (catch envelope/payload mismatches)**

- Enable: `SCOPE_TP_INPUT_DIGEST=1`
- Tune overhead vs coverage:
  - `SCOPE_TP_INPUT_DIGEST_EVERY_N=<int>` (bringup: `1`; later: increase)
  - `SCOPE_TP_INPUT_DIGEST_SAMPLE_BYTES=<int>` (set high when chasing drift)
- What to log (at least on rank0; ideally both ranks on mismatch):
  - `call_id`, `chunk_index`, action, cache/recompute flags
  - digest value(s) per rank
  - canonical meta JSON used for digest (or a stable summary)
  - tensor manifest: names, shapes, dtypes, devices

**C. Shard fingerprint (catch Franken-model / weight mutation)**

- Always record the **startup baseline** fingerprint vector and confirm it’s sane.
  - Bringup evidence: Run 3 logged identical fingerprints on both ranks (bug: `[6137181582272122429, 6137181582272122429]`); later runs confirmed the fix (`ed00094b`) with divergent fingerprints (e.g. `[-505995101758162675, 4154699338113877488]`).
- Add periodic checks when you suspect drift:
  - `SCOPE_TP_SHARD_FINGERPRINT_EVERY_N=128` (example cadence)
  - or `SCOPE_TP_SHARD_FINGERPRINT_ON_INIT_CACHE=1` (check at cache resets)
- What to log:
  - baseline fingerprint vector at init
  - periodic fingerprint vectors with `call_id` / `chunk_index`

**D. Backend pinning / startup report**

- Pin KV-bias backend across ranks: `SCOPE_KV_BIAS_BACKEND=flash|fa4` (avoid `auto` during bringup).
- Log a startup “backend report” on every rank (dist backend, device, KV-bias backend, compile mode).

**E. Hang hygiene (so divergence doesn’t waste 5 minutes)**

- When chasing control-flow divergence, enable the “fail sooner” knobs from the TP notes:
  - `SCOPE_TP_HEARTBEAT_S=1`
  - `SCOPE_TP_WORKER_WATCHDOG_S=30`
- Ensure logs include the last successfully processed `call_id` so you can localize where ranks stopped agreeing.

### Gotchas and failure modes

- **Digest checks can lie if serialization isn’t canonical.** If a digest includes JSON/meta that isn’t deterministically encoded, you’ll get spurious “mismatch” crashes (the run log explicitly calls out a deterministic-JSON fix: `8690b8d3`).
- **Sampling/stride can miss rare mismatches.** If `*_EVERY_N` is large or `*_SAMPLE_BYTES` is small, a one-off divergence can sneak through long enough to be painful. In bringup, pay the overhead and check frequently.
- **Digest ≠ weight integrity.** Input digests catch broadcast envelope/payload mismatches; they do not detect “rank1 weights changed” unless that change is itself part of broadcasted state. That’s why shard fingerprints exist.
- **Fingerprint baselines should be validated.** A baseline that is accidentally identical across ranks (like the Run 3 bug) provides false confidence; confirm fingerprints differ when shards differ.
- **Env parity is more than env vars.** Auto backend selection can depend on installed libs/versions (e.g., CUTLASS availability), so “same env var” doesn’t guarantee “same kernel.” Pin during bringup.
- **Lockstep RNG is brittle by design.** An extra RNG draw, a conditional branch, or a rank-only code path can desynchronize generator state. If you can’t guarantee lockstep, prefer broadcasting randomness tensors for debugging.
- **Tripwires don’t prevent mid-call control-flow divergence.** If ranks diverge *inside* the model forward, the next collective may hang before any digest check runs. Use heartbeat/watchdog + per-call logging to bound the search.

### Experiments to run

1. **Backend mismatch detection (FA4 vs flash)**
   - Force different `SCOPE_KV_BIAS_BACKEND` on rank0 vs rank1 (or keep `auto` but make CUTLASS available on only one rank).
   - Expected: env-parity / backend-report logic fails fast or emits an unmistakable “effective backend differs” signal before inference.

2. **Meta-serialization stability (kwargs key order)**
   - Build the same kwargs/envelope with different insertion order (or reorder keys in `SCOPE_TP_LOAD_PARAMS_JSON`).
   - Expected: with canonical serialization, input digests remain identical; if they differ, the digest machinery is too brittle (repro the class of bug that required “deterministic digest JSON”).

3. **RNG contract break (seed perturbation)**
   - Perturb the per-chunk seed on one rank, or intentionally consume an extra random draw on one rank.
   - Expected:
     - broadcast-noise mode: input digest trips immediately (noise bytes differ),
     - lockstep-generator mode: either the digest trips (if seed/state is part of the envelope) or you observe output drift → update the contract so RNG state is coupled to the broadcast.

4. **Weight mutation (fingerprint catches Franken-model)**
   - Apply an in-place weight tweak on one rank mid-run (or simulate a disallowed runtime LoRA scale update).
   - Enable `SCOPE_TP_SHARD_FINGERPRINT_EVERY_N=1` temporarily.
   - Expected: fingerprint mismatch triggers within a call or two, before you notice quality degradation.

5. **Control-flow divergence → hang (then shorten time-to-diagnosis)**
   - Create a deliberate branch divergence (e.g., force recompute path on one rank only).
   - Expected: a hang near a collective; with `SCOPE_TP_HEARTBEAT_S=1` + `SCOPE_TP_WORKER_WATCHDOG_S=30`, the system fails quickly and logs the last `call_id`, bounding where the divergence occurred.
