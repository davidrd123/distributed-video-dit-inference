## 0) Delta vs the prior 5-pro compile thread (05–07)

### Unchanged

* **The real enemy is graph fragmentation**, not “compile” per se. For TP, any per-collective `torch._dynamo.disable()` pattern explodes into a zillion tiny graphs and tanks throughput (Run 8–9b: ~9.6 FPS).
  Reference: `scope-drd/notes/FA4/h200/tp/bringup-run-log.md` (Run 8–9b).
* **Rank-symmetry is the correctness constraint**: if ranks don’t run the same collective program (same collectives, same order, compatible tensors), you get either an NCCL hang or a Franken-model.
  Reference: `scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md`.
* **Functional collectives are the compiler-friendly route** for TP collectives, and they must be gated to compile-only to avoid eager regressions.
  Reference: `refs/resources/funcol-rfc-93173.md`, `scope-drd/notes/FA4/h200/tp/bringup-run-log.md` (Run 12a vs 12b).

### Refined

* We now have **a working, measured TP=2+compile baseline** (Run 12b: ~24.5 FPS) and can write an operator manual around *protecting that baseline* rather than debating feasibility.
  Reference: `scope-drd/notes/FA4/h200/tp/session-state.md`, run log Run 12b.
* The “what to pin” story is now explicit:

  * **Env parity at init** for anything that changes compile graphs or kernel selection.
  * **In-envelope per call** for anything that changes control flow (cache lifecycle, recompute, plan).
  * **Tripwires** (digest/fingerprint) for drift.
    Reference: `runtime-checklist.md`, `06-failure-modes.md`, `04-determinism-across-ranks.md`.

### Overridden

* The earlier “maybe torch build is missing `_functional_collectives`” path is obsolete for this worktree: **the TP compile unlock already shipped via functional collectives** (Run 12b).
  Reference: `bringup-run-log.md` Run 12.
* The “`Tensor.item()` break is the main culprit” suspicion is explicitly disproven for TP=2: enabling `TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1` removed that warning, but FPS stayed 9.6 until collectives were fixed (Run 9b).
  Reference: `bringup-run-log.md` Run 9b.
* The “KV-cache slicing rewrite” is **not a ship requirement** right now. It is a **fullgraph diagnostic-only** problem and the narrow/SymInt attempt was reverted due to more breaks.
  Reference: `bringup-run-log.md` Known Issue 8.

---

## P0 Ship Checklist

This is the “if we ship TP=2 + compile in v0, what must be true” list. It is intentionally strict.

### P0.1 Safety gates for “compiled distributed regions”

1. **No rank divergence around collectives**

   * All ranks that participate in a process group must execute:

     * same collectives
     * same order
     * compatible tensors (shape, dtype, device)
       Reference: `06-failure-modes.md` invariant.
2. **Compile must not reintroduce collective-induced graph breaks**

   * In TP compiled region: collective ops must be traceable (functional collectives) rather than `dynamo.disable` wrappers.
     Reference: `bringup-run-log.md` Run 8–9b vs Run 12b; `funcol-rfc-93173.md`.
3. **Fail-fast posture**

   * Crashes are preferred to hangs or silent drift:

     * Worker watchdog + heartbeat for orphan detection.
     * Digest/fingerprint for drift detection.
       Reference: `06-failure-modes.md` Q2/Q7, `02-deadlock-patterns.md`.

### P0.2 Operator steps to enable TP+compile safely

* Required env (canonical):

  * `SCOPE_TENSOR_PARALLEL=2`
  * `SCOPE_TP_ALLOW_COMPILE=1`
  * `SCOPE_COMPILE_KREA_PIPELINE=1`
  * KV-bias backend pinned or parity-verified (`SCOPE_KV_BIAS_BACKEND=fa4|flash|auto` but see parity section)
* Required startup behavior:

  * **Lockstep warmup** enabled when compile is enabled (`SCOPE_TP_LOCKSTEP_WARMUP=auto` behavior).
    Warmup must be driven through the broadcast path in lockstep.
    Reference: `bringup-run-log.md` Known Issue 2, `session-state.md`.
* Required regression invariants (daily):

  * `scripts/tp_compile_repro.py` mode C: `graph_break_count=0`, `graph_count=1`.
    Reference: `research-program.md`.

### P0.3 “Stop the line” conditions

If any of these happen, do not try to “power through”:

* **Any hang** (GPU util drops to ~0, processes alive, last log near collective) → treat as rank divergence until proven otherwise.
  Reference: `06-failure-modes.md`.
* **Any silent quality drift** with no crash → treat as Franken-model risk, turn on digest + fingerprint and repro.
  Reference: `06-failure-modes.md` Q2.
* **Graph-break count increase** in the compile repro or E2E harness beyond baseline → treat as perf regression and investigate before merging.
  Reference: `dynamo-deep-dive.md`, `research-program.md`, run log.

---

## 1) Rank-parity invariants

This is the core operator manual. It tells you what must match across ranks, and where to enforce it.

### Parity invariants table

| Category                                      | Must be env-parity-checked at init                                                                                                                   | Must be in-envelope each call                                                          | Best-effort later                             |
| --------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | --------------------------------------------- |
| Topology                                      | `WORLD_SIZE==2` (TP v0 hard requirement), `SCOPE_TENSOR_PARALLEL=2`, rank roles (rank0 server, rank>0 worker)                                        | `TPAction` (`INFER/NOOP/SHUTDOWN`)                                                     | Multi-node topology (future)                  |
| Pipeline identity and weights                 | Pipeline ID frozen for job lifetime; reload disabled (HTTP 409 for mismatch)                                                                         | `control_epoch`/`cache_epoch` (or equivalent) if present                               | Periodic shard fingerprint (optional cadence) |
| Compile mode                                  | `SCOPE_TP_ALLOW_COMPILE`, `SCOPE_COMPILE_KREA_PIPELINE`, `SCOPE_TORCH_COMPILE_FULLGRAPH` (if used), any compile backend knobs you actually branch on | Lockstep warmup calls are “planned” calls (see warmup protocol)                        | AOTInductor cache strategy                    |
| Dynamo behavior knobs                         | `TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS` (if used), any `torch.compiler.config.*` you set explicitly                                                     | None (do not change at runtime)                                                        | Logging verbosity (`TORCH_LOGS`)              |
| Kernel/backend selection                      | `SCOPE_KV_BIAS_BACKEND` effective backend must match across ranks (pin during bringup), any attention backend flags that affect code path            | Per-call report fields can be logged (backend name) but do not branch on per-call      | Auto-selection once stable                    |
| Optional features that change graph structure | Any feature flags that add/remove ops: qkv pack flags, FA4 enable, VAE/text-enc compile toggles, etc                                                 | Recompute decisions, cache init flags, denoise schedule, expected generator call count | Adaptive policies                             |
| Control plane and ordering                    | One-call-at-a-time serialization on rank0 (no interleaving)                                                                                          | `call_id`, `chunk_index`, plan fields (see below)                                      | Async pipelining (future)                     |
| Drift tripwires                               | Enforce parity for enabling digest/fingerprint checks so they don’t diverge                                                                          | Input digest fields for this call (hash is computed per rank, compared via collective) | Sampling cadence knobs                        |

### What is already implemented today vs proposed

Already implemented (based on the selected notes):

* **Pipeline frozen for job lifetime** (no reload) and pipeline-ID mismatch returns 409.
  Reference: `session-state.md`, `06-failure-modes.md` Q1.
* **Env parity enforcement exists for TP debug/digest/fingerprint knobs** so ranks do not accidentally diverge on safety features.
  Reference: `06-failure-modes.md` (env parity mention), `runtime-checklist.md`.
* **KV-bias backend parity is a locked v0 decision**.
  Reference: `runtime-checklist.md`.
* **Compile-aware collectives**: eager uses in-place `dist.all_reduce`, compile uses functional collectives.
  Reference: `runtime-checklist.md`, `bringup-run-log.md` Run 12b.
* **Lockstep warmup** exists and is tied to compile enablement (auto). Warmup uses `chunk_index=-1` to exclude digest checks.
  Reference: `bringup-run-log.md` Known Issue 2.
* **Fail-fast hang hygiene**: heartbeat + watchdog exist as optional knobs.
  Reference: `06-failure-modes.md` Q7.

Proposed additions (docs-level requirements for ship readiness):

* **Per-call “plan” fields** in-envelope that make call-count and optional phases explicit, even if you think they’re implied:

  * `expected_generator_calls`
  * `do_recompute` (or recompute enum)
  * `init_cache`
  * denoise schedule identifier
    Why: this turns “implicit control flow” into “explicit contract,” preventing deadlocks.
    Reference: `02-deadlock-patterns.md` plan discipline; `06-failure-modes.md` invariant.
* **Parity assertion on “effective backend”**, not only the requested backend. Example: `SCOPE_KV_BIAS_BACKEND=auto` should log and assert that the selected backend name matches across ranks.
  Reference: `04-determinism-across-ranks.md` backend parity discussion.
* **Rank-symmetric compile health reporting**:

  * gather per-rank `graph_breaks` and `unique_graphs` counters at the end of warmup and at steady state.
  * crash if they differ across ranks when TP compile is enabled.
    Reference: `dynamo-deep-dive.md` (graph breaks, guards), `ezyang-state-of-compile.md` (rank-divergent compilation risk).

---

## 2) Graph breaks and guard churn triage

You want a fast flow that tells you whether you are dealing with:

* graph breaks (fragmentation)
* recompiles (guard churn)
* or distributed divergence (hang risk)

### Triage flow

1. **First question: is it hung or slow?**

   * Hung: GPUs go idle, no progress, last log near a collective or broadcast → treat as collective ordering mismatch until proven otherwise.
     Reference: `06-failure-modes.md`, `02-deadlock-patterns.md`.
   * Slow: system progresses but FPS regressed → continue.

2. **Measure graph-break health**

   * Enable: `TORCH_LOGS=graph_breaks`
   * Run:

     * `uv run torchrun --nproc_per_node=2 scripts/tp_compile_repro.py`
   * Interpret:

     * If breaks > 0 in the micro-repro mode that is supposed to be “0 breaks,” that is a regression (likely a stray `torch.compiler.disable` or Python side effect inside the compiled region).
       Reference: `research-program.md`.
     * If breaks increased in E2E but not in micro-repro, the regression is in non-TP blocks or glue code. Decide whether that matters for ship.

3. **Differentiate breaks vs recompiles**

   * If break count is stable but perf is jittery or progressively worse, check guard churn:

     * Enable: `TORCH_LOGS=recompiles,guards`
   * Signs of recompiles:

     * repeated “recompile” logs
     * `unique_graphs` grows without bound during a stable workload
       Reference: `dynamo-deep-dive.md` (guards/recompiles), `ezyang-state-of-compile.md` (static by default, specialization).

4. **Assess acceptable steady state**

   * Hard requirement for shipping TP compile:

     * **No collective-induced graph breaks** inside the TP compiled region. Your TP compile repro “Mode C” is the canary.
       Reference: `research-program.md`, `bringup-run-log.md` Run 12b.
   * Acceptable in v0 (current reality):

     * A small, stable number of graph breaks elsewhere (example noted: KV-cache dynamic slicing causes `fullgraph=True` failure and some breaks under `fullgraph=False`). This is acceptable only if it is:

       * stable
       * symmetric across ranks (same break sites)
       * not straddling collectives
         Reference: `bringup-run-log.md` Known Issue 8, `session-state.md` (steady state breaks).
   * Unacceptable:

     * Breaks that appear or disappear depending on rank, input, or timing. That is a hang seed.

5. **When to stop “tuning” and change structure**

   * If you see:

     * breaks caused by logging, Python control flow, `.tolist()`, debug flags, or disable wrappers in hot compiled code
       then tuning kernels is a waste. Remove the break source.
       Reference: `dynamo-deep-dive.md` (graph breaks as perf footgun).

---

## 3) Compiled distributed region contract

This is the code review contract for anything that will run under TP+compile.

### Do rules

* **Do treat compiled distributed code as SPMD inside its group**

  * same inputs (modulo sharding)
  * same non-tensor control decisions
  * same collectives, same order
    Reference: `06-failure-modes.md` invariant, `ezyang-state-of-compile.md` rank divergence warning.
* **Do gate functional collectives to compile mode**

  * eager: in-place `dist.all_reduce` (baseline)
  * compiled: functional collectives, consume returned tensors, respect wait semantics
    Reference: `runtime-checklist.md`, `bringup-run-log.md` Run 12a vs 12b, `funcol-rfc-93173.md`.
* **Do keep the compiled region tensor-pure**

  * no logging
  * no prints
  * no host callbacks
  * no passing `ProcessGroup` or `Work` objects through traced code
    Reference: `funcol-rfc-93173.md` (non-tensor objects pollute IR), `dynamo-deep-dive.md` (side effects cause breaks).
* **Do make optional phases explicit in the per-call plan**

  * If something can change call-count or branches (recompute, cache reset, schedule), put it in the envelope and assert it.
    Reference: `02-deadlock-patterns.md` (plan discipline), `06-failure-modes.md` Q4/Q6.
* **Do validate before you commit peers to blocking**

  * Any preflight that can throw must happen before sending an INFER header or entering a broadcast that will strand the peer.
    Reference: `02-deadlock-patterns.md` (anti-stranding), `06-failure-modes.md` Q7.

### Don’t rules

* **Don’t introduce conditional collectives**

  * “if debug: all_gather” is a deadlock trap unless debug is parity-checked and identical across all ranks.
  * Preferred: always run the collective, possibly NOOP the payload, or enforce parity + plan.
* **Don’t put `torch.compiler.disable()` or `torch._dynamo.disable()` inside hot TP paths**

  * That is how you get the Run 8–9b cliff (graph break per collective).
    Reference: `bringup-run-log.md` Run 8–9b.
* **Don’t rely on dynamic shape changes or data-dependent branches inside compiled distributed regions**

  * Guard failures and recompiles are rank divergence seeds.
    Reference: `dynamo-deep-dive.md` (guards), `ezyang-state-of-compile.md` (specialization, rank divergence).
* **Don’t mutate model weights at runtime**

  * LoRA runtime updates and pipeline reload are Franken-model risk unless they are explicitly synchronized and validated across ranks. v0 forbids reload and runtime LoRA scale changes.
    Reference: `06-failure-modes.md` Q1/Q3, `runtime-checklist.md`.

### Functional collective boundary rules

* **Where `wait` belongs**

  * Follow the “wait-before-storage-access” semantics from the RFC: treat functional collective outputs as async values and only access storage after wait (or rely on the AsyncTensor auto-wait behavior where applicable).
    Reference: `funcol-rfc-93173.md` (explicit wait semantics).
* **Never mix in-place and functional semantics casually**

  * If a tensor is used after a functional all-reduce, ensure downstream uses consume the returned tensor, not the pre-reduce one. This is a common correctness footgun when refactoring.

---

## 4) Warmup protocol and CUDA-graph caveats

### Warmup protocol

Goal: eliminate startup divergence and make first “real” inference hit steady-state compiled graphs.

**Warmup must:**

1. Run through the same TP broadcast path as real inference (rank0 drives, workers follow).
2. Use fixed shapes and flags matching the production mode you intend to serve.
3. Fill the same caches (KV cache) and trigger compilation of the same regions.

**Implementation policy (current behavior)**

* Use `SCOPE_TP_LOCKSTEP_WARMUP` (auto-enabled when TP compile is enabled).
* Warmup calls use `chunk_index=-1` so they are excluded from input digest checks, but still run in lockstep.
  Reference: `bringup-run-log.md` Known Issue 2.

**Recommended operator checklist for warmup**

* Before warmup:

  * log the “effective backend report” on every rank (kv-bias backend, attention backend, compile flags)
  * record shard fingerprint baseline
* During warmup:

  * keep `SCOPE_TP_HEARTBEAT_S` off (optional) if it pollutes logs, but keep watchdog off unless debugging hangs
  * ensure warmup count is fixed (`SCOPE_TP_LOCKSTEP_WARMUP_RUNS`)
* After warmup:

  * run a one-time “compiled health sync”:

    * gather per-rank `graph_breaks` and `unique_graphs` counters
    * assert they match across ranks in TP compile mode
  * optionally enable input digest every N calls once serving starts

### CUDA graphs caveats

This is explicitly a separate milestone, but the operator manual needs the “don’t accidentally do something dumb” list.

From `refs/topics/06-cuda-graphs.md` (selected slice), the practical policy is:

* **Do not try to capture collectives early**

  * CUDA graph capture + NCCL multiplies failure blast radius. A failed capture can strand peers.
    Reference: `06-cuda-graphs.md` guidance.
* **Freeze shapes first**

  * If shapes vary, you recapture or fail, and recapture in distributed is a deadlock seed.
    Reference: `06-cuda-graphs.md`.
* **Eliminate dynamic allocations inside the capture region**

  * Graph replay assumes pointer stability.
    Reference: `06-cuda-graphs.md`.
* **Start with a capture unit that can be static**

  * One stage forward, not the whole streaming loop.
    Reference: `06-cuda-graphs.md`.
* **Treat multi-stream overlap as a capture contract**

  * If you capture work across streams, the fork/join must be explicit.
    Reference: `06-cuda-graphs.md`.

Staged experiments (recommended, not required for ship):

1. Single-GPU capture of a fixed-shape non-collective region.
2. Multi-rank run where captured region contains no collectives.
3. Only then consider any collective-in-graph experiments, and only with explicit hang hygiene (watchdog) and strict parity.

---

## 5) Minimal daily regression suite

Target: 6–10 short checks that catch the failure classes early: parity bugs, graph breaks, recompiles, deadlocks.

I’m listing each test with “expected failure signature” so it’s actionable.

### Parity tests

1. **TP control plane smoke**

   * Command:
     `SCOPE_TENSOR_PARALLEL=2 uv run torchrun --nproc_per_node=2 scripts/tp_smoke_control_plane.py`
   * Expected: completes quickly; no hangs.
   * Failure signature: worker stuck in `recv_next()` or mismatched action handling.
     Reference: `research-program.md`.

2. **Env parity mismatch fails fast**

   * Intentionally set a TP-critical env var differently on rank1 (example: kv-bias backend or compile enable).
   * Expected: job crashes at init with a clear “env parity mismatch” error, not an NCCL hang.
     Reference: `runtime-checklist.md` (env parity), `02-deadlock-patterns.md` (crash > hang).

### Graph-break tests

3. **Compile repro break gate**

   * Command:
     `uv run torchrun --nproc_per_node=2 scripts/tp_compile_repro.py`
   * Expected: Mode C `graph_break_count=0`, `graph_count=1`.
   * Failure signature: nonzero breaks, or graph count explosion. Treat as regression.
     Reference: `research-program.md`.

4. **Fullgraph first-break census diagnostic**

   * Enable: `SCOPE_TORCH_COMPILE_FULLGRAPH=1`
   * Expected: known failure at KV-cache dynamic slicing (today). This is diagnostic-only.
   * Failure signature: failure moves earlier (new disable wrapper or logging introduced), or failure becomes rank-divergent.
     Reference: `research-program.md`, `bringup-run-log.md` Known Issue 8.

### Recompile and guard-churn tests

5. **Guard stability test**

   * Run the same fixed-shape inference call repeatedly (micro-repro or harness) with `TORCH_LOGS=recompiles,guards`.
   * Expected: no recompiles after warmup.
   * Failure signature: repeated recompiles under constant shapes, or recompiles differ across ranks.
     Reference: `dynamo-deep-dive.md` (guards/recompiles), `ezyang-state-of-compile.md`.

6. **One-dimension variation test**

   * Vary one “should-be-static” dimension (sequence length or resolution) once, identically on all ranks.
   * Expected: at most one synchronized recompile, then stable.
   * Failure signature: only one rank recompiles, or graph break sites differ across ranks.
     Reference: `dynamo-deep-dive.md`, `ezyang-state-of-compile.md`.

### Deadlock and hang hygiene tests

7. **Worker orphan test**

   * Enable: `SCOPE_TP_HEARTBEAT_S=1`, `SCOPE_TP_WORKER_WATCHDOG_S=30`
   * Kill rank0 abruptly during idle or between calls.
   * Expected: worker exits within watchdog window (crash > hang).
     Reference: `06-failure-modes.md` Q7.

8. **Anti-stranding preflight test**

   * Inject a non-picklable object into the broadcast meta payload (or otherwise force preflight failure).
   * Expected: rank0 fails before sending a header that commits workers to blocking, workers do not hang in recv.
     Reference: `02-deadlock-patterns.md` (anti-stranding).

### Drift detection tests

9. **Input digest mismatch test**

   * Enable: `SCOPE_TP_INPUT_DIGEST=1`, set sampling to every call.
   * Perturb a broadcast tensor on rank0 only (test harness).
   * Expected: crash with a clear digest mismatch diff, not a hang and not silent drift.
     Reference: `06-failure-modes.md` Q2.

10. **Shard fingerprint mutation test**

* Enable periodic fingerprint checks (example: `SCOPE_TP_SHARD_FINGERPRINT_EVERY_N=1` temporarily).
* Mutate a weight shard on one rank (test-only injection).
* Expected: crash on fingerprint mismatch (Franken-model tripwire).
  Reference: `06-failure-modes.md` Q2, `04-determinism-across-ranks.md`.

---

## Copy-paste snippets for operator manuals

If you want a short “contract blurb” to paste into `refs/topics/12-compile-distributed-interaction.md`:

> Compiled distributed code must be treated as SPMD inside its process group. Any rank divergence in graph breaks, guard-driven recompiles, or conditional collective behavior is a deadlock seed. Therefore: (1) parity-check all compile and backend selection knobs at init, (2) broadcast a per-call plan that fixes call-count and optional phases, (3) use functional collectives inside compiled regions and never wrap collectives in dynamo-disable, (4) warm up in lockstep before serving, and (5) keep crash > hang with watchdogs and drift tripwires.

If you want a short “triage blurb” to paste into `refs/topics/09-dynamo-tracing.md`:

> If perf regresses under compile, start with `TORCH_LOGS=graph_breaks`. Break count increases are guilty until proven innocent. If break count is stable but perf jitters, move to `TORCH_LOGS=recompiles,guards` and look for guard churn. In distributed mode, any rank-asymmetric breaks or recompiles must be treated as correctness bugs, not perf bugs.

If you want a short “funcol blurb” to paste into `refs/topics/11-functional-collectives.md`:

> Functional collectives exist because classic c10d collectives are in-place and return non-tensors, which breaks functional tracing. Under compile, use functional all-reduce and consume returned tensors; in eager, use in-place collectives for performance. Never add a `dynamo.disable` wrapper around collectives inside a compiled TP region.
