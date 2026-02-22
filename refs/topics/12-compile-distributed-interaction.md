---
status: draft
---

# Topic 12: Compile + distributed interaction — compile with DDP/FSDP/TP

The core challenge is that **DDP/FSDP use backward hooks for communication**, and these hooks create graph breaks in AOTAutograd. The solutions are: (1) **Compiled Autograd** (PyTorch 2.4+), which captures the full backward graph at runtime, (2) **FSDP2 built on DTensor**, which is compile-friendly by design, and (3) **functional collectives** for tensor parallelism.

## Resources

<!-- Resource IDs from manifest belonging to this topic -->
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| ezyang-state-of-compile | State of torch.compile for training (August 2025) | high | condensed |
| compiled-autograd-tutorial | Compiled Autograd Tutorial | medium | pending |
| ezyang-ways-to-compile | Ways to use torch.compile | medium | pending |
| torch-compiler-faq-dist | torch.compiler FAQ: Distributed Section | medium | pending |
| tp-tutorial | Large Scale Transformer Training with Tensor Parallel | medium | pending |
| vllm-torch-compile | Introduction to torch.compile and How It Works with vLLM | medium | pending |

## Implementation context

Compile interacted catastrophically with distributed collectives until Run 12b: `torch._dynamo.disable()` on each all-reduce caused ~160 graph breaks per forward and TP=2 throughput dropped from 16 FPS (Run 7) to **9.6 FPS** (Runs 8-9b). Replacing those with functional collectives made collectives traceable and restored end-to-end performance to **24.5 FPS** (Run 12b). Current steady state in the E2E harness is `unique_graphs=12, graph_breaks=2` after removing attention-backend logger breaks (Run 14).

See: `refs/implementation-context.md` → Phase 1, `scope-drd/notes/FA4/h200/tp/bringup-run-log.md` (Runs 7-14).

Relevant Scope code:
- `scope-drd/src/scope/core/tensor_parallel/linear.py` and `scope-drd/src/scope/core/tensor_parallel/rmsnorm.py` (compile-aware collective dispatch)
- `scope-drd/scripts/tp_compile_repro.py` (minimal repro + invariants that protect the baseline)

## Synthesis

<!-- To be filled during study -->

### Mental model

Distributed + `torch.compile` is hard because **compilation happens per process, but correctness is cross-process**.

- **Each rank compiles independently**: Dynamo traces what it sees on that rank (inputs, branches taken, shapes observed), emits one or more graphs, and installs guards. If anything about execution differs, different ranks can legitimately produce different graphs and/or hit different graph breaks.
- **Distributed expects rank symmetry**: whenever ranks participate in a collective, they must execute the same logical collective program (same ops, same order, compatible tensor metadata). If ranks diverge, the common failure mode is not “a bit slower” — it’s **hangs / NCCL timeouts**.

So the meta-rule is:

> In distributed, you don’t just need a fast compiled graph. You need *the same compiled behavior* on every participating rank.

Two compiler-compatible routes show up across the sources:

1. **DTensor / SPMD route (high-level)**: write a single global program on sharded tensors; compilation lowers placements into collectives and can optimize away eager DTensor overhead. This is the direction of “compiler-driven parallelism” (SimpleFSDP / AutoParallel).
2. **Functional collective route (low-level)**: keep your program explicit, but make collectives traceable as pure tensor ops (functional collectives) so they can live inside compiled graphs without forcing graph breaks around side-effecting `Work` objects.

In **Scope’s TP inference**, we took the second route: `torch._dynamo.disable()` around each all-reduce created ~160 graph breaks/forward and collapsed TP=2+compile to ~9.6 FPS; switching to functional collectives inside compiled regions restored end-to-end to **~24.5 FPS** (Run 12b). The remaining contract is still rank symmetry: even with funcol, ranks must see the same shapes/branches and execute collectives in the same order.

Finally, our PP design introduces an important nuance: **rank0-out-of-mesh** means rank0 is intentionally *not* symmetric with mesh ranks. That’s fine, but it means “rank symmetry” is a **per-process-group** contract (e.g., within `mesh_pg`), not a “the whole world does the same thing” contract.

### Key concepts

- **Rank-symmetry contract**: for all ranks in a given collective group, ensure the same collective sequence/order and compatible tensor metadata (shape/dtype), and avoid rank-dependent branches inside compiled regions.
- **Effective backend parity (not just requested backend parity)**: if a setting can resolve dynamically (e.g., `SCOPE_KV_BIAS_BACKEND=auto`), treat the *resolved* backend choice as part of the distributed contract: log it on every rank and assert it matches across ranks at startup (and ideally whenever it can change). “Auto” must never silently resolve to different kernels across ranks.
- **Graph breaks vs recompiles (Dynamo mechanics)**:
  - A **graph break** splits execution into multiple compiled segments with eager Python in between.
  - A **recompile** happens when a **guard fails** (e.g., a shape/value changes) and Dynamo emits a new specialization.
  - Either can cause *rank divergence* if different ranks break/recompile differently.
- **“Static by default” specialization**: Dynamo specializes on the first shapes/values it sees; dynamic shapes are an explicit decision (and a common source of guard churn).
- **Functional collectives (funcol)**: non-mutating collective ops that return tensors (often async) plus an explicit wait boundary; designed so collectives can be traced as Tensor ops rather than `ProcessGroup`/`Work` side effects.
- **DTensor**: a global tensor abstraction over a device mesh; placements (including `Partial`) represent distributed semantics; compilation lowers DTensor programs into explicit collectives and can remove eager DTensor overhead.
- **DDP/FSDP complication (training-oriented)**: communication is often triggered by backward hooks and autograd graph structure; “compiled autograd” and DTensor-native FSDP paths exist to make this more compile-friendly.
- **Regional compilation**: compile only the stable/hot region (e.g., one block or one stage) to reduce compile time and reduce the surface area for rank-divergent behavior.
- **`world_pg` vs `mesh_pg`**: Scope’s split isolates “who must be symmetric with whom”; compile decisions must respect group boundaries (rank0-out-of-mesh should not participate in `mesh_pg` collectives).

### Cross-resource agreement / disagreement

- **Agreement: distributed + compile fails via divergence**:
  - `ezyang-state-of-compile` highlights rank-divergent compilation as the core distributed pitfall (hangs/timeouts).
  - `dynamo-deep-dive` explains why divergence happens: tracing depends on runtime values; guards/recompiles and graph breaks are normal in the model.
- **Agreement: two “compile-compatible” routes**:
  - `ezyang-state-of-compile` frames DTensor and functional collectives as the two main roads: DTensor for high-level SPMD programs; funcol for explicit, compiler-friendly collectives.
  - `funcol-rfc-93173` makes the low-level argument concrete: in-place collectives + `Work` objects don’t fit a functional IR; you need Tensor-returning ops + explicit waits.
- **Difference: abstraction vs control**:
  - DTensor aims for “write global code, compiler inserts collectives and (eventually) schedules them well”.
  - Funcol aims for “write explicit collectives, but make them traceable and optimizable”.
  - In our TP bringup, funcol was the more predictable surface; DTensor/AutoParallel is promising but higher-variance and more sensitive to operator coverage/dynamism.
- **Nuance: overlap is not solved by funcol**:
  - The funcol RFC explicitly does *not* introduce multi-stream semantics in Inductor; overlap remains a separate stream/event design problem (relevant to PP).

### Practical checklist

> **Compiled distributed region contract**: Compiled distributed code must be treated as SPMD inside its process group. Any rank divergence in graph breaks, guard-driven recompiles, or conditional collective behavior is a deadlock seed. Therefore: (1) parity-check all compile and backend selection knobs at init, (2) broadcast a per-call plan that fixes call-count and optional phases, (3) use functional collectives inside compiled regions and never wrap collectives in dynamo-disable, (4) warm up in lockstep before serving, and (5) keep crash > hang with watchdogs and drift tripwires.

- **Define symmetry domains up front** (who must run the same compiled behavior?):
  - TP inside `mesh_pg`: all mesh ranks must be symmetric.
  - rank0-out-of-mesh: rank0 may diverge by design, but must not enter mesh collectives.
- **Lock down shapes and flags across symmetric ranks**: same input shapes, same optional feature flags, same code paths.
- **Track graph-break and recompile health**: monitor `TORCH_LOGS=graph_breaks,recompiles,guards` and keep a baseline regression gate (Scope steady state after Run 14: `unique_graphs≈12`, `graph_breaks=2`).
- **Keep compiled distributed regions “tensor-pure”**: avoid Python side effects/logging, and avoid runtime-varying non-tensor objects in the traced surface.
- **Use functional collectives under compile, not in eager**: gate funcol usage on compile mode; eager-mode funcol can regress due to extra allocations/wrapping (Run 12a).
- **Prefer regional compilation**: compile the per-block/per-stage forward where shapes are stable; keep control-plane and highly dynamic logic eager.
- **Establish a deterministic warmup**: run a fixed-shape warmup phase on all symmetric ranks before measuring/serving so compilation and caches are initialized in a known-safe regime.
- **Failure-mode hygiene**: use conservative timeouts during bringup and ensure “all ranks fail together” behavior (to avoid stranding peers at collectives).

#### P0 ship checklist (TP=2 + compile)

*Distilled from `deep-research/2026-02-22/compile-distributed-hardening/reply.md` §P0 Ship Checklist. Source citations: `scope-drd/notes/FA4/h200/tp/bringup-run-log.md` Runs 8–12b, `scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md`, `scope-drd/notes/FA4/h200/tp/research-program.md`.*

These are “stop the line” constraints if we want to keep the Run-12b baseline stable.

- **No rank divergence around collectives**: inside a collective group, ranks must execute the same collectives, in the same order, with compatible tensors (shape/dtype/device). If not, the expected outcomes are NCCL hang or Franken-model.
- **No collective-induced graph fragmentation**: the TP compiled region must not wrap collectives in `torch._dynamo.disable()` / `torch.compiler.disable()`; that was the Run 8–9b cliff (~160 breaks/forward).
- **Crash > hang, drift must trip**: keep watchdog/heartbeat available for orphaned workers; keep input digest + shard fingerprints available to catch envelope/weight drift.
- **Warmup in lockstep** when compile is enabled: warmup must run through the same TP broadcast path as “real” inference, with production shapes/flags.
- **Regression canary stays green**: `scope-drd/scripts/tp_compile_repro.py` Mode C invariants (no breaks / one graph) are treated as a daily gate.

#### Rank-parity invariants (what to pin, where)

*Distilled from `deep-research/2026-02-22/compile-distributed-hardening/reply.md` §1 Rank-parity invariants. Source citations: `scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md`, `scope-drd/notes/FA4/h200/tp/runtime-checklist.md`, `scope-drd/notes/FA4/h200/tp/bringup-run-log.md`.*

In distributed compile, compilation happens per-process but correctness is cross-process. The operator rule is: **pin anything that changes graphs or kernel selection at init, and carry anything that changes per-call control flow in the envelope**.

| Category | Enforce at init (env parity / startup assert) | Enforce per call (envelope / plan) | Notes |
|---|---|---|---|
| Topology | `WORLD_SIZE`, `SCOPE_TENSOR_PARALLEL`, role/plan (`SCOPE_TP_PLAN`, `SCOPE_PP_ENABLED`) | `action` (`INFER/NOOP/SHUTDOWN`) | “Wrong group membership” is instant deadlock risk. |
| Pipeline identity + weights | pipeline id frozen; reload disabled for TP v0 | `cache_epoch` / `control_epoch` (if used) | Periodic shard fingerprint is the drift backstop. |
| Compile mode | `SCOPE_TP_ALLOW_COMPILE`, `SCOPE_COMPILE_KREA_PIPELINE`, `SCOPE_TORCH_COMPILE_FULLGRAPH` (if used) | warmup calls treated as planned | Do not let some ranks compile while others don’t. |
| Kernel/backend selection | `SCOPE_KV_BIAS_BACKEND` (pin during bringup); any attention backend flags | (log only) | If you allow `auto`, assert the **effective** backend matches across ranks. |
| Optional phases that change call graph | feature flags that add/remove ops | `do_kv_recompute`, `num_denoise_steps`, `expected_generator_calls`, `init_cache` | Anything that changes call-count must be explicit. |
| Drift tripwires | parity-check enabling digest/fingerprint so ranks don’t diverge on safety | per-call digests (bringup) | Sampling cadence can be tuned later. |

#### Triage flow (graph breaks vs recompiles vs divergence)

*Distilled from `deep-research/2026-02-22/compile-distributed-hardening/reply.md` §2 Graph breaks and guard churn triage. Source citations: `refs/resources/dynamo-deep-dive.md`, `refs/resources/ezyang-state-of-compile.md`, `scope-drd/notes/FA4/h200/tp/research-program.md`.*

1. If it’s **hung** (GPUs idle, last log near a collective/broadcast): treat as collective ordering mismatch until proven otherwise.
2. If it’s **slow**, start with `TORCH_LOGS=graph_breaks`:
   - If breaks appear in `tp_compile_repro.py` “0-break mode”, treat as regression (often a stray disable wrapper or a Python side effect inside the compiled region).
3. If break count is stable but perf jitters or `unique_graphs` grows: switch to `TORCH_LOGS=recompiles,guards` to find guard churn.
4. In distributed mode, **rank-asymmetric breaks or recompiles are correctness bugs**, not “just perf.” Fix parity before tuning kernels.

#### Code-review contract: compiled distributed region

*Distilled from `deep-research/2026-02-22/compile-distributed-hardening/reply.md` §3 Compiled distributed region contract. Source citations: `scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md`, `refs/resources/funcol-rfc-93173.md`, `scope-drd/notes/FA4/h200/tp/explainers/02-deadlock-patterns.md`.*

Do:
- Treat compiled distributed code as SPMD inside its process group (same inputs/branches/collectives).
- Gate functional collectives to compile mode; eager should use in-place c10d collectives for performance.
- Keep the compiled region tensor-pure: no prints/logging/host callbacks; don’t thread `ProcessGroup` / `Work` objects through traced code.
- Make optional phases explicit in the per-call plan and assert `observed_generator_calls == expected_generator_calls`.
- Validate before committing peers to blocking (anti-stranding: preflight before sending `INFER` header / entering broadcast loops).

Don’t:
- Add conditional collectives behind per-rank flags unless the flag is parity-checked and treated as part of the plan.
- Wrap TP collectives in `torch._dynamo.disable()` / `torch.compiler.disable()` inside compiled hot paths.
- Rely on rank-dependent dynamic shapes or data-dependent branches inside compiled distributed regions.
- Introduce runtime weight mutation (runtime LoRA scale updates, reload) without explicit cross-rank synchronization; TP v0’s posture is “forbid.”

#### Warmup protocol (operator checklist)

*Distilled from `deep-research/2026-02-22/compile-distributed-hardening/reply.md` §4 Warmup protocol. Source citations: `scope-drd/notes/FA4/h200/tp/bringup-run-log.md` Known Issue 2, `scope-drd/notes/FA4/h200/tp/session-state.md`.*

Warmup exists to eliminate startup divergence and make the first “real” inference hit steady-state compiled graphs.

- Warmup must run through the same TP broadcast path as inference (rank0 drives; workers follow).
- Warmup must use fixed shapes/flags matching the production mode you intend to serve.
- Warmup should be excluded from input-digest checks (Scope uses `chunk_index=-1`) but still must run in lockstep.
- After warmup, perform a **rank-symmetric compile health gather**: all-gather per-rank `graph_breaks` / `unique_graphs` (or equivalent counters) and crash if they differ across TP ranks when compile is enabled. This turns “rank diverged during compile” into a fast error instead of a later NCCL hang.

#### Minimal daily regression suite (break-it tests)

*Distilled from `deep-research/2026-02-22/compile-distributed-hardening/reply.md` §5 Minimal daily regression suite. Source citations: `scope-drd/notes/FA4/h200/tp/research-program.md`, `scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md`, `refs/resources/dynamo-deep-dive.md`.*

These catch parity bugs, graph fragmentation, guard churn, hangs, and drift early.

1. TP control-plane smoke (no hangs): `SCOPE_TENSOR_PARALLEL=2 uv run torchrun --nproc_per_node=2 scripts/tp_smoke_control_plane.py`
2. Env parity mismatch fails fast (not hang): intentionally mismatch a TP-critical env var on rank1.
3. Compile repro break gate: `uv run torchrun --nproc_per_node=2 scripts/tp_compile_repro.py` (Mode C “0 breaks, 1 graph”).
4. Fullgraph “first break census” (diagnostic-only): `SCOPE_TORCH_COMPILE_FULLGRAPH=1` should fail at known KV slicing site; any earlier failure is regression.
5. Guard stability: `TORCH_LOGS=recompiles,guards` on repeated fixed-shape runs; expect no recompiles after warmup.
6. One-dimension variation (synchronized): vary one “should-be-static” dimension identically across ranks; expect at most one synchronized recompile.
7. Worker orphan test: enable heartbeat + watchdog; kill rank0; worker should exit within watchdog window (crash > hang).
8. Anti-stranding preflight test: inject non-picklable meta; sender must fail pre-header; receiver must not hang.
9. Input digest mismatch test: perturb broadcast tensor on rank0 only; expect crisp digest mismatch crash.
10. Shard fingerprint mutation test: mutate weight shard on one rank (test-only); expect fingerprint mismatch crash.

### Gotchas and failure modes

- **Rank-divergent graph breaks**: one rank hits an unsupported op (or a stray debug print) and breaks while another stays in-graph → different collective ordering → hang.
- **Rank-divergent recompiles**: one rank sees a new shape/value and recompiles while others don’t; the next collective boundary can deadlock.
- **The “dynamo.disable around collectives” trap**: disabling Dynamo at collective call sites fragments the forward into many tiny graphs and destroys performance (exactly what caused the Run 8–9b collapse).
- **Funcol eager-mode regression**: functional collectives can be slower than in-place c10d in eager due to extra allocations/wrapping; don’t “simplify” code by using funcol everywhere.
- **Autograd support mismatches**: funcol surfaces may have limitations for backward/autograd; training stacks may need DTensor/compiled-autograd paths instead.
- **Group boundary confusion**: using the wrong process group (or letting rank0 accidentally participate in a mesh collective) breaks the symmetry model and can deadlock.
- **Silent numeric drift under compile**: compile isn’t bitwise equivalent to eager; have a debug path that isolates tracing vs backend lowering when chasing correctness.
- **Cold-start compile cost**: compilation is a blocking first-call cost; for serving you need a warmup/AOT/caching story or you risk SLO violations.

### Experiments to run

- **Reproduce the “TP compile cliff” in isolation**: run `scope-drd/scripts/tp_compile_repro.py`-style microbenches and confirm:
  - eager in-place collectives baseline
  - compile + `dynamo.disable` on collectives → many breaks / poor throughput
  - compile + funcol-gated path → few breaks / improved throughput
- **Rank divergence drill**: intentionally perturb one rank (shape/flag) and observe the failure mode (timeout/hang), then fix by enforcing parity.
- **Guard/shape sensitivity sweep**: vary sequence length or other “maybe dynamic” dimensions and measure recompiles; decide which dimensions must be frozen for stable distributed compile.
- **DTensor vs explicit TP prototype**: implement a tiny DTensor shard/redistribute example and compare (a) eager overhead vs (b) compile lowering; use this to decide whether DTensor is worth revisiting for future TP/PP work.
- **Mesh vs world compilation split**: validate that rank0-out-of-mesh (stage0) can run without compiling the same graphs as mesh ranks, while mesh ranks stay symmetric within `mesh_pg`.
- **Compile caching/warmup strategy**: measure cold-start compile time vs warmed steady state; decide what “warmup contract” is acceptable before serving traffic.
