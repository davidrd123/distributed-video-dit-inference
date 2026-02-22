---
status: draft
---

# Topic 9: How Dynamo tracing works — graph breaks, guards, eager fallback

TorchDynamo uses CPython's **PEP 523 frame evaluation API** to intercept Python bytecode execution, symbolically trace tensor operations into an FX graph, and generate guard functions that check whether cached compilations remain valid. When it encounters untraceable code (data-dependent control flow, unsupported Python constructs), it inserts a **graph break** — splitting the code into multiple compiled subgraphs with eager Python between them.

## Resources

<!-- Resource IDs from manifest belonging to this topic -->
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| dynamo-deep-dive | Dynamo Deep-Dive | high | condensed |
| ezyang-state-of-compile | State of torch.compile for training (August 2025) | high | condensed |
| funcol-rfc-93173 | RFC #93173: PT2-Friendly Traceable, Functional Collective Communication APIs | high | condensed |
| torchdynamo-uwplse | How does torch.compile work? | medium | pending |
| torch-compile-missing-manual | torch.compile: the missing manual | medium | link_only |
| torch-compile-programming-model | torch.compile Programming Model | medium | pending |

## Implementation context

Dynamo tracing is central to two findings: (1) **`torch._dynamo.disable()` on collectives caused ~160 graph breaks per forward**, making compile a net negative at 9.6 FPS (Runs 8-9b). Fixed by functional collectives (Run 12b: 24.5 FPS). (2) **The remaining recurring graph break** is KV-cache dynamic slicing (`kv_cache["k"][:, local_start_index:local_end_index]`) with Tensor bounds. A `narrow()`/SymInt rewrite attempt produced more breaks and was reverted. Current steady state: `unique_graphs=12-14, graph_breaks=2`.

See: `refs/implementation-context.md` → Phase 1, `scope-drd/notes/FA4/h200/tp/bringup-run-log.md` Runs 8-14, Known Issue 8.

## Synthesis

<!-- To be filled during study -->

### Mental model

Dynamo is not a “whole-program compiler.” It’s a **tracing + fallback** system:
- It hooks CPython execution (PEP 523) and, while running your Python, records a **linear FX graph** of tensor ops (see `dynamo-deep-dive` claim 1 and claim 8).
- It caches compiled results behind **guards**; if a guard fails, it recompiles (see `dynamo-deep-dive` claim 4; and `ezyang-state-of-compile` claim 3).
- It hands the FX graph to a backend (often Inductor) to generate optimized code (see `dynamo-deep-dive` claim 10).

The most important operational concept is the **graph break**:
- When Dynamo can’t (or won’t) trace some line(s), it inserts a graph break, runs that region in eager Python, then resumes tracing later (see `dynamo-deep-dive` claim 6 and claim 7; `ezyang-state-of-compile` claim 4).
- Each compiled region is “one FX graph + one backend compile.” Graph breaks therefore create **boundaries** between compiled regions, and CPython overhead (and extra launch/sync surfaces) in between.

This gives a simple performance model:
- Fewer breaks → fewer compiled regions → less Python overhead → faster steady state.
- More breaks → many tiny graphs with Python in-between → slower.

Ground truth from our bringup: pre-funcol we hit ~160 breaks/forward and compile was net negative (9.6 FPS), while the funcol change removed the dominant break source and restored throughput to ~24.5 FPS; steady state is now `graph_breaks=2` with `unique_graphs=12–14` (see Implementation context; and the actionables in `dynamo-deep-dive`, `funcol-rfc-93173`, and `ezyang-state-of-compile`).

### Key concepts

Definitions below are the minimum vocabulary to debug “why did compile get slower/faster?”

- **FX graph**: the intermediate representation Dynamo records while executing Python; Dynamo’s traced output is a linear FX graph of tensor operations (see `dynamo-deep-dive` claim 1 and claim 9).
- **Compiled region**: one contiguous FX graph capture that gets handed to a backend (Inductor, etc.) for codegen; multiple regions exist when there are graph breaks (see `dynamo-deep-dive` claim 10; `ezyang-state-of-compile` claim 2).
- **Graph break**: an explicit boundary where Dynamo stops tracing, falls back to eager for a segment, and then may resume tracing; breaks are “first-class” behavior and can be banned with `fullgraph=True` (see `dynamo-deep-dive` claim 6–7; `ezyang-state-of-compile` claim 4).
- **Guard**: a runtime check Dynamo installs to ensure cached compiled graphs are valid for new inputs; guard failure triggers recompilation (see `dynamo-deep-dive` claim 4 and claim 5).
- **Recompile**: the event of compiling a new specialization because guard assumptions changed (often due to non-tensor values / sizes); distinct from graph breaks, but equally a source of jitter and divergence (see `ezyang-state-of-compile` claim 3; `dynamo-deep-dive` claim 4).
- **Continuation function / bytecode rewrite**: Dynamo’s mechanism for graph breaks: it rewrites Python bytecode to call compiled graphs, reconstruct the stack, execute the “break” bytecode, then continue (see `dynamo-deep-dive` claim 7 and claim 8).
- **SymInt / symbolic shapes**: a way to trace sizes/integers symbolically to reduce recompiles; Dynamo is “static by default” and only generalizes once it observes variation (see `dynamo-deep-dive` claim 2–3; `ezyang-state-of-compile` claim 3).

Debug knobs that matter in practice (all from `dynamo-deep-dive` API surface):
- `TORCH_LOGS=graph_breaks` — attribute breaks (first stop for perf regressions).
- `TORCH_LOGS=guards` / `TORCH_LOGS=recompiles` — explain guard assumptions and why they failed.
- `TORCH_LOGS=bytecode` — inspect the rewritten continuation functions around breaks.

### Cross-resource agreement / disagreement

All three sources tell the same story from different angles:

- **Graph breaks are the primary perf footgun**: Dynamo deep dive explains the mechanics (graph breaks + continuations) (see `dynamo-deep-dive` claim 6–7); ezyang frames breaks as expected compositional behavior and calls out `fullgraph=True` as the “ban breaks” mode (see `ezyang-state-of-compile` claim 4); funcol exists because classic in-place collectives don’t trace cleanly and can force breaks (see `funcol-rfc-93173` claim 1 and claim 5).
- **Distributed adds a new failure mode: rank-divergent compilation**: even if each rank “works,” if ranks take different break/guard paths, they can hit collectives in different orders and hang. Ezyang calls this out explicitly as a fragility of distributed compilation today (see `ezyang-state-of-compile` claim 10), and our own experience matches (Implementation context).

Differences in emphasis (not disagreements):
- **Regional compilation vs “compile everything”**: ezyang treats compilation as a compositional tool and recommends regional compilation (compile smaller units) as a lever to control compile time on Transformers (see `ezyang-state-of-compile` claim 5). Dynamo deep dive is more focused on soundness (guards) and the correctness mechanics of breaks/continuations (see `dynamo-deep-dive` claim 4 and claim 7).
- **Funcol’s scope is narrow on purpose**: funcol fixes one specific (but common) source of breaks—side-effecting collectives—by providing a traceable, functional tensor→tensor surface with explicit wait semantics (see `funcol-rfc-93173` claim 5–7). It does not “solve compile” broadly; it just makes one class of distributed operations tractable to the tracer.

### Practical checklist

Use this as the “first 10 minutes” playbook whenever compile performance regresses.

- **Always start by attributing breaks**: run with `TORCH_LOGS=graph_breaks` and count breaks per forward; treat break count as a regression signal (see `dynamo-deep-dive` API surface; and `ezyang-state-of-compile` claim 4 on breaks as first-class behavior).
- **Track steady-state health metrics**: our current steady state is `graph_breaks=2` and `unique_graphs=12–14`; increases are “guilty until proven innocent” (see Implementation context; and `ezyang-state-of-compile` actionables).
- **Use `fullgraph=True` during development of hot paths**: this is the fastest way to force yourself to remove/relocate the break source rather than “living with” a silent fallback (see `ezyang-state-of-compile` claim 4).
- **Separate breaks from recompiles**: if break count is stable but performance jitters, check `TORCH_LOGS=recompiles` / `TORCH_LOGS=guards` for dynamic-shape/flag churn (see `dynamo-deep-dive` claim 4; `ezyang-state-of-compile` claim 3).
- **For TP+compile, assume collectives are suspect until proven traceable**: if a collective is wrapped in `torch._dynamo.disable()` or threads non-tensor objects through traced code, you’re likely forcing breaks (see `funcol-rfc-93173` claim 1 and claim 4–5). Our measured outcome was ~160 breaks/forward pre-funcol → 2 breaks steady-state post-funcol (Implementation context).

### Gotchas and failure modes

- **Python side effects inside the hot path**: logging/printing/host callbacks can cause breaks just like unsupported ops (see `dynamo-deep-dive` claim 6 for “can’t trace → break”; and its actionables on side effects).
- **Guard churn from dynamic shapes / non-tensor values**: “static by default” specialization means you can recompile when sizes change; in distributed mode this becomes correctness risk if ranks don’t change in lockstep (see `dynamo-deep-dive` claim 3–5; `ezyang-state-of-compile` claim 3 and claim 10).
- **Rank-divergent compilation → collective ordering hangs**: the distributed nightmare scenario is “same code, different compiled path.” If ranks see different guards/breaks, they can enter collectives in different orders and deadlock (see `ezyang-state-of-compile` claim 10).
- **Our known remaining break: KV-cache dynamic slicing**: the steady-state break source is the KV-cache update slice with tensor start/stop (Implementation context; see also the final actionable in `dynamo-deep-dive`).
- **Eager regression trap (funcol)**: functional collectives are meant for compiled regions; using them unconditionally in eager can add overhead (our Run 12a regression; see `funcol-rfc-93173` actionables).

### Experiments to run

- **Break census before/after every “small” change**: run a fixed input through a short harness and record `graph_breaks`, `unique_graphs`, and wall-time. Treat break count as a first-order regression signal (Implementation context; `dynamo-deep-dive` API surface).
- **Guard stability test**: run the same shapes/flags repeatedly and confirm no recompiles; then vary one dimension intentionally and confirm you understand the new guards (see `dynamo-deep-dive` claim 3–5; `ezyang-state-of-compile` claim 3).
- **Fullgraph gate**: set `fullgraph=True` and ensure the hot forward path either compiles or fails loudly; use the failure to drive code motion (move logging, replace dynamic slicing, adopt funcol where appropriate) (see `ezyang-state-of-compile` claim 4).
- **Regional vs monolithic compile**: measure compile time and steady-state latency when compiling a block/stage function vs the full end-to-end loop (see `ezyang-state-of-compile` claim 5 on regional compilation).
- **Distributed symmetry check**: run N ranks with identical inputs and assert the same graph-break trace across ranks; treat any divergence as a correctness bug before it becomes an intermittent NCCL hang (see `ezyang-state-of-compile` claim 10).
