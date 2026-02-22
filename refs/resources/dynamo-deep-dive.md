# Dynamo Deep-Dive

| Field | Value |
|-------|-------|
| Source | https://docs.pytorch.org/docs/stable/torch.compiler_dynamo_deepdive.html |
| Type | docs |
| Topics | 9 |
| Status | condensed |

## Why it matters

The most comprehensive official resource on TorchDynamo internals: PEP 523 frame evaluation, Python bytecode translation via `VariableTracker`, guard generation, continuation functions at graph breaks, and SymInt/symbolic shapes. In our stack, “compile succeeded but perf collapsed” was explained by Dynamo-level mechanics (graph breaks + guard churn): ~160 breaks/forward yielded ~9.6 FPS (Runs 8–9b) until functional collectives removed the break source and restored ~24.5 FPS (Run 12b).

## Key sections

- [A Gentle Introduction to Dynamo](../../sources/dynamo-deep-dive/full.md#a-gentle-introduction-to-dynamo) — what Dynamo traces (linear FX graph) and why traces depend on inputs.
- [Making Dynamo Sound: Guards](../../sources/dynamo-deep-dive/full.md#making-dynamo-sound-guards) — what a guard is, how guard failures trigger recompilation, and `TORCH_LOGS=guards` / `TORCH_LOGS=recompiles`.
- [Symbolic Shapes](../../sources/dynamo-deep-dive/full.md#symbolic-shapes) — SymInt, “static by default”, and what “dynamic shapes” actually means in Dynamo.
- [Making Dynamo Complete: Graph Breaks](../../sources/dynamo-deep-dive/full.md#making-dynamo-complete-graph-breaks) — why Dynamo emits *multiple* graphs, how graph breaks work, and why they’re the main perf footgun.
- [PEP 523: Adding a frame evaluation API to CPython](../../sources/dynamo-deep-dive/full.md#pep-523-adding-a-frame-evaluation-api-to-cpython) — how Dynamo hooks CPython to rewrite/execute bytecode.
- [Generating the Output Graph](../../sources/dynamo-deep-dive/full.md#generating-the-output-graph) — FX graph + `OutputGraph` mental model.

## Core claims

1. **Claim**: Dynamo is a tracer that executes Python and records a **linear** (control-flow-free) sequence of operations into an FX graph; the traced graph depends on the runtime inputs.
   **Evidence**: [sources/dynamo-deep-dive/full.md#a-gentle-introduction-to-dynamo](../../sources/dynamo-deep-dive/full.md#a-gentle-introduction-to-dynamo)

2. **Claim**: Non-tensor inputs are often treated as constants in the graph, but Dynamo can trace integer inputs (and tensor sizes) symbolically via `torch.SymInt` to reduce recompilation.
   **Evidence**: [sources/dynamo-deep-dive/full.md#a-gentle-introduction-to-dynamo](../../sources/dynamo-deep-dive/full.md#a-gentle-introduction-to-dynamo), [sources/dynamo-deep-dive/full.md#symbolic-shapes](../../sources/dynamo-deep-dive/full.md#symbolic-shapes)

3. **Claim**: Dynamo’s symbolic-shape behavior is “static by default”: it specializes on concrete integers/shapes on the first run, and only starts tracing a dimension as symbolic once it observes that dimension change.
   **Evidence**: [sources/dynamo-deep-dive/full.md#static-by-default](../../sources/dynamo-deep-dive/full.md#static-by-default)

4. **Claim**: Guards are the mechanism Dynamo uses to safely reuse compiled graphs across calls: a guard is an assumption that must hold on new inputs; guard failures trigger recompilation, and `TORCH_LOGS=recompiles` can attribute recompiles to the failing guard.
   **Evidence**: [sources/dynamo-deep-dive/full.md#making-dynamo-sound-guards](../../sources/dynamo-deep-dive/full.md#making-dynamo-sound-guards)

5. **Claim**: Guards can arise from sources deeper than top-level inputs (e.g., `GetItemSource`), and Dynamo can also add non-trivial guards mid-execution when it encounters conditions on symbolic values; `TORCH_LOGS=dynamo` can show where a guard was added.
   **Evidence**: [sources/dynamo-deep-dive/full.md#making-dynamo-sound-guards](../../sources/dynamo-deep-dive/full.md#making-dynamo-sound-guards)

6. **Claim**: When Dynamo encounters code it cannot trace (e.g., Python bindings to C++/Rust libraries or unsupported ops), it can fall back to CPython by inserting a **graph break**: it emits multiple graphs and lets CPython execute the “problem” code between them.
   **Evidence**: [sources/dynamo-deep-dive/full.md#making-dynamo-complete-graph-breaks](../../sources/dynamo-deep-dive/full.md#making-dynamo-complete-graph-breaks)

7. **Claim**: Graph breaks are implemented by rewriting Python bytecode into (1) code that calls compiled graph(s), (2) stack reconstruction, (3) the graph-breaking bytecode, and (4) continuation code that runs subsequent graphs; `TORCH_LOGS=bytecode` can display this rewrite.
   **Evidence**: [sources/dynamo-deep-dive/full.md#making-dynamo-complete-graph-breaks](../../sources/dynamo-deep-dive/full.md#making-dynamo-complete-graph-breaks)

8. **Claim**: Dynamo’s execution model relies on CPython’s frame evaluation API (PEP 523) and a Python-level reimplementation of CPython’s stack machine semantics to interpret bytecode and produce the traced graph.
   **Evidence**: [sources/dynamo-deep-dive/full.md#pep-523-adding-a-frame-evaluation-api-to-cpython](../../sources/dynamo-deep-dive/full.md#pep-523-adding-a-frame-evaluation-api-to-cpython), [sources/dynamo-deep-dive/full.md#implementing-cpython-in-python](../../sources/dynamo-deep-dive/full.md#implementing-cpython-in-python)

9. **Claim**: Dynamo builds an FX graph via `OutputGraph`, where intermediates are `fx.Node`s wrapped by `fx.Proxy`s; operations on proxies record nodes into the graph.
   **Evidence**: [sources/dynamo-deep-dive/full.md#generating-the-output-graph](../../sources/dynamo-deep-dive/full.md#generating-the-output-graph)

10. **Claim**: The traced FX graph is then handed off to a compiler backend (e.g., Inductor) to produce low-level code; Dynamo’s job is the Python-level tracing/caching/bytecode rewrite that makes this backend invocation possible.
   **Evidence**: [sources/dynamo-deep-dive/full.md#making-dynamo-complete-graph-breaks](../../sources/dynamo-deep-dive/full.md#making-dynamo-complete-graph-breaks)

## API surface / configuration

**Primary debug knobs (from the doc examples):**
- `TORCH_LOGS=graph_code` — print the traced FX graph code.
- `TORCH_LOGS=guards` — print installed guards.
- `TORCH_LOGS=recompiles` — show guard failures that triggered recompilation.
- `TORCH_LOGS=graph_sizes` — show symbolic vs concrete tensor sizes.
- `TORCH_LOGS=bytecode` — show rewritten bytecode / continuation functions around graph breaks.
- `TORCH_LOGS=graph_breaks` — attribute graph breaks (best first stop for perf regressions).
- `TORCH_LOGS=dynamo` — lower-level Dynamo trace info (e.g., where a guard was added).

**Symbolic-shape controls mentioned by the doc:**
- `torch._dynamo.mark_dynamic(...)` (pre-mark a dimension as dynamic to avoid the first “static” compile).
- `torch.compile(dynamic=True)` (trace shapes/integers dynamically; “mostly useful for debugging purposes” per the doc).

## Actionables / gotchas

- **Graph breaks are the perf killer, not “compile” in general**: we saw ~160 breaks/forward in TP=2+compile (Runs 8–9b), collapsing throughput to **~9.6 FPS**. The fix was to remove the break source (collectives wrapped by `torch._dynamo.disable()`), switching to functional collectives under compile and restoring **~24.5 FPS** (Run 12b). See: `scope-drd/notes/FA4/h200/tp/bringup-run-log.md` (Runs 8–12b) and `refs/implementation-context.md` (guide row for `funcol-rfc-93173` + `dynamo-deep-dive`).
- **Use `TORCH_LOGS=graph_breaks` as first-line diagnosis**: the doc’s guidance matches our workflow — graph breaks split execution into multiple compiled segments with CPython in-between. If you’re chasing perf, start by counting and attributing breaks before tuning kernels.
- **Distinguish “graph breaks” vs “recompiles”**: breaks come from unsupported/escaped code paths; recompiles come from guard failures. In distributed mode, either can desynchronize ranks if shapes/flags differ, so enforce env parity and fixed-shape contracts across ranks. See: `scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md` (Q6).
- **Keep the compiled region “pure” (no Python side effects / logging / host callbacks)**: in the doc’s example, a `print()` forces a graph break. In our code, any incidental Python work inside the hot path has the same effect (and is harder to spot than a literal `print`).
- **Make dynamic shapes an explicit decision, not an accident**: Dynamo is “static by default” and will create an initial specialized graph; if variable video lengths / chunk shapes are expected, decide whether to lock them down (preferred for bringup) or to mark dynamics intentionally (`mark_dynamic` / `dynamic=True`) and accept the guard/compile complexity.
- **SymInt guards can appear mid-execution**: guards aren’t only “inputs”; they can be added when Dynamo sees control flow on symbolic values. If you see a surprising recompile/hang, `TORCH_LOGS=dynamo` can show where a guard was added.
- **After the funcol fix, the remaining steady-state break is KV-cache dynamic slicing**: our current steady state is `unique_graphs=12–14, graph_breaks=2` (Runs 13–14), and both breaks are the known KV-cache update slice with Tensor start/stop (shows up as `Dynamic slicing with Tensor arguments`). A narrow()/SymInt rewrite attempt (requiring `TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1`) introduced *more* breaks (e.g. `graph_breaks=7` vs `2`) and was reverted; treat this as an open Known Issue rather than something you can paper over with more compile flags. See: `scope-drd/notes/FA4/h200/tp/bringup-run-log.md` (Runs 13–14; Known Issue 8) and `scope-drd/notes/FA4/h200/tp/session-state.md` (“Graph-break hygiene”).
- **Distributed inference multiplies the blast radius**: Dynamo’s model is per-process. If different ranks take different graph-break paths or compile different specializations, they can hit NCCL collectives in different orders → hang. Treat “same shapes, same flags, same code path” as a contract whenever collectives are inside compiled regions. See: `scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md` (Q6).
- See: `refs/implementation-context.md` (Per-card actionables sharpening guide).

## Related resources

- [funcol-rfc-93173](funcol-rfc-93173.md) -- functional collectives designed to avoid graph breaks during tracing
- [ezyang-state-of-compile](ezyang-state-of-compile.md) -- practical compile + distributed interaction
- [cuda-graphs-guide](cuda-graphs-guide.md) -- CUDA graphs used by reduce-overhead compile mode
- [pytorch-cuda-semantics](pytorch-cuda-semantics.md) -- CUDA semantics underlying compiled execution
- [making-dl-go-brrrr](making-dl-go-brrrr.md) -- overhead elimination that motivates compilation
