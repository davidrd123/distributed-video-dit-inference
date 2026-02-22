# Topic 9: How Dynamo tracing works — graph breaks, guards, eager fallback

TorchDynamo uses CPython's **PEP 523 frame evaluation API** to intercept Python bytecode execution, symbolically trace tensor operations into an FX graph, and generate guard functions that check whether cached compilations remain valid. When it encounters untraceable code (data-dependent control flow, unsupported Python constructs), it inserts a **graph break** — splitting the code into multiple compiled subgraphs with eager Python between them.

## Resources

<!-- Resource IDs from manifest belonging to this topic -->
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| dynamo-deep-dive | Dynamo Deep-Dive | high | converted |
| torchdynamo-uwplse | How does torch.compile work? | medium | pending |
| torch-compile-missing-manual | torch.compile: the missing manual | medium | link_only |
| torch-compile-programming-model | torch.compile Programming Model | medium | pending |

## Implementation context

Dynamo tracing is central to two findings: (1) **`torch._dynamo.disable()` on collectives caused ~160 graph breaks per forward**, making compile a net negative at 9.6 FPS (Runs 8-9b). Fixed by functional collectives (Run 12b: 24.5 FPS). (2) **The remaining recurring graph break** is KV-cache dynamic slicing (`kv_cache["k"][:, local_start_index:local_end_index]`) with Tensor bounds. A `narrow()`/SymInt rewrite attempt produced more breaks and was reverted. Current steady state: `unique_graphs=12-14, graph_breaks=2`.

See: `refs/implementation-context.md` → Phase 1, `scope-drd/notes/FA4/h200/tp/bringup-run-log.md` Runs 8-14, Known Issue 8.

## Synthesis

<!-- To be filled during study -->

### Mental model

### Key concepts

### Cross-resource agreement / disagreement

### Practical checklist

### Gotchas and failure modes

### Experiments to run
