# Topic 9: How Dynamo tracing works — graph breaks, guards, eager fallback

TorchDynamo uses CPython's **PEP 523 frame evaluation API** to intercept Python bytecode execution, symbolically trace tensor operations into an FX graph, and generate guard functions that check whether cached compilations remain valid. When it encounters untraceable code (data-dependent control flow, unsupported Python constructs), it inserts a **graph break** — splitting the code into multiple compiled subgraphs with eager Python between them.

## Resources

<!-- Resource IDs from manifest belonging to this topic -->
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| dynamo-deep-dive | Dynamo Deep-Dive | high | pending |
| torchdynamo-uwplse | How does torch.compile work? | medium | pending |
| torch-compile-missing-manual | torch.compile: the missing manual | medium | link_only |
| torch-compile-programming-model | torch.compile Programming Model | medium | pending |

## Synthesis

<!-- To be filled during study -->

### Mental model

### Key concepts

### Cross-resource agreement / disagreement

### Practical checklist

### Gotchas and failure modes

### Experiments to run
