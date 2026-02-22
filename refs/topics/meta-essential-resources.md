# Meta: Essential resources

These resources cut across multiple topics and are worth bookmarking as ongoing references.

## Resources

<!-- Resource IDs from manifest belonging to this topic -->
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| ezyang-blog | Edward Yang's blog | medium | pending |
| pytorch-dev-podcast | PyTorch Developer Podcast | low | link_only |
| gpu-mode-lectures | GPU MODE lecture series | medium | link_only |
| ezyang-parallelism-mesh-zoo | The Parallelism Mesh Zoo | medium | pending |
| megatron-lm-repo | Megatron-LM repository | medium | pending |

## Implementation context

When debugging Scope bringup, the fastest “lookup path” usually isn’t a paper — it’s one of these living references plus the local working notes. In practice:

- For compile/distributed failures: start from `refs/implementation-context.md`, then search `scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md`.
- For topology questions (TP/PP/SP mixes): cross-check the plan in `scope-drd/notes/FA4/h200/tp/pp-next-steps.md` against the mental models in `ezyang-parallelism-mesh-zoo` / Megatron.
- For “what did upstream intend?” on compiler behavior: Edward Yang’s blog posts are usually the canonical explanation.

## Synthesis

<!-- To be filled during study -->

### Mental model

### Key concepts

### Cross-resource agreement / disagreement

### Practical checklist

### Gotchas and failure modes

### Experiments to run
