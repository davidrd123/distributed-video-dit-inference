---
status: stub
---

# Topic 5: CUDA streams — execution/dependency model, events, synchronization, NCCL interaction

Streams are the fundamental concurrency primitive for overlapping compute, communication, and memory transfers. NCCL operations execute on their own streams, and understanding how events synchronize across streams is essential for pipeline parallelism.

## Resources

<!-- Resource IDs from manifest belonging to this topic -->
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| pytorch-cuda-semantics | PyTorch CUDA Semantics | high | converted |
| cuda-async-execution | CUDA Programming Guide: Asynchronous Execution | medium | pending |
| leimao-cuda-stream | CUDA Stream | low | pending |

## Implementation context

CUDA streams show up in two load-bearing places: compile-aware collectives (Run 12b) require correct stream ordering when collectives are traced in-graph, and overlap experiments depend on explicit event/stream waits. The parked async-decode-overlap plan proposes launching VAE decode on a separate CUDA stream to hide ~3–10ms/chunk (≈2–8% ceiling) behind the next chunk’s denoise, but warns overlap collapses under semantics-preserving recompute (R0a). StreamDiffusionV2’s design similarly uses a dedicated comm stream plus waits to overlap stage transfers with compute.

See: `refs/implementation-context.md` → Phase 1-2, `scope-drd/notes/FA4/h200/tp/bringup-run-log.md` (Run 12b), `scope-drd/notes/FA4/h200/tp/async-decode-overlap-impl-plan.md` (CUDA stream overlap), `scope-drd/notes/FA4/h200/tp/streamdiffusion-v2-analysis-opus.md` (comm stream overlap).

## Synthesis

<!-- To be filled during study -->

### Mental model

### Key concepts

### Cross-resource agreement / disagreement

### Practical checklist

### Gotchas and failure modes

### Experiments to run
