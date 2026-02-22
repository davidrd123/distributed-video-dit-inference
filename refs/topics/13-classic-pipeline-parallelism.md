# Topic 13: Classic PP (GPipe, PipeDream) — micro-batching, 1F1B schedule, bubble fraction

Pipeline parallelism partitions a model across devices by layer. **GPipe** fills the pipeline with micro-batches and synchronizes at the end (high memory, simple). **PipeDream's 1F1B schedule** interleaves one forward and one backward per micro-batch, reducing peak activation memory from O(num_microbatches) to O(num_stages).

## Resources

<!-- Resource IDs from manifest belonging to this topic -->
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| gpipe | GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism | high | pending |
| pipedream | PipeDream: Generalized Pipeline Parallelism for DNN Training | medium | pending |
| pipedream-2bw | Memory-Efficient Pipeline-Parallel DNN Training (PipeDream-2BW) | high | pending |
| pp-siboehm | Pipeline-Parallelism: Distributed Training via Model Partitioning | medium | pending |

## Implementation context

PP is the **next topology play** after TP v0. The working design is rank0-out-of-mesh (Stage 0 = encode/decode, Stage 1 = generator-only with TP inside `mesh_pg`). Contracts (`PPEnvelopeV1`/`PPResultV1`) are defined, transport is implemented, smoke test passes. The key insight from StreamDiffusionV2 analysis: PP without batching at B=1 gives only 50% utilization (2 stages). PP becomes compelling when B>1 items are in flight.

Bringup plan is Steps A1-A5 in `scope-drd/notes/FA4/h200/tp/pp-next-steps.md`. Pseudocode reference in `pp-control-plane-pseudocode.md`. Runbook in `pp0-bringup-runbook.md`.

See: `refs/implementation-context.md` → Phase 3.

## Synthesis

<!-- To be filled during study -->

### Mental model

### Key concepts

### Cross-resource agreement / disagreement

### Practical checklist

### Gotchas and failure modes

### Experiments to run
