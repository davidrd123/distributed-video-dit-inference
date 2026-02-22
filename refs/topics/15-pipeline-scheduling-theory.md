# Topic 15: Pipeline scheduling theory — bubble fraction derivation, Little's law connection

The bubble fraction for a P-stage pipeline processing B micro-batches is **(P-1)/(B+P-1)**. This is a direct consequence of pipeline startup and drain latency. **Little's law** (L = lambda * W) provides the framework: to keep P stages busy, you need at least P micro-batches in flight, and throughput approaches the ideal rate as B approaches infinity.

## Resources

<!-- Resource IDs from manifest belonging to this topic -->
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| jax-scaling-book | How to Parallelize a Transformer for Training (JAX Scaling Book) | medium | pending |
| megatron-trillion-params | Scaling Language Model Training to a Trillion Parameters Using Megatron | medium | pending |
| megatron-pp-schedules | Megatron-LM Pipeline Parallel Schedules source | medium | pending |

## Implementation context

Scope already observed the classic pipeline-bubble reality: block-PP can reach **~1.87× throughput** with enough in-flight work, but single-stream per-chunk latency stayed flat (PP doesn’t shorten the critical path). StreamDiffusionV2’s takeaway is the same: PP alone with **2 stages and B=1** has a **50% bubble**, so you need “Stream Batch” / `max_outstanding≥2` to fill. The PP0 plan bakes this into bringup by starting with double-buffer depths `D_in=D_out=2` and an overlap pass gate `OverlapScore ≥ 0.30`.

See: `refs/implementation-context.md`, `scope-drd/notes/FA4/h200/tp/explainers/01-why-two-gpus.md` (PP throughput vs latency), `scope-drd/notes/FA4/h200/tp/streamdiffusion-v2-analysis-opus.md` (PP bubble + Stream Batch), `scope-drd/notes/FA4/h200/tp/pp0-bringup-runbook.md` (overlap gate), `scope-drd/notes/FA4/h200/tp/pp-topology-pilot-plan.md` (queue depths).

## Synthesis

<!-- To be filled during study -->

### Mental model

### Key concepts

### Cross-resource agreement / disagreement

### Practical checklist

### Gotchas and failure modes

### Experiments to run
