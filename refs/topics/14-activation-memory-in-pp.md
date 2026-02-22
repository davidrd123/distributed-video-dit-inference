# Topic 14: Activation memory in PP

Peak activation memory in PP depends on the schedule. GPipe stores activations for all micro-batches; 1F1B limits in-flight activations to `num_stages`. **Activation checkpointing** trades compute for memory by recomputing activations during backward.

## Resources

<!-- Resource IDs from manifest belonging to this topic -->
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| megatron-ptdp | Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM | medium | pending |
| zero-bubble-pp | Zero Bubble Pipeline Parallelism | high | pending |
| pytorch-pipelining-api | PyTorch torch.distributed.pipelining API | high | pending |

## Implementation context

PP0 bringup starts in R1 (recompute disabled) because the semantics-preserving recompute path is coupled to decoded pixels: steady-state `get_context_frames()` uses `decoded_frame_buffer[:, :1]` and a VAE re-encode. In the R0a plan, rank0 supplies `context_frames_override` in the envelope; the tensor is small (~0.26 MB/chunk at 320×576) but it creates a timing dependency that can constrain overlap. The PP pilot therefore treats `expected_generator_calls = (do_kv_recompute?1:0)+num_denoise_steps` as a deadlock tripwire on every chunk.

See: `refs/implementation-context.md` → Phase 2, `scope-drd/notes/FA4/h200/tp/pp-topology-pilot-plan.md` (recompute coupling + sizes), `scope-drd/notes/FA4/h200/tp/pp-next-steps.md` (expected_generator_calls), `scope-drd/notes/FA4/h200/tp/pp0-bringup-runbook.md` (per-chunk checklist).

## Synthesis

<!-- To be filled during study -->

### Mental model

### Key concepts

### Cross-resource agreement / disagreement

### Practical checklist

### Gotchas and failure modes

### Experiments to run
