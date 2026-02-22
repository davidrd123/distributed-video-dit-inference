# Topic 12: Compile + distributed interaction — compile with DDP/FSDP/TP

The core challenge is that **DDP/FSDP use backward hooks for communication**, and these hooks create graph breaks in AOTAutograd. The solutions are: (1) **Compiled Autograd** (PyTorch 2.4+), which captures the full backward graph at runtime, (2) **FSDP2 built on DTensor**, which is compile-friendly by design, and (3) **functional collectives** for tensor parallelism.

## Resources

<!-- Resource IDs from manifest belonging to this topic -->
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| ezyang-state-of-compile | State of torch.compile for training (August 2025) | high | pending |
| compiled-autograd-tutorial | Compiled Autograd Tutorial | medium | pending |
| ezyang-ways-to-compile | Ways to use torch.compile | medium | pending |
| torch-compiler-faq-dist | torch.compiler FAQ: Distributed Section | medium | pending |
| tp-tutorial | Large Scale Transformer Training with Tensor Parallel | medium | pending |
| vllm-torch-compile | Introduction to torch.compile and How It Works with vLLM | medium | pending |

## Implementation context

Compile interacted catastrophically with distributed collectives until Run 12b: `torch._dynamo.disable()` on each all-reduce caused ~160 graph breaks per forward and TP=2 throughput dropped from 16 FPS (Run 7) to **9.6 FPS** (Runs 8-9b). Replacing those with functional collectives made collectives traceable and restored end-to-end performance to **24.5 FPS** (Run 12b). Current steady state in the E2E harness is `unique_graphs=12, graph_breaks=2` after removing attention-backend logger breaks (Run 14).

See: `refs/implementation-context.md` → Phase 1, `scope-drd/notes/FA4/h200/tp/bringup-run-log.md` (Runs 7-14).

## Synthesis

<!-- To be filled during study -->

### Mental model

### Key concepts

### Cross-resource agreement / disagreement

### Practical checklist

### Gotchas and failure modes

### Experiments to run
