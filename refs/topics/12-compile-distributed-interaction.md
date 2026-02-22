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

## Synthesis

<!-- To be filled during study -->

### Mental model

### Key concepts

### Cross-resource agreement / disagreement

### Practical checklist

### Gotchas and failure modes

### Experiments to run
