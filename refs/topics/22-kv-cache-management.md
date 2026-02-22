# Topic 22: KV cache management in streaming inference

KV caches store the key/value projections from previous tokens to avoid recomputation during autoregressive generation. In video DiT streaming, this extends to **temporal KV caches** across denoising steps and frames. PagedAttention (from vLLM) introduced OS-style paged memory management, reducing waste from **60-80% to under 4%**.

## Resources

<!-- Resource IDs from manifest belonging to this topic -->
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| pagedattention | Efficient Memory Management for Large Language Model Serving with PagedAttention | high | pending |
| continuous-batching | Continuous Batching from First Principles | medium | pending |
| vllm-anatomy | Inside vLLM: Anatomy of a High-Throughput LLM Inference System | medium | pending |
| vllm-distributed | vLLM Distributed Inference Blog Post | medium | pending |

## Synthesis

<!-- To be filled during study -->

### Mental model

### Key concepts

### Cross-resource agreement / disagreement

### Practical checklist

### Gotchas and failure modes

### Experiments to run
