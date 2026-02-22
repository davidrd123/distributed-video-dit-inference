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

## Implementation context

The Scope KV-cache is a **fixed-size ring buffer** (~32K tokens, head-sharded in TP mode) with eviction and recompute. The cache lifecycle (hard cut, recompute, soft transition via `kv_cache_attention_bias`) is a major coupling point: recompute depends on `decoded_frame_buffer`, which is why v0 workers run the full pipeline. The `cache_epoch` counter in `PPEnvelopeV1` tracks hard-cut generations. The `SCOPE_KV_CACHE_RECOMPUTE_EVERY=2` experiment (Run 11) was a dead end — quality degradation with no net FPS gain.

See: `refs/implementation-context.md` → Phase 2-3, `scope-drd/notes/FA4/h200/tp/explainers/05-kv-cache-head-sharding.md`.

## Synthesis

<!-- To be filled during study -->

### Mental model

### Key concepts

### Cross-resource agreement / disagreement

### Practical checklist

### Gotchas and failure modes

### Experiments to run
