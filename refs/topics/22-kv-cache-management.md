---
status: draft
---

# Topic 22: KV cache management in streaming inference

KV caches store the key/value projections from previous tokens to avoid recomputation during autoregressive generation. In video DiT streaming, this extends to **temporal KV caches** across denoising steps and frames. PagedAttention (from vLLM) introduced OS-style paged memory management, reducing waste from **60-80% to under 4%**.

## Resources

<!-- Resource IDs from manifest belonging to this topic -->
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| pagedattention | Efficient Memory Management for Large Language Model Serving with PagedAttention | high | pending |
| causvid | From Slow Bidirectional to Fast Autoregressive Video Diffusion Models | high | converted |
| continuous-batching | Continuous Batching from First Principles | medium | pending |
| vllm-anatomy | Inside vLLM: Anatomy of a High-Throughput LLM Inference System | medium | pending |
| vllm-distributed | vLLM Distributed Inference Blog Post | medium | pending |

## Implementation context

The Scope KV-cache is a **fixed-size ring buffer** (~32K tokens, head-sharded in TP mode) with eviction and recompute. The cache lifecycle (hard cut, recompute, soft transition via `kv_cache_attention_bias`) is a major coupling point: recompute depends on `decoded_frame_buffer`, which is why v0 workers run the full pipeline. The `cache_epoch` counter in `PPEnvelopeV1` tracks hard-cut generations. The `SCOPE_KV_CACHE_RECOMPUTE_EVERY=2` experiment (Run 11) was a dead end — quality degradation with no net FPS gain.

See: `refs/implementation-context.md` → Phase 2-3, `scope-drd/notes/FA4/h200/tp/explainers/05-kv-cache-head-sharding.md`.

Relevant Scope code:
- `scope-drd/src/scope/core/distributed/pp_contract.py` (`cache_epoch`, `do_kv_recompute`, explicit reset flags)
- `scope-drd/src/scope/core/pipelines/krea_realtime_video/blocks/recompute_kv_cache.py` (recompute path + coupling to decoded frames)
- `scope-drd/src/scope/server/frame_processor.py` (hard cuts / soft transitions, `kv_cache_attention_bias` override lifecycle)

## Synthesis

### Mental model

In streaming video inference, the KV cache is the generator’s **hidden state across chunks**. It is simultaneously:

- a **performance feature** (reuse past K/V instead of recomputing), and
- a **correctness contract** (if cache state diverges, everything after can look “plausible” but be wrong).

Scope’s KV cache is a fixed-size **ring buffer** over *tokens* (frames expand to `frame_seq_length` tokens). Each chunk appends new entries and attends over the active window; when it fills, it evicts old entries and may run a **recompute** pass to rebuild the relevant cache slice. (`scope-drd/notes/FA4/h200/tp/explainers/05-kv-cache-head-sharding.md`)

The key operational takeaway is: **KV cache management is not a local optimization knob**. It’s part of the distributed protocol:

- In **TP v0/v1.1**, correctness comes from lockstep execution: both ranks receive identical `call_params`, update cache indices deterministically, and never broadcast cache contents. (`scope-drd/notes/FA4/h200/tp/explainers/05-kv-cache-head-sharding.md`, `scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md`)
- In **PP bringup**, the generator KV cache lives on the mesh side, but rank0 still controls cache lifecycle via the envelope: hard cuts must become epoch-bounded (`cache_epoch`), in-flight work must be droppable, and recompute coupling must be handled explicitly (e.g., `context_frames_override`). (`scope-drd/notes/FA4/h200/tp/pp-topology-pilot-plan.md`, `scope-drd/notes/FA4/h200/tp/pp-next-steps.md`)

### Key concepts

- **Head-sharded KV cache (TP correctness + memory)**
  - The cache is allocated per layer as `[batch, max_tokens, heads, head_dim]` and sharded by heads in TP mode (e.g., 40 → 20 heads per rank at TP=2). Cache contents are never exchanged; only the post-attention projection needs an all_reduce. (`scope-drd/notes/FA4/h200/tp/explainers/05-kv-cache-head-sharding.md`)
  - Practical implication: the cache is *huge* even when sharded (order-of-10GB per rank at TP=2 for the full model), so “copy/migrate the cache” is not a viable baseline mechanism. (`scope-drd/notes/FA4/h200/tp/explainers/05-kv-cache-head-sharding.md`)

- **Ring-buffer indexing: `global_end_index` vs `local_end_index`**
  - `global_end_index` is the monotonic position in the global token timeline; `local_end_index` maps that into the fixed-size ring after evictions/rolls. The write window is derived deterministically from `call_params` (start offset + token count), so ranks stay consistent without extra coordination. (`scope-drd/notes/FA4/h200/tp/explainers/05-kv-cache-head-sharding.md`)

- **Cache lifecycle events are correctness-critical**
  - **Hard cut** (`init_cache=True`): resets cache state; must happen on all participating ranks/stages at the same logical boundary. (`scope-drd/notes/FA4/h200/tp/explainers/05-kv-cache-head-sharding.md`, `scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md`)
  - **Recompute**: when the cache fills, the system re-encodes recent context and rewrites the cache; this triggers another generator pass and therefore participates in the same lockstep/collective invariants as normal denoise. (`scope-drd/notes/FA4/h200/tp/explainers/05-kv-cache-head-sharding.md`, `scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md`)
  - **Soft transition / forgetting** (`kv_cache_attention_bias`): a scalar bias applied in attention; must be identical across ranks or you induce drift. (`scope-drd/notes/FA4/h200/tp/explainers/05-kv-cache-head-sharding.md`)

- **The recompute coupling problem (why “generator-only workers” is non-trivial)**
  - In steady state, the recompute anchor frame is derived from `decoded_frame_buffer` via a VAE re-encode path. If a worker/mesh stage does not perform decode (or cannot VAE-encode), it must be provided an override tensor (`context_frames_override` / `context_frames` in the PP envelope), or recompute semantics break. (`scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md`, `scope-drd/notes/FA4/h200/tp/pp-topology-pilot-plan.md`)
  - This is why TP v0 kept the full pipeline on workers, and why PP plans explicitly stage recompute as “disable first (R1), then restore via rank0-provided override (R0a).” (`scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md`, `scope-drd/notes/FA4/h200/tp/pp-next-steps.md`, `scope-drd/notes/FA4/h200/tp/pp0-bringup-runbook.md`)

### Cross-resource agreement / disagreement

- **External resources emphasize memory efficiency; Scope emphasizes lifecycle correctness first**
  - PagedAttention / vLLM-style systems focus on reducing KV memory waste from variable sequence lengths via paged allocation and block managers.
  - Scope’s immediate KV pain is different: cache **lifecycle** (hard cuts, eviction, recompute) and **distributed coupling** (TP lockstep; PP stage boundary + recompute anchor). Head-sharding + deterministic indices already solve “don’t broadcast cache” efficiently; the next frontier is making lifecycle safe under PP. (`scope-drd/notes/FA4/h200/tp/explainers/05-kv-cache-head-sharding.md`, `scope-drd/notes/FA4/h200/tp/pp-topology-pilot-plan.md`)

- **Internal docs align on the central coupling**
  - The failure-modes explainer calls out KV lifecycle divergence and recompute coupling as a primary Franken-model risk. (`scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md`)
  - The PP topology plan and next-steps doc treat recompute as the “decide early” coupling and explicitly propose R1 → R0a sequencing, plus `cache_epoch`-based invalidation on hard cuts. (`scope-drd/notes/FA4/h200/tp/pp-topology-pilot-plan.md`, `scope-drd/notes/FA4/h200/tp/pp-next-steps.md`)

- **Measured performance supports the PP motivation**
  - Block profiling shows `decode` + `recompute_kv_cache` are ~33% of measured GPU time per chunk in TP mode, motivating a topology where rank0 owns decode and mesh owns generator. (`scope-drd/notes/FA4/h200/tp/bringup-run-log.md`)
  - “Reduce recompute frequency” (`SCOPE_KV_CACHE_RECOMPUTE_EVERY=2`) was empirically a dead end: quality loss with no net FPS gain. (`scope-drd/notes/FA4/h200/tp/bringup-run-log.md`)

### Practical checklist

**Goal**: keep cache lifecycle deterministic and auditable as you move from TP v0 → TP v1.1 → PP0/PP1.

1. **Make cache lifecycle explicit in the control-plane plan**
   - Drive all cache events from per-chunk params/envelopes: `init_cache`, `reset_kv_cache`, `reset_crossattn_cache`, `do_kv_recompute`, `kv_cache_attention_bias`, and `expected_generator_calls`. (`scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md`, `scope-drd/notes/FA4/h200/tp/pp-topology-pilot-plan.md`)
   - Assert `observed_generator_calls == expected_generator_calls` so “recompute happened on one side but not the other” can’t silently slip through. (`scope-drd/notes/FA4/h200/tp/pp0-bringup-runbook.md`)

2. **Pin backend selection across ranks**
   - Do not use per-rank `auto` backend selection for KV-bias attention; pin `SCOPE_KV_BIAS_BACKEND` (e.g., `fa4` vs `flash`) and include it in env parity checks. (`scope-drd/notes/FA4/h200/tp/explainers/05-kv-cache-head-sharding.md`)

3. **Treat recompute as a first-class distributed feature (not an afterthought)**
   - For PP0 bringup, start in **R1** (disable recompute) to prove contracts/queues first: `SCOPE_KV_CACHE_RECOMPUTE_EVERY=999999`. (`scope-drd/notes/FA4/h200/tp/pp-next-steps.md`, `scope-drd/notes/FA4/h200/tp/pp0-bringup-runbook.md`)
   - To restore correctness semantics, move to **R0a**: rank0 must provide `context_frames_override` / `context_frames` when `do_kv_recompute=True`; mesh must use the override and never fall back to VAE re-encode. (`scope-drd/notes/FA4/h200/tp/pp-next-steps.md`, `scope-drd/notes/FA4/h200/tp/pp0-bringup-runbook.md`)

4. **In PP, couple hard cuts to epoching + queue flush**
   - On hard cut, flush bounded queues and increment `cache_epoch` so stale in-flight results are self-identifying and droppable. (`scope-drd/notes/FA4/h200/tp/pp-topology-pilot-plan.md`)

5. **Audit for out-of-band cache mutation**
   - Any cache reset or lifecycle mutation that bypasses the broadcast/envelope (e.g., rank0-only side effects from a server endpoint) is a direct Franken-model risk. (`scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md`)

### Gotchas and failure modes

- **“Just skip decode on workers/mesh” breaks recompute**
  - Without `decoded_frame_buffer`, steady-state recompute either crashes or uses different inputs → cache divergence → silent corruption. This is the core blocker for generator-only workers unless you supply override tensors. (`scope-drd/notes/FA4/h200/tp/explainers/05-kv-cache-head-sharding.md`, `scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md`, `scope-drd/notes/FA4/h200/tp/pp-topology-pilot-plan.md`)

- **Hard cuts without epoching are dangerous under PP**
  - If you reset cache state but accept late results from the previous cache world, you effectively splice two different histories into one output stream. PP uses `cache_epoch` specifically to avoid this class. (`scope-drd/notes/FA4/h200/tp/pp-topology-pilot-plan.md`)

- **Backend mismatch can look like “mysterious quality drift”**
  - Different attention backends (FP8 FA4 vs FP16 flash) can introduce small numeric differences that compound across chunks; treat backend choice as a contract, not a suggestion. (`scope-drd/notes/FA4/h200/tp/explainers/05-kv-cache-head-sharding.md`)

- **Recompute frequency is not a free knob**
  - `SCOPE_KV_CACHE_RECOMPUTE_EVERY=2` caused visible quality glitches and did not improve end-to-end FPS in measured runs; don’t treat it as a safe perf lever without quality validation. (`scope-drd/notes/FA4/h200/tp/bringup-run-log.md`)

- **Mesh “generator-only” requires careful block refactors**
  - Some cache-setup paths currently assume VAE presence (e.g., `SetupCachesBlock` clearing VAE cache). In true generator-only mesh, these must be made optional or moved to rank0. (`scope-drd/notes/FA4/h200/tp/pp-topology-pilot-plan.md`)

### Experiments to run

1. **PP0/R0a recompute override enforcement**
   - Enable recompute (`SCOPE_KV_CACHE_RECOMPUTE_EVERY=1`) and intentionally omit `context_frames` when `do_kv_recompute=True`.
   - Expected: envelope validation fails pre-send or mesh refuses to proceed; logs show “override required” and no silent fallback to VAE re-encode. (`scope-drd/notes/FA4/h200/tp/pp-next-steps.md`, `scope-drd/notes/FA4/h200/tp/pp0-bringup-runbook.md`)

2. **Hard cut + late result (epoch drop)**
   - Trigger a hard cut while an older chunk is in flight and force its result to arrive late.
   - Expected: rank0 drops by `cache_epoch` mismatch and does not decode/emit it. (`scope-drd/notes/FA4/h200/tp/pp-topology-pilot-plan.md`)

3. **Backend mismatch tripwire**
   - Intentionally run mismatched KV-bias backends across ranks (e.g., `flash` vs `fa4`) and confirm drift detectors (digest/fingerprint) or explicit env-parity checks catch it early. (`scope-drd/notes/FA4/h200/tp/explainers/05-kv-cache-head-sharding.md`, `scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md`)

4. **Recompute frequency regression test**
   - Re-run the “every=2” experiment and verify the observed behavior (quality loss, no net FPS gain) to prevent future “quick win” regressions. (`scope-drd/notes/FA4/h200/tp/bringup-run-log.md`)
