# StreamDiffusionV2

| Field | Value |
|-------|-------|
| Source | https://arxiv.org/abs/2511.07399, https://streamdiffusionv2.github.io/, https://github.com/chenfengxu714/StreamDiffusionV2 |
| Type | paper |
| Topics | 24 |
| Authors | Tianrui Feng et al. |
| Year | 2025 |
| Status | condensed |

## Why it matters

The closest published + open-source system to our target: real-time, interactive streaming inference for a causal video DiT (Wan2.1 + CausVid-style causal conversion), including multi-GPU scaling under strict latency SLOs. It combines SLO-aware batching, pipeline-parallel "Stream Batch" scheduling, sink-token rolling KV cache + RoPE refresh for long-horizon stability, motion-aware noise control, and a dynamic DiT block scheduler; it reports **58.28 FPS** for a **14B** model on **4×H100** without TensorRT/quantization. (See also: `scope-drd/notes/FA4/h200/tp/streamdiffusion-v2-analysis.md` for how this maps to our H200 TP→PP roadmap.)

## Core claims

1. **Claim**: StreamDiffusionV2 hits interactive latency + throughput in practice: TTFF **0.47s** (16 FPS) / **0.37s** (30 FPS) for 1.3B, and multi-GPU throughput up to **58.28 FPS** (14B, 512×512) / **39.24 FPS** (14B, 480p) on **4×H100** (bf16, no TensorRT/quantization). It attributes the 1.3B vs 14B throughput gap being smaller than expected partly to the shared VAE weights, with VAE ≈ **30%** of end-to-end inference time.
   **Evidence**: [sources/streamdiffusionv2/full-paper.md#521-ttff-results](../../sources/streamdiffusionv2/full-paper.md#521-ttff-results), [sources/streamdiffusionv2/full-paper.md#522-fps-results](../../sources/streamdiffusionv2/full-paper.md#522-fps-results), [sources/streamdiffusionv2/full-paper.md#51-setup](../../sources/streamdiffusionv2/full-paper.md#51-setup)

2. **Claim**: Fixed large-chunk streaming inputs cannot meet live SLOs: they model TTFF as \(\mathrm{TTFF} \approx \frac{2 B T H W P_{\mathrm{model}}}{C_{\mathrm{device}}\rho_{\mathrm{VAE}}}\) and estimate ~**5.31s** TTFF for a 1.3B model at 480p when using an 81-frame chunk on a single H100.
   **Evidence**: [sources/streamdiffusionv2/full-paper.md#3-motivation--bottleneck-analysis](../../sources/streamdiffusionv2/full-paper.md#3-motivation--bottleneck-analysis)

3. **Claim**: Their SLO-aware batching scheduler treats inference as memory-bound and adapts stream-batch size \(B\) to hit a target per-stream rate \(f_{\mathrm{SLO}}\): \(L(T,B) \approx \frac{A(T,B)+P_{\mathrm{model}}}{\eta\,\mathrm{BW}_{\mathrm{HBM}}}\) with \(A(T,B)=\mathcal{O}(BT)\) (FlashAttention), implying \(f \propto \frac{B}{1+B}\) and an optimal \(B^*\) near the roofline knee.
   **Evidence**: [sources/streamdiffusionv2/full-paper.md#41-real-time-scheduling-and-quality-control](../../sources/streamdiffusionv2/full-paper.md#41-real-time-scheduling-and-quality-control)

4. **Claim**: StreamDiffusionV2's multi-GPU scaling comes from **pipeline orchestration**: partition DiT blocks across devices (pipeline-parallel ring) and extend the batching model by treating the \(n\) denoising steps as an "effective batch multiplier" (latency model \(L(T,nB)\)).
   **Evidence**: [sources/streamdiffusionv2/full-paper.md#42-scalable-pipeline-orchestration](../../sources/streamdiffusionv2/full-paper.md#42-scalable-pipeline-orchestration)

5. **Claim**: "Stream Batch" (keeping multiple denoising micro-steps in flight) improves throughput, and its benefit grows with more denoising steps (deeper in-flight pipeline).
   **Evidence**: [sources/streamdiffusionv2/full-paper.md#543-sequence-parallelism-vs-pipeline-orchestration](../../sources/streamdiffusionv2/full-paper.md#543-sequence-parallelism-vs-pipeline-orchestration), [sources/streamdiffusionv2/full.md#file-causvidmodelswancausal_stream_inferencepy](../../sources/streamdiffusionv2/full.md#file-causvidmodelswancausal_stream_inferencepy)

6. **Claim**: Compared to sequence-parallel attention approaches (DeepSpeed-Ulysses, Ring-Attention), their pipeline orchestration has far lower communication overhead: SP incurs ~**40–120 ms** cross-device latency, about **20–40×** higher than their approach; SP also only helps at high resolutions and can become memory-bound at moderate/low resolutions.
   **Evidence**: [sources/streamdiffusionv2/full-paper.md#543-sequence-parallelism-vs-pipeline-orchestration](../../sources/streamdiffusionv2/full-paper.md#543-sequence-parallelism-vs-pipeline-orchestration), [sources/streamdiffusionv2/full-paper.md#3-motivation--bottleneck-analysis](../../sources/streamdiffusionv2/full-paper.md#3-motivation--bottleneck-analysis)

7. **Claim**: Long-horizon stability is addressed by adaptive sink tokens + RoPE refresh: sink tokens are refreshed based on cosine similarity (threshold \(\tau\)), and RoPE phase is periodically reset after a threshold \(T_{\mathrm{reset}}\) to prevent positional drift; the repo implements sink-token–preserving rolling KV-cache updates with an (optional) similarity-based sink refresh.
   **Evidence**: [sources/streamdiffusionv2/full-paper.md#41-real-time-scheduling-and-quality-control](../../sources/streamdiffusionv2/full-paper.md#41-real-time-scheduling-and-quality-control), [sources/streamdiffusionv2/full.md#file-causvidmodelswancausal_modelpy](../../sources/streamdiffusionv2/full.md#file-causvidmodelswancausal_modelpy)

8. **Claim**: A motion-aware noise controller adapts the noise/noising schedule to motion magnitude (frame-difference metric + normalization + EMA smoothing); in ablations, sink tokens and dynamic noising improve pixel-level temporal consistency (lower Warp Error) and together yield the best CLIP/Warp metrics.
   **Evidence**: [sources/streamdiffusionv2/full-paper.md#41-real-time-scheduling-and-quality-control](../../sources/streamdiffusionv2/full-paper.md#41-real-time-scheduling-and-quality-control), [sources/streamdiffusionv2/full-paper.md#53-generation-quality-evaluation](../../sources/streamdiffusionv2/full-paper.md#53-generation-quality-evaluation), [sources/streamdiffusionv2/full.md#file-streamv2vinferencepy](../../sources/streamdiffusionv2/full.md#file-streamv2vinferencepy)

9. **Claim**: A dynamic DiT block scheduler mitigates pipeline stalls from stage imbalance (edge ranks also doing VAE encode/decode): it reallocates blocks based on measured execution times; the repo implements contiguous block-interval recomputation plus KV-cache migration when ownership changes.
   **Evidence**: [sources/streamdiffusionv2/full-paper.md#43-efficient-system-algorithm-co-design](../../sources/streamdiffusionv2/full-paper.md#43-efficient-system-algorithm-co-design), [sources/streamdiffusionv2/full-paper.md#542-effectiveness-of-the-dynamic-dit-block-scheduler](../../sources/streamdiffusionv2/full-paper.md#542-effectiveness-of-the-dynamic-dit-block-scheduler), [sources/streamdiffusionv2/full.md#file-streamv2vinference_pipepy](../../sources/streamdiffusionv2/full.md#file-streamv2vinference_pipepy), [sources/streamdiffusionv2/full.md#file-streamv2vcommunicationkv_cache_managerpy](../../sources/streamdiffusionv2/full.md#file-streamv2vcommunicationkv_cache_managerpy)

10. **Claim**: Communication is designed to be overlapped: each GPU uses separate compute + communication CUDA streams, runs P2P transfers asynchronously, and bounds in-flight sends (max outstanding) to avoid unbounded buffering.
    **Evidence**: [sources/streamdiffusionv2/full-paper.md#43-efficient-system-algorithm-co-design](../../sources/streamdiffusionv2/full-paper.md#43-efficient-system-algorithm-co-design), [sources/streamdiffusionv2/full.md#file-streamv2vinference_pipepy](../../sources/streamdiffusionv2/full.md#file-streamv2vinference_pipepy), [sources/streamdiffusionv2/full.md#file-streamv2vcommunicationdistributed_communicatorpy](../../sources/streamdiffusionv2/full.md#file-streamv2vcommunicationdistributed_communicatorpy)

## Key technical details

### SLO + latency model (paper)

- **TTFF scaling under fixed chunking**: \(\mathrm{TTFF} \propto BTHWP_{\mathrm{model}}\) (Section 3) and is the motivating reason to keep chunk sizes small and adapt batching.
  See: [sources/streamdiffusionv2/full-paper.md#3-motivation--bottleneck-analysis](../../sources/streamdiffusionv2/full-paper.md#3-motivation--bottleneck-analysis).
- **Memory-bound latency approximation for scheduling**: \(L(T,B) \approx \frac{A(T,B)+P_{\mathrm{model}}}{\eta\,\mathrm{BW}_{\mathrm{HBM}}}\), \(A(T,B)=\mathcal{O}(BT)\) (FlashAttention), and achieved frequency \(f \propto \frac{B}{1+B}\).
  See: [sources/streamdiffusionv2/full-paper.md#41-real-time-scheduling-and-quality-control](../../sources/streamdiffusionv2/full-paper.md#41-real-time-scheduling-and-quality-control).

### Stream Batch (denoising-steps-as-batch) mechanics (repo)

- In `CausalStreamInferencePipeline.prepare()`, the implementation sets `batch_size = len(denoising_step_list)` and **repeats** KV + cross-attn caches across that batch; it pre-allocates `hidden_states` and per-step `kv_cache_starts/ends`.
  See: [sources/streamdiffusionv2/full.md#file-causvidmodelswancausal_stream_inferencepy](../../sources/streamdiffusionv2/full.md#file-causvidmodelswancausal_stream_inferencepy).
- In `inference_stream()`, it shifts a step-pipeline by copying `[1:] <- [:-1]`, inserts the newest noise at slot 0, runs the generator on the whole step-batch, then re-noises intermediate slots to advance the step ladder — yielding a "clean latent" every micro-step (matches Fig. 6 conceptually).
  See: [sources/streamdiffusionv2/full.md#file-causvidmodelswancausal_stream_inferencepy](../../sources/streamdiffusionv2/full.md#file-causvidmodelswancausal_stream_inferencepy).

### Sink-token rolling KV cache (paper + repo)

- **Policy (paper)**: maintain a sink set \(\mathcal{S}_t\) and refresh sinks via cosine similarity threshold \(\tau\); periodically reset RoPE phase after \(T_{\mathrm{reset}}\).
  See: [sources/streamdiffusionv2/full-paper.md#41-real-time-scheduling-and-quality-control](../../sources/streamdiffusionv2/full-paper.md#41-real-time-scheduling-and-quality-control).
- **Mechanism (repo)**: per-block KV cache includes `k`, `v`, and end indices; on overflow it **rolls** the cache by evicting old tokens while preserving `sink_tokens = sink_size * frame_seqlen`, and calls `flash_attn_with_kvcache(...)` with `cache_seqlens`. The causal attention path optionally refreshes sink coverage when cosine similarity against the oldest sink region drops below `adapt_sink_thr`.
  See: [sources/streamdiffusionv2/full.md#file-causvidmodelswancausal_modelpy](../../sources/streamdiffusionv2/full.md#file-causvidmodelswancausal_modelpy), [sources/streamdiffusionv2/full.md#file-causvidmodelswancausal_stream_inferencepy](../../sources/streamdiffusionv2/full.md#file-causvidmodelswancausal_stream_inferencepy).

### Dynamic DiT block scheduler + KV migration (paper + repo)

- **Motivation (paper)**: VAE encode/decode on edge ranks induces imbalance; dynamic scheduling reallocates blocks to minimize per-stage latency.
  See: [sources/streamdiffusionv2/full-paper.md#542-effectiveness-of-the-dynamic-dit-block-scheduler](../../sources/streamdiffusionv2/full-paper.md#542-effectiveness-of-the-dynamic-dit-block-scheduler).
- **Implementation (repo)**: `compute_balanced_split(total_blocks, rank_times, dit_times, current_block_nums)` computes new contiguous `[start,end)` intervals; `KVCacheManager.rebalance_kv_cache_by_diff(...)` broadcasts KV blocks whose owner changes; ranks then offload KV for non-owned blocks to CPU.
  See: [sources/streamdiffusionv2/full.md#file-streamv2vcommunicationutilspy](../../sources/streamdiffusionv2/full.md#file-streamv2vcommunicationutilspy), [sources/streamdiffusionv2/full.md#file-streamv2vcommunicationkv_cache_managerpy](../../sources/streamdiffusionv2/full.md#file-streamv2vcommunicationkv_cache_managerpy), [sources/streamdiffusionv2/full.md#file-streamv2vinference_pipepy](../../sources/streamdiffusionv2/full.md#file-streamv2vinference_pipepy).

### Key repo entry points (JIT lookup)

- **Multi-GPU inference demo**: `streamv2v/inference_pipe.py` (pipeline stages, async send/recv, optional `--schedule_block`).
  See: [sources/streamdiffusionv2/full.md#file-streamv2vinference_pipepy](../../sources/streamdiffusionv2/full.md#file-streamv2vinference_pipepy).
- **Comm abstractions**: `streamv2v/communication/*` (tags + headers, buffer reuse, KV-cache rebalance).
  See: [sources/streamdiffusionv2/full.md#file-streamv2vcommunicationdistributed_communicatorpy](../../sources/streamdiffusionv2/full.md#file-streamv2vcommunicationdistributed_communicatorpy), [sources/streamdiffusionv2/full.md#file-streamv2vcommunicationbuffer_managerpy](../../sources/streamdiffusionv2/full.md#file-streamv2vcommunicationbuffer_managerpy).
- **Causal streaming pipeline**: `causvid/models/wan/causal_stream_inference.py` (Stream Batch and block-mode input/middle/output).
  See: [sources/streamdiffusionv2/full.md#file-causvidmodelswancausal_stream_inferencepy](../../sources/streamdiffusionv2/full.md#file-causvidmodelswancausal_stream_inferencepy).

## Actionables / gotchas

- **PP alone is not a single-stream latency win**: PP across blocks doesn't help at batch=1 unless you keep multiple micro-steps/chunks in flight; this matches our roadmap framing of "single-stream latency vs node throughput."
  See: `scope-drd/notes/FA4/h200/tp/streamdiffusion-v2-analysis.md`, `scope-drd/notes/FA4/h200/tp/streamdiffusion-v2-analysis-5pro.md`; [Stream Batch discussion](../../sources/streamdiffusionv2/full-paper.md#543-sequence-parallelism-vs-pipeline-orchestration).
- **Stage imbalance is the first-order obstacle for PP**: they explicitly call out VAE on edge stages; in our measurements, decode + KV-cache recompute are ~33% of wall time in the uncompiled TP baseline, and the "generator-only workers / rank specialization" scaffold is the Scope analog of their fix.
  See: `scope-drd/notes/FA4/h200/tp/streamdiffusion-v2-analysis-opus.md` ("Dynamic DiT Block Scheduler (Figure 13)"), `scope-drd/notes/FA4/h200/tp/bringup-run-log.md` (Run 10b), `scope-drd/notes/FA4/h200/tp/v1.1-generator-only-workers.md`, [paper block scheduler](../../sources/streamdiffusionv2/full-paper.md#542-effectiveness-of-the-dynamic-dit-block-scheduler).
- **Broadcast conditioning to every PP stage (adaLN-Zero coupling)**: DiT blocks consume timestep/text conditioning at every block, so PP stages must receive the same `conditioning_embeds` tensor every chunk (or replicate the conditioning MLP on each stage). Treat this as part of the stage-boundary envelope contract.
  See: `scope-drd/notes/FA4/h200/tp/pp-topology-pilot-plan.md` (Prompt encoding / conditioning broadcast), `scope-drd/notes/FA4/h200/tp/explainers/03-broadcast-envelope.md` (conditioning_embeds in broadcast envelope), [DiT conditioning via adaLN-Zero](../../sources/dit-paper/full.md#32-diffusion-transformer-design-space).
- **If you move blocks, you must move KV**: their repo's block scheduler isn't just "change block intervals" — it includes KV-cache ownership transfer (`KVCacheManager.rebalance_kv_cache_by_diff`). Any future "dynamic partition" in our PP mesh needs a KV migration story (or a hard "no repartition after warmup" constraint).
  See: [sources/streamdiffusionv2/full.md#file-streamv2vcommunicationkv_cache_managerpy](../../sources/streamdiffusionv2/full.md#file-streamv2vcommunicationkv_cache_managerpy), `scope-drd/notes/FA4/h200/tp/pp-topology-pilot-plan.md` (cache epoch + contracts).
- **Treat sink-token refresh + RoPE reset as part of the cache lifecycle contract**: for long-horizon streaming, these are not "model internals" — they are stateful cache policies that must be consistent across stages/ranks (especially with PP). In Scope terms, this should line up with `cache_epoch` and explicit cache-reset decisions in `PPEnvelopeV1`.
  See: [paper sink/RoPE policy](../../sources/streamdiffusionv2/full-paper.md#41-real-time-scheduling-and-quality-control), [repo sink-token KV](../../sources/streamdiffusionv2/full.md#file-causvidmodelswancausal_modelpy), `scope-drd/notes/FA4/h200/tp/pp-topology-pilot-plan.md`.
- **Motion-aware noise control is a concrete lever for high-motion failure modes**: the paper's ablation shows Warp Error gains; the repo implements a lightweight frame-difference heuristic (`compute_noise_scale_and_step`) that maps to our "rank0 computes knobs, workers execute generator" pattern. If adopted, include `current_step`/noise controls in the stage boundary envelope to preserve lockstep determinism.
  See: [paper motion-aware noise + ablation](../../sources/streamdiffusionv2/full-paper.md#541-effectiveness-of-sink-token-and-motion-aware-noise-controller), [repo heuristic](../../sources/streamdiffusionv2/full.md#file-streamv2vinferencepy), `scope-drd/notes/FA4/h200/tp/v1.1-generator-only-workers.md` (envelope completeness rules).
- **Use overlap only when you can prove it**: their design uses separate compute/comm streams; our PP pilot already defines an "OverlapScore" instrumentation plan. Don't assume overlap exists — measure it and gate on it.
  See: `scope-drd/notes/FA4/h200/tp/streamdiffusion-v2-analysis-opus.md` ("Async Communication Overlap (Figure 16)"), `scope-drd/notes/FA4/h200/tp/streamdiffusion-v2-analysis.md` ("Two CUDA streams overlap pattern"), [paper async overlap](../../sources/streamdiffusionv2/full-paper.md#43-efficient-system-algorithm-co-design), [paper execution timeline](../../sources/streamdiffusionv2/full-paper.md#c-figure-illustration), `scope-drd/notes/FA4/h200/tp/pp0-bringup-runbook.md` (OverlapScore), `scope-drd/notes/FA4/h200/tp/pp-control-plane-pseudocode.md`.
- **Interpret their "sequence parallelism is bad" result carefully**: their comparison is about attention SP (Ulysses/Ring), not hidden-dim TP. For us, NVLink all-reduces can be sub-ms but the *count* of collectives and compile graph-break hygiene still matters; use the same "measure-first" discipline before committing to a topology jump.
  As a concrete comm-cost anchor (their Figure 5 summary at 480p / NVLink H100): ~**2–5 ms** for their PP approach vs ~**60 ms** for DeepSpeed-Ulysses at 2 GPUs (and ~**5–10 ms** vs ~**120 ms** at 4 GPUs).
  See: `scope-drd/notes/FA4/h200/tp/streamdiffusion-v2-analysis-opus.md` ("Communication Cost (Figure 5)"), `scope-drd/notes/FA4/h200/tp/streamdiffusion-v2-analysis-5pro.md` (sync-points framing), `scope-drd/notes/FA4/h200/tp/feasibility.md` (collective census + timings), [paper comm-cost framing](../../sources/streamdiffusionv2/full-paper.md#543-sequence-parallelism-vs-pipeline-orchestration).
- **Step count changes the scheduling regime**: Stream Batch benefits increase with more denoising steps (deeper in-flight pipeline), but our product may want 1–2 steps for latency. Treat step count as a first-class control in PP bringup experiments.
  See: [paper Stream Batch note](../../sources/streamdiffusionv2/full-paper.md#543-sequence-parallelism-vs-pipeline-orchestration), [repo `--step` handling](../../sources/streamdiffusionv2/full.md#file-streamv2vinference_pipepy).

## Related resources

- [dit-paper](dit-paper.md) -- DiT architecture / block structure (PP-friendly uniform stack)
- [gpipe](gpipe.md) -- classic PP utilization + bubble fraction; Stream Batch is the "keep PP full" story
- [pipedream-2bw](pipedream-2bw.md) -- PP scheduling theory (training), useful context for future interleaving
- [pipedit](pipedit.md) -- alternative PP approach (patch/sequence pipelining) for DiT inference
- [pagedattention](pagedattention.md) -- memory management relevant to attention KV caches
- [pytorch-pipelining-api](pytorch-pipelining-api.md) -- PyTorch pipeline schedules (future adoption option)
- [nccl-user-guide](nccl-user-guide.md) -- stream semantics + gotchas when overlapping comm/compute
- [making-dl-go-brrrr](making-dl-go-brrrr.md) -- roofline/overhead framing for throughput vs latency decisions
