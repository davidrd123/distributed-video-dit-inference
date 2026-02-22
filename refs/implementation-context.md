# Implementation Context: Working Notes → Reference Library

This document maps from **measured findings and design decisions** in the Scope working notes (`scope-drd/notes/FA4/h200/tp/`) to **which reference library resources are load-bearing** and why. It is the orientation document for agent JIT lookup during implementation.

For the full working notes inventory, see `scope-drd/notes/FA4/h200/tp/session-state.md`.
For the reference library structure, see `CLAUDE.md`.

---

## System summary

Scope serves a **Wan2.1 14B video DiT** for real-time interactive streaming. The generator is a 40-layer transformer (dim=5120, 40 heads). Current deployment: 2×H200 GPUs (NVLink, ~450 GB/s bidirectional).

| Milestone | FPS | What unlocked it | Working notes reference |
|---|---|---|---|
| Single GPU + compile | ~20 | FA4 + torch.compile | baseline |
| TP=2, no compile | 16 | First working multi-GPU | Run 7, `bringup-run-log.md` |
| TP=2, FA4 backend | 19.5 | Correct KV-bias attention kernel | Run 10 |
| TP=2, FA4 + compile | **24.5** | Functional collectives fix graph breaks | Run 12b |
| TP=4, FA4 + compile | 27 | Just works, no code changes | Run 17 |

---

## Phase 1: TP v0 (done, protecting baseline)

### Finding: 160 collectives per forward, ~9ms overhead

**Source**: `feasibility.md` (Section 2), NCCL microbenchmarks in `outputs/h200/2026-02-19/tp_comm/`

Per chunk: 80 large all-reduces (O-proj + FFN-down, 2/block × 40 blocks) + 80 tiny all-reduces (distributed RMSNorm Q/K, 2/block × 40 blocks). Each large all-reduce on `[1, 2160, 5120]` BF16 is ~0.113ms on NVLink.

**Load-bearing resources**:

| Resource | Why | Card section to sharpen |
|---|---|---|
| `nccl-user-guide` | Algorithm selection (ring vs tree vs NVLS), `NCCL_ALGO`/`NCCL_PROTO` tuning, stream semantics for overlapping collectives | Actionables: cite the 0.113ms measurement and explain which algorithm NCCL auto-selects for 21 MiB all-reduce on NVLink |
| `scaling-dl-nccl` (medium priority) | NCCL architecture/topology overview — useful for translating our ~0.113ms / ~195 GB/s per-rank all-reduce on NVLink into expectations for PCIe and multi-node | Actionables: tie the 0.113ms measurement to topology/protocol choices and a “what to check first” tuning checklist |
| `nccl-tuning` (medium priority) | Deep dive into NCCL protocol selection (LL vs Simple) and tuning knobs; relevant if the 21 MiB all-reduce regresses vs the measured 0.113ms baseline | Actionables: list the minimal `NCCL_*` knobs to try when collective time regresses (and how to validate) |
| `pytorch-cuda-semantics` | CUDA stream ordering guarantees — the reason collectives must be issued on the correct stream for compile safety | Actionables: cite the compile-stream interaction from Run 12b |

**Relevant topics**: `01-nccl-internals`, `05-cuda-streams`

---

### Finding: FA4 KV-bias backend was the real bottleneck (16 → 19.5 FPS)

**Source**: `bringup-run-log.md` (Run 10), `feasibility.md` (Phase 2C benchmarks)

The 4 FPS gap between TP=2 (16 FPS, flash backend) and single-GPU (20 FPS) was primarily the attention backend, not collective overhead. FA4 score_mod with CUTLASS eliminated most of the gap.

**Load-bearing resources**:

| Resource | Why | Card section to sharpen |
|---|---|---|
| `flash-attention` (medium priority, not Phase 1) | Score_mod mechanism, FP8 attention accumulation, the arithmetic that makes FA4 faster than FlashAttention 3 for KV-biased attention | — |

**Relevant topics**: `06-cuda-graphs` (tangential), `22-kv-cache-management`

---

### Finding: Functional collectives fixed compile (9.6 → 24.5 FPS)

**Source**: `bringup-run-log.md` (Runs 8-9b diagnosis, Run 12b fix)

`torch._dynamo.disable()` on collective functions caused ~160 graph breaks per forward, fragmenting each block into tiny subgraphs. Switching to `torch.distributed._functional_collectives` (return new tensors, traceable by Inductor) eliminated all collective graph breaks.

**Load-bearing resources**:

| Resource | Why | Card section to sharpen |
|---|---|---|
| `funcol-rfc-93173` | The design RFC that defines the API: `all_reduce` returns a new tensor, `wait_tensor` for synchronization. The exact mechanism used in Run 12b | Actionables: cite the 9.6→24.5 FPS delta as evidence of functional collectives' real-world impact; note the eager-mode regression trap (Run 12a: funcol in eager was slower due to extra allocation) |
| `funcol-source` (medium priority) | Code-level reference for `AsyncCollectiveTensor` and `wait_tensor`; helpful when auditing our exact Run 12b `all_reduce`+wait pattern and stream semantics | Actionables: include the exact call pattern + note the eager/compile split (avoid extra allocation in eager) |
| `pytorch-issue-138773` (medium priority) | Documents cases where functional collectives can be much slower than in-place c10d — matches the Run 12a eager regression trap | Actionables: cite Run 12a and record how we avoid funcol in eager mode; track upstream fixes |
| `vllm-torch-compile` (medium priority) | Production-oriented writeup of torch.compile + tensor parallel with functional collectives; useful cross-check now that Run 12b is the baseline | Actionables: cross-check our debug workflow (`TORCH_LOGS`, graph break counters) against their recommendations |
| `ezyang-state-of-compile` | Broader context on compile + distributed interaction; where functional collectives fit in the torch.compile story | Actionables: cite the graph break census from Runs 13-14 (`unique_graphs=12-14, graph_breaks=2`) |
| `dynamo-deep-dive` | How Dynamo traces through Python, why `dynamo.disable()` causes graph breaks, what "graph break" means for dispatch overhead | Actionables: cite the 160 graph breaks diagnosis from Run 9b |

**Relevant topics**: `11-functional-collectives`, `12-compile-distributed-interaction`, `09-dynamo-tracing`

---

### Finding: Remaining graph break is KV-cache dynamic slicing

**Source**: `bringup-run-log.md` (Runs 13-14, Known Issue 8), `research-program.md`

`kv_cache["k"][:, local_start_index:local_end_index]` uses Tensor bounds, which Dynamo can't trace without `CAPTURE_SCALAR_OUTPUTS`. A `narrow()`/SymInt rewrite attempt produced *more* breaks and was reverted.

**Load-bearing resources**:

| Resource | Why | Card section to sharpen |
|---|---|---|
| `dynamo-deep-dive` | SymInt semantics, data-dependent guards, why `narrow()` with SymInt bounds can fail | Actionables: cite the KV-cache slicing break specifically |

---

## Phase 2: v1.1 Generator-Only Workers (scaffolded, verified safe)

### Finding: decode + recompute = 33% of wall time

**Source**: `bringup-run-log.md` (Run 10b block profile)

| Block | GPU ms/call | % |
|---|---|---|
| denoise | 435ms | 66.8% |
| decode | 107ms | 16.5% |
| recompute_kv_cache | 104ms | 16.0% |

Workers currently run the full pipeline (including decode) because recompute depends on `decoded_frame_buffer`. v1.1 broadcasts the minimum tensor set from rank0 so workers run only the generator.

**Load-bearing resources**:

| Resource | Why | Card section to sharpen |
|---|---|---|
| `making-dl-go-brrrr` | The roofline framing: decode is not TP-sharded, so it's pure overhead on worker ranks. Arithmetic intensity argument for why sharding compute doesn't help bandwidth-bound ops | Actionables: cite the 33% wall-time figure and explain why this is "overhead-bound" not "compute-bound" in the three-regime framework |
| `streamdiffusionv2` | Their dynamic block scheduler solves the same class of problem (VAE imbalance on edge ranks). Figure 13 is the warning | Actionables: cite the v1.1 scaffold as the Scope equivalent of their stage imbalance fix |
| `lightx2v-vae` (medium priority) | Streaming/chunked video VAE decode with temporal caching patterns; directly relevant to shrinking the 107ms decode slice (Run 10b) | Actionables: map their cache/tiling ideas to Scope’s decode path and propose a measurement plan |
| `cogvideox-vae` (medium priority) | Practical decode levers (tiling/slicing) for trading memory for latency; candidate if decode stays ≥10% of wall time | Actionables: tie to Run 10b decode ms and document a tiling/slicing experiment grid |
| `seedance` (medium priority) | Thin decoder + distributed VAE ideas; relevant if decode becomes a dedicated “Stage 0” bottleneck (rank0 imbalance) | Actionables: connect to the v1.1/PP motivation and define a “decode ms target” to justify deeper VAE work |

**Relevant topics**: `16-roofline-model`, `23-vae-latency-chunking`

---

## Phase 3: PP Topology (contracts defined, bringup planned)

### Design: rank0-out-of-mesh, Stage 0 / Stage 1 split

**Source**: `pp-topology-pilot-plan.md`, `pp-next-steps.md` (Steps A1-A5), `pp-control-plane-pseudocode.md`, `pp0-bringup-runbook.md`

Stage 0 (rank0): encode + decode + control plane. Stage 1 (mesh ranks 1..N): generator-only, TP collectives inside `mesh_pg`. Communication is one `PPEnvelopeV1` per chunk (p2p, overlapped with compute), vs 160 all-reduces per chunk in TP.

**Load-bearing resources**:

| Resource | Why | Card section to sharpen |
|---|---|---|
| `gpipe` | The foundational PP reference: micro-batching, bubble fraction `(P-1)/M`, synchronous scheduling. The "before" for understanding what Stream Batch improves | Actionables: cite the B/(B+P-1) utilization formula and note that PP without batching at B=1 gives 50% utilization (2 stages) |
| `pipedream-2bw` | 1F1B schedule, activation memory reduction. Relevant if the PP pilot evolves toward interleaved scheduling | Actionables: note that inference-only PP doesn't need backward scheduling, but the activation memory argument still applies to recompute |
| `zero-bubble-pp` | Further scheduling optimization. Less immediately relevant but establishes the state-of-the-art for PP scheduling | — |
| `pytorch-pipelining-api` | `torch.distributed.pipelining` — the PyTorch-native API for PP schedules. May be adoptable if the pilot script proves the topology | Actionables: cite the PP0 pilot plan (Steps A1-A5) as the bringup path that would feed into this API |
| `streamdiffusionv2` | The most directly relevant system: PP + Stream Batch on Wan2.1, 58 FPS on 4×H100. Their code in `streamv2v/` is the reference implementation for distributed inference with KV-cache management | Actionables: cite the PP comm cost comparison (Figure 5: PP ~2-5ms vs TP-like ~60ms at 2 GPUs), the overlap instrumentation (Figure 16), and the stage imbalance fix (Figure 13) |
| `pipedit` | Pipelined sequence parallelism for DiT — complementary approach. Useful for understanding the design space beyond pure layer-parallel PP | — |

**Relevant topics**: `13-classic-pipeline-parallelism`, `14-activation-memory-in-pp`, `15-pipeline-scheduling-theory`, `19-producer-consumer-backpressure`, `20-message-framing-versioning`, `24-video-dit-scheduling`

---

### Design: PPEnvelopeV1 / PPResultV1 contracts

**Source**: `pp-next-steps.md` (Section "Recent contract changes"), `pp-control-plane-pseudocode.md`

The stage boundary protocol: `PPAction` enum (NOOP/INFER/SHUTDOWN), `call_id` (globally monotonic), `chunk_index` (INFER-only monotonic), `cache_epoch` (increments on hard cut), `validate_before_send()`.

**Load-bearing resources**:

| Resource | Why | Card section to sharpen |
|---|---|---|
| `pagedattention` | Paged memory management for KV caches — relevant to the `cache_epoch` lifecycle and the ring-buffer eviction/recompute pattern | Actionables: cite the KV-cache lifecycle events (hard cut, recompute, soft transition) from explainer 05 |
| `pytorch-distributed-api` (medium priority) | Defines p2p/collective APIs, process groups, and timeout semantics used by `PPControlPlane` (anti-stranding ordering, `SCOPE_DIST_TIMEOUT_S` bringup) | Actionables: cite the bringup timeout choice (e.g. 60s) and document default timeout differences across backends |

**Relevant topics**: `20-message-framing-versioning`, `21-idempotency-and-replay`, `22-kv-cache-management`

---

### Design: Overlap instrumentation (how to prove PP is working)

**Source**: `pp-control-plane-pseudocode.md` (Section "Instrumentation"), `pp0-bringup-runbook.md` (Phase 2)

Overlap is proven by throughput period approaching `max(Stage0_ms, Stage1_ms)` rather than their sum. The `OverlapScore` metric: `median(hidden_k / min(stage0_k, stage1_k))`, pass gate ≥ 0.30.

**Load-bearing resources**:

| Resource | Why | Card section to sharpen |
|---|---|---|
| `making-dl-go-brrrr` | The "three regimes" framework (overhead/bandwidth/compute bound) directly applies to diagnosing whether PP overlap is working or whether you're in the "overhead-bound" regime where dispatch cost dominates | Actionables: cite the overlap instrumentation metrics |
| `profiling-torch-compile` (medium priority) | Practical guide for attributing time to compiled regions and separating Python overhead vs kernel time — needed when validating overlap and dispatch cost | Actionables: add a minimal profiler trace checklist for PP bringup (what events to look for) |
| `pytorch-profiler-tensorboard` (medium priority) | Reference for using TensorBoard traces to inspect GPU utilization/idle gaps; helpful for confirming `t_mesh_idle_ms` is small | Actionables: document the exact trace views to use for overlap validation |
| `cuda-async-execution` (medium priority) | Authoritative CUDA reference for stream/event ordering; needed if PP adds a dedicated comm stream and explicit waits (StreamDiffusionV2 pattern) | Actionables: spell out the event/wait rules we must follow when overlapping comm + compute |

**Relevant topics**: `16-roofline-model`, `17-reading-profiler-traces`, `18-bandwidth-accounting`

---

## Cross-cutting: DiT architecture

### Finding: 40 uniform blocks, no skip connections, adaLN-Zero conditioning

**Source**: `dit-paper` (already condensed), `explainers/01-why-two-gpus.md`, `explainers/04-tp-math.md`

The DiT's uniform block structure is what makes both TP and PP tractable: every block has the same shape (Q/K/V projection, attention, O-projection, FFN), so TP sharding and PP partitioning are straightforward. No skip connections means activation memory is O(1) per stage (no need to hold earlier activations for later layers). adaLN-Zero conditioning is injected at every block, which means PP stages need the conditioning embeddings broadcast to all stages.

**Load-bearing resources**:

| Resource | Why | Card section to sharpen |
|---|---|---|
| `dit-paper` | The architecture spec: model configs (layers, dim, heads, Gflops), adaLN-Zero mechanism, patchify pipeline. Already condensed with 7 claims and 8 actionables | Already done — verify actionables mention the PP-friendly uniform structure |
| `streamdiffusionv2` | Extends DiT to causal streaming with rolling KV-cache, sink tokens, and motion-aware noise | Actionables: cite the adaLN-Zero conditioning broadcast requirement for PP |

---

## Resource priority for PP bringup (recommended extraction order)

If PP bringup (Steps A1-A5) is the next implementation push, these are the most load-bearing resources in order of urgency:

1. **`streamdiffusionv2`** — reference implementation for distributed video DiT inference with PP + Stream Batch. Tier 2 + 3 needed. The `streamv2v/` code (already fetched) contains the distributed inference pipeline, KV-cache manager, and communication layer.

2. **`gpipe`** — foundational PP concept (micro-batching, bubble fraction). Tier 2 + 3 needed.

3. **`funcol-rfc-93173`** — functional collectives design. Critical context for the compile-aware collectives that will be used inside `mesh_pg` in PP1. Tier 2 + 3 needed.

4. **`pytorch-pipelining-api`** — the PyTorch-native PP scheduling API. May be adoptable once the topology is proven. Tier 2 + 3 needed.

5. **`pipedream-2bw`** — 1F1B scheduling, activation memory. Less urgent for inference-only PP but relevant if the system evolves. Tier 2 + 3 needed.

Resources already condensed that inform PP:
- `dit-paper` (architecture spec, uniform blocks)
- `making-dl-go-brrrr` (roofline analysis, three regimes)

---

## Per-card actionables sharpening guide

When completing Tier 3 for each Phase 1 resource, use this table to write targeted actionables instead of generic ones:

| Resource ID | Specific finding to cite | Working notes source |
|---|---|---|
| `nccl-user-guide` | 0.113ms per all-reduce on NVLink for 21 MiB tensor; algorithm auto-selection for this size | `feasibility.md` Section 1 |
| `pytorch-cuda-semantics` | Stream ordering for compile-aware collectives; why functional collectives need correct stream placement | Run 12b, `explainers/06-failure-modes.md` Q6 |
| `funcol-rfc-93173` | 9.6→24.5 FPS from eliminating dynamo.disable graph breaks; eager-mode regression trap (Run 12a) | `bringup-run-log.md` Runs 8-12b |
| `ezyang-state-of-compile` | graph_breaks=2 steady state, unique_graphs=12-14; the compile + distributed "functional collectives" story | Runs 13-14, `research-program.md` |
| `dynamo-deep-dive` | 160 graph breaks from dynamo.disable; KV-cache dynamic slicing as remaining break; SymInt/narrow attempt failure | Runs 9b, 13-14, Known Issue 8 |
| `cuda-graphs-guide` | CUDAGraph capture unstable/neutral on H200 (v0 roadmap item v2.0) | `feasibility.md` Section 3.4 |
| `gpipe` | B/(B+P-1) utilization; PP without batching at B=1 → 50% utilization for 2 stages | `streamdiffusion-v2-analysis.md`, `pp-topology-pilot-plan.md` |
| `pipedream-2bw` | Inference-only PP doesn't need backward scheduling but activation memory argument applies to recompute | `pp-topology-pilot-plan.md` recompute coupling |
| `zero-bubble-pp` | State-of-the-art PP scheduling; context for future optimization | — |
| `pytorch-pipelining-api` | Potential adoption path after PP0 pilot proves the topology | `pp-next-steps.md` Step A5 |
| `pagedattention` | KV-cache lifecycle: hard cut, recompute, eviction, ring-buffer semantics | `explainers/05-kv-cache-head-sharding.md` |
| `making-dl-go-brrrr` | 33% wall-time in decode+recompute; overhead-bound vs compute-bound diagnosis; overlap instrumentation | Run 10b, `pp0-bringup-runbook.md` Phase 2 |
| `streamdiffusionv2` | PP comm cost (Figure 5), overlap pattern (Figure 16), stage imbalance (Figure 13), Stream Batch fill requirement | all three analysis docs, `pp-topology-pilot-plan.md` |
| `pipedit` | Pipelined sequence parallelism design space; complementary to layer-parallel PP | — |
| `dit-paper` | Uniform block structure enables PP; adaLN-Zero requires conditioning at every stage; Gflops scale quadratically with tokens | `explainers/01-why-two-gpus.md`, `explainers/04-tp-math.md` |
