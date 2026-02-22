# Reading Guide — Bringup-Synced Study Order

> **This is a study guide, not the catalog.** The canonical exhaustive resource list is
> [`distributed_video_dit_inference.md`](../distributed_video_dit_inference.md) (24 topics, ~85 resources).
> This guide selects ~15 "spine" resources and sequences them against the actual PP bringup phases
> so you read what you need *when* you need it. For any topic where coverage here feels thin,
> the catalog has the full list.

Last updated: 2026-02-22

---

## 0) Crosswalks (when you're editing code, not “studying”)

- Working on the v1.1 “generator-only workers” scaffold? Start with `refs/v1.1-generator-only-workers-crosswalk.md` — section-by-section pointers into the topics/resources that matter *before* you change code.

## 1) The Spine (~15 load-bearing resources)

These are the beams. Everything else is lookup.

### A. Performance mental model (you're bandwidth-bound; act like it)

1. **Horace He — "Making Deep Learning Go Brrrr From First Principles"**
   Why: best single mental model for overhead- vs bandwidth- vs compute-bound regimes; makes "H200 FPS tracks HBM BW" feel obvious.

2. **Roofline guide (NERSC) + NVIDIA GPU Performance Background Guide**
   Why: one pedagogical roofline walkthrough (NERSC) + one "official NVIDIA definitions/counters" doc.

3. **Nsight Compute Roofline walkthrough**
   Why: fastest path from "I know roofline" → "I can run it on my kernels".

### B. GPU execution + overlap primitives (streams/events/graphs)

4. **CUDA Programming Guide: Asynchronous Execution + CUDA Graphs**
   Why: you will otherwise make bad assumptions about streams, dependencies, and why graph capture breaks.

5. **PyTorch CUDA Semantics + Zach DeVito caching allocator writeup**
   Why: this is the real story behind "reserved vs allocated", fragmentation, and why CUDA graph capture makes allocator behavior weird.

### C. NCCL + deadlocks + "multi-process-group reality"

6. **NCCL User Guide (env vars + group calls + CUDA graph capture + troubleshooting)**
   Why: your design has world_pg + mesh_pg + overlap. Group-call ordering rules are exactly the class of bug that yields "hangs only under load".

7. **NVIDIA blog: NCCL tuning (NCCL tests, NCCL_ALGO/NCCL_PROTO sweeps)**
   Why: practical knob-turning, and how to benchmark comm changes without self-gaslighting.

8. **Merlyn Wang — "NCCL Allreduce" (ring all-reduce intuition)**
   Why: if you don't have the ring pipeline in your head, you'll misdiagnose "why is this slow" when message sizes/topology change.

### D. torch.compile + distributed: what actually breaks graphs

9. **TorchDynamo deep dive + torch.compiler FAQ + `torch._dynamo.explain`**
   Why: prevents superstition. You need a crisp model of graph breaks, guards, and why functional collectives help.

10. **Inductor design / scheduling / fusion references**
    Pick one: Jason Ansel Inductor design doc OR the PyTorch 2 ASPLOS paper section.
    Why: you'll interpret profiler traces differently once you know what Inductor is trying to fuse and why.

11. **Edward Yang: "State of torch.compile" + "Ways to use torch.compile"**
    Why: this is where the real constraints + compiler/distributed interaction land.

12. **Functional collectives RFC + `_functional_collectives.py` source**
    Why: your own breakthrough depended on this (9.6 → 24.5 FPS); reading the RFC once pays for itself.

### E. Pipeline parallelism (only what you need for inference bringup)

13. **siboehm pipeline parallelism article + GPipe paper**
    Why: cleanest readable explanation + the canonical reference.

14. **Megatron pipeline schedules source (and/or scaling blog) + PyTorch `torch.distributed.pipelining` docs**
    Why: when you implement PP scheduling/queues, it helps to see how hardened code does it.

    Also load-bearing (not in the spine, but in the library and worth reading when you hit scheduling decisions): **PipeDream-2BW** (1F1B schedule), **Zero Bubble PP** (F/B/W backward split), **PipeDiT** (pipelined sequence parallelism for DiT).

### F. Message contracts + replay (PPEnvelopeV1 / PPResultV1)

15. **DDIA (schema evolution + dataflow) + Patterns of Distributed Systems (Idempotent Receiver, etc.)**
    Why: you're designing a protocol boundary; these prevent "cute contracts that collapse later".

---

## 2) Bringup-synced reading order (read → build → debug loop)

This matches your actual phases. Don't read everything up front; read just enough to unblock the next bringup step.

### Phase PP0 — contracts work, no overlap, no TP

**Read (minimum):**
- PyTorch distributed tutorial (send/recv, isend/irecv, process group basics)
- NCCL User Guide: **group calls** + troubleshooting sections
- PyTorch CUDA semantics (streams/events mental model)

**Build:**
- rank0 ↔ mesh leader P2P envelope/result transport
- versioned envelopes (PPEnvelopeV1 / PPResultV1), bounded queue stubs (D_IN/D_OUT)
- "hard cut" epoch handling (drop stale results)

**Debug checklist:**
- mismatched recv sizes / dtype / device
- wrong process group used for a collective
- rank0 accidentally participates in mesh collectives (→ hang)
- implicit sync from `tensor.cpu()` or print debugging

---

### Phase PP0 + overlap — non-blocking comms and real concurrency

**Read (minimum):**
- CUDA async execution (streams, events, sync primitives)
- PyTorch CUDA semantics: stream semantics + "streams and NCCL" implications

**Build:**
- non-blocking send/recv + bounded queues (pressure relief, not throughput magic)
- overlap rank0 work (VAE + control) with mesh work (generator)

**Debug checklist:**
- you *must* reason about where tensors live and which stream produced them
- add explicit events where you cross stream boundaries (rank0 side especially)
- avoid hidden sync: `.item()`, host prints, accidental `.to("cpu")`

---

### Phase PP0 + recompute — KV cache / rolling context plumbing

**Read (minimum):**
- CausVid paper (causal DiT framing + cache/recompute)
- StreamDiffusionV2 paper (rolling caches + Stream Batch motivation)
- vLLM PagedAttention paper/blog for cache-as-memory-management analogy (optional but helpful)

**Build:**
- reinstate KV cache recompute via `context_frames_override`
- check determinism and drift under "hard cut" resets

---

### Phase PP1 — TP only inside the mesh (rank0 excluded), compile survives

**Read (minimum):**
- If you're touching the v1.1 “generator-only workers” path: skim `refs/v1.1-generator-only-workers-crosswalk.md` first (it’s the fastest way to rehydrate the invariants).
- Functional collectives RFC + implementation
- Dynamo deep dive + FAQ (graph breaks, guards, explain)
- DTensor / TP tutorial (if migrating toward DeviceMesh/DTensor idioms)

**Build:**
- mesh_pg collectives only; rank0 never touches them
- envelope broadcast from mesh leader, TP all-reduces inside mesh_pg
- keep graphs intact: avoid in-place comm ops that reintroduce graph breaks

**Debug checklist:**
- ensure *all ranks in mesh_pg* execute the same collective ordering
- explicitly test `TORCH_LOGS="graph_breaks"` and keep a graph-break budget
- measure funcol overhead; validate you still net-win after compile

---

## 3) Topic index (24 topics → best picks)

Below: **first pick** (spine resource or closest) + **also useful** for each topic. Every Phase 1 manifest resource is named at least once. For the full resource list per topic, see the [canonical catalog](../distributed_video_dit_inference.md).

### Distributed systems fundamentals
1. **NCCL internals** → NCCL User Guide; NCCL tuning blog; Merlyn "NCCL allreduce"
2. **Deadlock patterns** → NCCL group-call ordering + troubleshooting; PyTorch distributed docs. Also: PyTorch issue #167775 (Ctrl+C handling), Lei Mao "Kill Distributed Processes"
3. **Graceful shutdown** → `destroy_process_group` docs. Also: PyTorch issue #115388 (destroy_process_group hangs after CUDA graph capture — directly relevant to your architecture)
4. **Determinism** → PyTorch reproducibility notes; `torch.use_deterministic_algorithms`. Also: FP non-associativity paper (2408.05148)

### GPU execution model
5. **CUDA streams** → CUDA async execution; PyTorch CUDA semantics
6. **CUDA graphs** → CUDA graphs guide; PyTorch CUDA graphs docs; **cuda-graphs-guide** (library resource). Also: NCCL with CUDA graphs doc
7. **GPU memory mgmt** → PyTorch CUDA semantics (allocator); Zach DeVito allocator post
8. **Kernel launch overhead** → Horace He "brrr" (**making-dl-go-brrrr**, library resource); CUDA graph best practices

### torch.compile / Inductor
9. **Dynamo tracing** → **dynamo-deep-dive** (library resource); UW PLSE TorchDynamo explainer; FAQ + `torch._dynamo.explain`
10. **Inductor fusion** → Inductor design doc; Inductor config/scheduler sources. Also: PyTorch 2 ASPLOS paper
11. **Functional collectives** → **funcol-rfc-93173** (library resource) + source; **ezyang-state-of-compile** (library resource). Also: PyTorch issue #138773 (funcol 67% slower — real overhead you experienced)
12. **Compile + distributed** → compiled autograd tutorial; DTensor TP tutorial; vLLM torch.compile blog

### Pipeline parallelism
13. **Classic PP** → siboehm article; **gpipe** (library resource); **pipedream-2bw** (library resource, 1F1B schedule)
14. **Activation memory** → GPipe; **pytorch-pipelining-api** (library resource); Megatron PTD-P paper (2104.04473, interleaved/virtual stages)
15. **Scheduling theory** → **zero-bubble-pp** (library resource, F/B/W split); bubble fraction derivations (JAX Scaling Book). Also: **pipedit** (library resource, pipelined sequence parallelism for DiT)

### Performance analysis
16. **Roofline for transformers** → NERSC roofline guide; NVIDIA GPU perf guide; Horace He "brrr"
17. **Profiler traces** → PyTorch profiler recipe; HTA "trace analysis for the masses"; GPU MODE profiling lecture
18. **Bandwidth accounting** → NVIDIA GPU perf guide; Horace He matmul-shapes post; Nsight Compute counters

### Systems patterns
19. **Backpressure** → bounded queues (your design) + Jeff Hodges notes + WarpStream rejection post
20. **Message framing/versioning** → proto3 guide; FlatBuffers evolution rules; Stephen Cleary framing post
21. **Idempotency/replay** → Fowler "Idempotent Receiver"; DDIA chapters on retries/dataflow

### Domain-specific: video DiT inference
22. **KV cache in streaming** → CausVid; **pagedattention** (library resource); vLLM "anatomy" blog
23. **VAE latency/chunking** → Diffusers CogVideoX VAE docs; LightX2V VAE system; StreamDiffusionV2 Stream-VAE
24. **Video DiT scheduling** → **streamdiffusionv2** (library resource); StreamDiffusion; **pipedit** (library resource); **dit-paper** (library resource, foundational architecture)

---

## 4) "Don't step on rakes" (specific to your architecture)

These are the mistakes that eat days.

- **Multi-PG ordering:** if any rank executes collectives in a different order (or on the wrong PG), you can hang without useful errors.
- **Hidden synchronization:** any host readback in the hot path (`.item()`, `.cpu()`) murders overlap and can deadlock if it forces stream sync at the wrong time.
- **Graph-capture vs shutdown:** CUDA graph capture + NCCL communicator destruction has known sharp edges (issue #115388). Plan shutdown deliberately, not as an afterthought.
- **Funcol overhead is real:** functional collectives can be slower than in-place c10d (issue #138773); you're "buying" traceability, and you need compile to pay you back.

---

## 5) Resources to evaluate for manifest

These appeared in the integrated learning path but are **not currently in the manifest**. Evaluate whether they should be added as Phase 2 (or promoted to Phase 1 if load-bearing for bringup).

### Papers (arxiv — directly relevant)

| arxiv ID | Paper | Why it appeared |
|---|---|---|
| 2503.20314 | **Wan 2.1** — the target model | Referenced in PipeDiT bibliography; the canonical catalog never links to it |
| 2412.07772 | **CausVid** — causal DiT framing + KV cache/recompute | Mentioned by name in bringup phases; URL found in StreamDiffusionV2 source |
| 2511.07399 | **StreamDiffusionV2 arxiv paper** | Catalog has project page + GitHub but not the arxiv paper |

### Blog posts / docs (supplemental)

| Resource | Why |
|---|---|
| Merlyn Wang "NCCL Allreduce" | Ring all-reduce intuition; fills gap in NCCL topic |
| Zach DeVito caching allocator writeup | Deeper than PyTorch docs on allocator internals |
| NERSC roofline guide | Pedagogical roofline walkthrough (complements NVIDIA's denser doc) |
| HTA "trace analysis for the masses" | Profiling tooling |

### Books / reference patterns

| Resource | Why |
|---|---|
| DDIA (Designing Data-Intensive Applications) | Schema evolution, dataflow — relevant to PPEnvelope versioning |
| Patterns of Distributed Systems (Fowler) | Idempotent Receiver pattern directly relevant |
| FlatBuffers evolution rules | Alternative to proto3 for schema evolution |

These are candidates, not automatic additions. The papers (especially Wan 2.1 and CausVid) are the strongest candidates for Phase 1 promotion.
