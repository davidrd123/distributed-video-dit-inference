# Distributed video DiT inference: a curated learning path

**This is a structured reading and watching list for building a pipeline-parallel inference system for a video Diffusion Transformer, organized across 24 topics spanning distributed systems, GPU execution, compiler internals, pipeline parallelism, performance analysis, systems patterns, and video generation.** Every resource was selected for depth and direct relevance to the engineering task at hand. Where possible, exact URLs are provided — papers link to arxiv, blog posts to their original sites, and documentation to specific pages.

---

## Distributed systems fundamentals

### 1. NCCL internals — ring vs tree algorithms, NVLink topology, GPU-level all-reduce/send/recv

NCCL's algorithm selection (ring, tree, NVLS, CollNet) is governed by message size, topology, and the `NCCL_ALGO` environment variable. Understanding this layer is essential for diagnosing why your pipeline send/recv calls behave differently across NVLink vs PCIe topologies.

- **NCCL User Guide** — covers communicators, collective operations, CUDA stream semantics, group calls, and algorithm selection. The environment variables page documents `NCCL_DEBUG`, `NCCL_ALGO`, and `NCCL_PROTO` for controlling and debugging algorithm choice.
  - https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html
  - Environment variables: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html

- **"Scaling Deep Learning Training with NCCL"** (NVIDIA Developer Blog) — architecture overview of multi-GPU/multi-node communication, NVLink/PCIe/InfiniBand support, and low-latency protocols.
  - https://developer.nvidia.com/blog/scaling-deep-learning-training-nccl/

- **"Understanding NCCL Tuning to Accelerate GPU-to-GPU Communication"** (NVIDIA Developer Blog) — CTA allocation, protocol selection (LL vs Simple), tuner plugins. The closest public documentation to NCCL's internal algorithm decisions.
  - https://developer.nvidia.com/blog/understanding-nccl-tuning-to-accelerate-gpu-to-gpu-communication/

- **Using NCCL with CUDA Graphs** — requirements and patterns for capturing NCCL operations inside CUDA graphs, which you will need for pipeline-parallel inference.
  - https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cudagraph.html

### 2. Deadlock patterns in multi-group distributed code

Deadlocks in multi-group code almost always stem from mismatched collective ordering across ranks or accidentally issuing operations on the wrong process group. `NCCL_DEBUG=INFO` (or `WARN`) is your primary diagnostic tool.

- **PyTorch Distributed API Reference** — documents `init_process_group`, process group creation, backend selection, and timeout semantics. The timeout (default 10 minutes for NCCL) is critical to understand for deadlock diagnosis.
  - https://docs.pytorch.org/docs/stable/distributed.html

- **"Writing Distributed Applications with PyTorch"** (tutorial) — hands-on walkthrough of point-to-point and collective communication, backends comparison, and process group setup. Good for building intuition about ordering constraints.
  - https://docs.pytorch.org/tutorials/intermediate/dist_tuto.html

- **"Building Scalable and Fault-Tolerant NCCL Applications"** (NVIDIA Developer Blog) — communicator management, dynamic rescaling, fault tolerance. Covers what happens when a rank fails mid-collective.
  - https://developer.nvidia.com/blog/building-scalable-and-fault-tolerant-nccl-applications

### 3. Graceful shutdown and draining in distributed PyTorch

Graceful shutdown in distributed PyTorch remains **underserved by documentation**. The core API is `destroy_process_group()`, but real-world challenges include ranks exiting at different times, CUDA graph capture preventing clean NCCL communicator destruction, and signal handling under `torchrun`.

- **PyTorch `destroy_process_group()` documentation** — within the distributed API reference. Call once near end of main(), blocking until all processes reach it.
  - https://docs.pytorch.org/docs/stable/distributed.html

- **GitHub Issue #115388: `destroy_process_group()` hangs after CUDA graph capture** — documents that `ncclCommDestroy` hangs when collectives were captured into CUDA graphs. Critical for inference workloads.
  - https://github.com/pytorch/pytorch/issues/115388

- **GitHub Issue #167775: Graceful Ctrl+C handling from torchrun** — active discussion on multi-GPU shutdown coordination under signal handling.
  - https://github.com/pytorch/pytorch/issues/167775

- **"Kill PyTorch Distributed Training Processes"** (Lei Mao) — practical patterns for fully releasing resources in distributed training, multi-node termination.
  - https://leimao.github.io/blog/Kill-PyTorch-Distributed-Training-Processes/

### 4. Determinism across ranks in distributed training/inference

Non-determinism in distributed settings comes from three sources: **CUDA atomicAdd operations** (non-associative floating-point), **cuDNN autotuning** selecting different algorithms per run, and **NCCL reduction order** across ranks.

- **PyTorch Reproducibility Guide** — the authoritative reference. Covers manual seeding, `torch.use_deterministic_algorithms()`, `cudnn.deterministic`, and lists all affected operations.
  - https://docs.pytorch.org/docs/stable/notes/randomness.html

- **`torch.use_deterministic_algorithms` API** — complete list of operations with deterministic overrides, including Inductor no-autotune mode and Triton config selection.
  - https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html

- **"Impacts of floating-point non-associativity on reproducibility for HPC and deep learning"** (arXiv) — rigorous study documenting how inter-chip communication introduces variation in distributed settings.
  - https://arxiv.org/abs/2408.05148

---

## GPU execution model

### 5. CUDA streams — execution/dependency model, events, synchronization, NCCL interaction

Streams are the fundamental concurrency primitive for overlapping compute, communication, and memory transfers. NCCL operations execute on their own streams, and understanding how events synchronize across streams is essential for pipeline parallelism.

- **PyTorch CUDA Semantics** — comprehensive treatment of streams, events, the caching allocator, and CUDA graphs. The sections on stream synchronization semantics and backward pass stream behavior are especially important.
  - https://docs.pytorch.org/docs/stable/notes/cuda.html

- **CUDA Programming Guide: Asynchronous Execution** — the authoritative NVIDIA reference on stream ordering, events, callbacks, and synchronization primitives.
  - https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html

- **"CUDA Stream"** (Lei Mao) — clear, practical explanation of serial vs concurrent stream models, overlapping compute with memory copy, and stream synchronization patterns with code examples.
  - https://leimao.github.io/blog/CUDA-Stream/

### 6. CUDA graphs — what they capture, what breaks them, relation to torch.compile

CUDA graphs eliminate kernel launch overhead by recording a sequence of GPU operations and replaying them. They are **critical for inference latency** but impose strict constraints: all tensor addresses must be fixed, control flow cannot change, and memory allocation patterns must be static.

- **CUDA Programming Guide: CUDA Graphs** — complete coverage of graph definition vs execution, stream capture, memory allocations within graphs, instantiation, and dependency management.
  - https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html

- **PyTorch CUDA Semantics (CUDA Graphs section)** — how PyTorch integrates CUDA graphs, graph-private memory pools, interaction with the caching allocator, and `torch.cuda.CUDAGraph` API.
  - https://docs.pytorch.org/docs/stable/notes/cuda.html

- **`torch.compile` with `mode="reduce-overhead"`** — uses CUDA graphs under the hood. The `torch.compile` API documentation describes the `triton.cudagraphs` option and related configuration.
  - https://docs.pytorch.org/docs/stable/generated/torch.compile.html

- **NVIDIA CUDA Graph Best Practices for NCCL** — requirements for capturing NCCL operations in CUDA graphs, with Megatron-LM's `CudaGraphManager` as a reference implementation.
  - https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cudagraph.html

### 7. GPU memory management — PyTorch caching allocator, fragmentation

The PyTorch caching allocator avoids expensive `cudaMalloc`/`cudaFree` calls by maintaining a pool of allocated blocks. Fragmentation under dynamic workloads (variable sequence lengths, different denoising steps) is the primary operational concern.

- **PyTorch CUDA Semantics: Memory Management** — documents `PYTORCH_CUDA_ALLOC_CONF` options (`max_split_size_mb`, `backend`, `expandable_segments`), memory snapshots, and the allocator's block-splitting behavior.
  - https://docs.pytorch.org/docs/stable/notes/cuda.html

- **`torch.cuda` Module Reference** — API for `memory_allocated()`, `memory_reserved()`, `memory_stats()`, `MemPool`, and the CUDA Sanitizer for detecting stream-ordered access errors.
  - https://docs.pytorch.org/docs/stable/cuda.html

### 8. Kernel launch overhead — Python-to-GPU dispatch path

Each PyTorch operator call traverses Python → C++ dispatch → CUDA kernel launch. At **~10μs per launch**, this overhead dominates when running many small operations. `torch.compile` and CUDA graphs are the primary mitigations.

- **"Making Deep Learning Go Brrrr From First Principles"** (Horace He) — the definitive blog post explaining overhead-bound, memory-bound, and compute-bound regimes. The overhead section explains exactly why Python dispatch matters and how operator fusion eliminates it.
  - https://horace.io/brrr_intro.html

- **GPU MODE Lecture 6: "Optimizing Optimizers in PyTorch"** (Jane Xu) — covers kernel launch overhead, kernel fusion, and multi-tensor apply as a mitigation. Available on the GPU MODE channel.
  - Channel: https://www.youtube.com/@GPUMODE

- **"PyTorch internals"** (Edward Yang) — deep dive into PyTorch's dispatch mechanism, tensor storage, and C++ code layout. Essential for understanding the full Python-to-GPU path.
  - https://blog.ezyang.com/2019/05/pytorch-internals/

---

## torch.compile and Inductor

### 9. How Dynamo tracing works — graph breaks, guards, eager fallback

TorchDynamo uses CPython's **PEP 523 frame evaluation API** to intercept Python bytecode execution, symbolically trace tensor operations into an FX graph, and generate guard functions that check whether cached compilations remain valid. When it encounters untraceable code (data-dependent control flow, unsupported Python constructs), it inserts a **graph break** — splitting the code into multiple compiled subgraphs with eager Python between them.

- **Dynamo Deep-Dive** (PyTorch docs) — the most comprehensive official resource on internals: PEP 523, VariableTracker system, guard generation, continuation functions at graph breaks, SymInt for dynamic shapes.
  - https://docs.pytorch.org/docs/stable/torch.compiler_dynamo_deepdive.html

- **"How does torch.compile work?"** (UW PLSE Blog, Megan Frisella) — excellent pedagogical explanation with clear diagrams of the TorchDynamo architecture, graph extraction, and guard functions.
  - https://uwplse.org/2025/04/28/torchdynamo.html

- **"torch.compile: the missing manual"** (Edward Yang, Google Doc) — hard-won debugging knowledge from Meta's PT2 deployments. Covers real-world graph break patterns, recompilation debugging, and distributed pitfalls.
  - https://docs.google.com/document/d/1y5CRfMLdwEoF1nTk9q8qEu1mgMUuUtvhklPKJ2emLU8/edit

- **`torch.compile` Programming Model** (PyTorch docs) — graph breaks, guards, recompilations, dynamic shapes, and debugging with `TORCH_LOGS`.
  - https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.html

### 10. Inductor fusion rules — what fuses, verification

TorchInductor's scheduler decides fusion using `score_fusion(node1, node2)`, which scores pairs of operations by **estimated memory traffic savings**. Pointwise-to-pointwise fusion is most common; reduction and template fusions have additional constraints.

- **"TorchInductor: a PyTorch-native Compiler with Define-by-Run IR and Symbolic Shapes"** (Jason Ansel, dev-discuss) — the foundational Inductor design document. Covers loop-level IR, TensorBox/StorageBox, Triton and C++ backends, and the breadth-first fusion approach.
  - https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747

- **TorchInductor `config.py`** (GitHub source) — the definitive reference for all fusion-related knobs: `epilogue_fusion`, `prologue_fusion`, `pattern_matcher`, `score_fusion_memory_threshold`, `benchmark_harness`.
  - https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py

- **"PyTorch 2: Faster Machine Learning Through Dynamic Python"** (ASPLOS 2024 paper) — Section on Inductor covers lowering (433 operators), `Scheduler.score_fusion`, persistent vs loop reductions in Triton codegen, and template epilogue fusions.
  - https://docs.pytorch.org/assets/pytorch2-2.pdf

- **Inductor scheduler source and fusion discussion** — Jason Ansel clarifies: "Fusions always happen in the scheduler" via `can_fuse` heuristics. Points to `scheduler.py` for flush boundaries.
  - https://dev-discuss.pytorch.org/t/disabling-codegen-specific-fusions-in-torchinductor-for-per-op-kernel-generation/3226

### 11. Functional collectives and Dynamo — why in-place breaks tracing, the funcol solution

Standard NCCL collectives (`all_reduce`, `all_gather`) are **in-place and side-effecting** — they mutate tensors and return opaque `Work` objects. This is fundamentally incompatible with functional graph tracing. Functional collectives (`torch.distributed._functional_collectives`) return new tensors and use `AsyncCollectiveTensor` subclasses for deferred synchronization, making them traceable by Dynamo.

- **RFC #93173: PT2-Friendly Traceable, Functional Collective Communication APIs** — the foundational design document. Explains why existing c10d APIs break tracing, defines the functional semantics, and specifies the `AsyncTensor` subclass approach.
  - https://github.com/pytorch/pytorch/issues/93173

- **`_functional_collectives.py` source** — the implementation. Shows `AsyncCollectiveTensor`, `traceable_collective_remaps`, `_c10d_functional` op registration, and `wait_tensor` synchronization.
  - https://github.com/pytorch/pytorch/blob/main/torch/distributed/_functional_collectives.py

- **"State of torch.compile for training (August 2025)"** (Edward Yang) — the most comprehensive treatment of functional collectives in context. Covers DTensor compilation, SimpleFSDP, async tensor parallelism as a compiler pass, and honest comparison with JAX's approach.
  - https://blog.ezyang.com/2025/08/state-of-torch-compile-august-2025/

- **GitHub Issue #138773: functional collectives 67% slower than torch.distributed** — important performance analysis showing the CPU overhead from extra copies and wrapping required for traceability. Includes benchmark code and profiler traces.
  - https://github.com/pytorch/pytorch/issues/138773

### 12. Compile + distributed interaction — compile with DDP/FSDP/TP

The core challenge is that **DDP/FSDP use backward hooks for communication**, and these hooks create graph breaks in AOTAutograd. The solutions are: (1) **Compiled Autograd** (PyTorch 2.4+), which captures the full backward graph at runtime, (2) **FSDP2 built on DTensor**, which is compile-friendly by design, and (3) **functional collectives** for tensor parallelism.

- **Compiled Autograd Tutorial** — key resource. Shows how `compiled_autograd = True` unifies backward graphs despite forward graph breaks, enabling DDP/FSDP communication to be captured and optimized.
  - https://docs.pytorch.org/tutorials/intermediate/compiled_autograd_tutorial.html

- **"Ways to use torch.compile"** (Edward Yang) — practical analysis covering small/medium/large scale training, load-bearing compilation, torchtitan/torchtune as showcase libraries, and when compilation's complexity budget is worth it.
  - https://blog.ezyang.com/2024/11/ways-to-use-torch-compile/

- **torch.compiler FAQ: Distributed Section** — explains why distributed code is hard for Dynamo, the AOTAutograd hook problem, and step-by-step debugging approach (`backend="eager"` → `"aot_eager"` → full compile).
  - https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_faq.html

- **Large Scale Transformer Training with Tensor Parallel** (PyTorch tutorial) — demonstrates `ColwiseParallel`, `RowwiseParallel`, `SequenceParallel` using DTensor and DeviceMesh. DTensor enables torch.compile compatibility for TP.
  - https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html

- **"Introduction to torch.compile and How It Works with vLLM"** (vLLM blog) — practical case study of compile + tensor parallelism with functional collectives in a production inference system.
  - https://blog.vllm.ai/2025/08/20/torch-compile.html

---

## Pipeline parallelism

### 13. Classic PP (GPipe, PipeDream) — micro-batching, 1F1B schedule, bubble fraction

Pipeline parallelism partitions a model across devices by layer. **GPipe** fills the pipeline with micro-batches and synchronizes at the end (high memory, simple). **PipeDream's 1F1B schedule** interleaves one forward and one backward per micro-batch, reducing peak activation memory from O(num_microbatches) to O(num_stages).

- **GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism** (Huang et al., 2018) — the foundational synchronous PP paper. Introduces micro-batch splitting and re-materialization (activation checkpointing) for memory efficiency.
  - https://arxiv.org/abs/1811.06965

- **PipeDream: Generalized Pipeline Parallelism for DNN Training** (Narayanan et al., 2019) — introduces the **1F1B schedule** and asynchronous weight updates with weight stashing. Up to 5x faster than data parallelism.
  - https://arxiv.org/abs/1806.03377

- **Memory-Efficient Pipeline-Parallel DNN Training (PipeDream-2BW)** (Narayanan et al., 2020) — introduces PipeDream-Flush (synchronous 1F1B) and double-buffered weight updates. The 1F1B schedule from this paper is what most modern systems actually implement.
  - https://arxiv.org/abs/2006.09503

- **"Pipeline-Parallelism: Distributed Training via Model Partitioning"** (Simon Boehm) — excellent educational blog post with clear diagrams covering naive PP, GPipe, PipeDream, pipeline bubbles, and gradient accumulation analysis.
  - https://siboehm.com/articles/22/pipeline-parallel-training

### 14. Activation memory in PP

Peak activation memory in PP depends on the schedule. GPipe stores activations for all micro-batches; 1F1B limits in-flight activations to `num_stages`. **Activation checkpointing** trades compute for memory by recomputing activations during backward.

- **Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM** (Narayanan et al., 2021) — the PTD-P paper. Proposes interleaved pipeline parallelism (virtual stages) that reduces bubble fraction from `(P-1)/(P-1+B)` to `(P-1)/(P-1+V·B)` at the cost of more communication. Detailed activation memory analysis.
  - https://arxiv.org/abs/2104.04473

- **Zero Bubble Pipeline Parallelism** (Qi et al., 2024) — achieves zero pipeline bubbles by splitting backward into B (input gradient) and W (parameter gradient) phases. Up to **23–31% throughput improvement** over 1F1B. The key insight is that W has no data dependency on the next micro-batch.
  - https://arxiv.org/abs/2401.10241

- **PyTorch `torch.distributed.pipelining` API** — documents `ScheduleGPipe`, `Schedule1F1B`, `ScheduleInterleaved1F1B`, `ScheduleInterleavedZeroBubble`, and `ScheduleZBVZeroBubble`. The code is your best reference for how these schedules manage activation memory.
  - https://docs.pytorch.org/docs/stable/distributed.pipelining.html

### 15. Pipeline scheduling theory — bubble fraction derivation, Little's law connection

The bubble fraction for a P-stage pipeline processing B micro-batches is **(P−1)/(B+P−1)**. This is a direct consequence of pipeline startup and drain latency. **Little's law** (L = λW) provides the framework: to keep P stages busy, you need at least P micro-batches in flight, and throughput approaches the ideal rate as B → ∞.

- **"How to Parallelize a Transformer for Training"** (JAX Scaling Book) — excellent treatment of all parallelism strategies with roofline analysis for communication bottlenecks. Covers pipeline bubble derivations and micro-batching tradeoffs.
  - https://jax-ml.github.io/scaling-book/training/

- **"Scaling Language Model Training to a Trillion Parameters Using Megatron"** (NVIDIA Developer Blog) — 3D parallelism analysis with interleaved pipeline scheduling, throughput measurements, and practical bubble fraction reduction.
  - https://developer.nvidia.com/blog/scaling-language-model-training-to-a-trillion-parameters-using-megatron/

- **Megatron-LM Pipeline Parallel Schedules source** — the reference implementation. `forward_backward_pipelining_with_interleaving` shows exactly how 1F1B and interleaved schedules manage micro-batch ordering.
  - https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/pipeline_parallel/schedules.py

---

## Performance analysis

### 16. Roofline model applied to transformers — arithmetic intensity for attention vs FFN

The roofline model plots achievable FLOPS against **arithmetic intensity** (FLOPS/byte of memory traffic). Attention is typically **memory-bandwidth-bound** (low arithmetic intensity due to reading/writing large KV matrices), while FFN layers are more likely **compute-bound** (high arithmetic intensity from large matrix multiplications). This distinction drives optimization strategy.

- **"Making Deep Learning Go Brrrr From First Principles"** (Horace He) — the single best resource for building roofline intuition. Explains overhead-bound, bandwidth-bound, and compute-bound regimes with concrete GPU examples. The starting point for any GPU performance reasoning.
  - https://horace.io/brrr_intro.html

- **"Roofline: An Insightful Visual Performance Model for Multicore Architectures"** (Williams, Waterman, Patterson, 2009) — the original paper. Defines the roofline model tying floating-point performance, operational intensity, and memory bandwidth.
  - https://dl.acm.org/doi/10.1145/1498765.1498785

- **NVIDIA GPU Performance Background User's Guide** — the canonical NVIDIA reference. Defines arithmetic intensity = ops/bytes, shows V100 ridge point at ~40–139 ops/byte, covers SM architecture, wave quantization, and tail effects.
  - https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html

- **GPU MODE Lecture 8: "CUDA Performance Checklist"** (Mark Saroufim) — covers roofline model concepts, memory coalescing, occupancy, and Nsight Compute profiling. Practical GPU performance reasoning.
  - Channel: https://www.youtube.com/@GPUMODE

### 17. Reading Nsight / torch.profiler traces

- **PyTorch Profiler with TensorBoard** (tutorial) — TensorBoard plugin integration with GPU utilization views, memory curves, kernel view, and operator view.
  - https://docs.pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html

- **Profiling torch.compile performance** — how to profile compiled models, identify `CompiledFunction` events, Triton kernels, and torch-compiled regions in traces.
  - https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_profiling_torch_compile.html

- **GPU MODE Lecture 1: "How to Profile CUDA Kernels in PyTorch"** (Mark Saroufim) — practical walkthrough of Nsight Compute profiling, reading kernel statistics, and identifying bottlenecks.
  - https://www.youtube.com/watch?v=LuhJEEJQgUM

- **NVIDIA Nsight Compute: Roofline Analysis** — how to use hierarchical roofline analysis within Nsight Compute to identify whether kernels are compute- or memory-bound.
  - https://developer.nvidia.com/blog/accelerating-hpc-applications-with-nsight-compute-roofline-analysis/

### 18. Bandwidth accounting from first principles

Bandwidth accounting means computing the **theoretical minimum memory traffic** for an operation, then comparing against measured bandwidth to determine utilization. For a matrix multiply C = A × B with dimensions (M, K) × (K, N), minimum traffic is `2(MK + KN + MN)` bytes (read A, read B, write C).

- **NVIDIA GPU Performance Background User's Guide** — defines ops:byte ratio, shows how to determine math-bound vs memory-bound, covers wave quantization and tail effects. The starting point for any bandwidth calculation.
  - https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html

- **"What is memory bandwidth?"** (Modal GPU Glossary) — concise definition with NVIDIA data center GPU bandwidth table from Ampere through Blackwell. B200 = **8 TB/s HBM3e**.
  - https://modal.com/gpu-glossary/perf/memory-bandwidth

- **"What is the roofline model?"** (Modal GPU Glossary) — companion piece explaining how to apply the roofline model to specific GPU architectures.
  - https://modal.com/gpu-glossary/perf/roofline-model

- **"What Shapes Do Matrix Multiplications Like?"** (Horace He) — deep dive into how matmul performance varies with input shapes on GPUs. Directly applicable to understanding DiT FFN and attention performance.
  - https://www.thonking.ai/p/what-shapes-do-matrix-multiplications

---

## Systems patterns

### 19. Producer-consumer with backpressure — bounded channels, ring buffers

In a pipeline-parallel inference system, each stage is a producer for the next stage. Without backpressure, a fast producer can overwhelm a slow consumer, causing OOM or unbounded latency. **Bounded queues** are the simplest correct solution: block the producer when the queue is full.

- **"Backpressure explained — the resisted flow of data through software"** (Jay Phelps) — the canonical backpressure explainer. Covers strategies: control the producer, buffer, or drop. Widely referenced.
  - https://medium.com/@jayphelps/backpressure-explained-the-flow-of-data-through-software-2350b3e77ce7

- **"Notes on Distributed Systems for Young Bloods"** (Jeff Hodges) — classic essay. The backpressure section is particularly well-known and covers practical implications for production systems.
  - https://www.somethingsimilar.com/2013/01/14/notes-on-distributed-systems-for-young-bloods/

- **"Dealing with rejection (in distributed systems)"** (WarpStream) — practical treatment from a production streaming system. Resource-based limits, the "Goldilocks zone" of queue depth, and circuit breaker patterns.
  - https://www.warpstream.com/blog/dealing-with-rejection-in-distributed-systems

### 20. Message framing and versioning

When pipeline stages communicate over network boundaries (multi-node inference), you need message framing (length-prefixed or delimited) and versioning for forward/backward compatibility as the system evolves.

- **"Message Framing"** (Stephen Cleary) — classic TCP message framing explainer. Length prefixing vs delimiters, implementation for partial receives.
  - https://blog.stephencleary.com/2009/04/message-framing.html

- **"2.3 Framing" from Computer Networks: A Systems Approach** (Peterson & Davie) — byte-oriented, bit-oriented, and count-based framing from the canonical networking textbook. Free online.
  - https://book.systemsapproach.org/direct/framing.html

- **Protocol Buffers Language Guide (proto3)** — the standard for schema evolution. Wire-safe vs wire-unsafe changes, field numbering, reserved fields.
  - https://protobuf.dev/programming-guides/proto3/

### 21. Idempotency and replay

For fault tolerance in a streaming video pipeline, operations should be idempotent — re-executing a denoising step or VAE decode with the same inputs produces the same output. Combined with **replay from checkpointed state**, this gives you exactly-once semantics without distributed transactions.

- **"What is Idempotency in Distributed Systems?"** (AlgoMaster) — idempotency keys, HTTP method idempotency, state management patterns.
  - https://blog.algomaster.io/p/idempotency-in-distributed-systems

- **"Exactly Once in Distributed Systems"** — why exactly-once is hard, and how at-least-once + idempotency achieves it in practice.
  - https://serverless-architecture.io/blog/exactly-once-in-distributed-systems/

---

## Domain-specific: video DiT inference

### 22. KV cache management in streaming inference

KV caches store the key/value projections from previous tokens to avoid recomputation during autoregressive generation. In video DiT streaming, this extends to **temporal KV caches** across denoising steps and frames. PagedAttention (from vLLM) introduced OS-style paged memory management, reducing waste from **60–80% to under 4%**.

- **"Efficient Memory Management for Large Language Model Serving with PagedAttention"** (Kwon et al., SOSP 2023) — the seminal paper. OS virtual memory analogy, block tables, non-contiguous allocation, copy-on-write sharing.
  - https://arxiv.org/abs/2309.06180

- **"Continuous Batching from First Principles"** (HuggingFace) — derives continuous batching from attention and KV caching fundamentals. Mixing prefill/decode in the same batch via attention masks.
  - https://huggingface.co/blog/continuous_batching

- **"Inside vLLM: Anatomy of a High-Throughput LLM Inference System"** (vLLM blog) — comprehensive deep-dive into vLLM V1: scheduling, paged attention, continuous batching, prefix caching, disaggregated prefill/decode.
  - https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html

- **vLLM Distributed Inference Blog Post** — detailed walkthrough of distributed serving architecture, tensor/pipeline parallelism configuration.
  - https://blog.vllm.ai/2025/02/17/distributed-inference.html

### 23. VAE latency and chunking for video

The VAE decoder is often the **latency bottleneck** in video generation pipelines. 3D VAEs compress both spatially and temporally (typical compression: 8×8×4), but decoding back to pixel space is expensive. **Tiled decoding** splits the latent spatially, **temporal chunking** with causal convolution caching enables frame-by-frame streaming decode.

- **AutoencoderKLCogVideoX** (Diffusers docs) — CogVideoX 3D VAE with `enable_tiling()` and `enable_slicing()`. Tile overlap blending for artifact reduction. Configurable tile dimensions.
  - https://huggingface.co/docs/diffusers/en/api/models/autoencoderkl_cogvideox

- **"VAE System and Video Encoding" (LightX2V)** — streaming VAE decode: CausalConv3d temporal caching (`CACHE_T=2`), frame-by-frame decoding, tiling configs, CPU offloading, distributed VAE with 2D grid splitting.
  - https://deepwiki.com/ModelTC/lightx2v/4.5-vae-system

- **"Seedance 1.0"** (ByteDance, 2025) — production video generation system. Thin VAE decoder (narrowed channels near pixel space) for **2x speedup**. Hybrid parallelism for distributed VAE, async offloading, temporally-causal compression.
  - https://arxiv.org/abs/2506.09113

- **"Improved Video VAE for Latent Video Diffusion Model"** (Wu et al., CVPR 2025) — dual-branch keyframe temporal compression. Addresses quality degradation from causal convolution. 3D causal VAE design.
  - Available via CVPR 2025 proceedings (no direct URL yet — check https://cvpr.thecvf.com/ when proceedings publish)

### 24. Video DiT scheduling — denoising steps, causal dependencies, rolling windows

Video DiT scheduling determines how denoising steps are ordered and parallelized across frames. **Rolling window** approaches generate video autoregressively: each window of N frames shares context from the previous window, enabling arbitrarily long generation. **Stream Batch** (from StreamDiffusion) batches denoising steps across time for throughput.

- **StreamDiffusion: A Pipeline-level Solution for Real-time Interactive Generation** (Kodaira et al., 2023) — introduces Stream Batch (staggered denoising across frames), Residual CFG for **2.05x speedup** over standard CFG, and Stochastic Similarity Filter. **91.07 fps** on RTX 4090.
  - https://arxiv.org/abs/2312.12491
  - GitHub: https://github.com/cumulo-autumn/StreamDiffusion

- **StreamDiffusionV2** (Feng et al., 2025) — extends to VIDEO diffusion models. SLO-aware batching, block scheduler, sink-token rolling KV cache, motion-aware noise. **58.28 FPS with a 14B model** on 4×H100. The most directly relevant system to what you're building.
  - Project page: https://streamdiffusionv2.github.io/
  - GitHub: https://github.com/chenfengxu714/StreamDiffusionV2

- **"Diffusion Models for Video Generation"** (Lilian Weng) — comprehensive survey covering cascaded pipelines, temporal super-resolution, Make-A-Video, Video LDM, SVD, and Sora-class architectures. Excellent for understanding the space of scheduling approaches.
  - https://lilianweng.github.io/posts/2024-04-12-diffusion-video/

- **"PipeDiT: Accelerating DiT in Video Generation with Pipelining and Decoupling"** — pipelined sequence parallelism (PipeSP), decoupled diffusion/VAE onto separate GPU groups (DeDiVAE), and attention co-processing for idle GPU utilization. Directly addresses pipeline-parallel DiT inference.
  - https://arxiv.org/abs/2511.12056

- **Scalable Diffusion Models with Transformers (DiT)** (Peebles & Xie, ICCV 2023) — the foundational DiT paper. Replaces U-Net with transformers in latent diffusion. Strong correlation between compute (Gflops) and sample quality. The architecture your system will be running.
  - https://arxiv.org/abs/2212.09748

---

## Essential meta-resources

These resources cut across multiple topics and are worth bookmarking as ongoing references.

- **Edward Yang's blog** — the single most authoritative source on PyTorch compiler and distributed internals. Start with "State of torch.compile for training (August 2025)" and "Ways to use torch.compile."
  - https://blog.ezyang.com/category/pytorch/

- **PyTorch Developer Podcast** (Edward Yang) — 10–20 minute episodes on compiler collectives, TORCH_TRACE, higher-order operators, tensor subclasses, compiled autograd. The podcast format provides context and reasoning that documentation omits.
  - https://podcasts.apple.com/us/podcast/pytorch-developer-podcast/id1566080008

- **GPU MODE lecture series** — 92+ lectures covering CUDA fundamentals, profiling, Flash Attention, ring attention, quantization, and Triton. The lecture GitHub repo has notes and code for each session.
  - YouTube: https://www.youtube.com/@GPUMODE
  - GitHub: https://github.com/gpu-mode/lectures

- **"The Parallelism Mesh Zoo"** (Edward Yang) — schematic discussion of all parallelization strategies through device meshes: DP, FSDP, HSDP, TP, SP, CP, PP, EP. Essential for understanding how your pipeline parallelism fits into a multi-dimensional parallelism scheme.
  - https://blog.ezyang.com/2025/08/the-parallelism-mesh-zoo/

- **Megatron-LM repository** — the reference implementation for large-scale distributed training with 3D parallelism. The pipeline parallel schedules and parallel state management code are the most production-hardened implementations available.
  - https://github.com/NVIDIA/Megatron-LM

---

## Suggested reading order

For someone building a pipeline-parallel video DiT inference system, the following sequence maximizes learning efficiency by building concepts in dependency order:

1. **Foundations first**: Horace He's "Making Deep Learning Go Brrrr" (topic 16) → PyTorch CUDA Semantics (topics 5–7) → CUDA graphs programming guide (topic 6)
2. **Compiler layer**: Dynamo Deep-Dive (topic 9) → Inductor design doc by Jason Ansel (topic 10) → Edward Yang's "State of torch.compile" (topics 11–12)
3. **Distributed layer**: NCCL User Guide (topic 1) → Functional collectives RFC (topic 11) → PyTorch distributed pipelining API (topics 13–14)
4. **Pipeline parallelism theory**: GPipe paper → PipeDream-2BW paper → Zero Bubble PP paper → Megatron-LM schedules source code (topics 13–15)
5. **Domain-specific**: DiT paper → StreamDiffusion → StreamDiffusionV2 → PipeDiT → CogVideoX VAE documentation (topics 22–24)
6. **Integration**: vLLM architecture blog → torch.compile + vLLM blog → profiling compiled distributed code (topics 17–18)

The total corpus is roughly **40–50 hours of reading** and **10–15 hours of video content**. Prioritize the bolded foundational resources in each section — you can reach working competence in ~20 hours by focusing on those alone.