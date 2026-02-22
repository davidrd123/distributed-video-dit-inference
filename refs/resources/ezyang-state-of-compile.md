# State of torch.compile for training (August 2025)

| Field | Value |
|-------|-------|
| Source | https://blog.ezyang.com/2025/08/state-of-torch-compile-august-2025/ |
| Type | blog |
| Topics | 11, 12 |
| Author | Edward Z. Yang |
| Status | condensed |

## Why it matters

The best single write-up of how `torch.compile` fits into PyTorch’s distributed story as of August 2025: graph breaks/recompiles, DTensor and functional collectives, compiler-driven parallelism (SimpleFSDP/AutoParallel), and the core distributed pitfall (rank-divergent compilation → NCCL timeouts). It’s the conceptual bridge between the functional collectives RFC and the “why did compile help/hurt?” questions that come up in real distributed inference.

## Core claims

1. **Claim**: `torch.compile` is **just-in-time**: the first call blocks on compilation; caching (local + remote) can skip repeated compilation, and ahead-of-time compilation is already possible for inference via AOTInductor.
   **Evidence**: [sources/ezyang-state-of-compile/full.md#what-is-torchcompiles-functionality](../../sources/ezyang-state-of-compile/full.md#what-is-torchcompiles-functionality)

2. **Claim**: `torch.compile` is designed to be **compositional with eager**: you can compile as small/large a region as you want, and compiled regions can interoperate with autograd/DDP/FSDP (with known limitations); when compilation fails, you can disable compilation for a region via `torch.compiler.disable()` and fall back to eager.
   **Evidence**: [sources/ezyang-state-of-compile/full.md#what-is-torchcompiles-functionality](../../sources/ezyang-state-of-compile/full.md#what-is-torchcompiles-functionality)

3. **Claim**: `torch.compile` aggressively specializes on non-Tensor values and sizes: graphs may be **recompiled** when non-Tensor args/globals change or when sizes vary; it is “static by default” and attempts to recompile to dynamic shapes after observing variation (and this may still fail).
   **Evidence**: [sources/ezyang-state-of-compile/full.md#what-is-torchcompiles-functionality](../../sources/ezyang-state-of-compile/full.md#what-is-torchcompiles-functionality)

4. **Claim**: **Graph breaks** are a first-class behavior: when the compiler cannot capture code, it can insert a graph break and run the uncapturable line(s) in eager, compiling regions before/after; you can ban this behavior with `fullgraph=True`.
   **Evidence**: [sources/ezyang-state-of-compile/full.md#what-is-torchcompiles-functionality](../../sources/ezyang-state-of-compile/full.md#what-is-torchcompiles-functionality)

5. **Claim**: Compile time can scale poorly for Transformer-style code because function calls are inlined and loops are unrolled by default; “regional compilation” (compile a block rather than the whole model) is a lever to reduce compile time.
   **Evidence**: [sources/ezyang-state-of-compile/full.md#what-is-torchcompiles-functionality](../../sources/ezyang-state-of-compile/full.md#what-is-torchcompiles-functionality)

6. **Claim**: `torch.compile` is **not bitwise equivalent** to eager by default: fusion and backend choices can change numerical behavior (e.g., fewer redundant precision casts; different reduction order). There are configuration knobs to emulate eager precision casting for debugging.
   **Evidence**: [sources/ezyang-state-of-compile/full.md#what-is-torchcompiles-functionality](../../sources/ezyang-state-of-compile/full.md#what-is-torchcompiles-functionality)

7. **Claim**: Distributed collectives and DTensor programs can be compiled, but distributed collectives are **unoptimized by default**, and PyTorch does not generally expect to trace through highly optimized distributed framework code.
   **Evidence**: [sources/ezyang-state-of-compile/full.md#what-is-torchcompiles-functionality](../../sources/ezyang-state-of-compile/full.md#what-is-torchcompiles-functionality)

8. **Claim**: DTensor is PyTorch’s “global tensor” abstraction for sharded tensors on an SPMD device mesh; its placements are device-mesh-oriented (including `Partial`), it supports autograd directly on DTensor programs, and it is implemented as a Python `Tensor` subclass with caching/overhead tradeoffs. Compiling DTensor desugars it into collectives and removes eager DTensor overhead.
   **Evidence**: [sources/ezyang-state-of-compile/full.md#state-of-advanced-parallelism](../../sources/ezyang-state-of-compile/full.md#state-of-advanced-parallelism)

9. **Claim**: “Functional collectives” provide non-mutating, compiler-friendly collectives; when compiling, PyTorch can translate traditional collectives into functional collectives for compiler passes, and under compilation the functional outputs can be re-inplaced to avoid forced allocations. Functional collectives (currently) do not support autograd.
   **Evidence**: [sources/ezyang-state-of-compile/full.md#state-of-advanced-parallelism](../../sources/ezyang-state-of-compile/full.md#state-of-advanced-parallelism)

10. **Claim**: `torch.compile` is not SPMD-by-default, and distributed compilation is fragile today because compilation happens in parallel across ranks: divergent compilation decisions (e.g., from dynamic inputs) can lead to NCCL timeouts. Practical mitigations are to eliminate rank divergence or to add extra collectives to synchronize decisions; the longer-term goal is “compile once and send to all nodes.”
   **Evidence**: [sources/ezyang-state-of-compile/full.md#state-of-advanced-parallelism](../../sources/ezyang-state-of-compile/full.md#state-of-advanced-parallelism)

11. **Claim**: Long-term compilation cost management is moving beyond caching: `torch.compile` runs compilation on the cluster by default, and PyTorch is working on “precompile” (ahead-of-time compilation producing binaries via an ABI-stable interface developed for AOTInductor).
   **Evidence**: [sources/ezyang-state-of-compile/full.md#state-of-compile-time](../../sources/ezyang-state-of-compile/full.md#state-of-compile-time)

12. **Claim**: A practical starting point for large-scale `torch.compile` training deployments is to fork `torchtitan`, which demonstrates how PyTorch native distributed features and compilation fit together.
   **Evidence**: [sources/ezyang-state-of-compile/full.md#how-do-i-get-started](../../sources/ezyang-state-of-compile/full.md#how-do-i-get-started)

## Key insights

- **`torch.compile` is a tracing + fallback system, not a “whole-program compiler”**: graph breaks and recompiles are expected behaviors, but they’re also the common sources of perf cliffs.
- **DTensor vs functional collectives are two “compiler-compatible” routes to distributed**: DTensor is the high-level abstraction; functional collectives are the low-level, explicit escape hatch.
- **Distributed compile requires rank-symmetry**: “works on one GPU” is not enough; divergent compilation decisions across ranks can hang the job.
- **Advanced parallelism is trending toward graph passes** (SimpleFSDP, AutoParallel, async TP passes) that insert naive collectives then optimize scheduling, rather than hand-coded distributed frameworks everywhere.

## Actionables / gotchas

- **Track graph-break health as a regression signal**: after the functional-collectives fix, our steady-state break census is `graph_breaks=2` and `unique_graphs=12–14` across configs (Runs 13–14). Treat increases as perf regressions until proven otherwise. See: `scope-drd/notes/FA4/h200/tp/bringup-run-log.md` (Runs 13–14) and `refs/implementation-context.md` (row for `ezyang-state-of-compile`).
- **Avoid rank-divergent compilation decisions**: lock down shapes/flags across ranks, and don’t let “one rank sees a different input” (or a different guard path) become possible when collectives occur inside compiled regions. This is the same class of failure as “different collective order” deadlocks.
- **Prefer functional collectives inside compiled distributed regions**: the post’s story matches our experience — compiler-friendly, non-mutating collectives are the path to getting compile speedups without fragmenting into many tiny graphs. Cross-check against `refs/resources/funcol-rfc-93173.md` and `refs/resources/dynamo-deep-dive.md`.
- **Use “regional compilation” to contain compile time**: compile time scaling with inlining/unrolling is a real cost for deep Transformers. For PP bringup, consider compiling per-block or per-stage functions rather than the entire end-to-end loop.
- **Have a correctness/debug mode for numeric drift**: if compiled results diverge from eager (especially under fusions), enable precision-emulation knobs and/or isolate front-end tracing vs backend lowering to localize the source of drift.
- **Treat DTensor as an integration point, not a free lunch**: it can eliminate eager overhead under compile, but dynamic shapes are a known pain point and operator coverage matters. For TP inference bringup, explicit functional collectives may be the more predictable surface.
- **Compile-time cost is a cluster cost**: JIT compilation blocks first-call execution and burns GPU cluster time. For stable inference loops, consider an AOT/precompile story or a deterministic warmup regime before serving traffic.
- See: `refs/implementation-context.md` (Per-card actionables sharpening guide).

## Related resources

- [funcol-rfc-93173](funcol-rfc-93173.md) -- the functional collectives RFC that this post contextualizes
- [dynamo-deep-dive](dynamo-deep-dive.md) -- Dynamo tracing internals referenced throughout
- [pytorch-cuda-semantics](pytorch-cuda-semantics.md) -- stream semantics underneath compiled execution and collectives
- [pytorch-pipelining-api](pytorch-pipelining-api.md) -- pipeline schedules that interact with compilation
- [making-dl-go-brrrr](making-dl-go-brrrr.md) -- performance motivation for compilation
- [cuda-graphs-guide](cuda-graphs-guide.md) -- CUDA graphs as reduce-overhead compile backend
