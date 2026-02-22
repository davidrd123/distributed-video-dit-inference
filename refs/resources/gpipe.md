# GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism

| Field | Value |
|-------|-------|
| Source | https://arxiv.org/abs/1811.06965 |
| Type | paper |
| Topics | 13 |
| Authors | Huang et al. |
| Year | 2018 |
| Status | stub |

## Why it matters

The foundational synchronous pipeline parallelism paper. Introduces micro-batch splitting and re-materialization (activation checkpointing) for memory efficiency. GPipe fills the pipeline with micro-batches and synchronizes at the end -- simple but high memory. Understanding GPipe is prerequisite to understanding the 1F1B and zero-bubble schedules that modern systems actually implement.

## Core claims

<!-- Each claim should cite evidence: sources/gpipe/full.md#<heading> -->

1. **Claim**: GPipe provides pipeline parallelism for any model that can be expressed as a **sequence of layers**, partitioning consecutive layer groups (“cells”) across accelerators and inserting communication at partition boundaries.
   **Evidence**: [sources/gpipe/full.md#abstract](../../sources/gpipe/full.md#abstract), [sources/gpipe/full.md#21-interface](../../sources/gpipe/full.md#21-interface), [sources/gpipe/full.md#22-algorithm](../../sources/gpipe/full.md#22-algorithm)

2. **Claim**: GPipe’s core algorithm is **batch splitting + pipelining**: split a mini-batch into $M$ micro-batches, pipeline them through $K$ partitions, and (for training) apply **one synchronous gradient update per mini-batch** by accumulating gradients over micro-batches.
   **Evidence**: [sources/gpipe/full.md#1-introduction](../../sources/gpipe/full.md#1-introduction), [sources/gpipe/full.md#22-algorithm](../../sources/gpipe/full.md#22-algorithm), [sources/gpipe/full.md#7-conclusion](../../sources/gpipe/full.md#7-conclusion)

3. **Claim**: GPipe supports (optional) **per-layer compute-cost estimates** and uses them to partition layers into cells by **minimizing variance** in estimated per-cell cost (to keep pipeline stage times balanced).
   **Evidence**: [sources/gpipe/full.md#21-interface](../../sources/gpipe/full.md#21-interface), [sources/gpipe/full.md#22-algorithm](../../sources/gpipe/full.md#22-algorithm)

4. **Claim**: GPipe’s **re-materialization** (activation recomputation) stores only boundary activations during forward and recomputes each partition’s forward function during backward; the paper states peak activation memory drops from $O(N\\times L)$ to $O(N + \\frac{L}{K}\\times\\frac{N}{M})$.
   **Evidence**: [sources/gpipe/full.md#23-performance-optimization](../../sources/gpipe/full.md#23-performance-optimization)

5. **Claim**: Pipeline parallelism introduces **bubble** (idle) time; the paper characterizes bubble time as $O(\\frac{K-1}{M+K-1})$ (amortized over $M$) and reports bubble overhead is negligible when $M \\ge 4\\times K$.
   **Evidence**: [sources/gpipe/full.md#23-performance-optimization](../../sources/gpipe/full.md#23-performance-optimization), [sources/gpipe/full.md#3-performance-analyses](../../sources/gpipe/full.md#3-performance-analyses)

6. **Claim**: GPipe’s communication overhead is low because it only transfers activation tensors at **partition boundaries**; the paper demonstrates scaling even on multi-GPU setups without high-speed interconnects.
   **Evidence**: [sources/gpipe/full.md#23-performance-optimization](../../sources/gpipe/full.md#23-performance-optimization), [sources/gpipe/full.md#3-performance-analyses](../../sources/gpipe/full.md#3-performance-analyses)

7. **Claim**: GPipe’s speedup depends on layer uniformity and micro-batch count: Transformer layers (more uniform) show near-linear scaling when $M \\gg K$, while AmoebaNet shows sub-linear scaling due to imbalance; with $M=1$ “there is effectively no pipeline parallelism.”
   **Evidence**: [sources/gpipe/full.md#3-performance-analyses](../../sources/gpipe/full.md#3-performance-analyses)

8. **Claim**: Compared to SPMD-style model parallelism, GPipe avoids an “abundance of AllReduce-like operations” and instead communicates only at boundaries; compared to PipeDream, GPipe avoids **weight staleness** by using synchronous updates. Limitations include assuming each single layer fits on one accelerator and complications for batch-dependent layers (e.g., BatchNorm).
   **Evidence**: [sources/gpipe/full.md#6-design-features-and-trade-offs](../../sources/gpipe/full.md#6-design-features-and-trade-offs), [sources/gpipe/full.md#22-algorithm](../../sources/gpipe/full.md#22-algorithm)

## Key technical details

### Notation (paper)

- $L$: number of layers; layer $L_i$ has forward $f_i$ and parameters $w_i$ (optional cost estimator $c_i$).
  **Evidence**: [sources/gpipe/full.md#21-interface](../../sources/gpipe/full.md#21-interface)
- $K$: number of partitions (“cells”) $p_k$; each cell is a contiguous subsequence of layers with composite forward $F_k$ and cost estimate $C_k$.
  **Evidence**: [sources/gpipe/full.md#21-interface](../../sources/gpipe/full.md#21-interface)
- $N$: mini-batch size; $M$: number of micro-batches (micro-batch size $N/M$).
  **Evidence**: [sources/gpipe/full.md#22-algorithm](../../sources/gpipe/full.md#22-algorithm)

### Schedule shape (what “GPipe” means operationally)

- **Training schedule** (paper): split mini-batch → pipeline **forward** micro-batches through $K$ cells → run **backward** per micro-batch → **accumulate gradients across micro-batches** → apply one synchronous update for the mini-batch.
  **Evidence**: [sources/gpipe/full.md#22-algorithm](../../sources/gpipe/full.md#22-algorithm)
- **Bubble heuristic** (paper): bubble time scales like $O(\\frac{K-1}{M+K-1})$ and is reported negligible when $M \\ge 4K$ (assuming balanced partitions as in Figure 2c).
  **Evidence**: [sources/gpipe/full.md#23-performance-optimization](../../sources/gpipe/full.md#23-performance-optimization)

### Re-materialization (activation recomputation)

- Store only **partition-boundary activations** in forward; recompute $F_k$ inside each partition during backward.
  **Evidence**: [sources/gpipe/full.md#23-performance-optimization](../../sources/gpipe/full.md#23-performance-optimization)
- Peak activation memory stated as $O(N + \\frac{L}{K}\\times\\frac{N}{M})$ vs $O(N\\times L)$ without re-materialization/partitioning.
  **Evidence**: [sources/gpipe/full.md#23-performance-optimization](../../sources/gpipe/full.md#23-performance-optimization)

### Partitioning objective

- GPipe’s heuristic partitioning aims to **minimize variance** of estimated cell costs (balance stage times); paper notes imbalance can dominate when layers have very different compute/memory.
  **Evidence**: [sources/gpipe/full.md#22-algorithm](../../sources/gpipe/full.md#22-algorithm), [sources/gpipe/full.md#23-performance-optimization](../../sources/gpipe/full.md#23-performance-optimization)

## Actionables / gotchas

- **Pipeline parallelism only buys throughput in steady state**: if you have $P$ pipeline stages and only $B=1$ in-flight item, ideal utilization is ~50% for $P=2$ (more generally ~$B/(B+P-1)$). For PP0 bringup, treat “micro-batches” as “in-flight chunks/envelopes” and provision queue depth so the mesh stays busy. See: `scope-drd/notes/FA4/h200/tp/pp-topology-pilot-plan.md`, `scope-drd/notes/FA4/h200/tp/streamdiffusion-v2-analysis.md`, and the bubble discussion in [sources/gpipe/full.md#23-performance-optimization](../../sources/gpipe/full.md#23-performance-optimization).
- **Don’t confuse throughput with latency**: GPipe’s micro-batch fill reduces bubbles but increases end-to-end latency for any single item (fill + drain). For interactive streaming inference, decide explicitly whether you’re optimizing throughput (FPS at steady state) or first-chunk latency.
- **Uniform-block DiT generators are “GPipe-friendly”**: the paper’s strongest scaling case is when per-layer costs are uniform (Transformer); our generator is a uniform block stack (40 similar blocks), so simple even block partitioning should get close to the “balanced partitions” assumption (see [dit-paper](dit-paper.md)).
- **Stage imbalance is the real enemy**: if you co-locate non-uniform work (I/O, embeddings, time-conditioning, VAE decode) with one stage, you violate GPipe’s balanced-stage assumption and lose the max-of-stages critical path benefit. Treat “minimize variance in per-stage cost” as the primary PP partitioning objective.
- **Re-materialization is a training lever; for inference, map it to “stash vs recompute” decisions**: GPipe recomputes activations in backward to trade FLOPs for memory. In our inference pipeline, the analogous pressure points are KV-cache / recompute policies and “what state is allowed to persist across chunks” (see `scope-drd/notes/FA4/h200/tp/pp-topology-pilot-plan.md`).
- **GPipe assumes each layer fits on one device**: if a single Transformer block (or attention/FFN matmul) doesn’t fit, you need intra-layer parallelism (TP/SP). Our expected path is TP+PP: TP inside a stage, PP across block ranges.
- **BatchNorm gotcha (mostly training-only)**: micro-batch splitting changes how BatchNorm statistics are computed (paper has explicit handling). For inference this is usually moot (eval-mode stats), but it’s a warning sign for any batch-dependent op you might accidentally leave enabled.
- **Know the “baseline schedule” you’re comparing against**: GPipe is the canonical fill/drain schedule; for more aggressive scheduling (mostly training-focused), compare to 1F1B (PipeDream-2BW) and zero-bubble schedules, and consider using PyTorch’s pipelining schedules as reference implementations.

## Related resources

- [dit-paper](dit-paper.md) -- DiT’s uniform block stack is ideal for block-wise PP partitioning
- [pipedream-2bw](pipedream-2bw.md) -- introduces 1F1B schedule that reduces GPipe's memory overhead
- [zero-bubble-pp](zero-bubble-pp.md) -- eliminates pipeline bubbles by splitting backward into B and W phases
- [pytorch-pipelining-api](pytorch-pipelining-api.md) -- ScheduleGPipe implements this paper's approach
- [pipedit](pipedit.md) -- pipeline-parallel DiT inference building on PP foundations
