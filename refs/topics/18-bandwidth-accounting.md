---
status: draft
---

# Topic 18: Bandwidth accounting from first principles

Bandwidth accounting means computing the **theoretical minimum memory traffic** for an operation, then comparing against measured bandwidth to determine utilization. For a matrix multiply C = A x B with dimensions (M, K) x (K, N), minimum traffic is `2(MK + KN + MN)` bytes (read A, read B, write C).

## Resources

<!-- Resource IDs from manifest belonging to this topic -->
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| nvidia-gpu-perf-guide | NVIDIA GPU Performance Background User's Guide | medium | pending |
| modal-memory-bandwidth | What is memory bandwidth? (Modal GPU Glossary) | low | pending |
| modal-roofline | What is the roofline model? (Modal GPU Glossary) | low | pending |
| matmul-shapes | What Shapes Do Matrix Multiplications Like? | medium | pending |

## Implementation context

Bandwidth accounting explains two observed ceilings: (1) single-stream FPS scales with HBM bandwidth across GPUs (B200/B300 ~8 TB/s ≈ 33–34 FPS vs H200 ~4.8 TB/s ≈ 20 FPS; ratio ~0.58–0.60), and (2) TP collectives are fast enough on NVLink to be plausible (BF16 all-reduce of a ~21 MiB `[1,2160,5120]` tensor is **~0.113ms**, ~195 GB/s per rank). These numbers anchor whether a proposed optimization is compute-bound vs bandwidth/overhead-bound.

See: `refs/implementation-context.md` → System summary + Phase 1, `scope-drd/notes/FA4/h200/tp/streamdiffusion-v2-analysis-opus.md` (H200 bandwidth-bound evidence), `scope-drd/notes/FA4/h200/tp/feasibility.md` (Section 1: collective bandwidth).

Relevant Scope artifacts:
- `scope-drd/scripts/bench_nccl_collectives.py` (collective latency/bandwidth measurements)
- `scope-drd/scripts/bench_gemm_shapes.py` / `scope-drd/scripts/gemm_shape_census.py` (compute-vs-bytes reasoning for common matmuls)

## Synthesis

### Mental model

- **Bandwidth accounting is the “bytes first” version of roofline**: estimate the *lower bound* bytes-to-HBM (or bytes-to-NVLink) implied by an algorithm and tensor shapes, then compare to measured time to infer whether you’re bandwidth-bound, overhead-bound, or compute-bound.
- **In practice you have multiple roofs** (and you should name which one you’re hitting):
  - **HBM roof** (intra-GPU): bytes moved between HBM and SMs.
  - **Interconnect roof** (NVLink/NVLS/PCIe): bytes moved by collectives or P2P activation transfers.
  - **Dispatch/launch roof**: many small kernels/subgraphs can dominate even if “bytes are small” (the TP+compile pre-funcol regression is the canonical example of this class; see `scope-drd/notes/FA4/h200/tp/bringup-run-log.md` Runs 8–9b vs 12b).
- **The project-level “anchor observation”** is that steady-state single-stream FPS tracks **HBM bandwidth across GPU generations** (B200/B300 ~8 TB/s → ~33–34 FPS vs H200 ~4.8 TB/s → ~20 FPS; see `scope-drd/notes/FA4/h200/tp/streamdiffusion-v2-analysis-opus.md`). Treat this as a strong prior that the core denoise path is bandwidth-limited on H200 until proven otherwise.
- **Bandwidth accounting is how you decide what kind of optimization is even plausible**:
  - If an operation is already close to the roof, only *byte reducers* can help (fusion, fewer intermediates, lower precision, better locality, fewer redundant passes).
  - If it’s far from the roof, you’re likely paying dispatch, sync, or algorithmic overhead (graph breaks, synchronizations, serialization points, copies/format conversions).

### Key concepts

- **Bytes-to-HBM lower bound**: the minimum bytes you must move if every tensor element is read/written exactly once from HBM. This is a lower bound; real kernels often move more due to materialized intermediates, read-modify-write, layout transforms, and cache misses.
- **Effective bandwidth (measured)**: `BW_eff = bytes_moved / time`. You can compute “implied BW” from your byte estimate and measured time, even before adding hardware counters. If implied BW is near the hardware peak, you’re bandwidth-limited (or your byte estimate is too low).
- **Arithmetic intensity vs pure-bytes accounting**:
  - Roofline uses `FLOPs/byte`.
  - Bandwidth accounting can skip FLOPs and still be decisive when you’re in the bandwidth-limited regime: if time scales ~linearly with bytes, FLOPs are not your binding constraint.
- **Token-driven scaling law (for our bringup regime)**:
  - In the “geometry-only” sweeps (vary `H×W`, keep denoise steps/chunking/local_attn/caches fixed), **tokens/frame ∝ pixel area** and many major buffers scale approximately linearly with token count.
  - This makes “time vs area” and “peak bytes vs area” plots extremely diagnostic for whether you’re bandwidth-bound vs overhead/allocator-bound.
- **Collective bandwidth accounting**: for TP, account for both (a) total bytes per step, and (b) number of synchronization points. Microbench shows the *large* BF16 all-reduce for `[1,2160,5120]` (~21 MiB) is ~0.113ms on 2×H200 NVLink (~195 GB/s per rank). That makes “many collectives” plausible from pure bandwidth, but still vulnerable to launch/ordering/compile overhead. (See `scope-drd/notes/FA4/h200/tp/feasibility.md`.)

### Cross-resource agreement / disagreement

- **Agreement (Scope measurements + roofline intuition)**:
  - `scope-drd/notes/FA4/h200/tp/streamdiffusion-v2-analysis-opus.md` argues the system is HBM-bandwidth-bound on H200 because FPS tracks HBM ratios across GPUs. That’s consistent with the roofline framing in `refs/topics/16-roofline-model.md`.
  - `scope-drd/notes/FA4/h200/tp/feasibility.md` shows the big “hidden-state” collectives are sub-millisecond on NVLink; this supports the claim that *collective bandwidth* is not the first-order blocker for TP=2 on-node (the harder parts are correctness/ordering, compile interaction, and kernel efficiency).
- **Nuance**:
  - “Bandwidth-bound” is often true for the denoise core, but system-level period can still be dominated by work that is duplicated or unsharded (e.g., decode/recompute in Run 10b) or by dispatch fragmentation (Runs 8–9b). Bandwidth accounting should be applied per stage/block, not only to the full pipeline.
  - A speedup claim that sounds like “reduce bytes” must specify *which roof* it reduces (HBM vs NVLink vs dispatch). Otherwise you end up tuning the wrong subsystem.

### Practical checklist

- **Write down your unit of analysis** (and keep it stable across experiments): per chunk, per denoise step, per DiT block, per stage in PP.
- **Measure the scaling slope first** (cheap, high-signal):
  - Sweep resolution by area multipliers (e.g., +10%, +20%, +50% area) with all other knobs held fixed.
  - If time scales ~linearly with area from the first +10%, assume bandwidth (or allocator/copy) is binding.
  - If time is sublinear at small deltas, you’re not saturating yet (launch/overhead share is still meaningful).
  - If time is superlinear, suspect allocator fragmentation, format conversions, or cache-unfriendly kernels.
- **Compute an implied bandwidth**:
  - Start with a conservative byte estimate (count the big tensors you know must be read/written each step) and compute `BW_implied = bytes_est / time_measured`.
  - If `BW_implied` is already near HBM peak for H200, only byte reducers can win.
- **Always separate HBM vs comm bytes**:
  - HBM: token-sized activations, KV traffic, projection intermediates, VAE decode outputs.
  - Comm: TP all-reduces/all-gathers; PP activation transfers; rank0 control-plane messages.
  - Use `scope-drd/scripts/bench_nccl_collectives.py` to keep a “known-good” comm baseline, and treat regressions as a red flag.
- **Treat “bytes moved” as an optimization spec**:
  - For any proposed change, state whether it reduces HBM bytes, reduces NVLink bytes, or reduces dispatch overhead. If you can’t name the roof, it’s probably not a real plan.
- **Keep a small toolkit of probes**:
  - Shape census: `scope-drd/scripts/gemm_shape_census.py`, `scope-drd/scripts/bench_gemm_shapes.py`.
  - Collective costs: `scope-drd/scripts/bench_nccl_collectives.py`.
  - End-to-end regression harness: `scope-drd/notes/FA4/h200/tp/research-program.md` (compile/graph-break gates + microbench sanity).

### Gotchas and failure modes

- **Byte undercounting is the default**: “theoretical minimum traffic” ignores real-world overheads (materialized intermediates, transposes, dtype conversions, allocator behavior). Use it as a floor, not a prediction.
- **Caching makes naive “read weights every step” reasoning misleading**: some weights may stay hot in cache, but you should not assume this holds across kernels, stream interactions, and large working sets. Prefer empirical slopes (time vs area/dtype) over assumptions.
- **Conflating bandwidth-bound with “parallelism won’t help”**: TP/PP can still help by changing *what bytes are duplicated* or by enabling overlap and better utilization. The StreamDiffusionV2 writeup’s stronger “TP can’t help per-sample latency” claim was revised in light of the compile unlock; treat bandwidth accounting as a guardrail, not a blanket veto. (See `scope-drd/notes/FA4/h200/tp/streamdiffusion-v2-analysis-opus.md` update section.)
- **Distributed “sync-point count” is its own hazard**: even if bytes are cheap (0.113ms for 21 MiB all-reduce), many collectives can still lose due to launch overhead, stream ordering bugs, graph breaks, or rank divergence. Bandwidth accounting must be paired with the operator-manual rules in `refs/topics/02-deadlock-patterns.md` and `refs/topics/12-compile-distributed-interaction.md`.
- **Allocator/fragmentation cliffs can look like bandwidth ceilings**: if peak reserved grows faster than allocated (or p99 latency spikes with resolution), you may be hitting allocator pathology rather than pure HBM limits; treat memory stats as part of the bandwidth-accounting workflow (see `refs/topics/07-gpu-memory-management.md`).

### Experiments to run

- **Geometry-only sweep (scaling law)**: measure steady-state chunk period at baseline, +10%, +20%, +50% area with fixed denoise steps, chunking, local attention, and cache window. Record (a) period ratio, (b) peak allocated/reserved, (c) any p99 jitter growth. (Tie into the “pixel multiplier” framing used in the TP/PP planning notes.)
- **HBM vs dispatch A/B**:
  - Reproduce an overhead-bound collapse by introducing graph breaks (e.g., disabling compile-aware collectives) and confirm that FPS drops disproportionately compared to any plausible “bytes changed” story (Runs 8–9b vs 12b are the historical anchor; see `scope-drd/notes/FA4/h200/tp/bringup-run-log.md`).
  - Then restore the functional-collectives path and verify the recovery.
- **Comm budget sanity**: periodically re-run `scope-drd/scripts/bench_nccl_collectives.py` for the canonical `[1,2160,5120]` BF16 shape. Treat a regression in p50 latency/bandwidth as a topology/config/environment issue to resolve before deeper model changes. (See `scope-drd/notes/FA4/h200/tp/feasibility.md` Section 1.)
- **Byte reducer validation**: pick one candidate byte reducer (e.g., attention kernel backend choice; fusion; fewer materialized intermediates) and test whether the period improvement scales with the expected byte reduction.
- **Duplicate-work accounting**: use block profiling (e.g., Run 10b numbers) to separate “HBM ceiling” from “duplicated unsharded work” (decode/recompute/other rank0-only blocks). This helps prioritize structural work like v1.1 generator-only workers.
