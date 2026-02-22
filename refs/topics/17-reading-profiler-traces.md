---
status: draft
---

# Topic 17: Reading Nsight / torch.profiler traces

Profiling is the primary tool for understanding where time is spent in distributed GPU workloads. PyTorch's built-in profiler integrates with TensorBoard for visualization, while NVIDIA Nsight Compute provides kernel-level analysis including roofline positioning.

## Resources

<!-- Resource IDs from manifest belonging to this topic -->
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| pytorch-profiler-tensorboard | PyTorch Profiler with TensorBoard | medium | pending |
| profiling-torch-compile | Profiling torch.compile performance | medium | pending |
| gpu-mode-lecture-1 | GPU MODE Lecture 1: How to Profile CUDA Kernels in PyTorch | low | link_only |
| nsight-roofline | NVIDIA Nsight Compute: Roofline Analysis | low | pending |

## Implementation context

Block profiling (Run 10b) is the main measured attribution we have: **denoise 435ms (66.8%)**, **decode 107ms (16.5%)**, **recompute_kv_cache 104ms (16.0%)** at TP=2, compile off. That profile also notes the measurement pitfall: per-block `synchronize()` perturbs overlap and inflates times, so use it for attribution rather than absolute throughput. For PP0/PP1, overlap is validated via derived metrics (`period`, `stage0`, `stage1`, `OverlapScore ≥ 0.30`) instead of relying on cross-rank trace alignment.

See: `refs/implementation-context.md` → Phase 2, `scope-drd/notes/FA4/h200/tp/bringup-run-log.md` (Run 10b), `scope-drd/notes/FA4/h200/tp/pp0-bringup-runbook.md` (Phase 2 overlap metrics).

## Synthesis

### Mental model

- **Start from a question, not a tool**: “why is FPS lower than expected?”, “is overlap real?”, “did compile reduce launch overhead?”, “is comm on the critical path?”, “is decode gating the period?”
- **There are three distinct profiling layers** (and they answer different questions):
  1. **Pipeline/block attribution** (system-level): where does wall time go across *blocks/stages* (A/B/C) and across ranks? (Run 10b is this.)
  2. **Timeline overlap** (systems-level): do CUDA streams and comm actually overlap, or are we serialized by waits/sync/allocator? (Nsight Systems + NVTX ranges is the cleanest.)
  3. **Kernel efficiency** (micro-level): is a hot kernel bandwidth-bound, compute-bound, or overhead-bound? (Nsight Compute / roofline; only after you know which kernel matters.)
- **GPU work is asynchronous by default**: “Python returns” does not mean “GPU finished.” Any measurement method that inserts `torch.cuda.synchronize()` changes the schedule and can destroy overlap. Treat synchronizing profilers as *attribution*, not a throughput oracle. (Run 10b note; reinforced by `scope-drd/notes/FA4/h200/tp/5pro/14-async-decode/review.md`.)
- **Distributed overlap cannot be “proven” by aligning rank0 vs rank1 timelines** unless you also handle clock sync. Prefer derived overlap metrics (Period/Stage0/Stage1/OverlapScore) and within-process timelines. (See `scope-drd/notes/FA4/h200/tp/pp-control-plane-pseudocode.md` warning and `scope-drd/notes/FA4/h200/tp/pp0-bringup-runbook.md` Phase 2.)

### Key concepts

- **Attribution vs throughput**:
  - Attribution: “decode is ~16.5% of chunk time” (Run 10b).
  - Throughput: “period_k improved” or “FPS improved” under the target settings (compile on, steady state).
  - Don’t treat an attribution-mode profile as a “FPS benchmark.”
- **Critical path**: the stage that determines the per-chunk period. For PP0/PP1 you want `period ≈ max(stage0, stage1)`; if `period ≈ stage0 + stage1`, you’re serialized.
- **Stage timing vocabulary (PP)** (from `pp0-bringup-runbook.md`):
  - rank0 timestamps: `tA0/tA1/tSend/tRecv/tDecode/tEmit`
  - mesh leader timestamps: `tB0/tB1`
  - derived: `period_k`, `stage0_k`, `stage1_k`, `hidden_k`, `OverlapScore`
- **“Overhead-bound” signatures**:
  - Many small kernels/subgraphs, lots of CPU time between GPU launches.
  - In our history, ~160 graph breaks/forward caused a collapse to 9.6 FPS; the fix (functional collectives) restored a single compiled graph per block and 24.5 FPS. (`scope-drd/notes/FA4/h200/tp/bringup-run-log.md` Runs 8–12b.)
- **NVTX ranges**: labels you add to create meaningful regions in Nsight Systems. 5Pro’s PP execution plan suggests stage-shaped ranges (e.g., `nvtx:pp:stage0:A_build_envelope`, `nvtx:pp:stage0:C_decode`, `nvtx:pp:stage1:B_phaseB`). (`scope-drd/notes/FA4/h200/tp/5pro/13-pp-execution-ready/response.md`.)
- **Stream semantics sanity**: “async” means “enqueued onto a CUDA stream + locally stream-ordered,” plus a cross-rank ordering contract for collectives. (See `refs/topics/02-deadlock-patterns.md`.)

### Cross-resource agreement / disagreement

- **Agreement**:
  - Scope’s bringup notes and 5Pro reviews converge on a practical rule: use *derived overlap metrics* for PP correctness and *timeline tools* only once instrumentation exists and you can interpret it safely.
  - The compile story (Runs 8–12b) is consistent with the “overhead-bound” framing in `refs/topics/16-roofline-model.md`: reducing dispatch fragmentation can dominate kernel-level optimizations.
- **Nuance**:
  - Torch profiler vs Nsight Systems: both can show “where time went,” but torch profiler often perturbs schedules more (Python-side instrumentation, synchronization choices). For overlap claims, Nsight Systems + NVTX is generally the more faithful tool.
  - Kernel roofline analysis is useful only after you’ve identified the hot kernels and validated that your end-to-end bottleneck is not elsewhere (serialization, duplicated work, rank0 gating).

### Practical checklist

- **0) Pin the scenario**: record exact env/config that affects kernels and graphs (compile flags, attention backend, resolution/chunking, recompute cadence). For distributed, enforce env parity (Topic 02/12).
- **1) Choose the correct measurement**:
  - “Where does time go?” → block profiling / per-stage timers (Run 10b style).
  - “Is overlap real?” → PP OverlapScore metrics + Nsight Systems ranges (no per-block synchronize).
  - “Is comm on the critical path?” → isolate comm cost (NCCL microbench, and/or per-collective timers) and compare to kernel time.
  - “Is kernel efficient?” → Nsight Compute on the top 1–3 kernels only.
- **2) Get stable numbers before deep dives**:
  - warm up (compile, cache fill) and then measure p50/p90/p99 period over a steady window.
  - treat p99 spikes as a separate bug class (allocator, synchronization, graph recompiles).
- **3) If investigating PP overlap** (bringup path):
  - instrument timestamps exactly as in `scope-drd/notes/FA4/h200/tp/pp0-bringup-runbook.md` Phase 2.
  - compute `OverlapScore` after warmup and require `OverlapScore ≥ 0.30` as a pass gate (bringup rule).
  - optionally add NVTX ranges for A/B/C regions for a “sanity look,” but don’t use cross-rank alignment as proof.
- **4) If investigating compile overhead**:
  - start with `TORCH_LOGS=graph_breaks` and `tp_compile_repro.py` (regression gate).
  - if traces show lots of tiny regions, assume graph breaks / recompiles until proven otherwise.
- **5) Write down your conclusion as an operator statement**:
  - “decode is X% of chunk time at resolution R” (attribution), and/or
  - “period ≈ max(stage0, stage1) with OverlapScore S at D_in/D_out=2” (overlap), and/or
  - “graph_breaks regressed from 0 to N in mode C” (compile regression).

### Gotchas and failure modes

- **Per-block `synchronize()` destroys overlap**: block profilers that synchronize after each block are great for attribution but can mask the very overlap you’re trying to create (async decode, PP queueing). Use them carefully. (Run 10b; `scope-drd/notes/FA4/h200/tp/5pro/14-async-decode/review.md`.)
- **Clock domains**: rank0 and rank1 CPU clocks are not aligned; “stage1 starts before stage0 ends” is not provable from raw timestamps without sync.
- **Profiler-induced Heisenbugs**: adding logging, printing, or profiler scopes inside compiled regions can create graph breaks and change performance.
- **Missing the real critical path**: it’s easy to spend days optimizing a kernel that isn’t on the steady-state period (e.g., optimizing a block that runs on rank0 while mesh is idle due to backpressure).
- **Interpreting “NCCL async” incorrectly**: NCCL calls enqueue work on streams; a hidden `cudaDeviceSynchronize` (or implicit sync via allocator) can serialize everything and make the trace look like “NCCL blocks everything.”

### Experiments to run

- **Reproduce Run 10b attribution at target settings**: capture an updated block breakdown with compile ON (Run 10b was compile OFF) to confirm whether decode/recompute shares remain similar. (`scope-drd/notes/FA4/h200/tp/async-decode-overlap-scoping.md` explicitly calls this out.)
- **Overlap proof A/B (PP0)**: run PP0 with `D_in/D_out=1` vs `2` and compare `OverlapScore`, `period`, and `t_mesh_idle_ms` (if logged). Expect D=1 to collapse overlap and D=2 to be the first stable overlap regime. (`pp0-bringup-runbook.md`, `pp-topology-pilot-plan.md`.)
- **Timeline sanity check (Nsight Systems)**: add NVTX ranges for A/B/C and confirm qualitatively that Stage 1 compute overlaps with Stage 0 decode when D_out=2 (while remembering cross-rank caveats). (`scope-drd/notes/FA4/h200/tp/5pro/13-pp-execution-ready/response.md`.)
- **Compile overhead regression test**: intentionally reintroduce a graph break around a collective (test-only) and observe the “many tiny subgraphs” signature + FPS collapse, then revert and confirm recovery. (Runs 8–12b anchor.)
