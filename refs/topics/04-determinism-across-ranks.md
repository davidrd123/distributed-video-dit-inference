# Topic 4: Determinism across ranks in distributed training/inference

Non-determinism in distributed settings comes from three sources: **CUDA atomicAdd operations** (non-associative floating-point), **cuDNN autotuning** selecting different algorithms per run, and **NCCL reduction order** across ranks.

## Resources

<!-- Resource IDs from manifest belonging to this topic -->
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| pytorch-reproducibility | PyTorch Reproducibility Guide | low | pending |
| pytorch-deterministic-api | torch.use_deterministic_algorithms API | low | pending |
| fp-non-assoc-reproducibility | Impacts of floating-point non-associativity on reproducibility for HPC and deep learning | low | pending |

## Implementation context

TP v0 treats cross-rank determinism as “detect drift and crash,” not bitwise reproducibility: optional per-call input digests (`SCOPE_TP_INPUT_DIGEST=1`) catch envelope/tensor mismatches, and shard fingerprints catch weight divergence (Franken-model). A concrete bringup bug: the initial fingerprint baseline was identical on both ranks (Run 3: `[6137181582272122429, 6137181582272122429]`), which was fixed so fingerprints diverge as expected (`[-505995101758162675, 4154699338113877488]`). Backend selection is also pinned via env parity (don’t let ranks auto-pick `flash` vs `fa4`).

See: `refs/implementation-context.md`, `scope-drd/notes/FA4/h200/tp/explainers/06-failure-modes.md` (Q2), `scope-drd/notes/FA4/h200/tp/bringup-run-log.md` (Run 3 + Run 8 shard fingerprint notes), `scope-drd/notes/FA4/h200/tp/runtime-checklist.md` (Locked v0 decisions: determinism + backend pinning).

## Synthesis

<!-- To be filled during study -->

### Mental model

### Key concepts

### Cross-resource agreement / disagreement

### Practical checklist

### Gotchas and failure modes

### Experiments to run
