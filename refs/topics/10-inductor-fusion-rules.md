# Topic 10: Inductor fusion rules — what fuses, verification

TorchInductor's scheduler decides fusion using `score_fusion(node1, node2)`, which scores pairs of operations by **estimated memory traffic savings**. Pointwise-to-pointwise fusion is most common; reduction and template fusions have additional constraints.

## Resources

<!-- Resource IDs from manifest belonging to this topic -->
| ID | Title | Priority | Status |
|----|-------|----------|--------|
| torchinductor-design | TorchInductor: a PyTorch-native Compiler with Define-by-Run IR and Symbolic Shapes | medium | pending |
| inductor-config | TorchInductor config.py | medium | pending |
| pytorch2-asplos | PyTorch 2: Faster Machine Learning Through Dynamic Python (ASPLOS 2024) | medium | pending |
| inductor-fusion-discussion | Inductor scheduler source and fusion discussion | low | pending |

## Implementation context

Run 12b’s performance win depended on Inductor being able to fuse through the TP block when collectives are traceable: functional collectives eliminate graph breaks and enable “one compiled graph per block” behavior. The regression harness treats this as an invariant: `tp_compile_repro.py` expects **mode_C graph_break_count=0** and **graph_count=1**. Any new graph breaks in the collective wrappers tend to push the system back into many tiny graphs and overhead-bound performance (Runs 8-9b).

See: `refs/implementation-context.md` → Phase 1, `scope-drd/notes/FA4/h200/tp/bringup-run-log.md` (Runs 8-12b), `scope-drd/notes/FA4/h200/tp/research-program.md` (compile micro-repro).

## Synthesis

<!-- To be filled during study -->

### Mental model

### Key concepts

### Cross-resource agreement / disagreement

### Practical checklist

### Gotchas and failure modes

### Experiments to run
