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

## Synthesis

<!-- To be filled during study -->

### Mental model

### Key concepts

### Cross-resource agreement / disagreement

### Practical checklist

### Gotchas and failure modes

### Experiments to run
