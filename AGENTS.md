# AGENTS.md — Project Architecture & Preferences

This file is a persistent reference for AI agents (and contributors) working
on this project. It describes the architecture, conventions, and the owner's
working preferences as they evolve. Update this file when anything significant
changes.

---

## Project Overview

**Goal:** Compare the performance of classical neural networks, quantum neural
networks (QNNs), and hybrid classical-quantum models on a binary classification
task. Intended as a university research project paper.

**Task:** Binary classification on sklearn's `make_moons` dataset — two
interleaved crescent-shaped point clouds that are not linearly separable.

**Stack:**
- Python 3.10+
- PyTorch — classical layers, optimizer, autograd
- PennyLane — quantum circuit simulation via `default.qubit` (CPU)
- scikit-learn — dataset generation and preprocessing
- matplotlib — visualization

---

## File Architecture

```
quantum-honors/
├── data.py        Dataset generation, train/test split, normalization
├── models.py      All three model classes + shared quantum circuit definition
├── train.py       Shared training loop (BCELoss + Adam, mini-batch)
├── evaluate.py    Test accuracy, parameter count, plots, summary table
├── main.py        Entry point — runs the full experiment end to end
├── Makefile       Convenience commands (install, run, clean)
├── requirements.txt
├── AGENTS.md      This file
├── README.md
└── docs/
    ├── overview.md         High-level project explanation
    ├── quantum_primer.md   Background on quantum computing concepts used
    ├── models.md           Detailed breakdown of each model
    └── results_guide.md    How to read and interpret the output
```

---

## Model Architecture Summary

### ClassicalNN
```
Input(2) → Linear(2→8) → ReLU → Linear(8→4) → ReLU → Linear(4→1) → Sigmoid
```
Intentionally small — comparable parameter count to the QNN for a fair comparison.

### QNN
```
Input(2) → AngleEmbedding(2 qubits) → StronglyEntanglingLayers(3 layers) → PauliZ expval → rescale → Output(1)
```
No classical layers. Trainable parameters are exclusively the quantum rotation
angles. Uses PennyLane `TorchLayer` to integrate with PyTorch autograd.
Quantum gradients computed via the parameter-shift rule.

### HybridNN
```
Input(2) → Linear(2→2)+Tanh → AngleEmbedding → StronglyEntanglingLayers → PauliZ expval → rescale → Linear(1→1)+Sigmoid → Output(1)
```
Classical pre-processing learns how to present data to the quantum circuit.
Classical post-processing learns to calibrate the quantum measurement output.

---

## Key Design Decisions

- **N_QUBITS = 2**: One qubit per input feature. Defined in `models.py`.
- **N_LAYERS = 3**: Depth of `StronglyEntanglingLayers`. More layers = more
  expressiveness but slower simulation and more risk of barren plateaus.
- **Tanh in Hybrid pre-layer**: Outputs in (-1,1), aligns well with rotation
  angle inputs for AngleEmbedding. ReLU would discard negative values.
- **Separate hyperparameters per model**: QNN trains slower, so it uses fewer
  epochs and a higher learning rate. See `CONFIG` dict in `main.py`.
- **Shared quantum circuit**: Both QNN and HybridNN use the same `quantum_circuit`
  qnode. This keeps things DRY and makes comparisons more controlled.

---

## Owner Preferences

*Updated as the project evolves. These guide how AI agents should approach
work on this codebase.*

- **Explain everything thoroughly.** The owner is learning quantum computing
  concepts through this project. When adding or modifying code, comments
  should explain the *why*, not just the *what*.
- **No unnecessary abstractions.** Don't create helper functions or classes
  for things that only happen once.
- **No silent changes.** Don't refactor surrounding code while fixing a bug
  or adding a feature. Change only what was asked.
- **Paper-first mindset.** Code decisions should be defensible in a research
  context. Favor clarity and reproducibility over cleverness.
- **Iterative.** Start with working boilerplate, then tune. Don't over-engineer
  the first pass.
- **QGANs are a future interest.** The owner expressed interest in quantum
  generative adversarial networks as a follow-up after this project.

---

## Experiment Parameters (current defaults)

| Model     | Epochs | LR    | Batch Size |
|-----------|--------|-------|------------|
| Classical | 100    | 0.01  | 32         |
| QNN       | 60     | 0.05  | 16         |
| Hybrid    | 60     | 0.02  | 16         |

Dataset: 200 samples, noise=0.1, 80/20 train/test split, seed=42.

---

## Known Limitations / Paper Considerations

- Running on CPU simulator — real quantum hardware would show very different
  timing and noise characteristics.
- Barren plateau problem: QNN gradients can vanish with deeper circuits.
  N_LAYERS=3 is a pragmatic compromise.
- Single seed, single dataset size — production research would average over
  multiple seeds and dataset sizes.
- `make_moons` is a toy dataset. Results should be contextualized accordingly.

---

## Future Work

- [ ] Add QGAN comparison (owner interest noted)
- [ ] Sweep over N_LAYERS to show expressiveness vs trainability tradeoff
- [ ] Add noise model to QNN to simulate real quantum hardware
- [ ] Multiple seeds + confidence intervals for paper rigor
- [ ] Benchmark on a second dataset (e.g., circles, blobs)
