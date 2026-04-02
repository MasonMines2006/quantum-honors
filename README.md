# quantum-honors

A university research project benchmarking classical neural networks, quantum
neural networks (QNNs), and hybrid classical-quantum models on a binary
classification task.

---

## What This Project Does

Three models — a standard neural network, a quantum neural network, and a
hybrid of both — are trained and evaluated on the same dataset. The goal is
to compare accuracy, training behavior, parameter efficiency, and training
cost across the three paradigms.

**Dataset:** `make_moons` (sklearn) — two interleaved crescent-shaped classes
that require a non-linear decision boundary to separate.

**Models:**
- `ClassicalNN` — standard feedforward network (PyTorch)
- `QNN` — parameterized quantum circuit (PennyLane), no classical layers
- `HybridNN` — classical preprocessing + quantum circuit + classical postprocessing

---

## Quickstart

### Option A — Docker (recommended, no local Python setup needed)

Requires [Docker Desktop](https://www.docker.com/products/docker-desktop/).

```bash
# 1. Build the image — installs all dependencies inside the container
make docker-build

# 2. Run the experiment — results.png is copied back to your current folder
make docker-run
```

First build takes ~2 minutes while pip installs everything. Subsequent builds
are instant unless `requirements.txt` changes.

### Option B — Local Python

Requires Python 3.10-3.12 (Python 3.11 recommended).

```bash
# 1. Install dependencies
make install

# 2. Run the full experiment
make run
```

If you use Python 3.13+, `torch` may not have compatible wheels in your
environment yet. In that case, create a Python 3.11 virtual environment or
use the Docker path above.

Output: training logs printed to terminal + `results.png` saved to the project root.

---

## Project Structure

```
quantum-honors/
├── Dockerfile       Container definition — builds the full environment
├── .dockerignore    Files excluded from the container image
├── Makefile         make docker-build / docker-run / install / run / clean
├── requirements.txt
├── AGENTS.md        Architecture reference and project preferences
├── scripts/
│   ├── data.py          Dataset generation and preprocessing
│   ├── models.py        All three model definitions
│   ├── train.py         Shared training loop
│   ├── evaluate.py      Metrics, plots, summary table
│   └── main.py          Entry point
└── docs/
    ├── overview.md        What this project is and why
    ├── quantum_primer.md  Background on qubits, circuits, and measurement
    ├── models.md          Deep dive into each model's design
    └── results_guide.md   How to read and interpret the output
```

---

## Dependencies

All managed automatically by Docker or `requirements.txt`:

- Python 3.11 (Docker image default)
- [PennyLane](https://pennylane.ai) — quantum circuit simulation
- PyTorch — neural network training
- scikit-learn — dataset generation
- matplotlib — visualization

---

## Expected Output

After training, you'll see a terminal summary:

```
════════════════════════════════════════════════════════════
  Model        Test Acc     Params    Time (s)
════════════════════════════════════════════════════════════
  Classical      0.9600         61        2.3s
  QNN            0.8800         18       84.7s
  Hybrid         0.9200         25       91.2s
════════════════════════════════════════════════════════════
```

And `results.png` containing decision boundary plots, training curves, and a
bar chart of test accuracy.

---

## Documentation

See the [docs/](docs/) folder for detailed explanations of the quantum
concepts, model designs, and how to interpret results.

For architecture and project conventions, see [AGENTS.md](AGENTS.md).
