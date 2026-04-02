# Model Deep Dive

This document explains the design of each model in detail — what each component
does, why it was chosen, and what it means for the experiment.

---

## Shared Setup: The Quantum Circuit

Both the QNN and HybridNN share the same underlying quantum circuit, defined
once in `models.py` as a PennyLane `qnode`:

```python
@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS))
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return qml.expval(qml.PauliZ(0))
```

The `@qml.qnode` decorator transforms this Python function into a quantum
node — a callable that, when invoked, actually runs the quantum circuit on
the simulator and returns a result. The `interface="torch"` argument is what
makes gradients flow between PyTorch and the quantum circuit.

**N_QUBITS = 2** — one qubit per input feature. This is a principled choice:
each qubit "owns" one feature's information and the CNOT gates then allow
correlations to be learned between features.

**N_LAYERS = 3** — chosen as a balance between expressiveness and trainability.
Too few layers and the circuit can't represent complex functions. Too many and
gradients vanish (barren plateaus). See `docs/quantum_primer.md`.

---

## Model 1: ClassicalNN

```
Input(2) → Linear(2→8) → ReLU → Linear(8→4) → ReLU → Linear(4→1) → Sigmoid
```

**Component breakdown:**

| Component | Role |
|---|---|
| `Linear(2, 8)` | Projects 2D input into 8D space. Learns a weighted combination of x and y. |
| `ReLU` | Non-linearity. Without this, stacking Linear layers is equivalent to one Linear layer. |
| `Linear(8, 4)` | Compresses 8 features down to 4. Learns higher-level combinations. |
| `ReLU` | Non-linearity again. |
| `Linear(4, 1)` | Final projection to a single score. |
| `Sigmoid` | Squashes score to (0, 1) → interpreted as P(class=1). |

**Parameter count:**
- Linear(2→8):  2×8 weights + 8 biases = 24
- Linear(8→4):  8×4 weights + 4 biases = 36
- Linear(4→1):  4×1 weights + 1 bias   = 5
- **Total: 65 parameters**

**Why this size?** Intentionally kept small to make the comparison fair. A
10,000-parameter classical network would trivially outperform an 18-parameter
QNN. The goal is to compare architecturally analogous models.

---

## Model 2: QNN

```
Input(2) → AngleEmbedding → StronglyEntanglingLayers → PauliZ measurement → rescale → Output(1)
```

**The full forward pass, step by step:**

```python
def forward(self, x):
    out = self.qlayer(x)      # (batch,)  → range [-1, 1]
    out = (out + 1) / 2       # rescale   → range [0, 1]
    return out.unsqueeze(1)   # (batch, 1)
```

**Step 1 — qlayer:**  
The `TorchLayer`-wrapped quantum circuit runs on the input. Internally, each
sample in the batch runs through:
1. `AngleEmbedding` encodes the 2 features as rotation angles on 2 qubits
2. 3 layers of `StronglyEntanglingLayers` apply parameterized rotations + CNOTs
3. `expval(PauliZ(0))` extracts a number in [-1, 1]

**Step 2 — rescale:**  
Binary Cross-Entropy loss requires inputs in [0, 1]. The linear map
`(x + 1) / 2` transforms [-1, 1] → [0, 1] without distorting the ordering.

**Step 3 — unsqueeze:**  
The quantum circuit outputs shape `(batch,)`. Our labels are `(batch, 1)`.
`unsqueeze(1)` adds the missing dimension.

**Parameter count:**
- StronglyEntanglingLayers: N_LAYERS × N_QUBITS × 3 = 3 × 2 × 3 = **18 parameters**

All 18 are rotation angles inside the quantum circuit. There are no bias
terms — quantum gates don't have biases.

**Key limitation:** Input features are embedded directly with no transformation.
The circuit must learn to classify the raw (standardized) features as-is.
If the quantum circuit's native geometry doesn't align with the data manifold,
this is a significant handicap.

---

## Model 3: HybridNN

```
Input(2) → Linear(2→2)+Tanh  →  quantum circuit  →  rescale  →  Linear(1→1)+Sigmoid → Output(1)
           ↑── classical pre ──↑  ↑──── quantum ────↑             ↑── classical post ──↑
```

**The full forward pass, step by step:**

```python
def forward(self, x):
    x = self.pre(x)           # classical: (batch, 2) → (batch, 2)
    x = self.qlayer(x)        # quantum:   (batch, 2) → (batch,)
    x = (x + 1) / 2          # rescale:   [-1,1]     → [0,1]
    x = x.unsqueeze(1)        # reshape:   (batch,)   → (batch, 1)
    x = self.post(x)          # classical: (batch, 1) → (batch, 1)
    return x
```

**The pre-processing layer:**
```
Linear(2, 2) + Tanh
```
This is a learned 2×2 linear transformation followed by Tanh. It can
represent any rotation, scaling, or shearing of the 2D input space. The Tanh
keeps outputs in (-1, 1), which is a natural range for rotation angles.

Think of it as: *the classical layer learns how to hand the data to the
quantum circuit in the best possible way.*

**Why Tanh instead of ReLU?**  
ReLU(x) = max(0, x) — clips all negative values to zero. For a 2D input
that has been standardized to zero mean, roughly half the values are negative.
Clipping them to zero before they reach the quantum circuit would throw away
half the information. Tanh preserves the sign while bounding the magnitude.

**The quantum layer:**  
Identical to the QNN's quantum circuit. Same N_QUBITS=2, same N_LAYERS=3,
same 18 trainable quantum parameters.

**The post-processing layer:**
```
Linear(1, 1) + Sigmoid
```
Takes the single quantum measurement output (after rescaling to [0,1]) and
applies a learned linear transformation + Sigmoid. This lets the model
learn to recalibrate the quantum output's scale and shift.

In practice, this adds just 2 parameters (1 weight + 1 bias) but can
significantly help when the quantum circuit's output distribution doesn't
naturally span the full [0, 1] range.

**Parameter count:**
- pre  Linear(2→2):  2×2 weights + 2 biases = 6
- quantum circuit:   18 weights              = 18
- post Linear(1→1):  1×1 weight  + 1 bias   = 2
- **Total: 26 parameters**

**The key insight for the paper:**  
The Hybrid model has only 26 parameters (vs 65 for Classical), yet the
classical layers give it the geometric flexibility to align the data with
the quantum circuit's native structure. Whether this advantage shows up in
accuracy is the empirical question.

---

## Side-by-Side Comparison

| Property | ClassicalNN | QNN | HybridNN |
|---|---|---|---|
| Parameters | 65 | 18 | 26 |
| Classical layers | 3 Linear | None | 2 Linear (pre + post) |
| Quantum layers | None | 1 circuit (3 layers) | 1 circuit (3 layers) |
| Gradient method | Backprop | Parameter-shift | Both |
| Expressiveness | Medium | Constrained by circuit depth | Higher than QNN alone |
| Training speed | Fast | Slow (2× circuit evals per param) | Slow (quantum bottleneck) |
| Sensitivity to input scale | Low (ReLU tolerant) | High (angle encoding is periodic) | Medium (Tanh normalizes first) |
