# Quantum Computing Primer

This document covers only the quantum concepts that actually appear in this
project. It's not a comprehensive introduction — it's a targeted reference
so you understand exactly what the code is doing and why.

---

## Qubits

A classical bit is either 0 or 1. A qubit can be in a **superposition** of
both states simultaneously:

```
|ψ⟩ = α|0⟩ + β|1⟩
```

Where α and β are complex numbers satisfying |α|² + |β|² = 1. When you
*measure* a qubit, you get 0 with probability |α|² and 1 with probability |β|².
Before measurement, the qubit genuinely holds both possibilities at once.

**The Bloch Sphere** — a useful mental model. Every possible qubit state
maps to a point on the surface of a sphere. The north pole is |0⟩, the south
pole is |1⟩, and every other point is some superposition. A rotation gate
moves the point around the sphere.

```
         |0⟩ (north pole)
          │
   ───────●───────
          │
         |1⟩ (south pole)
```

This project uses **2 qubits** — one for each input feature. With 2 qubits
the system lives in a 4-dimensional complex vector space (the tensor product
of two Bloch spheres). Entanglement links them.

---

## Quantum Gates

Gates are reversible operations that rotate a qubit's state. The ones that
appear in this project:

### Rotation Gates (single-qubit)

**Rx(θ)** — rotates around the X axis of the Bloch sphere by angle θ.
This is what `AngleEmbedding` uses (with `rotation="X"` by default).

**Rot(φ, θ, ω) = Rz(ω) · Ry(θ) · Rz(φ)** — the most general single-qubit
rotation. Takes three angles. This is what `StronglyEntanglingLayers` applies
to each qubit per layer. Three angles fully parameterize any point on the
Bloch sphere, so this gate can rotate a qubit to literally any state.

### CNOT (two-qubit)

**Controlled-NOT** — flips qubit 1 (the "target") if and only if qubit 0
(the "control") is in state |1⟩. This is the primary source of
**entanglement** — after a CNOT, the two qubits' states become correlated
in a way that can't be described by treating them independently.

Entanglement is important because it allows the circuit to learn joint
patterns across features, not just patterns in each feature independently.

---

## Angle Embedding

```python
qml.AngleEmbedding(inputs, wires=range(N_QUBITS))
```

This is how data enters the quantum circuit. Each input feature is encoded
as a rotation angle applied to its corresponding qubit:

```
feature 0  →  Rx(feature_0) on qubit 0
feature 1  →  Rx(feature_1) on qubit 1
```

After this operation, the quantum state of the circuit "contains" the input
data — it's been encoded into the rotation of the qubits.

**Why angles?** Quantum gates are parameterized by angles. Encoding classical
data as rotation angles is one of the most natural ways to inject classical
information into a quantum circuit.

---

## Strongly Entangling Layers

```python
qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
```

This is the trainable part of the circuit — the quantum analog of hidden
layers in a neural network. Each layer consists of:

1. A `Rot(φ, θ, ω)` gate on every qubit (parameterized, learnable)
2. CNOT gates connecting neighboring qubits (fixed structure, creates entanglement)

Repeated N_LAYERS times. The weights tensor has shape `(N_LAYERS, N_QUBITS, 3)`
because each qubit gets 3 angles (φ, θ, ω) per layer.

With N_LAYERS=3 and N_QUBITS=2, that's 3 × 2 × 3 = **18 trainable parameters**.

Visually, one layer looks like:

```
qubit 0: ──[Rot(φ,θ,ω)]──●──────────────
                          │
qubit 1: ──[Rot(φ,θ,ω)]──⊕──[Rot(φ,θ,ω)]──
```

---

## Measurement: Expectation Value

```python
return qml.expval(qml.PauliZ(0))
```

After all the gates, we need to extract a classical number from the quantum
state to use as a prediction.

**PauliZ** is an observable. On a single qubit, it has eigenvalues +1 (if
the qubit is |0⟩) and -1 (if the qubit is |1⟩).

**Expectation value** ⟨Z⟩ = P(qubit=0) × (+1) + P(qubit=1) × (-1)

This gives a continuous number in [-1, 1]:
- If the circuit drives qubit 0 strongly toward |0⟩, we get close to +1
- If it drives qubit 0 toward |1⟩, we get close to -1
- If it's uncertain, we get something near 0

We then rescale this to [0, 1] with `(output + 1) / 2` so it can be used
with Binary Cross-Entropy loss, which expects values in [0, 1].

---

## Parameter-Shift Rule (how gradients work)

Classical neural networks compute gradients via backpropagation — a chain
rule application through differentiable operations.

Quantum circuits can't use backprop directly because the operations are
unitary matrices over complex vector spaces, not simple arithmetic. Instead,
PennyLane uses the **parameter-shift rule**:

```
dL/dθ = [ L(θ + π/2) - L(θ - π/2) ] / 2
```

For each trainable angle θ, you run the circuit *twice* with θ shifted
forward and backward by π/2, and take the difference. This gives an exact
gradient — not an approximation. It's more expensive than backprop (2 circuit
evaluations per parameter) but it's mathematically exact.

With 18 parameters in our QNN, each training step requires up to 36 circuit
evaluations. This is why QNN training is so much slower than classical NN
training, even on this tiny problem.

---

## Barren Plateaus (why QNNs are hard to train)

As circuits get deeper and wider, gradients computed by the parameter-shift
rule tend to **vanish exponentially** — they become so small that the optimizer
can't tell which direction to move the parameters. This is called the
**barren plateau problem**.

It's the quantum analog of the vanishing gradient problem in deep networks —
but worse, because it scales exponentially with the number of qubits rather
than linearly with depth.

With N_LAYERS=3 and N_QUBITS=2, we're well within the tractable regime. If
you increase N_LAYERS to 10 or N_QUBITS to 6, expect training to fail silently
(loss barely moves). This is worth mentioning in the paper as a fundamental
limitation of current QNN approaches.
