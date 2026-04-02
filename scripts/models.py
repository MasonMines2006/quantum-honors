"""
models.py — All three model definitions

This file defines three classifiers that all solve the same task:
given a 2D point, predict whether it belongs to class 0 or class 1.

  1. ClassicalNN  — standard feedforward neural network
  2. QNN          — fully quantum parameterized circuit
  3. HybridNN     — classical preprocessing + quantum circuit + classical postprocessing

The quantum models use PennyLane's TorchLayer to make the quantum circuit
behave like a standard PyTorch nn.Module — meaning it participates in
automatic differentiation (autograd) and gradient-based training just like
a normal layer would.
"""

import torch
import torch.nn as nn
import pennylane as qml


# ─────────────────────────────────────────────────────────────────────────────
# Shared quantum circuit configuration
# ─────────────────────────────────────────────────────────────────────────────

N_QUBITS = 2   # One qubit per input feature (we have 2 features: x and y)
N_LAYERS = 3   # How many layers of parameterized gates to stack

# A PennyLane "device" is the backend that simulates or runs the circuit.
# "default.qubit" is the built-in exact statevector simulator — runs on CPU,
# no quantum hardware needed. "wires" is just PennyLane's word for qubits.
dev = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    """
    The parameterized quantum circuit used by both QNN and HybridNN.

    Think of this like a function: it takes input features and learnable
    weights, runs a sequence of quantum operations, and returns a number.

    Args:
        inputs   shape (2,)                   — the 2 input features
        weights  shape (N_LAYERS, N_QUBITS, 3) — trainable rotation angles

    Returns:
        A single float in [-1, 1]: the expectation value of PauliZ on qubit 0.

    ── Step 1: AngleEmbedding ───────────────────────────────────────────────
    Encodes each input feature as a rotation angle on its qubit.
    Feature 0 rotates qubit 0 by inputs[0] radians around the X axis.
    Feature 1 rotates qubit 1 by inputs[1] radians around the X axis.

    This is the quantum analog of passing data into a neural network layer.
    The difference is that on a qubit, a "rotation" changes its quantum state,
    which lives in a 2D complex vector space (a Bloch sphere).

    ── Step 2: StronglyEntanglingLayers ─────────────────────────────────────
    Applies N_LAYERS rounds of:
      a) A parameterized Rot gate on each qubit.
         Rot(φ, θ, ω) = Rz(ω) · Ry(θ) · Rz(φ)
         This is the most general single-qubit rotation — 3 angles per qubit.
      b) CNOT gates that entangle neighboring qubits.
         Entanglement lets the circuit learn correlations between features
         that a single-qubit approach couldn't capture.

    This is the quantum analog of hidden layers in a neural network.
    The weights ARE the trainable parameters, just like Linear layer weights.

    ── Step 3: Measurement ──────────────────────────────────────────────────
    Returns the *expectation value* of the Pauli-Z operator on qubit 0.
    Pauli-Z measures whether the qubit is closer to |0⟩ (+1) or |1⟩ (-1).
    The expectation value is the average over many repeated measurements,
    which gives a continuous value in [-1, 1] — perfect as a classifier output.
    """
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS))
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return qml.expval(qml.PauliZ(0))


# ─────────────────────────────────────────────────────────────────────────────
# 1. Classical Neural Network
# ─────────────────────────────────────────────────────────────────────────────

class ClassicalNN(nn.Module):
    """
    A standard feedforward (fully-connected) neural network.

    Architecture:   Input(2) → Linear(8) → ReLU → Linear(4) → ReLU → Linear(1) → Sigmoid

    The sigmoid on the final output squashes the value to (0, 1), which we
    interpret as P(class = 1). If this probability > 0.5, predict class 1.

    Why these layer sizes? This is intentionally small — we want a fair
    comparison with the QNN which has limited expressiveness. Using a massive
    classical network would make the comparison unfair.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),    # 2 inputs → 8 hidden units
            nn.ReLU(),
            nn.Linear(8, 4),    # 8 → 4 hidden units
            nn.ReLU(),
            nn.Linear(4, 1),    # 4 → 1 output (logit)
            nn.Sigmoid()        # squash to (0, 1)
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Quantum Neural Network (QNN)
# ─────────────────────────────────────────────────────────────────────────────

class QNN(nn.Module):
    """
    A fully quantum classifier — no classical layers.

    The raw input features are fed directly into the quantum circuit
    via AngleEmbedding. The only trainable parameters are the rotation
    angles inside the quantum circuit itself.

    How TorchLayer works:
        qml.qnn.TorchLayer wraps the qnode (quantum circuit function) so it
        registers its weights as nn.Parameters. This means PyTorch's optimizer
        can update them just like any other trainable weight. Gradients are
        computed via the "parameter-shift rule" — a quantum analog of backprop.

    Output rescaling:
        The circuit returns a value in [-1, 1] (PauliZ expectation).
        Binary cross-entropy loss (BCELoss) expects values in [0, 1].
        So we apply:  output = (raw + 1) / 2
        This maps:  -1 → 0,  0 → 0.5,  +1 → 1
    """
    def __init__(self):
        super().__init__()
        # weight_shapes tells TorchLayer how to initialize the circuit weights.
        # StronglyEntanglingLayers needs (n_layers, n_qubits, 3) because each
        # qubit gets 3 Euler rotation angles per layer.
        weight_shapes = {"weights": (N_LAYERS, N_QUBITS, 3)}
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

    def forward(self, x):
        out = self.qlayer(x)      # quantum circuit → shape (batch,), range [-1, 1]
        out = (out + 1) / 2       # rescale to [0, 1]
        return out.unsqueeze(1)   # shape (batch, 1) to match y_train shape


# ─────────────────────────────────────────────────────────────────────────────
# 3. Hybrid Neural Network
# ─────────────────────────────────────────────────────────────────────────────

class HybridNN(nn.Module):
    """
    A hybrid classical-quantum classifier.

    This is the most interesting model for the paper — it uses classical
    layers to do work that quantum circuits are bad at, and quantum layers
    to do work that classical networks may be inefficient at.

    Architecture:
        [Classical pre]   Linear(2→2) + Tanh
        [Quantum]         AngleEmbedding + StronglyEntanglingLayers (2 qubits, N_LAYERS)
        [Classical post]  Linear(1→1) + Sigmoid

    Why Tanh in the pre-processing layer?
        Tanh outputs values in (-1, 1), which maps naturally to rotation angles
        in AngleEmbedding. Using ReLU here would clip negative values to 0,
        potentially losing half the information before it even reaches the circuit.

    What the classical pre-layer does:
        It learns a linear transformation of the 2D input. This means it can
        rotate, scale, or shear the input space before the quantum circuit sees
        it. In effect, it learns "how to present data to the quantum circuit"
        in a way that makes the quantum processing maximally useful.

    What the classical post-layer does:
        It takes the single quantum measurement output and applies a learned
        linear rescaling + sigmoid before producing the final probability.
        This gives the model one more degree of freedom to calibrate its output.

    Gradient flow:
        Because TorchLayer integrates with PyTorch autograd, gradients flow
        backwards through the post layer → quantum circuit → pre layer in a
        single backward() call. The quantum gradients are computed via the
        parameter-shift rule and handed back to autograd seamlessly.
    """
    def __init__(self):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Linear(2, 2),
            nn.Tanh()
        )
        weight_shapes = {"weights": (N_LAYERS, N_QUBITS, 3)}
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        self.post = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.pre(x)           # classical feature transformation  (batch, 2)
        x = self.qlayer(x)        # quantum processing                (batch,)
        x = (x + 1) / 2          # rescale [-1,1] → [0,1]
        x = x.unsqueeze(1)        # (batch, 1)
        x = self.post(x)          # classical postprocessing           (batch, 1)
        return x
