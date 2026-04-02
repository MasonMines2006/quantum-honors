"""
models.py — All three model definitions

This file defines three classifiers that all solve the same task:
given a 2D point, predict whether it belongs to class 0 or class 1.

  1. ClassicalNN  — standard feedforward neural network (unchanged baseline)
  2. QNN          — improved quantum circuit with data re-uploading + dual measurement
  3. HybridNN     — wider classical preprocessing + improved quantum circuit + classical postprocessing

Improvements over the original architecture (v1):

  [1] N_LAYERS 3 → 4
      More layers = wider Fourier spectrum = richer function class the circuit
      can represent. Still safe from barren plateaus at 2 qubits.

  [2] Data re-uploading (Pérez-Salinas et al. 2020)
      The original circuit encoded inputs once at the very start. This limits
      the circuit to a Fourier series with frequencies determined only by the
      circuit depth — not by the data. Re-uploading encodes the inputs again
      at the start of every layer, which effectively multiplies up the accessible
      Fourier frequencies and makes the quantum model genuinely non-linear.
      Think of it like polynomial features in classical ML: encoding x once gives
      you linear functions of x; encoding it N times (interleaved with
      entangling operations) gives you degree-N polynomial-like functions.

  [3] Measure both qubits, not just qubit 0
      The original circuit discarded all information about qubit 1 at measurement
      time. The entangling CNOT gates spread information across both qubits —
      throwing away qubit 1's state wasted that computation. Measuring both
      gives a 2-dimensional output vector, which is then combined by a classical
      linear layer.

  [4] Wider classical pre-layer in HybridNN (2 → 4 → 2)
      The original pre-layer was Linear(2→2)+Tanh, which can only apply a
      linear transformation (rotation/scale/shear) before Tanh. A 2×2 linear
      map can't create a non-linear realignment of the data. Adding a hidden
      dimension (2→4→2) gives the pre-layer true non-linear capacity — it can
      now learn to reshape the data manifold, not just rotate it.
"""

import torch
import torch.nn as nn
import pennylane as qml
from pennylane.qnn.torch import TorchLayer


# ─────────────────────────────────────────────────────────────────────────────
# Shared quantum circuit configuration
# ─────────────────────────────────────────────────────────────────────────────

N_QUBITS = 2   # One qubit per input feature (x and y)
N_LAYERS = 4   # Increased from 3: more layers → wider Fourier spectrum

# "default.qubit" is the built-in exact statevector simulator.
# "wires" is PennyLane's word for qubits.
dev = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(dev, interface="torch")
def quantum_circuit_reupload(inputs, weights):
    """
    Parameterized quantum circuit with data re-uploading and dual measurement.

    Args:
        inputs   shape (2,)                    — the 2 input features
        weights  shape (N_LAYERS, N_QUBITS, 3) — trainable rotation angles

    Returns:
        A list of 2 floats, each in [-1, 1]:
        [expval(PauliZ on qubit 0), expval(PauliZ on qubit 1)]
        TorchLayer converts this to a (batch, 2) tensor.

    ── What changed vs the original circuit ────────────────────────────────

    Original:
        AngleEmbedding(inputs)              ← encode once
        StronglyEntanglingLayers(weights)   ← all layers at once
        expval(PauliZ(0))                   ← measure only qubit 0

    New (this circuit):
        for each layer:
            AngleEmbedding(inputs)          ← re-encode every layer
            StronglyEntanglingLayers(       ← one layer at a time
                weights[layer_idx])
        expval(PauliZ(0))                   ← measure qubit 0
        expval(PauliZ(1))                   ← AND qubit 1

    ── Why re-uploading works ───────────────────────────────────────────────

    A quantum circuit with a single AngleEmbedding can only represent
    functions whose Fourier coefficients have frequencies {0, ±1} per qubit
    (the "data encoding spectrum"). Re-uploading the same input N times,
    interleaved with entangling layers, expands this to frequencies
    {0, ±1, ±2, ..., ±N} per qubit — much richer, and enough to separate
    the interleaved crescents of make_moons.

    Classically, this is analogous to transforming feature x into
    [x, x², x³, ...] before fitting a linear model: same linear machinery,
    but now capable of fitting non-linear functions.

    ── Why measuring both qubits helps ─────────────────────────────────────

    The CNOT entangling gates create correlations between qubit 0 and qubit 1.
    Measuring only qubit 0 was like computing a dot product and then throwing
    away one of the two output dimensions. The post-processing layer (Linear
    2→1) now learns the optimal weighted combination of both measurements.
    """
    for layer_idx in range(N_LAYERS):
        # Re-encode: inject the input features as rotation angles on each qubit.
        # This is what "data re-uploading" means — the same data enters the
        # circuit N_LAYERS times, once per layer, not just once at the start.
        qml.AngleEmbedding(inputs, wires=range(N_QUBITS))

        # Apply one layer of parameterized rotations + CNOT entanglement.
        # weights[layer_idx:layer_idx+1] slices a single layer (shape (1, N_QUBITS, 3))
        # so StronglyEntanglingLayers treats it as exactly 1 layer.
        qml.StronglyEntanglingLayers(
            weights[layer_idx : layer_idx + 1], wires=range(N_QUBITS)
        )

    # Measure both qubits. PauliZ expectation is in [-1, 1]:
    #   close to +1 → qubit near |0⟩ state
    #   close to -1 → qubit near |1⟩ state
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Classical Neural Network  (unchanged — this is the baseline)
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

    Parameter count: Linear(2→8)=24, Linear(8→4)=36, Linear(4→1)=5 → Total: 65
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
    An improved fully-quantum classifier.

    Changes from v1:
      - Uses quantum_circuit_reupload: inputs re-encoded each layer
      - Measures both qubits, returning shape (batch, 2)
      - Adds a minimal post-processing layer Linear(2→1)+Sigmoid to combine
        the two qubit measurements into a final probability

    The QNN still has no classical preprocessing — the raw (standardized)
    inputs go directly into the quantum circuit. The only classical step is
    the final 2→1 combination of measurements.

    Forward pass:
        inputs (batch, 2)
            ↓  quantum_circuit_reupload
        raw (batch, 2)  ← two PauliZ expectations, range [-1, 1]
            ↓  (x + 1) / 2
        scaled (batch, 2)  ← range [0, 1]
            ↓  Linear(2→1) + Sigmoid
        output (batch, 1)  ← final probability P(class=1)

    Parameter count:
        quantum (N_LAYERS × N_QUBITS × 3): 4 × 2 × 3 = 24
        post Linear(2→1): 2 weights + 1 bias = 3
        Total: 27 parameters
    """
    def __init__(self):
        super().__init__()
        weight_shapes = {"weights": (N_LAYERS, N_QUBITS, 3)}
        self.qlayer = TorchLayer(quantum_circuit_reupload, weight_shapes)

        # Combines the 2 qubit measurements into a single classification score.
        # Linear(2→1) learns: "how much does each qubit's result matter?"
        self.post = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.qlayer(x)      # quantum circuit → shape (batch, 2), range [-1, 1]
        out = (out + 1) / 2       # rescale to [0, 1] for numerical stability
        out = self.post(out)      # combine measurements → shape (batch, 1)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 3. Hybrid Neural Network
# ─────────────────────────────────────────────────────────────────────────────

class HybridNN(nn.Module):
    """
    An improved hybrid classical-quantum classifier.

    Changes from v1:
      - Pre-layer widened: Linear(2→2) → Linear(2→4→2) with Tanh at each step.
        This gives the pre-layer genuine non-linear capacity. The original
        Linear(2→2) could only apply a rotation/scale/shear — a 2×2 matrix
        is still just a linear map. Going through a 4D hidden space lets the
        classical layer learn a non-linear reshaping of the input manifold
        before the quantum circuit sees it.
      - Uses quantum_circuit_reupload: inputs re-encoded each layer
      - Post-layer updated: Linear(1→1) → Linear(2→1) to consume both qubit
        measurements

    Architecture:
        [Classical pre]   Linear(2→4) + Tanh + Linear(4→2) + Tanh
        [Quantum]         data re-uploading circuit (N_LAYERS rounds of encode+entangle)
        [Classical post]  Linear(2→1) + Sigmoid

    Why Tanh at both pre-layer activations?
        Tanh bounds outputs to (-1, 1), which is a natural range for
        AngleEmbedding rotation angles. ReLU would clip all negative values to
        zero, discarding information about which direction a feature points.

    Why a hidden dimension of 4 in the pre-layer?
        The pre-layer now has the shape: 2 → 4 → 2
        This is the same idea as an "autoencoder bottleneck" — expand to a
        higher-dimensional space to let the network mix features non-linearly,
        then compress back down to the 2D space the quantum circuit expects.
        A 2→2 linear map can only rotate/scale; a 2→4→2 non-linear map can
        learn arbitrary continuous transformations in 2D.

    Forward pass:
        inputs (batch, 2)
            ↓  Linear(2→4) + Tanh + Linear(4→2) + Tanh
        transformed (batch, 2)  ← non-linearly re-shaped features
            ↓  quantum_circuit_reupload
        raw (batch, 2)  ← two PauliZ expectations, range [-1, 1]
            ↓  (x + 1) / 2
        scaled (batch, 2)
            ↓  Linear(2→1) + Sigmoid
        output (batch, 1)  ← final probability P(class=1)

    Parameter count:
        pre  Linear(2→4):  2×4 weights + 4 biases = 12
        pre  Linear(4→2):  4×2 weights + 2 biases = 10
        quantum (N_LAYERS × N_QUBITS × 3): 4 × 2 × 3 = 24
        post Linear(2→1):  2×1 weights + 1 bias   = 3
        Total: 49 parameters
    """
    def __init__(self):
        super().__init__()

        # Wider pre-processing: 2 → 4 → 2 with non-linear Tanh at each step.
        # The 4D hidden layer lets the network learn non-linear combinations
        # of the input features before they reach the quantum circuit.
        self.pre = nn.Sequential(
            nn.Linear(2, 4),    # expand to 4D hidden space
            nn.Tanh(),
            nn.Linear(4, 2),    # compress back to 2D for AngleEmbedding
            nn.Tanh()           # bound to (-1,1) — natural range for rotation angles
        )

        weight_shapes = {"weights": (N_LAYERS, N_QUBITS, 3)}
        self.qlayer = TorchLayer(quantum_circuit_reupload, weight_shapes)

        # Combines both qubit measurements into the final classification score.
        self.post = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.pre(x)           # classical feature transformation  (batch, 2)
        x = self.qlayer(x)        # quantum processing                (batch, 2)
        x = (x + 1) / 2          # rescale [-1,1] → [0,1]
        x = self.post(x)          # combine measurements              (batch, 1)
        return x
