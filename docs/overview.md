# Project Overview

## The Question This Project Asks

Can quantum neural networks do what classical neural networks do — and if so,
at what cost? And does combining the two (a hybrid model) give us something
better than either alone?

These aren't new questions in research, but they're still very much open ones.
The honest answer from the literature is: it depends. Quantum models sometimes
win on specific structured problems, but classical models are hard to beat on
generic tasks — especially when you're simulating the quantum circuit on
classical hardware. This project is a controlled experiment to see what
happens in practice on a small, well-understood problem.

---

## The Dataset: make_moons

We use sklearn's `make_moons` — a synthetic dataset of 200 2D points split
into two classes shaped like interleaved crescents:

```
  * * * *          Class 0 (top crescent)
*         *
            o o o  Class 1 (bottom crescent)
          o       o
```

Each point is described by two numbers (x-coordinate, y-coordinate). The task
is: given a point, which crescent does it belong to?

**Why this dataset?**
- It's not linearly separable — a straight line cannot divide the two classes.
  Any model that solves it has learned something genuinely non-linear.
- It's small enough to train quickly, even with the overhead of quantum simulation.
- It's a standard benchmark, so results are contextualized against known baselines.
- The 2D input maps naturally to 2 qubits (one per feature), which keeps the
  quantum circuit simple and interpretable.

---

## The Three Models

### 1. Classical Neural Network (the baseline)

A standard feedforward network with two hidden layers. This is the most
well-understood model of the three and serves as the baseline against which
the others are judged.

It works by applying learned linear transformations interleaved with non-linear
activation functions (ReLU), which together allow it to carve out arbitrarily
shaped decision regions in the input space.

### 2. Quantum Neural Network (the pure quantum model)

A parameterized quantum circuit — often called a "variational quantum circuit"
in the literature. Instead of matrix multiplications and activations, it
operates by rotating quantum bits (qubits) through a sequence of parameterized
gates, then measuring the result.

There are no classical layers at all. The input features are encoded directly
into the circuit, and the trainable weights are rotation angles inside the
quantum gates.

### 3. Hybrid Model (classical + quantum)

A model where a small classical network preprocesses the input, a quantum
circuit processes the transformed features, and another small classical network
turns the quantum output into a final prediction.

The hypothesis is that classical layers can handle the parts of the problem
they're good at (e.g., rescaling and rotating the feature space), while the
quantum circuit handles the parts it might be better at (e.g., exploring
exponentially large state spaces efficiently).

---

## What We're Measuring

| Metric | What it tells us |
|---|---|
| Test accuracy | Does the model actually learn to classify correctly? |
| Training loss curve | How smoothly and quickly does it converge? |
| Training accuracy curve | Does it learn on the training data itself? |
| Parameter count | How many numbers does each model need to learn? |
| Training time | What's the real-world cost of each approach? |

---

## The Honest Expectations

On this specific problem, the Classical NN will almost certainly win on
accuracy and training speed. That's expected and fine — it's not a flaw in
the experiment, it's a result. The interesting questions are:

- How close does the QNN get with far fewer parameters?
- Does the Hybrid model recover some of the QNN's accuracy gap?
- How much slower is quantum simulation, and what does that imply for scaling?
- What does the decision boundary of each model look like? Does the quantum
  circuit learn a different *shape* of boundary?

These are the questions worth discussing in the paper.
