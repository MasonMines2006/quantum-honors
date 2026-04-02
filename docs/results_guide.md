# Reading and Interpreting Results

After running `python -m scripts.main` (or `make run`), you get two outputs:
a printed summary table in the terminal, and `results.png` saved to the
project root. This document explains what each part means and what to
look for when writing up results.

---

## The Terminal Summary Table

```
════════════════════════════════════════════════════════════
  Model        Test Acc     Params    Time (s)
════════════════════════════════════════════════════════════
  Classical      0.9600         65        2.3s
  QNN            0.8800         18       84.7s
  Hybrid         0.9200         26       91.2s
════════════════════════════════════════════════════════════
```

**Test Accuracy** — the fraction of test set points classified correctly.
Since the test set has 40 points (20% of 200), each misclassified point
changes accuracy by 0.025. Don't over-interpret small differences — on a
40-point test set, 0.90 vs 0.92 is just 1 point.

**Params** — total number of trainable parameters. This is the model's
"budget." Comparing accuracy-per-parameter across models is often more
insightful than comparing raw accuracy.

**Time (s)** — wall-clock training time. Note that:
- Classical time scales with epoch count and dataset size (linearly-ish).
- QNN time scales with parameter count × 2 (parameter-shift rule) × batch
  count × epochs. It's fundamentally slower on classical hardware.
- If you double N_LAYERS, expect training time to roughly double too.

---

## results.png Layout

The figure has two rows and three columns:

```
┌─────────────────┬─────────────────┬─────────────────┐
│  Classical      │  QNN            │  Hybrid         │  ← Decision boundaries
│  boundary       │  boundary       │  boundary       │
├─────────────────┼─────────────────┼─────────────────┤
│  Training loss  │  Training acc   │  Test accuracy  │
│  curves         │  curves         │  bar chart      │
└─────────────────┴─────────────────┴─────────────────┘
```

---

## Row 1: Decision Boundary Plots

Each plot shows:
- **Background color gradient** — the model's predicted probability for
  class 1 at each point in the feature space. Deep red = high confidence
  class 1, deep blue = high confidence class 0.
- **Solid black line** — the decision boundary (where predicted probability = 0.5).
  Points on one side are classified as 0, points on the other as 1.
- **Scatter points** — the actual training data, colored by true class.

**What to look for:**

A good model has a boundary that:
- Passes cleanly between the two crescents
- Is smooth (not jagged/noisy)
- Doesn't hug individual points too tightly (overfitting)

**ClassicalNN** typically shows a smooth, curved boundary that correctly
separates the crescents. The ReLU network carves out piecewise-linear regions
that approximate the curve.

**QNN** may show a smoother but less precisely placed boundary, or a rounder
boundary that doesn't fully capture the crescent shape. The quantum circuit
is constrained in what decision boundaries it can represent — it essentially
learns a function that's a weighted sum of cosines (Fourier features), which
may not match the crescent geometry perfectly.

**HybridNN** should produce a boundary between the two — ideally closer to
the Classical boundary because the pre-processing layer has aligned the
feature space to the quantum circuit's strengths.

---

## Row 2, Column 1: Training Loss Curves

**X axis:** Training epoch number.  
**Y axis:** Average Binary Cross-Entropy loss on the training set.

**What healthy training looks like:**
- Loss starts high and decreases monotonically (or near-monotonically)
- Flattens out as training converges
- Doesn't spike or oscillate wildly

**Things to watch for:**

*Classical* should descend smoothly and quickly.

*QNN* may show a noisier descent. This is normal — quantum gradients are
computed via the parameter-shift rule which involves circuit evaluations
with real randomness. The gradient estimates are correct on average but
noisier per step than classical backprop.

*QNN plateau early* — if the QNN's loss barely moves for many epochs, this
is a sign of a barren plateau (vanishing gradients). With N_LAYERS=3 and
N_QUBITS=2 this shouldn't happen, but it can if you increase the circuit size.

---

## Row 2, Column 2: Training Accuracy Curves

**X axis:** Training epoch.  
**Y axis:** Fraction of training points classified correctly.

This tells the same story as the loss curve but in a more interpretable unit.
A model at 0.50 accuracy is no better than random guessing. A model at 0.95
is getting almost everything right on the training set.

**Gap between training and test accuracy:** If training accuracy is much higher
than test accuracy (from the bar chart), the model has overfit. With only 160
training points, this is a real risk for the Classical model.

---

## Row 2, Column 3: Test Accuracy Bar Chart

Each bar shows final test accuracy for one model. The number is printed above
the bar.

This is the headline result for your paper — but a single number needs context:

1. **Is the difference meaningful?** With 40 test points, a 5% difference
   is only 2 points. Run with multiple seeds (change `seed=` in `get_data()`)
   and report average ± standard deviation to make the comparison rigorous.

2. **What's the baseline?** Random guessing = 50%. A model that always predicts
   the majority class gets ~50% on a balanced dataset. Your models should be
   significantly above this to be meaningful.

3. **Compare accuracy/parameter:** Divide test accuracy by parameter count.
   The QNN may "lose" on raw accuracy but "win" on this efficiency metric.

---

## Discussion Angles for Your Paper

**If ClassicalNN wins clearly:** This is expected. The paper's contribution
is quantifying *how much* it wins by and *why*: training speed advantage,
parameter flexibility, ReLU's piecewise-linear boundary vs. QNN's Fourier-type
boundary.

**If QNN surprises:** Discuss whether it's due to the specific geometry of
make_moons being naturally suited to the quantum circuit's function class.
Try circles or blobs (both in sklearn) to test if the advantage generalizes.

**On the Hybrid:** If Hybrid > QNN, the paper can argue that classical
preprocessing is valuable because it bridges the gap between the data
distribution and the quantum circuit's native input geometry. If Hybrid ≈ QNN,
the pre-processing layer didn't add much, which is also an interesting result.

**On training time:** The 40–80× slowdown of quantum simulation is paper-worthy
in itself. This is the cost of running quantum circuits on classical hardware.
Real quantum hardware would flip this advantage for large enough systems —
but we're far from that regime for ML tasks.

---

## Reproducing Results

All randomness in the experiment is seeded. If you run `python -m scripts.main` twice,
you should get identical results. The seed is set in `scripts/data.py` via
`get_data(seed=42)`. To run multiple seeds for your paper:

```python
# In scripts/main.py, change:
X_train, X_test, y_train, y_test = get_data(n_samples=200, noise=0.1, seed=42)

# To loop over seeds:
for seed in [42, 7, 13, 99, 2025]:
    X_train, X_test, y_train, y_test = get_data(n_samples=200, noise=0.1, seed=seed)
    # ... train models, record test_acc ...
# Then report mean ± std across seeds
```
