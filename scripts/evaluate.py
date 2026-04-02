"""
evaluate.py — Metrics and visualization

Produces three outputs:
  1. Decision boundary plots  — shows what each model "learned"
  2. Training curves          — loss and accuracy over epochs
  3. Summary table            — test accuracy, parameter count, training time
  4. results.png              — all of the above saved to disk
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def evaluate_model(model, X_test, y_test):
    """
    Compute test accuracy for a trained model.

    model.eval() disables any training-only behavior (dropout, batchnorm).
    torch.no_grad() tells PyTorch not to build a computation graph during
    the forward pass — we don't need gradients here, so this saves memory.
    """
    model.eval()
    with torch.no_grad():
        preds     = model(X_test)
        predicted = (preds > 0.5).float()
        accuracy  = (predicted == y_test).float().mean().item()
    return accuracy


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.

    This is an important metric for the paper — it lets us ask:
    "Are we getting more or less accuracy per trainable parameter?"
    A QNN with N_LAYERS=3, N_QUBITS=2 has 3 × 2 × 3 = 18 quantum weights.
    The ClassicalNN has many more classical weights. Is the QNN more parameter-efficient?
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_decision_boundary(model, X, y, title, ax):
    """
    Visualize the decision boundary learned by a model.

    We create a dense grid of points covering the input space, run every
    point through the model, and color the background based on the predicted
    probability. The training points are overlaid on top.

    A sharp, clean boundary = confident model.
    A fuzzy/jagged boundary = uncertain or underfitted model.
    """
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    # h controls grid resolution — smaller = smoother but slower
    h  = 0.05
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Stack all grid points into a (N_grid, 2) tensor for batch inference
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        Z = model(grid).numpy().reshape(xx.shape)

    ax.contourf(xx, yy, Z, levels=20, alpha=0.4, cmap="RdBu")
    ax.contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=1.5)  # decision boundary line
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu", edgecolors="k", s=30, zorder=5)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")


def plot_results(models_dict, histories, X_train, y_train, X_test, y_test, times):
    """
    Generate and save the full results figure.

    Layout (2 rows × 3 columns):
      Row 1: Decision boundary for each model (3 plots)
      Row 2: Loss curve | Accuracy curve | Test accuracy bar chart
    """
    X_np = X_train.numpy()
    y_np = y_train.numpy().flatten()

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("NN vs QNN vs Hybrid — Binary Classification on make_moons",
                 fontsize=15, fontweight="bold", y=1.01)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    colors = {"Classical": "steelblue", "QNN": "coral", "Hybrid": "mediumseagreen"}

    # ── Row 1: Decision boundaries ──────────────────────────────────────────
    for i, (name, model) in enumerate(models_dict.items()):
        ax = fig.add_subplot(gs[0, i])
        plot_decision_boundary(model, X_np, y_np, name, ax)

    # ── Row 2, Col 0: Loss curves ────────────────────────────────────────────
    ax_loss = fig.add_subplot(gs[1, 0])
    for name, history in histories.items():
        ax_loss.plot(history["loss"], label=name, color=colors[name], linewidth=2)
    ax_loss.set_title("Training Loss", fontweight="bold")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("BCE Loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    # ── Row 2, Col 1: Accuracy curves ────────────────────────────────────────
    ax_acc = fig.add_subplot(gs[1, 1])
    for name, history in histories.items():
        ax_acc.plot(history["accuracy"], label=name, color=colors[name], linewidth=2)
    ax_acc.set_title("Training Accuracy", fontweight="bold")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(0, 1.05)
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)

    # ── Row 2, Col 2: Summary bar chart ──────────────────────────────────────
    ax_bar = fig.add_subplot(gs[1, 2])
    names     = list(models_dict.keys())
    test_accs = [evaluate_model(m, X_test, y_test) for m in models_dict.values()]

    bars = ax_bar.bar(names, test_accs,
                      color=[colors[n] for n in names],
                      edgecolor="black", linewidth=0.8)
    ax_bar.set_title("Test Accuracy", fontweight="bold")
    ax_bar.set_ylim(0, 1.1)
    ax_bar.set_ylabel("Accuracy")
    ax_bar.grid(True, axis="y", alpha=0.3)

    for bar, acc in zip(bars, test_accs):
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{acc:.3f}", ha="center", fontsize=11, fontweight="bold")

    plt.savefig("results.png", dpi=150, bbox_inches="tight")
    print("  Saved → results.png")
    plt.show()

    # ── Summary table ─────────────────────────────────────────────────────────
    param_counts = [count_parameters(m) for m in models_dict.values()]

    print("\n" + "=" * 60)
    print(f"  {'Model':<12} {'Test Acc':>10} {'Params':>10} {'Time (s)':>12}")
    print("=" * 60)
    for name, acc, params in zip(names, test_accs, param_counts):
        print(f"  {name:<12} {acc:>10.4f} {params:>10} {times[name]:>10.1f}s")
    print("=" * 60)
