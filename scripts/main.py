"""
main.py — Entry point: runs the full experiment end to end

Execution order:
  1. Generate and preprocess the dataset
  2. Train ClassicalNN, QNN, and HybridNN sequentially
  3. Evaluate all three on the test set
  4. Generate plots and print the results table

To run:
    python -m scripts.main
"""

from scripts.data     import get_data
from scripts.models   import ClassicalNN, QNN, HybridNN
from scripts.train    import train_model
from scripts.evaluate import evaluate_model, count_parameters, plot_results


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
#
# Why different settings per model?
#
# ClassicalNN  — fast to run, so we can afford more epochs. Lower batch size
#                is fine since gradient computation is cheap.
#
# QNN          — each forward pass involves a full quantum circuit simulation,
#                which is much slower than a matrix multiply. Fewer epochs and
#                a slightly higher learning rate helps it converge in reasonable
#                time. We also use a slightly smaller batch to get more gradient
#                update steps per epoch.
#
# HybridNN     — similar reasoning to QNN. The classical layers are fast but
#                the quantum layer is the bottleneck.
#
# These are starting defaults — feel free to tune them for your experiments.

CONFIG = {
    "Classical": {"epochs": 100, "lr": 0.01,  "batch_size": 32},
    "QNN":       {"epochs": 60,  "lr": 0.05,  "batch_size": 16},
    "Hybrid":    {"epochs": 60,  "lr": 0.02,  "batch_size": 16},
}


def main():
    # ── 1. Data ──────────────────────────────────────────────────────────────
    print("=" * 60)
    print("  Loading dataset...")
    print("=" * 60)
    X_train, X_test, y_train, y_test = get_data(n_samples=200, noise=0.1)
    print(f"  Train samples : {X_train.shape[0]}")
    print(f"  Test  samples : {X_test.shape[0]}")
    print(f"  Features      : {X_train.shape[1]}")
    print()

    # ── 2. Instantiate models ─────────────────────────────────────────────────
    models = {
        "Classical": ClassicalNN(),
        "QNN":       QNN(),
        "Hybrid":    HybridNN(),
    }

    # ── 3. Train each model ───────────────────────────────────────────────────
    histories = {}
    times     = {}

    for name, model in models.items():
        n_params = count_parameters(model)
        cfg      = CONFIG[name]
        print("=" * 60)
        print(f"  Training: {name}  ({n_params} trainable parameters)")
        print(f"  epochs={cfg['epochs']}  lr={cfg['lr']}  batch_size={cfg['batch_size']}")
        print("=" * 60)

        history, elapsed = train_model(model, X_train, y_train, **cfg)
        histories[name]  = history
        times[name]      = elapsed

        test_acc = evaluate_model(model, X_test, y_test)
        print(f"\n  ✓ Finished in {elapsed:.1f}s  |  Test Accuracy: {test_acc:.4f}\n")

    # ── 4. Results ────────────────────────────────────────────────────────────
    print("=" * 60)
    print("  Generating plots...")
    print("=" * 60)
    plot_results(models, histories, X_train, y_train, X_test, y_test, times)


if __name__ == "__main__":
    main()
