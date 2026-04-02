"""
train.py — Shared training loop

This single training function works for all three models (ClassicalNN,
QNN, HybridNN) because they all implement the nn.Module interface and
produce outputs in [0, 1] — so the loss function and optimizer work
identically regardless of what's inside the model.

Training uses:
  - BCELoss (Binary Cross-Entropy): the standard loss for binary classification.
    Defined as: L = -[y·log(p) + (1-y)·log(1-p)]
    where y is the true label (0 or 1) and p is the predicted probability.
    It penalizes confident wrong predictions very heavily.

  - Adam optimizer: an adaptive gradient descent algorithm. It tracks a
    moving average of gradients and their squares to automatically adjust
    the learning rate per parameter. Much more stable than vanilla SGD,
    especially for quantum circuits where gradients can be small.

  - Mini-batch training: instead of computing gradients on all 160 training
    points at once (too slow/noisy), we split the data into small batches
    and update after each batch. This also adds regularization-like noise.
"""

import time

import torch
import torch.nn as nn


def train_model(model, X_train, y_train, epochs=100, lr=0.01, batch_size=32):
    """
    Train a model on the given data and return its loss/accuracy history.

    Args:
        model       (nn.Module): Any of ClassicalNN, QNN, or HybridNN.
        X_train     (Tensor):    Training features, shape (N, 2).
        y_train     (Tensor):    Training labels,   shape (N, 1).
        epochs      (int):       Number of full passes through the training data.
        lr          (float):     Learning rate for the Adam optimizer.
        batch_size  (int):       Number of samples per gradient update step.

    Returns:
        history  (dict): {"loss": [...], "accuracy": [...]} — one value per epoch.
        elapsed  (float): Total training time in seconds.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    # History stores one loss value and one accuracy value per epoch.
    # We'll use these later to plot training curves.
    history = {"loss": [], "accuracy": []}

    # TensorDataset + DataLoader handles batching and shuffling for us.
    # Shuffling each epoch ensures the model doesn't memorize batch order.
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    start_time = time.time()

    for epoch in range(epochs):
        model.train()  # sets model to training mode (enables dropout, batchnorm if present)

        epoch_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in loader:
            # --- Forward pass ---
            optimizer.zero_grad()  # clear gradients from previous step
            preds = model(X_batch)  # shape (batch_size, 1), values in (0, 1)
            loss = loss_fn(preds, y_batch)

            # --- Backward pass ---
            # .backward() computes dLoss/dWeights for every trainable parameter.
            # For the quantum circuit, PennyLane computes these gradients using
            # the parameter-shift rule instead of standard backpropagation.
            loss.backward()
            optimizer.step()  # update weights: w ← w - lr * grad

            # --- Track metrics ---
            epoch_loss += loss.item() * len(X_batch)
            predicted = (preds > 0.5).float()  # threshold at 0.5 → class 0 or 1
            correct += (predicted == y_batch).sum().item()
            total += len(y_batch)

        avg_loss = epoch_loss / total
        accuracy = correct / total
        history["loss"].append(avg_loss)
        history["accuracy"].append(accuracy)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}  |  Loss: {avg_loss:.4f}  |  Acc: {accuracy:.4f}")

    elapsed = time.time() - start_time
    return history, elapsed
