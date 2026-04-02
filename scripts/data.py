"""
data.py — Dataset generation and preprocessing

We use sklearn's make_moons dataset: two interleaved crescent shapes
that are NOT linearly separable. This forces all three models to learn
a non-linear decision boundary, which is a more interesting test than
something a straight line could solve.

Data pipeline:
  1. Generate 200 points with a small amount of noise
  2. Split 80% train / 20% test
  3. Standardize (zero mean, unit variance) using only train statistics
  4. Convert to PyTorch tensors
"""

import torch
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def get_data(n_samples=200, noise=0.1, test_size=0.2, seed=42):
    """
    Generate and return the make_moons dataset as PyTorch tensors.

    Args:
        n_samples  (int):   Total number of data points to generate.
        noise      (float): Standard deviation of Gaussian noise added to
                            the data. Higher = harder classification task.
        test_size  (float): Fraction of data reserved for testing.
        seed       (int):   Random seed for reproducibility.

    Returns:
        X_train, X_test   shape (N, 2)  — the 2D input features
        y_train, y_test   shape (N, 1)  — binary labels (0 or 1)
    """
    # --- 1. Generate raw data ---
    # X has shape (n_samples, 2): two continuous features (x, y position)
    # y has shape (n_samples,): binary labels, 0 = top crescent, 1 = bottom
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)

    # --- 2. Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    # --- 3. Standardize features ---
    # StandardScaler transforms each feature to have mean=0, std=1.
    # We fit ONLY on training data, then apply the same transform to test.
    # This is critical — fitting on test data would be data leakage.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # --- 4. Convert to PyTorch tensors ---
    # float32 is standard for neural network weights and inputs.
    # y needs .unsqueeze(1) to go from shape (N,) → (N, 1) so it matches
    # the model output shape during loss computation.
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test  = torch.tensor(X_test,  dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_test  = torch.tensor(y_test,  dtype=torch.float32).unsqueeze(1)

    return X_train, X_test, y_train, y_test
