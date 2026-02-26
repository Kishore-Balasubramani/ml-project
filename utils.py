# utils.py

import numpy as np

def load_data():
    """
    Generates dummy training data.
    Returns:
        X: Feature matrix
        y: Target values
    """
    np.random.seed(42)

    # Example dataset: y = 2x + noise
    X = np.random.rand(100, 1) * 10
    y = 2 * X.flatten() + np.random.randn(100)

    return X, y