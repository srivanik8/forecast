"""error metrics, written out by hand so it is clear what each one measures."""

import numpy as np


def mae(actual, pred):
    """mean absolute error, in the units of the data."""
    actual, pred = np.asarray(actual, dtype=float), np.asarray(pred, dtype=float)
    if actual.shape != pred.shape:
        raise ValueError(f"shape mismatch: {actual.shape} vs {pred.shape}")
    return float(np.mean(np.abs(actual - pred)))
