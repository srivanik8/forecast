"""error metrics, written out by hand so it is clear what each one measures."""

import numpy as np


def _arrays(actual, pred):
    actual, pred = np.asarray(actual, dtype=float), np.asarray(pred, dtype=float)
    if actual.shape != pred.shape:
        raise ValueError(f"shape mismatch: {actual.shape} vs {pred.shape}")
    return actual, pred


def mae(actual, pred):
    """mean absolute error, in the units of the data."""
    actual, pred = _arrays(actual, pred)
    return float(np.mean(np.abs(actual - pred)))


def rmse(actual, pred):
    """root mean squared error. squaring punishes large misses harder than mae."""
    actual, pred = _arrays(actual, pred)
    return float(np.sqrt(np.mean((actual - pred) ** 2)))
