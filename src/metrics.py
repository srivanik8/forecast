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


def mape(actual, pred):
    """
    mean absolute percentage error, as a percentage.

    easy to communicate, but it divides by the actual value, so it breaks
    down near zero. refuse rather than return a meaningless number.
    """
    actual, pred = _arrays(actual, pred)
    if np.any(np.isclose(actual, 0.0)):
        raise ValueError("mape is undefined when an actual value is zero")
    return float(np.mean(np.abs((actual - pred) / actual)) * 100)


def mase(actual, pred, train, season=1):
    """
    mean absolute scaled error: forecast mae divided by the in-sample mae of a
    naive forecast on the training window (one step back, or one season back).

    below 1 beats that naive forecast, above 1 does not. the scale comes from
    the training window only, so no test data leaks into it.
    """
    actual, pred = _arrays(actual, pred)
    train = np.asarray(train, dtype=float)
    if len(train) <= season:
        raise ValueError("training window is too short to scale by")
    scale = np.mean(np.abs(train[season:] - train[:-season]))
    if np.isclose(scale, 0.0):
        raise ValueError("mase is undefined for a constant training window")
    return float(np.mean(np.abs(actual - pred)) / scale)
