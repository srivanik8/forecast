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


def _bounds(actual, lower, upper):
    actual, lower = _arrays(actual, lower)
    _, upper = _arrays(actual, upper)
    if np.any(lower > upper):
        raise ValueError("lower bound above upper bound")
    return actual, lower, upper


def coverage(actual, lower, upper):
    """share of actual values that fell inside their prediction interval."""
    actual, lower, upper = _bounds(actual, lower, upper)
    return float(np.mean((actual >= lower) & (actual <= upper)))


def interval_width(lower, upper):
    """mean width of the intervals. coverage means little without it."""
    lower, upper = _arrays(lower, upper)
    if np.any(lower > upper):
        raise ValueError("lower bound above upper bound")
    return float(np.mean(upper - lower))


def misses(actual, lower, upper):
    """how many actuals fell below and above their interval."""
    actual, lower, upper = _bounds(actual, lower, upper)
    return {"below": int(np.sum(actual < lower)), "above": int(np.sum(actual > upper))}


def wilson_interval(hits, n, z=1.96):
    """
    95% wilson score interval for a proportion such as coverage.

    with 96 intervals, observed coverage moves a few points by chance alone.
    if the nominal 95% sits inside this range, the gap is not evidence of
    miscalibration. it treats every interval as independent, which forecasts
    from the same origin are not, so the true range is wider than this.
    """
    if n <= 0 or not 0 <= hits <= n:
        raise ValueError("need 0 <= hits <= n and n > 0")
    p = hits / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return float(centre - half), float(centre + half)
