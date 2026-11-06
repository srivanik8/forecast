"""
naive benchmarks. every real model has to beat these to be worth anything.

each function takes a training series (monthly index) and a horizon, and
returns the forecast as a series indexed by the next `horizon` months. only
the training series is ever touched, so nothing from the future can leak in.
"""

import pandas as pd

SEASON = 12


def _future_index(train, horizon):
    return pd.date_range(train.index[-1], periods=horizon + 1, freq=train.index.freq)[1:]


def naive(train, horizon):
    """every future month equals the last observed month."""
    return pd.Series(float(train.iloc[-1]), index=_future_index(train, horizon), name="naive")


def seasonal_naive(train, horizon, season=SEASON):
    """each future month equals the same month in the last observed year."""
    last_year = train.iloc[-season:].to_numpy()
    values = [last_year[h % season] for h in range(horizon)]
    return pd.Series(values, index=_future_index(train, horizon), name="seasonal_naive")


def seasonal_naive_drift(train, horizon, season=SEASON, years=None):
    """
    seasonal naive plus the average year-on-year change in the training window.

    a forecast k years past the last observed season gets k years of drift.
    years limits the average to the most recent years, for a series whose
    growth is speeding up; None uses the whole training window.
    """
    changes = train.diff(season)
    if years is not None:
        changes = changes.iloc[-season * years:]
    drift = float(changes.mean())
    base = seasonal_naive(train, horizon, season)
    years_ahead = [h // season + 1 for h in range(horizon)]
    out = base + drift * pd.Series(years_ahead, index=base.index)
    out.name = "seasonal_naive_drift"
    return out


def recent_drift(years):
    """seasonal naive + drift averaged over only the last `years` years."""
    def forecaster(train, horizon):
        return seasonal_naive_drift(train, horizon, years=years)
    forecaster.__name__ = f"drift_{years}y"
    return forecaster


BASELINES = {
    "naive": naive,
    "seasonal_naive": seasonal_naive,
    "seasonal_naive_drift": seasonal_naive_drift,
}

#candidate harder benchmarks, run by src.experiments
CANDIDATE_BASELINES = {f"drift_{y}y": recent_drift(y) for y in (3, 5, 10)}
