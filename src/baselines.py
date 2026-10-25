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
