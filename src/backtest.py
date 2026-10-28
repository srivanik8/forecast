"""
rolling-origin backtest.

at each origin the forecaster is refit from scratch on everything before the
origin and forecasts the next `horizon` months. origins step forward a year
at a time and the last one ends at the end of the series. no forecaster ever
sees a value from on or after its origin.
"""

import numpy as np
import pandas as pd

HORIZON = 12
N_ORIGINS = 8
STEP = 12  #a year apart, so the forecast windows never overlap


def origins(n, horizon=HORIZON, n_origins=N_ORIGINS, step=STEP):
    """
    (train_end, test_end) positions for each origin, oldest first.

    train is series[:train_end], test is series[train_end:test_end].
    """
    last = n - horizon
    cuts = [last - step * (n_origins - 1 - i) for i in range(n_origins)]
    if cuts[0] < 2 * horizon:
        raise ValueError(f"series of length {n} is too short for {n_origins} origins")
    return [(c, c + horizon) for c in cuts]


def _as_frame(forecast):
    """baselines return a point series, models a mean/lower/upper frame."""
    if isinstance(forecast, pd.Series):
        return pd.DataFrame({"mean": forecast, "lower": np.nan, "upper": np.nan})
    return forecast


def run(series, forecasters, horizon=HORIZON, n_origins=N_ORIGINS, step=STEP):
    """
    refit every forecaster at every origin and record each forecast.

    one row per (model, origin, month ahead): the forecast, its 95% interval
    (empty for baselines), the actual value and the error (actual - forecast).
    """
    rows = []
    for train_end, test_end in origins(len(series), horizon, n_origins, step):
        #a fresh copy, so no forecaster can reach past the origin
        train = series.iloc[:train_end].copy()
        test = series.iloc[train_end:test_end]
        for name, fn in forecasters.items():
            f = _as_frame(fn(train, horizon))
            if not f.index.equals(test.index):
                raise ValueError(f"{name} forecast dates do not match the test window")
            for h, (date, actual) in enumerate(test.items(), start=1):
                rows.append({
                    "model": name,
                    "origin": test.index[0],
                    "train_end": train.index[-1],
                    "date": date,
                    "horizon": h,
                    "forecast": f.at[date, "mean"],
                    "lower": f.at[date, "lower"],
                    "upper": f.at[date, "upper"],
                    "actual": actual,
                    "error": actual - f.at[date, "mean"],
                })
    return pd.DataFrame(rows)
