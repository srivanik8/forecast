"""
rolling-origin backtest.

at each origin the forecaster is refit from scratch on everything before the
origin and forecasts the next `horizon` months. origins step forward a year
at a time and the last one ends at the end of the series. no forecaster ever
sees a value from on or after its origin.
"""

import numpy as np
import pandas as pd

from src.metrics import (coverage, interval_width, mae, mape, mase, misses, rmse,
                         wilson_interval)

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


def score(backtest, series):
    """
    every metric for every (model, origin) forecast window.

    mase is scaled on that origin's own training window - the data the model
    was fitted on - never on the full series.
    """
    rows = []
    for (model, origin), g in backtest.groupby(["model", "origin"], sort=False):
        train = series[series.index <= g["train_end"].iloc[0]]
        a, f = g["actual"], g["forecast"]
        rows.append({
            "model": model,
            "origin": origin,
            "mae": mae(a, f),
            "rmse": rmse(a, f),
            "mape": mape(a, f),
            "mase": mase(a, f, train),
        })
    return pd.DataFrame(rows)


def summarise(scores, reference="seasonal_naive_drift"):
    """
    mean of each metric across origins, plus the comparison that matters:
    mae relative to the hardest baseline (below 1 beats it) and how many
    origins each model beat it at.
    """
    ref = scores[scores["model"] == reference].set_index("origin")["mae"]
    if ref.empty:
        raise ValueError(f"reference model {reference!r} is not in the scores")
    scores = scores.assign(rel_mae=scores["mae"] / scores["origin"].map(ref))
    out = scores.groupby("model").agg(
        mae=("mae", "mean"),
        rmse=("rmse", "mean"),
        mape=("mape", "mean"),
        mase=("mase", "mean"),
        rel_mae=("rel_mae", "mean"),
        beats_reference=("rel_mae", lambda r: int((r < 1).sum())),
        origins=("origin", "nunique"),
    )
    return out.sort_values("mase")


def coverage_summary(backtest, nominal=0.95):
    """
    are the 95% intervals honest? one row per model that produced intervals.

    coverage well below nominal means overconfident, well above means the
    intervals are too wide to be useful.
    """
    rows = []
    with_intervals = backtest.dropna(subset=["lower", "upper"])
    for model, g in with_intervals.groupby("model"):
        a, lo, hi = g["actual"], g["lower"], g["upper"]
        n = len(g)
        hits = int(round(coverage(a, lo, hi) * n))
        ci_low, ci_high = wilson_interval(hits, n)
        rows.append({
            "model": model,
            "intervals": n,
            "inside": hits,
            "coverage": hits / n,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "nominal": nominal,
            "nominal_in_ci": ci_low <= nominal <= ci_high,
            "mean_width": interval_width(lo, hi),
            **{f"missed_{k}": v for k, v in misses(a, lo, hi).items()},
        })
    return pd.DataFrame(rows).set_index("model")


def coverage_by(backtest, by):
    """coverage per model split by another column, e.g. origin or horizon."""
    g = backtest.dropna(subset=["lower", "upper"])
    inside = (g["actual"] >= g["lower"]) & (g["actual"] <= g["upper"])
    return inside.groupby([g["model"], g[by]]).mean().unstack(by)


def error_by_horizon(backtest):
    """mae and rmse per model at each number of months ahead (1 to 12)."""
    err = backtest.assign(abs_err=backtest["error"].abs(), sq_err=backtest["error"] ** 2)
    g = err.groupby(["model", "horizon"])
    return pd.DataFrame({
        "mae": g["abs_err"].mean(),
        "rmse": np.sqrt(g["sq_err"].mean()),
        "n": g.size(),
    })


def horizon_decay(by_horizon, metric="mae"):
    """
    how much worse long-range forecasts are than short-range ones.

    ratio_12_1 compares 12 months ahead with 1 month ahead, as asked. each of
    those rests on only one forecast per origin, so ratio_q4_q1 compares the
    average of months 10-12 with months 1-3 as a steadier check.
    """
    t = by_horizon[metric].unstack("horizon")
    first, last = t.columns.min(), t.columns.max()
    out = pd.DataFrame({
        f"{metric}_h{first}": t[first],
        f"{metric}_h{last}": t[last],
        f"ratio_{last}_{first}": t[last] / t[first],
        f"ratio_q4_q1": t[t.columns[-3:]].mean(axis=1) / t[t.columns[:3]].mean(axis=1),
    })
    return out.sort_values(f"ratio_{last}_{first}", ascending=False)
