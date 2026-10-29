"""
step 4 - rolling-origin backtest of every baseline and model.

run from the repo root:  python -m src.run_backtest

8 origins a year apart (forecasting 1994 through 2001), 12 months ahead
from each, every forecaster refit from scratch at every origin. writes one
row per forecast to results/backtest.csv - the raw material for the metrics,
coverage and horizon steps that follow.
"""

import os
import time

from src.backtest import run
from src.baselines import BASELINES
from src.data import load_series
from src.metrics import mae
from src.models import MODELS
from src.plots import plot_backtest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(RESULTS, "figures")

FORECASTERS = {**BASELINES, **MODELS}


def main():
    series = load_series()
    start = time.time()
    out = run(series, FORECASTERS)

    os.makedirs(RESULTS, exist_ok=True)
    out.to_csv(os.path.join(RESULTS, "backtest.csv"), index=False,
               float_format="%.4f", date_format="%Y-%m")

    #mae of each 12-month forecast, one column per origin
    by_origin = (out.groupby(["model", "origin"])
                 .apply(lambda g: mae(g["actual"], g["forecast"]), include_groups=False)
                 .unstack("origin"))
    by_origin.columns = [f"{c:%Y}" for c in by_origin.columns]
    by_origin["mean"] = by_origin.mean(axis=1)
    by_origin = by_origin.round(3)
    by_origin = by_origin.sort_values("mean")
    by_origin.to_csv(os.path.join(RESULTS, "backtest_mae_by_origin.csv"))

    plot_backtest(series, out, by_origin, ["sarima", "holt_winters", "seasonal_naive_drift"],
                  os.path.join(FIGURES, "backtest.png"))

    n_origins = out["origin"].nunique()
    print(f"{len(out)} forecasts from {n_origins} origins in {time.time() - start:.0f}s\n")
    print(by_origin.to_string())


if __name__ == "__main__":
    main()
