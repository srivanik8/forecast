"""
step 2 - compare the baselines and find the one that is hardest to beat.

run from the repo root:  python -m src.compare_baselines

each baseline forecasts 12 months ahead from 8 yearly cut points at the end of
the series. it only ever sees the data before its cut point. the full
rolling-origin backtest comes in step 4; this is just enough to rank the
baselines.
"""

import os

import pandas as pd

from src.baselines import BASELINES
from src.data import load_series
from src.metrics import mae
from src.plots import plot_baseline_forecasts

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(RESULTS, "figures")

HORIZON = 12
N_ORIGINS = 8


def main():
    series = load_series()
    rows = []
    for i in range(N_ORIGINS):
        cut = len(series) - HORIZON * (N_ORIGINS - i)
        train, test = series.iloc[:cut], series.iloc[cut:cut + HORIZON]
        for name, fn in BASELINES.items():
            rows.append({
                "origin": f"{test.index[0]:%Y-%m}",
                "baseline": name,
                "mae": mae(test, fn(train, HORIZON)),
            })

    per_origin = pd.DataFrame(rows)
    summary = (per_origin.groupby("baseline")["mae"]
               .agg(mean_mae="mean", worst_mae="max")
               .round(3).sort_values("mean_mae"))

    os.makedirs(RESULTS, exist_ok=True)
    per_origin.round(3).to_csv(os.path.join(RESULTS, "baselines_by_origin.csv"), index=False)
    summary.to_csv(os.path.join(RESULTS, "baselines_summary.csv"))

    #picture of the last cut point, with three years of context
    train, test = series.iloc[:-HORIZON], series.iloc[-HORIZON:]
    forecasts = {name: fn(train, HORIZON) for name, fn in BASELINES.items()}
    plot_baseline_forecasts(train.iloc[-36:], test, forecasts,
                            os.path.join(FIGURES, "baseline_forecasts.png"))

    print(summary.to_string())
    print(f"\nhardest baseline to beat: {summary.index[0]}")


if __name__ == "__main__":
    main()
