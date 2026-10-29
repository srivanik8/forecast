"""
step 5 - score the backtest.

run from the repo root:  python -m src.evaluate   (after src.run_backtest)

reads results/backtest.csv and reports mae, rmse, mape and mase for every
forecaster, plus mae relative to the hardest baseline - the honest headline.
"""

import os

import pandas as pd

from src.backtest import score, summarise
from src.data import load_series
from src.plots import plot_metrics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(RESULTS, "figures")

REFERENCE = "seasonal_naive_drift"


def main():
    backtest = pd.read_csv(os.path.join(RESULTS, "backtest.csv"),
                           parse_dates=["origin", "train_end", "date"])
    scores = score(backtest, load_series())
    summary = summarise(scores, reference=REFERENCE)

    scores.to_csv(os.path.join(RESULTS, "metrics_by_origin.csv"), index=False,
                  float_format="%.4f", date_format="%Y-%m")
    summary.round(3).to_csv(os.path.join(RESULTS, "metrics.csv"))
    plot_metrics(summary, os.path.join(FIGURES, "metrics.png"), reference=REFERENCE)

    print(f"mean over {summary['origins'].iloc[0]} origins, "
          f"rel_mae is against {REFERENCE}\n")
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
