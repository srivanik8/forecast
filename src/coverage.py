"""
step 6 - are the 95% prediction intervals honest?

run from the repo root:  python -m src.coverage   (after src.run_backtest)

reads results/backtest.csv and checks how often the actual value fell inside
each model's 95% interval, across all origins and months ahead.
"""

import os

import pandas as pd

from src.backtest import coverage_by, coverage_summary
from src.plots import plot_coverage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(RESULTS, "figures")


def main():
    backtest = pd.read_csv(os.path.join(RESULTS, "backtest.csv"),
                           parse_dates=["origin", "train_end", "date"])
    summary = coverage_summary(backtest)
    by_origin = coverage_by(backtest, "origin")

    summary.round(3).to_csv(os.path.join(RESULTS, "coverage.csv"))
    out = by_origin.copy()
    out.columns = [f"{c:%Y}" for c in out.columns]
    out.round(3).to_csv(os.path.join(RESULTS, "coverage_by_origin.csv"))
    models = [m for m in ["sarima", "holt_winters"] if m in summary.index]
    plot_coverage(backtest, by_origin, models,
                  os.path.join(FIGURES, "coverage.png"))

    print(summary.round(3).to_string())
    print("\ncoverage by origin\n")
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
