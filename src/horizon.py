"""
step 7 - how fast does accuracy decay with the forecast horizon?

run from the repo root:  python -m src.horizon   (after src.run_backtest)

reads results/backtest.csv, groups every error by how many months ahead it
was, and compares 12-month-ahead error with 1-month-ahead error.
"""

import os

import pandas as pd

from src.backtest import error_by_horizon, horizon_decay, width_by_horizon
from src.plots import plot_horizon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(RESULTS, "figures")

MODELS = ["sarima", "holt_winters"]
BASELINES = ["seasonal_naive_drift", "seasonal_naive", "naive"]

#1998 was the year every model missed badly (see step 6). with only 8 forecasts
#per horizon one bad year can drive the ratio, so check it without that origin
SENSITIVITY_DROP = 1998


def main():
    backtest = pd.read_csv(os.path.join(RESULTS, "backtest.csv"),
                           parse_dates=["origin", "train_end", "date"])
    by_horizon = error_by_horizon(backtest)
    decay = horizon_decay(by_horizon)
    widths = width_by_horizon(backtest)

    mae = by_horizon["mae"].unstack("horizon")
    mae.round(3).to_csv(os.path.join(RESULTS, "mae_by_horizon.csv"))
    widths.round(3).to_csv(os.path.join(RESULTS, "width_by_horizon.csv"))
    decay.round(3).to_csv(os.path.join(RESULTS, "horizon_decay.csv"))
    kept = backtest[backtest["origin"].dt.year != SENSITIVITY_DROP]
    decay_kept = horizon_decay(error_by_horizon(kept))
    decay_kept.round(3).to_csv(
        os.path.join(RESULTS, f"horizon_decay_without_{SENSITIVITY_DROP}.csv"))
    plot_horizon(by_horizon, widths, os.path.join(FIGURES, "horizon.png"), MODELS, BASELINES)

    print("mae by months ahead\n")
    print(mae.round(3).to_string())
    print("\nhorizon decay\n")
    print(decay.round(2).to_string())
    print(f"\nhorizon decay without the {SENSITIVITY_DROP} origin\n")
    print(decay_kept.round(2).to_string())


if __name__ == "__main__":
    main()
