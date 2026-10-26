"""
step 3 - fit the real models and check they are sensible.

run from the repo root:  python -m src.fit_models

both models are fitted on everything except the final 12 months, then
forecast those 12 months. this is a sanity check on one split, not the
evaluation - the rolling-origin backtest in step 4 is what gets reported.
"""

import os

import pandas as pd

from src.baselines import seasonal_naive_drift
from src.data import load_series
from src.metrics import mae
from src.models import FITTERS, MODELS, diagnostics
from src.plots import plot_model_forecasts

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(RESULTS, "figures")

HORIZON = 12


def main():
    series = load_series()
    train, test = series.iloc[:-HORIZON], series.iloc[-HORIZON:]

    rows, forecasts = [], {}
    for name in MODELS:
        forecasts[name] = MODELS[name](train, HORIZON)
        rows.append({"model": name,
                     **diagnostics(FITTERS[name](train)),
                     "holdout_mae": round(mae(test, forecasts[name]["mean"]), 3)})
    rows.append({"model": "seasonal_naive_drift",
                 "holdout_mae": round(mae(test, seasonal_naive_drift(train, HORIZON)), 3)})

    summary = pd.DataFrame(rows).set_index("model")
    os.makedirs(RESULTS, exist_ok=True)
    summary.to_csv(os.path.join(RESULTS, "model_fit.csv"))
    plot_model_forecasts(train.iloc[-36:], test, forecasts,
                         os.path.join(FIGURES, "model_forecasts.png"))

    print(f"trained to {train.index[-1]:%Y-%m}, forecasting {test.index[0]:%Y-%m} "
          f"to {test.index[-1]:%Y-%m}\n")
    print(summary.to_string())


if __name__ == "__main__":
    main()
