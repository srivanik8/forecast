"""
step 1 - load and explore.

run from the repo root:  python -m src.explore

writes results/data_summary.csv and the exploration figures, and prints the
numbers the readme's data section is written from.
"""

import os

import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

from src.data import load_raw, missing_report, to_monthly
from src.plots import plot_decomposition, plot_seasonal_profile, plot_series

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(RESULTS, "figures")

SEASON = 12


def summarise(raw, monthly, decomp):
    raw_gaps = missing_report(raw)
    #months that had no weekly readings at all, i.e. the ones we interpolated
    empty_months = int(raw.resample("MS").mean().loc[monthly.index].isna().sum())

    #average yearly rise in the trend, first decade vs last decade
    rise = decomp.trend.dropna().diff(SEASON).dropna()
    seasonal_profile = decomp.seasonal.groupby(decomp.seasonal.index.month).mean()

    return {
        "raw_weekly_obs": raw_gaps["n_obs"],
        "raw_weekly_missing": raw_gaps["n_missing"],
        "raw_weekly_pct_missing": round(raw_gaps["pct_missing"], 2),
        "monthly_obs": len(monthly),
        "monthly_start": f"{monthly.index[0]:%Y-%m}",
        "monthly_end": f"{monthly.index[-1]:%Y-%m}",
        "months_interpolated": empty_months,
        "min_ppm": round(float(monthly.min()), 2),
        "max_ppm": round(float(monthly.max()), 2),
        "trend_rise_per_year_first_10y": round(float(rise.iloc[:120].mean()), 3),
        "trend_rise_per_year_last_10y": round(float(rise.iloc[-120:].mean()), 3),
        "seasonal_peak_month": int(seasonal_profile.idxmax()),
        "seasonal_trough_month": int(seasonal_profile.idxmin()),
        "seasonal_amplitude_ppm": round(float(seasonal_profile.max() - seasonal_profile.min()), 3),
        "residual_std_ppm": round(float(decomp.resid.std()), 3),
    }


def main():
    raw = load_raw()
    monthly = to_monthly(raw)
    #decomposing the full series is fine for looking at it. it must never feed
    #a model - that would leak the test period into training
    decomp = seasonal_decompose(monthly, model="additive", period=SEASON)

    summary = summarise(raw, monthly, decomp)
    os.makedirs(RESULTS, exist_ok=True)
    pd.Series(summary, name="value").rename_axis("metric").to_csv(
        os.path.join(RESULTS, "data_summary.csv"))

    plot_series(monthly, os.path.join(FIGURES, "series.png"))
    plot_decomposition(decomp, os.path.join(FIGURES, "decomposition.png"))
    plot_seasonal_profile(decomp.seasonal, os.path.join(FIGURES, "seasonal_profile.png"))

    for k, v in summary.items():
        print(f"{k:32s} {v}")


if __name__ == "__main__":
    main()
