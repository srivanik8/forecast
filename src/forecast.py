"""
monthly time series forecasting with a baseline-first evaluation.

the point of this script is not to fit a fancy model. it is to check, honestly,
whether the fancy models actually beat the obvious guess - and to say so when
they do not.
"""

import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

#statsmodels is chatty about convergence during the backtest, we handle it below
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")

SEASON = 12          #monthly data, so the season is twelve steps
HORIZON = 12         #we forecast a full year ahead
N_ORIGINS = 8        #eight rolling cut points to average over
STEP = 12            #origins are spaced a year apart so folds do not overlap
ALPHA = 0.05         #for the 95% prediction intervals


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load_series():
    """
    the only function you change to swap datasets.

    everything downstream just wants a pandas series with a monthly
    datetime index and no gaps.
    """
    #mauna loa atmospheric co2, weekly readings from march 1958 to december 2001
    raw = sm.datasets.co2.load_pandas().data["co2"]

    #weekly -> monthly means, which is the frequency we actually model at
    monthly = raw.resample("MS").mean()

    #a handful of months have no readings at all, bridge them by time
    n_missing = int(monthly.isna().sum())
    monthly = monthly.interpolate(method="time")

    #trim any leading or trailing nan the interpolation could not reach
    monthly = monthly.dropna()
    monthly.index.freq = "MS"

    print(f"loaded {len(monthly)} monthly observations "
          f"({monthly.index[0]:%Y-%m} to {monthly.index[-1]:%Y-%m}), "
          f"{n_missing} filled by interpolation")
    return monthly


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def mae(actual, pred):
    return float(np.mean(np.abs(actual - pred)))


def rmse(actual, pred):
    return float(np.sqrt(np.mean((actual - pred) ** 2)))


def mape(actual, pred):
    #expressed as a percentage, safe here because co2 is never near zero
    return float(np.mean(np.abs((actual - pred) / actual)) * 100)


def mase(actual, pred, train):
    """
    scaled error - a value of 1 means you did exactly as well as a
    one-step naive forecast on the training data.
    """
    scale = np.mean(np.abs(np.diff(train)))
    return float(np.mean(np.abs(actual - pred)) / scale)


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
# every model takes the training series and a horizon, and hands back a plain
# numpy array of length horizon. that uniform shape is what lets the backtest
# treat the baselines and the real models identically.

def naive(train, horizon):
    #carry the last observed value forward, flat
    return np.repeat(train.iloc[-1], horizon)


def seasonal_naive(train, horizon):
    #repeat the value from the same month last year
    last_season = train.iloc[-SEASON:].to_numpy()
    reps = int(np.ceil(horizon / SEASON))
    return np.tile(last_season, reps)[:horizon]


def seasonal_naive_drift(train, horizon):
    """
    same month last year, plus the average year on year climb.

    this is the baseline that actually deserves respect on a trending
    seasonal series, and it is the one people forget to try.
    """
    base = seasonal_naive(train, horizon)

    #average change between a month and the same month a year earlier
    drift = float(train.diff(SEASON).dropna().mean())

    #the value we borrowed is exactly one year stale, so add one year of climb.
    #past twelve months ahead we wrap around again and owe another year of it
    steps = np.arange(1, horizon + 1)
    years_stale = np.ceil(steps / SEASON)
    return base + drift * years_stale


def holt_winters(train, horizon):
    #additive trend and additive season, the shape this series obviously has
    fit = ExponentialSmoothing(
        train,
        trend="add",
        seasonal="add",
        seasonal_periods=SEASON,
        initialization_method="estimated",
    ).fit()
    return np.asarray(fit.forecast(horizon))


def sarima(train, horizon, with_interval=False):
    #(1,1,1)(1,1,1,12) - one regular difference for the trend, one seasonal
    fit = SARIMAX(
        train,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, SEASON),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)

    res = fit.get_forecast(steps=horizon)
    point = np.asarray(res.predicted_mean)

    if not with_interval:
        return point

    ci = res.conf_int(alpha=ALPHA)
    return point, np.asarray(ci.iloc[:, 0]), np.asarray(ci.iloc[:, 1])


MODELS = {
    "naive": naive,
    "seasonal_naive": seasonal_naive,
    "seasonal_naive_drift": seasonal_naive_drift,
    "holt_winters": holt_winters,
    "sarima": sarima,
}


# ---------------------------------------------------------------------------
# rolling origin backtest
# ---------------------------------------------------------------------------

def origins(n_obs):
    """
    the cut points we train up to. the last one leaves exactly one horizon
    of unseen data, and the earlier ones step back a year at a time.
    """
    last = n_obs - HORIZON
    cuts = [last - i * STEP for i in range(N_ORIGINS)]
    return sorted(c for c in cuts if c > SEASON * 4)


def backtest(y):
    """
    walk the origin forward, refit every model from scratch at each one, and
    score the next twelve months against what actually happened.

    refitting matters - reusing one fit across all origins would leak future
    data backwards and quietly flatter every model.
    """
    cuts = origins(len(y))
    print(f"backtesting {len(cuts)} origins, {HORIZON} months ahead from each")

    rows = []            #one row per model per origin
    horizon_rows = []    #one row per model per origin per step ahead
    coverage_rows = []   #interval hits, sarima only

    for cut in cuts:
        train = y.iloc[:cut]
        actual = y.iloc[cut:cut + HORIZON].to_numpy()
        origin_label = f"{y.index[cut - 1]:%Y-%m}"

        for name, fn in MODELS.items():
            if name == "sarima":
                pred, lo, hi = fn(train, HORIZON, with_interval=True)

                #did the 95% band actually contain the truth? measure, do not assume
                for step, (a, l, u) in enumerate(zip(actual, lo, hi), start=1):
                    coverage_rows.append({
                        "origin": origin_label,
                        "months_ahead": step,
                        "actual": a,
                        "lower": l,
                        "upper": u,
                        "covered": bool(l <= a <= u),
                        "width": float(u - l),
                    })
            else:
                pred = np.asarray(fn(train, HORIZON), dtype=float)

            rows.append({
                "model": name,
                "origin": origin_label,
                "mae": mae(actual, pred),
                "rmse": rmse(actual, pred),
                "mape": mape(actual, pred),
                "mase": mase(actual, pred, train.to_numpy()),
            })

            #keep the per step errors so we can watch accuracy decay later
            for step, (a, p) in enumerate(zip(actual, pred), start=1):
                horizon_rows.append({
                    "model": name,
                    "origin": origin_label,
                    "months_ahead": step,
                    "abs_error": abs(a - p),
                })

    return (pd.DataFrame(rows),
            pd.DataFrame(horizon_rows),
            pd.DataFrame(coverage_rows))


# ---------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------

def summarise(fold_scores):
    """average each metric across the folds and rank by mae"""
    summary = (fold_scores
               .groupby("model")[["mae", "rmse", "mape", "mase"]]
               .mean()
               .sort_values("mae"))

    #the honest comparison - everything measured against the seasonal naive,
    #not against the do nothing naive that makes any model look clever
    reference = summary.loc["seasonal_naive", "mae"]
    summary["vs_seasonal_naive"] = summary["mae"] / reference

    return summary.reset_index()


def coverage_summary(coverage):
    hit_rate = coverage["covered"].mean() * 100
    width = coverage["width"].mean()
    nominal = (1 - ALPHA) * 100

    verdict = "overconfident" if hit_rate < nominal else "conservative"
    print(f"\n{nominal:.0f}% intervals contained the actual value "
          f"{hit_rate:.1f}% of the time "
          f"(mean width {width:.2f} ppm) - slightly {verdict}")

    return pd.DataFrame([{
        "nominal_coverage": nominal,
        "empirical_coverage": hit_rate,
        "mean_interval_width": width,
        "n_intervals": len(coverage),
    }])


def horizon_summary(horizon_scores, model="sarima"):
    """how much worse does the forecast get the further out you look"""
    by_step = (horizon_scores[horizon_scores["model"] == model]
               .groupby("months_ahead")["abs_error"]
               .mean()
               .rename("mae")
               .reset_index())

    first, last = by_step["mae"].iloc[0], by_step["mae"].iloc[-1]
    print(f"\n{model} error grows from {first:.3f} ppm at one month "
          f"to {last:.3f} ppm at twelve ({last / first:.1f}x)")

    return by_step


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS, exist_ok=True)

    y = load_series()
    y.to_frame("value").to_csv(os.path.join(RESULTS, "series.csv"))

    fold_scores, horizon_scores, coverage = backtest(y)

    summary = summarise(fold_scores)
    print("\nmean error across all origins:")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    cov = coverage_summary(coverage)
    by_step = horizon_summary(horizon_scores)

    #everything the readme quotes is written out here, nothing is typed by hand
    summary.to_csv(os.path.join(RESULTS, "model_comparison.csv"), index=False)
    fold_scores.to_csv(os.path.join(RESULTS, "fold_scores.csv"), index=False)
    horizon_scores.to_csv(os.path.join(RESULTS, "horizon_errors.csv"), index=False)
    by_step.to_csv(os.path.join(RESULTS, "error_by_horizon.csv"), index=False)
    coverage.to_csv(os.path.join(RESULTS, "interval_coverage_detail.csv"), index=False)
    cov.to_csv(os.path.join(RESULTS, "interval_coverage.csv"), index=False)

    print(f"\nwrote csvs to {os.path.relpath(RESULTS, ROOT)}/")


if __name__ == "__main__":
    main()
