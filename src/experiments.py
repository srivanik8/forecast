"""
can the models be improved, or the benchmark made harder?

run from the repo root:  python -m src.experiments   (about a minute)

puts every candidate model and baseline through the same rolling-origin
backtest as the main results: same 8 origins, refit at each, no data from
after the origin. everything is reported, including what did not help.
kept out of run_all because the tuned sarima is slow.
"""

import os
import time

import pandas as pd

from src.backtest import coverage_by, coverage_summary, run, score, summarise
from src.baselines import BASELINES, CANDIDATE_BASELINES
from src.data import load_series
from src.models import CANDIDATES, MODELS, sarima_auto

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


def timed(fn, seconds, name):
    def wrapper(train, horizon):
        start = time.perf_counter()
        out = fn(train, horizon)
        seconds[name] = seconds.get(name, 0.0) + time.perf_counter() - start
        return out
    return wrapper


def main():
    series = load_series()
    seconds, orders = {}, []

    def sarima_auto_logged(train, horizon):
        f = sarima_auto(train, horizon)
        orders.append({"origin": f.index[0], "order": f.attrs["orders"][0],
                       "seasonal_order": f.attrs["orders"][1]})
        return f

    forecasters = {**BASELINES, **CANDIDATE_BASELINES, **MODELS,
                   **CANDIDATES, "sarima_auto": sarima_auto_logged}
    out = run(series, {n: timed(fn, seconds, n) for n, fn in forecasters.items()})
    scores = score(out, series)

    table = summarise(scores, reference="seasonal_naive_drift")
    table["rel_mae_vs_drift_10y"] = summarise(scores, reference="drift_10y")["rel_mae"]
    cov = coverage_summary(out)
    table["coverage"] = cov["coverage"]
    table["coverage_1998"] = coverage_by(out, "origin").filter(like="1998").iloc[:, 0]
    table["mean_width"] = cov["mean_width"]
    table["fit_seconds"] = pd.Series(seconds)
    table = table.drop(columns="origins").sort_values("mae")

    table.round(3).to_csv(os.path.join(RESULTS, "experiments.csv"))
    pd.DataFrame(orders).to_csv(os.path.join(RESULTS, "experiments_sarima_orders.csv"),
                                index=False, date_format="%Y-%m")

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table.round(3).to_string())
        print("\norders picked by sarima_auto\n")
        print(pd.DataFrame(orders).to_string(index=False))


if __name__ == "__main__":
    main()
