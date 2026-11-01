"""
rerun the backtest from scratch and check it matches the committed results.

if this fails, either the code changed the numbers (rerun python -m src.run_all
and update the readme) or a dependency version changed them (check
requirements.txt).
"""

import os

import numpy as np
import pandas as pd
import pytest

from src.backtest import coverage_summary, run, score, summarise
from src.data import load_series
from src.run_backtest import FORECASTERS

pytestmark = pytest.mark.slow

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


@pytest.fixture(scope="module")
def fresh():
    series = load_series()
    return series, run(series, FORECASTERS)


def committed(name, **kwargs):
    return pd.read_csv(os.path.join(RESULTS, name), **kwargs)


def test_backtest_matches_committed_results(fresh):
    _, out = fresh
    saved = committed("backtest.csv")
    assert len(out) == len(saved)
    assert list(out["model"]) == list(saved["model"])
    for col in ["forecast", "lower", "upper", "actual", "error"]:
        assert np.allclose(out[col], saved[col], atol=1e-4, equal_nan=True), col


def test_metrics_match_committed_results(fresh):
    series, out = fresh
    summary = summarise(score(out, series)).round(3)
    saved = committed("metrics.csv", index_col="model")
    pd.testing.assert_frame_equal(summary, saved.loc[summary.index], check_dtype=False)


def test_coverage_matches_committed_results(fresh):
    _, out = fresh
    cov = coverage_summary(out).round(3)
    saved = committed("coverage.csv", index_col="model")
    for col in ["inside", "coverage", "mean_width", "missed_above"]:
        assert np.allclose(cov[col], saved.loc[cov.index, col]), col


def test_a_second_run_is_identical(fresh):
    #nothing in the pipeline is random, so two runs must agree exactly
    series, out = fresh
    again = run(series, FORECASTERS)
    pd.testing.assert_frame_equal(out, again)
