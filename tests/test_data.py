import numpy as np
import pandas as pd

from src.data import load_series, missing_report, to_monthly


def weekly(values, start="2000-01-01"):
    idx = pd.date_range(start, periods=len(values), freq="W-SAT")
    return pd.Series(values, index=idx, dtype=float, name="toy")


def test_missing_report_counts_gaps():
    s = weekly([1.0, np.nan, 3.0, np.nan])
    rep = missing_report(s)
    assert rep["n_obs"] == 4
    assert rep["n_missing"] == 2
    assert rep["pct_missing"] == 50.0
    assert rep["first_missing"] == s.index[1]


def test_missing_report_on_complete_series():
    rep = missing_report(weekly([1.0, 2.0]))
    assert rep["n_missing"] == 0
    assert rep["first_missing"] is None


def test_to_monthly_averages_within_month():
    idx = pd.to_datetime(["2000-01-01", "2000-01-15", "2000-02-05"])
    s = pd.Series([1.0, 3.0, 10.0], index=idx)
    m = to_monthly(s)
    assert m.loc["2000-01-01"] == 2.0
    assert m.loc["2000-02-01"] == 10.0


def test_to_monthly_fills_interior_month_with_no_readings():
    #january and march have readings, february has none at all
    idx = pd.to_datetime(["2000-01-15", "2000-03-15"])
    s = pd.Series([1.0, 3.0], index=idx)
    m = to_monthly(s)
    assert list(m.index) == list(pd.date_range("2000-01-01", "2000-03-01", freq="MS"))
    assert not m.isna().any()
    #time-weighted: jan->feb is 31 days of the 60 between jan and mar
    assert np.isclose(m.loc["2000-02-01"], 1.0 + 2.0 * 31 / 60)


def test_to_monthly_drops_edge_gaps_instead_of_extrapolating():
    idx = pd.to_datetime(["2000-01-15", "2000-02-15", "2000-03-15"])
    s = pd.Series([np.nan, 2.0, np.nan], index=idx)
    m = to_monthly(s)
    assert list(m.index) == [pd.Timestamp("2000-02-01")]


def test_load_series_is_clean_and_monthly():
    s = load_series()
    assert s.index.freqstr == "MS"
    assert not s.isna().any()
    assert s.index.is_monotonic_increasing
    #no skipped months anywhere
    assert len(s) == len(pd.date_range(s.index[0], s.index[-1], freq="MS"))
    assert len(s) >= 200
