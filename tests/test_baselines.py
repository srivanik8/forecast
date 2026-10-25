import numpy as np
import pandas as pd
import pytest

from src.baselines import BASELINES, naive, seasonal_naive, seasonal_naive_drift


def monthly(values, start="2000-01-01"):
    idx = pd.date_range(start, periods=len(values), freq="MS")
    return pd.Series(values, index=idx, dtype=float)


#two years of a pattern that repeats every 12 months and rises by 10 a year
YEAR = np.arange(12, dtype=float)
TOY = monthly(np.concatenate([YEAR, YEAR + 10]))


def test_naive_repeats_last_value():
    f = naive(TOY, 3)
    assert list(f) == [21.0, 21.0, 21.0]


def test_forecast_index_starts_after_training():
    for fn in BASELINES.values():
        f = fn(TOY, 14)
        assert len(f) == 14
        assert f.index[0] == pd.Timestamp("2002-01-01")
        assert f.index.freqstr == "MS"


def test_seasonal_naive_repeats_last_year():
    f = seasonal_naive(TOY, 14)
    assert list(f.iloc[:12]) == list(YEAR + 10)
    #beyond one year it wraps round to the same months again
    assert list(f.iloc[12:]) == [10.0, 11.0]


def test_seasonal_naive_drift_adds_yearly_change():
    f = seasonal_naive_drift(TOY, 14)
    assert np.allclose(f.iloc[:12], YEAR + 20)
    assert np.allclose(f.iloc[12:], [30.0, 31.0])


def test_drift_recovers_a_perfect_trend_exactly():
    #season + linear trend: drift should forecast the continuation perfectly
    t = np.arange(48)
    full = monthly(np.sin(2 * np.pi * t / 12) + 0.5 * t)
    train, test = full.iloc[:36], full.iloc[36:]
    assert np.allclose(seasonal_naive_drift(train, 12), test)


@pytest.mark.parametrize("name", list(BASELINES))
def test_baselines_ignore_anything_after_training(name):
    #changing the future must not change the forecast: no leakage
    t = np.arange(60, dtype=float)
    a = monthly(t)
    b = a.copy()
    b.iloc[48:] = 999.0
    fn = BASELINES[name]
    assert fn(a.iloc[:48], 12).equals(fn(b.iloc[:48], 12))
