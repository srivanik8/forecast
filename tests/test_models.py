import numpy as np
import pandas as pd
import pytest

from src.models import MODELS


def synthetic(n=120, seed=0):
    #trend + annual season + a little noise, so both models have something to fit
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    values = 100 + 0.5 * t + 3 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 0.2, n)
    return pd.Series(values, index=pd.date_range("2000-01-01", periods=n, freq="MS"))


SERIES = synthetic()
TRAIN, TEST = SERIES.iloc[:-12], SERIES.iloc[-12:]


@pytest.mark.parametrize("name", list(MODELS))
def test_forecast_shape_and_index(name):
    f = MODELS[name](TRAIN, 12)
    assert list(f.columns) == ["mean", "lower", "upper"]
    assert f.index.equals(TEST.index)
    assert not f.isna().any().any()


@pytest.mark.parametrize("name", list(MODELS))
def test_interval_contains_point_forecast(name):
    f = MODELS[name](TRAIN, 12)
    assert (f["lower"] < f["mean"]).all()
    assert (f["mean"] < f["upper"]).all()


@pytest.mark.parametrize("name", list(MODELS))
def test_model_tracks_a_clean_seasonal_trend(name):
    f = MODELS[name](TRAIN, 12)
    assert np.mean(np.abs(f["mean"] - TEST)) < 1.0


@pytest.mark.parametrize("name", list(MODELS))
def test_forecast_ignores_data_after_training(name):
    #wrecking the future must not change the forecast: no leakage
    wrecked = SERIES.copy()
    wrecked.iloc[-12:] = 1e6
    a = MODELS[name](SERIES.iloc[:-12], 12)
    b = MODELS[name](wrecked.iloc[:-12], 12)
    pd.testing.assert_frame_equal(a, b)
