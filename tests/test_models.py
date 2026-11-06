import numpy as np
import pandas as pd
import pytest

from src.models import CANDIDATES, FITTERS, MODELS, diagnostics, fit_sarima, sarima_auto

pytestmark = pytest.mark.slow


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


@pytest.mark.parametrize("name", list(FITTERS))
def test_diagnostics_report_expected_fields(name):
    d = diagnostics(FITTERS[name](TRAIN))
    assert set(d) == {"aic", "resid_std", "ljung_box_p"}
    assert 0 <= d["ljung_box_p"] <= 1
    #noise was generated with std 0.2, residuals should be close to that
    assert d["resid_std"] < 0.5


@pytest.mark.parametrize("name", list(CANDIDATES))
def test_candidate_forecast_shape_intervals_and_no_leakage(name):
    fn = CANDIDATES[name]
    f = fn(TRAIN, 12)
    assert list(f.columns) == ["mean", "lower", "upper"]
    assert f.index.equals(TEST.index)
    assert (f["lower"] < f["mean"]).all() and (f["mean"] < f["upper"]).all()
    assert np.mean(np.abs(f["mean"] - TEST)) < 1.0
    wrecked = SERIES.copy()
    wrecked.iloc[-12:] = 1e6
    pd.testing.assert_frame_equal(f, fn(wrecked.iloc[:-12], 12))


def test_log_holt_winters_intervals_are_wider_above_than_below():
    f = CANDIDATES["holt_winters_log"](TRAIN, 12)
    assert ((f["upper"] - f["mean"]) > (f["mean"] - f["lower"])).all()


def test_auto_sarima_picks_the_lowest_aic_order_from_its_grid():
    grid = [((0, 1, 0), (0, 1, 0, 12)), ((1, 1, 1), (1, 1, 1, 12))]
    f = sarima_auto(TRAIN, 12, grid=grid)
    aics = {g: fit_sarima(TRAIN, order=g[0], seasonal_order=g[1]).aic for g in grid}
    assert f.attrs["orders"] == min(aics, key=aics.get)


def test_ensemble_is_the_average_of_sarima_and_holt_winters():
    f = CANDIDATES["ensemble"](TRAIN, 12)
    expected = (MODELS["sarima"](TRAIN, 12) + MODELS["holt_winters"](TRAIN, 12)) / 2
    pd.testing.assert_frame_equal(f, expected)
