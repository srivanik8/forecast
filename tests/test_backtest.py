import numpy as np
import pandas as pd
import pytest

from src.backtest import origins, run, score, summarise
from src.baselines import BASELINES


def monthly(values, start="1990-01-01"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="MS"),
                     dtype=float)


SERIES = monthly(np.arange(200))


def test_origins_step_a_year_and_end_at_series_end():
    cuts = origins(200)
    assert len(cuts) == 8
    assert cuts[-1] == (188, 200)
    assert [b - a for a, b in cuts] == [12] * 8
    assert [cuts[i + 1][0] - cuts[i][0] for i in range(7)] == [12] * 7


def test_origins_reject_a_series_that_is_too_short():
    with pytest.raises(ValueError):
        origins(100)


def test_training_never_reaches_the_forecast_window():
    seen = []

    def spy(train, horizon):
        seen.append(train.index[-1])
        return BASELINES["naive"](train, horizon)

    out = run(SERIES, {"spy": spy})
    for (train_end, _), last_seen in zip(origins(len(SERIES)), seen):
        assert last_seen == SERIES.index[train_end - 1]
    assert (out["train_end"] < out["date"]).all()
    assert (out["train_end"] < out["origin"]).all()


def test_changing_the_future_does_not_change_earlier_forecasts():
    #wreck the last two years: only the final origin trains on any of them
    wrecked = SERIES.copy()
    wrecked.iloc[-24:] = 1e6
    a = run(SERIES, BASELINES)
    b = run(wrecked, BASELINES)
    earlier = a["origin"] < SERIES.index[-24]
    last = a["origin"] == SERIES.index[-12]
    pd.testing.assert_series_equal(a.loc[earlier, "forecast"], b.loc[earlier, "forecast"])
    #sanity check that the test can fail: the final origin did see the change
    assert not np.allclose(a.loc[last, "forecast"], b.loc[last, "forecast"])


def test_one_row_per_model_origin_and_month_ahead():
    out = run(SERIES, BASELINES)
    assert len(out) == len(BASELINES) * 8 * 12
    assert sorted(out["horizon"].unique()) == list(range(1, 13))
    assert out.groupby(["model", "origin"]).size().eq(12).all()


def test_rows_record_actual_and_error_correctly():
    out = run(SERIES, BASELINES)
    assert np.allclose(out["actual"], SERIES.loc[out["date"]].to_numpy())
    assert np.allclose(out["error"], out["actual"] - out["forecast"])
    #naive on a straight line misses by exactly h steps of slope
    naive = out[out["model"] == "naive"]
    assert np.allclose(naive["error"], naive["horizon"])


def test_baselines_have_no_interval_and_models_keep_theirs():
    def with_interval(train, horizon):
        f = BASELINES["naive"](train, horizon)
        return pd.DataFrame({"mean": f, "lower": f - 1, "upper": f + 1})

    out = run(SERIES, {"naive": BASELINES["naive"], "banded": with_interval})
    assert out.loc[out["model"] == "naive", ["lower", "upper"]].isna().all().all()
    banded = out[out["model"] == "banded"]
    assert np.allclose(banded["upper"] - banded["lower"], 2.0)


def test_mismatched_forecast_dates_are_rejected():
    def wrong_dates(train, horizon):
        return BASELINES["naive"](train.iloc[:-1], horizon)

    with pytest.raises(ValueError):
        run(SERIES, {"wrong": wrong_dates})


def test_score_gives_one_row_per_model_and_origin():
    scores = score(run(SERIES, BASELINES), SERIES)
    assert len(scores) == len(BASELINES) * 8
    assert set(scores.columns) == {"model", "origin", "mae", "rmse", "mape", "mase"}


def test_score_scales_mase_on_each_origin_training_window_only():
    #a straight line has one-step naive error 1 everywhere, so naive's mase
    #equals its mae, which is the mean of 1..12 = 6.5
    scores = score(run(SERIES, BASELINES), SERIES)
    naive = scores[scores["model"] == "naive"]
    assert np.allclose(naive["mae"], 6.5)
    assert np.allclose(naive["mase"], 6.5)

    #make the last two years jumpy: only the final origin trains on them
    jumpy = SERIES.copy()
    jumpy.iloc[-24:] += np.tile([0, 50], 12)
    a = score(run(SERIES, BASELINES), SERIES)
    b = score(run(jumpy, BASELINES), jumpy)
    earlier = a["origin"] < SERIES.index[-24]
    last = a["origin"] == SERIES.index[-12]
    assert np.allclose(a.loc[earlier, "mase"], b.loc[earlier, "mase"])
    assert not np.allclose(a.loc[last, "mase"], b.loc[last, "mase"])


def test_summarise_compares_against_the_reference():
    scores = pd.DataFrame({
        "model": ["ref", "ref", "good", "good"],
        "origin": [1, 2, 1, 2],
        "mae": [2.0, 4.0, 1.0, 5.0],
        "rmse": 0.0, "mape": 0.0, "mase": [1.0, 2.0, 0.5, 2.5],
    })
    out = summarise(scores, reference="ref")
    #relative mae per origin: 0.5 and 1.25 -> mean 0.875, beats ref once
    assert out.loc["good", "rel_mae"] == pytest.approx(0.875)
    assert out.loc["good", "beats_reference"] == 1
    assert out.loc["ref", "rel_mae"] == pytest.approx(1.0)
    assert out.loc["ref", "beats_reference"] == 0
    assert out.loc["good", "mae"] == pytest.approx(3.0)


def test_summarise_needs_the_reference_model():
    scores = pd.DataFrame({"model": ["a"], "origin": [1], "mae": [1.0],
                           "rmse": 0.0, "mape": 0.0, "mase": 1.0})
    with pytest.raises(ValueError):
        summarise(scores, reference="missing")
