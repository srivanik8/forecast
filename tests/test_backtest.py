import numpy as np
import pandas as pd
import pytest

from src.backtest import origins, run
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
    wrecked = SERIES.copy()
    wrecked.iloc[-12:] = 1e6
    a = run(SERIES, BASELINES)
    b = run(wrecked, BASELINES)
    #every origin but the last forecasts before the wrecked year
    earlier = a["origin"] < SERIES.index[-12]
    pd.testing.assert_series_equal(a.loc[earlier, "forecast"], b.loc[earlier, "forecast"])
