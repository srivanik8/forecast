import numpy as np
import pytest

from src.metrics import mae, mape, rmse


def test_mae_on_known_example():
    assert mae([1, 2, 3], [2, 2, 5]) == pytest.approx(1.0)


def test_mae_is_zero_for_perfect_forecast():
    assert mae([4.5, 6.0], [4.5, 6.0]) == 0.0


def test_mae_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        mae([1, 2, 3], [1, 2])


def test_rmse_on_known_example():
    #errors 1, 0, 2 -> sqrt((1 + 0 + 4) / 3)
    assert rmse([1, 2, 3], [2, 2, 5]) == pytest.approx(np.sqrt(5 / 3))


def test_rmse_punishes_one_big_miss_more_than_mae():
    actual = [0, 0, 0, 0]
    spread, spike = [1, 1, 1, 1], [0, 0, 0, 4]
    assert mae(actual, spread) == mae(actual, spike)
    assert rmse(actual, spike) > rmse(actual, spread)


def test_mape_on_known_example():
    #10% and 5% off -> 7.5%
    assert mape([100, 200], [110, 190]) == pytest.approx(7.5)


def test_mape_refuses_zero_actuals():
    with pytest.raises(ValueError):
        mape([0, 1], [1, 1])
