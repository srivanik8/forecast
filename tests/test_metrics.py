import numpy as np
import pytest

from src.metrics import coverage, interval_width, mae, mape, mase, misses, rmse


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


def test_mase_on_known_example():
    #naive steps on training are all 2, forecast misses by 1 -> 0.5
    assert mase([10, 12], [11, 11], train=[0, 2, 4, 6]) == pytest.approx(0.5)


def test_mase_with_seasonal_scale():
    #season of 2: steps two back are all 4, forecast misses by 2 -> 0.5
    assert mase([10], [8], train=[0, 1, 4, 5, 8, 9], season=2) == pytest.approx(0.5)


def test_mase_of_naive_on_a_random_walk_is_about_one():
    rng = np.random.default_rng(0)
    walk = np.cumsum(rng.normal(size=20_000))
    train, test = walk[:-1], walk[1:]
    #one-step naive forecasts of each next value
    assert mase(test[-5000:], walk[-5001:-1], train) == pytest.approx(1.0, abs=0.05)


def test_mase_refuses_constant_training_window():
    with pytest.raises(ValueError):
        mase([1], [1], train=[3, 3, 3])


def test_coverage_counts_actuals_inside_the_interval():
    #inside, on the edge (counts), above, below
    actual = [5, 10, 12, -1]
    assert coverage(actual, [0, 0, 0, 0], [10, 10, 10, 10]) == pytest.approx(0.5)


def test_coverage_rejects_inverted_intervals():
    with pytest.raises(ValueError):
        coverage([1], [2], [0])


def test_interval_width_is_the_mean_width():
    assert interval_width([0, 1], [2, 5]) == pytest.approx(3.0)


def test_misses_split_by_direction():
    assert misses([5, 12, -1, 20], [0] * 4, [10] * 4) == {"below": 1, "above": 2}
