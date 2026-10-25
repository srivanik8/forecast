import pytest

from src.metrics import mae


def test_mae_on_known_example():
    assert mae([1, 2, 3], [2, 2, 5]) == pytest.approx(1.0)


def test_mae_is_zero_for_perfect_forecast():
    assert mae([4.5, 6.0], [4.5, 6.0]) == 0.0


def test_mae_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        mae([1, 2, 3], [1, 2])
