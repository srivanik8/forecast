"""
rolling-origin backtest.

at each origin the forecaster is refit from scratch on everything before the
origin and forecasts the next `horizon` months. origins step forward a year
at a time and the last one ends at the end of the series. no forecaster ever
sees a value from on or after its origin.
"""

HORIZON = 12
N_ORIGINS = 8
STEP = 12  #a year apart, so the forecast windows never overlap


def origins(n, horizon=HORIZON, n_origins=N_ORIGINS, step=STEP):
    """
    (train_end, test_end) positions for each origin, oldest first.

    train is series[:train_end], test is series[train_end:test_end].
    """
    last = n - horizon
    cuts = [last - step * (n_origins - 1 - i) for i in range(n_origins)]
    if cuts[0] < 2 * horizon:
        raise ValueError(f"series of length {n} is too short for {n_origins} origins")
    return [(c, c + horizon) for c in cuts]
