"""
the real models: sarima and holt-winters.

each function fits from scratch on the training series it is given and
returns a dataframe indexed by the next `horizon` months with columns
mean, lower and upper (the 95% prediction interval). nothing outside the
training series is used, so every call is a clean refit.
"""

import itertools
import warnings

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.statespace.sarimax import SARIMAX

SEASON = 12
ALPHA = 0.05  #95% prediction intervals


def _frame(mean, lower, upper, index):
    return pd.DataFrame({"mean": mean, "lower": lower, "upper": upper}, index=index)


def fit_sarima(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, SEASON)):
    """sarima(1,1,1)(1,1,1,12) - the standard starting point for monthly data."""
    with warnings.catch_warnings():
        #convergence chatter is expected on some windows, not an error
        warnings.simplefilter("ignore")
        return SARIMAX(train, order=order, seasonal_order=seasonal_order).fit(disp=False)


def sarima(train, horizon):
    fit = fit_sarima(train)
    pred = fit.get_forecast(horizon).summary_frame(alpha=ALPHA)
    return _frame(pred["mean"].to_numpy(), pred["mean_ci_lower"].to_numpy(),
                  pred["mean_ci_upper"].to_numpy(), pred.index)


def fit_holt_winters(train, seasonal="add", damped=False):
    """
    holt-winters with additive trend and seasonality.

    fitted as the ets(a,a,a) state-space model, which gives the same point
    forecasts as classic holt-winters but also proper prediction intervals.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ETSModel(train, error="add", trend="add", damped_trend=damped,
                        seasonal=seasonal, seasonal_periods=SEASON).fit(disp=False)


def _ets_forecast(fit, n, horizon):
    pred = fit.get_prediction(start=n, end=n + horizon - 1).summary_frame(alpha=ALPHA)
    return pred["mean"].to_numpy(), pred["pi_lower"].to_numpy(), pred["pi_upper"].to_numpy(), pred.index


def holt_winters(train, horizon):
    mean, lower, upper, index = _ets_forecast(fit_holt_winters(train), len(train), horizon)
    return _frame(mean, lower, upper, index)


def holt_winters_log(train, horizon):
    """
    holt-winters fitted to log co2, forecasts mapped back with exp.

    growth that is steady in percentage terms is linear on the log scale, and
    the intervals come back lopsided - wider above than below - which is the
    side every backtest miss fell on.
    """
    mean, lower, upper, _ = _ets_forecast(fit_holt_winters(np.log(train)), len(train), horizon)
    index = pd.date_range(train.index[-1], periods=horizon + 1, freq=train.index.freq)[1:]
    return _frame(np.exp(mean), np.exp(lower), np.exp(upper), index)


MODELS = {
    "sarima": sarima,
    "holt_winters": holt_winters,
}


def diagnostics(fit, skip=SEASON + 1, lags=24):
    """
    how well a fitted model describes its own training data.

    the first season or so of residuals is dropped: differencing and state
    initialisation make them meaningless. a ljung-box p-value below 0.05
    means pattern is left in the residuals that the model has not captured.
    """
    resid = np.asarray(fit.resid)[skip:]
    lb = acorr_ljungbox(resid, lags=[lags])
    return {
        "aic": round(float(fit.aic), 1),
        "resid_std": round(float(resid.std()), 3),
        "ljung_box_p": round(float(lb["lb_pvalue"].iloc[0]), 3),
    }


FITTERS = {
    "sarima": fit_sarima,
    "holt_winters": fit_holt_winters,
}


def holt_winters_damped(train, horizon):
    """holt-winters whose trend flattens out over the horizon - often safer long range."""
    mean, lower, upper, index = _ets_forecast(fit_holt_winters(train, damped=True),
                                              len(train), horizon)
    return _frame(mean, lower, upper, index)


#small grid around the default: d and D stay at 1, the series needs both
SARIMA_GRID = [((p, 1, q), (P, 1, Q, SEASON))
               for p, q, P, Q in itertools.product([0, 1, 2], [0, 1], [0, 1], [0, 1])]


def sarima_auto(train, horizon, grid=SARIMA_GRID):
    """
    sarima with its orders picked by aic, refit on each training window.

    the choice only ever sees the training window, so tuning cannot leak the
    test period. the chosen orders are kept in the result's attrs.
    """
    best, best_orders = None, None
    for order, seasonal_order in grid:
        try:
            fit = fit_sarima(train, order=order, seasonal_order=seasonal_order)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if best is None or fit.aic < best.aic:
            best, best_orders = fit, (order, seasonal_order)
    pred = best.get_forecast(horizon).summary_frame(alpha=ALPHA)
    out = _frame(pred["mean"].to_numpy(), pred["mean_ci_lower"].to_numpy(),
                 pred["mean_ci_upper"].to_numpy(), pred.index)
    out.attrs["orders"] = best_orders
    return out


#candidate improvements, kept out of MODELS so the main results stay as reported.
#src.experiments puts them through the same backtest
CANDIDATES = {
    "holt_winters_log": holt_winters_log,
    "holt_winters_damped": holt_winters_damped,
    "sarima_auto": sarima_auto,
}
