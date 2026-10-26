"""
the real models: sarima and holt-winters.

each function fits from scratch on the training series it is given and
returns a dataframe indexed by the next `horizon` months with columns
mean, lower and upper (the 95% prediction interval). nothing outside the
training series is used, so every call is a clean refit.
"""

import warnings

import pandas as pd
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.statespace.sarimax import SARIMAX

SEASON = 12
ALPHA = 0.05  #95% prediction intervals


def _frame(mean, lower, upper, index):
    return pd.DataFrame({"mean": mean, "lower": lower, "upper": upper}, index=index)


def sarima(train, horizon, order=(1, 1, 1), seasonal_order=(1, 1, 1, SEASON)):
    """sarima(1,1,1)(1,1,1,12) - the standard starting point for monthly data."""
    with warnings.catch_warnings():
        #convergence chatter is expected on some windows, not an error
        warnings.simplefilter("ignore")
        fit = SARIMAX(train, order=order, seasonal_order=seasonal_order).fit(disp=False)
    pred = fit.get_forecast(horizon).summary_frame(alpha=ALPHA)
    return _frame(pred["mean"].to_numpy(), pred["mean_ci_lower"].to_numpy(),
                  pred["mean_ci_upper"].to_numpy(), pred.index)


def holt_winters(train, horizon, seasonal="add"):
    """
    holt-winters with additive trend and seasonality.

    fitted as the ets(a,a,a) state-space model, which gives the same point
    forecasts as classic holt-winters but also proper prediction intervals.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ETSModel(train, error="add", trend="add", seasonal=seasonal,
                       seasonal_periods=SEASON).fit(disp=False)
    n = len(train)
    pred = fit.get_prediction(start=n, end=n + horizon - 1).summary_frame(alpha=ALPHA)
    return _frame(pred["mean"].to_numpy(), pred["pi_lower"].to_numpy(),
                  pred["pi_upper"].to_numpy(), pred.index)


MODELS = {
    "sarima": sarima,
    "holt_winters": holt_winters,
}
