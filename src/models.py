"""
the real models: sarima and holt-winters.

each function fits from scratch on the training series it is given and
returns a dataframe indexed by the next `horizon` months with columns
mean, lower and upper (the 95% prediction interval). nothing outside the
training series is used, so every call is a clean refit.
"""

import warnings

import pandas as pd
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
