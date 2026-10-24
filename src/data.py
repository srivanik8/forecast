"""
loading, cleaning and resampling.

everything downstream wants one thing: a pandas series with a monthly
datetime index and no gaps. swapping datasets means changing this file only.
"""

import pandas as pd
import statsmodels.api as sm


def load_raw():
    """mauna loa atmospheric co2 (ppm), weekly readings from 1958 to 2001."""
    raw = sm.datasets.co2.load_pandas().data["co2"]
    raw.name = "co2"
    return raw


def missing_report(series):
    """count and locate the gaps before deciding how to treat them."""
    gaps = series[series.isna()]
    return {
        "n_obs": int(len(series)),
        "n_missing": int(gaps.size),
        "pct_missing": float(100 * gaps.size / len(series)) if len(series) else 0.0,
        "first_missing": gaps.index.min() if gaps.size else None,
        "last_missing": gaps.index.max() if gaps.size else None,
    }


def to_monthly(series):
    """
    weekly -> monthly means, then bridge any months that had no readings.

    only interior gaps are filled (linear in time). leading or trailing gaps
    are dropped rather than extrapolated, since there is nothing either side
    to bridge from.
    """
    monthly = series.resample("MS").mean()
    monthly = monthly.interpolate(method="time", limit_area="inside")
    monthly = monthly.dropna()
    monthly = monthly.asfreq("MS")
    monthly.name = series.name
    return monthly


def load_series():
    """the clean monthly series every later step uses."""
    return to_monthly(load_raw())
