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
