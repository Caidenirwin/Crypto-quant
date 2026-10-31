"""
Indicator calculations shared by strategies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def returns(close: pd.Series, periods: int = 1) -> pd.Series:
    return close.pct_change(periods=periods)


def rolling_volatility(close: pd.Series, window: int = 20) -> pd.Series:
    return close.pct_change().rolling(window).std(ddof=0) * np.sqrt(365)


def rolling_mean(series: pd.Series, window: int = 20) -> pd.Series:
    return series.rolling(window).mean()


def rolling_std(series: pd.Series, window: int = 20) -> pd.Series:
    return series.rolling(window).std(ddof=0)


def zscore(series: pd.Series, window: int = 20) -> pd.Series:
    mu = rolling_mean(series, window)
    sigma = rolling_std(series, window).replace(0, np.nan)
    return (series - mu) / sigma


def volume_spike_score(volume: pd.Series, window: int = 20) -> pd.Series:
    return zscore(volume.astype(float), window)

