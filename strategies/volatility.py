"""
Volatility range strategy:
- Find low-volatility/range-bound assets
- Buy near lower band, sell near upper band
"""

from __future__ import annotations

from typing import Dict

import pandas as pd

from config import Config
from indicators import rolling_mean, rolling_std, rolling_volatility
from utils import normalize_weights


class VolatilityStrategy:
    def __init__(self, config: Config):
        self.config = config

    def generate_weights(
        self, market_data: Dict[str, pd.DataFrame], date: pd.Timestamp
    ) -> Dict[str, float]:
        raw: Dict[str, float] = {}
        window = self.config.strategy.volatility_window
        vol_threshold = self.config.strategy.volatility_threshold
        k = self.config.strategy.volatility_band_k

        for symbol, df in market_data.items():
            if date not in df.index:
                continue
            idx = df.index.get_loc(date)
            if isinstance(idx, slice) or idx < window:
                raw[symbol] = 0.0
                continue

            close = df["Close"]
            vol = rolling_volatility(close, window).iloc[idx]
            mu = rolling_mean(close, window).iloc[idx]
            sd = rolling_std(close, window).iloc[idx]
            px = close.iloc[idx]
            if pd.isna(vol) or pd.isna(mu) or pd.isna(sd):
                raw[symbol] = 0.0
                continue

            lower = mu - k * sd
            upper = mu + k * sd
            if vol < vol_threshold and px <= lower:
                raw[symbol] = 1.0
            elif vol < vol_threshold and px >= upper:
                raw[symbol] = -1.0
            else:
                raw[symbol] = 0.0

        return normalize_weights(raw)

