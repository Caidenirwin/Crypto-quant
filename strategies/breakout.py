"""
Breakout strategy:
- Detect 14-day highs
- Confirm with volume spike
- Enter long on confirmation
"""

from __future__ import annotations

from typing import Dict

import pandas as pd

from config import Config
from indicators import rolling_mean, volume_spike_score
from utils import normalize_weights


class BreakoutStrategy:
    def __init__(self, config: Config):
        self.config = config

    def generate_weights(
        self, market_data: Dict[str, pd.DataFrame], date: pd.Timestamp
    ) -> Dict[str, float]:
        raw: Dict[str, float] = {}
        lb = self.config.strategy.breakout_lookback
        vol_z = self.config.strategy.breakout_volume_spike_z
        vol_window = self.config.strategy.breakout_volume_filter_window
        vol_mult = self.config.strategy.breakout_volume_filter_multiplier
        trend_required = self.config.strategy.breakout_require_trend_confirmation
        trend_window = self.config.strategy.breakout_trend_ma_window

        for symbol, df in market_data.items():
            if date not in df.index:
                continue
            idx = df.index.get_loc(date)
            if isinstance(idx, slice) or idx < max(lb, vol_window, trend_window):
                raw[symbol] = 0.0
                continue

            close = df["Close"]
            volume = df["Volume"]
            px = close.iloc[idx]
            prev_high = close.iloc[idx - lb : idx].max()
            v_z = volume_spike_score(volume, lb).iloc[idx]
            vol_ma = rolling_mean(volume.astype(float), vol_window).iloc[idx]
            vol_pass = pd.notna(vol_ma) and float(volume.iloc[idx]) >= float(vol_ma) * float(vol_mult)
            trend_ma = rolling_mean(close.astype(float), trend_window).iloc[idx]
            trend_pass = (not trend_required) or (
                pd.notna(trend_ma) and float(px) > float(trend_ma)
            )

            if pd.notna(v_z) and vol_pass and trend_pass and px > prev_high and v_z >= vol_z:
                raw[symbol] = 1.0
            else:
                raw[symbol] = 0.0

        return normalize_weights(raw)
