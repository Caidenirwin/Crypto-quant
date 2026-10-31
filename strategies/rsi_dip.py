"""
RSI Dip Strategy — Aggressive mean-reversion on oversold crypto assets.

Pattern: crypto assets that hit extreme oversold (RSI < threshold) and then
show early signs of momentum recovery produce some of the highest-ROI setups
in the space. Bounces from RSI < 35 typically run 20-40% in 1-3 weeks.

Entry logic (two modes):
  1. Extreme oversold (RSI < 30): enter immediately — don't wait for recovery
  2. Moderate oversold (30 < RSI < threshold): enter only if 7d return is
     improving vs 14d return (first bounce signal)

Scoring: lower RSI = higher score, amplified by short-term recovery strength.
Allocation: top N assets, weighted by score (not equal weight) — more aggressive.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd

from config import Config
from utils import normalize_weights


class RSIDipStrategy:
    def __init__(self, config: Config):
        self.config = config

    def generate_weights(
        self, market_data: Dict[str, pd.DataFrame], date: pd.Timestamp
    ) -> Dict[str, float]:
        scores: Dict[str, float] = {}

        rsi_threshold  = float(self.config.strategy.rsi_dip_entry_threshold)
        rsi_extreme    = float(self.config.strategy.rsi_dip_extreme_threshold)
        min_history    = int(self.config.strategy.rsi_dip_min_history)
        top_n          = int(self.config.strategy.rsi_dip_top_n)
        vol_filter_on  = bool(self.config.strategy.rsi_dip_volume_filter_enabled)
        vol_window     = int(self.config.strategy.rsi_dip_volume_window)
        vol_mult       = float(self.config.strategy.rsi_dip_volume_multiplier)

        for symbol, df in market_data.items():
            if date not in df.index:
                continue
            idx = df.index.get_loc(date)
            if isinstance(idx, slice):
                continue
            if idx < min_history:
                continue

            close  = df["Close"].astype(float)
            volume = df["Volume"].astype(float)

            # RSI(14)
            rsi = self._rsi(close, idx, period=14)
            if rsi is None or rsi > rsi_threshold:
                continue

            # Sharpe30 gate: block assets in a genuine sustained crash.
            # IC analysis shows RSI<40 alone is a negative signal; we only
            # want temporary dips from otherwise healthy assets (Sharpe30 > -1.0).
            min_sharpe = float(self.config.strategy.rsi_dip_min_sharpe30)
            sharpe30 = self._sharpe30(close, idx, window=30)
            if sharpe30 is None or sharpe30 < min_sharpe:
                continue

            # Volume filter — skip if volume dried up (not a real dip, just illiquid)
            if vol_filter_on and idx >= vol_window:
                vol_ma = float(volume.iloc[idx - vol_window : idx].mean())
                cur_vol = float(volume.iloc[idx])
                if cur_vol < vol_ma * vol_mult:
                    continue

            # Short-term return windows for bounce confirmation
            r5  = (float(close.iloc[idx]) / float(close.iloc[idx - 5])  - 1) if idx >= 5  else 0.0
            r7  = (float(close.iloc[idx]) / float(close.iloc[idx - 7])  - 1) if idx >= 7  else 0.0
            r14 = (float(close.iloc[idx]) / float(close.iloc[idx - 14]) - 1) if idx >= 14 else 0.0

            # Asset must not be making new 10-day lows (confirms bounce has started)
            if idx >= 10:
                lo_10 = float(close.iloc[idx - 10 : idx].min())
                if float(close.iloc[idx]) < lo_10 * 1.01:
                    continue

            # Extreme oversold: still require 5-day bounce started (avoid pure freefall)
            if rsi <= rsi_extreme:
                if r5 < 0.0:
                    continue
                recovery_factor = 1.5  # bonus for extreme oversold
            else:
                # Moderate oversold: require positive short-term return (bounce confirmed)
                if r7 <= 0.02:  # must have at least 2% positive over 7 days
                    continue
                if r7 < r14:    # 7d must be improving vs 14d (momentum turning up)
                    continue
                recovery_factor = max(0.5, 1.0 + r7 * 3)

            # Score: how deeply oversold × bounce strength
            depth = rsi_threshold - rsi  # higher = more oversold
            scores[symbol] = depth * recovery_factor

        if not scores:
            return {s: 0.0 for s in market_data}

        # Rank and select top N
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected = ranked[:top_n]

        if not selected:
            return {s: 0.0 for s in market_data}

        # Score-proportional weighting (not equal weight) — more aggressive
        total_score = sum(sc for _, sc in selected)
        raw = {s: 0.0 for s in market_data}
        for sym, sc in selected:
            raw[sym] = sc / total_score if total_score > 0 else 1.0 / len(selected)

        return normalize_weights(raw)

    def _sharpe30(self, close: pd.Series, idx: int, window: int = 30) -> float | None:
        if idx < window:
            return None
        r = close.pct_change().iloc[idx - window + 1 : idx + 1].dropna()
        if len(r) < window // 2:
            return None
        sigma = float(r.std())
        if sigma <= 0:
            return None
        return float(r.mean()) / sigma

    def _rsi(self, close: pd.Series, idx: int, period: int = 14) -> float | None:
        if idx < period + 1:
            return None
        window = close.iloc[idx - period : idx + 1]
        delta = window.diff().dropna()
        gain = delta.clip(lower=0).mean()
        loss = (-delta.clip(upper=0)).mean()
        if loss == 0:
            return 100.0
        rs = gain / loss
        return float(100 - 100 / (1 + rs))
