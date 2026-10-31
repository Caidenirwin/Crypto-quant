"""
Yield strategy:
- Simulate daily yield accrual on held assets
- Convert accrued yield to core assets on a schedule
"""

from __future__ import annotations

from typing import Dict

import pandas as pd

from config import Config


class YieldStrategy:
    def __init__(self, config: Config):
        self.config = config
        self.yield_buffer = 0.0
        self.last_conversion_date: pd.Timestamp | None = None

    def accrue_yield(self, holdings_value: float) -> float:
        daily_yield = holdings_value * self.config.strategy.yield_daily_rate
        self.yield_buffer += daily_yield
        return daily_yield

    def conversion_amount(self, date: pd.Timestamp) -> float:
        if self.last_conversion_date is None:
            self.last_conversion_date = date
            return 0.0
        days = (date - self.last_conversion_date).days
        if days >= self.config.strategy.yield_conversion_days:
            amount = self.yield_buffer
            self.yield_buffer = 0.0
            self.last_conversion_date = date
            return amount
        return 0.0

    def generate_weights(
        self, market_data: Dict[str, pd.DataFrame], date: pd.Timestamp
    ) -> Dict[str, float]:
        symbols = sorted(market_data.keys())
        if not symbols or not self.config.strategy.yield_enabled:
            return {s: 0.0 for s in symbols}

        targets = [
            s
            for s in self.config.strategy.yield_preferred_targets
            if s in market_data
        ]
        if not targets and self.config.strategy.yield_fallback_to_quality_assets:
            targets = self._fallback_quality_targets(market_data, date)
        if not targets:
            return {s: 0.0 for s in symbols}

        w = 1.0 / float(len(targets))
        return {s: (w if s in set(targets) else 0.0) for s in symbols}

    def _fallback_quality_targets(
        self, market_data: Dict[str, pd.DataFrame], date: pd.Timestamp
    ) -> list[str]:
        scored: Dict[str, float] = {}
        for s, df in market_data.items():
            if date not in df.index:
                continue
            idx = df.index.get_loc(date)
            if isinstance(idx, slice) or idx < 1:
                continue
            hist = df.iloc[: idx + 1]
            if len(hist) < 20:
                continue
            window = hist.iloc[max(0, len(hist) - 60) :]
            dollar_vol = (window["Close"].astype(float) * window["Volume"].astype(float)).mean()
            if pd.isna(dollar_vol):
                continue
            scored[s] = float(dollar_vol)
        if not scored:
            return []
        ranked = sorted(scored.items(), key=lambda x: x[1], reverse=True)
        return [s for s, _ in ranked[:2]]
