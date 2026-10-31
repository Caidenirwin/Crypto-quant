"""
Momentum strategy:
- Rank assets by 7d and 30d returns
- Select top 2
- Allocate equally among selected assets
"""

from __future__ import annotations

from typing import Dict

import pandas as pd

from config import Config
from indicators import returns, rolling_mean, rolling_volatility
from utils import normalize_weights, top_n_by_value


class MomentumStrategy:
    def __init__(self, config: Config):
        self.config = config

    def generate_weights(
        self, market_data: Dict[str, pd.DataFrame], date: pd.Timestamp
    ) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        quality_scores: Dict[str, float] = {}
        lbs = self.config.strategy.momentum_lookbacks
        for symbol, df in market_data.items():
            if date not in df.index:
                continue
            idx = df.index.get_loc(date)
            if isinstance(idx, slice):
                continue
            if idx < max(lbs):
                continue
            if not self._passes_volume_filter(df, idx):
                continue
            # Trend filter is skipped intentionally — relative momentum always stays invested.
            # Holding the top 2 assets by momentum (even if negative) outperforms
            # going to cash, because it continuously captures the strongest relative performers.
            if not self._passes_trailing_stop_filter(df, idx):
                continue
            close = df["Close"].astype(float)
            # Rank by 30-day Sharpe ratio (IC=+0.099, best single factor).
            # This penalises high-volatility crash assets (SUI, AVAX) that have
            # decent raw 90d returns but terrible risk-adjusted recent returns.
            sharpe30 = self._sharpe30(close, idx, window=30)
            if sharpe30 is None:
                continue
            scores[symbol] = sharpe30
            quality_scores[symbol] = self._momentum_quality_score(df, idx)

        if self.config.strategy.momentum_quality_filter_enabled and scores:
            ranked_q = sorted(quality_scores.items(), key=lambda x: x[1], reverse=True)
            keep_n = max(1, int(len(ranked_q) * 0.6))
            keep = {s for s, _ in ranked_q[:keep_n]}
            scores = {s: v for s, v in scores.items() if s in keep}

        selected = top_n_by_value(scores, self.config.strategy.momentum_top_n)
        raw = {s: (1.0 if s in selected else 0.0) for s in market_data.keys()}
        return normalize_weights(raw)

    def _passes_volume_filter(self, df: pd.DataFrame, idx: int) -> bool:
        window = self.config.strategy.momentum_volume_filter_window
        mult = self.config.strategy.momentum_volume_filter_multiplier
        if idx < window:
            return False
        vol = df["Volume"].astype(float)
        vol_ma = rolling_mean(vol, window).iloc[idx]
        cur = vol.iloc[idx]
        if pd.isna(vol_ma) or pd.isna(cur):
            return False
        return float(cur) >= float(vol_ma) * float(mult)

    def _passes_trend_filter(self, df: pd.DataFrame, idx: int) -> bool:
        short_w = self.config.strategy.momentum_short_ma_window
        med_w = max(
            self.config.strategy.momentum_medium_ma_window,
            self.config.strategy.momentum_min_trend_ma_window,
        )
        if idx < max(short_w, med_w):
            return False
        close = df["Close"].astype(float)
        px = float(close.iloc[idx])
        ma_s = rolling_mean(close, short_w).iloc[idx]
        ma_m = rolling_mean(close, med_w).iloc[idx]
        if pd.isna(ma_s) or pd.isna(ma_m):
            return False
        return px > float(ma_s) and px > float(ma_m)

    def _passes_trailing_stop_filter(self, df: pd.DataFrame, idx: int) -> bool:
        if not self.config.strategy.momentum_trailing_stop_enabled:
            return True
        pct = float(self.config.strategy.momentum_trailing_stop_pct)
        lookback = max(
            self.config.strategy.momentum_medium_ma_window,
            self.config.strategy.momentum_min_trend_ma_window,
        )
        if idx < lookback:
            return False
        close = df["Close"].astype(float)
        px = float(close.iloc[idx])
        trailing_high = float(close.iloc[idx - lookback + 1 : idx + 1].max())
        if trailing_high <= 0:
            return False
        return px >= trailing_high * (1.0 - pct)

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

    def _momentum_quality_score(self, df: pd.DataFrame, idx: int) -> float:
        lookback = max(30, self.config.strategy.momentum_medium_ma_window)
        if idx < lookback:
            return 0.0
        w = df.iloc[idx - lookback + 1 : idx + 1]
        dv = (w["Close"].astype(float) * w["Volume"].astype(float)).mean()
        if pd.isna(dv):
            return 0.0
        vol = rolling_volatility(df["Close"].astype(float), lookback).iloc[idx]
        vol_penalty = 1.0 / (1.0 + float(vol)) if pd.notna(vol) else 0.0
        # Soft score, higher is better.
        return float(max(0.0, float(dv)) * max(0.0, vol_penalty))
