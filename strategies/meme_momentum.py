"""
Meme Momentum Strategy (hard-capped carve-out, outside the core V7 engine).

Logic:
  1. Every meme coin must first pass the anti-rug history check (min age,
     no crash, no liquidity drain, no parabolic pump).
  2. Entry filter: Sharpe14 >= gate, close above 20d MA, volume at least
     70% of its 20d average (real participation, not a thin-book drift).
  3. Rank passers by Sharpe14, hold the top-N equally weighted.
  4. When nothing qualifies, hold cash — same "go to cash in bear" idea
     that makes V7's sector rotation work.

Returned weights are FRACTIONS OF THE MEME BUDGET (sum <= 1). The engine scales
them by meme_max_allocation and applies the per-coin cap.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd

from anti_rug import AntiRugGuard
from config import Config


class MemeMomentumStrategy:
    def __init__(self, config: Config, guard: AntiRugGuard | None = None):
        self.config = config
        self.guard = guard or AntiRugGuard(config)
        self.last_rejections: Dict[str, str] = {}

    def generate_weights(
        self, market_data: Dict[str, pd.DataFrame], date: pd.Timestamp
    ) -> Dict[str, float]:
        mc = self.config.meme
        self.last_rejections = {}
        scores: Dict[str, float] = {}
        for sym, df in market_data.items():
            verdict = self.guard.history_check(sym, df, date)
            if not verdict.allow_entry:
                self.last_rejections[sym] = "; ".join(verdict.reasons)
                continue
            idx = df.index.get_loc(date)
            close = df["Close"].astype(float).iloc[: idx + 1]
            volume = df["Volume"].astype(float).iloc[: idx + 1]

            rets = close.pct_change().iloc[-int(mc.meme_sharpe_window):].dropna()
            sigma = float(rets.std())
            if len(rets) < int(mc.meme_sharpe_window) // 2 or sigma <= 0:
                continue
            sharpe = float(rets.mean()) / sigma
            if sharpe < float(mc.meme_min_entry_sharpe):
                continue
            ma = float(close.iloc[-int(mc.meme_trend_ma_window):].mean())
            if float(close.iloc[-1]) < ma:
                continue
            avg_vol = float(volume.iloc[-int(mc.meme_volume_window):].mean())
            if avg_vol > 0 and float(volume.iloc[-1]) < avg_vol * float(mc.meme_volume_multiplier):
                continue
            scores[sym] = sharpe

        chosen = sorted(scores, key=scores.get, reverse=True)[: int(mc.meme_top_n)]
        return {s: (1.0 / len(chosen) if s in chosen else 0.0) for s in market_data}

    def forced_exits(self, market_data: Dict[str, pd.DataFrame], date: pd.Timestamp) -> Dict[str, str]:
        """Symbols whose history check says a rug/crash is in progress -> sell everything."""
        out: Dict[str, str] = {}
        for sym, df in market_data.items():
            v = self.guard.history_check(sym, df, date)
            if v.force_exit:
                out[sym] = "; ".join(v.reasons)
        return out
