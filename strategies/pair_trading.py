"""
Pair trading strategy:
- Select pairs from active universe by objective quality/correlation criteria
- Enforce same quote currency pairing
- Trade mean reversion using ratio z-score
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Tuple

import pandas as pd

from config import Config
from indicators import rolling_std, zscore
from utils import normalize_weights


class PairTradingStrategy:
    def __init__(self, config: Config):
        self.config = config
        # Per-pair state: (a, b) -> direction where +1 long a/short b, -1 inverse.
        self.active_direction: Dict[Tuple[str, str], int] = {}
        self.last_selected_pairs: List[Tuple[str, str]] = []

    def generate_weights(
        self, market_data: Dict[str, pd.DataFrame], date: pd.Timestamp
    ) -> Dict[str, float]:
        symbols: List[str] = sorted(market_data.keys())
        raw: Dict[str, float] = {s: 0.0 for s in symbols}
        if len(symbols) < 2:
            self.last_selected_pairs = []
            return raw

        selected = self._select_pairs(market_data, date)
        self.last_selected_pairs = list(selected)
        if not selected:
            return raw

        for a, b in selected:
            dfa, dfb = market_data[a], market_data[b]
            idx_a = dfa.index.get_loc(date)
            idx_b = dfb.index.get_loc(date)
            if isinstance(idx_a, slice) or isinstance(idx_b, slice):
                continue
            ratio = dfa["Close"] / dfb["Close"]
            z = zscore(ratio, self.config.strategy.pair_window).iloc[idx_a]
            if pd.isna(z):
                continue

            key = (a, b)
            direction = int(self.active_direction.get(key, 0))
            z_entry = float(
                getattr(self.config.strategy, "pair_entry_zscore", self.config.strategy.pair_z_entry)
            )
            z_exit = float(
                getattr(self.config.strategy, "pair_exit_zscore", self.config.strategy.pair_z_exit)
            )
            if direction == 0:
                if z > z_entry:
                    direction = -1
                elif z < -z_entry:
                    direction = +1
            else:
                if abs(z) < z_exit:
                    direction = 0
            self.active_direction[key] = direction

            if direction > 0:
                raw[a] += 0.5
                raw[b] -= 0.5
            elif direction < 0:
                raw[a] -= 0.5
                raw[b] += 0.5

        return normalize_weights(raw)

    def _select_pairs(
        self, market_data: Dict[str, pd.DataFrame], date: pd.Timestamp
    ) -> List[Tuple[str, str]]:
        lookback = int(self.config.strategy.pair_selection_lookback)
        corr_min = float(self.config.strategy.pair_selection_correlation_min)
        ratio_vol_min = float(self.config.strategy.pair_min_ratio_volatility)
        ratio_vol_max = float(self.config.strategy.pair_selection_ratio_vol_max)
        pair_w = int(self.config.strategy.pair_window)

        candidates: List[Tuple[Tuple[str, str], float]] = []
        for a, b in combinations(sorted(market_data.keys()), 2):
            dfa, dfb = market_data[a], market_data[b]
            if date not in dfa.index or date not in dfb.index:
                continue
            if self._quote(a) != self._quote(b):
                continue
            idx_a = dfa.index.get_loc(date)
            idx_b = dfb.index.get_loc(date)
            if isinstance(idx_a, slice) or isinstance(idx_b, slice):
                continue
            if idx_a < max(lookback, pair_w) or idx_b < max(lookback, pair_w):
                continue

            w_a = dfa.iloc[idx_a - lookback + 1 : idx_a + 1]
            w_b = dfb.iloc[idx_b - lookback + 1 : idx_b + 1]
            if len(w_a) < lookback or len(w_b) < lookback:
                continue
            ret_a = w_a["Close"].pct_change()
            ret_b = w_b["Close"].pct_change()
            corr = ret_a.corr(ret_b)
            if pd.isna(corr) or float(corr) < corr_min:
                continue

            ratio = dfa["Close"] / dfb["Close"]
            ratio_vol = rolling_std(ratio, lookback).iloc[idx_a]
            if pd.isna(ratio_vol):
                continue
            ratio_vol = float(ratio_vol)
            if ratio_vol < ratio_vol_min or ratio_vol > ratio_vol_max:
                continue

            score = float(corr)
            # Favor higher-liquidity, stable-history pairs when requested.
            if self.config.strategy.pair_selection_use_quality_score:
                q_a = self._asset_quality_score(w_a, lookback)
                q_b = self._asset_quality_score(w_b, lookback)
                score *= 0.5 * (q_a + q_b)
            # Penalize very noisy ratio behavior.
            score /= (1.0 + ratio_vol)
            candidates.append(((a, b), score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        max_pairs = max(1, int(self.config.strategy.pair_max_pairs))
        return [pair for pair, _ in candidates[:max_pairs]]

    @staticmethod
    def _quote(symbol: str) -> str:
        s = str(symbol).strip().upper()
        if "(" in s and ")" in s and s.rfind("(") < s.rfind(")"):
            return s[s.rfind("(") + 1 : s.rfind(")")]
        return "UNKNOWN"

    @staticmethod
    def _asset_quality_score(df: pd.DataFrame, lookback: int) -> float:
        if df.empty:
            return 0.0
        history_score = min(1.0, len(df) / float(max(lookback, 1)))
        dollar_vol = (df["Close"].astype(float) * df["Volume"].astype(float)).replace([float("inf"), float("-inf")], pd.NA).dropna()
        if dollar_vol.empty:
            return 0.0
        avg_dv = float(dollar_vol.mean())
        med_dv = float(dollar_vol.median())
        liq_score = min(1.0, med_dv / max(avg_dv, 1e-9))
        return max(0.0, min(1.0, 0.6 * history_score + 0.4 * liq_score))
