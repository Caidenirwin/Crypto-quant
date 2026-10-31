"""
Risk management module:
- Position sizing limits
- Stop-loss checks
- Max drawdown protection
"""

from __future__ import annotations

from typing import Dict

import pandas as pd

from config import Config
from indicators import rolling_volatility
from utils import clamp, normalize_weights


class RiskManager:
    def __init__(self, config: Config):
        self.config = config
        self.trading_enabled = True

    def apply_position_limits(self, weights: Dict[str, float]) -> Dict[str, float]:
        max_w = self.config.risk.max_position_weight
        return {k: clamp(v, -max_w, max_w) for k, v in weights.items()}

    def apply_asset_caps(self, weights: Dict[str, float]) -> Dict[str, float]:
        capped = dict(weights)
        for symbol, cap in self.config.risk.high_risk_asset_caps.items():
            if symbol in capped:
                capped[symbol] = clamp(capped[symbol], -abs(cap), abs(cap))
        return capped

    def apply_asset_stability_penalties(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Softly down-weight assets that show unstable performance across robustness tests.
        """
        if not self.config.risk.asset_stability_penalty_enabled:
            return dict(weights)
        penalties = self.config.risk.asset_stability_penalties or {}
        if not penalties:
            return dict(weights)

        out: Dict[str, float] = dict(weights)
        min_penalty = float(self.config.backtest.asset_penalty_min)
        for symbol, weight in out.items():
            penalty = float(penalties.get(symbol, 1.0))
            penalty = clamp(penalty, min_penalty, 1.0)
            out[symbol] = float(weight) * penalty
        return normalize_weights(out)

    @staticmethod
    def apply_group_cap(weights: Dict[str, float], symbols: set[str], cap: float) -> Dict[str, float]:
        """
        Cap aggregate long exposure for a symbol group.
        Excess is reallocated proportionally to other positive-weight symbols.
        """
        out = dict(weights)
        cap = float(max(0.0, cap))
        if cap <= 0:
            for s in symbols:
                if s in out and out[s] > 0:
                    out[s] = 0.0
            return out

        group_pos = sum(max(0.0, out.get(s, 0.0)) for s in symbols)
        if group_pos <= cap + 1e-12:
            return out

        excess = group_pos - cap
        for s in symbols:
            w = max(0.0, out.get(s, 0.0))
            if group_pos > 0:
                out[s] = max(0.0, w - excess * (w / group_pos))

        others = [s for s, w in out.items() if s not in symbols and w > 0]
        other_pos = sum(out[s] for s in others)
        if other_pos > 1e-12:
            for s in others:
                out[s] += excess * (out[s] / other_pos)
        return out

    def apply_inverse_vol_sizing(
        self, weights: Dict[str, float], market_data: Dict[str, pd.DataFrame], date: pd.Timestamp
    ) -> Dict[str, float]:
        if not self.config.risk.volatility_sizing_enabled:
            return dict(weights)

        out: Dict[str, float] = {}
        window = self.config.risk.volatility_lookback
        vol_floor = self.config.risk.volatility_floor
        scale_max = self.config.risk.volatility_scale_max

        for symbol, w in weights.items():
            if abs(w) <= 1e-12:
                out[symbol] = 0.0
                continue
            df = market_data.get(symbol)
            if df is None or date not in df.index:
                out[symbol] = w
                continue
            idx = df.index.get_loc(date)
            if isinstance(idx, slice) or idx < window:
                out[symbol] = 0.0
                continue

            vol = rolling_volatility(df["Close"], window).iloc[idx]
            if pd.isna(vol):
                out[symbol] = 0.0
                continue
            inv_scale = 1.0 / max(float(vol), float(vol_floor))
            inv_scale = clamp(inv_scale, 0.0, float(scale_max))
            out[symbol] = float(w) * float(inv_scale)
        out = normalize_weights(out)

        # Keep active positions practical: drop tiny weights and cap oversized dominance.
        min_w = self.config.risk.active_weight_min
        max_w = self.config.risk.active_weight_max
        bounded: Dict[str, float] = {}
        for symbol, w in out.items():
            sign = 1.0 if w >= 0 else -1.0
            abs_w = abs(float(w))
            if abs_w < min_w:
                bounded[symbol] = 0.0
                continue
            bounded[symbol] = sign * min(abs_w, max_w)
        return normalize_weights(bounded)

    def stop_loss_triggered(self, entry_price: float, current_price: float, quantity: float = 1.0) -> bool:
        if entry_price <= 0:
            return False
        if quantity >= 0:
            pnl = (current_price / entry_price) - 1.0
        else:
            pnl = (entry_price / current_price) - 1.0
        return pnl <= -self.config.risk.stop_loss_pct

    def enforce_drawdown_guard(self, current_drawdown: float) -> None:
        if current_drawdown <= -self.config.risk.max_drawdown_pct:
            self.trading_enabled = False

    def maybe_resume(self, current_drawdown: float) -> None:
        threshold = -0.5 * self.config.risk.max_drawdown_pct
        if current_drawdown > threshold:
            self.trading_enabled = True
