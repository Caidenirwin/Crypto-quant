"""
Utility helpers used across modules.
"""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


def ensure_datetime_index(df: pd.DataFrame, date_column: str = "Date") -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()
    out = df.copy()
    out[date_column] = pd.to_datetime(out[date_column], utc=False, format="mixed", errors="coerce")
    out = out.dropna(subset=[date_column])
    out = out.set_index(date_column).sort_index()
    return out


def normalize_weights(raw_weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(abs(v) for v in raw_weights.values())
    if total <= 0:
        return {k: 0.0 for k in raw_weights}
    return {k: v / total for k, v in raw_weights.items()}


def top_n_by_value(values: Dict[str, float], n: int) -> List[str]:
    return [k for k, _ in sorted(values.items(), key=lambda item: item[1], reverse=True)[:n]]


def clamp(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


def annualize_return(total_return: float, periods: int, annualization_days: int = 365) -> float:
    if periods <= 1:
        return 0.0
    years = periods / float(annualization_days)
    if years <= 0:
        return 0.0
    return (1.0 + total_return) ** (1.0 / years) - 1.0


def sharpe_ratio(daily_returns: Iterable[float], annualization_days: int = 365) -> float:
    series = pd.Series(list(daily_returns)).dropna()
    if series.empty:
        return 0.0
    mean = series.mean()
    std = series.std(ddof=0)
    if std <= 0:
        return 0.0
    return float((mean / std) * np.sqrt(annualization_days))


def max_drawdown(equity_curve: Iterable[float]) -> float:
    curve = pd.Series(list(equity_curve), dtype=float)
    if curve.empty:
        return 0.0
    running_max = curve.cummax()
    drawdown = curve / running_max - 1.0
    return float(drawdown.min())
