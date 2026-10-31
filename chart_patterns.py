"""
Chart pattern scanner -> ML features.

Reads each coin's daily candles the way a chart trader would and turns what it
sees into numbers the model can learn from. Every value at day t uses ONLY candles
up to and including day t. Swing highs/lows need PIVOT_K bars on each side to be
confirmed, so a swing at day i is only "seen" from day i + PIVOT_K onward
(using it earlier would leak the future into the backtest).

Patterns scanned
  Candlesticks   body/wick shape, bullish/bearish engulfing, hammer, shooting
                 star, doji, inside bar, outside bar
  Breakouts      20- and 55-day breakout / breakdown (Donchian channels)
  Squeeze        Bollinger bandwidth percentile, NR7 (narrowest range in 7 days),
                 squeeze followed by a breakout
  Structure      higher-highs/higher-lows vs lower-highs/lower-lows from swings
  Support/resist distance to last swing high/low (in ATRs), how many past swings
                 cluster near today's price (resistance above, support below)
  Reversals      double bottom / double top (two swings within 1 ATR, neckline broken)
  Trendlines     regression slope (in ATRs/day) and straightness (R^2), 20 & 60 days
  Range          position inside the 1-year high-low range
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

PIVOT_K = 3

PATTERN_FEATURES: List[str] = [
    "cdl_body", "cdl_upper_wick", "cdl_lower_wick", "cdl_bull_engulf", "cdl_bear_engulf",
    "cdl_hammer", "cdl_shooting_star", "cdl_doji", "cdl_inside", "cdl_outside",
    "brk_up_20", "brk_dn_20", "brk_up_55", "brk_dn_55",
    "bb_width_pct", "nr7", "squeeze_breakout",
    "structure_score", "dist_swing_high_atr", "dist_swing_low_atr",
    "resistance_touches", "support_touches",
    "double_bottom", "double_top",
    "trend_slope_20", "trend_r2_20", "trend_slope_60", "trend_r2_60",
    "range_pos_365",
]

# Binary patterns shown in the research report's pattern scorecard.
BINARY_PATTERNS: List[str] = [
    "cdl_bull_engulf", "cdl_bear_engulf", "cdl_hammer", "cdl_shooting_star", "cdl_doji",
    "cdl_inside", "cdl_outside", "brk_up_20", "brk_dn_20", "brk_up_55", "brk_dn_55",
    "nr7", "squeeze_breakout", "double_bottom", "double_top",
]


def _confirmed_pivots(series: pd.Series, k: int, kind: str) -> pd.Series:
    """Pivot value placed on the day it becomes CONFIRMED (k bars later), NaN elsewhere."""
    win = series.rolling(2 * k + 1, center=True)
    extreme = win.max() if kind == "high" else win.min()
    is_pivot = series.eq(extreme)
    return series.where(is_pivot).shift(k)


def _last_two(confirmed: pd.Series):
    vals = confirmed.dropna()
    last = vals.reindex(confirmed.index).ffill()
    prev = vals.shift(1).reindex(confirmed.index).ffill()
    return last, prev


def _rolling_trend(close: pd.Series, n: int):
    """OLS slope and R^2 of log price over the last n bars."""
    y = np.log(close)
    x = np.arange(n, dtype=float)
    xm, xv = x.mean(), ((x - x.mean()) ** 2).sum()

    def slope(w):
        return ((x - xm) * (w - w.mean())).sum() / xv

    def r2(w):
        b = ((x - xm) * (w - w.mean())).sum() / xv
        fit = w.mean() + b * (x - xm)
        ss = ((w - w.mean()) ** 2).sum()
        return 1 - ((w - fit) ** 2).sum() / ss if ss > 0 else 0.0

    roll = y.rolling(n)
    return roll.apply(slope, raw=True), roll.apply(r2, raw=True)


def _touch_count(level_vals: pd.Series, close: pd.Series, atr: pd.Series, above: bool, lookback: int = 120):
    """How many confirmed swings in the last `lookback` days sit within 1 ATR of today's price."""
    vals = level_vals.to_numpy()
    c, a = close.to_numpy(), atr.to_numpy()
    out = np.zeros(len(c))
    idx = np.where(~np.isnan(vals))[0]
    for t in range(len(c)):
        if np.isnan(a[t]):
            out[t] = np.nan
            continue
        lo = np.searchsorted(idx, t - lookback)
        hi = np.searchsorted(idx, t, side="right")
        lv = vals[idx[lo:hi]]
        near = np.abs(lv - c[t]) <= a[t]
        side = lv >= c[t] if above else lv <= c[t]
        out[t] = float((near & side).sum())
    return pd.Series(out, index=close.index)


def pattern_features(df: pd.DataFrame, atr: pd.Series) -> pd.DataFrame:
    o, h, l, c = (df[k].astype(float) for k in ("Open", "High", "Low", "Close"))
    rng = (h - l).replace(0, np.nan)
    body = c - o
    f = pd.DataFrame(index=df.index)

    # --- Candlesticks -------------------------------------------------------
    f["cdl_body"] = body / rng
    f["cdl_upper_wick"] = (h - np.maximum(o, c)) / rng
    f["cdl_lower_wick"] = (np.minimum(o, c) - l) / rng
    po, pc = o.shift(1), c.shift(1)
    f["cdl_bull_engulf"] = ((pc < po) & (c > o) & (c >= po) & (o <= pc)).astype(float)
    f["cdl_bear_engulf"] = ((pc > po) & (c < o) & (c <= po) & (o >= pc)).astype(float)
    down_before = c.shift(1) < c.shift(6)
    up_before = c.shift(1) > c.shift(6)
    f["cdl_hammer"] = ((f["cdl_lower_wick"] >= 0.6) & (f["cdl_upper_wick"] <= 0.15) & down_before).astype(float)
    f["cdl_shooting_star"] = ((f["cdl_upper_wick"] >= 0.6) & (f["cdl_lower_wick"] <= 0.15) & up_before).astype(float)
    f["cdl_doji"] = (body.abs() / rng <= 0.1).astype(float)
    f["cdl_inside"] = ((h < h.shift(1)) & (l > l.shift(1))).astype(float)
    f["cdl_outside"] = ((h > h.shift(1)) & (l < l.shift(1))).astype(float)

    # --- Breakouts (prior channel, excluding today) --------------------------
    for n in (20, 55):
        f[f"brk_up_{n}"] = (c > h.shift(1).rolling(n).max()).astype(float)
        f[f"brk_dn_{n}"] = (c < l.shift(1).rolling(n).min()).astype(float)

    # --- Volatility squeeze -------------------------------------------------
    ma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    bbw = (4 * sd20) / ma20
    f["bb_width_pct"] = bbw.rolling(120).rank(pct=True)
    f["nr7"] = ((h - l) <= (h - l).rolling(7).min()).astype(float)
    squeezed_recently = (f["bb_width_pct"] <= 0.2).astype(float).rolling(5).max() == 1
    f["squeeze_breakout"] = (squeezed_recently & (f["brk_up_20"] == 1)).astype(float)

    # --- Swing structure (confirmed pivots only) -----------------------------
    sh = _confirmed_pivots(h, PIVOT_K, "high")
    sl = _confirmed_pivots(l, PIVOT_K, "low")
    sh_last, sh_prev = _last_two(sh)
    sl_last, sl_prev = _last_two(sl)
    hh = np.sign(sh_last - sh_prev)
    hl = np.sign(sl_last - sl_prev)
    f["structure_score"] = (hh + hl) / 2.0                     # +1 uptrend structure, -1 downtrend
    f["dist_swing_high_atr"] = (c - sh_last) / atr
    f["dist_swing_low_atr"] = (c - sl_last) / atr
    f["resistance_touches"] = _touch_count(sh, c, atr, above=True)
    f["support_touches"] = _touch_count(sl, c, atr, above=False)

    # --- Double bottom / top: two swings within 1 ATR + neckline broken -----
    f["double_bottom"] = (((sl_last - sl_prev).abs() <= atr) & (c > sh_last) & (sh_last > sl_last)).astype(float)
    f["double_top"] = (((sh_last - sh_prev).abs() <= atr) & (c < sl_last) & (sl_last < sh_last)).astype(float)

    # --- Trendlines -----------------------------------------------------------
    atr_pct = atr / c
    for n in (20, 60):
        slope, r2 = _rolling_trend(c, n)
        f[f"trend_slope_{n}"] = slope / atr_pct                # % per day, in ATR units
        f[f"trend_r2_{n}"] = r2

    lo365, hi365 = l.rolling(365, min_periods=200).min(), h.rolling(365, min_periods=200).max()
    f["range_pos_365"] = (c - lo365) / (hi365 - lo365).replace(0, np.nan)
    return f
