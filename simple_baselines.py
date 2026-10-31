"""
Simple honest baselines (backtest only - nothing here trades).  Run: python simple_baselines.py

Same honest setup as pit_backtest.py: Binance archive data (universe_data.py), 0.3% round-trip
costs (0.15% of traded notional per side), signal at close t -> fill at open t+1, mark at close.
Fixed rules, nothing tuned:
  (a) BTC buy & hold
  (b) BTC long while close > 200-day SMA, else cash
  (c) (b) applied to BTC and ETH separately, 50% of capital each (each half in cash when its own
      signal is off; holdings only re-sized when a signal flips - no extra rebalance schedule)
  (d) (b) with volatility targeting: weight = min(1, 0.40 / annualised 30-day realised vol)
      (30 = the project's existing risk.volatility_lookback, not a new parameter); re-sized daily.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from universe_data import load_all

ROOT = Path(__file__).resolve().parent
COST_PER_SIDE = 0.003 / 2
SMA = 200
VOL_TARGET = 0.40
VOL_LOOKBACK = 30            # = Config().risk.volatility_lookback
END = "2026-09-30"
PERIODS = {"2020-26": "2020-01-01", "2024-26": "2024-01-18"}


REBALANCE_BAND = 0.10        # (e): trade only if |target - current weight| > 10 percentage points


def target_weight_d(close: pd.Series) -> pd.Series:
    """Rule (d): long while close > 200-day SMA, sized min(1, 40% / 30-day realised vol). Known at the close."""
    above = close > close.rolling(SMA).mean()
    vol = np.log(close).diff().rolling(VOL_LOOKBACK).std() * math.sqrt(365)
    return (above.astype(float) * (VOL_TARGET / vol).clip(upper=1.0)).fillna(0.0)


def simulate(prices: Dict[str, pd.DataFrame], weights: pd.DataFrame, start: str, end: str,
             resize: str, band: float = REBALANCE_BAND) -> dict:
    """
    weights: target weight per asset decided at each CLOSE (index = signal date).
    resize = "on_change": trade only when an asset's target changes (else let it drift)
             "daily":     re-size to target every day.
             "band":      re-size only when |target - current weight| > band (current weight
                          = actual drifted weight at the open).
    """
    idx = weights.index[(weights.index >= start) & (weights.index <= end)]
    cash, units = 1.0, {a: 0.0 for a in weights}
    eq, exposure, trades, last_tgt = [], [], 0, {a: 0.0 for a in weights}
    off_but_held = 0                                               # band edge case: signal says cash, residual held
    for i in range(len(idx)):
        d = idx[i]
        if i > 0:                                                  # fill yesterday's signal at today's open
            sig = weights.loc[idx[i - 1]]
            opens = {a: float(prices[a].at[d, "Open"]) for a in weights}
            value = cash + sum(units[a] * opens[a] for a in weights)
            for a in weights:
                tgt = float(sig[a])
                if resize == "on_change" and abs(tgt - last_tgt[a]) < 1e-12:
                    continue
                if resize == "band":
                    current = units[a] * opens[a] / value if value > 0 else 0.0
                    if abs(tgt - current) <= band:
                        if tgt == 0.0 and units[a] > 0:
                            off_but_held += 1
                        continue
                want_units = tgt * value / opens[a]
                delta = want_units - units[a]
                if abs(delta * opens[a]) < 1e-9:
                    last_tgt[a] = tgt
                    continue
                cash -= delta * opens[a] + abs(delta * opens[a]) * COST_PER_SIDE
                units[a] = want_units
                last_tgt[a] = tgt
                trades += 1
        closes = {a: float(prices[a].at[d, "Close"]) for a in weights}
        held = sum(units[a] * closes[a] for a in weights)
        eq.append(cash + held)
        exposure.append(held / (cash + held) if cash + held > 0 else 0.0)
    eq = pd.Series(eq, index=idx)
    r = eq.pct_change().dropna()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    return {
        "start": str(eq.index[0].date()), "end": str(eq.index[-1].date()),
        "cagr": float(eq.iloc[-1] ** (1 / years) - 1),
        "sharpe": float(r.mean() / r.std() * math.sqrt(365)) if r.std() > 0 else 0.0,
        "max_dd": float((eq / eq.cummax() - 1).min()),
        "time_in_market": float((pd.Series(exposure, index=idx) > 1e-6).mean()),
        "avg_exposure": float(np.mean(exposure)),
        "trades": trades,
        "final_multiple": float(eq.iloc[-1]),
        "days_signal_off_but_still_holding": off_but_held,
        "equity": eq,
    }


def main() -> None:
    data = load_all()
    px = {a: data[f"{a}(USD)"][lambda df: df.index <= END] for a in ("BTC", "ETH")}
    dates = px["BTC"].index.intersection(px["ETH"].index)
    px = {a: df.loc[dates] for a, df in px.items()}

    close = pd.DataFrame({a: df["Close"] for a, df in px.items()})
    above = close > close.rolling(SMA).mean()                       # known at the close
    w_d = pd.DataFrame({"BTC": target_weight_d(close["BTC"])})

    rules = {
        "(a) BTC buy & hold": (pd.DataFrame({"BTC": 1.0}, index=dates), "on_change"),
        "(b) BTC > 200-day SMA, else cash": (pd.DataFrame({"BTC": above["BTC"].astype(float)}), "on_change"),
        "(c) 50/50 BTC/ETH, each > own 200-day SMA": (above.astype(float) * 0.5, "on_change"),
        "(d) (b) + 40% vol target, max 1x": (w_d, "daily"),
        "(e) (d) + 10-pt rebalance band": (w_d, "band"),
    }
    rows = []
    for period, start in PERIODS.items():
        for name, (w, resize) in rules.items():
            res = simulate({a: px[a] for a in w}, w, start, END, resize)
            res.pop("equity")
            rows.append({"period": period, "rule": name, **res})
            print(f"{period}  {name:44s} CAGR {res['cagr']:+7.1%}  Sharpe {res['sharpe']:5.2f}  "
                  f"MaxDD {res['max_dd']:6.1%}  in market {res['time_in_market']:5.1%} "
                  f"(avg exposure {res['avg_exposure']:5.1%})  trades {res['trades']:4d}"
                  + (f"  signal-off-but-held days {res['days_signal_off_but_still_holding']}" if resize == "band" else ""),
                  flush=True)
    out = ROOT / "outputs" / "simple_baselines.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
