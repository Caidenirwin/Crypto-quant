"""
PAPER-ONLY BTC trend strategy - rule (e) from simple_baselines.py, running alongside V7.

    Long BTC while close > 200-day SMA, sized min(1, 40% / 30-day realised vol); trade only when
    the target differs from the current weight by more than 10 percentage points.
    Signal at the daily close (UTC) -> fill at the next day's open; 0.15% cost per side.

No orders are sent anywhere. It simulates fills on real Binance daily candles (public data,
no API key) and keeps its own $500 ledger in logs/paper_trend/ - completely separate from V7.

Run as often as you like (the scheduled task runs it hourly): it only acts on completed daily
candles it hasn't processed yet, so it also catches up after the PC was off.

    python paper_trend.py                      process any new completed days
    python paper_trend.py --replay 2024-01-18 2026-09-30   parity check vs the backtest
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from simple_baselines import COST_PER_SIDE, REBALANCE_BAND, SMA, VOL_LOOKBACK, target_weight_d

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs" / "paper_trend"
STARTING_CAPITAL = 500.0
MIRROR = "https://data-api.binance.vision/api/v3/klines"
HISTORY_DAYS = SMA + VOL_LOOKBACK + 60          # enough closes for the 200-day SMA and 30-day vol


# --------------------------------------------------------------------------- #
# Data                                                                        #
# --------------------------------------------------------------------------- #
def completed_daily_candles() -> pd.DataFrame:
    r = requests.get(MIRROR, params={"symbol": "BTCUSDT", "interval": "1d", "limit": HISTORY_DAYS}, timeout=20)
    r.raise_for_status()
    now_ms = datetime.now(timezone.utc).timestamp() * 1000
    rows = [k for k in r.json() if k[6] < now_ms]                 # closed candles only
    df = pd.DataFrame({
        "Date": pd.to_datetime([k[0] for k in rows], unit="ms").normalize(),
        "Open": [float(k[1]) for k in rows], "Close": [float(k[4]) for k in rows],
    })
    return df.set_index("Date")


# --------------------------------------------------------------------------- #
# Ledger                                                                      #
# --------------------------------------------------------------------------- #
def _paths(log_dir: Path):
    return log_dir / "state.json", log_dir / "equity.csv", log_dir / "trades.csv", log_dir / "run.log"


def _append(path: Path, row: dict) -> None:
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def _save(path: Path, state: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(path)


def process(candles: pd.DataFrame, log_dir: Path = LOG_DIR, capital: float = STARTING_CAPITAL) -> int:
    """Apply rule (e) to every completed day not yet processed. Returns how many days were processed."""
    log_dir.mkdir(parents=True, exist_ok=True)
    state_p, equity_p, trades_p, _ = _paths(log_dir)
    targets = target_weight_d(candles["Close"])                  # value at d uses closes up to d only

    if not state_p.exists():                                     # first run: start flat at the latest close
        d = candles.index[-1]
        state = {"cash": capital, "units": 0.0, "start_date": str(d.date()),
                 "last_processed": str(d.date()), "pending_target": float(targets.loc[d])}
        _append(equity_p, {"date": str(d.date()), "open": candles.at[d, "Open"], "close": candles.at[d, "Close"],
                           "traded": False, "weight": 0.0, "next_target": round(state["pending_target"], 4),
                           "units": 0.0, "cash": round(capital, 6), "equity": round(capital, 6)})
        _save(state_p, state)
        return 1

    state = json.loads(state_p.read_text(encoding="utf-8"))
    new_days = candles.index[candles.index > pd.Timestamp(state["last_processed"])]
    for d in new_days:
        o, c = float(candles.at[d, "Open"]), float(candles.at[d, "Close"])
        cash, units, tgt = state["cash"], state["units"], state["pending_target"]
        traded = False
        value = cash + units * o
        current = units * o / value if value > 0 else 0.0
        if abs(tgt - current) > REBALANCE_BAND:                  # 10-point band, fill at today's open
            want = tgt * value / o
            delta = want - units
            fee = abs(delta * o) * COST_PER_SIDE
            cash -= delta * o + fee
            _append(trades_p, {"date": str(d.date()), "side": "buy" if delta > 0 else "sell",
                               "btc": round(delta, 8), "price": o, "notional": round(abs(delta * o), 6),
                               "fee": round(fee, 6), "from_weight": round(current, 4), "to_weight": round(tgt, 4)})
            units, traded = want, True
        equity = cash + units * c
        next_tgt = float(targets.loc[d])
        state.update({"cash": cash, "units": units, "last_processed": str(d.date()), "pending_target": next_tgt})
        _append(equity_p, {"date": str(d.date()), "open": o, "close": c, "traded": traded,
                           "weight": round(units * c / equity, 4) if equity > 0 else 0.0,
                           "next_target": round(next_tgt, 4), "units": round(units, 8),
                           "cash": round(cash, 6), "equity": round(equity, 6)})
        _save(state_p, state)                                    # persist after every day (crash-safe)
    return len(new_days)


# --------------------------------------------------------------------------- #
def run_live() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = _paths(LOG_DIR)[3]
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    try:
        n = process(completed_daily_candles())
        state = json.loads(_paths(LOG_DIR)[0].read_text(encoding="utf-8"))
        msg = (f"{stamp} processed {n} day(s); last close {state['last_processed']}; "
               f"next target weight {state['pending_target']:.3f}")
    except Exception as exc:                                     # never crash noisily; next run retries
        msg = f"{stamp} ERROR {type(exc).__name__}: {exc}"
    with log.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg)


def replay(start: str, end: str) -> None:
    """Run the live ledger code over archive history and compare with simple_baselines.simulate."""
    from simple_baselines import simulate
    from universe_data import load_all

    btc = load_all()["BTC(USD)"]
    btc = btc[btc.index <= end]
    tmp = Path(tempfile.mkdtemp(prefix="paper_trend_replay_"))
    first = btc.index[btc.index >= start][0]
    process(btc[btc.index <= first], tmp, capital=1.0)            # initialise exactly like the first live run
    process(btc, tmp, capital=1.0)
    live_eq = pd.read_csv(tmp / "equity.csv", parse_dates=["date"]).set_index("date")["equity"]
    w = pd.DataFrame({"BTC": target_weight_d(btc["Close"])})
    bt = simulate({"BTC": btc}, w, start, end, "band")
    diff = (live_eq - bt["equity"].reindex(live_eq.index)).abs().max()
    trades = len(pd.read_csv(tmp / "trades.csv")) if (tmp / "trades.csv").exists() else 0
    print(f"replay {start}..{end}: paper ledger final x{live_eq.iloc[-1]:.6f}, trades {trades} | "
          f"backtest final x{bt['final_multiple']:.6f}, trades {bt['trades']} | max equity diff {diff:.2e}")


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--replay":
        replay(sys.argv[2], sys.argv[3])
    else:
        run_live()
