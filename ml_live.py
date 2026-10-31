"""
ML bracket paper trader - its own $500 virtual account on Binance Spot Testnet,
separate from V7 (own ledger in logs/ml/, never touches V7's paper account).

Called every hour from live_runner.py (failures never affect V7). Each cycle:
  EXITS  (every hour) for each open trade, walk the 1-hour candles since the last
         check (Binance public data): stop-loss first, then take-profit, then the
         14-day time limit. Severe news (hack / exploit / delisting) or an
         anti-rug crash also force an exit.
  ENTRIES (once per new daily candle) only if ml_autoresearch.py says the edge
         is healthy: score every coin with the champion model, keep EV >= 0.2R,
         drop coins with bad news or failing anti-rug checks, buy the best few.
         Risk 1% of the account per trade, stop 1.5 x ATR, target 2R.

Run one cycle by hand: python ml_live.py
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

from config import Config

ROOT = Path(__file__).resolve().parent
ML_LOGS = ROOT / "logs" / "ml"
STATE_PATH = ML_LOGS / "state.json"
TRADES_PATH = ML_LOGS / "ml_trades.csv"
EQUITY_PATH = ML_LOGS / "ml_equity.csv"
SIGNALS_PATH = ML_LOGS / "ml_signals.csv"
STATUS_PATH = ROOT / "models" / "ml_status.json"
MIRROR = "https://data-api.binance.vision/api/v3"
STARTING_CAPITAL = 500.0
MAX_MODEL_AGE_DAYS = 10


# --------------------------------------------------------------------------- #
# Ledger                                                                      #
# --------------------------------------------------------------------------- #
def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"cash": STARTING_CAPITAL, "positions": {}, "last_signal_date": None, "realized_pnl": 0.0}


def save_state(state: dict) -> None:
    ML_LOGS.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(STATE_PATH)


def _append(path: Path, row: dict) -> None:
    ML_LOGS.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def _now() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# Market data (Binance public mirror = real prices)                           #
# --------------------------------------------------------------------------- #
def _pair(sym: str) -> str:
    return sym.replace("/", "")


def mirror_price(sym: str) -> float:
    r = requests.get(f"{MIRROR}/ticker/price", params={"symbol": _pair(sym)}, timeout=15)
    r.raise_for_status()
    return float(r.json()["price"])


def hourly_bars_since(sym: str, since: datetime) -> List[list]:
    r = requests.get(
        f"{MIRROR}/klines",
        params={"symbol": _pair(sym), "interval": "1h", "startTime": int(since.timestamp() * 1000), "limit": 1000},
        timeout=15,
    )
    r.raise_for_status()
    now_ms = _now().timestamp() * 1000
    return [k for k in r.json() if k[6] < now_ms]   # closed bars only


# --------------------------------------------------------------------------- #
# Orders                                                                      #
# --------------------------------------------------------------------------- #
def _executor():
    from execution.binance_executor import BinanceSpotTestnetExecutor

    ex = BinanceSpotTestnetExecutor(dry_run=False)
    exchange = ex.connect_to_exchange()
    if "testnet" not in str(getattr(exchange, "urls", "")).lower():
        raise RuntimeError("ML sleeve safety: exchange is not Testnet")
    ex._ensure_markets_loaded()
    return ex


def _fill(order: dict, fallback_px: float) -> dict:
    from live_runner import _extract_fill_metrics

    return _extract_fill_metrics(order, fallback_price=fallback_px)


def _close(ex, state: dict, sym: str, reason: str, ref_px: float) -> Optional[dict]:
    pos = state["positions"][sym]
    try:
        order = ex.place_market_sell(sym, float(pos["qty"]))
    except Exception as exc:
        print(f"[ml] exit {sym} ({reason}) failed: {exc}")
        return None
    f = _fill(order, ref_px)
    if f["filled_qty"] <= 0:
        print(f"[ml] exit {sym} ({reason}) not filled (testnet liquidity); will retry next cycle")
        return None
    qty = min(f["filled_qty"], float(pos["qty"]))
    px = f["avg_price"]
    proceeds = qty * px - f["fee"]
    pnl = proceeds - qty * float(pos["entry"])
    r_mult = (px - float(pos["entry"])) / float(pos["risk"])
    state["cash"] += proceeds
    state["realized_pnl"] = float(state.get("realized_pnl", 0.0)) + pnl
    remaining = float(pos["qty"]) - qty
    if remaining * px < 1.0:
        del state["positions"][sym]
    else:
        pos["qty"] = remaining
    row = {"timestamp": _now().isoformat(), "symbol": sym, "side": "sell", "qty": qty, "price": px,
           "notional": qty * px, "reason": reason, "r_multiple": round(r_mult, 3), "pnl": round(pnl, 4),
           "cash_after": round(state["cash"], 4)}
    _append(TRADES_PATH, row)
    print(f"[ml] EXIT {sym} {reason}: {qty:.6g} @ {px:.6g}  R={r_mult:+.2f}  pnl=${pnl:+.2f}")
    return row


# --------------------------------------------------------------------------- #
# Cycle                                                                       #
# --------------------------------------------------------------------------- #
def manage_exits(ex, state: dict, cfg: Config, news: Optional[dict]) -> int:
    from anti_rug import AntiRugGuard

    guard = AntiRugGuard(cfg)
    coin_risk = (news or {}).get("coin_risk", {})
    closed = 0
    for sym in list(state["positions"]):
        pos = state["positions"][sym]
        since = datetime.fromisoformat(pos.get("last_check") or pos["entry_time"])
        reason, ref_px = None, None
        try:
            bars = hourly_bars_since(sym, since)
        except Exception as exc:
            print(f"[ml] {sym}: price check failed ({exc}); keeping position")
            continue
        for k in bars:                                # stop-loss checked before take-profit
            lo, hi = float(k[3]), float(k[2])
            if lo <= pos["sl"]:
                reason, ref_px = "stop_loss", pos["sl"]
                break
            if hi >= pos["tp"]:
                reason, ref_px = "take_profit", pos["tp"]
                break
        if bars:
            pos["last_check"] = datetime.fromtimestamp(bars[-1][6] / 1000, tz=timezone.utc).isoformat()
        base = sym.split("/")[0]
        if reason is None and coin_risk.get(base, {}).get("level") == "severe":
            reason = "news: " + coin_risk[base]["headlines"][0][:80]
        if reason is None and _now() >= datetime.fromisoformat(pos["expiry"]):
            reason = "time_limit"
        if reason is None:
            v = guard.market_check(f"{base}(USD)")
            if v.force_exit:
                reason = "anti_rug: " + "; ".join(v.reasons)
        if reason:
            px = ref_px or mirror_price(sym)
            closed += _close(ex, state, sym, reason, px) is not None
    return closed


def generate_entries(ex, state: dict, cfg: Config, news: Optional[dict]) -> int:
    import joblib

    from anti_rug import AntiRugGuard
    from ml_strategy import build_dataset, expected_r, load_panel, load_reference

    mc = cfg.ml
    status = json.loads(STATUS_PATH.read_text(encoding="utf-8")) if STATUS_PATH.exists() else {}
    model_path = ROOT / mc.model_path
    if not model_path.exists():
        print("[ml] no trained model yet (run ml_autoresearch.py) - no entries")
        return 0
    bundle = joblib.load(model_path)
    if _now() - pd.Timestamp(bundle["trained_at"]).to_pydatetime() > timedelta(days=MAX_MODEL_AGE_DAYS):
        print("[ml] model is stale (>10 days) - no entries until research retrains it")
        return 0

    panel = load_panel(cfg)
    latest = max(df.index.max() for df in panel.values())
    if state.get("last_signal_date") and pd.Timestamp(state["last_signal_date"]) >= latest:
        return 0                                       # already traded this daily candle (cheap check first)
    ds = build_dataset(panel, load_reference(cfg), cfg, with_labels=False)
    latest = ds["date"].max()
    if latest < pd.Timestamp(_now().date()) - pd.Timedelta(days=2):
        print(f"[ml] daily data stale (latest {latest.date()}) - no entries")
        return 0
    state["last_signal_date"] = str(latest.date())

    # Each coin's own newest candle: Yahoo coins publish ~1 day after Binance ones, so
    # requiring date == latest would silently drop most of the universe.
    newest = ds[ds["tradable"]].sort_values("date").groupby("symbol").tail(1)
    today = newest[newest["date"] >= latest - pd.Timedelta(days=1)].dropna(subset=bundle["features"] + ["atr"]).copy()
    today["p"] = bundle["model"].predict_proba(today[bundle["features"]])[:, 1]
    today["ev"] = expected_r(today["p"].to_numpy(), mc.reward_risk)
    today = today.sort_values("ev", ascending=False)

    allowed = bool(status.get("entries_allowed", False))
    coin_risk = (news or {}).get("coin_risk", {})
    guard = AntiRugGuard(cfg)
    equity = portfolio_equity(state)
    slots = mc.max_open_trades - len(state["positions"])
    opened = 0
    for _, row in today.iterrows():
        sym = f"{row['symbol'].split('(')[0]}/USDT"
        decision, why = "skip", ""
        if row["ev"] < mc.min_expected_r:
            why = f"EV {row['ev']:+.2f}R below gate"
        elif not allowed:
            why = "PAUSED by research health gate: " + str(status.get("health_reason", "no research yet"))
        elif sym in state["positions"]:
            why = "already holding"
        elif slots - opened <= 0:
            why = "no free slots"
        elif sym not in ex.exchange.markets:
            why = "not on testnet"
        elif row["symbol"].split("(")[0] in coin_risk:
            info = coin_risk[row["symbol"].split("(")[0]]
            why = f"news {info['level']}: {info['headlines'][0][:70]}"
        else:
            hist = guard.history_check(row["symbol"], panel[row["symbol"]], row["date"])
            mkt = guard.market_check(row["symbol"], order_usd=equity * mc.max_position_weight)
            if not (hist.allow_entry and mkt.allow_entry):
                why = "anti-rug: " + "; ".join(hist.reasons + mkt.reasons)
            else:
                ok = _open(ex, state, cfg, sym, float(row["atr"]), equity, float(row["p"]))
                decision, why = ("BUY", "") if ok else ("skip", "order not filled / too small")
                opened += ok
        if row["ev"] >= mc.min_expected_r or decision == "BUY":
            _append(SIGNALS_PATH, {"signal_date": str(latest.date()), "symbol": sym, "p": round(row["p"], 4),
                                   "ev_r": round(row["ev"], 3), "decision": decision, "reason": why})
    print(f"[ml] signals {latest.date()}: {int((today['ev'] >= mc.min_expected_r).sum())} above EV gate, "
          f"opened {opened}, entries {'ALLOWED' if allowed else 'PAUSED'} (champion={bundle['variant']})")
    return opened


def _open(ex, state: dict, cfg: Config, sym: str, atr: float, equity: float, p: float) -> bool:
    mc = cfg.ml
    px = mirror_price(sym)
    risk = mc.stop_atr_mult * atr
    if not (risk > 0 and px > risk):
        return False
    qty = equity * mc.risk_per_trade / risk
    qty = min(qty, equity * mc.max_position_weight / px, state["cash"] * 0.98 / px)
    min_cost = float((ex.exchange.markets[sym].get("limits", {}).get("cost", {}) or {}).get("min") or 0.0)
    if qty <= 0 or qty * px < max(min_cost, 5.0):
        return False
    try:
        order = ex.place_market_buy(sym, qty)
    except Exception as exc:
        print(f"[ml] buy {sym} failed: {exc}")
        return False
    f = _fill(order, px)
    if f["filled_qty"] <= 0:
        print(f"[ml] buy {sym} not filled (testnet liquidity)")
        return False
    fill = f["avg_price"]
    state["cash"] -= f["filled_qty"] * fill + f["fee"]
    now = _now()
    state["positions"][sym] = {
        "qty": f["filled_qty"], "entry": fill, "risk": risk,
        "sl": fill - risk, "tp": fill + mc.reward_risk * risk,
        "entry_time": now.isoformat(), "last_check": now.isoformat(),
        "expiry": (now + timedelta(days=mc.horizon_days)).isoformat(), "p": p,
    }
    _append(TRADES_PATH, {"timestamp": now.isoformat(), "symbol": sym, "side": "buy", "qty": f["filled_qty"],
                          "price": fill, "notional": f["filled_qty"] * fill, "reason": f"ML p={p:.2f}",
                          "r_multiple": "", "pnl": "", "cash_after": round(state["cash"], 4)})
    print(f"[ml] BUY {sym}: {f['filled_qty']:.6g} @ {fill:.6g}  stop {fill - risk:.6g}  "
          f"target {fill + mc.reward_risk * risk:.6g}  (R:R {mc.reward_risk:.1f})")
    return True


def portfolio_equity(state: dict, prices: Optional[Dict[str, float]] = None) -> float:
    total = float(state["cash"])
    for sym, pos in state["positions"].items():
        try:
            px = (prices or {}).get(sym) or mirror_price(sym)
        except Exception:
            px = float(pos["entry"])
        total += float(pos["qty"]) * px
    return total


def run_ml_cycle(cfg: Optional[Config] = None, news: Optional[dict] = None) -> dict:
    cfg = cfg or Config()
    if not cfg.ml.ml_enabled:
        return {"enabled": False}
    state = load_state()
    ex = _executor()
    closed = manage_exits(ex, state, cfg, news)
    save_state(state)
    opened = generate_entries(ex, state, cfg, news)
    save_state(state)
    equity = portfolio_equity(state)
    status = json.loads(STATUS_PATH.read_text(encoding="utf-8")) if STATUS_PATH.exists() else {}
    _append(EQUITY_PATH, {"timestamp": _now().isoformat(), "cash": round(state["cash"], 4),
                          "equity": round(equity, 4), "open_trades": len(state["positions"]),
                          "realized_pnl": round(float(state.get("realized_pnl", 0.0)), 4),
                          "entries_allowed": status.get("entries_allowed")})
    summary = {"enabled": True, "equity": equity, "open": len(state["positions"]),
               "opened": opened, "closed": closed, "entries_allowed": status.get("entries_allowed")}
    print(f"[ml] ML sleeve equity ${equity:,.2f} (start ${STARTING_CAPITAL:,.0f}) | open trades {len(state['positions'])}")
    return summary


if __name__ == "__main__":
    from news_feed import news_snapshot

    print(run_ml_cycle(news=news_snapshot()))
