"""
12. Execution. (a) Reconstruct the 2026-09-29 HBAR/ONDO/XRP buys from the logs.
(b) Prove whether an ML signal can fire twice: run ml_live.run_ml_cycle with a FAKE
exchange in a temp folder (no orders sent, project logs untouched) where the network
fails right after the first fill.
"""
import json
import tempfile
from pathlib import Path

import pandas as pd

import _common as c

# ---------------------------------------------------------------- (a) Sep 29
tr = pd.read_csv(c.ROOT / "logs" / "live_trades.csv", parse_dates=["timestamp"])
runs = pd.read_csv(c.ROOT / "logs" / "live_runner_runs.csv", parse_dates=["timestamp"])
day = tr[tr["timestamp"].dt.date == pd.Timestamp("2026-09-29").date()]
print("(a) 2026-09-29 V7 fills (UTC):")
print(day[["timestamp", "symbol", "side", "quantity", "price", "order_value_usd"]].to_string(index=False))
cum = day[day.side == "buy"].groupby("symbol")["order_value_usd"].sum()
print("\ncumulative buys per coin:", cum.round(2).to_dict(),
      "| V7 target that day ~26.7% x $500 = $133.4; per-order cap 12% x equity = $60")
print("runner log rows that day:")
print(runs[runs["timestamp"].dt.date == pd.Timestamp("2026-09-29").date()][["run_number", "timestamp", "trades_executed"]]
      .to_string(index=False))

# ---------------------------------------------------------------- (b) ML double-fire
import ml_live  # noqa: E402
import ml_strategy  # noqa: E402
import anti_rug  # noqa: E402

tmp = Path(tempfile.mkdtemp(prefix="ml_audit_"))
for name in ("STATE_PATH", "TRADES_PATH", "EQUITY_PATH", "SIGNALS_PATH"):
    setattr(ml_live, name, tmp / getattr(ml_live, name).name)
ml_live.ML_LOGS = tmp
status = tmp / "ml_status.json"
status.write_text(json.dumps({"entries_allowed": True, "health_reason": "audit"}))
ml_live.STATUS_PATH = status


class FakeExchange:
    markets = {f"{b}/USDT": {"limits": {"cost": {"min": 1.0}}} for b in
               ["XRP", "HBAR", "ONDO", "SOL", "AVAX", "LINK", "NEAR", "SUI", "AAVE", "UNI", "ARB", "OP",
                "PENDLE", "DOGE", "SHIB", "PEPE", "BONK", "WIF", "FLOKI", "CHAINLINK"]}


class FakeEx:
    exchange = FakeExchange()
    buys = 0

    def place_market_buy(self, sym, qty):
        FakeEx.buys += 1
        return {"filled": qty, "average": 1.0, "cost": qty, "fee": {"cost": 0.0}}


calls = {"n": 0}


def flaky_price(sym):
    calls["n"] += 1
    if calls["n"] >= 2:                                   # network drops after the first fill
        raise ConnectionError("simulated: data-api.binance.vision unreachable")
    return 1.0


def run_cycle():
    try:
        ml_live.run_ml_cycle(c.cfg_for(), news=None)
        return "ok"
    except Exception as exc:
        return f"raised {type(exc).__name__}: {exc}"


ml_live._executor = lambda: FakeEx()
ml_live.mirror_price = flaky_price
ml_strategy.expected_r = lambda p, rr: p * 0 + 1.0           # force every coin over the EV gate
anti_rug.AntiRugGuard.history_check = lambda self, *a, **k: anti_rug.RugVerdict("x")
anti_rug.AntiRugGuard.market_check = lambda self, *a, **k: anti_rug.RugVerdict("x")

first = run_cycle()
state1 = json.loads(ml_live.STATE_PATH.read_text()) if ml_live.STATE_PATH.exists() else {}
log1 = pd.read_csv(ml_live.TRADES_PATH) if ml_live.TRADES_PATH.exists() else pd.DataFrame()
print("\n(b) cycle 1:", first)
print(f"    exchange buys sent: {FakeEx.buys} | trade-log buy rows: {len(log1)} | "
      f"state.json positions: {list((state1.get('positions') or {}).keys())} | "
      f"state last_signal_date: {state1.get('last_signal_date')}")
calls["n"] = 0
second = run_cycle()
log2 = pd.read_csv(ml_live.TRADES_PATH)
print("    cycle 2 (next hour, network back for one call):", second)
print(f"    exchange buys sent so far: {FakeEx.buys} | trade-log rows: {len(log2)} | symbols bought: "
      f"{log2['symbol'].tolist()}")
dup = log2["symbol"].duplicated().any()
print("    SAME COIN BOUGHT TWICE FOR THE SAME DAILY SIGNAL:", bool(dup))
c.save("a12_execution.json", {"cycle1": first, "cycle2": second, "buys": FakeEx.buys,
                               "log_symbols": log2["symbol"].tolist(), "duplicate": bool(dup)})
