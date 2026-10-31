"""
Crash-safety of the ML ledger (ml_live.py). Offline: fake exchange, fake prices, temp folder -
no orders are sent and the real logs/ml/ files are never touched.
Run: python tests/test_ml_persistence.py

A) Network drops right after a BUY fills  -> position + signal date already on disk;
   the next cycle must not buy the same signal again.
B) Process dies right after an EXIT fill  -> position already gone from disk;
   the next cycle must not sell it a second time.
"""

import json
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import anti_rug  # noqa: E402
import ml_live  # noqa: E402
import ml_strategy  # noqa: E402
from config import Config  # noqa: E402

fails = 0


def check(name, ok, detail=""):
    global fails
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")


def isolate():
    tmp = Path(tempfile.mkdtemp(prefix="ml_persist_"))
    for name in ("STATE_PATH", "TRADES_PATH", "EQUITY_PATH", "SIGNALS_PATH"):
        setattr(ml_live, name, tmp / getattr(ml_live, name).name)
    ml_live.ML_LOGS = tmp
    status = tmp / "ml_status.json"
    status.write_text(json.dumps({"entries_allowed": True, "health_reason": "test"}))
    ml_live.STATUS_PATH = status


class FakeExchange:
    markets = {f"{b}/USDT": {"limits": {"cost": {"min": 1.0}}} for b in
               ["XRP", "HBAR", "ONDO", "SOL", "AVAX", "LINK", "NEAR", "SUI", "AAVE", "UNI", "ARB", "OP",
                "PENDLE", "DOGE", "SHIB", "PEPE", "BONK", "WIF", "FLOKI", "CHAINLINK"]}


class FakeEx:
    exchange = FakeExchange()
    buys = sells = 0

    def place_market_buy(self, sym, qty):
        FakeEx.buys += 1
        return {"filled": qty, "average": 1.0, "cost": qty, "fee": {"cost": 0.0}}

    def place_market_sell(self, sym, qty):
        FakeEx.sells += 1
        return {"filled": qty, "average": 0.5, "cost": qty * 0.5, "fee": {"cost": 0.0}}


def run_cycle():
    try:
        ml_live.run_ml_cycle(Config(), news=None)
        return None
    except Exception as exc:
        return exc


# Common offline wiring
ml_live._executor = lambda: FakeEx()
ml_live.MAX_MODEL_AGE_DAYS = 10_000
ml_strategy.expected_r = lambda p, rr: p * 0 + 1.0                  # every coin clears the EV gate
anti_rug.AntiRugGuard.history_check = lambda self, *a, **k: anti_rug.RugVerdict("x")
anti_rug.AntiRugGuard.market_check = lambda self, *a, **k: anti_rug.RugVerdict("x")
ml_live.hourly_bars_since = lambda sym, since: []                   # no stop/target touches

# ---------------------------------------------------------------- A) crash after BUY
isolate()
calls = {"n": 0}


def flaky_price(sym):
    calls["n"] += 1
    if calls["n"] >= 2:
        raise ConnectionError("simulated network drop after first fill")
    return 1.0


ml_live.mirror_price = flaky_price
err = run_cycle()
state = json.loads(ml_live.STATE_PATH.read_text())
check("A1 cycle crashed after first fill (simulated)", isinstance(err, ConnectionError), repr(err))
check("A2 filled position is on disk", len(state["positions"]) == 1, str(list(state["positions"])))
check("A3 signal date is on disk", bool(state.get("last_signal_date")), str(state.get("last_signal_date")))
check("A4 cash was debited on disk", state["cash"] < 500.0, f"cash={state['cash']:.2f}")
buys_after_1 = FakeEx.buys

ml_live.mirror_price = lambda sym: 1.0                              # network back
err2 = run_cycle()
check("A5 next cycle runs cleanly", err2 is None, repr(err2))
check("A6 NO second buy of the same signal", FakeEx.buys == buys_after_1, f"buys={FakeEx.buys}")

# ---------------------------------------------------------------- B) crash after EXIT
isolate()
now = datetime.now(timezone.utc)
ml_live.save_state({
    "cash": 400.0, "realized_pnl": 0.0, "last_signal_date": "2999-01-01",   # no new entries
    "positions": {"XRP/USDT": {"qty": 100.0, "entry": 1.0, "risk": 0.1, "sl": 0.9, "tp": 1.2,
                               "entry_time": now.isoformat(), "last_check": now.isoformat(),
                               "expiry": (now - timedelta(minutes=1)).isoformat(), "p": 0.5}},
})
real_append = ml_live._append


def crashing_append(path, row):
    if path == ml_live.TRADES_PATH and row.get("side") == "sell":
        raise OSError("simulated crash right after the exit fill")
    return real_append(path, row)


ml_live._append = crashing_append
err = run_cycle()
state = json.loads(ml_live.STATE_PATH.read_text())
check("B1 cycle crashed right after the exit fill (simulated)", isinstance(err, OSError), repr(err))
check("B2 sold position is gone from disk", "XRP/USDT" not in state["positions"], str(list(state["positions"])))
check("B3 sale proceeds are on disk", abs(state["cash"] - 450.0) < 1e-9, f"cash={state['cash']:.2f}")
sells_after_1 = FakeEx.sells
ml_live._append = real_append
err2 = run_cycle()
check("B4 next cycle runs cleanly", err2 is None, repr(err2))
check("B5 NO second sell of the same position", FakeEx.sells == sells_after_1, f"sells={FakeEx.sells}")

print(f"\n{'ALL PASSED' if not fails else f'{fails} FAILED'}")
sys.exit(1 if fails else 0)
