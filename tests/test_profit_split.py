"""
Offline test of the 50/50 profit split (tax reserve vs. reinvested), V7 and ML ledgers.
Run: python tests/test_profit_split.py   (uses a fake exchange - no orders are sent)
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ml_live  # noqa: E402
from config import Config  # noqa: E402
from paper_account import PaperAccount  # noqa: E402
from tax_reserve import TaxReserveManager  # noqa: E402

RATE = Config().portfolio.tax_reserve_rate
fails = 0


def check(name, got, want):
    global fails
    ok = abs(got - want) < 1e-9
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {name:46s} got {got:10.4f}  want {want:10.4f}")


# ---- V7 paper account -------------------------------------------------------
acct = PaperAccount(starting_capital_usd=500, cash_usd=500, tax_manager=TaxReserveManager(RATE))
acct.apply_fill("XRP/USDT", "buy", 10, 10.0, fee=0.0)      # spend $100
acct.apply_fill("XRP/USDT", "sell", 10, 12.0, fee=0.0)     # +$20 profit
check("V7 rate from config", RATE, 0.50)
check("V7 tax reserve = half of $20 profit", acct.tax_manager.total_tax_reserve, 10.0)
check("V7 tradable cash = $500 + $10 reinvested", acct.cash_usd, 510.0)
check("V7 trading equity excludes reserve", acct.trading_equity({}), 510.0)
check("V7 total equity includes reserve", acct.total_equity({}), 520.0)
acct.apply_fill("XRP/USDT", "buy", 10, 10.0, fee=0.0)
acct.apply_fill("XRP/USDT", "sell", 10, 9.0, fee=0.0)      # -$10 loss
check("V7 loss doesn't release reserve", acct.tax_manager.total_tax_reserve, 10.0)
check("V7 loss comes out of tradable cash", acct.cash_usd, 500.0)

# ---- ML ledger (fake exchange fill) -----------------------------------------
class FakeEx:
    def place_market_sell(self, sym, qty):
        return {"filled": qty, "average": 12.0, "cost": qty * 12.0, "fee": {"cost": 0.0}}


tmp = Path(tempfile.mkdtemp())
ml_live.ML_LOGS, ml_live.TRADES_PATH, ml_live.STATE_PATH = tmp, tmp / "ml_trades.csv", tmp / "state.json"
state = {"cash": 400.0, "positions": {"XRP/USDT": {"qty": 10.0, "entry": 10.0, "risk": 1.0}},
         "realized_pnl": 0.0}
ml_live._close(FakeEx(), state, "XRP/USDT", "take_profit", 12.0)
check("ML tax reserve = half of $20 profit", state["tax_reserve"], 10.0)
check("ML cash = 400 + 120 proceeds - 10 reserve", state["cash"], 510.0)
check("ML trading equity excludes reserve", ml_live.trading_equity(state), 510.0)
check("ML total equity includes reserve", ml_live.portfolio_equity(state), 520.0)

print(f"\n{'ALL PASSED' if not fails else f'{fails} FAILED'}")
sys.exit(1 if fails else 0)
