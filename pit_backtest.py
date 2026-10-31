"""
Point-in-time (hindsight-free) backtest of the V7 rule-based strategy.

What changes vs. backtest.BacktestEngine:  ONLY how coins are chosen.
  - Universe = top N coins by trailing 30-day median dollar volume AS OF EACH DATE, drawn from
    every Binance USDT pair ever listed (incl. delisted coins) - universe_data.py.
    (Dollar volume is the market-cap proxy: free market-cap history needs a paid API key.)
  - Hand-picked inputs removed: sector list (XRP+HBAR / ONDO) -> one "sector" per universe coin
    (sector_rotation_top_n etc. unchanged); per-coin risk caps chosen with hindsight; the
    hand-picked meme list.
  - Coins enter/leave the universe over time; a coin whose data ends (delisting) is sold at its
    last available close.
Everything else - sleeves and their effective weights, risk overlays, stops, take-profit,
trend-break exits, rebalance cadence, next-open execution - is the original engine's code.

Run: python pit_backtest.py
"""

from __future__ import annotations

import io
import json
import math
from contextlib import redirect_stdout
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

import numpy as np
import pandas as pd

from backtest import BacktestEngine
from config import Config
from utils import annualize_return, max_drawdown, sharpe_ratio

ROOT = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# Universe                                                                    #
# --------------------------------------------------------------------------- #
class PointInTimeUniverse:
    """Top-N by trailing median dollar volume, using only data up to each date."""

    def __init__(self, data: Dict[str, pd.DataFrame], top_n: int = 20, lookback: int = 30,
                 min_history: int = 90, min_days_present: int = 20):
        self.top_n = top_n
        cols = {}
        for sym, df in data.items():
            qv = df["QuoteVolume"] if "QuoteVolume" in df else df["Close"] * df["Volume"]
            med = qv.rolling(lookback, min_periods=min_days_present).median()
            age = pd.Series(np.arange(1, len(df) + 1), index=df.index)
            cols[sym] = med.where(age >= min_history)
        self.score = pd.DataFrame(cols).sort_index()

    def at(self, date: pd.Timestamp) -> Set[str]:
        if date not in self.score.index:
            prior = self.score.index[self.score.index <= date]
            if prior.empty:
                return set()
            date = prior[-1]
        row = self.score.loc[date].dropna()
        return set(row.nlargest(self.top_n).index)


def honest_config(roundtrip_cost: float = 0.003) -> Config:
    """V7 config with every hand-picked / hindsight input removed; all parameters unchanged."""
    cfg = Config()
    cfg.portfolio.fee_rate = roundtrip_cost / 2 * (2 / 3)        # e.g. 0.10% fee per side ...
    cfg.portfolio.slippage_rate = roundtrip_cost / 2 * (1 / 3)   # ... + 0.05% slippage per side
    cfg.risk.high_risk_asset_caps = {}                           # "OP was worst" etc. = hindsight
    cfg.meme.meme_enabled = False
    cfg.meme.meme_symbols = []                                   # DOGE etc. are ordinary coins here
    cfg.strategy.yield_enabled = False
    cfg.data.use_rules_based_universe = True
    return cfg


# --------------------------------------------------------------------------- #
# Engine                                                                      #
# --------------------------------------------------------------------------- #
class PITBacktestEngine(BacktestEngine):
    """
    Same daily loop as BacktestEngine.run (signal at close t, execute at open t+1, mark at
    close t+1), but over the UNION of dates with a changing coin set.
    universe_fn(date) -> set of symbols eligible that day.
    dynamic_sectors=True turns each eligible coin into its own sector for sector rotation.
    """

    def __init__(self, config: Config, universe_fn: Callable[[pd.Timestamp], Set[str]],
                 dynamic_sectors: bool = True, view_all_live: bool = False):
        super().__init__(config)
        self.universe_fn = universe_fn
        self.dynamic_sectors = dynamic_sectors
        # view_all_live=True mirrors BacktestEngine.run exactly (overlays/meme sleeve see every
        # loaded coin) - used to validate this loop against the original engine.
        self.view_all_live = view_all_live
        self.delist_exits: List[Dict[str, object]] = []

    def _select_universe(self, market_data, date):           # rule-based universe replaces the file list
        return {s for s in self.universe_fn(date) if s in market_data}

    def run_pit(self, data: Dict[str, pd.DataFrame], start: str, end: Optional[str] = None) -> dict:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end) if end else max(df.index.max() for df in data.values())
        dates = sorted({d for df in data.values() for d in df.index if start_ts <= d <= end_ts})
        last_bar = {s: df.index.max() for s, df in data.items()}
        last_px: Dict[str, float] = {}

        def px_at(sym, date, col):
            df = data[sym]
            if date in df.index:
                v = float(df.at[date, col])
                if col == "Close":
                    last_px[sym] = v
                return v
            return last_px.get(sym, 0.0)

        equity_points: List[float] = [self.portfolio.mark_to_market({})]
        date_points = [dates[0]]
        universe_hist = []
        current_universe: Set[str] = set()
        last_refresh = None

        for i in range(1, len(dates)):
            sig_d, exe_d = dates[i - 1], dates[i]
            held = {s for s, p in self.portfolio.state.positions.items() if abs(p.quantity) > 1e-12}

            # Delisting: data has ended -> sell at the last available close.
            for s in list(held):
                if last_bar[s] < exe_d:
                    px = last_px.get(s) or float(data[s]["Close"].iloc[-1])
                    self.portfolio.reduce_position(s, 1.0, px, exe_d)
                    self.delist_exits.append({"date": str(exe_d.date()), "symbol": s, "price": px})
                    held.discard(s)

            if self._should_refresh_universe(sig_d, last_refresh):
                current_universe = self._select_universe(
                    {s: df for s, df in data.items() if sig_d in df.index}, sig_d)
                last_refresh = sig_d
                universe_hist.append({"date": str(sig_d.date()), "coins": sorted(current_universe)})

            pool = set(data) if self.view_all_live else (current_universe | held)
            live = {s for s in pool if sig_d in data[s].index and exe_d in data[s].index}
            md = {s: data[s] for s in live}
            eligible = current_universe & live
            if self.dynamic_sectors:
                self.config.strategy.sector_definitions = {s.split("(")[0]: [s] for s in sorted(eligible)}

            signal_close = {s: px_at(s, sig_d, "Close") for s in md}
            exec_open = {s: px_at(s, exe_d, "Open") for s in md}
            for s in held - set(md):                                # held but no bar today: carry last price
                signal_close[s] = exec_open[s] = last_px.get(s, 0.0)

            signal_equity = self.portfolio.mark_to_market(signal_close)
            current_dd = self._current_drawdown(equity_points, signal_equity)
            self.risk.enforce_drawdown_guard(current_dd)
            self.risk.maybe_resume(current_dd)

            if md:
                self._apply_meme_rug_exits(md, sig_d, exec_open, exe_d)
                self._apply_stop_losses(signal_close, exec_open, exe_d)
                self._apply_take_profits(md, sig_d, signal_close, exec_open, exe_d)
                self._apply_trend_break_exits(md, sig_d, exec_open, exe_d)
                if self.risk.trading_enabled and self.portfolio.should_rebalance(sig_d):
                    w = self._target_weights_with_risk_overlays(md, sig_d, current_dd, eligible_symbols=eligible)
                    self.portfolio.rebalance(w, exec_open, exe_d)

            exec_close = {s: px_at(s, exe_d, "Close") for s in md}
            for s, p in self.portfolio.state.positions.items():
                if s not in exec_close:
                    exec_close[s] = last_px.get(s, 0.0)
            equity_points.append(self.portfolio.mark_to_market(exec_close))
            date_points.append(exe_d)

        eq = pd.Series(equity_points, index=date_points)
        rets = eq.pct_change().fillna(0.0)
        total = eq.iloc[-1] / eq.iloc[0] - 1
        return {
            "start": str(eq.index[0].date()), "end": str(eq.index[-1].date()),
            "cagr": annualize_return(total_return=float(total), periods=len(eq), annualization_days=365),
            "sharpe": sharpe_ratio(rets, annualization_days=365), "max_dd": max_drawdown(eq),
            "total_return": float(total), "trades": len(self.portfolio.trade_log),
            "fees_and_slippage": float(self.portfolio.total_fee_paid + self.portfolio.total_slippage_cost),
            "delist_exits": len(self.delist_exits), "equity": eq, "universe_history": universe_hist,
        }


# --------------------------------------------------------------------------- #
# Benchmarks                                                                  #
# --------------------------------------------------------------------------- #
def buy_and_hold(df: pd.DataFrame, start: str, end: str, roundtrip_cost: float) -> dict:
    s = df["Close"][(df.index >= start) & (df.index <= end)]
    eq = s / s.iloc[0] * (1 - roundtrip_cost / 2)
    return _stats(eq)


def equal_weight_universe(data, universe: PointInTimeUniverse, start: str, end: str, roundtrip_cost: float) -> dict:
    """Hold the point-in-time top N equally weighted, rebalanced monthly."""
    dates = sorted({d for df in data.values() for d in df.index if pd.Timestamp(start) <= d <= pd.Timestamp(end)})
    eq, val, weights, last_month = [1.0], 1.0, {}, None
    for prev, d in zip(dates[:-1], dates[1:]):
        if prev.month != last_month:
            new = universe.at(prev)
            tgt = {s: 1 / len(new) for s in new} if new else {}
            turnover = sum(abs(tgt.get(s, 0) - weights.get(s, 0)) for s in set(tgt) | set(weights))
            val *= 1 - turnover * roundtrip_cost / 2
            weights, last_month = tgt, prev.month
        r = 0.0
        for s, w in weights.items():
            df = data[s]
            if prev in df.index and d in df.index:
                r += w * (df.at[d, "Close"] / df.at[prev, "Close"] - 1)
            elif prev in df.index:                                  # delisted: position lost at last close
                pass
        val *= 1 + r
        eq.append(val)
    return _stats(pd.Series(eq, index=dates))


def _v7_original_data(end: str) -> Dict[str, pd.DataFrame]:
    """The original hand-picked V7 dataset (historic_data/, quality pre-filter, aligned) up to `end`."""
    from data import DataManager
    from live_runner import align_with_min_history
    from main import build_symbol_file_map

    cfg = Config()
    dm = DataManager(cfg)
    with redirect_stdout(io.StringIO()):
        loaded = dm.load_multi_csv(build_symbol_file_map(cfg, ROOT))
        usd, _, _ = dm.filter_by_allowed_quote_currency(loaded)
        pre, _ = dm.pre_filter_quality(usd)
    pre = {s: df[df.index <= end] for s, df in pre.items()}
    return align_with_min_history(pre, cfg.data.min_history_days)


def main() -> None:
    from universe_data import load_all

    END = "2026-09-30"            # last full month in Binance's archive
    V7_START = "2024-01-18"       # V7's original backtest window start
    LONG_START = "2020-01-01"
    rows, extra = [], {}

    def add(label, r, note=""):
        rows.append({"run": label, "window": f"{r['start']} .. {r['end']}", "cagr": r["cagr"],
                     "sharpe": r["sharpe"], "max_dd": r["max_dd"], "total_return": r["total_return"],
                     "trades": r.get("trades", ""), "note": note})
        print(f"{label:58s} CAGR {r['cagr']:+7.1%}  Sharpe {r['sharpe']:5.2f}  MaxDD {r['max_dd']:6.1%}  "
              f"total {r['total_return']:+8.1%}  {note}", flush=True)

    # --- A/B: original engine, original hand-picked data -----------------------------------
    v7 = _v7_original_data(END)
    for label, cost in (("A  V7 as-is (hand-picked, 0.2% round trip)", None),
                        ("B  V7 as-is, costs raised to 0.3% round trip", 0.003)):
        cfg = Config()
        if cost:
            cfg.portfolio.fee_rate, cfg.portfolio.slippage_rate = cost / 3, cost / 6
        res = BacktestEngine(cfg).run(v7)
        add(label, {"start": str(res.equity_curve.index[0].date()), "end": str(res.equity_curve.index[-1].date()),
                    "cagr": res.cagr, "sharpe": res.sharpe, "max_dd": res.max_drawdown,
                    "total_return": res.total_return, "trades": res.diagnostics["trade_count"]})

    # --- Binance archive (incl. delisted) ----------------------------------------------------
    allb = {s: df for s, df in load_all().items() if df.index.min() <= pd.Timestamp(END)}
    allb = {s: df[df.index <= END] for s, df in allb.items()}
    uni = PointInTimeUniverse(allb, top_n=20)

    # C: same hand-picked V7 coin files + sectors, but on Binance data at 0.3% (isolates data source)
    hand = {s for s in v7}
    base_map = {s.split("(")[0].replace("CHAINLINK", "LINK"): s for s in hand}
    hand_b = {f"{b}(USD)": allb[f"{b}(USD)"] for b in base_map if f"{b}(USD)" in allb}
    cfg_c = Config()
    cfg_c.portfolio.fee_rate, cfg_c.portfolio.slippage_rate = 0.001, 0.0005
    cfg_c.strategy.sector_definitions = {"HighBeta": ["XRP(USD)", "HBAR(USD)"], "ONDO": ["ONDO(USD)"]}
    cfg_c.meme.meme_symbols = [f"{m.split('(')[0]}(USD)" for m in Config().meme.meme_symbols]
    cfg_c.risk.high_risk_asset_caps = {"OP(USD)": 0.05}
    picker = BacktestEngine(cfg_c)
    eng_c = PITBacktestEngine(cfg_c, universe_fn=lambda d: BacktestEngine._select_universe(
        picker, {s: df for s, df in hand_b.items() if d in df.index}, d), dynamic_sectors=False)
    add("C  hand-picked coins & sectors, Binance data, 0.3%", eng_c.run_pit(hand_b, V7_START),
        f"{len(hand_b)} coins")

    # D/E: HONEST - point-in-time top 20, no hand-picked inputs, 0.3%
    for label, start in (("D  HONEST: point-in-time top-20, 0.3% (V7 window)", V7_START),
                         ("E  HONEST: point-in-time top-20, 0.3% (2020-2026)", LONG_START)):
        eng = PITBacktestEngine(honest_config(0.003), universe_fn=uni.at, dynamic_sectors=True)
        r = eng.run_pit(allb, start)
        add(label, r, f"{r['delist_exits']} delisting exits")
        extra[label[0]] = {"universe_history": r["universe_history"][::13], "delist_exits": eng.delist_exits}

    # Benchmarks
    for start, tag in ((V7_START, "V7 window"), (LONG_START, "2020-2026")):
        add(f"   benchmark: BTC buy & hold ({tag})", buy_and_hold(allb["BTC(USD)"], start, END, 0.003))
        add(f"   benchmark: equal-weight top-20, monthly ({tag})", equal_weight_universe(allb, uni, start, END, 0.003))

    out = ROOT / "outputs" / "honest_baseline.json"
    out.write_text(json.dumps({"results": rows, "details": extra}, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved {out}")


def _stats(eq: pd.Series) -> dict:
    r = eq.pct_change().fillna(0.0)
    total = eq.iloc[-1] / eq.iloc[0] - 1
    return {"start": str(eq.index[0].date()), "end": str(eq.index[-1].date()),
            "cagr": annualize_return(total_return=float(total), periods=len(eq), annualization_days=365),
            "sharpe": sharpe_ratio(r, annualization_days=365), "max_dd": max_drawdown(eq),
            "total_return": float(total)}


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    main()
