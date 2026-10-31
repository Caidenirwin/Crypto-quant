"""
Forward testing (paper trading) module.

Execution timing (same as backtest):
- signal on close(t)
- execute on open(t+1)
- mark equity on close(t+1)
"""

from __future__ import annotations

import json
from copy import deepcopy
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd

from backtest import BacktestEngine
from config import Config
from data import DataManager
import main as _main_module
from indicators import rolling_mean, rolling_std, rolling_volatility, returns, zscore
from utils import max_drawdown


# Consistent forward reason codes.
INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"
VOLUME_FILTER_FAILED = "VOLUME_FILTER_FAILED"
VOLATILITY_CONDITION_FAILED = "VOLATILITY_CONDITION_FAILED"
TREND_FILTER_FAILED = "TREND_FILTER_FAILED"
REGIME_FILTER_BLOCKED = "REGIME_FILTER_BLOCKED"
DRAWDOWN_PROTECTION_BLOCKED = "DRAWDOWN_PROTECTION_BLOCKED"
ALLOCATION_CAP_BLOCKED = "ALLOCATION_CAP_BLOCKED"
NO_ELIGIBLE_ASSETS = "NO_ELIGIBLE_ASSETS"
NO_SIGNAL = "NO_SIGNAL"
SIGNAL_ACCEPTED = "SIGNAL_ACCEPTED"
EXECUTED = "EXECUTED"
EXECUTION_SKIPPED_ZERO_SIZE = "EXECUTION_SKIPPED_ZERO_SIZE"
PAIR_CONDITION_NOT_MET = "PAIR_CONDITION_NOT_MET"
NO_BREAKOUT_SIGNAL = "NO_BREAKOUT_SIGNAL"
NO_MOMENTUM_SIGNAL = "NO_MOMENTUM_SIGNAL"
DATA_VALIDATION_WARNING = "DATA_VALIDATION_WARNING"


def auto_discover_symbol_files(root: Path, search_roots: list | None = None) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    dirs = [root / d for d in search_roots] if search_roots else [root]
    for search_dir in dirs:
        glob_root = search_dir if search_dir.is_dir() else root
        for p in glob_root.glob("*.csv"):
            if p.name.lower().startswith("forward_"):
                continue
            symbol = p.stem.split("-")[0].strip().upper().replace(" ", "_")
            if symbol:
                mapping[symbol] = str(p)
    return mapping


@dataclass
class ForwardTestResult:
    equity_log: pd.DataFrame
    trade_log: pd.DataFrame
    position_log: pd.DataFrame
    signal_trace_log: pd.DataFrame
    audit: Dict[str, object]


class ForwardTester:
    def __init__(self, config: Config):
        self.config = config
        self.data_manager = DataManager(config)
        self.engine = BacktestEngine(config)

    def run(self, market_data: Dict[str, pd.DataFrame]) -> ForwardTestResult:
        if not market_data:
            raise ValueError("market_data is empty")
        if self.config.backtest.execution_timing != "signal_close_execute_next_open":
            raise ValueError("Unsupported execution_timing for forward test")

        validation_warnings, unusable_symbols = self._data_quality_checks(market_data)

        (
            working_data,
            eval_dates,
            eval_start,
            eval_end,
            pre_eval_dates,
        ) = self._build_forward_windows(market_data)
        if len(eval_dates) < 2:
            raise ValueError("Forward window too short; need at least 2 bars")

        history_ok, required_history, symbol_history = self._symbol_history_diagnostics(working_data)
        if self.config.forward.forward_require_full_history:
            keep_symbols = {
                s
                for s, info in symbol_history.items()
                if bool(info.get("passes_min_required_history", False))
            }
            working_data = {s: df for s, df in working_data.items() if s in keep_symbols}
            eval_dates = self._common_dates_subset(working_data, eval_start, eval_end)
            if len(eval_dates) < 2:
                raise ValueError("No symbols left after forward history requirement filter")

        if unusable_symbols:
            working_data = {s: df for s, df in working_data.items() if s not in unusable_symbols}
            eval_dates = self._common_dates_subset(working_data, eval_start, eval_end)
            if len(eval_dates) < 2:
                raise ValueError("No symbols left after data validation filter")

        mode = self.config.forward.forward_start_mode
        bootstrap_note = ""
        if mode == "carry_backtest":
            bootstrap_note = self._carry_backtest_state(working_data, pre_eval_dates)
        elif mode != "fresh":
            raise ValueError("forward_start_mode must be 'fresh' or 'carry_backtest'")

        all_dates = self.engine._common_dates(working_data)
        first_exec_date = eval_dates[0]
        first_idx = all_dates.index(first_exec_date)
        if first_idx <= 0:
            raise ValueError("Not enough pre-forward bar for signal generation")
        seed_date = all_dates[first_idx - 1]
        seed_close = {s: float(df.loc[seed_date, "Close"]) for s, df in working_data.items()}

        equity_points: List[float] = [self.engine.portfolio.mark_to_market(seed_close)]
        equity_dates: List[pd.Timestamp] = [seed_date]
        drawdowns: List[float] = [0.0]
        returns_list: List[float] = [0.0]
        position_rows: List[Dict[str, object]] = [
            {"date": seed_date, "positions": self._positions_snapshot(seed_close), "gross_exposure": 0.0}
        ]
        trade_rows: List[Dict[str, object]] = []
        signal_trace_rows: List[Dict[str, object]] = []

        current_universe: Set[str] = set(working_data.keys())
        last_universe_refresh_date: pd.Timestamp | None = None
        trade_cursor = len(self.engine.portfolio.trade_log)
        strategy_audit = self._empty_strategy_audit()
        rejection_reasons = Counter()
        signals_generated = Counter()
        signals_rejected = 0
        risk_suppression_events = 0
        latest_regime_state = "unknown"

        for exec_date in eval_dates:
            j = all_dates.index(exec_date)
            signal_date = all_dates[j - 1]
            signal_close = {s: float(df.loc[signal_date, "Close"]) for s, df in working_data.items()}
            exec_open = {s: float(df.loc[exec_date, "Open"]) for s, df in working_data.items()}
            exec_close = {s: float(df.loc[exec_date, "Close"]) for s, df in working_data.items()}

            signal_equity = self.engine.portfolio.mark_to_market(signal_close)
            y = self.engine.strategies["yield"]
            y.accrue_yield(self.engine.portfolio.holdings_value(signal_close))
            self.engine.portfolio.state.cash += y.conversion_amount(signal_date)

            current_dd = self.engine._current_drawdown(equity_points, signal_equity)
            self.engine.risk.enforce_drawdown_guard(current_dd)
            self.engine.risk.maybe_resume(current_dd)
            if not self.engine.risk.trading_enabled:
                rejection_reasons[DRAWDOWN_PROTECTION_BLOCKED] += 1
                risk_suppression_events += 1

            if self.engine._should_refresh_universe(signal_date, last_universe_refresh_date):
                current_universe = self.engine._select_universe(working_data, signal_date)
                last_universe_refresh_date = signal_date
            if not current_universe:
                rejection_reasons[NO_ELIGIBLE_ASSETS] += 1

            self._audit_strategy_signals(
                working_data,
                signal_date,
                current_universe,
                strategy_audit,
                rejection_reasons,
                signals_generated,
            )

            self.engine._apply_stop_losses(signal_close, exec_open, exec_date)
            self.engine._apply_take_profits(
                working_data, signal_date, signal_close, exec_open, exec_date
            )
            self.engine._apply_trend_break_exits(working_data, signal_date, exec_open, exec_date)

            strat_signals = self.engine._strategy_signal_weights(
                working_data, signal_date, eligible_symbols=current_universe
            )
            final_weights: Dict[str, float] = {}
            if self.engine.risk.trading_enabled and self.engine.portfolio.should_rebalance(signal_date):
                latest_regime_state = self.engine._regime_state(
                    working_data, signal_date, eligible_symbols=current_universe
                )
                final_weights = self.engine._target_weights_with_risk_overlays(
                    working_data, signal_date, current_dd, eligible_symbols=current_universe
                )
                self._capture_signal_trace(
                    signal_trace_rows,
                    strat_signals,
                    final_weights,
                    signal_date,
                    latest_regime_state,
                    current_dd,
                    current_universe,
                )
                self.engine.portfolio.rebalance(final_weights, exec_open, exec_date)
            else:
                self._capture_signal_trace(
                    signal_trace_rows,
                    strat_signals,
                    final_weights,
                    signal_date,
                    latest_regime_state,
                    current_dd,
                    current_universe,
                    blocked_code=DRAWDOWN_PROTECTION_BLOCKED if not self.engine.risk.trading_enabled else NO_SIGNAL,
                )

            if strat_signals and final_weights:
                for sname, sw in strat_signals.items():
                    for sym, w in sw.items():
                        if abs(w) <= 1e-12:
                            continue
                        if abs(final_weights.get(sym, 0.0)) <= 1e-12:
                            signals_rejected += 1
                            rejection_reasons[ALLOCATION_CAP_BLOCKED] += 1

            new_trades = self.engine.portfolio.trade_log[trade_cursor:]
            traded_symbols = set()
            for tr in new_trades:
                traded_symbols.add(tr.symbol)
                trade_rows.append(
                    {
                        "date": tr.date,
                        "symbol": tr.symbol,
                        "quantity": tr.quantity,
                        "reference_price": tr.reference_price,
                        "execution_price": tr.execution_price,
                        "gross_notional": tr.gross_notional,
                        "fee": tr.fee,
                        "slippage_cost": tr.slippage_cost,
                        "realized_pnl": tr.realized_pnl,
                        "reason_code": EXECUTED,
                    }
                )
            trade_cursor = len(self.engine.portfolio.trade_log)
            self._mark_trace_execution(signal_trace_rows, exec_date, traded_symbols)

            end_equity = self.engine.portfolio.mark_to_market(exec_close)
            prev_equity = equity_points[-1]
            day_ret = (end_equity / prev_equity - 1.0) if prev_equity > 0 else 0.0

            equity_points.append(end_equity)
            equity_dates.append(exec_date)
            returns_list.append(day_ret)
            drawdowns.append(self.engine._current_drawdown(equity_points, end_equity))
            gross_exp = self.engine.portfolio.gross_exposure(exec_close)
            position_rows.append(
                {
                    "date": exec_date,
                    "positions": self._positions_snapshot(exec_close),
                    "gross_exposure": (gross_exp / end_equity) if end_equity > 0 else 0.0,
                }
            )

        equity_log = pd.DataFrame(
            {"date": equity_dates, "equity": equity_points, "daily_return": returns_list, "drawdown": drawdowns}
        )
        trade_log = pd.DataFrame(trade_rows)
        position_log = pd.DataFrame(position_rows)
        signal_trace_log = pd.DataFrame(signal_trace_rows)

        latest = equity_log.iloc[-1]
        latest_positions = position_log.iloc[-1]["positions"]
        top3 = sorted(rejection_reasons.items(), key=lambda x: x[1], reverse=True)[:3]
        flat_status, flat_expl = self._flat_status(
            is_flat=(latest_positions == "FLAT"),
            symbols_loaded=len(working_data),
            active_universe=current_universe,
            history_ok=history_ok,
            risk_suppression=(risk_suppression_events > 0),
            signals_generated=sum(signals_generated.values()),
            trades_executed=len(trade_log),
            rejection_reasons=rejection_reasons,
        )

        audit = {
            "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "forward_mode": mode,
            "bootstrap_note": bootstrap_note,
            "execution_timing_assumption": "signal_close(t)_execute_open(t+1)_mark_close(t+1)",
            "forward_window_start": str(eval_dates[0].date()),
            "forward_window_end": str(eval_dates[-1].date()),
            "latest_data_date": str(equity_dates[-1].date()),
            "symbols_loaded": len(working_data),
            "active_universe": sorted(current_universe),
            "active_universe_size": len(current_universe),
            "current_equity": float(latest["equity"]),
            "daily_return": float(latest["daily_return"]),
            "current_positions": latest_positions,
            "drawdown": float(latest["drawdown"]),
            "max_drawdown_to_date": float(max_drawdown(equity_log["equity"].tolist())),
            "strategy_signal_counts": dict(signals_generated),
            "strategy_audit_results": self._summarize_strategy_audit(strategy_audit),
            "rejection_reason_breakdown": dict(rejection_reasons),
            "top3_no_entry_reasons": [{k: int(v)} for k, v in top3],
            "risk_suppression_flag": bool(risk_suppression_events > 0),
            "history_sufficiency_status": bool(history_ok),
            "required_history_bars": int(required_history),
            "per_symbol_history_diagnostics": symbol_history,
            "trades_executed": int(len(trade_log)),
            "is_flat": bool(latest_positions == "FLAT"),
            "flat_status": flat_status,
            "flat_state_explanation": flat_expl,
            "warnings": validation_warnings + ([] if history_ok else [INSUFFICIENT_HISTORY]),
            "latest_regime_state": latest_regime_state,
            "signals_rejected": int(signals_rejected),
        }

        return ForwardTestResult(
            equity_log=equity_log,
            trade_log=trade_log,
            position_log=position_log,
            signal_trace_log=signal_trace_log,
            audit=audit,
        )

    def _build_forward_windows(
        self, market_data: Dict[str, pd.DataFrame]
    ) -> Tuple[Dict[str, pd.DataFrame], List[pd.Timestamp], pd.Timestamp, pd.Timestamp, List[pd.Timestamp]]:
        common = self.engine._common_dates(market_data)
        if len(common) < 3:
            raise ValueError("Need at least 3 bars to run forward test")
        end = common[-1]
        start = end - pd.Timedelta(days=max(1, int(self.config.forward.forward_window_days) - 1))
        hist_start = start - pd.Timedelta(days=max(1, int(self.config.forward.forward_history_buffer_days)))
        working = {s: df.loc[(df.index >= hist_start) & (df.index <= end)].copy() for s, df in market_data.items()}
        eval_dates = self._common_dates_subset(working, start, end)
        pre_eval_dates = [d for d in self.engine._common_dates(working) if d < start]
        return working, eval_dates, start, end, pre_eval_dates

    def _common_dates_subset(
        self, market_data: Dict[str, pd.DataFrame], start: pd.Timestamp, end: pd.Timestamp
    ) -> List[pd.Timestamp]:
        all_common = self.engine._common_dates(market_data)
        return [d for d in all_common if d >= start and d <= end]

    def _carry_backtest_state(
        self, working_data: Dict[str, pd.DataFrame], pre_eval_dates: List[pd.Timestamp]
    ) -> str:
        if len(pre_eval_dates) < 2:
            return "carry_backtest requested but insufficient pre-forward history; using fresh-equivalent state"
        warm_end = pre_eval_dates[-1]
        warm_data = {s: df.loc[df.index <= warm_end].copy() for s, df in working_data.items()}
        bootstrap_engine = BacktestEngine(self.config)
        bootstrap_engine.run(warm_data)
        self.engine = bootstrap_engine
        return f"carried state through {str(warm_end.date())}"

    def _data_quality_checks(self, market_data: Dict[str, pd.DataFrame]) -> Tuple[List[str], Set[str]]:
        warnings: List[str] = []
        unusable: Set[str] = set()
        required = {"Open", "High", "Low", "Close", "Volume"}
        for s, df in market_data.items():
            cols = set(df.columns)
            miss = sorted(required - cols)
            if miss:
                warnings.append(f"{DATA_VALIDATION_WARNING}:{s}:MISSING_COLS:{','.join(miss)}")
                unusable.add(s)
                continue
            if df.index.has_duplicates:
                warnings.append(f"{DATA_VALIDATION_WARNING}:{s}:DUPLICATE_DATES")
            if not df.index.is_monotonic_increasing:
                warnings.append(f"{DATA_VALIDATION_WARNING}:{s}:UNSORTED_DATES")
            bad_num = (
                pd.to_numeric(df["Open"], errors="coerce").isna().sum()
                + pd.to_numeric(df["Close"], errors="coerce").isna().sum()
            )
            if bad_num > 0:
                warnings.append(f"{DATA_VALIDATION_WARNING}:{s}:NON_NUMERIC_PRICES")
            if (df["Open"].astype(float) <= 0).any() or (df["Close"].astype(float) <= 0).any():
                warnings.append(f"{DATA_VALIDATION_WARNING}:{s}:NON_POSITIVE_PRICES")
            if (df["Volume"].astype(float) == 0).any():
                warnings.append(f"{DATA_VALIDATION_WARNING}:{s}:ZERO_VOLUME_PRESENT")
        return warnings, unusable

    def _symbol_history_diagnostics(
        self, market_data: Dict[str, pd.DataFrame]
    ) -> Tuple[bool, int, Dict[str, Dict[str, object]]]:
        required = int(self.config.forward.min_required_history_bars)
        out: Dict[str, Dict[str, object]] = {}
        ok = True
        for s, df in market_data.items():
            bars = len(df)
            first = str(df.index.min().date()) if bars else ""
            last = str(df.index.max().date()) if bars else ""
            exp = pd.date_range(df.index.min(), df.index.max(), freq="D") if bars else pd.DatetimeIndex([])
            missing = int(len(exp.difference(df.index))) if bars else 0
            indicator_issues: List[str] = []
            if bars < max(self.config.strategy.momentum_lookbacks):
                indicator_issues.append("momentum_returns")
            if bars < self.config.strategy.momentum_medium_ma_window:
                indicator_issues.append("momentum_trend_ma")
            if bars < self.config.strategy.breakout_lookback:
                indicator_issues.append("breakout_high")
            if bars < self.config.strategy.breakout_volume_filter_window:
                indicator_issues.append("breakout_volume_ma")
            if bars < self.config.strategy.pair_correlation_window:
                indicator_issues.append("pair_correlation")
            if bars < self.config.risk.regime_slow_ma_window:
                indicator_issues.append("regime_ma")
            passes = bars >= required
            if not passes:
                ok = False
            out[s] = {
                "first_available_date": first,
                "last_available_date": last,
                "bars": int(bars),
                "missing_bar_count": missing,
                "passes_min_required_history": bool(passes),
                "status": "OK" if passes else "INSUFFICIENT_HISTORY",
                "indicators_may_be_unavailable": indicator_issues,
            }
        return ok, required, out

    @staticmethod
    def _empty_strategy_audit() -> Dict[str, Dict[str, object]]:
        return {
            "momentum": {"checked": 0, "passed": 0, "rejected": 0, "reasons": Counter()},
            "breakout": {"checked": 0, "passed": 0, "rejected": 0, "reasons": Counter()},
            "pair_trading": {"checked": 0, "passed": 0, "rejected": 0, "reasons": Counter()},
            "volatility": {"checked": 0, "passed": 0, "rejected": 0, "reasons": Counter()},
            "yield": {"checked": 0, "passed": 0, "rejected": 0, "reasons": Counter()},
        }

    def _audit_strategy_signals(
        self,
        market_data: Dict[str, pd.DataFrame],
        signal_date: pd.Timestamp,
        current_universe: Set[str],
        strategy_audit: Dict[str, Dict[str, object]],
        rejection_reasons: Counter,
        signals_generated: Counter,
    ) -> None:
        tradable = {s: df for s, df in market_data.items() if s in current_universe}
        if not tradable:
            rejection_reasons[NO_ELIGIBLE_ASSETS] += 1
            return
        self._audit_momentum(tradable, signal_date, strategy_audit, rejection_reasons, signals_generated)
        self._audit_breakout(tradable, signal_date, strategy_audit, rejection_reasons, signals_generated)
        self._audit_pair(tradable, signal_date, strategy_audit, rejection_reasons, signals_generated)
        self._audit_volatility(tradable, signal_date, strategy_audit, rejection_reasons, signals_generated)
        y = strategy_audit["yield"]
        y["checked"] += len(tradable)
        y["passed"] += len(tradable)
        signals_generated["yield"] += len(tradable)

    def _audit_momentum(self, tradable, date, strategy_audit, rejection_reasons, signals_generated) -> None:
        a = strategy_audit["momentum"]
        need = max(
            max(self.config.strategy.momentum_lookbacks),
            self.config.strategy.momentum_volume_filter_window,
            self.config.strategy.momentum_short_ma_window,
            self.config.strategy.momentum_medium_ma_window,
            30,
        )
        for _, (s, df) in enumerate(tradable.items()):
            a["checked"] += 1
            if date not in df.index:
                a["rejected"] += 1
                a["reasons"][INSUFFICIENT_HISTORY] += 1
                rejection_reasons[INSUFFICIENT_HISTORY] += 1
                continue
            idx = df.index.get_loc(date)
            if isinstance(idx, slice) or idx < need:
                a["rejected"] += 1
                a["reasons"][INSUFFICIENT_HISTORY] += 1
                rejection_reasons[INSUFFICIENT_HISTORY] += 1
                continue
            close = df["Close"].astype(float)
            vol = df["Volume"].astype(float)
            vol_ma = rolling_mean(vol, self.config.strategy.momentum_volume_filter_window).iloc[idx]
            if pd.isna(vol_ma) or float(vol.iloc[idx]) < float(vol_ma) * float(
                self.config.strategy.momentum_volume_filter_multiplier
            ):
                a["rejected"] += 1
                a["reasons"][VOLUME_FILTER_FAILED] += 1
                rejection_reasons[VOLUME_FILTER_FAILED] += 1
                continue
            ma_s = rolling_mean(close, self.config.strategy.momentum_short_ma_window).iloc[idx]
            ma_m = rolling_mean(close, self.config.strategy.momentum_medium_ma_window).iloc[idx]
            px = float(close.iloc[idx])
            if pd.isna(ma_s) or pd.isna(ma_m) or not (px > float(ma_s) and px > float(ma_m)):
                a["rejected"] += 1
                a["reasons"][TREND_FILTER_FAILED] += 1
                rejection_reasons[TREND_FILTER_FAILED] += 1
                continue
            r7 = returns(close, 7).iloc[idx]
            r30 = returns(close, 30).iloc[idx]
            if pd.isna(r7) or pd.isna(r30) or float(r7) <= 0 or float(r30) <= 0:
                a["rejected"] += 1
                a["reasons"][NO_MOMENTUM_SIGNAL] += 1
                rejection_reasons[NO_MOMENTUM_SIGNAL] += 1
                continue
            a["passed"] += 1
            signals_generated["momentum"] += 1

    def _audit_breakout(self, tradable, date, strategy_audit, rejection_reasons, signals_generated) -> None:
        a = strategy_audit["breakout"]
        lb = self.config.strategy.breakout_lookback
        vw = self.config.strategy.breakout_volume_filter_window
        tw = self.config.strategy.breakout_trend_ma_window
        need = max(lb, vw, tw)
        for _, (s, df) in enumerate(tradable.items()):
            a["checked"] += 1
            if date not in df.index:
                a["rejected"] += 1
                a["reasons"][INSUFFICIENT_HISTORY] += 1
                rejection_reasons[INSUFFICIENT_HISTORY] += 1
                continue
            idx = df.index.get_loc(date)
            if isinstance(idx, slice) or idx < need:
                a["rejected"] += 1
                a["reasons"][INSUFFICIENT_HISTORY] += 1
                rejection_reasons[INSUFFICIENT_HISTORY] += 1
                continue
            close = df["Close"].astype(float)
            vol = df["Volume"].astype(float)
            px = float(close.iloc[idx])
            prev_high = float(close.iloc[idx - lb : idx].max())
            vol_ma = rolling_mean(vol, vw).iloc[idx]
            vol_pass = pd.notna(vol_ma) and float(vol.iloc[idx]) >= float(vol_ma) * float(
                self.config.strategy.breakout_volume_filter_multiplier
            )
            if not vol_pass:
                a["rejected"] += 1
                a["reasons"][VOLUME_FILTER_FAILED] += 1
                rejection_reasons[VOLUME_FILTER_FAILED] += 1
                continue
            if self.config.strategy.breakout_require_trend_confirmation:
                tma = rolling_mean(close, tw).iloc[idx]
                if pd.isna(tma) or px <= float(tma):
                    a["rejected"] += 1
                    a["reasons"][TREND_FILTER_FAILED] += 1
                    rejection_reasons[TREND_FILTER_FAILED] += 1
                    continue
            vz = zscore(vol, lb).iloc[idx]
            if pd.isna(vz) or px <= prev_high or float(vz) < float(self.config.strategy.breakout_volume_spike_z):
                a["rejected"] += 1
                a["reasons"][NO_BREAKOUT_SIGNAL] += 1
                rejection_reasons[NO_BREAKOUT_SIGNAL] += 1
                continue
            a["passed"] += 1
            signals_generated["breakout"] += 1

    def _audit_pair(self, tradable, date, strategy_audit, rejection_reasons, signals_generated) -> None:
        a = strategy_audit["pair_trading"]
        syms = sorted(tradable.keys())
        a["checked"] += 1
        if len(syms) < 2:
            a["rejected"] += 1
            a["reasons"][NO_ELIGIBLE_ASSETS] += 1
            rejection_reasons[NO_ELIGIBLE_ASSETS] += 1
            return
        s1, s2 = syms[0], syms[1]
        d1, d2 = tradable[s1], tradable[s2]
        if date not in d1.index or date not in d2.index:
            a["rejected"] += 1
            a["reasons"][INSUFFICIENT_HISTORY] += 1
            rejection_reasons[INSUFFICIENT_HISTORY] += 1
            return
        idx = d1.index.get_loc(date)
        need = max(self.config.strategy.pair_window, self.config.strategy.pair_correlation_window)
        if isinstance(idx, slice) or idx < need:
            a["rejected"] += 1
            a["reasons"][INSUFFICIENT_HISTORY] += 1
            rejection_reasons[INSUFFICIENT_HISTORY] += 1
            return
        ratio = d1["Close"].astype(float) / d2["Close"].astype(float)
        corr = d1["Close"].pct_change().rolling(self.config.strategy.pair_correlation_window).corr(
            d2["Close"].pct_change()
        ).iloc[idx]
        ratio_vol = rolling_std(ratio, self.config.strategy.pair_correlation_window).iloc[idx]
        if pd.isna(corr) or pd.isna(ratio_vol):
            a["rejected"] += 1
            a["reasons"][INSUFFICIENT_HISTORY] += 1
            rejection_reasons[INSUFFICIENT_HISTORY] += 1
            return
        if float(corr) < float(self.config.strategy.pair_min_correlation) or float(ratio_vol) < float(
            self.config.strategy.pair_min_ratio_volatility
        ):
            a["rejected"] += 1
            a["reasons"][PAIR_CONDITION_NOT_MET] += 1
            rejection_reasons[PAIR_CONDITION_NOT_MET] += 1
            return
        z = zscore(ratio, self.config.strategy.pair_window).iloc[idx]
        if pd.isna(z) or abs(float(z)) < float(self.config.strategy.pair_z_entry):
            a["rejected"] += 1
            a["reasons"][PAIR_CONDITION_NOT_MET] += 1
            rejection_reasons[PAIR_CONDITION_NOT_MET] += 1
            return
        a["passed"] += 1
        signals_generated["pair_trading"] += 1

    def _audit_volatility(self, tradable, date, strategy_audit, rejection_reasons, signals_generated) -> None:
        a = strategy_audit["volatility"]
        w = self.config.strategy.volatility_window
        for _, (s, df) in enumerate(tradable.items()):
            a["checked"] += 1
            if date not in df.index:
                a["rejected"] += 1
                a["reasons"][INSUFFICIENT_HISTORY] += 1
                rejection_reasons[INSUFFICIENT_HISTORY] += 1
                continue
            idx = df.index.get_loc(date)
            if isinstance(idx, slice) or idx < w:
                a["rejected"] += 1
                a["reasons"][INSUFFICIENT_HISTORY] += 1
                rejection_reasons[INSUFFICIENT_HISTORY] += 1
                continue
            close = df["Close"].astype(float)
            vol = rolling_volatility(close, w).iloc[idx]
            mu = rolling_mean(close, w).iloc[idx]
            sd = rolling_std(close, w).iloc[idx]
            px = float(close.iloc[idx])
            if pd.isna(vol) or pd.isna(mu) or pd.isna(sd):
                a["rejected"] += 1
                a["reasons"][INSUFFICIENT_HISTORY] += 1
                rejection_reasons[INSUFFICIENT_HISTORY] += 1
                continue
            lower = float(mu) - float(self.config.strategy.volatility_band_k) * float(sd)
            upper = float(mu) + float(self.config.strategy.volatility_band_k) * float(sd)
            trig = (float(vol) < float(self.config.strategy.volatility_threshold)) and (px <= lower or px >= upper)
            if not trig:
                a["rejected"] += 1
                a["reasons"][VOLATILITY_CONDITION_FAILED] += 1
                rejection_reasons[VOLATILITY_CONDITION_FAILED] += 1
                continue
            a["passed"] += 1
            signals_generated["volatility"] += 1

    def _capture_signal_trace(
        self,
        rows: List[Dict[str, object]],
        strat_signals: Dict[str, Dict[str, float]],
        final_weights: Dict[str, float],
        signal_date: pd.Timestamp,
        regime_state: str,
        current_dd: float,
        current_universe: Set[str],
        blocked_code: str | None = None,
    ) -> None:
        for strat, sig in strat_signals.items():
            scale = float(self.config.strategy.strategy_weights.get(strat, 0.0))
            for sym, sw in sig.items():
                if abs(sw) <= 1e-12:
                    continue
                proposed = scale * float(sw)
                reason = SIGNAL_ACCEPTED
                status = "accepted"
                if blocked_code is not None:
                    reason = blocked_code
                    status = "rejected"
                elif sym not in current_universe:
                    reason = NO_ELIGIBLE_ASSETS
                    status = "rejected"
                elif abs(final_weights.get(sym, 0.0)) <= 1e-12:
                    status = "rejected"
                    if regime_state != "bull" and self.config.risk.regime_block_new_noncore_positions:
                        reason = REGIME_FILTER_BLOCKED
                    elif current_dd <= -self.config.risk.drawdown_pause_adds_level:
                        reason = DRAWDOWN_PROTECTION_BLOCKED
                    else:
                        reason = ALLOCATION_CAP_BLOCKED
                rows.append(
                    {
                        "signal_date": signal_date,
                        "strategy": strat,
                        "target_asset": sym,
                        "signal_generated": 1,
                        "proposed_allocation": proposed,
                        "final_target_weight": float(final_weights.get(sym, 0.0)),
                        "status": status,
                        "reason_code": reason,
                        "execution_result": EXECUTION_SKIPPED_ZERO_SIZE,
                    }
                )

    @staticmethod
    def _mark_trace_execution(rows: List[Dict[str, object]], exec_date: pd.Timestamp, traded_symbols: Set[str]) -> None:
        if not rows:
            return
        for r in rows:
            if r.get("execution_result") != EXECUTION_SKIPPED_ZERO_SIZE:
                continue
            if r.get("target_asset") in traded_symbols:
                r["execution_result"] = EXECUTED

    @staticmethod
    def _summarize_strategy_audit(strategy_audit: Dict[str, Dict[str, object]]) -> Dict[str, Dict[str, object]]:
        out: Dict[str, Dict[str, object]] = {}
        for strat, vals in strategy_audit.items():
            reasons: Counter = vals["reasons"]
            top_reason = reasons.most_common(1)[0][0] if reasons else ""
            out[strat] = {
                "checked": int(vals["checked"]),
                "passed": int(vals["passed"]),
                "rejected": int(vals["rejected"]),
                "top_rejected_reason": top_reason,
            }
        return out

    def _flat_status(
        self,
        is_flat: bool,
        symbols_loaded: int,
        active_universe: Set[str],
        history_ok: bool,
        risk_suppression: bool,
        signals_generated: int,
        trades_executed: int,
        rejection_reasons: Counter,
    ) -> Tuple[str, str]:
        if not is_flat:
            return "NOT_FLAT", "Positions are active."
        if symbols_loaded == 0 or len(active_universe) == 0:
            return "FLAT_DATA_OR_UNIVERSE_ISSUE", "No symbols or no active universe."
        if not history_ok:
            return "FLAT_INSUFFICIENT_HISTORY", "Insufficient indicator history for broad signal generation."
        if risk_suppression:
            return "FLAT_BLOCKED_BY_RISK", "Risk controls are suppressing entries."
        if signals_generated == 0:
            return "FLAT_VALID_NO_SIGNALS", "No valid strategy opportunities met current rules."
        if signals_generated > 0 and trades_executed == 0 and not rejection_reasons:
            return "FLAT_POSSIBLE_BUG", "Signals passed but no trades/rejections recorded."
        return "FLAT_VALID_NO_SIGNALS", "Signals did not survive filters/allocation constraints."

    def _positions_snapshot(self, prices: Dict[str, float]) -> str:
        rows: List[str] = []
        for symbol, pos in sorted(self.engine.portfolio.state.positions.items()):
            px = prices.get(symbol, 0.0)
            mv = pos.quantity * px
            rows.append(
                f"{symbol}: qty={pos.quantity:.6f}, avg={pos.avg_entry_price:.6f}, mv={mv:.2f}"
            )
        return " | ".join(rows) if rows else "FLAT"


def main() -> None:
    config = Config()
    root = Path(__file__).resolve().parent
    data_manager = DataManager(config)
    symbol_to_file = _main_module.build_symbol_file_map(config, root)
    if not symbol_to_file:
        raise RuntimeError("No CSV files found. Add OHLCV CSV files to project root.")

    market_data = data_manager.load_multi_csv(symbol_to_file)
    # Keep only USD-quoted symbols — non-USD assets trade on different calendars
    # and collapse the date intersection to zero (same filter the backtest applies).
    allowed_quote = config.data.allowed_quote_currency
    overrides = set(config.data.symbol_quote_overrides.keys())
    market_data = {
        sym: df for sym, df in market_data.items()
        if data_manager.detect_quote_currency(sym) == allowed_quote or sym in overrides
    }
    market_data = data_manager.align_on_common_dates(market_data, drop_insufficient=True)
    tester = ForwardTester(config)
    result = tester.run(market_data)

    equity_path = root / "forward_equity.csv"
    trades_path = root / "forward_trades.csv"
    positions_path = root / "forward_positions.csv"
    trace_path = root / "forward_signal_trace.csv"
    audit_path = root / "forward_audit.json"
    audit_hist_path = root / "forward_audit_history.jsonl"
    coverage_path = root / "data_coverage_report.csv"
    mode_cmp_path = root / "forward_mode_comparison.json"

    result.equity_log.to_csv(equity_path, index=False)
    result.trade_log.to_csv(trades_path, index=False)
    result.position_log.to_csv(positions_path, index=False)
    result.signal_trace_log.to_csv(trace_path, index=False)

    coverage_df = _build_data_coverage_report(result.audit, config)
    coverage_df.to_csv(coverage_path, index=False)

    with audit_path.open("w", encoding="utf-8") as f:
        json.dump(result.audit, f, indent=2, default=str)
    with audit_hist_path.open("a", encoding="utf-8") as f:
        compact = {
            "run_timestamp_utc": result.audit.get("run_timestamp_utc"),
            "forward_mode": result.audit.get("forward_mode"),
            "window_start": result.audit.get("forward_window_start"),
            "window_end": result.audit.get("forward_window_end"),
            "equity": result.audit.get("current_equity"),
            "daily_return": result.audit.get("daily_return"),
            "drawdown": result.audit.get("drawdown"),
            "max_drawdown_to_date": result.audit.get("max_drawdown_to_date"),
            "is_flat": result.audit.get("is_flat"),
            "flat_status": result.audit.get("flat_status"),
            "risk_suppression_flag": result.audit.get("risk_suppression_flag"),
            "top3_no_entry_reasons": result.audit.get("top3_no_entry_reasons"),
            "trades_executed": result.audit.get("trades_executed"),
            "active_universe_size": result.audit.get("active_universe_size"),
            "history_sufficiency_status": result.audit.get("history_sufficiency_status"),
        }
        f.write(json.dumps(compact, default=str) + "\n")

    mode_cmp = _run_mode_comparison(base_config=config, market_data=market_data)
    with mode_cmp_path.open("w", encoding="utf-8") as f:
        json.dump(mode_cmp, f, indent=2, default=str)

    a = result.audit
    print("Forward Test Snapshot")
    print(f"Forward Mode: {a.get('forward_mode')}")
    if a.get("bootstrap_note"):
        print(f"Bootstrap: {a.get('bootstrap_note')}")
    print(f"Execution Timing: {a.get('execution_timing_assumption')}")
    print(f"Current Equity: {a.get('current_equity', 0.0):,.2f}")
    print(f"Daily Return: {float(a.get('daily_return', 0.0)):.2%}")
    print(f"Drawdown: {float(a.get('drawdown', 0.0)):.2%}")
    print(f"Max Drawdown to Date: {float(a.get('max_drawdown_to_date', 0.0)):.2%}")
    print(f"Current Positions: {a.get('current_positions')}")

    print(f"\nFLAT Status: {a.get('flat_status')}")
    print(f"FLAT Explanation: {a.get('flat_state_explanation')}")
    if not a.get("history_sufficiency_status", True):
        print("INSUFFICIENT HISTORY FOR FULL SIGNAL GENERATION")
    print(f"Symbols Loaded: {a.get('symbols_loaded')}")
    print(f"Active Universe ({a.get('active_universe_size')}): {a.get('active_universe')}")
    print(f"Trades Executed: {a.get('trades_executed')}")
    print(f"Signals Generated by Strategy: {a.get('strategy_signal_counts')}")
    print(f"Signals Rejected: {a.get('signals_rejected')}")
    print(f"Risk Suppression Flag: {a.get('risk_suppression_flag')}")
    print(f"Top 3 No-entry Reasons: {a.get('top3_no_entry_reasons')}")
    _print_coverage_summary(coverage_df)
    recommendation = _next_step_recommendation(audit=a, coverage_df=coverage_df)
    print(f"\nNext-step Recommendation: {recommendation}")

    print("\nForward Mode Comparison")
    for mode in ["fresh", "carry_backtest"]:
        m = mode_cmp.get(mode, {})
        print(
            f"{mode}: equity={float(m.get('current_equity', 0.0)):,.2f}, "
            f"drawdown={float(m.get('drawdown', 0.0)):.2%}, "
            f"trades={int(m.get('trades_executed', 0))}, "
            f"active_universe={m.get('active_universe')}, "
            f"positions={m.get('current_positions')}"
        )
        print(f"  top_no_entry_reasons={m.get('top3_no_entry_reasons')}")

    if config.forward.forward_verbose_audit:
        print("\nSignal Audit")
        for strat, vals in (a.get("strategy_audit_results") or {}).items():
            print(
                f"{strat}: checked={vals.get('checked')}, passed={vals.get('passed')}, "
                f"rejected={vals.get('rejected')}, top_rejected_reason={vals.get('top_rejected_reason')}"
            )
        print("\nRejection Reason Breakdown")
        for k, v in sorted((a.get("rejection_reason_breakdown") or {}).items(), key=lambda x: x[1], reverse=True):
            print(f"- {k}: {v}")
        print("\nPer-symbol History Diagnostics")
        for sym, info in (a.get("per_symbol_history_diagnostics") or {}).items():
            print(
                f"{sym}: bars={info.get('bars')}, first={info.get('first_available_date')}, "
                f"last={info.get('last_available_date')}, missing={info.get('missing_bar_count')}, "
                f"status={info.get('status')}"
            )

    if a.get("warnings"):
        print("\nWarnings")
        for w in a["warnings"]:
            print(f"- {w}")

    print(
        f"\nSaved: {equity_path.name}, {trades_path.name}, {positions_path.name}, "
        f"{trace_path.name}, {audit_path.name}, {audit_hist_path.name}, "
        f"{coverage_path.name}, {mode_cmp_path.name}"
    )


def _build_data_coverage_report(audit: Dict[str, object], config: Config) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    per_symbol = audit.get("per_symbol_history_diagnostics", {}) or {}
    active = set(audit.get("active_universe", []) or [])
    require_full = bool(config.forward.forward_require_full_history)
    for symbol, info in sorted(per_symbol.items()):
        passes = bool(info.get("passes_min_required_history", False))
        included = symbol in active
        if included:
            excluded_reason = ""
        elif require_full and not passes:
            excluded_reason = INSUFFICIENT_HISTORY
        elif not active:
            excluded_reason = NO_ELIGIBLE_ASSETS
        else:
            excluded_reason = "UNIVERSE_FILTER_EXCLUDED"
        rows.append(
            {
                "symbol": symbol,
                "first_date": info.get("first_available_date", ""),
                "last_date": info.get("last_available_date", ""),
                "bars": int(info.get("bars", 0)),
                "missing_dates": int(info.get("missing_bar_count", 0)),
                "passes_min_history": passes,
                "included_in_active_universe": included,
                "excluded_reason": excluded_reason,
                "status": info.get("status", ""),
            }
        )
    return pd.DataFrame(rows)


def _print_coverage_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("\nData Coverage Summary: no symbols")
        return
    total = len(df)
    pass_hist = int(df["passes_min_history"].sum())
    active = int(df["included_in_active_universe"].sum())
    insuff = int((df["status"] == "INSUFFICIENT_HISTORY").sum())
    print("\nData Coverage Summary")
    print(
        f"Symbols={total} | PassMinHistory={pass_hist}/{total} | "
        f"ActiveUniverse={active}/{total} | InsufficientHistory={insuff}/{total}"
    )


def _run_mode_comparison(base_config: Config, market_data: Dict[str, pd.DataFrame]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for mode in ["fresh", "carry_backtest"]:
        cfg = deepcopy(base_config)
        cfg.forward.forward_start_mode = mode
        tester = ForwardTester(cfg)
        res = tester.run(market_data)
        a = res.audit
        out[mode] = {
            "forward_mode": mode,
            "active_universe": a.get("active_universe"),
            "current_equity": a.get("current_equity"),
            "current_positions": a.get("current_positions"),
            "drawdown": a.get("drawdown"),
            "trades_executed": a.get("trades_executed"),
            "top3_no_entry_reasons": a.get("top3_no_entry_reasons"),
        }
    return out


def _next_step_recommendation(audit: Dict[str, object], coverage_df: pd.DataFrame) -> str:
    total = len(coverage_df)
    insuff = int((coverage_df["status"] == "INSUFFICIENT_HISTORY").sum()) if total > 0 else 0
    insuff_ratio = (insuff / total) if total > 0 else 1.0
    if insuff_ratio > 0.20:
        return "DATA_FIRST"
    if bool(audit.get("risk_suppression_flag", False)):
        return "RISK_REVIEW"

    reasons = audit.get("rejection_reason_breakdown", {}) or {}
    filter_codes = {
        VOLATILITY_CONDITION_FAILED,
        VOLUME_FILTER_FAILED,
        TREND_FILTER_FAILED,
        NO_BREAKOUT_SIGNAL,
        NO_MOMENTUM_SIGNAL,
        PAIR_CONDITION_NOT_MET,
    }
    total_rej = sum(int(v) for v in reasons.values()) if reasons else 0
    filter_rej = sum(int(reasons.get(k, 0)) for k in filter_codes)
    if total_rej > 0 and (filter_rej / total_rej) > 0.5:
        return "STRATEGY_REVIEW"

    trades = int(audit.get("trades_executed", 0))
    active_u = int(audit.get("active_universe_size", 0))
    if trades > 0 and active_u > 0:
        return "READY_FOR_PAPER_TRACKING"
    return "DATA_FIRST"


if __name__ == "__main__":
    main()
