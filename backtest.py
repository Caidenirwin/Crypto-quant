"""
Backtest engine:
- Combines strategy weights
- Applies risk controls
- Rebalances portfolio
- Produces performance metrics
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Dict, List, Set

import pandas as pd

from config import Config
from data import DataManager
from indicators import rolling_mean, rolling_volatility
from portfolio import PortfolioManager
from risk import RiskManager
from strategies.breakout import BreakoutStrategy
from strategies.meme_momentum import MemeMomentumStrategy
from strategies.momentum import MomentumStrategy
from strategies.pair_trading import PairTradingStrategy
from strategies.rsi_dip import RSIDipStrategy
from strategies.sector_rotation import SectorRotationStrategy
from strategies.volatility import VolatilityStrategy
from utils import annualize_return, clamp, max_drawdown, normalize_weights, sharpe_ratio


@dataclass
class BacktestResult:
    total_return: float
    cagr: float
    sharpe: float
    max_drawdown: float
    equity_curve: pd.Series
    daily_returns: pd.Series
    diagnostics: Dict[str, object]
    warnings: List[str]


class BacktestEngine:
    def __init__(self, config: Config):
        self.config = config
        self.data_manager = DataManager(config)
        self.portfolio = PortfolioManager(config)
        self.risk = RiskManager(config)
        YieldStrategy = import_module("strategies.yield").YieldStrategy
        self.strategies = {
            "momentum":        MomentumStrategy(config),
            "rsi_dip":         RSIDipStrategy(config),
            "sector_rotation": SectorRotationStrategy(config),
            "pair_trading":    PairTradingStrategy(config),
            "volatility":      VolatilityStrategy(config),
            "breakout":        BreakoutStrategy(config),
            "yield":           YieldStrategy(config),
        }
        self.meme_strategy = MemeMomentumStrategy(config)
        self.meme_rug_exits: List[Dict[str, object]] = []

    def run(self, market_data: Dict[str, pd.DataFrame]) -> BacktestResult:
        if not market_data:
            raise ValueError("market_data is empty")
        if self.config.backtest.execution_timing != "signal_close_execute_next_open":
            raise ValueError("Unsupported execution_timing. Use signal_close_execute_next_open.")

        common_dates = self._common_dates(market_data)
        if len(common_dates) < 2:
            raise ValueError("Need at least 2 aligned bars for next-open execution timing.")

        # Timing convention:
        # - Generate signals using bar t close data.
        # - Execute orders on bar t+1 open with slippage/fees.
        # - Mark portfolio at bar t+1 close for PnL/equity.
        first_date = common_dates[0]
        first_close = {sym: float(df.loc[first_date, "Close"]) for sym, df in market_data.items()}

        active_sleeves = self._active_enabled_strategies()
        equity_points: List[float] = [self.portfolio.mark_to_market(first_close)]
        date_points: List[pd.Timestamp] = [first_date]
        daily_ret_values: List[float] = [0.0]
        exposure_values: List[float] = [0.0]
        strategy_pnl_proxy: Dict[str, float] = {k: 0.0 for k in active_sleeves}
        current_universe: Set[str] = set(market_data.keys())
        last_universe_refresh_date: pd.Timestamp | None = None

        for i in range(1, len(common_dates)):
            signal_date = common_dates[i - 1]
            exec_date = common_dates[i]
            signal_close = {sym: float(df.loc[signal_date, "Close"]) for sym, df in market_data.items()}
            exec_open = {sym: float(df.loc[exec_date, "Open"]) for sym, df in market_data.items()}
            exec_close = {sym: float(df.loc[exec_date, "Close"]) for sym, df in market_data.items()}

            signal_equity = self.portfolio.mark_to_market(signal_close)
            if "yield" in active_sleeves and self.config.strategy.yield_enabled:
                y = self.strategies["yield"]
                y.accrue_yield(self.portfolio.holdings_value(signal_close))
                self.portfolio.state.cash += y.conversion_amount(signal_date)

            current_dd = self._current_drawdown(equity_points, signal_equity)
            self.risk.enforce_drawdown_guard(current_dd)
            self.risk.maybe_resume(current_dd)

            if self._should_refresh_universe(signal_date, last_universe_refresh_date):
                current_universe = self._select_universe(market_data, signal_date)
                last_universe_refresh_date = signal_date

            # Diagnostics: realized next-bar contribution proxy per strategy sleeve.
            strat_weights = self._strategy_signal_weights(
                market_data, signal_date, eligible_symbols=current_universe
            )
            next_bar_rets = {
                sym: (exec_close[sym] / signal_close[sym] - 1.0)
                for sym in signal_close.keys()
                if signal_close[sym] > 0 and sym in exec_close
            }
            effective_sleeve_w = self._effective_strategy_weights(list(strat_weights.keys()))
            for strat_name, sw in strat_weights.items():
                sleeve_ret = sum(sw.get(sym, 0.0) * next_bar_rets.get(sym, 0.0) for sym in sw.keys())
                scale = effective_sleeve_w.get(strat_name, 0.0)
                strategy_pnl_proxy[strat_name] = strategy_pnl_proxy.get(strat_name, 0.0) + (
                    signal_equity * scale * sleeve_ret
                )

            # Evaluate exits on signal_date close; execute on next open.
            self._apply_meme_rug_exits(market_data, signal_date, exec_open, exec_date)
            self._apply_stop_losses(signal_close, exec_open, exec_date)
            self._apply_take_profits(market_data, signal_date, signal_close, exec_open, exec_date)
            self._apply_trend_break_exits(market_data, signal_date, exec_open, exec_date)

            if self.risk.trading_enabled and self.portfolio.should_rebalance(signal_date):
                weights = self._target_weights_with_risk_overlays(
                    market_data,
                    signal_date,
                    current_dd,
                    eligible_symbols=current_universe,
                )
                self.portfolio.rebalance(weights, exec_open, exec_date)

            end_equity = self.portfolio.mark_to_market(exec_close)
            prev_equity = equity_points[-1]
            day_ret = (end_equity / prev_equity - 1.0) if prev_equity > 0 else 0.0

            equity_points.append(end_equity)
            date_points.append(exec_date)
            daily_ret_values.append(day_ret)
            gross_exp = self.portfolio.gross_exposure(exec_close)
            exposure_values.append(gross_exp / end_equity if end_equity > 0 else 0.0)

        equity_curve = pd.Series(equity_points, index=date_points, name="equity")
        daily_returns = pd.Series(daily_ret_values, index=date_points, name="daily_return")
        total_ret = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0 if len(equity_curve) > 1 else 0.0
        diagnostics: Dict[str, object] = {}
        if self.config.backtest.include_diagnostics:
            diagnostics = self._diagnostics(
                equity_curve=equity_curve,
                exposure_series=pd.Series(exposure_values, index=date_points, name="gross_exposure"),
                end_prices={sym: float(df.loc[date_points[-1], "Close"]) for sym, df in market_data.items()},
                strategy_pnl_proxy=strategy_pnl_proxy,
            )
        warnings = self._audit_warnings(market_data)

        return BacktestResult(
            total_return=float(total_ret),
            cagr=annualize_return(
                total_return=float(total_ret),
                periods=len(equity_curve),
                annualization_days=self.config.backtest.annualization_days,
            ),
            sharpe=sharpe_ratio(
                daily_returns, annualization_days=self.config.backtest.annualization_days
            ),
            max_drawdown=max_drawdown(equity_curve),
            equity_curve=equity_curve,
            daily_returns=daily_returns,
            diagnostics=diagnostics,
            warnings=warnings,
        )

    def _combined_strategy_weights(
        self,
        market_data: Dict[str, pd.DataFrame],
        date: pd.Timestamp,
        eligible_symbols: Set[str] | None = None,
    ) -> Dict[str, float]:
        combined: Dict[str, float] = {sym: 0.0 for sym in market_data.keys()}
        strat_weights = self._strategy_signal_weights(
            market_data, date, eligible_symbols=eligible_symbols
        )
        weights = self._effective_strategy_weights(list(strat_weights.keys()))
        for name, signal_w in strat_weights.items():
            scale = weights.get(name, 0.0)
            for sym in combined.keys():
                combined[sym] += scale * signal_w.get(sym, 0.0)
        return normalize_weights(combined)

    def _effective_strategy_weights(self, available: List[str]) -> Dict[str, float]:
        """
        Build sleeve weights with optional stability penalties, diversification blend, and cap.
        """
        raw = {
            name: max(0.0, float(self.config.strategy.strategy_weights.get(name, 0.0)))
            for name in available
        }
        if not self.config.strategy.yield_enabled and "yield" in raw:
            raw["yield"] = 0.0
        if not raw:
            return {}

        penalty_map = self.config.strategy.sleeve_stability_penalties or {}
        penalized = {
            name: raw[name] * clamp(float(penalty_map.get(name, 1.0)), 0.0, 1.0)
            for name in raw
        }
        weights = normalize_weights(penalized if sum(penalized.values()) > 0 else raw)
        if not weights:
            return {}

        blend = clamp(float(self.config.strategy.sleeve_diversification_blend), 0.0, 1.0)
        if blend > 0 and len(weights) > 1:
            eq = 1.0 / float(len(weights))
            weights = {k: (1.0 - blend) * v + blend * eq for k, v in weights.items()}
            weights = normalize_weights(weights)

        if self.config.strategy.enforce_sleeve_caps:
            cap = float(self.config.strategy.sleeve_max_weight)
            if cap > 0:
                weights = self._cap_weights(weights, cap)

        # Keep yield sleeve conservative to reduce fragility.
        if "yield" in weights:
            ycap = max(0.0, float(self.config.strategy.yield_max_allocation))
            if weights["yield"] > ycap:
                excess = weights["yield"] - ycap
                weights["yield"] = ycap
                others = [k for k in weights.keys() if k != "yield"]
                other_total = sum(weights[k] for k in others)
                if other_total > 1e-12:
                    for k in others:
                        weights[k] += excess * (weights[k] / other_total)

        if "momentum" in weights:
            mcap = max(0.0, float(self.config.strategy.momentum_max_allocation))
            if weights["momentum"] > mcap:
                excess = weights["momentum"] - mcap
                weights["momentum"] = mcap
                others = [k for k in weights.keys() if k != "momentum"]
                other_total = sum(weights[k] for k in others)
                if other_total > 1e-12:
                    for k in others:
                        weights[k] += excess * (weights[k] / other_total)

        if "pair_trading" in weights:
            pcap = max(0.0, float(self.config.strategy.pair_max_allocation))
            if weights["pair_trading"] > pcap:
                excess = weights["pair_trading"] - pcap
                weights["pair_trading"] = pcap
                others = [k for k in weights.keys() if k != "pair_trading"]
                other_total = sum(weights[k] for k in others)
                if other_total > 1e-12:
                    for k in others:
                        weights[k] += excess * (weights[k] / other_total)

        if "breakout" in weights:
            bcap = max(0.0, float(self.config.strategy.breakout_max_allocation))
            if weights["breakout"] > bcap:
                excess = weights["breakout"] - bcap
                weights["breakout"] = bcap
                others = [k for k in weights.keys() if k != "breakout"]
                other_total = sum(weights[k] for k in others)
                if other_total > 1e-12:
                    for k in others:
                        weights[k] += excess * (weights[k] / other_total)
        return normalize_weights(weights)

    @staticmethod
    def _cap_weights(weights: Dict[str, float], cap: float) -> Dict[str, float]:
        out = dict(weights)
        cap = max(0.0, float(cap))
        if cap <= 0 or not out:
            return normalize_weights(out)

        # Iteratively clip oversized sleeves and redistribute to uncapped sleeves.
        for _ in range(6):
            over = {k: v - cap for k, v in out.items() if v > cap}
            if not over:
                break
            excess = sum(over.values())
            for k in over:
                out[k] = cap
            recipients = [k for k, v in out.items() if v < cap - 1e-12]
            recv_total = sum(out[k] for k in recipients)
            if recv_total <= 1e-12:
                break
            for k in recipients:
                out[k] += excess * (out[k] / recv_total)
        return normalize_weights(out)

    def _strategy_signal_weights(
        self,
        market_data: Dict[str, pd.DataFrame],
        date: pd.Timestamp,
        eligible_symbols: Set[str] | None = None,
    ) -> Dict[str, Dict[str, float]]:
        enabled = self._active_enabled_strategies()
        tradable = set(market_data.keys()) if eligible_symbols is None else set(eligible_symbols)
        tradable_data = {
            s: df for s, df in market_data.items() if s in tradable and not self.config.is_meme(s)
        }
        out: Dict[str, Dict[str, float]] = {}
        if not tradable_data:
            return out
        for name in enabled:
            strat = self.strategies.get(name)
            if strat is None:
                continue
            signal_w = strat.generate_weights(tradable_data, date)
            out[name] = signal_w
        return out

    def _active_enabled_strategies(self) -> List[str]:
        enabled = list(self.config.strategy.enabled_strategies)
        toggles = {
            "momentum":        bool(self.config.strategy.momentum_enabled),
            "rsi_dip":         bool(self.config.strategy.rsi_dip_enabled),
            "sector_rotation": bool(self.config.strategy.sector_rotation_enabled),
            "breakout":        bool(self.config.strategy.breakout_enabled),
            "pair_trading":    bool(self.config.strategy.pair_trading_enabled),
            "volatility":      bool(self.config.strategy.volatility_enabled),
            "yield":           bool(self.config.strategy.yield_enabled),
        }
        return [s for s in enabled if toggles.get(s, True)]

    def _split_meme_data(
        self, market_data: Dict[str, pd.DataFrame]
    ) -> tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        core = {s: df for s, df in market_data.items() if not self.config.is_meme(s)}
        memes = {s: df for s, df in market_data.items() if self.config.is_meme(s)}
        return core, memes

    def _target_weights_with_risk_overlays(
        self,
        market_data: Dict[str, pd.DataFrame],
        date: pd.Timestamp,
        current_drawdown: float,
        eligible_symbols: Set[str] | None = None,
    ) -> Dict[str, float]:
        core_data, meme_data = self._split_meme_data(market_data)
        core_eligible = (
            None if eligible_symbols is None else {s for s in eligible_symbols if s in core_data}
        )
        target = {sym: 0.0 for sym in market_data.keys()}
        target.update(
            self._core_target_weights(core_data, date, current_drawdown, eligible_symbols=core_eligible)
        )
        meme_alloc = self._meme_target_weights(meme_data, date, current_drawdown)
        used = sum(meme_alloc.values())
        if used > 0:
            # Fund memes from idle cash first; only trim core if the book would exceed ~100%.
            core_sum = sum(max(0.0, w) for w in target.values())
            headroom = 0.98 - used
            if core_sum > headroom and core_sum > 0:
                scale = max(0.0, headroom) / core_sum
                target = {s: w * scale for s, w in target.items()}
            target.update(meme_alloc)
        return target

    def _meme_target_weights(
        self,
        meme_data: Dict[str, pd.DataFrame],
        date: pd.Timestamp,
        current_drawdown: float,
    ) -> Dict[str, float]:
        mc = self.config.meme
        if not mc.meme_enabled or not meme_data:
            return {}
        # No meme risk while the portfolio is already in a meaningful drawdown.
        if current_drawdown <= -self.config.risk.drawdown_pause_adds_level:
            return {s: 0.0 for s in meme_data}
        signal = self.meme_strategy.generate_weights(meme_data, date)
        budget = max(0.0, float(mc.meme_max_allocation))
        per_coin = max(0.0, float(mc.meme_max_per_coin))
        return {s: min(w * budget, per_coin) for s, w in signal.items()}

    def _core_target_weights(
        self,
        market_data: Dict[str, pd.DataFrame],
        date: pd.Timestamp,
        current_drawdown: float,
        eligible_symbols: Set[str] | None = None,
    ) -> Dict[str, float]:
        combined = self._combined_strategy_weights(
            market_data, date, eligible_symbols=eligible_symbols
        )
        combined = self.risk.apply_inverse_vol_sizing(combined, market_data, date)
        combined = self.risk.apply_asset_stability_penalties(combined)

        # Long-only target for this portfolio model.
        core_assets = [s for s in self.config.portfolio.core_assets if s in market_data]
        all_symbols = set(market_data.keys())
        non_core_assets = sorted(all_symbols - set(core_assets))
        active_weights = {s: combined.get(s, 0.0) for s in non_core_assets}

        regime_state = self._regime_state(market_data, date, eligible_symbols=eligible_symbols)
        core_target = self._core_target_by_regime(regime_state)
        active_multiplier = self._active_multiplier_by_regime(regime_state)
        if self._universe_trend_negative(market_data, date, eligible_symbols):
            active_multiplier *= 0.5

        pause_new_aggressive_adds = current_drawdown <= -self.config.risk.drawdown_pause_adds_level
        if current_drawdown <= -self.config.risk.drawdown_level1:
            active_multiplier *= self.config.risk.drawdown_level1_risk_multiplier
        if current_drawdown <= -self.config.risk.drawdown_level2:
            active_multiplier *= self.config.risk.drawdown_level2_risk_multiplier
        if current_drawdown <= -self.config.risk.drawdown_level3:
            active_multiplier *= self.config.risk.drawdown_level3_risk_multiplier
            core_target = max(core_target, self.config.portfolio.core_extreme_defensive_allocation)

        active_budget = (1.0 - core_target) * max(0.0, active_multiplier)
        active_alloc = normalize_weights(active_weights)
        active_alloc = {s: active_alloc.get(s, 0.0) * active_budget for s in non_core_assets}
        quality_snapshot = self.data_manager.universe_quality_snapshot(market_data, date)
        active_alloc = self._apply_universe_quality_weighting(active_alloc, quality_snapshot)
        active_alloc = self._blend_active_equal_weight(active_alloc)

        if pause_new_aggressive_adds or (
            regime_state != "bull" and self.config.risk.regime_block_new_noncore_positions
        ):
            held = self._currently_held_non_core_symbols(set(core_assets))
            for sym in non_core_assets:
                if sym not in held:
                    active_alloc[sym] = 0.0

        core_alloc_value = max(0.0, core_target)
        core_alloc: Dict[str, float] = {}
        if core_assets:
            core_alloc = self._core_bucket_allocation(market_data, core_assets, date, core_alloc_value)

        target = {sym: 0.0 for sym in market_data.keys()}
        for sym, w in active_alloc.items():
            target[sym] = w
        for sym, w in core_alloc.items():
            target[sym] = target.get(sym, 0.0) + w

        target = self.risk.apply_asset_caps(target)
        target = self._apply_dynamic_high_risk_group_cap(
            target=target,
            market_data=market_data,
            date=date,
            core_assets=set(core_assets),
        )
        target = self.risk.apply_position_limits(target)
        target = self._restore_min_core_after_caps(target, set(core_assets), core_target)
        target = self._apply_fixed_hold_allocations(target, market_data)
        return target

    def _apply_fixed_hold_allocations(
        self, target: Dict[str, float], market_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, float]:
        """
        Reserve configured minimum portfolio slices for specific symbols (if available).
        This is structural allocation logic, not a strategy signal.
        """
        if not bool(self.config.portfolio.fixed_hold_enabled):
            return dict(target)

        raw_holds = dict(self.config.portfolio.fixed_hold_allocations or {})
        if not raw_holds:
            return dict(target)

        resolved_holds: Dict[str, float] = {}
        symbols = set(market_data.keys())
        for configured_sym, w in raw_holds.items():
            wv = max(0.0, float(w))
            if wv <= 0:
                continue
            resolved = self._resolve_symbol_key(configured_sym, symbols)
            if resolved is None:
                continue
            resolved_holds[resolved] = resolved_holds.get(resolved, 0.0) + wv

        if not resolved_holds:
            return dict(target)

        fixed_total = sum(resolved_holds.values())
        if fixed_total >= 0.999:
            # Keep a tiny non-fixed budget for operational flexibility.
            scale = 0.999 / fixed_total
            resolved_holds = {s: w * scale for s, w in resolved_holds.items()}
            fixed_total = sum(resolved_holds.values())

        out = dict(target)
        non_fixed_symbols = [s for s in out.keys() if s not in resolved_holds]
        non_fixed_total = sum(max(0.0, out.get(s, 0.0)) for s in non_fixed_symbols)
        remaining_budget = max(0.0, 1.0 - fixed_total)

        if non_fixed_total > 1e-12 and remaining_budget >= 0:
            for s in non_fixed_symbols:
                out[s] = max(0.0, out.get(s, 0.0)) * (remaining_budget / non_fixed_total)
        else:
            for s in non_fixed_symbols:
                out[s] = 0.0

        for s, w in resolved_holds.items():
            out[s] = max(float(w), out.get(s, 0.0))

        return normalize_weights(out)

    @staticmethod
    def _resolve_symbol_key(configured_sym: str, available_symbols: Set[str]) -> str | None:
        """
        Resolve flexible config names like XRP, OP, SOL to actual universe keys such as
        XRP, OP(USD), SOL(USD), etc.
        """
        key = str(configured_sym).strip().upper()
        if key in available_symbols:
            return key
        # Try base-token matching against symbols like TOKEN(USD).
        for s in sorted(available_symbols):
            su = str(s).strip().upper()
            base = su.split("(")[0].strip() if "(" in su else su
            if base == key:
                return s
        return None

    def _universe_trend_negative(
        self,
        market_data: Dict[str, pd.DataFrame],
        date: pd.Timestamp,
        eligible_symbols: Set[str] | None = None,
    ) -> bool:
        if not self.config.risk.universe_trend_filter_enabled:
            return False
        ma_w = int(self.config.risk.universe_trend_ma_window)
        symbols = sorted(eligible_symbols) if eligible_symbols is not None else sorted(market_data.keys())
        if not symbols:
            return False
        trend_scores: List[int] = []
        for s in symbols:
            df = market_data.get(s)
            if df is None or date not in df.index:
                continue
            idx = df.index.get_loc(date)
            if isinstance(idx, slice) or idx < ma_w:
                continue
            px = float(df.loc[date, "Close"])
            ma = rolling_mean(df["Close"].astype(float), ma_w).iloc[idx]
            if pd.isna(ma):
                continue
            trend_scores.append(1 if px > float(ma) else 0)
        if not trend_scores:
            return False
        return (sum(trend_scores) / float(len(trend_scores))) < 0.5

    def _apply_meme_rug_exits(
        self,
        market_data: Dict[str, pd.DataFrame],
        signal_date: pd.Timestamp,
        execution_prices: Dict[str, float],
        execution_date: pd.Timestamp,
    ) -> None:
        """Anti-rug: flatten any held meme whose history check flags a crash/drain (daily, not just on rebalance)."""
        held = {
            s: market_data[s]
            for s, pos in self.portfolio.state.positions.items()
            if pos.quantity > 0 and self.config.is_meme(s) and s in market_data
        }
        if not held:
            return
        for symbol, reason in self.meme_strategy.forced_exits(held, signal_date).items():
            exec_px = execution_prices.get(symbol, 0.0)
            if exec_px <= 0:
                continue
            self.portfolio.reduce_position(symbol, 1.0, exec_px, execution_date)
            self.meme_rug_exits.append({"date": str(signal_date.date()), "symbol": symbol, "reason": reason})

    def _apply_stop_losses(
        self,
        trigger_prices: Dict[str, float],
        execution_prices: Dict[str, float],
        execution_date: pd.Timestamp,
    ) -> None:
        to_close = []
        for symbol, pos in self.portfolio.state.positions.items():
            px = trigger_prices.get(symbol, 0.0)
            if px <= 0:
                continue
            if self.risk.stop_loss_triggered(pos.avg_entry_price, px, pos.quantity):
                to_close.append(symbol)

        for symbol in to_close:
            pos = self.portfolio.state.positions.get(symbol)
            exec_px = execution_prices.get(symbol, 0.0)
            if pos is None or exec_px <= 0:
                continue
            # Flatten at next open under explicit timing assumptions.
            self.portfolio.reduce_position(symbol, 1.0, exec_px, execution_date)

    def _apply_take_profits(
        self,
        market_data: Dict[str, pd.DataFrame],
        signal_date: pd.Timestamp,
        trigger_prices: Dict[str, float],
        execution_prices: Dict[str, float],
        execution_date: pd.Timestamp,
    ) -> None:
        gain_th = self.config.portfolio.take_profit_gain_threshold
        strong_sell = self.config.portfolio.take_profit_sell_fraction
        weak_sell = self.config.portfolio.take_profit_weak_trend_sell_fraction
        ma_window = self.config.portfolio.take_profit_trend_ma_window
        cooldown_days = self.config.portfolio.take_profit_cooldown_days

        for symbol, pos in list(self.portfolio.state.positions.items()):
            if pos.quantity <= 0:
                continue
            px = trigger_prices.get(symbol, 0.0)
            if px <= 0 or pos.avg_entry_price <= 0:
                continue

            last_tp = self.portfolio.state.last_take_profit_date.get(symbol)
            if last_tp is not None and (signal_date - last_tp).days < cooldown_days:
                continue

            gain = px / pos.avg_entry_price - 1.0
            if gain < gain_th:
                continue

            trend_ok = self._asset_above_ma(market_data, symbol, signal_date, ma_window)
            if trend_ok:
                sell_fraction = strong_sell
            elif self.config.portfolio.take_profit_full_exit_on_trend_break:
                sell_fraction = 1.0
            else:
                sell_fraction = weak_sell
            exec_px = execution_prices.get(symbol, 0.0)
            if exec_px > 0:
                self.portfolio.reduce_position(symbol, sell_fraction, exec_px, execution_date)

    def _apply_trend_break_exits(
        self,
        market_data: Dict[str, pd.DataFrame],
        signal_date: pd.Timestamp,
        execution_prices: Dict[str, float],
        execution_date: pd.Timestamp,
    ) -> None:
        ma_window = self.config.portfolio.trend_break_exit_ma_window
        exit_fraction = self.config.portfolio.trend_break_exit_fraction
        for symbol, pos in list(self.portfolio.state.positions.items()):
            if pos.quantity <= 0:
                continue
            if self._asset_above_ma(market_data, symbol, signal_date, ma_window):
                continue
            px = execution_prices.get(symbol, 0.0)
            if px <= 0:
                continue
            self.portfolio.reduce_position(symbol, exit_fraction, px, execution_date)

    def _asset_above_ma(
        self, market_data: Dict[str, pd.DataFrame], symbol: str, date: pd.Timestamp, window: int
    ) -> bool:
        df = market_data.get(symbol)
        if df is None or date not in df.index:
            return False
        idx = df.index.get_loc(date)
        if isinstance(idx, slice) or idx < window:
            return False
        ma = rolling_mean(df["Close"], window).iloc[idx]
        if pd.isna(ma):
            return False
        return float(df.loc[date, "Close"]) >= float(ma)

    def _currently_held_non_core_symbols(self, core_assets: Set[str]) -> Set[str]:
        return {
            s for s, pos in self.portfolio.state.positions.items() if abs(pos.quantity) > 1e-12 and s not in core_assets
        }

    def _restore_min_core_after_caps(
        self, target: Dict[str, float], core_assets: Set[str], core_floor: float
    ) -> Dict[str, float]:
        if not core_assets:
            return dict(target)
        out = dict(target)
        core_current = sum(out.get(s, 0.0) for s in core_assets)
        min_core = core_floor
        if core_current >= min_core:
            return out

        needed = min_core - core_current
        non_core = [s for s in out.keys() if s not in core_assets]
        reducible = sum(max(0.0, out.get(s, 0.0)) for s in non_core)
        if reducible <= 1e-12:
            per_core = needed / float(len(core_assets))
            for s in core_assets:
                out[s] = out.get(s, 0.0) + per_core
            return out

        for s in non_core:
            w = max(0.0, out.get(s, 0.0))
            cut = needed * (w / reducible) if reducible > 0 else 0.0
            out[s] = max(0.0, w - cut)
        per_core = needed / float(len(core_assets))
        for s in core_assets:
            out[s] = out.get(s, 0.0) + per_core
        return out

    def _blend_active_equal_weight(self, active_alloc: Dict[str, float]) -> Dict[str, float]:
        """
        Reduce concentration by blending positive active allocations with equal weighting.
        """
        blend = float(self.config.risk.active_equal_weight_blend)
        if blend <= 0:
            return dict(active_alloc)
        out = dict(active_alloc)
        pos_symbols = [s for s, w in out.items() if w > 0]
        if len(pos_symbols) <= 1:
            return out
        pos_budget = sum(out[s] for s in pos_symbols)
        if pos_budget <= 0:
            return out
        eq = pos_budget / float(len(pos_symbols))
        for s in pos_symbols:
            out[s] = (1.0 - blend) * out[s] + blend * eq
        return out

    def _apply_universe_quality_weighting(
        self,
        active_alloc: Dict[str, float],
        quality_snapshot: Dict[str, Dict[str, object]],
    ) -> Dict[str, float]:
        if not self.config.backtest.universe_quality_weighting_enabled:
            return dict(active_alloc)
        strength = clamp(float(self.config.backtest.universe_quality_weight_strength), 0.0, 0.5)
        if strength <= 0:
            return dict(active_alloc)

        out = dict(active_alloc)
        pos_symbols = [s for s, w in out.items() if w > 0]
        if not pos_symbols:
            return out
        for sym in pos_symbols:
            score = float(quality_snapshot.get(sym, {}).get("quality_score", 0.5))
            # Soft tilt around neutral 0.5 quality level.
            tilt = 1.0 + strength * (2.0 * score - 1.0)
            tilt = clamp(float(tilt), 1.0 - strength, 1.0 + strength)
            out[sym] = out[sym] * tilt
        total = sum(max(0.0, out[s]) for s in pos_symbols)
        if total <= 1e-12:
            return active_alloc
        target_total = sum(max(0.0, active_alloc[s]) for s in pos_symbols)
        scale = target_total / total if total > 0 else 1.0
        for sym in pos_symbols:
            out[sym] = max(0.0, out[sym]) * scale
        return out

    def _apply_dynamic_high_risk_group_cap(
        self,
        target: Dict[str, float],
        market_data: Dict[str, pd.DataFrame],
        date: pd.Timestamp,
        core_assets: Set[str],
    ) -> Dict[str, float]:
        cap = float(self.config.risk.high_risk_group_cap)
        if cap <= 0:
            return dict(target)

        candidates = [s for s, w in target.items() if s not in core_assets and w > 0]
        if not candidates:
            return dict(target)

        lookback = int(self.config.risk.high_risk_vol_lookback)
        vols: Dict[str, float] = {}
        for s in candidates:
            df = market_data.get(s)
            if df is None or date not in df.index:
                continue
            idx = df.index.get_loc(date)
            if isinstance(idx, slice) or idx < lookback:
                continue
            vol = rolling_volatility(df["Close"], lookback).iloc[idx]
            if pd.isna(vol):
                continue
            vols[s] = float(vol)
        if not vols:
            return dict(target)

        top_frac = float(self.config.risk.high_risk_top_fraction)
        n = max(1, int(len(vols) * top_frac))
        ranked = sorted(vols.items(), key=lambda x: x[1], reverse=True)
        high_risk_group = {s for s, _ in ranked[:n]}
        return self.risk.apply_group_cap(dict(target), high_risk_group, cap)

    def _diagnostics(
        self,
        equity_curve: pd.Series,
        exposure_series: pd.Series,
        end_prices: Dict[str, float],
        strategy_pnl_proxy: Dict[str, float],
    ) -> Dict[str, object]:
        closed = self.portfolio.closed_trade_pnls
        wins = [x for x in closed if x > 0]
        losses = [x for x in closed if x < 0]
        unrealized = self.portfolio.unrealized_pnl_by_asset(end_prices)
        per_asset: Dict[str, float] = {}
        for sym, pnl in self.portfolio.realized_pnl_by_asset.items():
            per_asset[sym] = per_asset.get(sym, 0.0) + pnl
        for sym, pnl in unrealized.items():
            per_asset[sym] = per_asset.get(sym, 0.0) + pnl

        avg_equity = float(equity_curve.mean()) if len(equity_curve) else 0.0
        turnover = self.portfolio.total_turnover_notional / avg_equity if avg_equity > 0 else 0.0
        strategy_total_abs = sum(abs(v) for v in strategy_pnl_proxy.values())
        strategy_share = {
            k: (abs(v) / strategy_total_abs if strategy_total_abs > 0 else 0.0)
            for k, v in strategy_pnl_proxy.items()
        }
        asset_total_abs = sum(abs(v) for v in per_asset.values())
        asset_share = {
            k: (abs(v) / asset_total_abs if asset_total_abs > 0 else 0.0)
            for k, v in per_asset.items()
        }
        top_asset_share = max(asset_share.values()) if asset_share else 0.0
        traded_symbols = sorted(
            {
                str(getattr(t, "symbol"))
                for t in self.portfolio.trade_log
                if getattr(t, "symbol", None)
            }
        )
        effective_weights = self._effective_strategy_weights(self._active_enabled_strategies())
        pair_strat = self.strategies.get("pair_trading")
        selected_pairs = []
        if pair_strat is not None and hasattr(pair_strat, "last_selected_pairs"):
            selected_pairs = [f"{a}|{b}" for a, b in getattr(pair_strat, "last_selected_pairs")]
        y_strat = self.strategies.get("yield")
        yield_targets = []
        if y_strat is not None and hasattr(y_strat, "config"):
            # Diagnostic intent: show configured preferred targets.
            yield_targets = list(self.config.strategy.yield_preferred_targets)
        return {
            "trade_count": len(self.portfolio.trade_log),
            "turnover": float(turnover),
            "average_holding_period_days": float(
                pd.Series(self.portfolio.closed_holding_period_days).mean()
            )
            if self.portfolio.closed_holding_period_days
            else 0.0,
            "win_rate": float(len(wins) / len(closed)) if closed else 0.0,
            "average_gain": float(pd.Series(wins).mean()) if wins else 0.0,
            "average_loss": float(pd.Series(losses).mean()) if losses else 0.0,
            "average_exposure": float(exposure_series.mean()) if len(exposure_series) else 0.0,
            "per_asset_contribution_pnl": per_asset,
            "per_asset_contribution_share_abs": asset_share,
            "top_asset_contribution_share_abs": float(top_asset_share),
            "strategy_contribution_proxy_pnl": strategy_pnl_proxy,
            "strategy_contribution_proxy_share_abs": strategy_share,
            "effective_strategy_weights": effective_weights,
            "yield_enabled": bool(self.config.strategy.yield_enabled),
            "yield_max_allocation": float(self.config.strategy.yield_max_allocation),
            "yield_preferred_targets": yield_targets,
            "pair_selected_pairs": selected_pairs,
            "fee_impact": float(self.portfolio.total_fee_paid),
            "slippage_impact": float(self.portfolio.total_slippage_cost),
            "fee_plus_slippage_impact": float(
                self.portfolio.total_fee_paid + self.portfolio.total_slippage_cost
            ),
            "traded_symbols": traded_symbols,
        }

    def _audit_warnings(self, market_data: Dict[str, pd.DataFrame]) -> List[str]:
        warnings: List[str] = []
        if self.config.backtest.selection_bias_warning:
            if self.config.data.use_rules_based_universe:
                warnings.append(
                    "Universe is rules-based (liquidity screen), but selection/survivorship bias may still remain."
                )
            else:
                warnings.append(
                    "Universe appears hand-selected; survivorship/selection bias may remain."
                )
        warnings.append(
            "Data is aligned on common dates across assets, which may exclude assets/dates "
            "with missing history and can bias results."
        )
        warnings.append(
            "Execution model assumes signals at close(t) and fills at open(t+1) with fixed fee/slippage rates."
        )
        if len(market_data) < 5:
            warnings.append("Small universe size increases overfitting risk.")
        return warnings

    def _regime_state(
        self,
        market_data: Dict[str, pd.DataFrame],
        date: pd.Timestamp,
        eligible_symbols: Set[str] | None = None,
    ) -> str:
        if not self.config.risk.regime_filter_enabled:
            return "bull"
        symbol = self._regime_symbol_at_date(market_data, date, eligible_symbols)
        if symbol is None:
            return "neutral"
        df = market_data.get(symbol)
        if df is None or date not in df.index:
            return "neutral"
        idx = df.index.get_loc(date)
        if isinstance(idx, slice):
            return "neutral"
        fast_w = self.config.risk.regime_fast_ma_window
        slow_w = self.config.risk.regime_slow_ma_window
        if idx < max(fast_w, slow_w):
            return "neutral"

        close = float(df.loc[date, "Close"])
        ma_fast = rolling_mean(df["Close"], fast_w).iloc[idx]
        ma_slow = rolling_mean(df["Close"], slow_w).iloc[idx]
        if pd.isna(ma_fast) or pd.isna(ma_slow):
            return "neutral"
        if close < float(ma_slow):
            return "bear"
        if close < float(ma_fast):
            return "neutral"
        return "bull"

    def _regime_symbol_at_date(
        self,
        market_data: Dict[str, pd.DataFrame],
        date: pd.Timestamp,
        eligible_symbols: Set[str] | None,
    ) -> str | None:
        fixed = self.config.risk.regime_symbol
        if fixed and fixed in market_data:
            return fixed
        if not self.config.risk.regime_use_dynamic_symbol:
            return None
        scores = self.data_manager.liquidity_scores_at_date(market_data, date)
        if eligible_symbols is not None:
            scores = {s: v for s, v in scores.items() if s in eligible_symbols}
        if not scores:
            return None
        return max(scores.items(), key=lambda item: item[1])[0]

    def _core_target_by_regime(self, regime_state: str) -> float:
        if regime_state == "bull":
            return self.config.portfolio.core_bull_allocation
        if regime_state == "bear":
            return self.config.portfolio.core_bear_allocation
        return self.config.portfolio.core_neutral_allocation

    def _active_multiplier_by_regime(self, regime_state: str) -> float:
        if regime_state == "bear":
            return self.config.risk.regime_active_multiplier_bear
        if regime_state == "neutral":
            return self.config.risk.regime_active_multiplier_neutral
        return 1.0

    def _core_bucket_allocation(
        self,
        market_data: Dict[str, pd.DataFrame],
        core_assets: List[str],
        date: pd.Timestamp,
        core_alloc_value: float,
    ) -> Dict[str, float]:
        if not core_assets or core_alloc_value <= 0:
            return {}

        inv_vol_scores: Dict[str, float] = {}
        vol_lb = self.config.portfolio.core_vol_lookback
        for symbol in core_assets:
            df = market_data.get(symbol)
            if df is None or date not in df.index:
                continue
            idx = df.index.get_loc(date)
            if isinstance(idx, slice) or idx < vol_lb:
                continue
            vol = rolling_volatility(df["Close"], vol_lb).iloc[idx]
            if pd.isna(vol) or float(vol) <= 0:
                continue
            inv_vol_scores[symbol] = 1.0 / float(vol)

        if not inv_vol_scores:
            per = core_alloc_value / float(len(core_assets))
            return {s: per for s in core_assets}

        rel = normalize_weights(inv_vol_scores)
        min_frac = self.config.portfolio.core_min_fraction_per_asset
        bounded_rel = {s: max(min_frac, rel.get(s, 0.0)) for s in core_assets}
        rel_sum = sum(bounded_rel.values())
        if rel_sum <= 0:
            per = core_alloc_value / float(len(core_assets))
            return {s: per for s in core_assets}
        return {s: core_alloc_value * (bounded_rel[s] / rel_sum) for s in core_assets}

    def _should_refresh_universe(
        self, date: pd.Timestamp, last_refresh_date: pd.Timestamp | None
    ) -> bool:
        if not self.config.data.use_rules_based_universe:
            return last_refresh_date is None
        if last_refresh_date is None:
            return True
        return (date - last_refresh_date).days >= int(self.config.data.universe_refresh_days)

    def _select_universe(
        self, market_data: Dict[str, pd.DataFrame], date: pd.Timestamp
    ) -> Set[str]:
        # Memes never compete for core universe slots (they have their own sleeve).
        core_data, _ = self._split_meme_data(market_data)
        if not self.config.data.use_rules_based_universe:
            return set(core_data.keys())
        sel = self.data_manager.select_universe_at_date(
            market_data=core_data,
            date=date,
            target_size=int(self.config.data.universe_size),
        )
        selected = set(sel.get("selected_symbols", set()))
        if not selected:
            return set(core_data.keys())
        return selected

    @staticmethod
    def _common_dates(market_data: Dict[str, pd.DataFrame]) -> List[pd.Timestamp]:
        common = None
        for df in market_data.values():
            common = df.index if common is None else common.intersection(df.index)
        if common is None:
            return []
        return list(common.sort_values())

    @staticmethod
    def _current_drawdown(equity_points: List[float], current_equity: float) -> float:
        if not equity_points:
            return 0.0
        peak = max(equity_points)
        if peak <= 0:
            return 0.0
        return current_equity / peak - 1.0
