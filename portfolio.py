"""
Portfolio module:
- Track positions and cash
- Apply target allocations
- Rebalance on schedule
- Support partial sells
- Record auditable trade diagnostics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd

from config import Config
from utils import clamp


@dataclass
class Position:
    symbol: str
    quantity: float = 0.0
    avg_entry_price: float = 0.0
    entry_date: pd.Timestamp | None = None


@dataclass
class TradeRecord:
    date: pd.Timestamp
    symbol: str
    quantity: float
    reference_price: float
    execution_price: float
    gross_notional: float
    fee: float
    slippage_cost: float
    realized_pnl: float


@dataclass
class PortfolioState:
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    last_rebalance_date: pd.Timestamp | None = None
    start_date: pd.Timestamp | None = None
    last_take_profit_date: Dict[str, pd.Timestamp] = field(default_factory=dict)


class PortfolioManager:
    def __init__(self, config: Config):
        self.config = config
        self.state = PortfolioState(cash=config.data.start_cash)
        self.trade_log: List[TradeRecord] = []
        self.total_fee_paid: float = 0.0
        self.total_slippage_cost: float = 0.0
        self.total_turnover_notional: float = 0.0
        self.closed_trade_pnls: List[float] = []
        self.closed_holding_period_days: List[int] = []
        self.realized_pnl_by_asset: Dict[str, float] = {}

    def mark_to_market(self, prices: Dict[str, float]) -> float:
        holdings = sum(
            pos.quantity * prices.get(sym, 0.0) for sym, pos in self.state.positions.items()
        )
        return self.state.cash + holdings

    def holdings_value(self, prices: Dict[str, float]) -> float:
        return sum(pos.quantity * prices.get(sym, 0.0) for sym, pos in self.state.positions.items())

    def gross_exposure(self, prices: Dict[str, float]) -> float:
        return sum(abs(pos.quantity * prices.get(sym, 0.0)) for sym, pos in self.state.positions.items())

    def should_rebalance(self, date: pd.Timestamp) -> bool:
        if self.state.start_date is None:
            self.state.start_date = date
        last = self.state.last_rebalance_date
        if last is None:
            offset = self.config.portfolio.rebalance_initial_offset_days
            return (date - self.state.start_date).days >= offset
        delta = (date - last).days
        return delta >= self.config.portfolio.rebalance_frequency_days

    def rebalance(self, target_weights: Dict[str, float], prices: Dict[str, float], date: pd.Timestamp) -> None:
        """
        Execute target rebalance using provided reference prices.
        Execution price applies configurable slippage around those references.
        """
        equity = self.mark_to_market(prices)

        for symbol, target_w in target_weights.items():
            reference_price = prices.get(symbol, 0.0)
            if reference_price <= 0:
                continue

            target_value = equity * target_w
            current_pos = self.state.positions.get(symbol, Position(symbol=symbol))
            current_value = current_pos.quantity * reference_price
            trade_value = target_value - current_value

            if trade_value < 0 and not self.config.portfolio.allow_partial_sells:
                trade_value = -current_value

            trade_qty = trade_value / reference_price
            if abs(trade_qty) < 1e-12:
                continue

            self._execute_trade(symbol, trade_qty, reference_price, date)

        self.state.last_rebalance_date = date

    def reduce_position(self, symbol: str, sell_fraction: float, price: float, date: pd.Timestamp) -> bool:
        """
        Partial exit helper used by risk logic.
        Uses execution assumptions consistently (fee/slippage included).
        """
        pos = self.state.positions.get(symbol)
        if pos is None or price <= 0:
            return False
        sell_fraction = clamp(sell_fraction, 0.0, 1.0)
        if sell_fraction <= 0:
            return False

        if pos.quantity > 0:
            trade_qty = -abs(pos.quantity) * sell_fraction
        else:
            trade_qty = abs(pos.quantity) * sell_fraction
        if abs(trade_qty) <= 1e-12:
            return False

        self._execute_trade(symbol, trade_qty, price, date)
        self.state.last_take_profit_date[symbol] = date
        return True

    def unrealized_pnl_by_asset(self, prices: Dict[str, float]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for symbol, pos in self.state.positions.items():
            px = prices.get(symbol, 0.0)
            if px <= 0 or abs(pos.quantity) <= 1e-12:
                out[symbol] = 0.0
                continue
            if pos.quantity > 0:
                out[symbol] = (px - pos.avg_entry_price) * pos.quantity
            else:
                out[symbol] = (pos.avg_entry_price - px) * abs(pos.quantity)
        return out

    def _effective_fee_rate(self) -> float:
        if self.config.portfolio.fee_rate is not None:
            return float(self.config.portfolio.fee_rate)
        return float(self.config.portfolio.transaction_cost_bps) / 10_000.0

    def _execution_price(self, reference_price: float, trade_qty: float) -> float:
        slip = float(self.config.portfolio.slippage_rate)
        if trade_qty > 0:
            return reference_price * (1.0 + slip)
        return reference_price * (1.0 - slip)

    def _execute_trade(
        self, symbol: str, trade_qty: float, reference_price: float, date: pd.Timestamp
    ) -> None:
        current_pos = self.state.positions.get(symbol, Position(symbol=symbol))
        old_qty = current_pos.quantity
        old_avg = current_pos.avg_entry_price
        old_entry_date = current_pos.entry_date

        exec_price = self._execution_price(reference_price, trade_qty)
        gross_notional = trade_qty * exec_price
        fee_rate = self._effective_fee_rate()
        fee = abs(gross_notional) * fee_rate
        slippage_cost = abs(trade_qty) * abs(exec_price - reference_price)
        realized_pnl = self._realized_pnl(old_qty, old_avg, trade_qty, exec_price)

        self.state.cash -= gross_notional + fee
        self.total_fee_paid += fee
        self.total_slippage_cost += slippage_cost
        self.total_turnover_notional += abs(gross_notional)

        new_qty = old_qty + trade_qty
        flipped = abs(old_qty) > 1e-12 and abs(new_qty) > 1e-12 and (old_qty * new_qty < 0)
        if abs(new_qty) <= 1e-12:
            if symbol in self.state.positions:
                del self.state.positions[symbol]
            if abs(old_qty) > 1e-12 and old_entry_date is not None:
                self.closed_holding_period_days.append((date - old_entry_date).days)
            if abs(realized_pnl) > 1e-12:
                self.closed_trade_pnls.append(realized_pnl)
            self.realized_pnl_by_asset[symbol] = self.realized_pnl_by_asset.get(symbol, 0.0) + realized_pnl
        else:
            new_avg = self._updated_avg_entry(old_qty, old_avg, trade_qty, exec_price, new_qty)
            new_entry_date = self._updated_entry_date(old_qty, trade_qty, new_qty, old_entry_date, date)
            self.state.positions[symbol] = Position(
                symbol=symbol,
                quantity=new_qty,
                avg_entry_price=new_avg,
                entry_date=new_entry_date,
            )
            if abs(realized_pnl) > 1e-12:
                self.closed_trade_pnls.append(realized_pnl)
                self.realized_pnl_by_asset[symbol] = (
                    self.realized_pnl_by_asset.get(symbol, 0.0) + realized_pnl
                )
            if flipped and old_entry_date is not None:
                self.closed_holding_period_days.append((date - old_entry_date).days)

        self.trade_log.append(
            TradeRecord(
                date=date,
                symbol=symbol,
                quantity=trade_qty,
                reference_price=reference_price,
                execution_price=exec_price,
                gross_notional=abs(gross_notional),
                fee=fee,
                slippage_cost=slippage_cost,
                realized_pnl=realized_pnl,
            )
        )

    @staticmethod
    def _realized_pnl(old_qty: float, old_avg: float, trade_qty: float, exec_price: float) -> float:
        if abs(old_qty) <= 1e-12:
            return 0.0
        if (old_qty > 0 and trade_qty > 0) or (old_qty < 0 and trade_qty < 0):
            return 0.0
        close_qty = min(abs(old_qty), abs(trade_qty))
        if old_qty > 0:
            return close_qty * (exec_price - old_avg)
        return close_qty * (old_avg - exec_price)

    @staticmethod
    def _updated_avg_entry(
        old_qty: float, old_avg: float, trade_qty: float, exec_price: float, new_qty: float
    ) -> float:
        if abs(old_qty) <= 1e-12:
            return exec_price
        same_direction_increase = (old_qty > 0 and trade_qty > 0) or (old_qty < 0 and trade_qty < 0)
        if same_direction_increase:
            total_abs = abs(old_qty) + abs(trade_qty)
            if total_abs <= 1e-12:
                return exec_price
            return ((abs(old_qty) * old_avg) + (abs(trade_qty) * exec_price)) / total_abs
        if (old_qty > 0 and new_qty > 0) or (old_qty < 0 and new_qty < 0):
            return old_avg
        return exec_price

    @staticmethod
    def _updated_entry_date(
        old_qty: float,
        trade_qty: float,
        new_qty: float,
        old_entry_date: pd.Timestamp | None,
        trade_date: pd.Timestamp,
    ) -> pd.Timestamp | None:
        if abs(old_qty) <= 1e-12:
            return trade_date
        same_direction_increase = (old_qty > 0 and trade_qty > 0) or (old_qty < 0 and trade_qty < 0)
        if same_direction_increase:
            return old_entry_date or trade_date
        if (old_qty > 0 and new_qty > 0) or (old_qty < 0 and new_qty < 0):
            return old_entry_date or trade_date
        return trade_date
