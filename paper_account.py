"""
Virtual paper account with persistent state reconstruction from trade logs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd

from tax_reserve import TaxReserveManager


@dataclass
class PaperPosition:
    symbol: str
    quantity: float = 0.0
    avg_entry_price: float = 0.0


@dataclass
class PaperAccount:
    starting_capital_usd: float = 500.0
    cash_usd: float = 500.0
    tax_manager: TaxReserveManager = field(default_factory=TaxReserveManager)
    positions: Dict[str, PaperPosition] = field(default_factory=dict)
    realized_pnl: float = 0.0
    realized_losses: float = 0.0

    @classmethod
    def from_trade_log(
        cls,
        trade_log_path: str,
        starting_capital_usd: float,
        tax_reserve_rate: float,
    ) -> "PaperAccount":
        acct = cls(
            starting_capital_usd=float(starting_capital_usd),
            cash_usd=float(starting_capital_usd),
            tax_manager=TaxReserveManager(tax_reserve_rate=float(tax_reserve_rate)),
        )
        try:
            df = pd.read_csv(trade_log_path)
        except FileNotFoundError:
            return acct
        if df.empty:
            return acct

        required = {"symbol", "side", "quantity", "price", "fees_estimate"}
        if not required.issubset(set(df.columns)):
            return acct

        for _, row in df.iterrows():
            acct.apply_fill(
                symbol=str(row["symbol"]),
                side=str(row["side"]).lower(),
                quantity=float(row["quantity"]),
                price=float(row["price"]),
                fee=float(row.get("fees_estimate", 0.0) or 0.0),
            )
        return acct

    def position_value(self, symbol: str, price: float) -> float:
        pos = self.positions.get(symbol)
        if pos is None:
            return 0.0
        return float(pos.quantity) * float(price)

    def open_position_value(self, price_map: Dict[str, float]) -> float:
        out = 0.0
        for sym, pos in self.positions.items():
            px = float(price_map.get(sym, 0.0))
            out += float(pos.quantity) * px
        return out

    def unrealized_pnl(self, price_map: Dict[str, float]) -> float:
        pnl = 0.0
        for sym, pos in self.positions.items():
            px = float(price_map.get(sym, 0.0))
            if px <= 0:
                continue
            pnl += float(pos.quantity) * (px - float(pos.avg_entry_price))
        return pnl

    def total_equity(self, price_map: Dict[str, float]) -> float:
        return float(self.cash_usd) + self.open_position_value(price_map) + float(
            self.tax_manager.total_tax_reserve
        )

    def apply_fill(
        self, symbol: str, side: str, quantity: float, price: float, fee: float
    ) -> Dict[str, float]:
        """
        Apply a filled spot trade to virtual account.
        Returns realized pnl/tax/cash snapshot for logging.
        """
        symbol = str(symbol)
        side = str(side).lower().strip()
        qty = float(quantity)
        px = float(price)
        fee = float(fee)
        if qty <= 0 or px <= 0:
            raise ValueError("quantity and price must be > 0")
        if side not in {"buy", "sell"}:
            raise ValueError("side must be buy/sell")

        pos = self.positions.get(symbol, PaperPosition(symbol=symbol))
        realized = 0.0
        tax_reserved = 0.0
        notional = qty * px

        if side == "buy":
            total_cost = notional + fee
            if total_cost > self.cash_usd + 1e-12:
                raise RuntimeError(
                    f"Insufficient virtual cash for buy: need {total_cost:.4f}, have {self.cash_usd:.4f}"
                )
            new_qty = pos.quantity + qty
            if new_qty <= 0:
                raise RuntimeError("Invalid buy resulting quantity.")
            new_avg = (
                (pos.quantity * pos.avg_entry_price + qty * px) / new_qty
                if pos.quantity > 0
                else px
            )
            pos.quantity = new_qty
            pos.avg_entry_price = new_avg
            self.cash_usd -= total_cost
        else:
            if qty > pos.quantity + 1e-12:
                raise RuntimeError(
                    f"Insufficient position for sell {symbol}: have {pos.quantity:.8f}, trying {qty:.8f}"
                )
            realized = (px - pos.avg_entry_price) * qty - fee
            pos.quantity -= qty
            self.cash_usd += notional - fee
            if pos.quantity <= 1e-12:
                pos.quantity = 0.0
                pos.avg_entry_price = 0.0
            tax_reserved = self.tax_manager.reserve_from_realized_profit(realized)
            if tax_reserved > 0:
                self.cash_usd -= tax_reserved

        self.positions[symbol] = pos
        if pos.quantity <= 0:
            del self.positions[symbol]

        self.realized_pnl += realized
        if realized < 0:
            self.realized_losses += abs(realized)

        return {
            "realized_pnl": float(realized),
            "tax_reserved": float(tax_reserved),
            "cash_usd": float(self.cash_usd),
            "total_tax_reserve": float(self.tax_manager.total_tax_reserve),
        }

    def exposure_pct(self, price_map: Dict[str, float]) -> float:
        equity = self.total_equity(price_map)
        if equity <= 0:
            return 0.0
        return self.open_position_value(price_map) / equity

    def cash_buffer_pct(self, price_map: Dict[str, float]) -> float:
        equity = self.total_equity(price_map)
        if equity <= 0:
            return 0.0
        return self.cash_usd / equity
