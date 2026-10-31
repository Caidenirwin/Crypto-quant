"""
Tax reserve utilities for paper trading.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TaxReserveManager:
    tax_reserve_rate: float = 0.30
    total_tax_reserve: float = 0.0

    def reserve_from_realized_profit(self, realized_profit: float) -> float:
        """
        Reserve tax only on positive realized profit.
        """
        if realized_profit <= 0:
            return 0.0
        tax = float(realized_profit) * float(self.tax_reserve_rate)
        self.total_tax_reserve += tax
        return tax
