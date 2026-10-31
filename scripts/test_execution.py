"""
Simple execution connectivity test for Binance Spot Testnet.

Usage examples:
  python scripts/test_execution.py
  python scripts/test_execution.py --dry-run false --symbol BTC/USDT --amount 0.001 --side buy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# Ensure project root is importable when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from execution.binance_executor import BinanceSpotTestnetExecutor


def _parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Binance Spot Testnet execution test")
    parser.add_argument("--dry-run", default="true", help="true/false (default: true)")
    parser.add_argument("--symbol", default="BTC/USDT", help="Order symbol (default: BTC/USDT)")
    parser.add_argument("--amount", type=float, default=0.001, help="Base amount for market order")
    parser.add_argument("--side", choices=["buy", "sell"], default="buy", help="Order side")
    args = parser.parse_args()

    dry_run = _parse_bool(args.dry_run)
    executor = BinanceSpotTestnetExecutor(dry_run=dry_run)

    exchange = executor.connect_to_exchange()
    print(f"successful connection: {exchange.id} (sandbox={True})")

    bal = executor.fetch_balance()
    print("account balance (free):")
    print(json.dumps(bal.get("free", {}), indent=2, default=str))

    if dry_run:
        result = (
            executor.place_market_buy(args.symbol, args.amount)
            if args.side == "buy"
            else executor.place_market_sell(args.symbol, args.amount)
        )
        print("test order result:")
        print(json.dumps(result, indent=2, default=str))
        print("dry run message: no order sent.")
        return

    # Real testnet order mode (still sandbox/testnet, not production).
    result = (
        executor.place_market_buy(args.symbol, args.amount)
        if args.side == "buy"
        else executor.place_market_sell(args.symbol, args.amount)
    )
    print("test order result:")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
