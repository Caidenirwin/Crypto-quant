"""
Binance Spot Testnet execution layer using ccxt.

This module intentionally does not depend on strategy or backtest logic.
It only provides safe exchange connectivity and order execution helpers.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

try:
    import ccxt
except Exception as exc:  # pragma: no cover
    ccxt = None  # type: ignore[assignment]
    _CCXT_IMPORT_ERROR = exc
else:
    _CCXT_IMPORT_ERROR = None


LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


class BinanceSpotTestnetExecutor:
    """
    Safe Binance Spot Testnet executor.

    - Uses API keys from environment variables.
    - Uses sandbox mode.
    - Supports dry-run mode to avoid sending real testnet orders.
    """

    SYMBOL_PATTERN = re.compile(r"^[A-Z0-9]+/[A-Z0-9]+$")

    def __init__(self, dry_run: bool = True) -> None:
        if ccxt is None:  # pragma: no cover
            raise RuntimeError(
                "ccxt is not installed or failed to import. "
                "Install with: pip install ccxt"
            ) from _CCXT_IMPORT_ERROR
        self.dry_run = bool(dry_run)
        self.exchange: Any | None = None
        self._markets_loaded = False

    @staticmethod
    def _load_env_file(env_path: Path) -> None:
        """
        Minimal .env loader fallback for local runs.
        Does not overwrite already-set process environment variables.
        """
        if not env_path.exists():
            return
        try:
            for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and os.getenv(key) is None:
                    os.environ[key] = value
        except Exception:
            # Keep connection logic deterministic; explicit checks below still validate.
            return

    def connect_to_exchange(self) -> Any:
        """
        Connect to Binance Spot Testnet using env vars:
        - BINANCE_API_KEY
        - BINANCE_SECRET
        """
        # Local convenience: auto-load from project/root .env when available.
        cwd_env = Path.cwd() / ".env"
        self._load_env_file(cwd_env)
        module_env = Path(__file__).resolve().parents[1] / ".env"
        if module_env != cwd_env:
            self._load_env_file(module_env)

        api_key = os.getenv("BINANCE_API_KEY")
        secret = os.getenv("BINANCE_SECRET")
        if not api_key or not secret:
            raise RuntimeError(
                "Missing Binance credentials. Set BINANCE_API_KEY and BINANCE_SECRET."
            )

        exchange = ccxt.binance(
            {
                "apiKey": api_key,
                "secret": secret,
                "enableRateLimit": True,
                "options": {
                    "defaultType": "spot",
                    # Mitigate Binance -1021 timestamp drift by auto-applying server offset.
                    "adjustForTimeDifference": True,
                    # Slightly wider receive window for unstable local clock/network jitter.
                    "recvWindow": 15000,
                },
            }
        )
        exchange.set_sandbox_mode(True)
        self.exchange = exchange
        # Pre-sync immediately to reduce first private-call nonce failures.
        try:
            self._sync_time_difference()
        except Exception as exc:
            LOGGER.warning("Initial server time sync warning: %s", exc)
        # Soft assert that sandbox URLs look like testnet (defensive safety check).
        urls_blob = str(getattr(exchange, "urls", "")).lower()
        if "testnet" not in urls_blob:
            raise RuntimeError(
                "Binance executor safety check failed: sandbox/testnet endpoints not detected."
            )
        LOGGER.info("Connected to Binance Spot Testnet (sandbox mode ON).")
        return exchange

    def fetch_balance(self) -> Dict[str, Any]:
        self._require_connection()
        try:
            balance = self.exchange.fetch_balance()
            LOGGER.info("Fetched account balance successfully.")
            return balance
        except Exception as exc:
            if self._is_time_sync_error(exc):
                LOGGER.warning("Time drift detected on fetch_balance; syncing server time and retrying once.")
                self._sync_time_difference()
                balance = self.exchange.fetch_balance()
                LOGGER.info("Fetched account balance successfully after time sync retry.")
                return balance
            LOGGER.exception("Failed to fetch balance: %s", exc)
            raise

    def place_market_buy(self, symbol: str, amount: float) -> Dict[str, Any]:
        return self._place_market_order(symbol=symbol, side="buy", amount=amount)

    def place_market_sell(self, symbol: str, amount: float) -> Dict[str, Any]:
        return self._place_market_order(symbol=symbol, side="sell", amount=amount)

    def _place_market_order(self, symbol: str, side: str, amount: float) -> Dict[str, Any]:
        self._require_connection()
        side = side.lower().strip()
        if side not in {"buy", "sell"}:
            raise ValueError("side must be 'buy' or 'sell'")
        self._validate_symbol(symbol)
        if amount <= 0:
            raise ValueError("amount must be > 0")

        self._ensure_markets_loaded()
        if symbol not in self.exchange.markets:
            raise ValueError(f"Symbol not found on exchange: {symbol}")

        market = self.exchange.market(symbol)
        base = market["base"]
        quote = market["quote"]

        amount = self._validate_balance_for_order(symbol=symbol, side=side, amount=amount, base=base, quote=quote)

        order_preview = {
            "symbol": symbol,
            "side": side,
            "type": "market",
            "amount": amount,
            "dry_run": self.dry_run,
        }
        print(f"[ORDER PREVIEW] {order_preview}")
        LOGGER.info("Order preview: %s", order_preview)

        if self.dry_run:
            LOGGER.info("DRY RUN enabled. No order sent.")
            return {
                "status": "dry_run",
                "message": "Dry run enabled. Order not sent.",
                "order": order_preview,
            }

        try:
            if side == "buy":
                order = self.exchange.create_market_buy_order(symbol, amount)
            else:
                order = self.exchange.create_market_sell_order(symbol, amount)
            LOGGER.info("Order placed: %s", order)
            return order
        except Exception as exc:
            if self._is_time_sync_error(exc):
                LOGGER.warning("Time drift detected on order; syncing server time and retrying once.")
                self._sync_time_difference()
                if side == "buy":
                    order = self.exchange.create_market_buy_order(symbol, amount)
                else:
                    order = self.exchange.create_market_sell_order(symbol, amount)
                LOGGER.info("Order placed after time sync retry: %s", order)
                return order
            LOGGER.exception("Order failed: %s", exc)
            raise

    def _validate_balance_for_order(
        self, symbol: str, side: str, amount: float, base: str, quote: str
    ) -> float:
        """Validate balance; returns the (possibly clipped) amount to send."""
        balance = self.fetch_balance()
        free_bal = balance.get("free", {}) if isinstance(balance, dict) else {}

        if side == "sell":
            base_free = float(free_bal.get(base, 0.0) or 0.0)
            if base_free + 1e-12 < amount:
                # Buy fees are often charged in the base asset, so the exchange holds
                # slightly less than the paper fill. Sell what's there if the gap is tiny.
                if base_free >= amount * 0.98:
                    LOGGER.info("Clipping %s sell from %s to free balance %s", symbol, amount, base_free)
                    return base_free
                raise RuntimeError(
                    f"Insufficient {base} balance for sell. Free={base_free}, required={amount}"
                )
            return amount

        # Buy check: verify enough quote balance using latest last/close price.
        ticker = self.exchange.fetch_ticker(symbol)
        px = ticker.get("last") or ticker.get("close")
        if px is None:
            raise RuntimeError(f"Could not fetch price for symbol {symbol} to validate buy balance.")
        px = float(px)
        cost = amount * px
        quote_free = float(free_bal.get(quote, 0.0) or 0.0)
        if quote_free + 1e-12 < cost:
            raise RuntimeError(
                f"Insufficient {quote} balance for buy. Free={quote_free}, est_cost={cost:.8f}"
            )
        return amount

    def _validate_symbol(self, symbol: str) -> None:
        s = symbol.strip().upper()
        if not self.SYMBOL_PATTERN.match(s):
            raise ValueError(
                f"Invalid symbol format: {symbol}. Expected format like 'BTC/USDT'."
            )

    def _ensure_markets_loaded(self) -> None:
        if not self._markets_loaded:
            self.exchange.load_markets()
            self._markets_loaded = True

    def _sync_time_difference(self) -> None:
        """
        Sync local request timestamp offset with Binance server time.
        """
        self._require_connection()
        # CCXT helper computes and stores timeDifference internally.
        self.exchange.load_time_difference()
        td = getattr(self.exchange, "timeDifference", None)
        if td is not None:
            LOGGER.info("Binance time difference synced: %sms", td)

    @staticmethod
    def _is_time_sync_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return ("-1021" in msg) or ("timestamp for this request" in msg) or ("invalidnonce" in msg)

    def _require_connection(self) -> None:
        if self.exchange is None:
            raise RuntimeError("Exchange is not connected. Call connect_to_exchange() first.")
