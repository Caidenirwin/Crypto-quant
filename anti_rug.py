"""
Anti-rug-pull guard for the meme coin sleeve.

Three independent layers (a coin must pass ALL of them to be bought):

1. history_check   — daily price/volume behaviour (works in backtest AND live):
                     min coin age, 1-day crash, crash from recent high,
                     liquidity drain, parabolic pump (pump-and-dump risk).
2. market_check    — live order book / 24h volume from Binance mainnet public
                     market data: min volume, max spread, min bid depth,
                     intraday crash.
3. contract_check  — on-chain token contract audit via GoPlus Security:
                     honeypot, can't-sell, hidden owner, taxes, mint/freeze
                     authority, owner-controlled balances, holder concentration.

Semantics:
- allow_entry=False  -> no NEW buys for this coin.
- force_exit=True    -> sell the whole position immediately (rug in progress).
- Data/API failures are fail-closed for buys but never force a sell on their own.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

from config import Config

_BURN_ADDRESSES = {
    "0x0000000000000000000000000000000000000000",
    "0x000000000000000000000000000000000000dead",
    "0xdead000000000000000042069420694206942069",
}
# Owner-gated EVM powers: dangerous only while someone still owns the contract.
_OWNER_GATED_FLAGS = {
    "owner_change_balance": "OWNER_CAN_CHANGE_BALANCES",
    "is_mintable": "MINTABLE",
    "transfer_pausable": "TRANSFERS_PAUSABLE",
    "is_blacklisted": "BLACKLIST_FUNCTION",
    "slippage_modifiable": "TAX_MODIFIABLE",
    "personal_slippage_modifiable": "PER_WALLET_TAX_MODIFIABLE",
    "trading_cooldown": "TRADING_COOLDOWN",
    "is_proxy": "UPGRADEABLE_PROXY",
}
# EVM flags that are a rug signal no matter who owns the contract.
_HARD_EVM_FLAGS = {
    "is_honeypot": "HONEYPOT",
    "cannot_sell_all": "CANNOT_SELL_ALL",
    "hidden_owner": "HIDDEN_OWNER",
    "selfdestruct": "SELFDESTRUCT",
    "can_take_back_ownership": "CAN_RECLAIM_OWNERSHIP",
}


@dataclass
class RugVerdict:
    symbol: str
    allow_entry: bool = True
    force_exit: bool = False
    reasons: List[str] = field(default_factory=list)

    def merge(self, other: "RugVerdict") -> "RugVerdict":
        return RugVerdict(
            symbol=self.symbol,
            allow_entry=self.allow_entry and other.allow_entry,
            force_exit=self.force_exit or other.force_exit,
            reasons=self.reasons + other.reasons,
        )

    def block(self, reason: str, exit_now: bool = False) -> None:
        self.allow_entry = False
        self.force_exit = self.force_exit or exit_now
        self.reasons.append(reason)


def meme_base(symbol: str) -> str:
    s = str(symbol).strip().upper()
    return s.split("(")[0].strip() if "(" in s else s.split("/")[0].strip()


class AntiRugGuard:
    def __init__(self, config: Config, cache_path: Optional[Path] = None, timeout: float = 15.0):
        self.cfg = config.meme
        self.cache_path = cache_path
        self.timeout = timeout
        self._contract_cache: Dict[str, Dict[str, object]] = self._load_cache()

    # ------------------------------------------------------------------ #
    # Layer 1: daily price/volume history (pure function of the data)     #
    # ------------------------------------------------------------------ #
    def history_check(self, symbol: str, df: pd.DataFrame, date: pd.Timestamp) -> RugVerdict:
        v = RugVerdict(symbol=symbol)
        if df is None or date not in df.index:
            v.block("NO_DATA")
            return v
        idx = df.index.get_loc(date)
        if isinstance(idx, slice):
            v.block("DUPLICATE_DATES")
            return v
        hist = df.iloc[: idx + 1]
        close = hist["Close"].astype(float)
        volume = hist["Volume"].astype(float)

        if len(hist) < int(self.cfg.rug_min_history_days):
            v.block(f"TOO_YOUNG ({len(hist)}d < {self.cfg.rug_min_history_days}d)")
            return v

        px = float(close.iloc[-1])
        prev = float(close.iloc[-2])
        if prev > 0 and px / prev - 1.0 <= -float(self.cfg.rug_crash_1d_pct):
            v.block(f"CRASH_1D ({px / prev - 1.0:.1%})", exit_now=True)

        hw = int(self.cfg.rug_crash_high_window)
        recent_high = float(hist["High"].astype(float).iloc[-hw:].max())
        if recent_high > 0 and px / recent_high - 1.0 <= -float(self.cfg.rug_crash_from_high_pct):
            v.block(f"CRASH_FROM_{hw}D_HIGH ({px / recent_high - 1.0:.1%})", exit_now=True)

        # Median (not mean) so one hype day doesn't make normal volume look like a drain.
        vol_base = float(volume.iloc[-60:].median())
        vol_recent = float(volume.iloc[-5:].mean())
        if vol_base > 0 and vol_recent / vol_base < float(self.cfg.rug_liquidity_drain_ratio):
            v.block(f"LIQUIDITY_DRAIN (5d/60d vol={vol_recent / vol_base:.2f})", exit_now=True)

        if len(close) >= 8 and float(close.iloc[-8]) > 0:
            r7 = px / float(close.iloc[-8]) - 1.0
            if r7 >= float(self.cfg.rug_pump_7d_pct):
                v.block(f"PARABOLIC_PUMP_7D (+{r7:.0%})")
        return v

    # ------------------------------------------------------------------ #
    # Layer 2: live market microstructure (Binance mainnet public data)   #
    # ------------------------------------------------------------------ #
    def market_check(self, symbol: str, order_usd: float = 0.0) -> RugVerdict:
        v = RugVerdict(symbol=symbol)
        pair = f"{meme_base(symbol)}USDT"
        base_url = str(self.cfg.rug_market_data_url).rstrip("/")
        try:
            t = self._get_json(f"{base_url}/api/v3/ticker/24hr", {"symbol": pair})
            ob = self._get_json(f"{base_url}/api/v3/depth", {"symbol": pair, "limit": 100})
            kl = self._get_json(f"{base_url}/api/v3/klines", {"symbol": pair, "interval": "1h", "limit": 5})
        except Exception as exc:
            v.block(f"MARKET_DATA_UNAVAILABLE ({exc})")
            return v

        qv = float(t.get("quoteVolume", 0.0) or 0.0)
        if qv < float(self.cfg.rug_min_quote_volume_24h):
            v.block(f"LOW_24H_VOLUME (${qv:,.0f})")
        chg24 = float(t.get("priceChangePercent", 0.0) or 0.0) / 100.0
        if chg24 <= -float(self.cfg.rug_crash_1d_pct):
            v.block(f"CRASH_24H ({chg24:.1%})", exit_now=True)

        bids, asks = ob.get("bids") or [], ob.get("asks") or []
        if not bids or not asks:
            v.block("EMPTY_ORDER_BOOK", exit_now=True)
            return v
        bid, ask = float(bids[0][0]), float(asks[0][0])
        mid = (bid + ask) / 2.0
        spread_bps = (ask - bid) / mid * 1e4 if mid > 0 else float("inf")
        if spread_bps > float(self.cfg.rug_max_spread_bps):
            v.block(f"WIDE_SPREAD ({spread_bps:.1f}bps)")
        depth = sum(float(p) * float(q) for p, q in bids if float(p) >= mid * 0.98)
        need = max(float(self.cfg.rug_min_bid_depth_usd), float(order_usd) * float(self.cfg.rug_min_depth_order_multiple))
        if depth < need:
            v.block(f"THIN_BID_DEPTH (${depth:,.0f} < ${need:,.0f})")

        if len(kl) >= 5:
            c_then, c_now = float(kl[0][4]), float(kl[-1][4])
            if c_then > 0 and c_now / c_then - 1.0 <= -float(self.cfg.rug_intraday_crash_pct):
                v.block(f"INTRADAY_CRASH_4H ({c_now / c_then - 1.0:.1%})", exit_now=True)
        return v

    # ------------------------------------------------------------------ #
    # Layer 3: on-chain contract audit (GoPlus Security)                  #
    # ------------------------------------------------------------------ #
    def contract_check(self, symbol: str) -> RugVerdict:
        v = RugVerdict(symbol=symbol)
        if not self.cfg.rug_contract_check_enabled:
            return v
        base = meme_base(symbol)
        meta = self.cfg.meme_contracts.get(base)
        if not meta:
            # Native L1 coin (e.g. DOGE) — no token contract to exploit.
            return v
        chain, addr = str(meta["chain"]), str(meta["address"])
        try:
            res = self._contract_data(base, chain, addr)
        except Exception as exc:
            v.block(f"CONTRACT_DATA_UNAVAILABLE ({exc})")
            return v
        if not res:
            v.block("CONTRACT_NOT_FOUND")
            return v
        if chain == "solana":
            self._check_solana(res, v)
        else:
            self._check_evm(res, v)
        return v

    def _check_evm(self, res: Dict[str, object], v: RugVerdict) -> None:
        for key, label in _HARD_EVM_FLAGS.items():
            if str(res.get(key, "0")) == "1":
                v.block(label, exit_now=True)
        if str(res.get("is_open_source", "1")) == "0":
            v.block("CLOSED_SOURCE")
        for key in ("buy_tax", "sell_tax"):
            try:
                tax = float(res.get(key) or 0.0)
            except (TypeError, ValueError):
                tax = 0.0
            if tax > float(self.cfg.rug_max_tax):
                v.block(f"HIGH_{key.upper()} ({tax:.0%})", exit_now=True)

        owner = str(res.get("owner_address") or "").lower()
        renounced = owner in ("", "none") or owner in _BURN_ADDRESSES
        if not renounced:
            for key, label in _OWNER_GATED_FLAGS.items():
                if str(res.get(key, "0")) == "1":
                    v.block(f"{label} (owner={owner[:10]}...)")

        top10 = 0.0
        for h in (res.get("holders") or [])[:10]:
            addr = str(h.get("address", "")).lower()
            tag = str(h.get("tag", "")).lower()
            if addr in _BURN_ADDRESSES or "dead" in tag or "burn" in tag or str(h.get("is_locked")) == "1":
                continue
            try:
                top10 += float(h.get("percent") or 0.0)
            except (TypeError, ValueError):
                continue
        if top10 > float(self.cfg.rug_max_top10_holder_pct):
            v.block(f"HOLDER_CONCENTRATION (top10={top10:.0%})")

    def _check_solana(self, res: Dict[str, object], v: RugVerdict) -> None:
        def _status(key: str) -> str:
            node = res.get(key)
            return str(node.get("status", "0")) if isinstance(node, dict) else str(node or "0")

        if _status("mintable") == "1":
            v.block("MINT_AUTHORITY_ACTIVE", exit_now=True)
        if _status("freezable") == "1":
            v.block("FREEZE_AUTHORITY_ACTIVE", exit_now=True)
        if _status("balance_mutable_authority") == "1":
            v.block("BALANCE_MUTABLE", exit_now=True)
        if _status("closable") == "1":
            v.block("ACCOUNT_CLOSABLE")
        if str(res.get("non_transferable", "0")) == "1":
            v.block("NON_TRANSFERABLE", exit_now=True)
        if res.get("transfer_hook"):
            v.block("TRANSFER_HOOK")
        if res.get("transfer_fee"):
            v.block("TRANSFER_FEE")

    def _contract_data(self, base: str, chain: str, addr: str) -> Dict[str, object]:
        cached = self._contract_cache.get(base)
        max_age = float(self.cfg.rug_contract_cache_hours) * 3600.0
        if cached and (time.time() - float(cached.get("fetched_at", 0))) < max_age:
            return dict(cached.get("data") or {})
        if chain == "solana":
            url = "https://api.gopluslabs.io/api/v1/solana/token_security"
            key = addr
        else:
            url = f"https://api.gopluslabs.io/api/v1/token_security/{chain}"
            key = addr.lower()
        payload = self._get_json(url, {"contract_addresses": addr})
        data = (payload.get("result") or {}).get(key) or {}
        if data:
            self._contract_cache[base] = {"fetched_at": time.time(), "data": data}
            self._save_cache()
        return data

    # ------------------------------------------------------------------ #
    # Combined live check                                                  #
    # ------------------------------------------------------------------ #
    def live_check(self, symbol: str, df: Optional[pd.DataFrame], order_usd: float = 0.0) -> RugVerdict:
        v = RugVerdict(symbol=symbol)
        if df is not None and not df.empty:
            v = v.merge(self.history_check(symbol, df, df.index[-1]))
        v = v.merge(self.market_check(symbol, order_usd))
        v = v.merge(self.contract_check(symbol))
        return v

    # ------------------------------------------------------------------ #
    def _get_json(self, url: str, params: Dict[str, object]) -> Dict:
        r = requests.get(url, params=params, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def _load_cache(self) -> Dict[str, Dict[str, object]]:
        if self.cache_path is None or not self.cache_path.exists():
            return {}
        try:
            return json.loads(self.cache_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_cache(self) -> None:
        if self.cache_path is None:
            return
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self._contract_cache), encoding="utf-8")
        except Exception:
            pass
