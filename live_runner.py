"""
One-shot paper trading runner on Binance Spot Testnet.

This runner:
- Uses existing strategy logic via BacktestEngine target-weight generation.
- Uses Binance Spot Testnet execution via ccxt.
- Enforces an internal virtual paper account (default $500).
- Applies tax reserve on realized gains.
- Logs trades/equity/tax reserve.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

# Configure logging BEFORE project imports so this wins the basicConfig race.
# Uses local (ET) time with an explicit "ET" label on every INFO line.
_DISPLAY_TZ_EARLY = ZoneInfo("America/New_York")


class _ETFormatter(logging.Formatter):
    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        dt = datetime.fromtimestamp(record.created, tz=_DISPLAY_TZ_EARLY)
        return dt.strftime("%Y-%m-%d %H:%M:%S ET")


_root = logging.getLogger()
if not _root.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(_ETFormatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    _root.addHandler(_handler)
    _root.setLevel(logging.INFO)

from anti_rug import AntiRugGuard, RugVerdict
from backtest import BacktestEngine
from config import Config
from data import DataManager
from execution.binance_executor import BinanceSpotTestnetExecutor
from main import build_symbol_file_map
from paper_account import PaperAccount


@dataclass
class LiveSafetyConfig:
    # Order/position limits scale with starting_capital_usd: ~12% per order, ~36% per position
    max_order_usd: float = 60.0
    max_position_usd: float = 180.0
    max_total_exposure_pct: float = 0.95
    min_cash_buffer_pct: float = 0.05
    stop_daily_drawdown_pct: float = 0.12
    # V7 backtest max DD is -20%; stop at 30% gives 1.5x breathing room for live volatility.
    stop_total_drawdown_pct: float = 0.30
    starting_capital_usd: float = 500.0
    tax_reserve_rate: float = 0.30
    profit_to_core_hold_fraction: float = 0.0  # No forced core holds — let strategy decide
    core_hold_symbols: Tuple[str, ...] = ()

    def order_limit(self, equity: float) -> float:
        """Dynamic order cap: 12% of current equity (min $10, max $500)."""
        return max(10.0, min(500.0, equity * 0.12))

    def position_limit(self, equity: float) -> float:
        """Dynamic position cap: 36% of current equity."""
        return max(30.0, min(2000.0, equity * 0.36))


@dataclass
class RunSummary:
    timestamp_utc: str
    equity: float
    drawdown_pct: float
    trades_executed: int
    tax_reserve: float
    fatal_stop: bool = False
    stop_reason: str = ""


DISPLAY_TZ = ZoneInfo("America/New_York")
DISPLAY_TZ_LABEL = "ET"

# Process exit codes (read by autopilot.ps1).
EXIT_SAFETY_STOP = 2  # drawdown/API safety stop -> autopilot halts and waits for a human


TOKEN_ALIAS_TO_BINANCE = {
    # CSV long-name → Binance base symbol
    "CHAINLINK":  "LINK",
    "CENTRIFUGE": "CFG",
    "CLEARPOOL":  "CPOOL",
    "MAPLE":      "MPL",
    "HEDERA":     "HBAR",
    "ZEBEC":      "ZBCN",
    "SOLANA":     "SOL",
    # New additions — AVAX/SUI/NEAR/NEAR map 1:1 so no alias needed,
    # but POLYX/PRO/RIO are unlikely to be on Binance Spot Testnet;
    # the exchange order-placement code will skip them gracefully if not found.
}


def _load_env_file(env_path: Path) -> None:
    """
    Minimal .env loader to avoid extra dependencies.
    Loads KEY=VALUE pairs into process env only when key is not already set.
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
        # Keep runner resilient; explicit key checks below will still guard safety.
        return


def ensure_csv(path: Path, headers: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)


def append_csv_row(path: Path, row: Dict[str, object], headers: List[str]) -> None:
    ensure_csv(path, headers)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writerow({k: row.get(k, "") for k in headers})


def write_positions_snapshot(
    path: Path,
    timestamp_utc: str,
    paper: PaperAccount,
    latest_prices: Dict[str, float],
) -> None:
    """
    Write a full mark-to-market positions snapshot each run.
    This keeps dashboard unrealized PnL current even when no new trades occur.
    """
    headers = [
        "timestamp",
        "symbol",
        "quantity",
        "avg_entry_price",
        "last_price",
        "market_value",
        "unrealized_pnl",
        "allocation_pct",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    total_equity = float(paper.total_equity(latest_prices))
    rows: List[Dict[str, object]] = []
    for sym, pos in sorted(paper.positions.items()):
        qty = float(pos.quantity)
        if qty <= 0:
            continue
        last_px = float(latest_prices.get(sym, 0.0))
        mv = qty * last_px if last_px > 0 else 0.0
        upnl = qty * (last_px - float(pos.avg_entry_price)) if last_px > 0 else 0.0
        alloc = (mv / total_equity) if total_equity > 0 else 0.0
        rows.append(
            {
                "timestamp": timestamp_utc,
                "symbol": sym,
                "quantity": qty,
                "avg_entry_price": float(pos.avg_entry_price),
                "last_price": last_px,
                "market_value": mv,
                "unrealized_pnl": upnl,
                "allocation_pct": alloc,
            }
        )
    # Overwrite snapshot each run (single source of truth for "now").
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def internal_symbol_to_binance(internal_symbol: str) -> str:
    s = str(internal_symbol).strip().upper()
    if "(" in s:
        base = s.split("(")[0].strip()
    else:
        base = s.strip()
    base = TOKEN_ALIAS_TO_BINANCE.get(base, base)
    return f"{base}/USDT"


def align_with_min_history(data: Dict[str, pd.DataFrame], min_hist: int) -> Dict[str, pd.DataFrame]:
    working = dict(data)
    while len(working) >= 2:
        common = None
        for df in working.values():
            common = df.index if common is None else common.intersection(df.index)
        if common is None or len(common) == 0:
            shortest = min(working.items(), key=lambda kv: len(kv[1]))[0]
            del working[shortest]
            continue
        too_short = []
        aligned = {}
        for sym, df in working.items():
            adf = df.loc[common].copy().sort_index()
            if len(adf) < min_hist:
                too_short.append(sym)
            else:
                aligned[sym] = adf
        if not too_short:
            return aligned
        for sym in too_short:
            del working[sym]
    return {}


def current_utc_ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def drawdown_status(equity_path: Path, current_equity: float) -> Tuple[float, float]:
    """
    Returns (daily_drawdown, total_drawdown) as negative numbers on drawdown.
    """
    if not equity_path.exists():
        return 0.0, 0.0
    try:
        df = pd.read_csv(equity_path)
    except Exception:
        return 0.0, 0.0
    if df.empty or "timestamp" not in df.columns or "total_equity" not in df.columns:
        return 0.0, 0.0

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"])
    if df.empty:
        return 0.0, 0.0
    df["date"] = df["timestamp"].dt.date
    today = pd.Timestamp.now(tz="UTC").date()
    today_df = df[df["date"] == today]
    if today_df.empty:
        day_start_equity = float(current_equity)
    else:
        day_start_equity = float(today_df.iloc[0]["total_equity"])
    peak_equity = max(float(df["total_equity"].max()), float(current_equity))

    daily_dd = (float(current_equity) / day_start_equity - 1.0) if day_start_equity > 0 else 0.0
    total_dd = (float(current_equity) / peak_equity - 1.0) if peak_equity > 0 else 0.0
    return daily_dd, total_dd


def load_market_data(config: Config) -> Dict[str, pd.DataFrame]:
    dm = DataManager(config)
    symbol_to_file = build_symbol_file_map(config, Path(__file__).resolve().parent)
    loaded = dm.load_multi_csv(symbol_to_file)
    filtered, _, _ = dm.filter_by_allowed_quote_currency(loaded)
    # Same quality pre-filter as main.py: without it a stale symbol (MPL, data ends
    # May 2025) caps the aligned window and live signals come from 2025 prices.
    pre_filtered, _ = dm.pre_filter_quality(filtered)
    aligned = align_with_min_history(pre_filtered, int(config.data.min_history_days))
    if len(aligned) < 2:
        raise RuntimeError("Not enough market symbols after filtering/alignment.")
    latest = max(df.index.max() for df in aligned.values())
    age_days = (pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() - latest).days
    if age_days > 3:
        raise RuntimeError(
            f"Local CSV data is stale (latest bar {latest.date()}, {age_days}d old). Run: python update_data.py"
        )
    return aligned


def load_market_data_from_binance(
    config: Config,
    exchange: object,
    symbols: List[str],
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
    """
    Load daily OHLCV directly from Binance (testnet/public market data).
    Returns (data, errors_by_symbol). Data shape matches CSV loader output.
    """
    out: Dict[str, pd.DataFrame] = {}
    errors: Dict[str, str] = {}

    # Pull enough bars for indicator warmup.
    limit = max(int(config.data.min_history_days) + 120, 300)
    for internal_sym in symbols:
        bin_sym = internal_symbol_to_binance(internal_sym)
        try:
            if bin_sym not in exchange.markets:
                errors[internal_sym] = f"{bin_sym} not in Binance markets"
                continue
            raw = exchange.fetch_ohlcv(bin_sym, timeframe="1d", limit=limit)
            if not raw:
                errors[internal_sym] = "empty OHLCV response"
                continue
            df = pd.DataFrame(raw, columns=["timestamp_ms", "open", "high", "low", "close", "volume"])
            df["date"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
            df = df.drop(columns=["timestamp_ms"]).set_index("date").sort_index()
            for c in ["open", "high", "low", "close", "volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=["open", "high", "low", "close", "volume"])
            # Keep timezone-naive UTC-like index for compatibility with the existing engine/dataframes.
            df.index = df.index.tz_convert("UTC").tz_localize(None)
            if df.empty:
                errors[internal_sym] = "no valid OHLCV rows after cleanup"
                continue
            out[internal_sym] = df[["open", "high", "low", "close", "volume"]].copy()
        except Exception as exc:
            errors[internal_sym] = str(exc)

    dm = DataManager(config)
    filtered, _, _ = dm.filter_by_allowed_quote_currency(out)
    aligned = align_with_min_history(filtered, int(config.data.min_history_days))
    return aligned, errors


def _extract_fill_metrics(order: Dict[str, object], fallback_price: float) -> Dict[str, float]:
    """
    Extract filled amount, average fill price, notional cost, and fee from ccxt order payload.
    """
    filled = float(order.get("filled", 0.0) or 0.0)
    avg = float(order.get("average", 0.0) or 0.0)
    cost = float(order.get("cost", 0.0) or 0.0)

    if filled <= 0:
        info = order.get("info", {})
        if isinstance(info, dict):
            try:
                filled = float(info.get("executedQty", 0.0) or 0.0)
            except Exception:
                filled = 0.0
            try:
                cost = float(info.get("cummulativeQuoteQty", 0.0) or cost)
            except Exception:
                pass

    if avg <= 0:
        if filled > 0 and cost > 0:
            avg = cost / filled
        else:
            avg = float(fallback_price)
    if cost <= 0 and filled > 0 and avg > 0:
        cost = filled * avg

    fee_total = 0.0
    fee_obj = order.get("fee")
    if isinstance(fee_obj, dict):
        try:
            fee_total += float(fee_obj.get("cost", 0.0) or 0.0)
        except Exception:
            pass
    fees_list = order.get("fees")
    if isinstance(fees_list, list):
        for f in fees_list:
            if not isinstance(f, dict):
                continue
            try:
                fee_total += float(f.get("cost", 0.0) or 0.0)
            except Exception:
                continue

    return {
        "filled_qty": max(0.0, float(filled)),
        "avg_price": max(0.0, float(avg)),
        "cost": max(0.0, float(cost)),
        "fee": max(0.0, float(fee_total)),
    }


def _is_testnet_sandbox(exchange: object) -> bool:
    urls = getattr(exchange, "urls", None)
    if urls is None:
        return False
    return "testnet" in str(urls).lower()


def _market_min_notional(exchange: object, symbol: str) -> float:
    try:
        market = exchange.markets.get(symbol, {})
        limits = market.get("limits", {})
        cost_limits = limits.get("cost", {})
        min_cost = cost_limits.get("min")
        if min_cost is None:
            return 0.0
        return float(min_cost)
    except Exception:
        return 0.0


def _is_notional_filter_error(exc: Exception) -> bool:
    msg = str(exc).upper()
    return ("FILTER FAILURE: NOTIONAL" in msg) or ("NOTIONAL" in msg and "-1013" in msg)


def _format_hhmm(dt_utc: Optional[datetime]) -> str:
    if dt_utc is None:
        return "N/A"
    return dt_utc.astimezone(DISPLAY_TZ).strftime(f"%Y-%m-%d %H:%M {DISPLAY_TZ_LABEL}")


def _print_run_summary(run_number: int, summary: RunSummary, next_run_utc: Optional[datetime]) -> None:
    ts = pd.to_datetime(summary.timestamp_utc, errors="coerce", utc=True)
    ts_label = (
        ts.tz_convert(DISPLAY_TZ).strftime(f"%Y-%m-%d %H:%M {DISPLAY_TZ_LABEL}")
        if not pd.isna(ts)
        else summary.timestamp_utc
    )
    print(f"Run #{run_number}")
    print(f"Time: {ts_label}")
    print(f"Equity: ${summary.equity:,.2f}")
    print(f"Drawdown: {summary.drawdown_pct:.2%}")
    print(f"Trades: {summary.trades_executed}")
    print(f"Tax Reserve: ${summary.tax_reserve:,.2f}")
    if next_run_utc is not None:
        print(f"Next run: {_format_hhmm(next_run_utc)}")
    if summary.stop_reason:
        print(f"Stop reason: {summary.stop_reason}")


def _append_run_log(
    logs_dir: Path,
    run_number: int,
    summary: RunSummary,
    next_run_utc: Optional[datetime],
) -> None:
    run_log_path = logs_dir / "live_runner_runs.csv"
    headers = [
        "run_number",
        "timestamp",
        "equity",
        "drawdown",
        "trades_executed",
        "tax_reserve",
        "next_run_utc",
        "fatal_stop",
        "stop_reason",
    ]
    ensure_csv(run_log_path, headers)
    append_csv_row(
        run_log_path,
        {
            "run_number": run_number,
            "timestamp": summary.timestamp_utc,
            "equity": summary.equity,
            "drawdown": summary.drawdown_pct,
            "trades_executed": summary.trades_executed,
            "tax_reserve": summary.tax_reserve,
            "next_run_utc": next_run_utc.isoformat() if next_run_utc is not None else "",
            "fatal_stop": summary.fatal_stop,
            "stop_reason": summary.stop_reason,
        },
        headers,
    )


def _append_scheduler_stop_event(logs_dir: Path, run_number: int, reason: str) -> None:
    """
    Write a scheduler-level stop marker so graceful stops are visible in logs.
    """
    summary = RunSummary(
        timestamp_utc=current_utc_ts(),
        equity=float("nan"),
        drawdown_pct=float("nan"),
        trades_executed=0,
        tax_reserve=float("nan"),
        fatal_stop=False,
        stop_reason=reason,
    )
    _append_run_log(logs_dir=logs_dir, run_number=run_number, summary=summary, next_run_utc=None)


def _export_excel_backup(logs_dir: Path) -> None:
    """
    Export CSV logs into a single Excel workbook as a backup for dashboard data.
    This is read-only reporting and does not affect strategy/execution behavior.
    """
    backup_path = logs_dir / "trading_backup.xlsx"
    files = {
        "trades": logs_dir / "live_trades.csv",
        "equity": logs_dir / "paper_equity.csv",
        "tax_reserve": logs_dir / "tax_reserve.csv",
        "runner_runs": logs_dir / "live_runner_runs.csv",
    }

    dataframes: Dict[str, pd.DataFrame] = {}
    for sheet, path in files.items():
        if not path.exists():
            dataframes[sheet] = pd.DataFrame()
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            dataframes[sheet] = pd.DataFrame()
            continue

        # Add an ET display timestamp column for easier spreadsheet review.
        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
            df["timestamp_et"] = ts.dt.tz_convert(DISPLAY_TZ).dt.strftime("%Y-%m-%d %H:%M:%S")
        dataframes[sheet] = df

    # Derived trade summary sheet.
    trade_summary = pd.DataFrame()
    trades_df = dataframes.get("trades", pd.DataFrame())
    if trades_df is not None and not trades_df.empty:
        td = trades_df.copy()
        for c in ["realized_pnl", "tax_reserved", "order_value_usd", "fees_estimate"]:
            if c in td.columns:
                td[c] = pd.to_numeric(td[c], errors="coerce").fillna(0.0)
        if "timestamp" in td.columns:
            td["timestamp"] = pd.to_datetime(td["timestamp"], errors="coerce", utc=True)
            td["trade_date_et"] = td["timestamp"].dt.tz_convert(DISPLAY_TZ).dt.date
        if "realized_pnl" in td.columns:
            td["cumulative_realized_pnl"] = td["realized_pnl"].cumsum()
        trade_summary = pd.DataFrame(
            {
                "metric": [
                    "total_trades",
                    "total_realized_pnl",
                    "total_tax_reserved",
                    "total_fees_estimate",
                    "total_order_value_usd",
                ],
                "value": [
                    int(len(td)),
                    float(td["realized_pnl"].sum()) if "realized_pnl" in td.columns else 0.0,
                    float(td["tax_reserved"].sum()) if "tax_reserved" in td.columns else 0.0,
                    float(td["fees_estimate"].sum()) if "fees_estimate" in td.columns else 0.0,
                    float(td["order_value_usd"].sum()) if "order_value_usd" in td.columns else 0.0,
                ],
            }
        )

    # Add metadata sheet.
    metadata = pd.DataFrame(
        {
            "key": [
                "generated_utc",
                "generated_et",
                "timezone_display",
                "note",
            ],
            "value": [
                datetime.now(timezone.utc).isoformat(),
                datetime.now(DISPLAY_TZ).strftime("%Y-%m-%d %H:%M:%S"),
                str(DISPLAY_TZ),
                "Auto-generated backup for dashboard + audit. Source remains CSV logs.",
            ],
        }
    )

    try:
        with pd.ExcelWriter(backup_path) as writer:
            metadata.to_excel(writer, sheet_name="metadata", index=False)
            dataframes["trades"].to_excel(writer, sheet_name="trades", index=False)
            dataframes["equity"].to_excel(writer, sheet_name="equity", index=False)
            dataframes["tax_reserve"].to_excel(writer, sheet_name="tax_reserve", index=False)
            dataframes["runner_runs"].to_excel(writer, sheet_name="runner_runs", index=False)
            trade_summary.to_excel(writer, sheet_name="trade_summary", index=False)
    except Exception as exc:
        print(f"excel backup warning: could not export trading_backup.xlsx ({exc})")


def apply_anti_rug_guard(
    cfg: Config,
    logs_dir: Path,
    market_data: Dict[str, pd.DataFrame],
    paper: PaperAccount,
    tradable_targets: Dict[str, float],
    latest_prices: Dict[str, float],
) -> set:
    """
    Run the full anti-rug check on every meme we hold or want to buy.
    Mutates tradable_targets in place:
      - rug in progress (force_exit)  -> target removed, returned for a full exit
      - fails entry checks            -> no adds (target capped at current weight)
    Fail-closed: if the check itself errors, no meme buys happen this run.
    """
    meme_by_bin = {internal_symbol_to_binance(s): s for s in cfg.meme.meme_symbols}
    held = {s for s, p in paper.positions.items() if s in meme_by_bin and p.quantity > 0}
    wanted = {s for s, w in tradable_targets.items() if s in meme_by_bin and w > 0}
    to_check = sorted(held | wanted)
    if not to_check:
        return set()

    guard = AntiRugGuard(cfg, cache_path=logs_dir / "anti_rug_cache.json")
    equity = paper.total_equity(latest_prices)
    log_path = logs_dir / "anti_rug_log.csv"
    headers = ["timestamp", "symbol", "held", "target_weight", "allow_entry", "force_exit", "reasons"]
    rug_exits: set = set()
    print("anti-rug checks:")
    for bin_sym in to_check:
        target_w = float(tradable_targets.get(bin_sym, 0.0))
        try:
            verdict = guard.live_check(
                meme_by_bin[bin_sym],
                market_data.get(meme_by_bin[bin_sym]),
                order_usd=target_w * equity,
            )
        except Exception as exc:
            verdict = RugVerdict(symbol=bin_sym, allow_entry=False, reasons=[f"GUARD_ERROR ({exc})"])

        if verdict.force_exit and bin_sym in held:
            rug_exits.add(bin_sym)
            tradable_targets.pop(bin_sym, None)
            status = "FORCE EXIT"
        elif not verdict.allow_entry:
            px = float(latest_prices.get(bin_sym, 0.0))
            cur_w = paper.position_value(bin_sym, px) / equity if equity > 0 and px > 0 else 0.0
            capped = min(target_w, cur_w)
            if capped > 0:
                tradable_targets[bin_sym] = capped
            else:
                tradable_targets.pop(bin_sym, None)
            status = "BLOCKED"
        else:
            status = "PASS"
        print(f"- {bin_sym}: {status}" + (f" | {'; '.join(verdict.reasons)}" if verdict.reasons else ""))
        append_csv_row(
            log_path,
            {
                "timestamp": current_utc_ts(),
                "symbol": bin_sym,
                "held": bin_sym in held,
                "target_weight": target_w,
                "allow_entry": verdict.allow_entry,
                "force_exit": verdict.force_exit,
                "reasons": "; ".join(verdict.reasons),
            },
            headers,
        )
    return rug_exits


def run_once(cfg: Config, safety: LiveSafetyConfig) -> RunSummary:

    logs_dir = Path(__file__).resolve().parent / str(cfg.data.log_output_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    live_trades_path = logs_dir / "live_trades.csv"
    equity_path = logs_dir / "paper_equity.csv"
    tax_path = logs_dir / "tax_reserve.csv"
    positions_snapshot_path = logs_dir / "positions_snapshot.csv"

    trade_headers = [
        "timestamp",
        "symbol",
        "side",
        "quantity",
        "price",
        "order_value_usd",
        "fees_estimate",
        "realized_pnl",
        "tax_reserved",
        "remaining_cash",
        "total_equity",
    ]
    eq_headers = [
        "timestamp",
        "cash",
        "open_position_value",
        "tax_reserve",
        "total_equity",
        "drawdown",
    ]
    tax_headers = ["timestamp", "realized_profit", "tax_reserved", "total_tax_reserve"]
    ensure_csv(live_trades_path, trade_headers)
    ensure_csv(equity_path, eq_headers)
    ensure_csv(tax_path, tax_headers)

    executor = BinanceSpotTestnetExecutor(dry_run=False)
    exchange = executor.connect_to_exchange()
    if not _is_testnet_sandbox(exchange):
        raise RuntimeError("Fatal safety condition: sandbox mode appears OFF.")
    if str(exchange.options.get("defaultType", "")).lower() != "spot":
        raise RuntimeError("Fatal safety condition: exchange defaultType is not spot.")
    try:
        _ = executor.fetch_balance()
    except Exception as exc:
        raise RuntimeError(f"Fatal safety condition: API key/connectivity check failed: {exc}") from exc
    print("Binance Testnet connected")

    paper = PaperAccount.from_trade_log(
        trade_log_path=str(live_trades_path),
        starting_capital_usd=safety.starting_capital_usd,
        tax_reserve_rate=safety.tax_reserve_rate,
    )
    print(f"starting capital used: ${safety.starting_capital_usd:.2f}")

    executor._ensure_markets_loaded()
    symbol_to_file = build_symbol_file_map(cfg, Path(__file__).resolve().parent)
    candidate_symbols = sorted(symbol_to_file.keys())

    # Prefer live Binance OHLCV for signal generation to avoid stale local CSVs.
    market_data, ohlcv_errors = load_market_data_from_binance(cfg, exchange, candidate_symbols)
    if len(market_data) < 2:
        print("Warning: Binance OHLCV load returned insufficient symbols; falling back to local CSV market data.")
        market_data = load_market_data(cfg)
        print("Market data source: LOCAL_CSV_FALLBACK")
    else:
        print("Market data source: BINANCE_OHLCV")
    if ohlcv_errors:
        print("Binance OHLCV symbol notes:")
        for sym, reason in sorted(ohlcv_errors.items()):
            print(f"- {sym}: {reason}")

    # Keep only symbols whose mapped Binance spot symbols exist on testnet.
    tradable_market_data: Dict[str, pd.DataFrame] = {}
    skipped_unavailable: Dict[str, str] = {}
    for internal_sym, df in market_data.items():
        bin_sym = internal_symbol_to_binance(internal_sym)
        if bin_sym in exchange.markets:
            tradable_market_data[internal_sym] = df
        else:
            skipped_unavailable[internal_sym] = bin_sym

    if skipped_unavailable:
        print("Skipped unavailable testnet symbols (not in exchange markets):")
        for k, v in sorted(skipped_unavailable.items()):
            print(f"- {k} -> {v}")
    if len(tradable_market_data) >= 2:
        market_data = tradable_market_data
    else:
        print(
            "Warning: fewer than 2 tradable symbols on Binance testnet after filtering. "
            "Using full market_data fallback."
        )

    engine = BacktestEngine(cfg)
    common_dates = engine._common_dates(market_data)
    if not common_dates:
        raise RuntimeError("No common aligned market dates.")
    signal_date = common_dates[-1]

    # Existing strategy logic: universe + target allocation
    eligible_universe = engine._select_universe(market_data, signal_date)
    # Use current paper drawdown as risk overlay input.
    price_map_for_dd = {}
    for _, pos in paper.positions.items():
        try:
            t = exchange.fetch_ticker(pos.symbol)
            px = float(t.get("last") or t.get("close") or 0.0)
        except Exception:
            px = 0.0
        price_map_for_dd[pos.symbol] = px
    current_equity = paper.total_equity(price_map_for_dd)
    _, total_dd = drawdown_status(equity_path, current_equity)
    target_weights = engine._target_weights_with_risk_overlays(
        market_data=market_data,
        date=signal_date,
        current_drawdown=float(total_dd),
        eligible_symbols=eligible_universe,
    )

    # Convert target allocations to Binance symbols where possible.
    executor._ensure_markets_loaded()  # reuse validated/loaded market list
    tradable_targets: Dict[str, float] = {}
    skipped_symbols: Dict[str, str] = {}
    for internal_sym, w in target_weights.items():
        if w <= 0:
            continue
        bin_sym = internal_symbol_to_binance(internal_sym)
        if bin_sym not in exchange.markets:
            skipped_symbols[internal_sym] = f"{bin_sym} not available on Binance Spot Testnet"
            continue
        tradable_targets[bin_sym] = tradable_targets.get(bin_sym, 0.0) + float(w)

    if skipped_symbols:
        print("Skipped non-tradable symbols:")
        for k, v in sorted(skipped_symbols.items()):
            print(f"- {k}: {v}")

    # Normalize tradable target weights to <= max_total_exposure budget.
    weight_sum = sum(max(0.0, x) for x in tradable_targets.values())
    if weight_sum > 0:
        scale = min(1.0, safety.max_total_exposure_pct / weight_sum)
        tradable_targets = {k: max(0.0, v) * scale for k, v in tradable_targets.items()}
    print("target allocations:")
    print(tradable_targets)

    # Fetch latest prices for required symbols.
    latest_prices: Dict[str, float] = {}
    for sym in sorted(set(list(tradable_targets.keys()) + list(paper.positions.keys()))):
        try:
            t = exchange.fetch_ticker(sym)
            px = float(t.get("last") or t.get("close") or 0.0)
        except Exception:
            px = 0.0
        latest_prices[sym] = px

    # Anti-rug-pull layer for meme coins (live order book + on-chain contract + history).
    rug_exit_symbols = apply_anti_rug_guard(
        cfg=cfg,
        logs_dir=logs_dir,
        market_data=market_data,
        paper=paper,
        tradable_targets=tradable_targets,
        latest_prices=latest_prices,
    )

    # Combine strategy targeting with profit-funded core accumulation.
    # Rule: allocate 65% of profits (above starting capital) into core hold symbols.
    current_equity = paper.total_equity(latest_prices)
    profit_pool = max(0.0, float(current_equity) - float(safety.starting_capital_usd))
    profit_core_budget = float(profit_pool) * float(safety.profit_to_core_hold_fraction)
    eligible_core = [
        s
        for s in safety.core_hold_symbols
        if (s in exchange.markets) and (float(latest_prices.get(s, 0.0)) > 0.0)
    ]
    if eligible_core and current_equity > 0 and profit_core_budget > 0:
        per_symbol_budget = profit_core_budget / float(len(eligible_core))
        for sym in eligible_core:
            floor_weight = per_symbol_budget / float(current_equity)
            tradable_targets[sym] = max(float(tradable_targets.get(sym, 0.0)), float(floor_weight))
        # Re-apply exposure normalization after adding core floors.
        weight_sum = sum(max(0.0, x) for x in tradable_targets.values())
        if weight_sum > 0:
            scale = min(1.0, safety.max_total_exposure_pct / weight_sum)
            tradable_targets = {k: max(0.0, v) * scale for k, v in tradable_targets.items()}
    print(
        "profit hold overlay: "
        f"profit_pool=${profit_pool:.2f}, "
        f"core_budget(65%)=${profit_core_budget:.2f}, "
        f"core_symbols={eligible_core if eligible_core else list(safety.core_hold_symbols)}"
    )
    print("combined target allocations:")
    print(tradable_targets)

    # Stop checks before trading.
    current_equity = paper.total_equity(latest_prices)
    daily_dd, total_dd = drawdown_status(equity_path, current_equity)
    if daily_dd <= -safety.stop_daily_drawdown_pct:
        reason = (
            f"safety checks failed: daily drawdown {daily_dd:.2%} breached "
            f"{safety.stop_daily_drawdown_pct:.2%}. stopping."
        )
        print(reason)
        return RunSummary(
            timestamp_utc=current_utc_ts(),
            equity=float(current_equity),
            drawdown_pct=float(total_dd),
            trades_executed=0,
            tax_reserve=float(paper.tax_manager.total_tax_reserve),
            fatal_stop=True,
            stop_reason=reason,
        )
    if total_dd <= -safety.stop_total_drawdown_pct:
        reason = (
            f"safety checks failed: total drawdown {total_dd:.2%} breached "
            f"{safety.stop_total_drawdown_pct:.2%}. stopping."
        )
        print(reason)
        return RunSummary(
            timestamp_utc=current_utc_ts(),
            equity=float(current_equity),
            drawdown_pct=float(total_dd),
            trades_executed=0,
            tax_reserve=float(paper.tax_manager.total_tax_reserve),
            fatal_stop=True,
            stop_reason=reason,
        )

    # Build proposed trades (single step toward target, capped per order).
    proposed_trades = []
    for sym, tgt_w in tradable_targets.items():
        px = float(latest_prices.get(sym, 0.0))
        if px <= 0:
            continue
        equity_now = paper.total_equity(latest_prices)
        target_value = float(tgt_w) * equity_now
        current_value = paper.position_value(sym, px)
        delta = target_value - current_value
        if abs(delta) < 0.25:
            continue

        dyn_order_cap = safety.order_limit(equity_now)
        dyn_pos_cap = safety.position_limit(equity_now)
        if delta > 0:
            allowed_by_position = max(0.0, dyn_pos_cap - current_value)
            order_value = min(delta, dyn_order_cap, allowed_by_position)
            side = "buy"
        else:
            order_value = min(abs(delta), dyn_order_cap, current_value)
            side = "sell"
        if order_value <= 0:
            continue
        min_notional = _market_min_notional(exchange, sym)
        if min_notional > 0 and order_value < min_notional:
            print(
                f"skipping {sym} {side}: order notional ${order_value:.4f} "
                f"below exchange minimum ${min_notional:.4f}"
            )
            continue
        qty = order_value / px
        if qty <= 0:
            continue
        proposed_trades.append(
            {"symbol": sym, "side": side, "quantity": qty, "price": px, "order_value_usd": order_value}
        )

    # Also consider trimming symbols not in current target.
    for sym, pos in list(paper.positions.items()):
        if sym in tradable_targets:
            continue
        px = float(latest_prices.get(sym, 0.0))
        if px <= 0 or pos.quantity <= 0:
            continue
        current_value = pos.quantity * px
        dyn_order_cap_exit = safety.order_limit(paper.total_equity(latest_prices))
        # Rug in progress: dump the whole position now, not one capped slice per run.
        order_value = current_value if sym in rug_exit_symbols else min(current_value, dyn_order_cap_exit)
        min_notional = _market_min_notional(exchange, sym)
        if min_notional > 0 and order_value < min_notional:
            print(
                f"skipping {sym} sell: order notional ${order_value:.4f} "
                f"below exchange minimum ${min_notional:.4f}"
            )
            continue
        qty = order_value / px
        if qty > 0:
            proposed_trades.append(
                {"symbol": sym, "side": "sell", "quantity": qty, "price": px, "order_value_usd": order_value}
            )

    print("proposed trades:")
    for t in proposed_trades:
        print(t)

    # Execute with full safety checks.
    any_order = False
    trades_executed = 0
    for t in proposed_trades:
        sym = str(t["symbol"])
        side = str(t["side"])
        qty = float(t["quantity"])
        px = float(t["price"])
        order_value = float(t["order_value_usd"])
        fee_estimate = order_value * float(cfg.portfolio.fee_rate)

        # Safety: symbol format validation + order caps.
        try:
            executor._validate_symbol(sym)
        except Exception:
            print(f"safety checks failed for {sym}: invalid symbol format")
            continue
        live_equity = paper.total_equity(latest_prices)
        dyn_order_limit = safety.order_limit(live_equity)
        dyn_pos_limit = safety.position_limit(live_equity)
        is_rug_exit = side == "sell" and sym in rug_exit_symbols
        if order_value > dyn_order_limit + 1e-12 and not is_rug_exit:
            print(f"safety checks failed for {sym}: order ${order_value:.2f} exceeds dynamic cap ${dyn_order_limit:.2f}")
            continue

        # Projected state checks.
        projected_cash = paper.cash_usd
        projected_position_value = paper.open_position_value(latest_prices)
        cur_pos_value = paper.position_value(sym, px)
        if side == "buy":
            projected_cash -= (order_value + fee_estimate)
            projected_position_value += order_value
            new_pos_value = cur_pos_value + order_value
        else:
            projected_cash += (order_value - fee_estimate)
            projected_position_value -= order_value
            new_pos_value = max(0.0, cur_pos_value - order_value)

        projected_equity = projected_cash + projected_position_value + paper.tax_manager.total_tax_reserve
        if projected_equity <= 0:
            print(f"safety checks failed for {sym}: projected equity invalid")
            continue
        if new_pos_value > dyn_pos_limit + 1e-12:
            print(f"safety checks failed for {sym}: position ${new_pos_value:.2f} exceeds dynamic cap ${dyn_pos_limit:.2f}")
            continue
        if projected_position_value / projected_equity > safety.max_total_exposure_pct + 1e-12:
            print(f"safety checks failed for {sym}: max_total_exposure_pct exceeded")
            continue
        if projected_cash / projected_equity < safety.min_cash_buffer_pct - 1e-12:
            print(f"safety checks failed for {sym}: min_cash_buffer_pct violated")
            continue

        print(f"safety checks passed: {sym} {side} {qty:.8f}")

        # Send order to Binance testnet (dry_run=False, sandbox=True).
        try:
            if side == "buy":
                order = executor.place_market_buy(sym, qty)
            else:
                order = executor.place_market_sell(sym, qty)
        except Exception as exc:
            if _is_notional_filter_error(exc):
                print(f"skipping {sym}: exchange NOTIONAL filter rejected small order ({exc})")
                continue
            reason = f"API failure while placing order for {sym}: {exc}"
            print(reason)
            return RunSummary(
                timestamp_utc=current_utc_ts(),
                equity=float(paper.total_equity(latest_prices)),
                drawdown_pct=float(total_dd),
                trades_executed=trades_executed,
                tax_reserve=float(paper.tax_manager.total_tax_reserve),
                fatal_stop=True,
                stop_reason=reason,
            )
        fill_exec = _extract_fill_metrics(order, fallback_price=px)
        filled_qty = float(fill_exec["filled_qty"])
        fill_price = float(fill_exec["avg_price"])
        fill_cost = float(fill_exec["cost"])
        fee_paid = float(fill_exec["fee"])
        if filled_qty <= 0:
            print(f"order status: no filled quantity for {sym}, skipping paper accounting update")
            continue

        # Guard for floating precision / step-size differences.
        if side == "sell":
            pos_obj = paper.positions.get(sym)
            if pos_obj is not None:
                filled_qty = min(filled_qty, float(pos_obj.quantity))
                fill_cost = filled_qty * fill_price
            if filled_qty <= 0:
                print(f"order status: sell fill clipped to zero for {sym}, skipping")
                continue

        any_order = True
        trades_executed += 1
        print(
            f"order status: {order.get('status', 'unknown')} | symbol={sym} side={side} "
            f"filled_qty={filled_qty:.8f} avg_fill_price={fill_price:.8f} cost={fill_cost:.4f}"
        )

        fill = paper.apply_fill(
            symbol=sym,
            side=side,
            quantity=filled_qty,
            price=fill_price,
            fee=fee_paid,
        )
        latest_prices[sym] = fill_price
        total_equity = paper.total_equity(latest_prices)
        ts = current_utc_ts()

        append_csv_row(
            live_trades_path,
            {
                "timestamp": ts,
                "symbol": sym,
                "side": side,
                "quantity": filled_qty,
                "price": fill_price,
                "order_value_usd": fill_cost if fill_cost > 0 else order_value,
                "fees_estimate": fee_paid if fee_paid > 0 else fee_estimate,
                "realized_pnl": fill["realized_pnl"],
                "tax_reserved": fill["tax_reserved"],
                "remaining_cash": fill["cash_usd"],
                "total_equity": total_equity,
            },
            trade_headers,
        )
        if fill["tax_reserved"] > 0:
            append_csv_row(
                tax_path,
                {
                    "timestamp": ts,
                    "realized_profit": max(0.0, float(fill["realized_pnl"])),
                    "tax_reserved": float(fill["tax_reserved"]),
                    "total_tax_reserve": float(fill["total_tax_reserve"]),
                },
                tax_headers,
            )

    # Final equity snapshot log.
    final_equity = paper.total_equity(latest_prices)
    _, final_total_dd = drawdown_status(equity_path, final_equity)
    append_csv_row(
        equity_path,
        {
            "timestamp": current_utc_ts(),
            "cash": paper.cash_usd,
            "open_position_value": paper.open_position_value(latest_prices),
            "tax_reserve": paper.tax_manager.total_tax_reserve,
            "total_equity": final_equity,
            "drawdown": final_total_dd,
        },
        eq_headers,
    )

    # Refresh mark-to-market positions snapshot every run.
    write_positions_snapshot(
        path=positions_snapshot_path,
        timestamp_utc=current_utc_ts(),
        paper=paper,
        latest_prices=latest_prices,
    )

    open_positions = {
        s: {"qty": p.quantity, "avg_entry": p.avg_entry_price}
        for s, p in sorted(paper.positions.items())
    }
    print(f"current paper equity: ${current_equity:,.2f}")
    print(f"paper cash remaining: ${paper.cash_usd:,.2f}")
    print(f"open positions: {open_positions if open_positions else 'none'}")
    print(f"tax reserve balance: ${paper.tax_manager.total_tax_reserve:,.2f}")
    print(f"final equity after run: ${final_equity:,.2f}")
    if not any_order:
        print("orders sent: none")

    # Maintain an Excel backup on each run for external audit/review.
    _export_excel_backup(logs_dir)

    return RunSummary(
        timestamp_utc=current_utc_ts(),
        equity=float(final_equity),
        drawdown_pct=float(final_total_dd),
        trades_executed=trades_executed,
        tax_reserve=float(paper.tax_manager.total_tax_reserve),
        fatal_stop=False,
        stop_reason="",
    )


def _run_side_sleeves(cfg: Config) -> None:
    """News refresh + ML paper sleeve. Isolated: any failure here never affects V7."""
    news = None
    try:
        from news_feed import news_snapshot

        news = news_snapshot()
        flagged = {c: v["level"] for c, v in news["coin_risk"].items()}
        print(f"news: Fear&Greed={news['fear_greed']} headlines={news['headlines']} "
              f"(+{news['new_archived']} archived) flagged={flagged or 'none'}")
    except Exception as exc:
        print(f"news plug-in error (trading unaffected): {exc}")
    try:
        from ml_live import run_ml_cycle

        run_ml_cycle(Config(), news)
    except Exception as exc:
        print(f"ML sleeve error (V7 unaffected): {exc}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run crypto quant paper runner once or on a fixed schedule.")
    parser.add_argument("--loop", action="store_true", help="Enable scheduled loop mode.")
    parser.add_argument(
        "--interval-minutes",
        type=int,
        default=60,
        help="Loop interval in minutes (used only when --loop is set).",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=0,
        help="Stop after this many runs in loop mode. 0 means no max-runs limit.",
    )
    parser.add_argument(
        "--stop-after-hours",
        type=float,
        default=0.0,
        help="Stop loop after this many hours. 0 means no time limit.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.interval_minutes <= 0:
        raise ValueError("--interval-minutes must be > 0")
    if args.max_runs < 0:
        raise ValueError("--max-runs must be >= 0")
    if args.stop_after_hours < 0:
        raise ValueError("--stop-after-hours must be >= 0")

    # Load .env from project root (if present), then enforce explicit key checks.
    _load_env_file(Path(__file__).resolve().parent / ".env")

    # Explicit safety pre-check for API key presence.
    if not os.getenv("BINANCE_API_KEY") or not os.getenv("BINANCE_SECRET"):
        raise RuntimeError("Fatal safety condition: missing BINANCE_API_KEY/BINANCE_SECRET environment variables.")

    cfg = Config()
    # Yield is simulated (not real staking) — keep off for live runner.
    cfg.strategy.yield_enabled = False
    # Momentum is enabled by default in Config; override here only if you want to suppress it.
    # cfg.strategy.momentum_enabled = False
    safety = LiveSafetyConfig()
    logs_dir = Path(__file__).resolve().parent / str(cfg.data.log_output_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    emergency_stop_flag = logs_dir / "emergency_stop.flag"

    if not args.loop:
        summary = run_once(cfg, safety)
        _print_run_summary(run_number=1, summary=summary, next_run_utc=None)
        _append_run_log(logs_dir=logs_dir, run_number=1, summary=summary, next_run_utc=None)
        _run_side_sleeves(cfg)
        if summary.fatal_stop:
            sys.exit(EXIT_SAFETY_STOP)
        return

    interval = timedelta(minutes=args.interval_minutes)
    loop_start = datetime.now(timezone.utc)
    run_number = 0
    consecutive_errors = 0
    max_consecutive_errors = 3
    print(
        f"Loop mode enabled | interval={args.interval_minutes}m "
        f"| max_runs={args.max_runs if args.max_runs > 0 else 'unbounded'} "
        f"| stop_after_hours={args.stop_after_hours if args.stop_after_hours > 0 else 'unbounded'}"
    )
    try:
        while True:
            if emergency_stop_flag.exists():
                print("Emergency stop flag detected. stopping loop safely.")
                _append_scheduler_stop_event(
                    logs_dir=logs_dir,
                    run_number=run_number,
                    reason="EMERGENCY_STOP_FLAG",
                )
                break
            now = datetime.now(timezone.utc)
            if args.stop_after_hours > 0 and (now - loop_start).total_seconds() >= args.stop_after_hours * 3600.0:
                print("stop_after_hours reached. stopping loop.")
                _append_scheduler_stop_event(
                    logs_dir=logs_dir,
                    run_number=run_number,
                    reason="STOP_AFTER_HOURS_REACHED",
                )
                break
            if args.max_runs > 0 and run_number >= args.max_runs:
                print("max_runs reached. stopping loop.")
                _append_scheduler_stop_event(
                    logs_dir=logs_dir,
                    run_number=run_number,
                    reason="MAX_RUNS_REACHED",
                )
                break

            run_number += 1
            try:
                summary = run_once(cfg, safety)
                consecutive_errors = 0
            except Exception as exc:
                consecutive_errors += 1
                reason = f"RUN_EXCEPTION: {exc}"
                print(f"Run #{run_number} failed: {exc}")
                summary = RunSummary(
                    timestamp_utc=current_utc_ts(),
                    equity=float("nan"),
                    drawdown_pct=float("nan"),
                    trades_executed=0,
                    tax_reserve=float("nan"),
                    fatal_stop=False,
                    stop_reason=reason,
                )
                # Log failed cycle and continue unless repeated failures suggest a hard outage.
                next_run_err = loop_start + interval * run_number
                _append_run_log(logs_dir=logs_dir, run_number=run_number, summary=summary, next_run_utc=next_run_err)
                if consecutive_errors >= max_consecutive_errors:
                    print(
                        f"Fatal scheduler condition: {consecutive_errors} consecutive run errors. stopping loop."
                    )
                    _append_scheduler_stop_event(
                        logs_dir=logs_dir,
                        run_number=run_number,
                        reason="MAX_CONSECUTIVE_RUN_ERRORS_REACHED",
                    )
                    sys.exit(1)
                sleep_seconds = max(30.0, min(interval.total_seconds(), 300.0))
                print(f"Retrying after {sleep_seconds:.0f}s (consecutive errors: {consecutive_errors}).")
                time.sleep(sleep_seconds)
                continue

            next_run = None
            if not summary.fatal_stop:
                next_run = loop_start + interval * run_number
            _print_run_summary(run_number=run_number, summary=summary, next_run_utc=next_run)
            _append_run_log(logs_dir=logs_dir, run_number=run_number, summary=summary, next_run_utc=next_run)
            _run_side_sleeves(cfg)

            if summary.fatal_stop:
                print("Fatal safety condition triggered. loop stopped.")
                sys.exit(EXIT_SAFETY_STOP)

            if args.max_runs > 0 and run_number >= args.max_runs:
                print("max_runs reached. stopping loop.")
                _append_scheduler_stop_event(
                    logs_dir=logs_dir,
                    run_number=run_number,
                    reason="MAX_RUNS_REACHED",
                )
                break
            if args.stop_after_hours > 0:
                elapsed = datetime.now(timezone.utc) - loop_start
                if elapsed.total_seconds() >= args.stop_after_hours * 3600.0:
                    print("stop_after_hours reached. stopping loop.")
                    _append_scheduler_stop_event(
                        logs_dir=logs_dir,
                        run_number=run_number,
                        reason="STOP_AFTER_HOURS_REACHED",
                    )
                    break

            # Anchored schedule to avoid timing drift.
            sleep_seconds = (next_run - datetime.now(timezone.utc)).total_seconds() if next_run else 0.0
            if sleep_seconds > 0:
                print(
                    "Sleeping "
                    f"{sleep_seconds:.0f}s until next run at "
                    f"{next_run.astimezone(DISPLAY_TZ).strftime(f'%Y-%m-%d %H:%M:%S {DISPLAY_TZ_LABEL}')}"
                )
                # Sleep in 60s steps against the wall clock: Windows does not count
                # time spent in laptop sleep toward one long time.sleep().
                while datetime.now(timezone.utc) < next_run and not emergency_stop_flag.exists():
                    time.sleep(min(60.0, max(1.0, (next_run - datetime.now(timezone.utc)).total_seconds())))
            else:
                print("Next run time already passed; starting next cycle immediately.")
    except KeyboardInterrupt:
        print("Bot stopped safely")
        _append_scheduler_stop_event(
            logs_dir=logs_dir,
            run_number=run_number,
            reason="MANUAL_STOP_CTRL_C",
        )


if __name__ == "__main__":
    main()
