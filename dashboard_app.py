from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
try:
    from streamlit_autorefresh import st_autorefresh  # type: ignore
except Exception:
    st_autorefresh = None

st.set_page_config(
    page_title="Crypto Quant Command Center",
    page_icon="C",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------
# Paths (read-only)
# ----------------------------
LOG_DIR = Path("logs")
PATHS = {
    "trades": LOG_DIR / "live_trades.csv",
    "equity": LOG_DIR / "paper_equity.csv",
    "equity_legacy": Path("forward_equity.csv"),
    "tax": LOG_DIR / "tax_reserve.csv",
    "positions_snapshot": LOG_DIR / "positions_snapshot.csv",
    "audit": Path("forward_audit.json"),
    "signals": Path("forward_signal_trace.csv"),
}
DISPLAY_TZ = ZoneInfo("America/New_York")
DISPLAY_TZ_LABEL = "ET"
FORWARD_SIGNAL_LOOKBACK_DAYS = 14
LONG_TERM_CORE_BASES = ("XRP", "HBAR", "ONDO")

# Starting capital — used as the baseline for all return % calculations.
STARTING_CAPITAL = 500.0
LIVE_START_DATE = "2026-04-26"

# V7 strategy description (display only)
STRATEGY_INFO = {
    "name": "V7 HighBeta + ONDO Sector Rotation",
    "sectors": "HighBeta (XRP + HBAR) | ONDO",
    "signal": "Sharpe14 > 0.1 gate (else cash)",
    "rebalance": "Every 2 days",
    "backtest_cagr": "+34.3%",
    "backtest_sharpe": "1.30",
    "backtest_maxdd": "-20.0%",
    "backtest_period": "Jan 2024 – Jun 2026 (2.43 yrs)",
}

# Display-only risk constants — kept in sync with LiveSafetyConfig in live_runner.py.
RISK_DISPLAY = {
    "Max Order Size (USD)": 60.0,
    "Max Position Size (USD)": 180.0,
    "Max Exposure %": 95.0,
    "Daily DD Stop %": 12.0,
    "Total DD Stop %": 30.0,
}


def inject_css() -> None:
    st.markdown(
        """
<style>
:root {
  --bg: #080808;
  --card: #111111;
  --card2: #151515;
  --text: #f2f2f2;
  --muted: #9a9a9a;
  --border: #2a2a2a;
  --green: #00c076;
  --red: #e03d3d;
  --white: #ffffff;
}
html, body, [class*="css"] {
  font-family: Consolas, 'Courier New', monospace !important;
  font-variant-numeric: tabular-nums;
}
.stApp { background: var(--bg); color: var(--text); }
[data-testid="stSidebar"] > div:first-child {
  background: var(--bg);
  border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
.block-container { max-width: 96% !important; padding-top: 0.8rem !important; }
[data-testid="stHeader"] { background: transparent; }
.hdr {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 14px 16px;
  margin-bottom: 14px;
}
.hdr-title {
  letter-spacing: .14em;
  text-transform: uppercase;
  color: var(--white);
  font-size: 18px;
}
.hdr-sub { color: var(--muted); font-size: 12px; margin-top: 6px; }
.badge {
  display: inline-block;
  border: 1px solid var(--border);
  background: var(--card2);
  color: var(--text);
  border-radius: 999px;
  padding: 3px 9px;
  font-size: 10px;
  letter-spacing: .1em;
  text-transform: uppercase;
  margin-right: 6px;
}
.badge-green { color: var(--green); border-color: rgba(0,192,118,.45); background: rgba(0,192,118,.08); }
.badge-red { color: var(--red); border-color: rgba(224,61,61,.45); background: rgba(224,61,61,.08); }
.section-title {
  letter-spacing: .14em;
  text-transform: uppercase;
  color: var(--text);
  font-size: 13px;
  margin: 18px 0 8px;
}
.metric-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(150px, 1fr));
  gap: 10px;
  margin-bottom: 10px;
}
.metric-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px;
}
.metric-label {
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: .1em;
  font-size: 10px;
}
.metric-value {
  color: var(--white);
  font-size: 22px;
  margin-top: 6px;
}
.val-pos { color: var(--green); }
.val-neg { color: var(--red); }
.box {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 10px 12px;
  margin: 6px 0;
}
.warn { border-left: 3px solid #c8a13f; color: #e7c872; }
.info { border-left: 3px solid #555; color: var(--muted); }
</style>
        """,
        unsafe_allow_html=True,
    )


def warn(msg: str) -> None:
    st.markdown(f'<div class="box warn">{msg}</div>', unsafe_allow_html=True)


def info(msg: str) -> None:
    st.markdown(f'<div class="box info">{msg}</div>', unsafe_allow_html=True)


def load_csv(path: Path) -> Tuple[pd.DataFrame | None, str | None]:
    if not path.exists():
        return None, f"Missing file: {path}"
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        return None, f"Malformed CSV {path}: {exc}"
    if df.empty:
        return df, f"Empty file: {path}"
    return df, None


def load_json(path: Path) -> Tuple[dict | None, str | None]:
    if not path.exists():
        return None, f"Missing file: {path}"
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f), None
    except Exception as exc:
        return None, f"Malformed JSON {path}: {exc}"


def to_ts(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def to_display_ts(series: pd.Series) -> pd.Series:
    return to_ts(series).dt.tz_convert(DISPLAY_TZ)


def fmt_money(v: float | None) -> str:
    if v is None or pd.isna(v):
        return "N/A"
    return f"${v:,.2f}"


def fmt_pct(v: float | None) -> str:
    if v is None or pd.isna(v):
        return "N/A"
    return f"{v:.2f}%"


def drawdown_pct(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return (equity / peak - 1.0) * 100.0


def prepare_equity_history(df: pd.DataFrame, eq_col: str, ts_col: str) -> pd.DataFrame:
    """
    Normalize equity history for accurate charting/metrics:
    - parse timestamps in UTC then convert to display tz
    - parse equity values as numeric
    - drop invalid rows
    - sort chronologically
    - keep the latest record per timestamp
    """
    out = df.copy()
    out[ts_col] = to_display_ts(out[ts_col])
    out[eq_col] = pd.to_numeric(out[eq_col], errors="coerce")
    out = out.dropna(subset=[ts_col, eq_col]).sort_values(ts_col)
    out = out.drop_duplicates(subset=[ts_col], keep="last")
    return out


def merge_equity_history(
    primary_df: pd.DataFrame | None,
    legacy_df: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """
    Merge current paper equity with legacy forward equity so chart history remains continuous.
    """
    frames: List[pd.DataFrame] = []
    if primary_df is not None and not primary_df.empty:
        frames.append(primary_df.copy())
    if legacy_df is not None and not legacy_df.empty:
        frames.append(legacy_df.copy())
    if not frames:
        return None

    merged = pd.concat(frames, ignore_index=True, sort=False)
    ts_col = "timestamp" if "timestamp" in merged.columns else None
    if ts_col is None:
        ts_col = next((c for c in merged.columns if "time" in c.lower() or "date" in c.lower()), None)
    eq_col = "total_equity" if "total_equity" in merged.columns else None
    if eq_col is None:
        eq_col = next((c for c in merged.columns if "equity" in c.lower()), None)
    if ts_col is None or eq_col is None:
        return merged

    merged[ts_col] = to_ts(merged[ts_col])
    merged[eq_col] = pd.to_numeric(merged[eq_col], errors="coerce")
    merged = merged.dropna(subset=[ts_col, eq_col]).sort_values(ts_col)
    merged = merged.drop_duplicates(subset=[ts_col], keep="last")
    return merged


def metric_card(label: str, value: str, sign: str = "") -> str:
    cls = "val-pos" if sign == "pos" else ("val-neg" if sign == "neg" else "")
    return (
        '<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value {cls}">{value}</div>'
        "</div>"
    )


def build_positions_from_trades(trades: pd.DataFrame) -> pd.DataFrame:
    req = {"symbol", "side", "quantity", "price"}
    if trades is None or trades.empty or not req.issubset(trades.columns):
        return pd.DataFrame()

    df = trades.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = to_ts(df["timestamp"])
        df = df.sort_values("timestamp")

    positions: Dict[str, Dict[str, float]] = {}
    for _, r in df.iterrows():
        sym = str(r["symbol"])
        side = str(r["side"]).strip().lower()
        qty = float(pd.to_numeric(r.get("quantity", 0), errors="coerce") or 0)
        px = float(pd.to_numeric(r.get("price", 0), errors="coerce") or 0)
        if qty <= 0 or px <= 0:
            continue

        pos = positions.get(sym, {"qty": 0.0, "avg": 0.0})
        if side == "buy":
            new_qty = pos["qty"] + qty
            pos["avg"] = (pos["qty"] * pos["avg"] + qty * px) / new_qty if new_qty > 0 else 0.0
            pos["qty"] = new_qty
        elif side == "sell":
            pos["qty"] = max(0.0, pos["qty"] - qty)
            if pos["qty"] == 0:
                pos["avg"] = 0.0
        positions[sym] = pos

    rows = [{"Symbol": s, "Qty": p["qty"], "Entry Price": p["avg"]} for s, p in positions.items() if p["qty"] > 0]
    return pd.DataFrame(rows)


inject_css()

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.markdown("### Controls")
    auto_refresh = st.checkbox("Auto-refresh", value=True)
    refresh_interval = st.selectbox("Refresh interval", ["15s", "30s", "60s", "Manual"], index=1)
    interval_map = {"15s": 15, "30s": 30, "60s": 60, "Manual": 0}
    interval_sec = interval_map[refresh_interval]
    info("Chart scaling is now controlled directly on the chart (zoom/pan/reset).")
    if auto_refresh and interval_sec > 0:
        if st_autorefresh is not None:
            st_autorefresh(interval=interval_sec * 1000, key="dashboard_autorefresh")
        else:
            info("Auto-refresh package not available. Use Refresh button.")

    section_overview = st.checkbox("Overview", value=True)
    section_positions = st.checkbox("Positions", value=True)
    section_long_term_core = st.checkbox("Long-Term Core", value=True)
    section_trades = st.checkbox("Trades", value=True)
    section_signals = st.checkbox("Signals", value=True)
    section_risk = st.checkbox("Risk", value=True)
    section_optimization = st.checkbox("Optimization Data", value=False)

    st.markdown("---")
    range_choice = st.selectbox("Time Range", ["ALL", "1D", "1W", "1M"], index=0)

    if st.button("Refresh"):
        st.rerun()

# Keep the dashboard updating automatically by default so each live-run write
# to logs (equity/trades/positions) appears without manual page reload.
if auto_refresh and interval_sec > 0:
    st.caption(f"Auto-refresh active: every {interval_sec}s")
else:
    st.caption("Auto-refresh is off (manual refresh mode)")

# ----------------------------
# Load data
# ----------------------------
trades_df, trades_err = load_csv(PATHS["trades"])
equity_df, equity_err = load_csv(PATHS["equity"])
legacy_equity_df, legacy_equity_err = load_csv(PATHS["equity_legacy"])
signals_df, signals_err = load_csv(PATHS["signals"])
positions_snap_df, positions_snap_err = load_csv(PATHS["positions_snapshot"])
audit, audit_err = load_json(PATHS["audit"])

for msg in [trades_err, equity_err, signals_err, audit_err]:
    if msg:
        if "Missing" in msg or "Empty" in msg:
            warn(msg)
        else:
            info(msg)
# positions snapshot is optional; ignore missing/empty warnings.
if positions_snap_err and "Malformed" in positions_snap_err:
    info(positions_snap_err)
# Legacy equity file is optional; only surface malformed errors.
if legacy_equity_err and "Malformed" in legacy_equity_err:
    info(legacy_equity_err)

merged_equity_df = merge_equity_history(equity_df, legacy_equity_df)
if merged_equity_df is not None and not merged_equity_df.empty:
    equity_df = merged_equity_df

# ----------------------------
# Core metrics from real logs
# ----------------------------
current_equity = start_equity = total_return_pct = current_dd = max_dd = None
cash = open_value = None
realized_pnl = unrealized_pnl = None
trade_count = trades_today = 0
win_rate = None
last_updated = None
forward_start_utc = None

if equity_df is not None and not equity_df.empty:
    if "timestamp" in equity_df.columns:
        equity_df["timestamp"] = to_ts(equity_df["timestamp"])
        equity_df = equity_df.dropna(subset=["timestamp"]).sort_values("timestamp")
        if not equity_df.empty:
            last_updated = equity_df["timestamp"].iloc[-1]

    eq_col = "total_equity" if "total_equity" in equity_df.columns else None
    if eq_col is None:
        eq_col = next((c for c in equity_df.columns if "equity" in c.lower()), None)

    if eq_col:
        eq = pd.to_numeric(equity_df[eq_col], errors="coerce").dropna()
        if not eq.empty:
            start_equity = STARTING_CAPITAL  # anchor to known $500 starting capital
            current_equity = float(eq.iloc[-1])
            total_return_pct = (current_equity / start_equity - 1.0) * 100.0
            dd = drawdown_pct(eq)
            current_dd = float(dd.iloc[-1])
            max_dd = float(dd.min())

    if "cash" in equity_df.columns:
        s = pd.to_numeric(equity_df["cash"], errors="coerce").dropna()
        cash = float(s.iloc[-1]) if not s.empty else None
    if "open_position_value" in equity_df.columns:
        s = pd.to_numeric(equity_df["open_position_value"], errors="coerce").dropna()
        open_value = float(s.iloc[-1]) if not s.empty else None

if trades_df is not None and not trades_df.empty:
    trade_count = len(trades_df)
    if "timestamp" in trades_df.columns:
        trades_df["timestamp"] = to_ts(trades_df["timestamp"])
        today = datetime.now(DISPLAY_TZ).date()
        trades_today = int((trades_df["timestamp"].dt.tz_convert(DISPLAY_TZ).dt.date == today).sum())

    if "realized_pnl" in trades_df.columns:
        rp = pd.to_numeric(trades_df["realized_pnl"], errors="coerce").fillna(0)
        realized_pnl = float(rp.sum())
        closed = rp[rp != 0]
        if not closed.empty:
            win_rate = float((closed > 0).mean() * 100.0)

positions_df = build_positions_from_trades(trades_df if trades_df is not None else pd.DataFrame())

if positions_snap_df is not None and not positions_snap_df.empty:
    ps = positions_snap_df.copy()
    # Keep only latest snapshot timestamp.
    if "timestamp" in ps.columns:
        ps["timestamp"] = to_ts(ps["timestamp"])
        ps = ps.dropna(subset=["timestamp"]).sort_values("timestamp")
        if not ps.empty:
            latest_ts = ps["timestamp"].iloc[-1]
            ps = ps[ps["timestamp"] == latest_ts].copy()
    required_cols = {
        "symbol",
        "quantity",
        "avg_entry_price",
        "last_price",
        "market_value",
        "unrealized_pnl",
        "allocation_pct",
    }
    if required_cols.issubset(ps.columns):
        positions_df = pd.DataFrame(
            {
                "Symbol": ps["symbol"].astype(str),
                "Qty": pd.to_numeric(ps["quantity"], errors="coerce").fillna(0.0),
                "Entry Price": pd.to_numeric(ps["avg_entry_price"], errors="coerce").fillna(0.0),
                "Last Price": pd.to_numeric(ps["last_price"], errors="coerce").fillna(0.0),
                "Market Value": pd.to_numeric(ps["market_value"], errors="coerce").fillna(0.0),
                "Unrealized PnL": pd.to_numeric(ps["unrealized_pnl"], errors="coerce").fillna(0.0),
                "Allocation %": pd.to_numeric(ps["allocation_pct"], errors="coerce").fillna(0.0) * 100.0,
            }
        )
        positions_df = positions_df[positions_df["Qty"] > 0].sort_values("Market Value", ascending=False)
        unrealized_pnl = float(positions_df["Unrealized PnL"].sum()) if not positions_df.empty else 0.0
        open_value = float(positions_df["Market Value"].sum()) if not positions_df.empty else 0.0

if not positions_df.empty and (positions_snap_df is None or positions_snap_df.empty):
    # last trade price proxy
    last_price = {}
    if trades_df is not None and {"symbol", "price"}.issubset(trades_df.columns):
        tmp = trades_df.copy()
        tmp["price"] = pd.to_numeric(tmp["price"], errors="coerce")
        if "timestamp" in tmp.columns:
            tmp = tmp.sort_values("timestamp")
        for _, r in tmp.dropna(subset=["price"]).iterrows():
            last_price[str(r["symbol"])] = float(r["price"])

    rows = []
    total_mv = 0.0
    total_upnl = 0.0
    for _, r in positions_df.iterrows():
        sym = str(r["Symbol"])
        qty = float(r["Qty"])
        ep = float(r["Entry Price"])
        lp = float(last_price.get(sym, ep))
        mv = qty * lp
        up = qty * (lp - ep)
        total_mv += mv
        total_upnl += up
        alloc = (mv / current_equity * 100.0) if current_equity else 0.0
        rows.append({
            "Symbol": sym,
            "Qty": qty,
            "Entry Price": ep,
            "Last Price": lp,
            "Market Value": mv,
            "Unrealized PnL": up,
            "Allocation %": alloc,
        })
    positions_df = pd.DataFrame(rows).sort_values("Market Value", ascending=False)
    unrealized_pnl = total_upnl
    if open_value is None:
        open_value = total_mv

if current_equity is None:
    current_equity = 0.0
if open_value is None:
    open_value = 0.0
exposure_pct = (open_value / current_equity * 100.0) if current_equity > 0 else 0.0

# Infer forward-testing start from earliest live timestamp available.
forward_start_candidates = []
if equity_df is not None and not equity_df.empty and "timestamp" in equity_df.columns:
    s = equity_df["timestamp"].dropna()
    if not s.empty:
        forward_start_candidates.append(s.min())
if trades_df is not None and not trades_df.empty and "timestamp" in trades_df.columns:
    s = trades_df["timestamp"].dropna()
    if not s.empty:
        forward_start_candidates.append(s.min())
if forward_start_candidates:
    forward_start_utc = min(forward_start_candidates)

status_fresh = bool(last_updated is not None and (datetime.now(timezone.utc) - last_updated.to_pydatetime()).total_seconds() < 300)
risk_ok = (current_dd is None or current_dd > -RISK_DISPLAY["Total DD Stop %"]) and exposure_pct <= RISK_DISPLAY["Max Exposure %"]

# Runner status details (last run + next run) for monitoring header.
runner_last_run_txt = "N/A"
runner_next_run_txt = "N/A"
run_log_path = LOG_DIR / "live_runner_runs.csv"
if run_log_path.exists():
    try:
        rr = pd.read_csv(run_log_path)
        if rr is not None and not rr.empty:
            if "timestamp" in rr.columns:
                rr["timestamp"] = pd.to_datetime(rr["timestamp"], errors="coerce", utc=True)
            rr = rr.dropna(subset=["timestamp"]).sort_values("timestamp")
            if not rr.empty:
                runner_last_run_txt = rr["timestamp"].iloc[-1].tz_convert(DISPLAY_TZ).strftime(
                    f"%Y-%m-%d %H:%M:%S {DISPLAY_TZ_LABEL}"
                )
                if "next_run_utc" in rr.columns:
                    nr = pd.to_datetime(rr["next_run_utc"], errors="coerce", utc=True).dropna()
                    if not nr.empty:
                        runner_next_run_txt = nr.iloc[-1].tz_convert(DISPLAY_TZ).strftime(
                            f"%Y-%m-%d %H:%M:%S {DISPLAY_TZ_LABEL}"
                        )
    except Exception:
        pass

# ----------------------------
# Header
# ----------------------------
last_txt = (
    last_updated.tz_convert(DISPLAY_TZ).strftime(f"%Y-%m-%d %H:%M:%S {DISPLAY_TZ_LABEL}")
    if last_updated is not None
    else "N/A"
)
st.markdown(
    f"""
<div class="hdr">
  <div style="display:flex; justify-content:space-between; align-items:center; gap:8px; flex-wrap:wrap;">
    <div>
      <div class="hdr-title">Crypto Quant Command Center</div>
      <div class="hdr-sub">
        Strategy: {STRATEGY_INFO['name']} &nbsp;|&nbsp;
        Live Since: {LIVE_START_DATE} &nbsp;|&nbsp;
        Starting Capital: ${STARTING_CAPITAL:,.0f} &nbsp;|&nbsp;
        Mode: Binance Spot Testnet
      </div>
    </div>
    <div>
      <span class="badge">TESTNET</span>
      <span class="badge">PAPER MODE</span>
      <span class="badge badge-green">SANDBOX ON</span>
      <span class="badge {'badge-green' if risk_ok else 'badge-red'}">{'RISK OK' if risk_ok else 'RISK STOP'}</span>
      <span class="badge">TAX RESERVE ACTIVE</span>
    </div>
  </div>
  <div class="hdr-sub">Last Updated: {last_txt} &nbsp;|&nbsp; Bot: {'RUNNING' if status_fresh else 'STOPPED'} &nbsp;|&nbsp; Last Run: {runner_last_run_txt} &nbsp;|&nbsp; Next Run: {runner_next_run_txt}</div>
</div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Overview
# ----------------------------
if section_overview:
    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)

    cards = [
        metric_card("Current Equity", fmt_money(current_equity)),
        metric_card("Total Return %", fmt_pct(total_return_pct), "pos" if (total_return_pct or 0) > 0 else "neg" if (total_return_pct or 0) < 0 else ""),
        metric_card("Cash Available", fmt_money(cash)),
        metric_card("Open Position Value", fmt_money(open_value)),
        metric_card("Current Drawdown", fmt_pct(current_dd), "neg" if (current_dd or 0) < 0 else ""),
        metric_card("Max Drawdown", fmt_pct(max_dd), "neg" if (max_dd or 0) < 0 else ""),
        metric_card("Realized PnL", fmt_money(realized_pnl), "pos" if (realized_pnl or 0) > 0 else "neg" if (realized_pnl or 0) < 0 else ""),
        metric_card("Unrealized PnL", fmt_money(unrealized_pnl), "pos" if (unrealized_pnl or 0) > 0 else "neg" if (unrealized_pnl or 0) < 0 else ""),
        metric_card("Trades Today", str(trades_today)),
        metric_card("Win Rate", fmt_pct(win_rate)),
        metric_card("Exposure", fmt_pct(exposure_pct), "neg" if exposure_pct > RISK_DISPLAY["Max Exposure %"] else ""),
    ]
    st.markdown(f'<div class="metric-grid">{"".join(cards[:4])}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-grid">{"".join(cards[4:8])}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-grid">{"".join(cards[8:12])}</div>', unsafe_allow_html=True)

    # V7 strategy reference card
    live_return_str = f"{total_return_pct:+.2f}%" if total_return_pct is not None else "N/A"
    live_return_color = "val-pos" if (total_return_pct or 0) > 0 else ("val-neg" if (total_return_pct or 0) < 0 else "")
    st.markdown(
        f"""
<div class="box" style="margin-bottom:10px;">
  <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
    <div>
      <div style="color:#9a9a9a; font-size:10px; text-transform:uppercase; letter-spacing:.1em;">Active Strategy</div>
      <div style="color:#f2f2f2; font-size:13px; margin-top:4px;">{STRATEGY_INFO['name']}</div>
      <div style="color:#9a9a9a; font-size:11px; margin-top:2px;">
        Sectors: {STRATEGY_INFO['sectors']} &nbsp;·&nbsp;
        Signal: {STRATEGY_INFO['signal']} &nbsp;·&nbsp;
        Rebalance: {STRATEGY_INFO['rebalance']}
      </div>
    </div>
    <div style="display:flex; gap:24px; text-align:center;">
      <div>
        <div style="color:#9a9a9a; font-size:10px; text-transform:uppercase; letter-spacing:.08em;">Backtest CAGR</div>
        <div style="color:#00c076; font-size:16px; margin-top:2px;">{STRATEGY_INFO['backtest_cagr']}</div>
        <div style="color:#9a9a9a; font-size:9px;">{STRATEGY_INFO['backtest_period']}</div>
      </div>
      <div>
        <div style="color:#9a9a9a; font-size:10px; text-transform:uppercase; letter-spacing:.08em;">Backtest Sharpe</div>
        <div style="color:#f2f2f2; font-size:16px; margin-top:2px;">{STRATEGY_INFO['backtest_sharpe']}</div>
        <div style="color:#9a9a9a; font-size:9px;">Max DD {STRATEGY_INFO['backtest_maxdd']}</div>
      </div>
      <div>
        <div style="color:#9a9a9a; font-size:10px; text-transform:uppercase; letter-spacing:.08em;">Live Return</div>
        <div class="{live_return_color}" style="font-size:16px; margin-top:2px;">{live_return_str}</div>
        <div style="color:#9a9a9a; font-size:9px;">Since {LIVE_START_DATE} · Start ${STARTING_CAPITAL:,.0f}</div>
      </div>
    </div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Equity & Drawdown</div>', unsafe_allow_html=True)
    if equity_df is None or equity_df.empty:
        info("No equity data available.")
    else:
        eq_col = "total_equity" if "total_equity" in equity_df.columns else next((c for c in equity_df.columns if "equity" in c.lower()), None)
        ts_col = "timestamp" if "timestamp" in equity_df.columns else next((c for c in equity_df.columns if "time" in c.lower() or "date" in c.lower()), None)
        if eq_col is None or ts_col is None:
            warn("Could not detect required columns for equity chart.")
        else:
            chart_df = prepare_equity_history(equity_df, eq_col=eq_col, ts_col=ts_col)

            if range_choice != "ALL" and not chart_df.empty:
                max_ts = chart_df[ts_col].max()
                offset = {"1D": pd.Timedelta(days=1), "1W": pd.Timedelta(days=7), "1M": pd.Timedelta(days=30)}[range_choice]
                chart_df = chart_df[chart_df[ts_col] >= max_ts - offset]

            if chart_df.empty:
                info("No data in selected range.")
            else:
                chart_df["dd"] = drawdown_pct(chart_df[eq_col])
                chart_df["peak"] = chart_df[eq_col].cummax()

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=chart_df[ts_col],
                        y=chart_df[eq_col],
                        mode="lines",
                        name="Equity",
                        line=dict(color="#f2f2f2", width=2),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=chart_df[ts_col],
                        y=chart_df["peak"],
                        mode="lines",
                        name="Peak",
                        line=dict(color="#8a8a8a", width=1, dash="dot"),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=chart_df[ts_col],
                        y=chart_df["dd"],
                        mode="lines",
                        name="Drawdown %",
                        yaxis="y2",
                        line=dict(color="#e03d3d", width=1.5),
                    )
                )
                # Auto ranges; user can control interactively using Plotly in-chart tools.
                y1_cfg = dict(title="Equity", tickprefix="$", gridcolor="#1f1f1f", fixedrange=False)
                y2_cfg = dict(
                    title="Drawdown %",
                    overlaying="y",
                    side="right",
                    ticksuffix="%",
                    showgrid=False,
                    fixedrange=False,
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="#111111",
                    font=dict(color="#d0d0d0"),
                    margin=dict(l=40, r=40, t=20, b=30),
                    dragmode="zoom",
                    xaxis=dict(
                        gridcolor="#1f1f1f",
                        rangeslider=dict(visible=True, thickness=0.08),
                        rangeselector=dict(
                            buttons=[
                                dict(count=1, label="1D", step="day", stepmode="backward"),
                                dict(count=7, label="1W", step="day", stepmode="backward"),
                                dict(count=1, label="1M", step="month", stepmode="backward"),
                                dict(step="all", label="ALL"),
                            ]
                        ),
                        fixedrange=False,
                    ),
                    yaxis=y1_cfg,
                    yaxis2=y2_cfg,
                    legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
                    height=360,
                )
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={
                        "scrollZoom": True,
                        "displaylogo": False,
                        "modeBarButtonsToAdd": ["pan2d", "zoom2d", "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"],
                    },
                )

# ----------------------------
# Positions
# ----------------------------
if section_positions:
    st.markdown('<div class="section-title">Positions</div>', unsafe_allow_html=True)
    if positions_df is None or positions_df.empty:
        info("No open positions.")
    else:
        out = positions_df.copy()
        out["Qty"] = out["Qty"].map(lambda x: f"{x:,.6f}")
        for col in ["Entry Price", "Last Price"]:
            out[col] = out[col].map(lambda x: f"${x:,.6f}")
        out["Market Value"] = out["Market Value"].map(lambda x: f"${x:,.2f}")
        out["Unrealized PnL"] = out["Unrealized PnL"].map(lambda x: f"${x:,.2f}")
        out["Allocation %"] = out["Allocation %"].map(lambda x: f"{x:,.2f}%")
        st.dataframe(out, use_container_width=True, hide_index=True)

# ----------------------------
# Long-term core panel
# ----------------------------
if section_long_term_core:
    st.markdown('<div class="section-title">Rotation Assets (XRP · HBAR · ONDO)</div>', unsafe_allow_html=True)

    if positions_df is None or positions_df.empty:
        info("No open positions in rotation assets (XRP, HBAR, ONDO).")
    else:
        core_rows = []
        p = positions_df.copy()
        p["base_symbol"] = p["Symbol"].astype(str).str.upper().map(
            lambda s: s.split("/")[0].split("(")[0].strip()
        )
        core_positions = p[p["base_symbol"].isin(LONG_TERM_CORE_BASES)].copy()

        # Optional net invested estimate from trade history (buys - sells notional).
        invested_map: Dict[str, float] = {}
        if trades_df is not None and not trades_df.empty and {"symbol", "side"}.issubset(trades_df.columns):
            t = trades_df.copy()
            if "order_value_usd" in t.columns:
                t["order_value_usd"] = pd.to_numeric(t["order_value_usd"], errors="coerce").fillna(0.0)
            else:
                q = pd.to_numeric(t.get("quantity", 0.0), errors="coerce").fillna(0.0)
                px = pd.to_numeric(t.get("price", 0.0), errors="coerce").fillna(0.0)
                t["order_value_usd"] = q * px
            t["base_symbol"] = t["symbol"].astype(str).str.upper().map(
                lambda s: s.split("/")[0].split("(")[0].strip()
            )
            t = t[t["base_symbol"].isin(LONG_TERM_CORE_BASES)]
            if not t.empty:
                t["signed_notional"] = t["order_value_usd"] * t["side"].astype(str).str.lower().map(
                    lambda side: 1.0 if side == "buy" else -1.0
                )
                invested = t.groupby("base_symbol", dropna=False)["signed_notional"].sum()
                invested_map = {str(k): float(v) for k, v in invested.items()}

        if core_positions.empty:
            info("No current holdings in rotation asset set (XRP, HBAR, ONDO).")
        else:
            for _, r in core_positions.iterrows():
                base = str(r["base_symbol"])
                market_value = float(r.get("Market Value", 0.0) or 0.0)
                net_invested = float(invested_map.get(base, 0.0))
                core_rows.append(
                    {
                        "Base": base,
                        "Symbol": str(r["Symbol"]),
                        "Qty": float(r.get("Qty", 0.0) or 0.0),
                        "Entry Price": float(r.get("Entry Price", 0.0) or 0.0),
                        "Last Price": float(r.get("Last Price", 0.0) or 0.0),
                        "Market Value": market_value,
                        "Allocation %": float(r.get("Allocation %", 0.0) or 0.0),
                        "Unrealized PnL": float(r.get("Unrealized PnL", 0.0) or 0.0),
                        "Net Invested (Buys-Sells)": net_invested,
                    }
                )

            core_df = pd.DataFrame(core_rows).sort_values("Market Value", ascending=False)
            core_total_value = float(core_df["Market Value"].sum()) if not core_df.empty else 0.0
            core_total_alloc = float(core_df["Allocation %"].sum()) if not core_df.empty else 0.0
            core_total_upnl = float(core_df["Unrealized PnL"].sum()) if not core_df.empty else 0.0

            st.markdown(
                f"""
<div class="box">
  <div style="font-size:12px; color:#f2f2f2;">Core Value: ${core_total_value:,.2f}</div>
  <div style="font-size:11px; color:#9a9a9a;">Core Allocation: {core_total_alloc:,.2f}% | Core Unrealized PnL: ${core_total_upnl:,.2f}</div>
</div>
                """,
                unsafe_allow_html=True,
            )

            out = core_df.copy()
            out["Qty"] = out["Qty"].map(lambda x: f"{x:,.6f}")
            out["Entry Price"] = out["Entry Price"].map(lambda x: f"${x:,.6f}")
            out["Last Price"] = out["Last Price"].map(lambda x: f"${x:,.6f}")
            out["Market Value"] = out["Market Value"].map(lambda x: f"${x:,.2f}")
            out["Allocation %"] = out["Allocation %"].map(lambda x: f"{x:,.2f}%")
            out["Unrealized PnL"] = out["Unrealized PnL"].map(lambda x: f"${x:,.2f}")
            out["Net Invested (Buys-Sells)"] = out["Net Invested (Buys-Sells)"].map(lambda x: f"${x:,.2f}")
            st.dataframe(out, use_container_width=True, hide_index=True)

# ----------------------------
# Trades
# ----------------------------
if section_trades:
    st.markdown('<div class="section-title">Trades</div>', unsafe_allow_html=True)
    if trades_df is None or trades_df.empty:
        info("No trades available.")
    else:
        t = trades_df.copy()
        if "timestamp" in t.columns:
            t["timestamp"] = to_display_ts(t["timestamp"])
            t = t.sort_values("timestamp", ascending=False)
        if "symbol" in t.columns:
            symbols = ["ALL"] + sorted(t["symbol"].dropna().astype(str).unique().tolist())
            pick = st.selectbox("Filter symbol", symbols, index=0)
            if pick != "ALL":
                t = t[t["symbol"] == pick]

        cols = [c for c in ["timestamp", "symbol", "side", "quantity", "price", "order_value_usd", "realized_pnl"] if c in t.columns]
        out = t[cols].head(100).copy()
        st.dataframe(out, use_container_width=True, hide_index=True)

# ----------------------------
# Signals
# ----------------------------
if section_signals:
    st.markdown('<div class="section-title">Signals</div>', unsafe_allow_html=True)
    if signals_df is None or signals_df.empty:
        info("No signal trace available.")
    else:
        s = signals_df.copy()
        ts_col = next(
            (
                c
                for c in ["timestamp", "signal_date", "signal_timestamp", "date", "created_at"]
                if c in s.columns
            ),
            None,
        )
        if ts_col is None:
            # No timestamp-like column detected; keep fallback behavior.
            st.dataframe(s.tail(50), use_container_width=True, hide_index=True)
        else:
            s["_signal_ts_utc"] = pd.to_datetime(s[ts_col], errors="coerce", utc=True)
            s = s.dropna(subset=["_signal_ts_utc"])

            recent_cutoff = datetime.now(timezone.utc) - pd.Timedelta(days=FORWARD_SIGNAL_LOOKBACK_DAYS)
            if forward_start_utc is not None:
                cutoff_utc = max(recent_cutoff, forward_start_utc.to_pydatetime())
            else:
                cutoff_utc = recent_cutoff

            filtered = s[s["_signal_ts_utc"] >= cutoff_utc].copy()
            dropped_count = len(s) - len(filtered)
            cutoff_ts = pd.Timestamp(cutoff_utc)
            if cutoff_ts.tzinfo is None:
                cutoff_ts = cutoff_ts.tz_localize("UTC")
            else:
                cutoff_ts = cutoff_ts.tz_convert("UTC")
            cutoff_et_str = cutoff_ts.tz_convert(DISPLAY_TZ).strftime("%Y-%m-%d %H:%M:%S")
            if filtered.empty:
                info(
                    "No recent forward-test signals in the selected window "
                    f"(cutoff: {cutoff_et_str} {DISPLAY_TZ_LABEL})."
                )
            else:
                filtered[ts_col] = (
                    filtered["_signal_ts_utc"].dt.tz_convert(DISPLAY_TZ).dt.strftime("%Y-%m-%d %H:%M:%S")
                )
                filtered = filtered.sort_values("_signal_ts_utc", ascending=False)
                info(
                    f"Showing recent forward-test signals only (last {FORWARD_SIGNAL_LOOKBACK_DAYS} days "
                    f"and since live start). Filtered out {dropped_count} older rows."
                )
                st.dataframe(
                    filtered.drop(columns=["_signal_ts_utc"]).head(100),
                    use_container_width=True,
                    hide_index=True,
                )

# ----------------------------
# Risk panel
# ----------------------------
if section_risk:
    st.markdown('<div class="section-title">Risk</div>', unsafe_allow_html=True)
    risk_rows = pd.DataFrame(
        {
            "Parameter": list(RISK_DISPLAY.keys()) + ["Current Exposure %", "Current Drawdown %", "Open Positions"],
            "Value": [
                f"${RISK_DISPLAY['Max Order Size (USD)']:.2f}",
                f"${RISK_DISPLAY['Max Position Size (USD)']:.2f}",
                f"{RISK_DISPLAY['Max Exposure %']:.2f}%",
                f"{RISK_DISPLAY['Daily DD Stop %']:.2f}%",
                f"{RISK_DISPLAY['Total DD Stop %']:.2f}%",
                f"{exposure_pct:.2f}%",
                f"{(current_dd if current_dd is not None else 0.0):.2f}%",
                str(len(positions_df) if positions_df is not None else 0),
            ],
        }
    )
    st.dataframe(risk_rows, use_container_width=True, hide_index=True)

    bar_color = "#00c076" if exposure_pct < 50 else ("#d4a017" if exposure_pct <= 80 else "#e03d3d")
    st.markdown(
        f"""
<div class="box">
  <div style="color:#9a9a9a; font-size:10px; text-transform:uppercase; letter-spacing:.1em; margin-bottom:6px;">Exposure Gauge</div>
  <div style="height:8px; background:#1f1f1f; border:1px solid #2a2a2a; border-radius:999px; overflow:hidden;">
    <div style="height:100%; width:{min(max(exposure_pct, 0), 100):.2f}%; background:{bar_color};"></div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

# ----------------------------
# Optimization data panel
# ----------------------------
if section_optimization:
    st.markdown('<div class="section-title">Optimization Data</div>', unsafe_allow_html=True)

    # 1) Data health / coverage
    health_rows = []
    for name, df in [
        ("trades", trades_df),
        ("equity", equity_df),
        ("signals", signals_df),
    ]:
        row_count = int(len(df)) if df is not None else 0
        start_ts = "N/A"
        end_ts = "N/A"
        if df is not None and not df.empty and "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dropna()
            if not ts.empty:
                start_ts = ts.min().tz_convert(DISPLAY_TZ).strftime("%Y-%m-%d %H:%M:%S")
                end_ts = ts.max().tz_convert(DISPLAY_TZ).strftime("%Y-%m-%d %H:%M:%S")
        elif df is not None and not df.empty and "signal_date" in df.columns:
            ts = pd.to_datetime(df["signal_date"], errors="coerce", utc=True).dropna()
            if not ts.empty:
                start_ts = ts.min().tz_convert(DISPLAY_TZ).strftime("%Y-%m-%d %H:%M:%S")
                end_ts = ts.max().tz_convert(DISPLAY_TZ).strftime("%Y-%m-%d %H:%M:%S")
        health_rows.append(
            {
                "dataset": name,
                "rows": row_count,
                f"start ({DISPLAY_TZ_LABEL})": start_ts,
                f"end ({DISPLAY_TZ_LABEL})": end_ts,
            }
        )
    st.dataframe(pd.DataFrame(health_rows), use_container_width=True, hide_index=True)

    # 2) Trade quality by symbol
    if trades_df is None or trades_df.empty:
        info("No trade data available for optimization metrics.")
    else:
        t = trades_df.copy()
        for c in ["realized_pnl", "fees_estimate", "order_value_usd"]:
            if c in t.columns:
                t[c] = pd.to_numeric(t[c], errors="coerce").fillna(0.0)
        if "timestamp" in t.columns:
            t["timestamp"] = pd.to_datetime(t["timestamp"], errors="coerce", utc=True)
            t["trade_date_et"] = t["timestamp"].dt.tz_convert(DISPLAY_TZ).dt.date
            t["hour_et"] = t["timestamp"].dt.tz_convert(DISPLAY_TZ).dt.hour

        if {"symbol", "realized_pnl"}.issubset(t.columns):
            grp = t.groupby("symbol", dropna=False)
            by_symbol = pd.DataFrame(
                {
                    "trades": grp.size(),
                    "realized_pnl_sum": grp["realized_pnl"].sum(),
                    "realized_pnl_avg": grp["realized_pnl"].mean(),
                    "wins": grp["realized_pnl"].apply(lambda s: int((s > 0).sum())),
                    "losses": grp["realized_pnl"].apply(lambda s: int((s < 0).sum())),
                    "win_rate_%": grp["realized_pnl"].apply(
                        lambda s: float((s > 0).mean() * 100.0) if len(s) > 0 else 0.0
                    ),
                }
            ).reset_index()
            if "fees_estimate" in t.columns:
                by_symbol = by_symbol.merge(
                    grp["fees_estimate"].sum().reset_index(name="fees_sum"), on="symbol", how="left"
                )
            st.markdown('<div class="section-title">Trade Quality By Symbol</div>', unsafe_allow_html=True)
            st.dataframe(by_symbol.sort_values("realized_pnl_sum", ascending=False), use_container_width=True, hide_index=True)

        # 3) Time-of-day / day-level performance
        if {"hour_et", "realized_pnl"}.issubset(t.columns):
            by_hour = (
                t.groupby("hour_et", dropna=False)["realized_pnl"]
                .agg(["count", "sum", "mean"])
                .reset_index()
                .rename(columns={"count": "trades", "sum": "realized_pnl_sum", "mean": "realized_pnl_avg"})
            )
            st.markdown('<div class="section-title">Performance By Hour (ET)</div>', unsafe_allow_html=True)
            st.dataframe(by_hour.sort_values("hour_et"), use_container_width=True, hide_index=True)

        if {"trade_date_et", "realized_pnl"}.issubset(t.columns):
            by_day = (
                t.groupby("trade_date_et", dropna=False)["realized_pnl"]
                .agg(["count", "sum", "mean"])
                .reset_index()
                .rename(columns={"count": "trades", "sum": "realized_pnl_sum", "mean": "realized_pnl_avg"})
            )
            st.markdown('<div class="section-title">Performance By Day (ET)</div>', unsafe_allow_html=True)
            st.dataframe(by_day.sort_values("trade_date_et", ascending=False), use_container_width=True, hide_index=True)

    # 4) Equity optimization stats
    if equity_df is None or equity_df.empty:
        info("No equity data available for optimization stats.")
    else:
        eq_col = "total_equity" if "total_equity" in equity_df.columns else next(
            (c for c in equity_df.columns if "equity" in c.lower()), None
        )
        ts_col = "timestamp" if "timestamp" in equity_df.columns else next(
            (c for c in equity_df.columns if "time" in c.lower() or "date" in c.lower()), None
        )
        if eq_col is not None and ts_col is not None:
            e = equity_df.copy()
            e[ts_col] = pd.to_datetime(e[ts_col], errors="coerce", utc=True)
            e[eq_col] = pd.to_numeric(e[eq_col], errors="coerce")
            e = e.dropna(subset=[ts_col, eq_col]).sort_values(ts_col)
            if not e.empty:
                e["ret"] = e[eq_col].pct_change().fillna(0.0)
                e["dd_pct"] = drawdown_pct(e[eq_col])
                stats = pd.DataFrame(
                    [
                        {"metric": "equity_points", "value": int(len(e))},
                        {"metric": "start_equity", "value": float(e[eq_col].iloc[0])},
                        {"metric": "end_equity", "value": float(e[eq_col].iloc[-1])},
                        {"metric": "total_return_%", "value": float((e[eq_col].iloc[-1] / e[eq_col].iloc[0] - 1.0) * 100.0) if e[eq_col].iloc[0] != 0 else 0.0},
                        {"metric": "max_drawdown_%", "value": float(e["dd_pct"].min())},
                        {"metric": "avg_period_return_%", "value": float(e["ret"].mean() * 100.0)},
                        {"metric": "return_std_%", "value": float(e["ret"].std(ddof=0) * 100.0)},
                    ]
                )
                st.markdown('<div class="section-title">Equity Statistics</div>', unsafe_allow_html=True)
                st.dataframe(stats, use_container_width=True, hide_index=True)

    # 5) Runner stop reasons (critical for optimization/debug)
    run_log_path = LOG_DIR / "live_runner_runs.csv"
    if run_log_path.exists():
        try:
            rr = pd.read_csv(run_log_path)
            if not rr.empty:
                if "timestamp" in rr.columns:
                    rr["timestamp"] = pd.to_datetime(rr["timestamp"], errors="coerce", utc=True)
                    rr["timestamp_et"] = rr["timestamp"].dt.tz_convert(DISPLAY_TZ).dt.strftime("%Y-%m-%d %H:%M:%S")
                if "stop_reason" in rr.columns:
                    reason_counts = (
                        rr["stop_reason"]
                        .fillna("")
                        .replace("", "NO_STOP_REASON")
                        .value_counts(dropna=False)
                        .reset_index()
                    )
                    reason_counts.columns = ["stop_reason", "count"]
                    st.markdown('<div class="section-title">Runner Stop Reason Counts</div>', unsafe_allow_html=True)
                    st.dataframe(reason_counts, use_container_width=True, hide_index=True)
                st.markdown('<div class="section-title">Recent Runner Runs</div>', unsafe_allow_html=True)
                show_cols = [c for c in ["run_number", "timestamp_et", "equity", "drawdown", "trades_executed", "fatal_stop", "stop_reason"] if c in rr.columns]
                if show_cols:
                    st.dataframe(rr.sort_values("run_number", ascending=False)[show_cols].head(50), use_container_width=True, hide_index=True)
        except Exception as exc:
            info(f"Could not read runner run log for optimization panel: {exc}")
