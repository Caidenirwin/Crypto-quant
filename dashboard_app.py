"""
Crypto Quant Command Center (Streamlit).  Run: streamlit run dashboard.py

Mostly read-only: it reads the bot's logs and pulls live prices / news from the
same public sources the bot uses (Binance public market data, alternative.me,
crypto news RSS). The ONE action is the "Run bot now" button, which launches
run_bot_now.ps1 (one Testnet paper-trading cycle) in the background after a
confirmation; its output goes to logs/manual_runs/.

Tabs: Overview | Live Prices | V7 Portfolio | ML Sleeve | News | Safety
"""

from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="Crypto Quant", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "logs"
ML_LOGS = LOGS / "ml"
MIRROR = "https://data-api.binance.vision/api/v3"
TZ = ZoneInfo("America/New_York")
V7_START = 500.0
ML_START = 500.0

# Watchlist: everything the bot trades or watches, plus BTC/ETH as market context.
WATCHLIST = [
    "BTC", "ETH", "XRP", "HBAR", "ONDO", "SOL", "AVAX", "LINK", "NEAR", "SUI", "AAVE",
    "UNI", "ARB", "OP", "PENDLE", "DOGE", "SHIB", "PEPE", "BONK", "WIF", "FLOKI",
]
MEMES = {"DOGE", "SHIB", "PEPE", "BONK", "WIF", "FLOKI"}
# Binance kline intervals offered on the candle chart.
TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]
MANUAL_DIR = LOGS / "manual_runs"
MANUAL_STATE = MANUAL_DIR / "current.json"

# ---- Palette (dataviz reference instance, dark mode) -------------------------
C = {
    "page": "#0d0d0d", "surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
    "grid": "#2c2c2a", "axis": "#383835", "border": "rgba(255,255,255,0.10)",
    "s1": "#3987e5", "s2": "#d95926", "s3": "#199e70",          # categorical slots 1-3
    "up": "#3987e5", "down": "#e66767",                          # diverging pair (price polarity)
    "good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b",
}


def inject_css() -> None:
    st.markdown(
        f"""
<style>
.stApp {{ background: {C['page']}; color: {C['ink']}; }}
html, body, [class*="css"] {{ font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }}
.block-container {{ padding-top: 1.2rem; max-width: 1500px; }}
[data-testid="stHeader"] {{ background: transparent; }}
.hdr {{ display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;
        background:{C['surface']}; border:1px solid {C['border']}; border-radius:12px; padding:14px 18px; margin-bottom:12px; }}
.hdr-title {{ font-size:20px; font-weight:600; color:{C['ink']}; }}
.hdr-sub {{ font-size:12px; color:{C['muted']}; margin-top:4px; }}
.pill {{ display:inline-block; font-size:11px; padding:3px 10px; border-radius:999px; margin-left:6px;
         border:1px solid {C['border']}; color:{C['ink2']}; background:#222220; }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(170px, 1fr)); gap:10px; margin:6px 0 14px; }}
.tile {{ background:{C['surface']}; border:1px solid {C['border']}; border-radius:10px; padding:12px 14px; }}
.tile-label {{ font-size:11px; color:{C['muted']}; text-transform:uppercase; letter-spacing:.06em; }}
.tile-value {{ font-size:24px; font-weight:600; color:{C['ink']}; margin-top:4px; }}
.tile-sub {{ font-size:12px; color:{C['ink2']}; margin-top:2px; }}
.up {{ color:{C['good']}; }} .down {{ color:{C['critical']}; }}
.ticker {{ display:flex; gap:18px; overflow-x:auto; white-space:nowrap; padding:8px 12px; margin-bottom:12px;
           background:{C['surface']}; border:1px solid {C['border']}; border-radius:10px; font-size:13px; }}
.ticker b {{ color:{C['ink']}; }} .ticker span.px {{ color:{C['ink2']}; font-variant-numeric: tabular-nums; }}
.news {{ background:{C['surface']}; border:1px solid {C['border']}; border-radius:10px; padding:10px 14px; margin:6px 0; }}
.news a {{ color:{C['ink']}; text-decoration:none; font-weight:500; }}
.news .meta {{ font-size:11px; color:{C['muted']}; margin-top:3px; }}
.tag {{ display:inline-block; font-size:10px; padding:1px 7px; border-radius:6px; margin-right:4px;
        border:1px solid {C['border']}; color:{C['ink2']}; }}
.status {{ display:inline-block; font-size:12px; padding:3px 10px; border-radius:6px; font-weight:600; }}
.st-good {{ color:{C['good']}; border:1px solid {C['good']}; }}
.st-warn {{ color:{C['warning']}; border:1px solid {C['warning']}; }}
.st-crit {{ color:{C['critical']}; border:1px solid {C['critical']}; }}
.note {{ font-size:12px; color:{C['muted']}; }}
</style>""",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Formatting helpers                                                          #
# --------------------------------------------------------------------------- #
def money(v: Optional[float], dp: int = 2) -> str:
    if v is None or pd.isna(v):
        return "—"
    return f"-${abs(v):,.{dp}f}" if v < 0 else f"${v:,.{dp}f}"


def price_fmt(v: Optional[float]) -> str:
    if v is None or pd.isna(v):
        return "—"
    if v >= 1000:
        return f"${v:,.2f}"
    if v >= 1:
        return f"${v:,.4f}"
    return f"${v:.8f}".rstrip("0")


def delta_html(pct: Optional[float], dp: int = 2) -> str:
    """Gain/loss text: color + arrow, never color alone."""
    if pct is None or pd.isna(pct):
        return '<span class="note">—</span>'
    cls, arrow = ("up", "▲") if pct >= 0 else ("down", "▼")
    return f'<span class="{cls}">{arrow} {pct:+.{dp}f}%</span>'


def tile(label: str, value: str, sub: str = "") -> str:
    return (f'<div class="tile"><div class="tile-label">{label}</div>'
            f'<div class="tile-value">{value}</div><div class="tile-sub">{sub}</div></div>')


def status_badge(level: str, text: str) -> str:
    icon = {"good": "✔", "warn": "⚠", "crit": "✖"}[level]
    return f'<span class="status st-{level}">{icon} {text}</span>'


def et(ts) -> str:
    try:
        return pd.Timestamp(ts).tz_convert(TZ).strftime("%b %d %H:%M ET")
    except Exception:
        return "—"


def style_fig(fig: go.Figure, height: int = 320, title: str = "") -> go.Figure:
    fig.update_layout(
        height=height, title=dict(text=title, font=dict(size=13, color=C["ink2"]), x=0, xanchor="left"),
        paper_bgcolor=C["surface"], plot_bgcolor=C["surface"], font=dict(color=C["ink2"], size=12),
        margin=dict(l=50, r=20, t=40 if title else 16, b=30), hovermode="x unified",
        legend=dict(orientation="h", y=1.08, x=1, xanchor="right", bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#222220", bordercolor=C["axis"], font=dict(color=C["ink"])),
    )
    fig.update_xaxes(gridcolor=C["grid"], linecolor=C["axis"], zeroline=False, showspikes=True,
                     spikecolor=C["muted"], spikethickness=1, spikedash="solid", spikemode="across")
    fig.update_yaxes(gridcolor=C["grid"], linecolor=C["axis"], zeroline=False)
    return fig


def show_chart(fig: go.Figure, key: str) -> None:
    st.plotly_chart(fig, width="stretch", key=key, config={"displaylogo": False, "scrollZoom": True})


# --------------------------------------------------------------------------- #
# Data loaders (read-only)                                                    #
# --------------------------------------------------------------------------- #
def read_csv(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    return df


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


@st.cache_data(ttl=15, show_spinner=False)
def live_tickers(bases: tuple) -> pd.DataFrame:
    syms = json.dumps([f"{b}USDT" for b in bases], separators=(",", ":"))
    r = requests.get(f"{MIRROR}/ticker/24hr", params={"symbols": syms}, timeout=15)
    r.raise_for_status()
    rows = []
    for t in r.json():
        rows.append({
            "Coin": t["symbol"][:-4], "Price": float(t["lastPrice"]),
            "24h %": float(t["priceChangePercent"]), "24h High": float(t["highPrice"]),
            "24h Low": float(t["lowPrice"]), "24h Volume ($)": float(t["quoteVolume"]),
        })
    df = pd.DataFrame(rows)
    df["order"] = df["Coin"].map({b: i for i, b in enumerate(bases)})
    return df.sort_values("order").drop(columns="order").reset_index(drop=True)


@st.cache_data(ttl=10, show_spinner=False)
def klines(base: str, interval: str, limit: int = 200) -> pd.DataFrame:
    r = requests.get(f"{MIRROR}/klines", params={"symbol": f"{base}USDT", "interval": interval, "limit": limit},
                     timeout=15)
    r.raise_for_status()
    df = pd.DataFrame(r.json()).iloc[:, :6]
    df.columns = ["t", "Open", "High", "Low", "Close", "Volume"]
    df["t"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(TZ)
    return df.astype({c: float for c in ["Open", "High", "Low", "Close", "Volume"]})


@st.cache_data(ttl=300, show_spinner=False)
def headlines() -> List[dict]:
    from news_feed import fetch_headlines   # read-only fetch; the bot does the archiving
    return fetch_headlines()


@st.cache_data(ttl=600, show_spinner=False)
def fear_greed(days: int = 365) -> pd.DataFrame:
    try:
        r = requests.get("https://api.alternative.me/fng/", params={"limit": days, "format": "json"}, timeout=15)
        r.raise_for_status()
        rows = [{"date": pd.Timestamp(int(x["timestamp"]), unit="s"), "fng": float(x["value"]),
                 "label": x["value_classification"]} for x in r.json()["data"]]
        return pd.DataFrame(rows).sort_values("date")
    except Exception:
        from news_feed import load_fear_greed
        df = load_fear_greed().reset_index().tail(days)
        df["label"] = ""
        return df


def live_price_map(bases) -> Dict[str, float]:
    try:
        t = live_tickers(tuple(dict.fromkeys(bases)))
        return dict(zip(t["Coin"], t["Price"]))
    except Exception:
        return {}


def bot_status() -> dict:
    """Autopilot health from its own log + lock file."""
    out = {"running": False, "last_activity": None, "next_run": None, "safety_stop": False,
           "emergency_flag": (LOGS / "emergency_stop.flag").exists()}
    log = LOGS / "autopilot.log"
    if log.exists():
        out["last_activity"] = datetime.fromtimestamp(log.stat().st_mtime, tz=timezone.utc)
        try:
            tail = log.read_text(encoding="utf-8", errors="ignore").splitlines()[-400:]
        except Exception:
            tail = []
        for line in reversed(tail):
            m = re.search(r"until next run at (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) ET", line)
            if m:
                out["next_run"] = m.group(1)
                break
        out["safety_stop"] = any("SAFETY STOP" in l for l in tail[-20:])
        fresh = datetime.now(timezone.utc) - out["last_activity"] < timedelta(minutes=75)
        out["running"] = (LOGS / "autopilot.lock").exists() and fresh and not out["emergency_flag"]
    return out


# --------------------------------------------------------------------------- #
# "Run bot now" button (the dashboard's only action)                          #
# --------------------------------------------------------------------------- #
def _pid_alive(pid: int) -> bool:
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {int(pid)}", "/NH"], capture_output=True,
                             text=True, timeout=10).stdout
        return str(int(pid)) in out
    except Exception:
        return False


def manual_run_status() -> Optional[dict]:
    s = read_json(MANUAL_STATE)
    if not s:
        return None
    log = Path(s["log"])
    text = log.read_text(encoding="utf-8", errors="ignore") if log.exists() else ""
    started = pd.Timestamp(s["started"])
    # Guard against Windows reusing the PID long after our run ended.
    running = _pid_alive(s["pid"]) and pd.Timestamp.now(tz="UTC") - started < pd.Timedelta(minutes=30)
    if "Cycle finished" in text:
        outcome = ("good", "Finished OK")
    elif "SAFETY STOP" in text:
        outcome = ("crit", "SAFETY STOP - review before trading again")
    elif "Emergency stop is ON" in text:
        outcome = ("crit", "Refused: emergency stop is on")
    elif "Cycle failed" in text:
        outcome = ("crit", "Failed - see output")
    elif running:
        outcome = ("warn", "Running...")
    else:
        outcome = ("warn", "Ended without a result - see output")
    noise = (" | INFO | ", "Order placed:", "[ORDER PREVIEW]", "RemoteException", "CategoryInfo",
             "FullyQualifiedErrorId", "At line:", "+ ")
    lines = [l for l in text.splitlines() if l.strip() and not any(n in l for n in noise)]
    return {"running": running, "outcome": outcome, "started": started, "tail": lines[-14:], "log": log}


def launch_manual_run(skip_data: bool) -> None:
    MANUAL_DIR.mkdir(parents=True, exist_ok=True)
    log = MANUAL_DIR / f"run_{datetime.now(TZ):%Y%m%d-%H%M%S}.log"
    args = ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ROOT / "run_bot_now.ps1")]
    if skip_data:
        args.append("-SkipData")
    with log.open("w", encoding="utf-8") as fh:
        # Detached + no window: the run keeps going even if the dashboard page reloads.
        proc = subprocess.Popen(
            args, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP,
        )
    MANUAL_STATE.write_text(json.dumps({"pid": proc.pid, "started": datetime.now(timezone.utc).isoformat(),
                                        "log": str(log), "skip_data": skip_data}), encoding="utf-8")


def run_control_bar() -> None:
    # Inline confirmation (not st.dialog): this page has auto-refreshing fragments, and
    # a dialog is closed by the next fragment refresh. The button lives outside them.
    flag = (LOGS / "emergency_stop.flag").exists()
    c1, c2 = st.columns([1, 4], vertical_alignment="center")
    with c1:
        if st.button("▶ Run bot now", type="primary", width="stretch", disabled=flag,
                     help="Emergency stop is on" if flag else "Run one trading cycle now (Testnet, paper money)"):
            st.session_state["confirm_run"] = True
    with c2:
        run_status_panel(flag)

    if st.session_state.get("confirm_run") and not flag:
        with st.container(border=True):
            current = manual_run_status()
            if current and current["running"]:
                st.warning("A manual run is already in progress - wait for it to finish.")
                if st.button("OK"):
                    st.session_state["confirm_run"] = False
                    st.rerun()
                return
            st.markdown("**Run one trading cycle now?** V7 + memes (anti-rug) + news + the ML sleeve, once, on "
                        "Binance **Testnet** with paper money - same as `.\\run_bot_now.ps1`. If an automatic "
                        "cycle is in progress it waits for it to finish first.")
            skip = st.checkbox("Skip the market-data refresh (about 1 minute faster)", value=False)
            b1, b2, _ = st.columns([1, 1, 3])
            if b1.button("Yes, run it", type="primary", width="stretch"):
                launch_manual_run(skip)
                st.session_state["confirm_run"] = False
                st.rerun()
            if b2.button("Cancel", width="stretch"):
                st.session_state["confirm_run"] = False
                st.rerun()


@st.fragment(run_every=5)
def run_status_panel(flag: bool) -> None:
    status = manual_run_status()
    if flag:
        st.markdown(status_badge("crit", "Emergency stop is on - button disabled"), unsafe_allow_html=True)
    elif status is None:
        st.markdown('<span class="note">No manual run yet. The autopilot also trades every hour on its own.</span>',
                    unsafe_allow_html=True)
    else:
        lvl, text = status["outcome"]
        st.markdown(status_badge(lvl, f"Last manual run: {text}")
                    + f'<span class="note"> &nbsp;started {et(status["started"])}</span>',
                    unsafe_allow_html=True)
        with st.expander("Manual run output", expanded=status["running"]):
            st.code("\n".join(status["tail"]) or "(starting...)", language=None)
            st.caption(f"Full log: {status['log']}")


# --------------------------------------------------------------------------- #
# Derived portfolio views                                                     #
# --------------------------------------------------------------------------- #
def v7_positions(prices: Dict[str, float]) -> pd.DataFrame:
    snap = read_csv(LOGS / "positions_snapshot.csv")
    if snap.empty:
        return snap
    snap = snap[snap["timestamp"] == snap["timestamp"].max()]
    rows = []
    for _, r in snap.iterrows():
        base = str(r["symbol"]).split("/")[0]
        qty, entry = float(r["quantity"]), float(r["avg_entry_price"])
        if qty <= 0:
            continue
        px = prices.get(base, float(r["last_price"]))
        rows.append({"Coin": base, "Qty": qty, "Entry": entry, "Live Price": px, "Value": qty * px,
                     "P&L $": qty * (px - entry), "P&L %": (px / entry - 1) * 100 if entry else 0.0})
    return pd.DataFrame(rows)


def ml_positions(prices: Dict[str, float]) -> pd.DataFrame:
    state = read_json(ML_LOGS / "state.json")
    rows = []
    for sym, p in (state.get("positions") or {}).items():
        base = sym.split("/")[0]
        px = prices.get(base, float(p["entry"]))
        risk = float(p["risk"])
        rows.append({"Coin": base, "Qty": float(p["qty"]), "Entry": float(p["entry"]), "Live Price": px,
                     "Stop": float(p["sl"]), "Target": float(p["tp"]),
                     "Open R": (px - float(p["entry"])) / risk if risk else 0.0,
                     "Expires": et(p["expiry"]), "Model p": float(p.get("p", 0.0))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Sections                                                                    #
# --------------------------------------------------------------------------- #
def header(status: dict) -> None:
    if status["emergency_flag"]:
        badge = status_badge("crit", "EMERGENCY STOP")
    elif status["safety_stop"]:
        badge = status_badge("crit", "SAFETY STOP")
    elif status["running"]:
        badge = status_badge("good", "AUTOPILOT RUNNING")
    else:
        badge = status_badge("warn", "AUTOPILOT NOT RUNNING")
    nxt = f"Next run {status['next_run'][11:16]} ET" if status["next_run"] else "Next run —"
    last = et(status["last_activity"]) if status["last_activity"] else "—"
    st.markdown(
        f"""<div class="hdr"><div><div class="hdr-title">Crypto Quant Command Center</div>
<div class="hdr-sub">Paper trading on Binance Spot Testnet · last bot activity {last} · {nxt}</div></div>
<div>{badge}<span class="pill">TESTNET · PAPER MONEY</span></div></div>""",
        unsafe_allow_html=True,
    )


@st.fragment(run_every=15)
def ticker_strip() -> None:
    try:
        t = live_tickers(tuple(WATCHLIST))
    except Exception as exc:
        st.markdown(f'<div class="ticker note">Live prices unavailable: {exc}</div>', unsafe_allow_html=True)
        return
    items = "".join(
        f'<div><b>{r["Coin"]}</b> <span class="px">{price_fmt(r["Price"])}</span> {delta_html(r["24h %"])}</div>'
        for r in t.to_dict("records")
    )
    st.markdown(f'<div class="ticker">{items}</div>', unsafe_allow_html=True)


def overview_tab(prices: Dict[str, float]) -> None:
    v7_eq = read_csv(LOGS / "paper_equity.csv")
    ml_eq = read_csv(ML_LOGS / "ml_equity.csv")
    status = read_json(ROOT / "models" / "ml_status.json")

    v7_pos = v7_positions(prices)
    v7_cash = float(v7_eq["cash"].iloc[-1]) if not v7_eq.empty else V7_START
    v7_now = v7_cash + (v7_pos["Value"].sum() if not v7_pos.empty else 0.0) + (
        float(v7_eq["tax_reserve"].iloc[-1]) if not v7_eq.empty and "tax_reserve" in v7_eq else 0.0)
    ml_state = read_json(ML_LOGS / "state.json")
    ml_pos = ml_positions(prices)
    ml_now = float(ml_state.get("cash", ML_START)) + float(ml_state.get("tax_reserve", 0.0)) + (
        (ml_pos["Qty"] * ml_pos["Live Price"]).sum() if not ml_pos.empty else 0.0)
    total = v7_now + ml_now
    fng = fear_greed(30)
    fng_now = fng["fng"].iloc[-1] if not fng.empty else None
    fng_lbl = fng["label"].iloc[-1] if not fng.empty else ""

    st.markdown(
        '<div class="tiles">'
        + tile("Total (both accounts)", money(total), delta_html((total / (V7_START + ML_START) - 1) * 100) + " since start")
        + tile("V7 account", money(v7_now), delta_html((v7_now / V7_START - 1) * 100) + " · live-priced")
        + tile("ML account", money(ml_now), delta_html((ml_now / ML_START - 1) * 100)
               + f" · {len(ml_pos)} open trade{'s' if len(ml_pos) != 1 else ''}")
        + tile("ML new entries", "Allowed" if status.get("entries_allowed") else "Paused",
               f"champion: {status.get('champion', '—')}")
        + tile("Fear & Greed", f"{fng_now:.0f}" if fng_now is not None else "—", fng_lbl)
        + "</div>",
        unsafe_allow_html=True,
    )

    # Account equity over time - one axis, two accounts (categorical slots 1, 2).
    fig = go.Figure()
    if not v7_eq.empty:
        fig.add_trace(go.Scatter(x=v7_eq["timestamp"].dt.tz_convert(TZ), y=v7_eq["total_equity"], name="V7",
                                 mode="lines", line=dict(color=C["s1"], width=2)))
    if not ml_eq.empty:
        fig.add_trace(go.Scatter(x=ml_eq["timestamp"].dt.tz_convert(TZ), y=ml_eq["equity"], name="ML sleeve",
                                 mode="lines", line=dict(color=C["s2"], width=2)))
    fig.add_hline(y=500, line=dict(color=C["axis"], width=1), annotation_text="$500 start",
                  annotation_font_color=C["muted"], annotation_position="bottom left")
    fig.update_yaxes(tickprefix="$")
    show_chart(style_fig(fig, 320, "Account value"), "ov_equity")
    st.markdown('<p class="note">Logs are written by the bot once per hourly run; the tiles above re-price open '
                'positions with live prices.</p>', unsafe_allow_html=True)

    profit_split_section()

    with st.expander("▶ Run the bot now (one trading cycle, any time)"):
        st.markdown("Paste into **PowerShell**. Safe while the autopilot runs: it waits for any cycle in progress.")
        st.code("cd C:\\Users\\caide\\Downloads\\CRPTO-QUANT\n.\\run_bot_now.ps1", language="powershell")
        st.markdown('<p class="note">Add <code>-SkipData</code> to skip the ~1 minute data refresh. '
                    'Stop everything: <code>.\\stop_autopilot.ps1</code> · restart: '
                    '<code>.\\install_autopilot.ps1</code></p>', unsafe_allow_html=True)


def _realized(df: pd.DataFrame, pnl_col: str, tax_col: str, account: str) -> pd.DataFrame:
    if df.empty or pnl_col not in df.columns:
        return pd.DataFrame(columns=["timestamp", "account", "pnl", "tax"])
    out = pd.DataFrame({
        "timestamp": df["timestamp"], "account": account,
        "pnl": pd.to_numeric(df[pnl_col], errors="coerce").fillna(0.0),
        "tax": pd.to_numeric(df[tax_col], errors="coerce").fillna(0.0) if tax_col in df.columns else 0.0,
    })
    return out[out["pnl"] != 0]


def profit_split_section() -> None:
    from config import Config

    rate = float(Config().portfolio.tax_reserve_rate)
    st.markdown(f"#### Profit split · {rate:.0%} tax reserve / {1 - rate:.0%} reinvested")
    st.markdown(
        f'<p class="note">Every winning sale: {rate:.0%} of the profit moves into a locked tax reserve that is '
        f'never traded; the other {1 - rate:.0%} stays in the account and is reinvested to compound. Losing '
        'trades don\'t take money back out of the reserve (conservative).</p>',
        unsafe_allow_html=True,
    )
    fills = pd.concat([
        _realized(read_csv(LOGS / "live_trades.csv"), "realized_pnl", "tax_reserved", "V7"),
        _realized(read_csv(ML_LOGS / "ml_trades.csv"), "pnl", "tax_reserved", "ML"),
    ], ignore_index=True).sort_values("timestamp")
    gains = fills.loc[fills["pnl"] > 0, "pnl"].sum()
    losses = -fills.loc[fills["pnl"] < 0, "pnl"].sum()
    reserved = fills["tax"].sum()
    reinvested = gains - reserved
    st.markdown(
        '<div class="tiles">'
        + tile("Tax reserve (locked)", money(reserved), f"{rate:.0%} of winning-trade profit")
        + tile("Reinvested profit", money(reinvested), f"{1 - rate:.0%} back to work in the accounts")
        + tile("Profit from winning trades", money(gains),
               f"{int((fills['pnl'] > 0).sum())} winning sale{'s' if (fills['pnl'] > 0).sum() != 1 else ''}")
        + tile("Net realized P&L", money(gains - losses), f"after {money(losses)} of realized losses")
        + "</div>",
        unsafe_allow_html=True,
    )
    by_acct = (fills.assign(win=fills["pnl"].clip(lower=0), loss=(-fills["pnl"]).clip(lower=0))
               .groupby("account")[["win", "loss", "tax"]].sum()
               .reindex(["V7", "ML"]).fillna(0.0))
    by_acct["reinvested"] = by_acct["win"] - by_acct["tax"]
    table = by_acct.reset_index().rename(columns={
        "account": "Account", "win": "Winning profit", "loss": "Realized losses",
        "tax": "Tax reserve", "reinvested": "Reinvested"})
    money_cols = ["Winning profit", "Realized losses", "Tax reserve", "Reinvested"]
    st.dataframe(table, width="stretch", hide_index=True,
                 column_config={c: st.column_config.NumberColumn(format="$%.2f") for c in money_cols})

    wins = fills[fills["pnl"] > 0]
    if wins.empty:
        st.markdown('<p class="note">No winning trade has closed yet, so the reserve is still $0. It starts '
                    'filling on the first profitable sale.</p>', unsafe_allow_html=True)
        return
    x = wins["timestamp"].dt.tz_convert(TZ)
    f = go.Figure()
    f.add_trace(go.Scatter(x=x, y=wins["tax"].cumsum(), mode="lines", name="Tax reserve",
                           line=dict(color=C["s3"], width=2, shape="hv")))
    f.add_trace(go.Scatter(x=x, y=(wins["pnl"] - wins["tax"]).cumsum(), mode="lines", name="Reinvested",
                           line=dict(color=C["s1"], width=2, shape="hv")))
    f.update_yaxes(tickprefix="$")
    show_chart(style_fig(f, 260, "Cumulative profit split"), "profit_split")


def prices_tab() -> None:
    @st.fragment(run_every=15)
    def table() -> None:
        try:
            t = live_tickers(tuple(WATCHLIST))
        except Exception as exc:
            st.warning(f"Live prices unavailable: {exc}")
            return
        t["Type"] = t["Coin"].map(lambda b: "meme" if b in MEMES else ("market" if b in ("BTC", "ETH") else "core"))
        st.caption(f"Live from Binance public market data · refreshes every 15s · {datetime.now(TZ):%H:%M:%S} ET")
        st.dataframe(
            t, width="stretch", hide_index=True, height=min(780, 36 * (len(t) + 1)),
            column_config={
                "Price": st.column_config.NumberColumn(format="$%.8g"),
                "24h %": st.column_config.NumberColumn(format="%+.2f%%"),
                "24h High": st.column_config.NumberColumn(format="$%.8g"),
                "24h Low": st.column_config.NumberColumn(format="$%.8g"),
                "24h Volume ($)": st.column_config.NumberColumn(format="compact"),
            },
        )

    left, right = st.columns([5, 7], gap="medium")
    with left:
        table()
    with right:
        c1, c2 = st.columns([2, 5])
        coin = c1.selectbox("Coin", WATCHLIST, index=WATCHLIST.index("XRP"))
        interval = c2.segmented_control("Timeframe", TIMEFRAMES, default="1h", key="timeframe") or "1h"
        candle_chart(coin, interval)


@st.fragment(run_every=15)
def candle_chart(coin: str, interval: str) -> None:
    try:
        k = klines(coin, interval)
    except Exception as exc:
        st.warning(f"Chart unavailable: {exc}")
        return
    k["MA20"] = k["Close"].rolling(20).mean()
    k["MA50"] = k["Close"].rolling(50).mean()
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=k["t"], open=k["Open"], high=k["High"], low=k["Low"], close=k["Close"], name=coin,
        increasing=dict(line=dict(color=C["up"], width=1), fillcolor=C["up"]),
        decreasing=dict(line=dict(color=C["down"], width=1), fillcolor=C["down"]),
    ))
    fig.add_trace(go.Scatter(x=k["t"], y=k["MA20"], name="20-period avg", mode="lines",
                             line=dict(color=C["ink2"], width=1.5)))
    fig.add_trace(go.Scatter(x=k["t"], y=k["MA50"], name="50-period avg", mode="lines",
                             line=dict(color=C["s3"], width=1.5)))
    # Overlay the bot's own levels on this coin, if it holds it.
    levels = []
    ml_state = read_json(ML_LOGS / "state.json")
    pos = (ml_state.get("positions") or {}).get(f"{coin}/USDT")
    if pos:
        levels += [(pos["entry"], "ML entry", C["muted"]), (pos["sl"], "ML stop", C["muted"]),
                   (pos["tp"], "ML target", C["muted"])]
    v7 = read_csv(LOGS / "positions_snapshot.csv")
    if not v7.empty:
        row = v7[(v7["timestamp"] == v7["timestamp"].max()) & (v7["symbol"] == f"{coin}/USDT")]
        if not row.empty and float(row["quantity"].iloc[0]) > 0:
            levels.append((float(row["avg_entry_price"].iloc[0]), "V7 entry", C["s1"]))
    # Keep the y-axis on the candles: a level far outside them (e.g. an old entry price on
    # a 5-minute chart) would squash the candles flat - list those as text instead.
    lo, hi = float(k["Low"].min()), float(k["High"].max())
    pad = (hi - lo) * 0.08 or hi * 0.01
    off_chart = []
    for y, name, color in levels:
        if lo - 3 * pad <= y <= hi + 3 * pad:
            fig.add_hline(y=y, line=dict(color=color, width=1), annotation_text=name,
                          annotation_font_color=C["ink2"], annotation_position="top left")
            lo, hi = min(lo, y), max(hi, y)
        else:
            off_chart.append(f"{name} {price_fmt(y)} ({'above' if y > hi else 'below'} this chart)")
    fig.update_yaxes(range=[lo - pad, hi + pad])
    fig.update_layout(xaxis_rangeslider_visible=False)
    last = k["Close"].iloc[-1]
    chg = (last / k["Close"].iloc[0] - 1) * 100
    show_chart(style_fig(fig, 470, f"{coin}/USDT · {interval} candles · last {price_fmt(last)} "
                                    f"({chg:+.2f}% over chart) · updated {datetime.now(TZ):%H:%M:%S} ET"),
               f"candles_{coin}_{interval}")
    if off_chart:
        st.caption("Bot levels off this chart: " + " · ".join(off_chart))
    with st.expander("Table view"):
        st.dataframe(k.drop(columns=["MA20", "MA50"]).tail(50).iloc[::-1], width="stretch", hide_index=True)


def v7_tab(prices: Dict[str, float]) -> None:
    eq = read_csv(LOGS / "paper_equity.csv")
    trades = read_csv(LOGS / "live_trades.csv")
    pos = v7_positions(prices)
    st.markdown("**V7 — HighBeta + ONDO sector rotation** (+ meme sleeve, 3% cap, anti-rug guarded)")
    if eq.empty:
        st.info("No V7 equity logged yet.")
    else:
        x = eq["timestamp"].dt.tz_convert(TZ)
        y = eq["total_equity"].astype(float)
        dd = (y / y.cummax() - 1) * 100
        # Two charts, shared time axis - never one chart with two y-scales.
        f1 = go.Figure(go.Scatter(x=x, y=y, mode="lines", name="Equity", line=dict(color=C["s1"], width=2)))
        f1.update_yaxes(tickprefix="$")
        show_chart(style_fig(f1, 280, "V7 equity"), "v7_eq")
        f2 = go.Figure(go.Scatter(x=x, y=dd, mode="lines", name="Drawdown", fill="tozeroy",
                                  line=dict(color=C["down"], width=1.5), fillcolor="rgba(230,103,103,0.15)"))
        f2.add_hline(y=-30, line=dict(color=C["critical"], width=1), annotation_text="-30% safety stop",
                     annotation_font_color=C["ink2"], annotation_position="bottom left")
        f2.update_yaxes(ticksuffix="%")
        show_chart(style_fig(f2, 200, "Drawdown from peak"), "v7_dd")

    st.markdown("**Open positions (live-priced)**")
    if pos.empty:
        st.info("V7 holds no positions right now (in cash).")
    else:
        st.dataframe(pos, width="stretch", hide_index=True, column_config={
            "Entry": st.column_config.NumberColumn(format="$%.6g"),
            "Live Price": st.column_config.NumberColumn(format="$%.6g"),
            "Value": st.column_config.NumberColumn(format="$%.2f"),
            "P&L $": st.column_config.NumberColumn(format="$%+.2f"),
            "P&L %": st.column_config.NumberColumn(format="%+.2f%%"),
        })
    st.markdown("**Trade history**")
    if trades.empty:
        st.info("No trades yet.")
    else:
        t = trades.sort_values("timestamp", ascending=False).copy()
        t["timestamp"] = t["timestamp"].dt.tz_convert(TZ).dt.strftime("%Y-%m-%d %H:%M")
        cols = [c for c in ["timestamp", "symbol", "side", "quantity", "price", "order_value_usd", "realized_pnl"]
                if c in t.columns]
        st.dataframe(t[cols], width="stretch", hide_index=True)


def ml_tab(prices: Dict[str, float]) -> None:
    status = read_json(ROOT / "models" / "ml_status.json")
    eq = read_csv(ML_LOGS / "ml_equity.csv")
    trades = read_csv(ML_LOGS / "ml_trades.csv")
    signals = read_csv(ML_LOGS / "ml_signals.csv")
    pos = ml_positions(prices)

    allowed = bool(status.get("entries_allowed"))
    st.markdown(
        (status_badge("good", "NEW ENTRIES ALLOWED") if allowed else status_badge("warn", "NEW ENTRIES PAUSED"))
        + f'<span class="note">&nbsp; {status.get("health_reason", "no research run yet")} · champion '
          f'<b>{status.get("champion", "—")}</b> · last research {str(status.get("last_research", "—"))[:10]}</span>',
        unsafe_allow_html=True,
    )
    st.markdown('<p class="note">Bracket trades at 2:1 reward:risk · 1% of the account risked per trade · '
                'stop 1.5×ATR · 14-day limit · own $500 ledger, separate from V7.</p>', unsafe_allow_html=True)

    closed = trades[trades.get("side", pd.Series(dtype=str)) == "sell"] if not trades.empty else trades
    r = pd.to_numeric(closed["r_multiple"], errors="coerce").dropna() if not closed.empty else pd.Series(dtype=float)
    ml_state = read_json(ML_LOGS / "state.json")
    ml_now = (float(ml_state.get("cash", ML_START)) + float(ml_state.get("tax_reserve", 0.0))
              + ((pos["Qty"] * pos["Live Price"]).sum() if not pos.empty else 0.0))
    st.markdown(
        '<div class="tiles">'
        + tile("ML account", money(ml_now), delta_html((ml_now / ML_START - 1) * 100))
        + tile("Closed trades", str(len(r)), f"win rate {(r > 0).mean():.0%}" if len(r) else "none yet")
        + tile("Avg R per trade", f"{r.mean():+.2f}R" if len(r) else "—", "target +2R / stop −1R")
        + tile("Open trades", str(len(pos)), "max 5")
        + "</div>",
        unsafe_allow_html=True,
    )
    if not eq.empty:
        f = go.Figure(go.Scatter(x=eq["timestamp"].dt.tz_convert(TZ), y=eq["equity"], mode="lines",
                                 name="ML equity", line=dict(color=C["s2"], width=2)))
        f.update_yaxes(tickprefix="$")
        show_chart(style_fig(f, 260, "ML account value"), "ml_eq")

    st.markdown("**Open bracket trades (live-priced)**")
    if pos.empty:
        st.info("No open ML trades.")
    else:
        st.dataframe(pos, width="stretch", hide_index=True, column_config={
            c: st.column_config.NumberColumn(format="$%.6g") for c in ("Entry", "Live Price", "Stop", "Target")
        } | {"Open R": st.column_config.NumberColumn(format="%+.2fR"),
             "Model p": st.column_config.NumberColumn(format="percent")})

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Latest daily signals** (why each coin was or wasn't bought)")
        if signals.empty:
            st.caption("No signal has cleared the EV gate yet.")
        else:
            st.dataframe(signals.iloc[::-1].head(50), width="stretch", hide_index=True)
    with c2:
        st.markdown("**ML trade log**")
        if trades.empty:
            st.caption("No ML trades yet.")
        else:
            t = trades.sort_values("timestamp", ascending=False).copy()
            t["timestamp"] = t["timestamp"].dt.tz_convert(TZ).dt.strftime("%Y-%m-%d %H:%M")
            st.dataframe(t, width="stretch", hide_index=True)

    reports = sorted((ROOT / "outputs" / "research_reports").glob("*.md"))
    if reports:
        with st.expander(f"Latest weekly research report ({reports[-1].stem})"):
            st.markdown(reports[-1].read_text(encoding="utf-8"))


TREND_LOGS = LOGS / "paper_trend"
TREND_START = 500.0


@st.cache_data(ttl=300, show_spinner=False)
def trend_signal() -> dict:
    """Why the paper strategy wants its weight: recomputed from the same completed daily candles and
    the same rule function paper_trend.py uses (read-only imports - nothing is modified)."""
    import math

    import numpy as np
    from paper_trend import completed_daily_candles
    from simple_baselines import SMA, VOL_LOOKBACK, VOL_TARGET, target_weight_d

    c = completed_daily_candles()["Close"]
    sma = float(c.rolling(SMA).mean().iloc[-1])
    vol = float(np.log(c).diff().rolling(VOL_LOOKBACK).std().iloc[-1] * math.sqrt(365))
    return {"date": c.index[-1], "close": float(c.iloc[-1]), "sma": sma, "vol": vol,
            "vol_target": VOL_TARGET, "target": float(target_weight_d(c).iloc[-1]),
            "closes": c}


def trend_tab(prices: Dict[str, float]) -> None:
    st.markdown("**Trend (paper)** - BTC long while above its 200-day average, sized to 40% volatility (max 1x), "
                "trades only when the weight is off by more than 10 points. Paper-only: no orders are sent.")
    state = read_json(TREND_LOGS / "state.json")
    if not state:
        st.info("The paper trend strategy hasn't started yet (no logs/paper_trend/state.json).")
        return
    eq = read_csv_plain(TREND_LOGS / "equity.csv", "date")
    trades = read_csv_plain(TREND_LOGS / "trades.csv", "date")

    # --- current equity / weights (live-priced) --------------------------------------------
    btc_px = prices.get("BTC")
    units, cash = float(state["units"]), float(state["cash"])
    last_close = float(eq["close"].iloc[-1]) if not eq.empty else None
    px = btc_px or last_close or 0.0
    equity = cash + units * px
    weight = units * px / equity if equity > 0 else 0.0
    target = float(state["pending_target"])
    # Signal at the last processed close -> fill at the OPEN of the next daily candle (00:00 UTC),
    # booked by paper_trend.py once that candle has completed (the following 00:00 UTC).
    fill_day = pd.Timestamp(state["last_processed"]) + pd.Timedelta(days=1)
    next_open = fill_day.tz_localize("UTC")
    booked_at = next_open + pd.Timedelta(days=1)
    gap = abs(target - weight)
    action = (f"trades to {target:.0%} at the {fill_day:%b %d} open ({et(next_open)}); "
              f"booked when that candle closes ({et(booked_at)})"
              if gap > 0.10 else f"no trade needed (gap {gap * 100:.0f} pts <= 10-pt band)")
    st.markdown(
        '<div class="tiles">'
        + tile("Trend paper equity", money(equity), delta_html((equity / TREND_START - 1) * 100)
               + f" vs {money(TREND_START)} start · live-priced")
        + tile("BTC weight now", f"{weight:.0%}", f"{units:.6f} BTC · cash {money(cash)}")
        + tile("Target weight", f"{target:.0%}", action)
        + tile("Running since", str(state["start_date"]), f"last close processed {state['last_processed']}")
        + "</div>",
        unsafe_allow_html=True,
    )

    # --- why -----------------------------------------------------------------------------
    try:
        sig = trend_signal()
        above = sig["close"] > sig["sma"]
        raw = min(1.0, sig["vol_target"] / sig["vol"]) if sig["vol"] > 0 else 0.0
        st.markdown(
            '<div class="tiles">'
            + tile("BTC vs 200-day avg", status_badge("good" if above else "warn", "ABOVE" if above else "BELOW"),
                   f"close {price_fmt(sig['close'])} vs avg {price_fmt(sig['sma'])} "
                   f"({(sig['close'] / sig['sma'] - 1) * 100:+.1f}%) on {sig['date'].date()}")
            + tile("30-day volatility vs 40%", f"{sig['vol']:.1%}",
                   f"40% / {sig['vol']:.1%} = {sig['vol_target'] / sig['vol']:.2f}x -> capped at 1x = {raw:.0%}"
                   if raw >= 1.0 else f"40% / {sig['vol']:.1%} = {raw:.0%} of capital")
            + tile("Rule result", f"{sig['target']:.0%}",
                   ("above the average, so size = min(1, 40% / vol)" if above else "below the average, so cash")
                   + ("" if abs(sig["target"] - target) < 1e-9 else " · differs from ledger: next run will update"))
            + "</div>",
            unsafe_allow_html=True,
        )
    except Exception as exc:
        st.warning(f"Couldn't recompute the signal right now: {exc}")

    # --- last run / errors -----------------------------------------------------------------
    log = TREND_LOGS / "run.log"
    lines = log.read_text(encoding="utf-8", errors="ignore").splitlines() if log.exists() else []
    errors = [l for l in lines if " ERROR " in l]
    if lines:
        last = lines[-1]
        fresh = datetime.now(timezone.utc) - pd.Timestamp(last[:19]).tz_localize("UTC").to_pydatetime() < timedelta(hours=2)
        st.markdown(
            status_badge("good" if fresh and " ERROR " not in last else "warn",
                         f"Last run {et(pd.Timestamp(last[:19]).tz_localize('UTC'))}")
            + f'<span class="note"> &nbsp;{last[24:]}</span>', unsafe_allow_html=True)
    if errors:
        st.markdown(status_badge("crit", f"{len(errors)} error(s) in run.log") + " - most recent:",
                    unsafe_allow_html=True)
        st.code("\n".join(errors[-5:]), language=None)
    else:
        st.markdown(status_badge("good", "No errors in run.log"), unsafe_allow_html=True)

    # --- comparison chart: all rebased to $500 at the trend start close ---------------------
    start = pd.Timestamp(state["start_date"])
    series = {}
    if not eq.empty:
        series["Trend (paper)"] = eq.set_index("date")["equity"].astype(float)
    v7 = read_csv(LOGS / "paper_equity.csv")
    last_done = pd.Timestamp(state["last_processed"])
    if not v7.empty:
        ts = v7["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
        val = v7["total_equity"].astype(float)
        after_close = val[ts >= start + pd.Timedelta(days=1)]          # V7 logs hourly, not at 00:00 UTC
        if not after_close.empty:
            anchor = float(after_close.iloc[0])                      # first V7 value after the start close
            daily = val.groupby(ts.dt.normalize()).last()
            daily = daily[(daily.index > start) & (daily.index <= last_done)]   # completed days only
            v7s = pd.concat([pd.Series({start: anchor}), daily])
            series["V7 (paper)"] = v7s / anchor * TREND_START
    try:
        closes = trend_signal()["closes"]
        btc = closes[closes.index >= start]
        if not btc.empty:
            series["BTC buy & hold"] = btc / btc.iloc[0] * TREND_START
    except Exception:
        pass
    colors = {"Trend (paper)": C["s3"], "V7 (paper)": C["s1"], "BTC buy & hold": C["ink2"]}
    f = go.Figure()
    for name, s in series.items():
        f.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines+markers", name=name,
                               line=dict(color=colors[name], width=2), marker=dict(size=8)))
    f.add_hline(y=TREND_START, line=dict(color=C["axis"], width=1))
    f.update_yaxes(tickprefix="$")
    last_x = max((s.index.max() for s in series.values()), default=start)
    f.update_xaxes(type="date", tickformat="%b %d", dtick=86400000,          # whole days, never sub-second ticks
                   range=[start - pd.Timedelta(days=1), max(last_x, start + pd.Timedelta(days=7))])
    show_chart(style_fig(f, 320, f"Daily equity, all rebased to $500 at the {start.date()} close"), "trend_eq")
    days = max(len(s) for s in series.values()) if series else 0
    if days < 5:
        st.caption(f"Only {days} daily point(s) so far - the comparison fills in one point per completed day (00:00 UTC).")

    dd_rows = []
    for name, s in series.items():
        dd_rows.append({"Strategy": name, "Value now (rebased)": float(s.iloc[-1]),
                        "Return": float(s.iloc[-1] / TREND_START - 1),
                        "Max drawdown": float((s / s.cummax() - 1).min()), "Daily points": len(s)})
    st.dataframe(pd.DataFrame(dd_rows), width="stretch", hide_index=True, column_config={
        "Value now (rebased)": st.column_config.NumberColumn(format="$%.2f"),
        "Return": st.column_config.NumberColumn(format="percent"),
        "Max drawdown": st.column_config.NumberColumn(format="percent")})

    st.markdown("**Trades** (simulated fills at the daily open, 0.15% cost per side)")
    if trades.empty:
        st.caption(f"No trades booked yet - next fill uses the {fill_day:%b %d} open price ({et(next_open)}) "
                   f"and is booked at {et(booked_at)}." if gap > 0.10 else "No trades booked yet.")
    else:
        st.dataframe(trades.sort_values("date", ascending=False), width="stretch", hide_index=True)


def read_csv_plain(path: Path, date_col: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, parse_dates=[date_col])
        return df.sort_values(date_col)
    except Exception:
        return pd.DataFrame()


def news_tab() -> None:
    from news_feed import coin_news_risk, coins_mentioned, headline_sentiment, market_mood

    fng = fear_greed(365)
    try:
        items = headlines()
    except Exception as exc:
        items = []
        st.warning(f"Headlines unavailable: {exc}")
    risk = coin_news_risk(items)
    mood = market_mood(items)

    left, right = st.columns([5, 7], gap="medium")
    with left:
        if not fng.empty:
            now, lbl = fng["fng"].iloc[-1], fng["label"].iloc[-1]
            wk = fng["fng"].iloc[-8] if len(fng) > 8 else None
            st.markdown(
                '<div class="tiles">'
                + tile("Fear & Greed today", f"{now:.0f}", f"{lbl} · 0 = extreme fear, 100 = extreme greed")
                + tile("1 week ago", f"{wk:.0f}" if wk is not None else "—",
                       f"{now - wk:+.0f} pts" if wk is not None else "")
                + tile("Headline mood (24h)", f"{mood:+.2f}" if mood is not None else "—",
                       "word-count score, + positive / − negative")
                + "</div>",
                unsafe_allow_html=True,
            )
            f = go.Figure(go.Scatter(x=fng["date"], y=fng["fng"], mode="lines", name="Fear & Greed",
                                     line=dict(color=C["s1"], width=2)))
            for y in (25, 50, 75):
                f.add_hline(y=y, line=dict(color=C["axis"], width=1))
            # Zone names live on the axis, so they never sit on top of the line.
            f.update_yaxes(range=[0, 100], tickvals=[0, 25, 50, 75, 100],
                           ticktext=["0", "25 extreme fear", "50 neutral", "75 extreme greed", "100"])
            f = style_fig(f, 300, "Fear & Greed Index · 1 year (ML research feature)")
            f.update_layout(margin=dict(l=120))
            show_chart(f, "fng")
        st.markdown("**Coins flagged by the news filter**")
        if not risk:
            st.markdown(status_badge("good", "No coin flagged in the last 48h"), unsafe_allow_html=True)
        for coin, info in risk.items():
            lvl = "crit" if info["level"] == "severe" else "warn"
            action = "forces an exit if held" if info["level"] == "severe" else "blocks new buys"
            st.markdown(status_badge(lvl, f"{coin}: {info['level'].upper()}") + f'<span class="note"> &nbsp;{action}</span>',
                        unsafe_allow_html=True)
            for h in info["headlines"][:3]:
                st.markdown(f'<div class="note">• {h}</div>', unsafe_allow_html=True)

    with right:
        c1, c2 = st.columns([3, 2])
        query = c1.text_input("Filter headlines", placeholder="e.g. XRP, ETF, hack")
        only_ours = c2.toggle("Only coins the bot trades", value=False)
        st.caption(f"{len(items)} headlines from Cointelegraph, CoinDesk, Decrypt, The Block · refreshed every 5 min")
        shown = 0
        for x in sorted(items, key=lambda i: i["published"], reverse=True):
            coins = coins_mentioned(x["title"])
            if only_ours and not coins:
                continue
            if query and query.lower() not in (x["title"] + " " + x.get("summary", "")).lower():
                continue
            s = headline_sentiment(x["title"])
            mood_tag = (f'<span class="tag up">▲ positive</span>' if s > 0 else
                        f'<span class="tag down">▼ negative</span>' if s < 0 else "")
            coin_tags = "".join(f'<span class="tag">{c}</span>' for c in coins)
            title = x["title"].replace("<", "&lt;")
            st.markdown(
                f'<div class="news"><a href="{x["link"]}" target="_blank">{title}</a>'
                f'<div class="meta">{x["source"]} · {et(x["published"])} &nbsp;{mood_tag}{coin_tags}</div></div>',
                unsafe_allow_html=True,
            )
            shown += 1
            if shown >= 40:
                break
        if shown == 0:
            st.caption("No headlines match.")


def safety_tab(status: dict) -> None:
    st.markdown("**Safety systems**")
    rows = [
        ("Autopilot", "good" if status["running"] else "warn",
         "running" if status["running"] else "not running - start with install_autopilot.ps1"),
        ("Emergency stop flag", "crit" if status["emergency_flag"] else "good",
         "SET - bot will not trade" if status["emergency_flag"] else "not set"),
        ("Drawdown safety stop", "crit" if status["safety_stop"] else "good",
         "TRIGGERED - review logs" if status["safety_stop"] else "not triggered (stops at -12% day / -30% total)"),
    ]
    for name, lvl, text in rows:
        st.markdown(f'{status_badge(lvl, name)} <span class="note">&nbsp;{text}</span>', unsafe_allow_html=True)

    st.markdown("**Anti-rug-pull checks on meme coins** (history · live order book · on-chain contract)")
    ar = read_csv(LOGS / "anti_rug_log.csv")
    if ar.empty:
        st.caption("No anti-rug checks logged yet.")
    else:
        ar = ar.sort_values("timestamp", ascending=False).copy()
        ar["timestamp"] = ar["timestamp"].dt.tz_convert(TZ).dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(ar.head(100), width="stretch", hide_index=True)

    st.markdown("**Risk limits**")
    st.dataframe(pd.DataFrame([
        ("V7 max order", "12% of equity"), ("V7 max position", "36% of equity"),
        ("V7 max exposure", "95%"), ("Daily loss stop", "-12%"), ("Total drawdown stop", "-30%"),
        ("Meme sleeve cap", "3% total / 1.5% per coin"),
        ("ML risk per trade", "1% of ML account"), ("ML max position", "20% of ML account"),
        ("ML reward:risk", "2.0"), ("ML entry gate", "expected value ≥ +0.2R and research health OK"),
    ], columns=["Limit", "Setting"]), width="stretch", hide_index=True)

    log = LOGS / "autopilot.log"
    if log.exists():
        with st.expander("Autopilot log (last 60 lines)"):
            lines = log.read_text(encoding="utf-8", errors="ignore").splitlines()[-60:]
            st.code("\n".join(lines), language=None)


# --------------------------------------------------------------------------- #
# Page                                                                        #
# --------------------------------------------------------------------------- #
inject_css()
status = bot_status()
header(status)
run_control_bar()
ticker_strip()

held = [p.split("/")[0] for p in (read_json(ML_LOGS / "state.json").get("positions") or {})]
snap = read_csv(LOGS / "positions_snapshot.csv")
if not snap.empty:
    held += [s.split("/")[0] for s in snap["symbol"].astype(str)]
prices = live_price_map(WATCHLIST + held)

tabs = st.tabs(["Overview", "Live Prices", "V7 Portfolio", "ML Sleeve", "Trend (paper)", "News", "Safety"])
with tabs[0]:
    overview_tab(prices)
with tabs[1]:
    prices_tab()
with tabs[2]:
    v7_tab(prices)
with tabs[3]:
    ml_tab(prices)
with tabs[4]:
    trend_tab(prices)
with tabs[5]:
    news_tab()
with tabs[6]:
    safety_tab(status)
