"""
Update historic_data CSVs with missing price data via yfinance.
Run: python update_data.py

Reads each USD CSV file, detects the last date, fetches forward to today,
and prepends the new rows in the same reverse-chronological Yahoo Finance format.
"""

import csv
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).parent / "historic_data"
# Only store COMPLETE daily candles (Yahoo crypto days are UTC). Today's bar is still
# forming; writing it would freeze a partial candle into the file forever.
TODAY = datetime.now(timezone.utc).date() - timedelta(days=1)
CSV_HEADER = "Date,Open,High,Low,Close,Adj Close,Volume"

# Map from CSV filename prefix to Yahoo Finance ticker
# Only USD-quoted assets (non-USD are excluded from trading anyway)
FILE_TICKER_MAP = {
    "AAVE(USD)-CRYPTO - Sheet1 (1)": "AAVE-USD",
    "AAVE(USD)-CRYPTO - Sheet1":     "AAVE-USD",
    "ARB(USD)-CRYPTO - Sheet1":      "ARB11841-USD",  # ARB-USD is wrong ticker; 11841 is Arbitrum
    "CENTRIFUGE(USD)-CRYPTO - Sheet1": "CFG-USD",
    "CLEARPOOL(usd)-CRYPTO - Sheet1":  "CPOOL-USD",
    "Chainlink(USD)-crypto - Sheet1":  "LINK-USD",
    "HBAR(USD)-CRYPTO - Sheet1":       "HBAR-USD",
    "MAPLE(USD)-CRYPTO - Sheet1":      "MPL-USD",
    "ONDO(USD)- crypto - Sheet1":      "ONDO-USD",
    "OP(USD)-CRYPTO - Sheet1":         "OP-USD",
    "PENDLE(USD)-CRYPTO - Sheet1":     "PENDLE-USD",
    "SKY(USD)-CRYPTO - Sheet1":        "SKY-USD",
    "SOL(usd)-CRYPTO - Sheet1":        "SOL-USD",
    "UNI(USD)-CRYPTO - Sheet1":        "UNI7083-USD",  # Yahoo Finance ID for Uniswap
    "XRP-CRYPTO - Sheet1":             "XRP-USD",
    "ZEBEC(USD)-CRYPTO - Sheet1":      "ZBCN-USD",
    # New assets added June 2027
    "AVAX(USD)-CRYPTO - Sheet1":       "AVAX-USD",
    "SUI(USD)-CRYPTO - Sheet1":        "SUI20947-USD",   # SUI20947-USD is correct Yahoo ticker
    "NEAR(USD)-CRYPTO - Sheet1":       "NEAR-USD",
    "POLYX(USD)-CRYPTO - Sheet1":      "POLYX-USD",      # experimental
    "PRO(USD)-CRYPTO - Sheet1":        "PRO-USD",        # experimental (Propy)
    "RIO(USD)-CRYPTO - Sheet1":        "RIO-USD",        # experimental (Realio)
    # Non-USD — update anyway so they stay current if config changes
    "MKR(EUR)-CRYPTO - Sheet1":        "MKR-EUR",
    "SNX(AUD)-CRYPTO - Sheet1":        "SNX-AUD",
    "VECHAIN(EUR)-CRYPTO - Sheet1":    "VET-EUR",
    "COMP(CAD)-CRYPTO - Sheet1":       "COMP-CAD",
    # Meme coins (Sep 2026) — created from full history on first run if missing.
    # Yahoo quantizes daily prices to $0.000001, which destroys sub-cent tokens
    # (SHIB/PEPE/BONK/FLOKI), so those come from Binance's public data mirror.
    "DOGE(USD)-CRYPTO - Sheet1":       "DOGE-USD",
    "SHIB(USD)-CRYPTO - Sheet1":       "binance:SHIBUSDT",
    "PEPE(USD)-CRYPTO - Sheet1":       "binance:PEPEUSDT",
    "BONK(USD)-CRYPTO - Sheet1":       "binance:BONKUSDT",
    "WIF(USD)-CRYPTO - Sheet1":        "WIF-USD",
    "FLOKI(USD)-CRYPTO - Sheet1":      "binance:FLOKIUSDT",
}
BINANCE_DATA_URL = "https://data-api.binance.vision/api/v3/klines"
REFERENCE_DIR = Path(__file__).parent / "reference_data"
REFERENCE_TICKERS = {
    "BTC(USD)": "binance:BTCUSDT",
    "ETH(USD)": "binance:ETHUSDT",
}
BOOTSTRAP_IF_MISSING = {
    "DOGE(USD)-CRYPTO - Sheet1", "SHIB(USD)-CRYPTO - Sheet1", "PEPE(USD)-CRYPTO - Sheet1",
    "BONK(USD)-CRYPTO - Sheet1", "WIF(USD)-CRYPTO - Sheet1", "FLOKI(USD)-CRYPTO - Sheet1",
}

# CoinGecko fallbacks for tokens Yahoo Finance doesn't carry
COINGECKO_IDS = {
    "CFG-USD":   "centrifuge",
    "CPOOL-USD": "clearpool",
    "ZBCN-USD":  "zebec-network",
    "MPL-USD":   "maple",
    "SKY-USD":   "sky-governance-token",
}


def find_latest_date(csv_path: Path) -> date | None:
    """Return the most-recent date found in a Yahoo Finance-format CSV."""
    import csv as csvmod
    try:
        with csv_path.open("r", encoding="utf-8-sig", errors="ignore", newline="") as f:
            reader = csvmod.reader(f)
            for row in reader:
                if not row or all(c.strip() == "" for c in row):
                    continue
                date_val = row[0].strip()
                if date_val.lower() == "date":
                    continue
                for fmt in ("%b %d, %Y", "%b  %d, %Y", "%Y-%m-%d"):
                    try:
                        return datetime.strptime(date_val, fmt).date()
                    except ValueError:
                        continue
    except Exception as e:
        print(f"  [warn] Could not read {csv_path.name}: {e}")
    return None


def format_volume(vol) -> str:
    """Format volume as integer with commas, or '-' if zero/NaN."""
    try:
        v = float(vol)
        if v <= 0 or pd.isna(v):
            return "-"
        return f"{int(v):,}"
    except Exception:
        return "-"


def format_price(val) -> str:
    """Format price — strip unnecessary trailing zeros."""
    try:
        v = float(val)
        if pd.isna(v):
            return "0"
        if v >= 1000:
            return f"{v:.2f}"
        if v >= 1:
            return f"{v:.4f}".rstrip("0").rstrip(".")
        # Sub-$1: keep 6 significant digits in plain decimal form, so sub-cent
        # meme prices (e.g. SHIB 0.00000577) are not rounded away.
        return np.format_float_positional(v, precision=6, unique=False, fractional=False, trim="-")
    except Exception:
        return "0"


def rows_to_yahoo_lines(df: pd.DataFrame) -> list[str]:
    """Convert a DataFrame (ascending date index) to Yahoo Finance CSV lines (newest first)."""
    lines = []
    for ts, row in df.iloc[::-1].iterrows():
        d = ts.date() if hasattr(ts, "date") else ts
        # Produce "Jun 5, 2026" (no zero-padding on day)
        date_str = f"{d.strftime('%b')} {d.day}, {d.year}"
        o  = format_price(row.get("Open",  row.get("open",  0)))
        h  = format_price(row.get("High",  row.get("high",  0)))
        lo = format_price(row.get("Low",   row.get("low",   0)))
        c  = format_price(row.get("Close", row.get("close", 0)))
        vol = format_volume(row.get("Volume", row.get("volume", 0)))
        # Yahoo format: Date,Open,High,Low,Close,Adj Close,Volume
        # Volume is comma-formatted so must be quoted to avoid breaking CSV parsing
        lines.append(f'"{date_str}",{o},{h},{lo},{c},{c},"{vol}"')
    return lines


def fetch_yfinance(ticker: str, start: date, end: date) -> pd.DataFrame | None:
    try:
        df = yf.download(ticker, start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(),
                         progress=False, auto_adjust=True)
        if df.empty:
            return None
        # yfinance may return MultiIndex columns — flatten
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index).normalize()
        return df
    except Exception as e:
        print(f"  [yfinance error] {ticker}: {e}")
        return None


def fetch_binance(pair: str, start: date, end: date) -> pd.DataFrame | None:
    """
    Daily klines from Binance's public market-data mirror (no API key).
    Volume is written as QUOTE volume (USDT) to match Yahoo's USD-volume convention.
    """
    import requests

    rows = []
    cursor = int(datetime(start.year, start.month, start.day, tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = int(datetime(end.year, end.month, end.day, 23, 59, tzinfo=timezone.utc).timestamp() * 1000)
    try:
        while cursor <= end_ms:
            resp = requests.get(
                BINANCE_DATA_URL,
                params={"symbol": pair, "interval": "1d", "startTime": cursor, "endTime": end_ms, "limit": 1000},
                timeout=20,
            )
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            for k in batch:
                rows.append({
                    "Date": pd.Timestamp(k[0], unit="ms").normalize(),
                    "Open": float(k[1]), "High": float(k[2]), "Low": float(k[3]),
                    "Close": float(k[4]), "Volume": float(k[7]),
                })
            cursor = int(batch[-1][0]) + 86_400_000
            if len(batch) < 1000:
                break
            time.sleep(0.2)
    except Exception as e:
        print(f"  [binance error] {pair}: {e}")
        return None
    if not rows:
        return None
    return pd.DataFrame(rows).set_index("Date").sort_index()


def fetch_prices(ticker: str, start: date, end: date) -> pd.DataFrame | None:
    if ticker.startswith("binance:"):
        return fetch_binance(ticker.split(":", 1)[1], start, end)
    return fetch_yfinance(ticker, start, end)


def fetch_coingecko(cg_id: str, start: date, end: date) -> pd.DataFrame | None:
    """Fetch OHLCV from CoinGecko public API (no key needed, free tier)."""
    try:
        base = "https://api.coingecko.com/api/v3"
        # CoinGecko returns daily OHLC (open/high/low/close) for >90d ranges
        start_ts = int(datetime(start.year, start.month, start.day).timestamp())
        end_ts   = int(datetime(end.year,   end.month,   end.day,   23, 59).timestamp())
        url = f"{base}/coins/{cg_id}/ohlc/range?vs_currency=usd&from={start_ts}&to={end_ts}&interval=daily"
        resp = __import__("requests").get(url, timeout=15)
        if resp.status_code != 200:
            # Fallback: market_chart endpoint for price + volume
            url2 = (f"{base}/coins/{cg_id}/market_chart/range"
                    f"?vs_currency=usd&from={start_ts}&to={end_ts}")
            resp = __import__("requests").get(url2, timeout=15)
            if resp.status_code != 200:
                print(f"  [coingecko error] {cg_id}: HTTP {resp.status_code}")
                return None
            data = resp.json()
            prices  = {int(x[0]//1000): x[1] for x in data.get("prices", [])}
            volumes = {int(x[0]//1000): x[1] for x in data.get("total_volumes", [])}
            rows = []
            for ts, price in prices.items():
                d = datetime.utcfromtimestamp(ts).date()
                rows.append({"Date": d, "Open": price, "High": price,
                              "Low": price, "Close": price,
                              "Volume": volumes.get(ts, 0)})
            if not rows:
                return None
            df = pd.DataFrame(rows).set_index("Date")
            df.index = pd.to_datetime(df.index)
            return df
        data = resp.json()
        rows = []
        for item in data:
            # [timestamp_ms, open, high, low, close]
            d = datetime.utcfromtimestamp(item[0] / 1000).date()
            rows.append({"Date": d, "Open": item[1], "High": item[2],
                          "Low": item[3], "Close": item[4], "Volume": 0})
        if not rows:
            return None
        df = pd.DataFrame(rows).set_index("Date")
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        print(f"  [coingecko error] {cg_id}: {e}")
        return None


def _line_date(line: str) -> date | None:
    try:
        cell = next(csv.reader([line]))[0].strip()
    except Exception:
        return None
    for fmt in ("%b %d, %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(cell, fmt).date()
        except ValueError:
            continue
    return None


def prepend_rows_to_csv(csv_path: Path, new_lines: list[str], replace_from: date | None = None) -> None:
    """
    Insert new data lines right after the header row in the existing CSV.
    Existing rows dated >= replace_from are dropped first (they are being re-downloaded,
    which repairs any partial candle written by an earlier intraday run).
    """
    content = csv_path.read_text(encoding="utf-8-sig", errors="ignore")
    lines = content.splitlines()
    if replace_from is not None:
        kept = []
        for line in lines:
            d = _line_date(line)
            if d is not None and d >= replace_from:
                continue
            kept.append(line)
        lines = kept

    # Find the header line index
    header_idx = None
    for i, line in enumerate(lines):
        parts = [p.strip().strip('"').lower() for p in line.split(",")]
        if "date" in parts and "open" in parts and "close" in parts:
            header_idx = i
            break

    if header_idx is None:
        print(f"  [error] No header found in {csv_path.name}, skipping.")
        return

    updated = (
        lines[: header_idx + 1]
        + new_lines
        + lines[header_idx + 1 :]
    )
    csv_path.write_text("\n".join(updated) + "\n", encoding="utf-8")


def bootstrap_file(csv_path: Path, ticker: str) -> bool:
    """Create a new Yahoo-format CSV from the ticker's full daily history."""
    print(f"  [new]  Creating {csv_path.name} from full {ticker} history ...", end=" ", flush=True)
    df = fetch_prices(ticker, date(2015, 1, 1), TODAY)
    if df is None or df.empty:
        print("no data returned.")
        return False
    df = df[df.index.date <= TODAY]  # type: ignore[operator]
    lines = rows_to_yahoo_lines(df)
    csv_path.write_text("\n".join([CSV_HEADER] + lines) + "\n", encoding="utf-8")
    print(f"{len(lines)} bars ({df.index[0].date()} -> {df.index[-1].date()}).")
    return True


def update_file(csv_path: Path, ticker: str) -> bool:
    latest = find_latest_date(csv_path)
    if latest is None:
        print(f"  [skip] Could not determine latest date in {csv_path.name}")
        return False

    # Always re-download the newest stored bar: it may be a partial intraday candle.
    fetch_start = min(latest, TODAY)
    print(f"  Fetching {ticker} from {fetch_start} to {TODAY} ...", end=" ", flush=True)

    df = fetch_prices(ticker, fetch_start, TODAY)

    if df is None or df.empty:
        cg_id = COINGECKO_IDS.get(ticker)
        if cg_id:
            print(f"yfinance empty, trying CoinGecko ({cg_id}) ...", end=" ", flush=True)
            time.sleep(1.2)  # respect free-tier rate limit
            df = fetch_coingecko(cg_id, fetch_start, TODAY)

    if df is None or df.empty:
        print(f"no data returned.")
        return False

    # Filter to only complete bars in the requested range
    df = df[(df.index.date >= fetch_start) & (df.index.date <= TODAY)]  # type: ignore[operator]
    if df.empty:
        print(f"0 new bars.")
        return True

    new_lines = rows_to_yahoo_lines(df)
    replace_from = df.index.min().date()
    print(f"{len(new_lines)} bars (replacing stored rows from {replace_from}).")
    prepend_rows_to_csv(csv_path, new_lines, replace_from=replace_from)
    return True


def main():
    print(f"Updating historic_data CSVs through {TODAY}\n")
    success, skipped, failed = 0, 0, 0

    for stem, ticker in FILE_TICKER_MAP.items():
        # Match by stem (handle spaces in filename)
        candidates = list(DATA_DIR.glob(f"{stem}.csv"))
        if not candidates:
            # Try partial match
            candidates = [f for f in DATA_DIR.iterdir() if f.stem == stem]
        if not candidates and stem in BOOTSTRAP_IF_MISSING:
            if bootstrap_file(DATA_DIR / f"{stem}.csv", ticker):
                success += 1
            else:
                failed += 1
            time.sleep(0.5)
            continue
        if not candidates:
            print(f"  [miss] No file found for '{stem}'")
            skipped += 1
            continue

        for csv_path in candidates:
            ok = update_file(csv_path, ticker)
            if ok:
                success += 1
            else:
                failed += 1
        time.sleep(0.5)  # polite delay between tickers

    # Market reference series for the ML model. Kept OUT of historic_data/ so the
    # core V7 universe (which auto-discovers that folder) is unchanged.
    REFERENCE_DIR.mkdir(exist_ok=True)
    for name, ticker in REFERENCE_TICKERS.items():
        path = REFERENCE_DIR / f"{name}.csv"
        ok = update_file(path, ticker) if path.exists() else bootstrap_file(path, ticker)
        if ok:
            success += 1
        else:
            failed += 1
        time.sleep(0.5)

    print(f"\nDone. Updated: {success}  Skipped/Missing: {skipped}  Failed: {failed}")


if __name__ == "__main__":
    main()
