"""
Survivorship-free daily data for EVERY Binance USDT spot pair ever listed (incl. delisted
coins such as LUNA, FTT, SRM), from Binance's public archive data.binance.vision.

Used for point-in-time universe selection ("top N coins by size as of each date").
Kept in universe_data/ - NOT historic_data/ - so V7's auto-discovered file list is untouched.

Run: python universe_data.py           (resumable; only fetches months not on disk)
"""

from __future__ import annotations

import io
import re
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "universe_data" / "binance_1d"
S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
FILES = "https://data.binance.vision"

# Not crypto exposure: stablecoins / fiat / wrapped duplicates / tokenized gold & stocks.
EXCLUDE_BASES = {
    "USDC", "BUSD", "TUSD", "USDP", "PAX", "DAI", "FDUSD", "USDE", "PYUSD", "RLUSD", "USD1", "XUSD",
    "BFUSD", "UST", "USTC", "SUSD", "USDS", "USDSB", "EUR", "EURI", "AEUR", "GBP", "AUD", "TRY", "BRL",
    "RUB", "UAH", "NGN", "BIDR", "IDRT", "BVND", "ZAR", "JPY", "WBTC", "WBETH", "BETH", "PAXG", "XAUT",
    "BTCST",
}
LEVERAGED = re.compile(r".+(UP|DOWN|BULL|BEAR)$")
TOKENIZED_STOCK = re.compile(r"^[A-Z]{2,6}B$")   # 2026 tokenized equities (NVDAB, MSTRB, ...)
KNOWN_B_COINS = {"BNB", "ARB", "SB", "FB", "CVB", "LAB"}   # real coins that happen to end in B


def is_crypto_base(base: str) -> bool:
    if base in EXCLUDE_BASES or LEVERAGED.match(base):
        return False
    if TOKENIZED_STOCK.match(base) and base not in KNOWN_B_COINS and len(base) >= 4:
        return False
    return True


def _list(prefix: str) -> List[str]:
    keys, marker = [], ""
    while True:
        for attempt in range(4):                                   # the archive's S3 listing times out now and then
            try:
                r = requests.get(S3, params={"delimiter": "/", "prefix": prefix, "marker": marker}, timeout=60)
                r.raise_for_status()
                break
            except requests.RequestException:
                if attempt == 3:
                    raise
                time.sleep(2 * (attempt + 1))
        found = re.findall(r"<(?:Prefix|Key)>([^<]+)</(?:Prefix|Key)>", r.text)
        found = [f for f in found if f != prefix]
        keys += found
        if "<IsTruncated>true</IsTruncated>" not in r.text or not found:
            return keys
        marker = found[-1]


def usdt_symbols() -> List[str]:
    syms = [p.rstrip("/").split("/")[-1] for p in _list("data/spot/monthly/klines/")]
    return sorted(s for s in syms if s.endswith("USDT") and is_crypto_base(s[:-4]))


def _parse(raw: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        txt = z.read(z.namelist()[0]).decode()
    df = pd.read_csv(io.StringIO(txt), header=None)
    if not str(df.iloc[0, 0]).lstrip("-").isdigit():            # some files carry a header row
        df = df.iloc[1:].astype(float)
    t = df[0].astype("int64")
    unit = "us" if t.max() > 1e14 else "ms"                        # archive switched to microseconds in 2025
    return pd.DataFrame({
        "Date": pd.to_datetime(t, unit=unit).dt.normalize(),
        "Open": df[1].astype(float), "High": df[2].astype(float), "Low": df[3].astype(float),
        "Close": df[4].astype(float), "Volume": df[5].astype(float), "QuoteVolume": df[7].astype(float),
    })


def fetch_symbol(sym: str) -> str:
    path = OUT / f"{sym}.csv"
    have = pd.read_csv(path, parse_dates=["Date"]) if path.exists() else pd.DataFrame()
    have_months = set(have["Date"].dt.strftime("%Y-%m")) if not have.empty else set()
    keys = [k for k in _list(f"data/spot/monthly/klines/{sym}/1d/") if k.endswith(".zip")]
    frames = [have] if not have.empty else []
    newest = max(have_months) if have_months else None
    for k in keys:
        month = k[-11:-4]                                          # ...-YYYY-MM.zip
        if month in have_months and month != newest:               # re-fetch the newest stored month
            continue
        for attempt in range(3):
            try:
                r = requests.get(f"{FILES}/{k}", timeout=30)
                r.raise_for_status()
                frames.append(_parse(r.content))
                break
            except Exception:
                time.sleep(1 + attempt)
    if not frames:
        return f"{sym}: no data"
    df = pd.concat(frames).drop_duplicates("Date", keep="last").sort_values("Date")
    df.to_csv(path, index=False)
    return f"{sym}: {len(df)} days {df['Date'].min().date()}..{df['Date'].max().date()}"


def download(workers: int = 16) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    syms = usdt_symbols()
    print(f"{len(syms)} USDT crypto pairs in the archive (listed + delisted)")
    done, failed = 0, []
    with ThreadPoolExecutor(workers) as ex:
        futures = {ex.submit(fetch_symbol, s): s for s in syms}
        for fut in as_completed(futures):
            done += 1
            try:
                msg = fut.result()
            except Exception as exc:                              # one bad symbol must not stop the rest
                failed.append(futures[fut])
                msg = f"{futures[fut]}: FAILED ({exc})"
            if done % 50 == 0 or done == len(syms):
                print(f"  {done}/{len(syms)}  last: {msg}", flush=True)
    print(f"done. failed: {failed or 'none'} (rerun to retry - completed months are kept)")


def _split_reused_tickers(base: str, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Binance re-used some tickers for a different asset after a delisting (LUNA: old Terra until
    2022-05-13, Terra 2.0 from 2022-05-31 at a 20,000x different price) or redenominated tokens
    (VEN->VET, STRAX, VIDT). A gap > 7 days with a > e (2.7x) price jump splits the file into
    separate coins, so the backtest never sees a fake jump - the first segment simply ends
    (treated as a delisting).
    """
    idx = df.index
    gaps = idx.to_series().diff().dt.days
    cuts = []
    for day in gaps[gaps > 7].index:
        pos = idx.get_loc(day)
        before, after = float(df["Close"].iloc[pos - 1]), float(df["Open"].iloc[pos])
        if before > 0 and abs(np.log(after / before)) > 1.0:
            cuts.append(pos)
    if not cuts:
        return {f"{base}(USD)": df}
    out, start = {}, 0
    for n, cut in enumerate(cuts + [len(df)]):
        out[f"{base}{'' if n == 0 else f'#{n + 1}'}(USD)"] = df.iloc[start:cut]
        start = cut
    return out


def load_all(min_days: int = 30) -> Dict[str, pd.DataFrame]:
    """{BASE(USD): OHLCV df indexed by date} for every downloaded pair (reused tickers split)."""
    out = {}
    for p in sorted(OUT.glob("*USDT.csv")):
        df = pd.read_csv(p, parse_dates=["Date"]).set_index("Date").sort_index()
        df = df[(df[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]
        df = df[df.index >= "2017-01-01"]                        # drop corrupt epoch-era timestamps
        df = df[~df.index.duplicated(keep="last")]
        for sym, part in _split_reused_tickers(p.stem[:-4], df).items():
            if len(part) >= min_days:
                out[sym] = part
    return out


if __name__ == "__main__":
    download()
