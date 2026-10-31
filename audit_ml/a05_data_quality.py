"""
5. Data quality. data.py:341-347 (_drop_invalid_rows) only drops NaN rows - it never
rejects zero/negative prices, broken OHLC, or flat lines. Scan every file, then check
which bad rows survive into the ML training set (label not NaN, bars >= 200).
Also: are non-USD files (data.py:126-157) kept out, and do stored Yahoo candles still
match a fresh download (stale partial candles from pre-fix update runs)?
"""
import io
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

import _common as c
from data import DataManager
from main import build_symbol_file_map

cfg = c.cfg_for()
dm = DataManager(cfg)
with redirect_stdout(io.StringIO()):
    loaded = dm.load_multi_csv(build_symbol_file_map(cfg, c.ROOT))
usd, _, excluded = dm.filter_by_allowed_quote_currency(loaded)
ds = c.dataset()
train = ds[ds["label"].notna()]
train_keys = set(zip(train["symbol"], train["date"]))

rows = []
for sym, df in sorted(loaded.items()):
    o, h, l, cl, v = (df[k].astype(float) for k in ("Open", "High", "Low", "Close", "Volume"))
    ret = cl.pct_change()
    flat = (cl.diff() == 0)
    run = flat.groupby((~flat).cumsum()).cumsum()                 # consecutive unchanged closes
    issues = {
        "nonpositive_price": (o <= 0) | (h <= 0) | (l <= 0) | (cl <= 0),
        "broken_ohlc": (h < np.maximum(o, cl) * (1 - 1e-9)) | (l > np.minimum(o, cl) * (1 + 1e-9)),
        "ohlc_all_equal": (o == h) & (h == l) & (l == cl),
        "flatline_5d+": run >= 4,
        "zero_volume": v <= 0,
        "jump_over_60pct": ret.abs() > 0.60,
    }
    gaps = df.index.to_series().diff().dt.days
    for name, mask in issues.items():
        hits = df.index[mask.fillna(False).to_numpy()]
        rows.append({"symbol": sym, "issue": name, "rows": len(hits),
                     "in_ml_training": sum((sym, d) in train_keys for d in hits),
                     "examples": ", ".join(str(d.date()) for d in hits[:3])})
    rows.append({"symbol": sym, "issue": "date_gap_>1d", "rows": int((gaps > 1).sum()), "in_ml_training": "",
                 "examples": ", ".join(f"{d.date()}(+{int(g)}d)" for d, g in gaps[gaps > 1].head(3).items())})
    rows.append({"symbol": sym, "issue": "duplicate_dates", "rows": int(df.index.duplicated().sum()),
                 "in_ml_training": "", "examples": ""})
q = pd.DataFrame(rows)
bad = q[q["rows"] > 0]
print(bad.to_string(index=False))

print("\nNon-USD files excluded by the quote filter (data.py:126-157):", excluded)
print("Any non-USD symbol in the ML dataset:", sorted(set(ds["symbol"]) & set(excluded)) or "none")

# Fresh-download comparison for Yahoo-sourced coins (stale partial candles).
import yfinance as yf  # noqa: E402

cmp_rows = []
for sym, tick in (("XRP", "XRP-USD"), ("HBAR(USD)", "HBAR-USD"), ("ONDO(USD)", "ONDO-USD")):
    fresh = yf.download(tick, start="2024-01-01", progress=False, auto_adjust=True)
    if isinstance(fresh.columns, pd.MultiIndex):
        fresh.columns = fresh.columns.get_level_values(0)
    fresh.index = pd.to_datetime(fresh.index).normalize()
    stored = loaded[sym]
    j = stored[["Close", "High", "Low"]].join(fresh[["Close", "High", "Low"]], rsuffix="_fresh", how="inner")
    j = j[j.index < j.index.max()]                                  # ignore today's still-forming candle
    for col in ("Close", "High", "Low"):
        diff = (j[col] / j[f"{col}_fresh"] - 1).abs()
        off = diff[diff > 0.005]
        cmp_rows.append({"symbol": sym, "field": col, "days_compared": len(j), "days_off_>0.5%": len(off),
                         "worst": f"{diff.max():.2%}", "dates": ", ".join(str(d.date()) for d in off.index[:6])})
cmp = pd.DataFrame(cmp_rows)
print("\nStored vs fresh Yahoo download (2024+):\n" + cmp.to_string(index=False))
c.save("a05_quality.csv", q)
c.save("a05_yahoo_compare.csv", cmp)
