"""
11b. Reproduce the common live situation where Yahoo-sourced coins lag Binance-sourced
coins by one day (seen on 2026-10-30). On the newest date only the Binance-sourced coins
exist, so breadth_* and xs_rank_* (ml_strategy.py:211-217) for those rows are computed
over a 4-6 coin cross-section instead of ~25 - a value range training never saw.
"""
import pandas as pd

import _common as c
import ml_strategy as ms
import update_data

cfg = c.cfg_for()
panel, ref = c.panel_ref()
binance_bases = {stem.split("(")[0].upper() for stem, t in update_data.FILE_TICKER_MAP.items() if t.startswith("binance:")}
last = max(df.index.max() for df in panel.values())
lagged = {s: (df if s.split("(")[0].upper() in binance_bases else df[df.index < last]) for s, df in panel.items()}

honest = ms.build_dataset(panel, ref, cfg, with_labels=False)
skewed = ms.build_dataset(lagged, ref, cfg, with_labels=False)
xs = ["breadth_ma50", "breadth_ma200", "xs_rank_ret7", "xs_rank_ret30", "xs_rank_vol30"]
a = honest[honest["date"] == last].set_index("symbol")[xs]
b = skewed[skewed["date"] == last].set_index("symbol")[xs]
print(f"Binance-sourced coins: {sorted(binance_bases)}")
print(f"cross-section on {last.date()}: full={len(a)} coins, lagged-Yahoo scenario={len(b)} coins")
print("\nsame coins, same day - cross-sectional features (full cross-section vs what live would see):")
print(a.loc[b.index].round(3).join(b.round(3), rsuffix="_LIVE_LAGGED").T.to_string())

bundle = __import__("joblib").load(c.ROOT / cfg.ml.model_path)
f = bundle["features"]
pa = bundle["model"].predict_proba(honest[honest["date"] == last].set_index("symbol").loc[b.index, f])[:, 1]
pb = bundle["model"].predict_proba(skewed[skewed["date"] == last].set_index("symbol")[f])[:, 1]
print("\nmodel p, full vs lagged cross-section:", {s: (round(x, 3), round(y, 3)) for s, x, y in zip(b.index, pa, pb)})
