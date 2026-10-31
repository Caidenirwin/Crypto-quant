"""
4. Survivorship. Which coins exist at each date, when they were added to the project,
and which coins feed the cross-sectional features (ml_strategy.py:211-217) over time.
"""
import os
from datetime import datetime

import pandas as pd

import _common as c
from main import build_symbol_file_map

cfg = c.cfg_for()
panel, _ = c.panel_ref()
files = build_symbol_file_map(cfg, c.ROOT)
ds = c.dataset()

norm = {}
for raw, path in files.items():
    from data import DataManager
    norm[DataManager(cfg).normalize_symbol(raw)] = path

rows = []
for sym, df in sorted(panel.items(), key=lambda kv: kv[1].index.min()):
    path = norm.get(sym, "")
    added = datetime.fromtimestamp(os.path.getctime(path)).date() if path and os.path.exists(path) else None
    rows.append({"symbol": sym, "first_bar": df.index.min().date(), "last_bar": df.index.max().date(),
                 "bars": len(df), "file_created": added, "tradable": sym not in cfg.ml.no_trade_symbols,
                 "meme": cfg.is_meme(sym), "still_trading_in_data": (df.index.max() >= ds["date"].max() - pd.Timedelta(days=5))})
uni = pd.DataFrame(rows)
print(uni.to_string(index=False))

# Cross-section membership (rows that feed breadth / xs_rank on each date, BEFORE the 200-bar filter)
from ml_strategy import symbol_features  # noqa: E402
counts = []
for y in range(2018, 2027):
    d = pd.Timestamp(f"{y}-06-30") if y < 2026 else pd.Timestamp("2026-10-15")
    members = sorted(s for s, df in panel.items() if d in df.index)
    counts.append({"date": d.date(), "coins_in_cross_section": len(members),
                   "of_which_added_to_project_in_2026": sum(1 for s in members
                       if (u := uni.set_index("symbol").loc[s, "file_created"]) is not None and u.year == 2026),
                   "members": ",".join(m.split("(")[0] for m in members)})
cs = pd.DataFrame(counts)
print("\n" + cs.drop(columns="members").to_string(index=False))
dead = uni[~uni.still_trading_in_data]
print("\ncoins in the panel whose data has ENDED (non-survivors):", dead.symbol.tolist())
c.save("a04_universe.csv", uni)
c.save("a04_cross_section.csv", cs)
