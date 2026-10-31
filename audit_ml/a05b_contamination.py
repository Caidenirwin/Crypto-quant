"""
5b. Bad rows excluded by the 200-bar rule can still sit inside the rolling windows
(up to 365 bars for range_pos_365) used to compute LATER training rows' features.
Count training rows whose lookback window contains a zero / placeholder price.
"""
import pandas as pd

import _common as c

panel, _ = c.panel_ref()
ds = c.dataset()
train = ds[ds["label"].notna()]
out = []
for sym, df in panel.items():
    o, h, l, cl = (df[k].astype(float) for k in ("Open", "High", "Low", "Close"))
    bad = ((cl <= 0) | (o <= 0) | (l <= 0) | ((o == h) & (h == l) & (l == cl))).astype(int)
    if not bad.any():
        continue
    hit200 = bad.rolling(200, min_periods=1).max().astype(bool)     # MA200 / dist_ma200 window
    hit365 = bad.rolling(365, min_periods=1).max().astype(bool)     # range_pos_365 window
    tr = train[train["symbol"] == sym].set_index("date")
    out.append({"symbol": sym, "bad_rows": int(bad.sum()), "training_rows": len(tr),
                "train_rows_with_bad_bar_in_200d_window": int(hit200.reindex(tr.index).sum()),
                "train_rows_with_bad_bar_in_365d_window": int(hit365.reindex(tr.index).sum()),
                "tradable": sym not in c.cfg_for().ml.no_trade_symbols})
df = pd.DataFrame(out)
print(df.to_string(index=False))
c.save("a05b_contamination.csv", df)
