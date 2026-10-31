"""
1. Label leakage / look-ahead shift test.
For several cut-off dates T: delete every row after T (all coins, BTC/ETH reference,
Fear & Greed), rebuild features, and require the rows at T to be identical to the
full-history build. Also checks that no label column is a feature.
"""
import numpy as np
import pandas as pd

import _common as c
import news_feed
import ml_strategy as ms

full_fng = news_feed.load_fear_greed()
panel, ref = c.panel_ref()
cfg = c.cfg_for()
full = ms.build_dataset(panel, ref, cfg, with_labels=True)
LABEL_COLS = {"label", "r_net", "exit_offset", "label_end"}

out = {"label_columns_in_FEATURES": sorted(LABEL_COLS & set(ms.FEATURES)), "cuts": []}
last = full["date"].max()
cuts = [pd.Timestamp("2021-06-15"), pd.Timestamp("2023-02-01"), pd.Timestamp("2024-11-20"),
        pd.Timestamp("2026-05-01"), last - pd.Timedelta(days=20)]
for T in cuts:
    p_cut = {s: df[df.index <= T] for s, df in panel.items() if (df.index <= T).sum() > 0}
    r_cut = {k: v[v.index <= T] for k, v in ref.items()}
    news_feed.load_fear_greed = lambda: full_fng[full_fng.index <= T]          # truncate sentiment too
    cut = ms.build_dataset(p_cut, r_cut, cfg, with_labels=False)
    news_feed.load_fear_greed = lambda: full_fng
    a = full[full["date"] == T].set_index("symbol")[ms.FEATURES].sort_index()
    b = cut[cut["date"] == T].set_index("symbol")[ms.FEATURES].sort_index()
    common = a.index.intersection(b.index)
    a, b = a.loc[common].astype(float), b.loc[common].astype(float)
    same = np.isclose(a.to_numpy(), b.to_numpy(), rtol=1e-9, atol=1e-12, equal_nan=True)
    bad_cols = sorted({ms.FEATURES[j] for j in np.where(~same)[1]})
    out["cuts"].append({"cut": str(T.date()), "coins_compared": len(common),
                        "values_compared": int(same.size), "mismatches": int((~same).sum()),
                        "mismatched_features": bad_cols})
    print(f"cut {T.date()}: {len(common)} coins x {len(ms.FEATURES)} features -> "
          f"{int((~same).sum())} mismatches {bad_cols or ''}")

print("label columns inside FEATURES:", out["label_columns_in_FEATURES"] or "none")
c.save("a01_leakage.json", out)
