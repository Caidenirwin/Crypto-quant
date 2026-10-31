"""
11. Live vs training feature parity.
Training: ml_strategy.build_dataset(with_labels=True) (ml_autoresearch / train_final_model).
Live:     ml_live.generate_entries -> build_dataset(with_labels=False), then each coin's
          newest row within 1 day of the latest date (ml_live.py:226-237).
Diff the exact vectors fed to the saved model, check column order vs the model, and the
cross-section size behind breadth / xs_rank on the live day vs in training.
"""
import joblib
import numpy as np
import pandas as pd

import _common as c
import ml_strategy as ms

cfg = c.cfg_for()
panel, ref = c.panel_ref()
bundle = joblib.load(c.ROOT / cfg.ml.model_path)
feats = bundle["features"]

train_ds = ms.build_dataset(panel, ref, cfg, with_labels=True)
live_ds = ms.build_dataset(panel, ref, cfg, with_labels=False)

# exactly what ml_live does
latest = live_ds["date"].max()
newest = live_ds[live_ds["tradable"]].sort_values("date").groupby("symbol").tail(1)
live_rows = newest[newest["date"] >= latest - pd.Timedelta(days=1)].dropna(subset=feats + ["atr"])

key = ["symbol", "date"]
tr = train_ds.set_index(key).loc[live_rows.set_index(key).index, feats]
lv = live_rows.set_index(key)[feats]
same = np.isclose(tr.to_numpy(float), lv.to_numpy(float), equal_nan=True, rtol=1e-12, atol=0)
print(f"coins scored live: {len(lv)} | values compared: {same.size} | mismatches: {int((~same).sum())}")

hgb = bundle["model"].base
names = list(getattr(hgb, "feature_names_in_", []))
print("model feature names == bundle['features'] (same order):", names == feats)
print("bundle features == variant_features(champion):", feats == ms.variant_features(bundle["variant"]))

# rows training keeps vs rows live keeps
train_nan_rows = train_ds[train_ds["label"].notna()][feats].isna().any(axis=1).mean()
print(f"training rows with >=1 NaN feature (kept; HGB handles NaN): {train_nan_rows:.1%} | "
      f"live drops any coin with a NaN feature (ml_live.py:236)")

# cross-section size behind breadth / xs_rank
size_train = train_ds.groupby("date")["symbol"].transform("size")
pre = ms.build_dataset.__wrapped__ if hasattr(ms.build_dataset, "__wrapped__") else None
members = {d: sum(1 for df in panel.values() if d in df.index) for d in sorted(set(live_rows["date"]))}
recent_typical = pd.Series({d: sum(1 for df in panel.values() if d in df.index)
                            for d in pd.date_range(latest - pd.Timedelta(days=60), latest - pd.Timedelta(days=3))})
print(f"\ncoins in the cross-section on the live scoring dates: {members}")
print(f"typical cross-section in the prior 2 months: {int(recent_typical.median())} coins")
for d, n in members.items():
    rows = live_rows[live_rows["date"] == d]
    print(f"  {d.date()}: {len(rows)} live rows scored; breadth_ma50 values {sorted(rows['breadth_ma50'].round(3).unique())}")

src_dates = live_rows.groupby("date")["symbol"].apply(lambda s: ", ".join(x.split('(')[0] for x in s)).to_dict()
print("live rows by date:", {str(k.date()): v for k, v in src_dates.items()})
c.save("a11_live_parity.json", {"mismatches": int((~same).sum()), "values": int(same.size),
                                 "cross_section_live": {str(k.date()): v for k, v in members.items()},
                                 "cross_section_typical": int(recent_typical.median())})
