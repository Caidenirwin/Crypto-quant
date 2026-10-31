"""
2. Purge / embargo check (ml_strategy.py:292-297).
The purge is horizon_days + 2 = 16 CALENDAR days, but a label spans 14 ROWS of that
coin's own history. If a coin has missing days, 14 rows span > 16 calendar days and
a training label can reach into the test block. Count it per fold.
Also: inside CalibratedModel (ml_strategy.py:258-265) the base model trains on the
first 80% of rows and the isotonic calibrator on the last 20% with NO gap between them.
"""
import pandas as pd

import _common as c

cfg = c.cfg_for()
ds = c.dataset()
lab = ds[ds["label"].notna()]
step = pd.Timedelta(days=cfg.ml.retrain_every_days)
purge = pd.Timedelta(days=cfg.ml.horizon_days + 2)

rows = []
t0, end = pd.Timestamp("2021-01-01"), ds["date"].max()
while t0 <= end:
    train = lab[lab["date"] < t0 - purge]
    if len(train) > 2000:
        leak = train[train["label_end"] >= t0]
        # inside the calibrated model: 80/20 split on date-sorted training rows
        cut = int(len(train) * 0.8)
        cal_start = train["date"].iloc[cut]
        base_into_cal = train.iloc[:cut]
        base_into_cal = base_into_cal[base_into_cal["label_end"] >= cal_start]
        rows.append({"fold_start": t0.date(), "train_rows": len(train),
                     "labels_reaching_test": len(leak),
                     "leak_symbols": ",".join(sorted(leak["symbol"].unique())),
                     "max_label_span_days": int(((train["label_end"] - train["date"]).dt.days).max()),
                     "base_labels_reaching_calibration": len(base_into_cal)})
    t0 += step

df = pd.DataFrame(rows)
print(df.to_string(index=False))
print(f"\nfolds: {len(df)} | folds with train labels reaching the test block: "
      f"{int((df.labels_reaching_test > 0).sum())} | total such rows: {int(df.labels_reaching_test.sum())}")
print(f"base-model rows whose label overlaps the calibration slice: {int(df.base_labels_reaching_calibration.sum())} "
      f"(avg {df.base_labels_reaching_calibration.mean():.0f}/fold)")
span = (lab["label_end"] - lab["date"]).dt.days
print("label span in calendar days (all labelled rows): " + span.describe()[["min", "50%", "max"]].to_string().replace("\n", " | "))
gappy = lab[span > cfg.ml.horizon_days + 2]
print("rows whose 14-row label spans > 16 calendar days:", len(gappy),
      gappy.groupby("symbol").size().sort_values(ascending=False).head(8).to_dict())
c.save("a02_purge.csv", df)
