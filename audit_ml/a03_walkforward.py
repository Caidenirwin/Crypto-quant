"""
3. Walk-forward integrity. Instrument ml_strategy.walk_forward (unchanged) with a
recording model factory: capture the exact training rows each fold sees, then verify
chronology, refit-per-fold, and that nothing is fit on test data.
"""
import inspect

import pandas as pd

import _common as c
import ml_strategy as ms

ds = c.dataset()
cfg = c.cfg_for()
fits = []


class Recorder(ms.CalibratedModel):
    def fit(self, X, y):
        fits.append({"rows": X.index.to_numpy(), "n": len(X)})
        return super().fit(X, y)


wf = ms.walk_forward(ds, cfg, model_factory=Recorder, features=ms.variant_features(c.CHAMPION))
step = pd.Timedelta(days=cfg.ml.retrain_every_days)
purge = pd.Timedelta(days=cfg.ml.horizon_days + 2)

t0, i, rows = pd.Timestamp("2021-01-01"), 0, []
while t0 <= ds["date"].max():
    test = wf[(wf["date"] >= t0) & (wf["date"] < t0 + step)]
    if i < len(fits) and test["p"].notna().any():
        tr = ds.loc[fits[i]["rows"]]
        rows.append({"fold": t0.date(), "train_first": tr["date"].min().date(), "train_last": tr["date"].max().date(),
                     "test_first": test["date"].min().date(), "test_last": test["date"].max().date(),
                     "gap_days": (test["date"].min() - tr["date"].max()).days,
                     "train_sorted": bool(tr["date"].is_monotonic_increasing),
                     "train_overlaps_test": bool((tr["date"] >= t0).any()),
                     "train_n": fits[i]["n"]})
        i += 1
    t0 += step
df = pd.DataFrame(rows)
print(df.to_string(index=False))
print(f"\nfits: {len(fits)} (one fresh model per fold) | any train row on/after its test start: "
      f"{bool(df.train_overlaps_test.any())} | min gap train->test: {df.gap_days.min()} days | "
      f"train rows date-sorted (needed by the 80/20 calibration split): {bool(df.train_sorted.all())}")
print("strictly chronological folds:", bool(pd.Series(pd.to_datetime(df.test_first)).is_monotonic_increasing))

src = inspect.getsource(ms)
for word in ("StandardScaler", "MinMaxScaler", "fit_transform", "SelectKBest", "PCA", "LabelEncoder"):
    print(f"'{word}' used anywhere in ml_strategy.py: {word in src}")
c.save("a03_walkforward.csv", df)
