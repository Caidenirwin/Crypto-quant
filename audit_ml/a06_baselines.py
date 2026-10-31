"""
6. Sanity baselines for the champion (calibrated HGB, baseline features, RR 2.0, EV gate 0.2R).
 (a) same pipeline trained on SHUFFLED labels (labels permuted inside each training fold)
 (b) random entries: same number of trades drawn at random from the same OOS rows
 (c) simple trend filter (close > MA50 and BTC > MA200), no model
"""
import numpy as np
import pandas as pd

import _common as c
import ml_strategy as ms

cfg = c.cfg_for()
ds = c.dataset()
feats = ms.variant_features(c.CHAMPION)


def summarize(name, wf):
    oos = wf[wf["p"].notna() & wf["tradable"] & wf["label"].notna()]
    g = oos[ms.expected_r(oos["p"].to_numpy(), c.RR) >= cfg.ml.min_expected_r]
    return {"model": name, "oos_auc": round(c.auc(oos["label"], oos["p"]), 4), "gated_trades": len(g),
            "gated_avg_r": round(g["r_net"].mean(), 4) if len(g) else float("nan"),
            "gated_win_rate": round((g["r_net"] > 0).mean(), 4) if len(g) else float("nan")}


rows = [summarize("CHAMPION (real labels)", c.champion_oos())]
for seed in (1, 2, 3):
    rng = np.random.default_rng(seed)

    class Shuffled(ms.CalibratedModel):
        def fit(self, X, y):
            return super().fit(X, pd.Series(rng.permutation(y.to_numpy()), index=y.index))

    rows.append(summarize(f"shuffled labels (seed {seed})", ms.walk_forward(ds, cfg, model_factory=Shuffled, features=feats)))

champ = c.gated(c.champion_oos())
champ = champ[champ["label"].notna()]
pool = c.champion_oos()
pool = pool[pool["p"].notna() & pool["tradable"] & pool["label"].notna()]
rng = np.random.default_rng(7)
draws = np.array([pool["r_net"].sample(len(champ), random_state=int(rng.integers(1e9))).mean() for _ in range(2000)])
pct = float((draws < champ["r_net"].mean()).mean())
rows.append({"model": "random entries (2000 draws, same n)", "oos_auc": 0.5, "gated_trades": len(champ),
             "gated_avg_r": round(float(draws.mean()), 4), "gated_win_rate": round(float((pool["r_net"] > 0).mean()), 4),
             "note": f"champion avgR beats {pct:.1%} of random draws; 95% band "
                     f"[{np.percentile(draws, 2.5):+.3f}, {np.percentile(draws, 97.5):+.3f}]"})
trend = pool[(pool["dist_ma50"] > 0) & (pool["btc_dist_ma200"] > 0)]
rows.append({"model": "trend filter (no model)", "oos_auc": round(c.auc(pool["label"], (pool["dist_ma50"] > 0) & (pool["btc_dist_ma200"] > 0)), 4),
             "gated_trades": len(trend), "gated_avg_r": round(trend["r_net"].mean(), 4),
             "gated_win_rate": round((trend["r_net"] > 0).mean(), 4)})

# Is the champion's avgR distinguishable from zero / from random? (trade-level t-stat, overlapping trades => optimistic)
r = champ["r_net"]
rows.append({"model": "champion t-stat vs 0 (naive, trades overlap)", "oos_auc": "", "gated_trades": len(r),
             "gated_avg_r": round(r.mean(), 4), "gated_win_rate": "",
             "note": f"t = {r.mean() / (r.std() / np.sqrt(len(r))):.2f}"})
out = pd.DataFrame(rows)
print(out.to_string(index=False))
c.save("a06_baselines.csv", out)
