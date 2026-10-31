"""
7. Overfitting.
 (a) Count configurations evaluated on the same 2021-2026 OOS period (from the saved
     outputs + the trials that produced them).
 (b) Deflated Sharpe ratio (Bailey & Lopez de Prado 2014) for the champion's portfolio.
 (c) Pseudo-holdout: choose the best of 12 configs using ONLY signals before 2026-05-01,
     then report May-Oct 2026 alone. NOTE: no truly untouched holdout exists - every
     date through Oct 2026 was already looked at during development.
"""
import json
import math

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

import _common as c
import ml_strategy as ms

# ---------------------------------------------------------------- (a) trial count
TRIALS = [
    # (source, what, count)
    ("ml_research.py:136-185 evaluate()", "raw HGB, RR {1.5,2.0} x EV gates {0,0.1,0.2,0.3} + take-all + trend", 2 * 6),
    ("ml_research.py:188-221 evaluate_variants()", "calibrated A / A+trend at RR {1.5,2.0}", 4),
    ("ablation (scratch)", "orig+F&G, +new coins (no F&G)", 2),
    ("ml_autoresearch run #1", "new_coins, F&G+new_coins", 2),
    ("ml_autoresearch run #2 (VARIANTS ml_strategy.py:310-315)", "chart_patterns, F&G+chart_patterns", 2),
    ("outputs/rr_sweep.csv", "RR {1.0,1.25,1.5,2.0} x 2 variants (new combos only)", 6),
    ("outputs/rr_sweep_high.csv", "RR {2.5,3.0} x 2 variants", 4),
    ("outputs/short_walk_forward.csv", "short side x 2 variants", 2),
]
n_trials = sum(t[2] for t in TRIALS)

# Annualised Sharpe of every trial we have a portfolio number for.
sharpes = []
rr = pd.read_csv(c.ROOT / "outputs" / "rr_sweep.csv"); sharpes += rr["sharpe21"].tolist()
rh = pd.read_csv(c.ROOT / "outputs" / "rr_sweep_high.csv"); sharpes += rh["sharpe21"].tolist()
sh = pd.read_csv(c.ROOT / "outputs" / "short_walk_forward.csv"); sharpes += sh["sharpe21"].tolist()
for v in json.loads((c.ROOT / "outputs" / "ml_walk_forward_variants.json").read_text()):
    sharpes += [x["sharpe"] for k, x in v["variants"].items() if "2021" in k]
for v in json.loads((c.ROOT / "outputs" / "ml_walk_forward.json").read_text()):
    sharpes += [x["ml"]["sharpe"] for k, x in v["portfolio"].items() if k.startswith("2021")]
sharpes = np.array(sharpes, float)


# ---------------------------------------------------------------- (b) champion daily returns
def simulate_curve(sig, panel, cfg, start):
    """Copy of ml_research.simulate (long side) that returns the daily equity curve."""
    mc = cfg.ml
    fee = mc.roundtrip_cost / 2.0
    start_ts = pd.Timestamp(start)
    dates = sorted({d for df in panel.values() for d in df.index if d >= start_ts})
    by_date = {d: g for d, g in sig[sig["date"] >= start_ts].groupby("date")}
    cash, open_pos, pending, curve = 1.0, {}, [], []
    for d in dates:
        for p in pending:
            df = panel[p["symbol"]]
            if d not in df.index or p["symbol"] in open_pos or len(open_pos) >= mc.max_open_trades:
                continue
            entry = float(df.at[d, "Open"]); risk = mc.stop_atr_mult * p["atr"]
            eq = cash + sum(x["qty"] * x["last"] for x in open_pos.values())
            qty = min(eq * mc.risk_per_trade / risk, eq * mc.max_position_weight / entry, cash / (entry * (1 + fee)))
            if qty <= 0:
                continue
            cash -= qty * entry * (1 + fee)
            open_pos[p["symbol"]] = {"qty": qty, "entry": entry, "sl": entry - risk,
                                      "tp": entry + mc.reward_risk * risk, "bars": 0, "last": entry}
        pending = []
        for sym in list(open_pos):
            df = panel[sym]
            if d not in df.index:
                continue
            pos = open_pos[sym]
            o, h, l, cl = (float(df.at[d, k]) for k in ("Open", "High", "Low", "Close"))
            pos["bars"] += 1
            px = (o if pos["bars"] > 1 and o <= pos["sl"] else pos["sl"] if l <= pos["sl"]
                  else pos["tp"] if h >= pos["tp"] else cl if pos["bars"] >= mc.horizon_days else None)
            pos["last"] = cl
            if px is not None:
                cash += pos["qty"] * px * (1 - fee)
                del open_pos[sym]
        curve.append((d, cash + sum(x["qty"] * x["last"] for x in open_pos.values())))
        g = by_date.get(d)
        if g is not None:
            g = g[~g["symbol"].isin(open_pos)].sort_values("ev", ascending=False)
            pending = g.head(max(0, mc.max_open_trades - len(open_pos))).to_dict("records")
    return pd.Series(dict(curve)).sort_index()


panel, _ = c.panel_ref()
cfg = c.cfg_for()
eq = simulate_curve(c.gated(c.champion_oos()), panel, cfg, "2021-01-01")
ret = eq.pct_change().dropna()
sr_d = ret.mean() / ret.std()
T, g3, g4 = len(ret), skew(ret), kurtosis(ret, fisher=False)
var_sr_ann = np.var(sharpes, ddof=1)
var_sr_d = var_sr_ann / 365.0
emc = 0.5772156649
sr0 = math.sqrt(var_sr_d) * ((1 - emc) * norm.ppf(1 - 1 / n_trials) + emc * norm.ppf(1 - 1 / (n_trials * math.e)))
dsr = norm.cdf((sr_d - sr0) * math.sqrt(T - 1) / math.sqrt(1 - g3 * sr_d + (g4 - 1) / 4 * sr_d ** 2))
psr0 = norm.cdf(sr_d * math.sqrt(T - 1) / math.sqrt(1 - g3 * sr_d + (g4 - 1) / 4 * sr_d ** 2))

print(f"(a) configurations evaluated on 2021-2026: {n_trials}")
for t in TRIALS:
    print(f"    {t[2]:>3}  {t[0]}: {t[1]}")
print(f"    trial Sharpes available: {len(sharpes)} (annualised; mean {sharpes.mean():.2f}, sd {sharpes.std(ddof=1):.2f}, max {sharpes.max():.2f})")
print(f"(b) champion portfolio 2021+: annualised Sharpe {sr_d * math.sqrt(365):.2f}, T={T} days, skew {g3:.2f}, kurtosis {g4:.1f}")
print(f"    expected max Sharpe of {n_trials} zero-skill trials: {sr0 * math.sqrt(365):.2f} (annualised)")
print(f"    Probabilistic Sharpe (vs 0): {psr0:.3f} | DEFLATED Sharpe ratio: {dsr:.3f}  (>0.95 = significant)")

# ---------------------------------------------------------------- (c) pseudo-holdout
CUT = pd.Timestamp("2026-05-01")
rows = []
for rrv in (1.5, 2.0, 2.5):
    cfg_r = c.cfg_for(rrv)
    ds = c.dataset(rrv)
    for v in ms.VARIANTS:
        wf = c.champion_oos(rrv, v)
        g = wf[wf["p"].notna() & wf["tradable"] & wf["label"].notna()]
        g = g[ms.expected_r(g["p"].to_numpy(), rrv) >= cfg_r.ml.min_expected_r]
        sel = g[g["label_end"] < CUT]                       # outcome fully known before the cut
        ho = g[g["date"] >= CUT]
        rows.append({"rr": rrv, "variant": v, "sel_trades": len(sel), "sel_avg_r": sel["r_net"].mean(),
                     "holdout_trades": len(ho), "holdout_avg_r": ho["r_net"].mean(),
                     "holdout_win": (ho["r_net"] > 0).mean() if len(ho) else float("nan")})
tab = pd.DataFrame(rows)
eligible = tab[tab["sel_trades"] >= 300]
pick = eligible.loc[eligible["sel_avg_r"].idxmax()]
base = c.dataset(2.0)
pool = base[(base["date"] >= CUT) & base["tradable"] & base["label"].notna()]
trend = pool[(pool["dist_ma50"] > 0) & (pool["btc_dist_ma200"] > 0)]
print("\n(c) 12 configs: selection = signals whose outcome resolved before 2026-05-01; holdout = signals from 2026-05-01")
print(tab.round(3).to_string(index=False))
print(f"\n    picked on pre-May data: RR {pick.rr} / {pick.variant} (selection avg {pick.sel_avg_r:+.3f}R)")
print(f"    its HOLDOUT (May-Oct 2026): {int(pick.holdout_trades)} trades, avg {pick.holdout_avg_r:+.3f}R, win {pick.holdout_win:.1%}")
cur = tab[(tab.rr == 2.0) & (tab.variant == "baseline")].iloc[0]
print(f"    current champion (RR 2 / baseline) holdout: {int(cur.holdout_trades)} trades, avg {cur.holdout_avg_r:+.3f}R")
print(f"    take-every-setup holdout: {len(pool)} setups, avg {pool['r_net'].mean():+.3f}R | "
      f"trend filter holdout: {len(trend)} trades, avg {trend['r_net'].mean():+.3f}R")
c.save("a07_holdout.csv", tab)
c.save("a07_overfitting.json", {"n_trials": n_trials, "trial_sharpes": sharpes.tolist(),
                                "champion_sharpe_ann": sr_d * math.sqrt(365), "dsr": dsr, "psr": psr0,
                                "expected_max_null_sharpe_ann": sr0 * math.sqrt(365),
                                "picked": pick.to_dict()})
