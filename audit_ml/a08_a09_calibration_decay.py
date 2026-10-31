"""
8. Calibration: OOS predicted-probability deciles vs actual hit rate.
9. Decay: rolling 90-day AUC / avg R, monthly signal counts, and where the health gate's
   "last 180d: 531 trades, -0.70R" (ml_autoresearch.py:65,120-123) actually comes from.
Plus a date-clustered t-stat (trades overlap heavily in time).
"""
import numpy as np
import pandas as pd

import _common as c

wf = c.champion_oos()
oos = wf[wf["p"].notna() & wf["tradable"] & wf["label"].notna()].copy()
g = c.gated(wf)
g = g[g["label"].notna()]

# ---- 8. calibration by decile
oos["decile"] = pd.qcut(oos["p"].rank(method="first"), 10, labels=False) + 1
cal = oos.groupby("decile").agg(n=("p", "size"), predicted=("p", "mean"), actual=("label", "mean"),
                                avg_r=("r_net", "mean")).reset_index()
cal["gap_pts"] = (cal["actual"] - cal["predicted"]) * 100
print("8. CALIBRATION (all OOS tradable rows, deciles of predicted p)")
print(cal.round(3).to_string(index=False))
brier = float(((oos["p"] - oos["label"]) ** 2).mean())
brier_base = float(((oos["label"].mean() - oos["label"]) ** 2).mean())
print(f"Brier score {brier:.4f} vs base-rate forecast {brier_base:.4f} (lower is better)")
gate_rows = oos[oos["p"] >= (1 + 0.2) / (1 + c.RR)]
print(f"rows the EV gate trades (p >= {(1.2 / 3):.3f}): predicted {gate_rows['p'].mean():.3f} vs actual "
      f"{gate_rows['label'].mean():.3f} (n={len(gate_rows)})")

# ---- 9. decay
print("\n9. DECAY")
m = g.groupby(g["date"].dt.to_period("M")).agg(trades=("r_net", "size"), avg_r=("r_net", "mean")).reset_index()
m_all = oos.groupby(oos["date"].dt.to_period("M")).apply(lambda x: c.auc(x["label"], x["p"])).rename("auc").reset_index()
mm = m_all.merge(m, on="date", how="left").fillna({"trades": 0})
print("monthly (2025-01+): OOS AUC, gated trades, gated avg R")
print(mm[mm["date"] >= pd.Period("2025-01")].round(3).to_string(index=False))

roll = []
for end in pd.date_range("2021-06-30", oos["date"].max(), freq="30D"):
    w = oos[(oos["date"] > end - pd.Timedelta(days=90)) & (oos["date"] <= end)]
    wg = g[(g["date"] > end - pd.Timedelta(days=90)) & (g["date"] <= end)]
    roll.append({"window_end": end.date(), "auc_90d": c.auc(w["label"], w["p"]),
                 "gated_trades_90d": len(wg), "gated_avg_r_90d": wg["r_net"].mean() if len(wg) else np.nan})
roll = pd.DataFrame(roll)
yr = oos.groupby(oos["date"].dt.year).apply(lambda x: c.auc(x["label"], x["p"]))
print("\nAUC by year:", {int(k): round(v, 3) for k, v in yr.items()})
slope = np.polyfit(np.arange(len(roll)), roll["auc_90d"].fillna(roll["auc_90d"].mean()), 1)[0] * 12
print(f"rolling-90d AUC trend: {slope:+.4f} per year | first-year mean {roll.auc_90d.head(12).mean():.3f} "
      f"vs last-year mean {roll.auc_90d.tail(12).mean():.3f}")

recent_cut = g["date"].max() - pd.Timedelta(days=180)
rec = g[g["date"] > recent_cut]
by_day = rec.groupby("date").size()
print(f"\nhealth-gate window: signals after {recent_cut.date()} -> {len(rec)} trades, avg {rec['r_net'].mean():+.3f}R")
print(f"  those trades fall on {by_day.size} distinct days, {by_day.index.min().date()} .. {by_day.index.max().date()}; "
      f"busiest day {by_day.max()} signals")
print(f"  coins involved: {rec['symbol'].nunique()} | last gated signal of all: {g['date'].max().date()} | "
      f"last labelled OOS row: {oos['date'].max().date()}")

# Clustered significance: average the R of all signals taken on the same week, t-test across weeks.
wk = g.groupby(g["date"].dt.to_period("W"))["r_net"].mean()
print(f"\nclustered t-stat (weekly means, n={len(wk)} weeks): {wk.mean() / (wk.std() / np.sqrt(len(wk))):.2f} "
      f"(naive per-trade t was 1.72)")
c.save("a08_calibration.csv", cal)
c.save("a09_monthly.csv", mm.astype(str))
c.save("a09_rolling90.csv", roll)
