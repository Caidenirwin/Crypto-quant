"""
10. Cost sensitivity. The classifier predicts "take-profit first" (cost-independent), so the
same OOS signals are re-priced: r_net(cost) = gross_R - cost * entry / risk
(ml_strategy.py:185). Portfolio re-simulated with ml_research.simulate (unchanged).
"""
import io
from contextlib import redirect_stdout

import pandas as pd

import _common as c
from ml_research import simulate

panel, _ = c.panel_ref()
wf = c.champion_oos()
g = c.gated(wf)
lab = g[g["label"].notna()].copy()

# entry = next bar's open, risk = 1.5 * ATR at signal (same as the label code)
nxt = {s: df["Open"].astype(float).shift(-1) for s, df in panel.items()}
lab["entry"] = [nxt[s].get(d) for s, d in zip(lab["symbol"], lab["date"])]
lab["risk"] = 1.5 * lab["atr"]
lab["cost_r_per_1pct"] = 0.01 * lab["entry"] / lab["risk"]
base_cost = c.cfg_for().ml.roundtrip_cost
gross = lab["r_net"] + base_cost * lab["entry"] / lab["risk"]
rows = []
for cost in (0.0, 0.003, 0.006):
    r = gross - cost * lab["entry"] / lab["risk"]
    cfg = c.cfg_for(cost=cost)
    with redirect_stdout(io.StringIO()):
        p21 = simulate(g, panel, cfg, "2021-01-01")
        p24 = simulate(g, panel, cfg, "2024-01-18")
    rows.append({"roundtrip_cost": f"{cost:.1%}", "trades": len(r), "avg_r": round(r.mean(), 4),
                 "win_rate": round((r > 0).mean(), 4), "profit_factor": round(r[r > 0].sum() / -r[r < 0].sum(), 3),
                 "portfolio_cagr_2021": round(p21["cagr"], 4), "sharpe_2021": round(p21["sharpe"], 3),
                 "max_dd_2021": round(p21["max_dd"], 4), "portfolio_cagr_2024": round(p24["cagr"], 4)})
out = pd.DataFrame(rows)
print(out.to_string(index=False))
print(f"\nmedian cost per trade in R at 0.3% round trip: {(lab['cost_r_per_1pct'] * 0.3).median():.3f}R "
      f"(stop distance median {(lab['risk'] / lab['entry']).median():.1%} of price)")
c.save("a10_costs.csv", out)
