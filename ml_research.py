"""
Walk-forward evaluation of the ML bracket sleeve.  Run: python ml_research.py

Everything reported here is OUT-OF-SAMPLE: each prediction comes from a model
trained only on data that ended before the prediction date (minus a purge gap).
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config
from ml_strategy import (
    FEATURES, build_dataset, expected_r, load_panel, load_reference, walk_forward,
)

ROOT = Path(__file__).resolve().parent


def trade_stats(r: pd.Series) -> dict:
    r = r.dropna()
    if r.empty:
        return {"trades": 0}
    wins, losses = r[r > 0].sum(), -r[r < 0].sum()
    return {
        "trades": int(len(r)),
        "win_rate": float((r > 0).mean()),
        "avg_r": float(r.mean()),
        "profit_factor": float(wins / losses) if losses > 0 else float("inf"),
    }


def simulate(sig: pd.DataFrame, panel: dict, cfg: Config, start: str, end: str | None = None,
             side: str = "long") -> dict:
    """
    Daily portfolio simulation. Signals at close d -> entry at the coin's next open.
    Stop/take-profit checked on each bar (stop first), gap-through fills at the open,
    time exit at close after horizon bars. 1% equity risked per trade.
    side="short": 1x short (entry notional held as collateral, gains as price falls).
    """
    mc = cfg.ml
    fee = mc.roundtrip_cost / 2.0
    s = 1.0 if side == "long" else -1.0

    def value(pos, px):           # what the position is worth if closed at px (before fees)
        return pos["qty"] * (pos["entry"] + s * (px - pos["entry"]))
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) if end else max(df.index.max() for df in panel.values())
    dates = sorted({d for df in panel.values() for d in df.index if start_ts <= d <= end_ts})
    by_date = {d: g for d, g in sig[(sig["date"] >= start_ts) & (sig["date"] <= end_ts)].groupby("date")}

    cash, open_pos, pending, curve, trades = 1.0, {}, [], [], []
    for d in dates:
        # 1) fill entries signalled yesterday at today's open
        for p in pending:
            df = panel[p["symbol"]]
            if d not in df.index or p["symbol"] in open_pos or len(open_pos) >= mc.max_open_trades:
                continue
            entry = float(df.at[d, "Open"])
            risk = mc.stop_atr_mult * p["atr"]
            equity = cash + sum(value(x, x["last"]) for x in open_pos.values())
            qty = equity * mc.risk_per_trade / risk
            qty = min(qty, equity * mc.max_position_weight / entry, cash / (entry * (1 + fee)))
            if qty <= 0:
                continue
            cash -= qty * entry * (1 + fee)          # long: cost; short: 1x collateral
            open_pos[p["symbol"]] = {"qty": qty, "entry": entry, "sl": entry - s * risk,
                                      "tp": entry + s * mc.reward_risk * risk, "risk": risk,
                                      "bars": 0, "last": entry, "entry_date": d}
        pending = []

        # 2) exits on today's bar
        for sym in list(open_pos):
            df = panel[sym]
            if d not in df.index:
                continue
            pos = open_pos[sym]
            o, h, l, c = (float(df.at[d, k]) for k in ("Open", "High", "Low", "Close"))
            adverse, favorable = (l, h) if s > 0 else (h, l)
            pos["bars"] += 1
            px = None
            if pos["bars"] > 1 and s * (o - pos["sl"]) <= 0:
                px = o
            elif s * (adverse - pos["sl"]) <= 0:
                px = pos["sl"]
            elif s * (favorable - pos["tp"]) >= 0:
                px = pos["tp"]
            elif pos["bars"] >= mc.horizon_days:
                px = c
            pos["last"] = c
            if px is not None:
                cash += value(pos, px) - pos["qty"] * px * fee
                trades.append(s * (px - pos["entry"]) / pos["risk"] - mc.roundtrip_cost * pos["entry"] / pos["risk"])
                del open_pos[sym]

        equity = cash + sum(value(x, x["last"]) for x in open_pos.values())
        curve.append((d, equity))

        # 3) new signals at today's close, best expected value first
        g = by_date.get(d)
        if g is not None:
            g = g[~g["symbol"].isin(open_pos)].sort_values("ev", ascending=False)
            slots = mc.max_open_trades - len(open_pos)
            pending = g.head(max(0, slots)).to_dict("records")

    eq = pd.Series(dict(curve)).sort_index()
    rets = eq.pct_change().dropna()
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    return {
        "start": str(eq.index[0].date()), "end": str(eq.index[-1].date()),
        "total_return": float(eq.iloc[-1] / eq.iloc[0] - 1),
        "cagr": float((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1),
        "sharpe": float(rets.mean() / rets.std() * np.sqrt(365)) if rets.std() > 0 else 0.0,
        "max_dd": float((eq / eq.cummax() - 1).min()),
        **trade_stats(pd.Series(trades)),
    }


def fmt(name: str, s: dict) -> str:
    if s.get("trades", 0) == 0:
        return f"  {name:34s} no trades"
    pf = s["profit_factor"]
    base = (f"  {name:34s} trades={s['trades']:5d} win={s['win_rate']:5.1%} "
            f"avgR={s['avg_r']:+.3f} PF={pf:4.2f}")
    if "cagr" in s:
        base += f" | CAGR={s['cagr']:+7.1%} Sharpe={s['sharpe']:5.2f} MaxDD={s['max_dd']:6.1%}"
    return base


def evaluate(rr: float) -> dict:
    cfg = Config()
    cfg.ml.reward_risk = rr
    panel = load_panel(cfg)
    ds = build_dataset(panel, load_reference(cfg), cfg)
    from ml_strategy import make_model

    wf = walk_forward(ds, cfg, model_factory=make_model)   # raw (uncalibrated) model
    oos = wf[wf["p"].notna() & wf["tradable"]].copy()
    oos["ev"] = expected_r(oos["p"].to_numpy(), rr)
    lab = oos[oos["label"].notna()]

    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(lab["label"].astype(int), lab["p"])
    breakeven = 1.0 / (1.0 + rr)
    print(f"\n=== R:R {rr:.1f}  (break-even win rate {breakeven:.1%} before costs) ===")
    print(f"  rows={len(ds):,}  out-of-sample tradable rows={len(oos):,}  "
          f"OOS window {oos['date'].min().date()} -> {oos['date'].max().date()}  AUC={auc:.3f}")

    print("  Per-trade outcomes (every qualifying signal, before position limits):")
    print(fmt("take EVERY setup (no filter)", trade_stats(lab["r_net"])))
    trend = lab[(lab["dist_ma50"] > 0) & (lab["btc_dist_ma200"] > 0)]
    print(fmt("simple trend filter (no ML)", trade_stats(trend["r_net"])))
    for gate in (0.0, 0.1, 0.2, 0.3):
        sel = lab[lab["ev"] >= gate]
        print(fmt(f"ML: expected R >= {gate:.1f}", trade_stats(sel["r_net"])))

    # Calibration: does a higher predicted p actually win more often?
    lab = lab.assign(bucket=pd.qcut(lab["p"], 5, labels=False, duplicates="drop"))
    cal = lab.groupby("bucket").agg(pred=("p", "mean"), actual=("label", "mean"), avg_r=("r_net", "mean"), n=("p", "size"))
    print("  Calibration by predicted-probability quintile:")
    for _, row in cal.iterrows():
        print(f"    predicted {row.pred:5.1%} -> actual win {row.actual:5.1%}  avgR={row.avg_r:+.3f}  n={int(row.n)}")

    gate = cfg.ml.min_expected_r
    ml_sig = oos[oos["ev"] >= gate]
    trend_sig = oos[(oos["dist_ma50"] > 0) & (oos["btc_dist_ma200"] > 0)].assign(ev=lambda x: x["ret_30"])
    print(f"  Portfolio simulation (1% risk/trade, max {cfg.ml.max_open_trades} open, costs included):")
    res = {}
    for label_, start in (("2021-01-01 -> now", "2021-01-01"), ("2024-01-18 -> now (V7 window)", "2024-01-18")):
        s_ml = simulate(ml_sig, panel, cfg, start)
        s_tr = simulate(trend_sig, panel, cfg, start)
        print(fmt(f"ML  {label_}", s_ml))
        print(fmt(f"trend {label_}", s_tr))
        res[label_] = {"ml": s_ml, "trend": s_tr}

    by_year = lab[lab["ev"] >= gate].groupby(lab["date"].dt.year)["r_net"]
    print("  ML gated trades by year: " + "  ".join(
        f"{y}: n={len(g)} avgR={g.mean():+.2f}" for y, g in by_year))
    return {"rr": rr, "auc": float(auc), "portfolio": res}


def evaluate_variants(rr: float) -> dict:
    """The two pre-declared fixes: (A) calibrated model, (B) calibrated model AND trend filter."""
    from ml_strategy import make_calibrated_model

    cfg = Config()
    cfg.ml.reward_risk = rr
    panel = load_panel(cfg)
    ds = build_dataset(panel, load_reference(cfg), cfg)
    wf = walk_forward(ds, cfg, model_factory=make_calibrated_model)
    oos = wf[wf["p"].notna() & wf["tradable"]].copy()
    oos["ev"] = expected_r(oos["p"].to_numpy(), rr)
    lab = oos[oos["label"].notna()]
    print(f"\n=== R:R {rr:.1f} - CALIBRATED model ===")
    lab = lab.assign(bucket=pd.qcut(lab["p"].rank(method="first"), 5, labels=False))
    cal = lab.groupby("bucket").agg(pred=("p", "mean"), actual=("label", "mean"), avg_r=("r_net", "mean"))
    print("  Calibration: " + "  ".join(f"{r.pred:4.0%}->{r.actual:4.0%}" for _, r in cal.iterrows()))

    gate = cfg.ml.min_expected_r
    trend = (oos["dist_ma50"] > 0) & (oos["btc_dist_ma200"] > 0)
    variants = {
        "A calibrated ML": oos[oos["ev"] >= gate],
        "B calibrated ML + trend": oos[(oos["ev"] >= gate) & trend],
    }
    res = {}
    for name, sig in variants.items():
        lab_sig = sig[sig["label"].notna()]
        print(fmt(f"{name} (all signals)", trade_stats(lab_sig["r_net"])))
        for label_, start in (("2021+", "2021-01-01"), ("2024-01-18+", "2024-01-18")):
            s = simulate(sig, panel, cfg, start)
            print(fmt(f"{name} portfolio {label_}", s))
            res[f"{name} {label_}"] = s
        yr = lab_sig.groupby(lab_sig["date"].dt.year)["r_net"]
        print("    by year: " + "  ".join(f"{y}:{g.mean():+.2f}R(n={len(g)})" for y, g in yr))
    return {"rr": rr, "variants": res}


if __name__ == "__main__":
    if "--variants" in sys.argv:
        rrs = [float(x) for x in sys.argv[1:] if x != "--variants"] or [1.5, 2.0]
        results = [evaluate_variants(rr) for rr in rrs]
        out = ROOT / "outputs" / "ml_walk_forward_variants.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        print(f"\nSaved {out}")
        sys.exit(0)
    rrs = [float(x) for x in sys.argv[1:]] or [1.5, 2.0]
    results = [evaluate(rr) for rr in rrs]
    out = ROOT / "outputs" / "ml_walk_forward.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out}")
