"""
Automatic ML research - runs itself weekly from autopilot.ps1.
Run manually: python ml_autoresearch.py [--force]

Each run:
  1. Refresh Fear & Greed history.
  2. Walk-forward test the CHAMPION and every CHALLENGER variant on the newest data
     (baseline / +Fear&Greed - see ml_strategy.VARIANTS).
  3. Promote a challenger ONLY if it beats the champion on all three tests:
       avg R per signal (+0.05R margin), portfolio Sharpe 2021+, and the last 180 days.
  4. Retrain the champion on all labelled history -> models/ml_bracket_model.joblib.
  5. Health gate: NEW ML entries are allowed only if the champion has >= 40 out-of-sample
     trades in the last 180 days (counted back from today) AND they average >= 0R.
     Too few recent trades counts as paused - no evidence is not a reason to trade.
  6. Write a report to outputs/research_reports/<date>.md.

Headline sentiment: archived daily by news_feed.py. Once 180+ days exist, the
report says so and it can be added as a challenger variant.
"""

from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

from config import Config  # noqa: E402
from ml_strategy import (  # noqa: E402
    VARIANTS, build_dataset, expected_r, load_panel, load_reference, train_final_model,
    variant_features, walk_forward,
)

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
CHAMPION_PATH = MODELS / "ml_champion.json"
STATUS_PATH = MODELS / "ml_status.json"
REPORTS = ROOT / "outputs" / "research_reports"
RUN_EVERY_DAYS = 7
RECENT_DAYS = 180
HEALTH_MIN_TRADES = 40


def _read(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def recent_window_start() -> pd.Timestamp:
    """Start of the health-gate window: RECENT_DAYS before today (UTC, tz-naive like the dataset)."""
    return pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() - pd.Timedelta(days=RECENT_DAYS)


def evaluate_variant(name: str, panel: dict, ds: pd.DataFrame, cfg: Config) -> dict:
    from ml_research import simulate, trade_stats

    p = panel
    wf = walk_forward(ds, cfg, features=variant_features(name))
    oos = wf[wf["p"].notna() & wf["tradable"]].copy()
    oos["ev"] = expected_r(oos["p"].to_numpy(), cfg.ml.reward_risk)
    sig = oos[oos["ev"] >= cfg.ml.min_expected_r]
    lab = sig[sig["label"].notna()]
    # Health window counts back from TODAY. (It used to count back from the last signal,
    # so a model that stopped signalling kept being judged on trades 9-15 months old.)
    recent_cut = recent_window_start()
    port = simulate(sig, p, cfg, "2021-01-01")
    return {
        "variant": name,
        "coins": len(p),
        "rows": int(len(ds)),
        "signals": trade_stats(lab["r_net"]),
        "recent_180d": trade_stats(lab[lab["date"] > recent_cut]["r_net"]),
        "portfolio_2021": port,
        "by_year": {int(y): round(float(g.mean()), 3) for y, g in lab.groupby(lab["date"].dt.year)["r_net"]},
        "_dataset": ds,
    }


def beats(ch: dict, champ: dict) -> bool:
    def avg(d, k):
        return d[k].get("avg_r", -9.0) if d[k].get("trades", 0) else -9.0
    return (
        avg(ch, "signals") >= avg(champ, "signals") + 0.05
        and ch["portfolio_2021"]["sharpe"] >= champ["portfolio_2021"]["sharpe"]
        and avg(ch, "recent_180d") >= avg(champ, "recent_180d")
    )


def run(force: bool = False) -> dict:
    cfg = Config()
    status = _read(STATUS_PATH, {})
    last = status.get("last_research")
    if not force and last and datetime.fromisoformat(last) > datetime.now(timezone.utc) - timedelta(days=RUN_EVERY_DAYS):
        print(f"[research] last run {last[:10]}; next due in <{RUN_EVERY_DAYS}d. Skipping (use --force).")
        return status

    from news_feed import archive_days, update_fear_greed

    print("[research] refreshing Fear & Greed ...")
    fng = update_fear_greed()

    panel = load_panel(cfg)
    ds = build_dataset(panel, load_reference(cfg), cfg)
    champion = _read(CHAMPION_PATH, {}).get("variant", "baseline")
    if champion not in VARIANTS:
        champion = "baseline"
    results = {}
    for name in VARIANTS:
        print(f"[research] walk-forward: {name} ...")
        results[name] = evaluate_variant(name, panel, ds, cfg)

    promoted = None
    contenders = [n for n in VARIANTS if n != champion and beats(results[n], results[champion])]
    if contenders:
        promoted = max(contenders, key=lambda n: results[n]["signals"]["avg_r"])
        champion = promoted

    print(f"[research] training champion '{champion}' on all history ...")
    train_final_model(results[champion]["_dataset"], cfg, variant=champion)

    rec = results[champion]["recent_180d"]
    # Entries need POSITIVE EVIDENCE: enough recent out-of-sample trades AND a non-negative
    # average. Too few recent trades (e.g. a model that stopped signalling) means paused.
    enough = rec.get("trades", 0) >= HEALTH_MIN_TRADES
    healthy = enough and rec.get("avg_r", -1.0) >= 0.0
    verdict = ("" if healthy else
               f" - paused: needs >= {HEALTH_MIN_TRADES} recent trades" if not enough else
               " - paused: recent average below 0R")
    now = datetime.now(timezone.utc).isoformat()
    status = {
        "last_research": now,
        "champion": champion,
        "promoted_this_run": promoted,
        "entries_allowed": bool(healthy),
        "health_reason": (
            f"last {RECENT_DAYS}d (since {recent_window_start().date()}): {rec.get('trades', 0)} OOS trades, "
            f"avg {(format(rec['avg_r'], '+.3f') + 'R') if rec.get('trades') else 'n/a'}{verdict}"
        ),
        "news_archive_days": archive_days(),
        "fear_greed_latest": float(fng["fng"].iloc[-1]) if not fng.empty else None,
    }
    MODELS.mkdir(exist_ok=True)
    STATUS_PATH.write_text(json.dumps(status, indent=2), encoding="utf-8")
    CHAMPION_PATH.write_text(json.dumps({"variant": champion, "since": now}, indent=2), encoding="utf-8")
    write_report(results, status, pattern_scorecard(ds))
    return status


def pattern_scorecard(ds: pd.DataFrame) -> pd.DataFrame:
    """
    For each chart pattern: how did 2:1 bracket trades started on that day turn out?
    Descriptive (whole history, every tradable coin) - it shows what the scanner
    sees; whether the MODEL can profit from it is decided by the walk-forward table.
    """
    from chart_patterns import BINARY_PATTERNS

    lab = ds[ds["label"].notna() & ds["tradable"]]
    rows = [{"pattern": "ALL SETUPS (baseline)", "n": len(lab), "win": lab["label"].mean(), "avg_r": lab["r_net"].mean()}]
    for p in BINARY_PATTERNS:
        hit = lab[lab[p] == 1]
        if len(hit) >= 50:
            rows.append({"pattern": p, "n": len(hit), "win": hit["label"].mean(), "avg_r": hit["r_net"].mean()})
    return pd.DataFrame(rows).sort_values("avg_r", ascending=False)


def write_report(results: dict, status: dict, scorecard: pd.DataFrame | None = None) -> Path:
    REPORTS.mkdir(parents=True, exist_ok=True)
    day = status["last_research"][:10]
    lines = [
        f"# ML research report - {day}",
        "",
        f"- **Champion:** `{status['champion']}`"
        + (f" (promoted this run)" if status["promoted_this_run"] else " (unchanged)"),
        f"- **New ML entries allowed:** {'YES' if status['entries_allowed'] else 'NO - paused'} "
        f"({status['health_reason']})",
        f"- **Fear & Greed today:** {status['fear_greed_latest']}",
        f"- **Headline archive:** {status['news_archive_days']} days collected "
        f"({'ready to test as a feature' if status['news_archive_days'] >= 180 else 'need 180 before it can be tested'})",
        "",
        "## Walk-forward results (out-of-sample, R:R 2.0, costs included)",
        "",
        "| Variant | Coins | Signals | Win | Avg R | PF | Last 180d avg R | Portfolio CAGR 2021+ | Sharpe | Max DD |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        s, rec, p = r["signals"], r["recent_180d"], r["portfolio_2021"]
        mark = " **(champion)**" if name == status["champion"] else ""
        lines.append(
            f"| {name}{mark} | {r['coins']} | {s.get('trades', 0)} | {s.get('win_rate', 0):.1%} | "
            f"{s.get('avg_r', 0):+.3f} | {s.get('profit_factor', 0):.2f} | {rec.get('avg_r', float('nan')):+.3f} "
            f"(n={rec.get('trades', 0)}) | {p['cagr']:+.1%} | {p['sharpe']:.2f} | {p['max_dd']:.1%} |"
        )
    lines += ["", "## Avg R per signal by year", ""]
    for name, r in results.items():
        lines.append(f"- `{name}`: " + ", ".join(f"{y}: {v:+.2f}R" for y, v in r["by_year"].items()))
    if scorecard is not None and not scorecard.empty:
        lines += ["", "## Chart pattern scorecard (2:1 trades started on the pattern day, all history)", "",
                  "| Pattern | Times seen | Hit target first | Avg R after costs |", "|---|---|---|---|"]
        for _, r in scorecard.iterrows():
            lines.append(f"| {r['pattern']} | {int(r['n'])} | {r['win']:.1%} | {r['avg_r']:+.3f} |")
    lines.append("")
    path = REPORTS / f"{day}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[research] report: {path}")
    return path


if __name__ == "__main__":
    st = run(force="--force" in sys.argv)
    print(json.dumps(st, indent=2))
