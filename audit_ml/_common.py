"""
Shared loaders for the ML audit. READ-ONLY with respect to the project: everything
written goes to audit_ml/cache or audit_ml/results. No project code is modified.
"""

from __future__ import annotations

import io
import json
import sys
import warnings
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT.parent
CACHE = AUDIT / "cache"
RESULTS = AUDIT / "results"
CACHE.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT))

from config import Config  # noqa: E402
import ml_strategy as ms  # noqa: E402

CHAMPION = "baseline"
RR = 2.0


def cfg_for(rr: float = RR, cost: float | None = None) -> Config:
    c = Config()
    c.ml.reward_risk = rr
    if cost is not None:
        c.ml.roundtrip_cost = cost
    return c


def panel_ref():
    cfg = cfg_for()
    with redirect_stdout(io.StringIO()):
        return ms.load_panel(cfg), ms.load_reference(cfg)


def label_end_dates(panel: dict, horizon: int) -> pd.DataFrame:
    """Calendar date of the last bar each row's label can look at (row position t + H)."""
    rows = []
    for sym, df in panel.items():
        idx = df.index
        end = pd.Series(idx, index=idx).shift(-horizon)
        rows.append(pd.DataFrame({"date": idx, "symbol": sym, "label_end": end.values}))
    return pd.concat(rows, ignore_index=True)


def dataset(rr: float = RR, cost: float | None = None, refresh: bool = False) -> pd.DataFrame:
    tag = f"ds_rr{rr}_c{cost if cost is not None else 'default'}.pkl"
    path = CACHE / tag
    if path.exists() and not refresh:
        return pd.read_pickle(path)
    panel, ref = panel_ref()
    ds = ms.build_dataset(panel, ref, cfg_for(rr, cost))
    ds = ds.merge(label_end_dates(panel, cfg_for().ml.horizon_days), on=["date", "symbol"], how="left")
    ds.to_pickle(path)
    return ds


def champion_oos(rr: float = RR, variant: str = CHAMPION, refresh: bool = False) -> pd.DataFrame:
    """Walk-forward OOS predictions exactly as ml_autoresearch.evaluate_variant produces them."""
    path = CACHE / f"wf_{variant}_rr{rr}.pkl"
    if path.exists() and not refresh:
        return pd.read_pickle(path)
    ds = dataset(rr)
    wf = ms.walk_forward(ds, cfg_for(rr), features=ms.variant_features(variant))
    wf["ev"] = ms.expected_r(wf["p"].to_numpy(), rr)
    wf.to_pickle(path)
    return wf


def gated(wf: pd.DataFrame, rr: float = RR) -> pd.DataFrame:
    gate = cfg_for(rr).ml.min_expected_r
    oos = wf[wf["p"].notna() & wf["tradable"]]
    return oos[oos["ev"] >= gate]


def auc(y, p) -> float:
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y, int)
    return float(roc_auc_score(y, p)) if len(set(y)) == 2 else float("nan")


def save(name: str, obj) -> Path:
    path = RESULTS / name
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(path, index=False)
    else:
        path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    return path
