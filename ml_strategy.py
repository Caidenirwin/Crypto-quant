"""
ML bracket-trade model.

Question the model answers for every (coin, day):
    "If I buy at tomorrow's open with a stop 1.5 x ATR below and a take-profit
     reward_risk x that distance above, will the take-profit be hit first?"

Pipeline
  1. load_panel()        every USD coin's FULL history (not V7's aligned window)
                         + BTC/ETH reference series for market regime
  2. build_dataset()     ~45 leak-free features per row (only data <= close t)
                         + triple-barrier label from bars t+1 .. t+horizon
  3. walk_forward()      train only on the past, predict the next 90 days,
                         purge rows whose label window overlaps the test period
  4. expected_r()        EV in R units = p*RR - (1-p)*1 - costs; trade if >= gate

Conservative label rules: stop is checked before take-profit inside a bar
(if both are touched the same day we assume the loss), gaps through the stop
fill at the open, and costs are charged on every trade.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from chart_patterns import PATTERN_FEATURES, pattern_features
from config import Config

ROOT = Path(__file__).resolve().parent

FEATURES: List[str] = [
    "ret_1", "ret_3", "ret_7", "ret_14", "ret_30", "ret_60", "ret_90",
    "vol_7", "vol_30", "vol_ratio", "atr_pct", "rsi_14",
    "dist_ma20", "dist_ma50", "dist_ma200", "ma50_slope",
    "sharpe_14", "sharpe_30", "dd_30", "dd_90", "up_30", "range_pos_20",
    "volu_5_30", "volu_1_20",
    "btc_ret_7", "btc_ret_30", "btc_dist_ma50", "btc_dist_ma200", "btc_vol_30",
    "eth_ret_30", "rel_btc_30",
    "breadth_ma50", "breadth_ma200", "xs_rank_ret7", "xs_rank_ret30", "xs_rank_vol30",
    "is_meme",
    # Real-world sentiment (news_feed.py): Crypto Fear & Greed Index, history since 2018.
    "fng", "fng_chg_7", "fng_ma_30",
] + PATTERN_FEATURES   # chart scanner (chart_patterns.py): candlesticks, breakouts, S/R, structure


# --------------------------------------------------------------------------- #
# Data                                                                        #
# --------------------------------------------------------------------------- #
def load_panel(cfg: Config) -> Dict[str, pd.DataFrame]:
    """Every USD-quoted coin with its own full history (no cross-coin alignment)."""
    from data import DataManager
    from main import build_symbol_file_map

    dm = DataManager(cfg)
    with redirect_stdout(io.StringIO()):
        loaded = dm.load_multi_csv(build_symbol_file_map(cfg, ROOT))
        usd, _, _ = dm.filter_by_allowed_quote_currency(loaded)
    return {s: df.sort_index() for s, df in usd.items() if len(df) > cfg.ml.min_history_bars + cfg.ml.horizon_days}


def load_reference(cfg: Config) -> Dict[str, pd.DataFrame]:
    from data import DataManager

    dm = DataManager(cfg)
    out = {}
    for name in ("BTC(USD)", "ETH(USD)"):
        path = ROOT / cfg.ml.reference_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing - run: python update_data.py")
        with redirect_stdout(io.StringIO()):
            out[name] = dm.load_symbol_csv(str(path), name)[name].sort_index()
    return out


# --------------------------------------------------------------------------- #
# Features                                                                    #
# --------------------------------------------------------------------------- #
def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    c_prev = df["Close"].shift(1)
    tr = pd.concat(
        [df["High"] - df["Low"], (df["High"] - c_prev).abs(), (df["Low"] - c_prev).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(n).mean()


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def symbol_features(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v = (df[k].astype(float) for k in ("Close", "High", "Low", "Volume"))
    lr = np.log(c).diff()
    f = pd.DataFrame(index=df.index)
    for w in (1, 3, 7, 14, 30, 60, 90):
        f[f"ret_{w}"] = np.log(c / c.shift(w))
    f["vol_7"] = lr.rolling(7).std()
    f["vol_30"] = lr.rolling(30).std()
    f["vol_ratio"] = f["vol_7"] / f["vol_30"]
    atr = _atr(df)
    f["atr"] = atr                              # absolute, used for stop sizing (not a feature)
    f["atr_pct"] = atr / c
    f["rsi_14"] = _rsi(c)
    for w in (20, 50, 200):
        f[f"dist_ma{w}"] = c / c.rolling(w).mean() - 1
    ma50 = c.rolling(50).mean()
    f["ma50_slope"] = ma50 / ma50.shift(10) - 1
    for w in (14, 30):
        f[f"sharpe_{w}"] = lr.rolling(w).mean() / lr.rolling(w).std()
    f["dd_30"] = c / c.rolling(30).max() - 1
    f["dd_90"] = c / c.rolling(90).max() - 1
    f["up_30"] = c / c.rolling(30).min() - 1
    lo20, hi20 = l.rolling(20).min(), h.rolling(20).max()
    f["range_pos_20"] = (c - lo20) / (hi20 - lo20).replace(0, np.nan)
    vpos = v.where(v > 0)
    f["volu_5_30"] = np.log(vpos.rolling(5).mean() / vpos.rolling(30).mean())
    f["volu_1_20"] = np.log(vpos / vpos.rolling(20).mean())
    f["bars"] = np.arange(len(df))
    return f.join(pattern_features(df, atr))


def market_features(ref: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    btc = ref["BTC(USD)"]["Close"].astype(float)
    eth = ref["ETH(USD)"]["Close"].astype(float)
    lr = np.log(btc).diff()
    m = pd.DataFrame(index=btc.index)
    m["btc_ret_7"] = np.log(btc / btc.shift(7))
    m["btc_ret_30"] = np.log(btc / btc.shift(30))
    m["btc_dist_ma50"] = btc / btc.rolling(50).mean() - 1
    m["btc_dist_ma200"] = btc / btc.rolling(200).mean() - 1
    m["btc_vol_30"] = lr.rolling(30).std()
    m["eth_ret_30"] = np.log(eth / eth.shift(30)).reindex(m.index)
    return m


# --------------------------------------------------------------------------- #
# Labels                                                                      #
# --------------------------------------------------------------------------- #
def triple_barrier(df: pd.DataFrame, atr: pd.Series, cfg: Config, side: str = "long") -> pd.DataFrame:
    """
    Outcome of a bracket trade signalled at close t, entered at open t+1.
    side="long":  stop below entry, target above.
    side="short": stop ABOVE entry, target below (profit when price falls).
    Returns label (1 = take-profit first), r_net (R multiple after costs),
    exit_offset (bars after the signal bar).
    """
    mc = cfg.ml
    o, h, l, c = (df[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))
    a = atr.to_numpy(float)
    n, H, rr = len(df), int(mc.horizon_days), float(mc.reward_risk)
    s = 1.0 if side == "long" else -1.0           # direction sign
    label = np.full(n, np.nan)
    r_net = np.full(n, np.nan)
    exit_off = np.full(n, np.nan)
    for t in range(n - H):
        entry = o[t + 1]
        risk = mc.stop_atr_mult * a[t]
        if not (entry > 0 and risk > 0) or np.isnan(risk):
            continue
        sl, tp = entry - s * risk, entry + s * rr * risk
        r, won, k_exit = None, 0, t + H
        for k in range(t + 1, t + 1 + H):
            adverse = l[k] if s > 0 else h[k]     # the side of the bar that hurts us
            favorable = h[k] if s > 0 else l[k]
            if k > t + 1 and s * (o[k] - sl) <= 0:  # gapped through the stop overnight
                r, k_exit = s * (o[k] - entry) / risk, k
                break
            if s * (adverse - sl) <= 0:           # stop checked first (conservative)
                r, k_exit = -1.0, k
                break
            if s * (favorable - tp) >= 0:
                r, won, k_exit = rr, 1, k
                break
        if r is None:                             # time exit at close of the last day
            r = s * (c[t + H] - entry) / risk
        label[t] = won
        r_net[t] = r - mc.roundtrip_cost * entry / risk
        exit_off[t] = k_exit - t
    return pd.DataFrame({"label": label, "r_net": r_net, "exit_offset": exit_off}, index=df.index)


# --------------------------------------------------------------------------- #
# Dataset                                                                     #
# --------------------------------------------------------------------------- #
def build_dataset(
    panel: Dict[str, pd.DataFrame], ref: Dict[str, pd.DataFrame], cfg: Config, with_labels: bool = True,
    side: str = "long",
) -> pd.DataFrame:
    mkt = market_features(ref)
    frames = []
    for sym, df in panel.items():
        f = symbol_features(df)
        if with_labels:
            f = f.join(triple_barrier(df, f["atr"], cfg, side=side))
        f["close"] = df["Close"].astype(float)
        f["symbol"] = sym
        f["is_meme"] = float(cfg.is_meme(sym))
        frames.append(f)
    ds = pd.concat(frames)
    ds.index.name = "date"
    ds = ds.reset_index()

    # Cross-sectional context: how this coin ranks vs. all coins that day.
    g = ds.groupby("date")
    ds["breadth_ma50"] = g["dist_ma50"].transform(lambda x: (x > 0).mean())
    ds["breadth_ma200"] = g["dist_ma200"].transform(lambda x: (x > 0).mean())
    ds["xs_rank_ret7"] = g["ret_7"].rank(pct=True)
    ds["xs_rank_ret30"] = g["ret_30"].rank(pct=True)
    ds["xs_rank_vol30"] = g["vol_30"].rank(pct=True)

    ds = ds.merge(mkt, left_on="date", right_index=True, how="left")
    from news_feed import fear_greed_features, load_fear_greed

    ds = ds.merge(fear_greed_features(load_fear_greed()), left_on="date", right_index=True, how="left")
    ds["rel_btc_30"] = ds["ret_30"] - ds["btc_ret_30"]
    ds = ds[ds["bars"] >= int(cfg.ml.min_history_bars)]
    ds = ds.replace([np.inf, -np.inf], np.nan)
    ds["tradable"] = ~ds["symbol"].isin(cfg.ml.no_trade_symbols)
    return ds.sort_values(["date", "symbol"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Model                                                                       #
# --------------------------------------------------------------------------- #
def make_model():
    from sklearn.ensemble import HistGradientBoostingClassifier

    # Shallow, heavily regularised trees: financial labels are very noisy.
    return HistGradientBoostingClassifier(
        learning_rate=0.03,
        max_iter=250,
        max_depth=3,
        min_samples_leaf=400,
        l2_regularization=2.0,
        random_state=42,
    )


class CalibratedModel:
    """
    Boosted trees fit on the older 80% of the training window, then an isotonic
    calibrator fit on the newest 20% (time-ordered, so no look-ahead). Fixes the
    raw model's over-confidence: it said ~48% when the real hit rate was ~32%.
    """

    def __init__(self, base_factory=make_model, holdout_frac: float = 0.2):
        self.base_factory = base_factory
        self.holdout_frac = holdout_frac

    def fit(self, X: pd.DataFrame, y: pd.Series):
        from sklearn.isotonic import IsotonicRegression

        cut = int(len(X) * (1 - self.holdout_frac))   # rows are date-sorted
        self.base = self.base_factory().fit(X.iloc[:cut], y.iloc[:cut])
        raw = self.base.predict_proba(X.iloc[cut:])[:, 1]
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw, y.iloc[cut:])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = self.iso.predict(self.base.predict_proba(X)[:, 1])
        return np.column_stack([1 - p, p])


def make_calibrated_model():
    return CalibratedModel()


def expected_r(p: np.ndarray, rr: float) -> np.ndarray:
    """EV per trade in R, before costs (costs are already inside r_net for evaluation)."""
    return p * rr - (1.0 - p)


def walk_forward(
    ds: pd.DataFrame,
    cfg: Config,
    start: str = "2021-01-01",
    model_factory=make_calibrated_model,
    features: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Out-of-sample predictions: each test block only ever sees a model trained on its past."""
    feats = list(features or FEATURES)
    out = ds.copy()
    out["p"] = np.nan
    step = pd.Timedelta(days=int(cfg.ml.retrain_every_days))
    purge = pd.Timedelta(days=int(cfg.ml.horizon_days) + 2)
    t0, end = pd.Timestamp(start), out["date"].max()
    while t0 <= end:
        t1 = t0 + step
        train = out[(out["date"] < t0 - purge) & out["label"].notna()]
        test_mask = (out["date"] >= t0) & (out["date"] < t1)
        if len(train) > 2000 and test_mask.any():
            model = model_factory().fit(train[feats], train["label"].astype(int))
            out.loc[test_mask, "p"] = model.predict_proba(out.loc[test_mask, feats])[:, 1]
        t0 = t1
    return out


# --------------------------------------------------------------------------- #
# Research variants (champion / challengers, see ml_autoresearch.py)          #
# --------------------------------------------------------------------------- #
FNG_FEATURES = ["fng", "fng_chg_7", "fng_ma_30"]
VARIANTS: Dict[str, dict] = {
    "baseline":                {"use_fng": False, "use_patterns": False},
    "fear_greed":              {"use_fng": True,  "use_patterns": False},
    "chart_patterns":          {"use_fng": False, "use_patterns": True},
    "fear_greed+chart_patterns": {"use_fng": True, "use_patterns": True},
}


def variant_features(name: str) -> List[str]:
    v = VARIANTS[name]
    return [
        f for f in FEATURES
        if (v["use_fng"] or f not in FNG_FEATURES) and (v["use_patterns"] or f not in PATTERN_FEATURES)
    ]


def train_final_model(ds: pd.DataFrame, cfg: Config, variant: str = "baseline", path: Optional[Path] = None):
    """Fit the calibrated model on ALL labelled history and save it for the paper trader."""
    import joblib

    feats = variant_features(variant)
    train = ds[ds["label"].notna()]
    model = make_calibrated_model().fit(train[feats], train["label"].astype(int))
    path = path or (ROOT / cfg.ml.model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "features": feats, "variant": variant,
         "symbols": sorted(ds["symbol"].unique()),
         "trained_through": str(train["date"].max().date()),
         "trained_at": pd.Timestamp.now(tz="UTC").isoformat(),
         "reward_risk": cfg.ml.reward_risk, "stop_atr_mult": cfg.ml.stop_atr_mult,
         "horizon_days": cfg.ml.horizon_days},
        path,
    )
    return model
