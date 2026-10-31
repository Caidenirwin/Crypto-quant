"""
Robustness checks for fragility and asset dependency.
"""

from __future__ import annotations

from copy import deepcopy
from math import comb
from random import Random
from statistics import median
from typing import Dict, List, Set, Tuple

import pandas as pd

from backtest import BacktestEngine
from config import Config
from data import DataManager


def run_robustness_suite(config: Config, market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    data_manager = DataManager(config)

    def _run_case(
        category: str,
        case_name: str,
        cfg: Config,
        data: Dict[str, pd.DataFrame],
        extra: Dict[str, object] | None = None,
    ) -> None:
        base_row: Dict[str, object] = {"category": category, "case": case_name}
        if extra:
            base_row.update(extra)
        try:
            result = BacktestEngine(cfg).run(data)
            diag = result.diagnostics if result.diagnostics else {}
            per_asset = diag.get("per_asset_contribution_pnl", {})
            worst_asset = None
            worst_asset_pnl = 0.0
            if isinstance(per_asset, dict) and per_asset:
                worst_asset, worst_asset_pnl = min(
                    ((str(k), float(v)) for k, v in per_asset.items()),
                    key=lambda x: x[1],
                )
            row = dict(base_row)
            row.update(
                {
                    "total_return": result.total_return,
                    "cagr": result.cagr,
                    "sharpe": result.sharpe,
                    "max_drawdown": result.max_drawdown,
                    "worst_asset_contributor": worst_asset,
                    "worst_asset_pnl": worst_asset_pnl,
                    "best_asset_contributor": (
                        max(
                            ((str(k), float(v)) for k, v in per_asset.items()),
                            key=lambda x: x[1],
                        )[0]
                        if isinstance(per_asset, dict) and per_asset
                        else None
                    ),
                    "best_asset_pnl": (
                        max(
                            (float(v) for v in per_asset.values()),
                            default=0.0,
                        )
                        if isinstance(per_asset, dict)
                        else 0.0
                    ),
                    "strategy_contribution_proxy_pnl": diag.get(
                        "strategy_contribution_proxy_pnl", {}
                    ),
                    "per_asset_contribution_pnl": per_asset,
                    "traded_symbols": diag.get("traded_symbols", []),
                    "yield_enabled_in_case": bool(cfg.strategy.yield_enabled),
                    "pair_selected_pairs": diag.get("pair_selected_pairs", []),
                }
            )
            rows.append(row)
        except Exception as exc:
            row = dict(base_row)
            row["error"] = str(exc)
            rows.append(row)

    _run_case("base", "baseline", deepcopy(config), market_data)

    # 1) Drop-one-asset test.
    for symbol in sorted(market_data.keys()):
        dropped = {s: df for s, df in market_data.items() if s != symbol}
        if len(dropped) < 2:
            continue
        _run_case("drop_one", f"drop_{symbol}", deepcopy(config), dropped)

    # 2) Randomized universes (quality-filtered by default).
    rng = Random(config.backtest.random_seed)
    all_symbols = sorted(market_data.keys())
    common_dates = _common_dates(market_data)
    if len(common_dates) >= 2:
        select_idx = min(
            max(int(config.data.universe_quality_lookback_days) - 1, 0),
            len(common_dates) - 2,
        )
        selection_date = common_dates[select_idx]
    else:
        selection_date = common_dates[0] if common_dates else None

    def _run_random_cases(
        category: str,
        candidates: List[str],
        label: str,
        desired_size: int,
        min_size: int,
        requested_trials: int,
    ) -> None:
        if not selection_date:
            rows.append(
                {"category": category, "case": f"{category}_no_dates", "error": "No aligned dates"}
            )
            return
        if len(candidates) < 2:
            rows.append(
                {
                    "category": category,
                    "case": f"{category}_insufficient_candidates",
                    "error": "Not enough candidate symbols",
                    "universe_label": label,
                }
            )
            return
        # Quality-random subsets should be < eligible universe when possible.
        desired_size = int(max(2, desired_size))
        min_size = int(max(2, min_size))
        n = len(candidates)
        if n > desired_size:
            subset_size = desired_size
        elif n > min_size:
            subset_size = max(min_size, n - 1)
        else:
            subset_size = min(n, desired_size)
        subset_size = max(2, min(subset_size, n))

        max_unique = comb(n, subset_size) if n >= subset_size else 0
        target_trials = min(int(max(1, requested_trials)), max_unique if max_unique > 0 else 1)
        seen: Set[Tuple[str, ...]] = set()
        duplicate_skipped = 0
        quality_snapshot = data_manager.universe_quality_snapshot(market_data, selection_date)
        i = 0
        attempts = 0
        max_attempts = max(50, target_trials * 20)
        while i < target_trials and attempts < max_attempts:
            attempts += 1
            picked = tuple(sorted(rng.sample(candidates, subset_size)))
            if picked in seen:
                duplicate_skipped += 1
                continue
            seen.add(picked)
            picked_list = list(picked)
            sub = {
                s: market_data[s].loc[market_data[s].index >= selection_date].copy()
                for s in picked_list
            }
            case_cfg = deepcopy(config)
            if bool(config.backtest.robustness_disable_momentum):
                case_cfg.strategy.momentum_enabled = False
            if bool(config.backtest.robustness_disable_yield) or bool(
                config.backtest.yield_disable_in_random_universe_tests
            ):
                case_cfg.strategy.yield_enabled = False
            qual = [float(quality_snapshot.get(s, {}).get("quality_score", 0.0)) for s in picked_list]
            avg_vol = [float(quality_snapshot.get(s, {}).get("avg_volume", 0.0)) for s in picked_list]
            avg_dv = [
                float(quality_snapshot.get(s, {}).get("avg_dollar_volume", 0.0)) for s in picked_list
            ]
            avg_hist = [float(quality_snapshot.get(s, {}).get("history_bars", 0.0)) for s in picked_list]
            i += 1
            _run_case(
                category,
                f"{category}_{i}_{'_'.join(picked_list)}",
                case_cfg,
                sub,
                extra={
                    "selected_assets": ",".join(picked_list),
                    "selection_date": str(selection_date.date()),
                    "universe_label": label,
                    "universe_avg_quality_score": float(sum(qual) / max(len(qual), 1)),
                    "universe_avg_volume": float(sum(avg_vol) / max(len(avg_vol), 1)),
                    "universe_avg_dollar_volume": float(sum(avg_dv) / max(len(avg_dv), 1)),
                    "universe_avg_history_bars": float(sum(avg_hist) / max(len(avg_hist), 1)),
                    "yield_disabled_for_random_test": bool(
                        config.backtest.robustness_disable_yield
                        or config.backtest.yield_disable_in_random_universe_tests
                    ),
                    "momentum_disabled_for_random_test": bool(
                        config.backtest.robustness_disable_momentum
                    ),
                    "requested_trials": int(requested_trials),
                    "successful_unique_trials": int(target_trials),
                    "duplicate_trials_skipped": int(duplicate_skipped),
                    "subset_size_used": int(subset_size),
                },
            )
        if i < target_trials:
            rows.append(
                {
                    "category": category,
                    "case": f"{category}_insufficient_unique_samples",
                    "error": "Could not generate enough unique random universes",
                    "requested_trials": int(requested_trials),
                    "successful_unique_trials": int(i),
                    "duplicate_trials_skipped": int(duplicate_skipped),
                    "subset_size_used": int(subset_size),
                    "candidate_count": int(n),
                    "max_unique_combinations": int(max_unique),
                    "universe_label": label,
                }
            )

    eligible_symbols = all_symbols
    eligible_count = len(all_symbols)
    if selection_date and config.backtest.random_universe_use_quality_filter:
        eligible_set, _ = data_manager.quality_eligible_symbols(market_data, selection_date)
        eligible_symbols = sorted(eligible_set)
        eligible_count = len(eligible_symbols)
    quality_label = (
        "LOW_QUALITY_UNIVERSE"
        if eligible_count < int(config.backtest.min_eligible_universe_size)
        else "QUALITY_FILTERED_UNIVERSE"
    )
    _run_random_cases(
        "random_universe",
        eligible_symbols,
        quality_label,
        desired_size=int(config.backtest.quality_random_universe_size),
        min_size=int(config.backtest.quality_random_universe_min_size),
        requested_trials=int(config.backtest.quality_random_universe_trials),
    )
    if config.backtest.random_universe_include_raw_stress_test:
        _run_random_cases(
            "random_universe_raw",
            all_symbols,
            "RAW_STRESS_UNIVERSE",
            desired_size=int(config.backtest.random_universe_size),
            min_size=2,
            requested_trials=int(config.backtest.random_universe_trials),
        )

    # 3) Parameter variation test (small +/- shifts around key thresholds).
    variation = float(config.backtest.parameter_variation_pct)
    perturbations = [
        ("pair_z_entry_down", ("strategy", "pair_z_entry"), 1.0 - variation),
        ("pair_z_entry_up", ("strategy", "pair_z_entry"), 1.0 + variation),
        ("stop_loss_down", ("risk", "stop_loss_pct"), 1.0 - variation),
        ("stop_loss_up", ("risk", "stop_loss_pct"), 1.0 + variation),
        ("breakout_vol_z_down", ("strategy", "breakout_volume_spike_z"), 1.0 - variation),
        ("breakout_vol_z_up", ("strategy", "breakout_volume_spike_z"), 1.0 + variation),
        ("momentum_vol_mult_down", ("strategy", "momentum_volume_filter_multiplier"), 1.0 - variation),
        ("momentum_vol_mult_up", ("strategy", "momentum_volume_filter_multiplier"), 1.0 + variation),
    ]
    for name, path, multiplier in perturbations:
        cfg = deepcopy(config)
        obj = getattr(cfg, path[0])
        cur = float(getattr(obj, path[1]))
        setattr(obj, path[1], max(0.0, cur * multiplier))
        _run_case("parameter_variation", name, cfg, market_data)

    # 4) Rebalance offset sensitivity.
    for offset in config.backtest.robustness_rebalance_offsets:
        cfg = deepcopy(config)
        cfg.portfolio.rebalance_initial_offset_days = int(offset)
        _run_case("rebalance_offset", f"offset_{int(offset)}", cfg, market_data)

    # 5) Fee/slippage sensitivity.
    for mult in config.backtest.robustness_fee_multipliers:
        cfg = deepcopy(config)
        m = float(max(0.0, mult))
        cfg.portfolio.fee_rate = float(config.portfolio.fee_rate) * m
        cfg.portfolio.slippage_rate = float(config.portfolio.slippage_rate) * m
        _run_case("fee_slippage_sensitivity", f"fee_slip_x{m:.2f}", cfg, market_data)

    # 6) Pair-focused diagnostic sweep.
    pair_scenarios = [
        ("full_default", dict()),
        (
            "pair_only",
            {
                "momentum_enabled": False,
                "breakout_enabled": False,
                "volatility_enabled": False,
                "pair_trading_enabled": True,
                "yield_enabled": False,
            },
        ),
        (
            "pair_plus_breakout",
            {
                "momentum_enabled": False,
                "breakout_enabled": True,
                "volatility_enabled": False,
                "pair_trading_enabled": True,
                "yield_enabled": False,
            },
        ),
        (
            "pair_plus_volatility",
            {
                "momentum_enabled": False,
                "breakout_enabled": False,
                "volatility_enabled": True,
                "pair_trading_enabled": True,
                "yield_enabled": False,
            },
        ),
        (
            "pair_plus_breakout_volatility",
            {
                "momentum_enabled": False,
                "breakout_enabled": True,
                "volatility_enabled": True,
                "pair_trading_enabled": True,
                "yield_enabled": False,
            },
        ),
    ]
    for name, flags in pair_scenarios:
        cfg = deepcopy(config)
        for k, v in flags.items():
            setattr(cfg.strategy, k, bool(v))
        _run_case("pair_focus", name, cfg, market_data)

    return pd.DataFrame(rows)


def cagr_dispersion_by_category(results_df: pd.DataFrame) -> pd.DataFrame:
    usable = results_df.dropna(subset=["category", "cagr"]).copy()
    if usable.empty:
        return pd.DataFrame(
            columns=["category", "best_cagr", "worst_cagr", "median_cagr", "spread"]
        )
    out_rows: List[Dict[str, float | str]] = []
    for category, grp in usable.groupby("category"):
        cagr_vals = [float(v) for v in grp["cagr"].tolist()]
        if not cagr_vals:
            continue
        out_rows.append(
            {
                "category": str(category),
                "best_cagr": max(cagr_vals),
                "worst_cagr": min(cagr_vals),
                "median_cagr": float(median(cagr_vals)),
                "spread": max(cagr_vals) - min(cagr_vals),
            }
        )
    out = pd.DataFrame(out_rows)
    if out.empty:
        return out
    return out.sort_values("spread", ascending=False).reset_index(drop=True)


def largest_cagr_spread_category(results_df: pd.DataFrame) -> Dict[str, object]:
    disp = cagr_dispersion_by_category(results_df)
    if disp.empty:
        return {
            "category": "none",
            "best_cagr": 0.0,
            "worst_cagr": 0.0,
            "median_cagr": 0.0,
            "spread": 0.0,
        }
    top = disp.iloc[0]
    return {
        "category": str(top["category"]),
        "best_cagr": float(top["best_cagr"]),
        "worst_cagr": float(top["worst_cagr"]),
        "median_cagr": float(top["median_cagr"]),
        "spread": float(top["spread"]),
    }


def derive_asset_stability_penalties(
    results_df: pd.DataFrame,
    config: Config,
) -> Dict[str, float]:
    """
    Build soft per-asset penalties from drop-one sensitivity.
    Larger CAGR change when an asset is removed => larger penalty (still soft).
    """
    usable = results_df.dropna(subset=["category", "case", "cagr"]).copy()
    if usable.empty:
        return {}

    base_rows = usable[(usable["category"] == "base") & (usable["case"] == "baseline")]
    if base_rows.empty:
        return {}
    base_cagr = float(base_rows.iloc[0]["cagr"])
    denom = max(float(config.backtest.fragility_cagr_range_warn), 1e-9)
    strength = float(config.backtest.asset_penalty_strength)
    floor = float(config.backtest.asset_penalty_min)

    penalties: Dict[str, float] = {}
    drop_rows = usable[usable["category"] == "drop_one"]
    for _, row in drop_rows.iterrows():
        case = str(row["case"])
        if not case.startswith("drop_"):
            continue
        symbol = case[len("drop_") :]
        delta = abs(base_cagr - float(row["cagr"]))
        severity = min(1.0, delta / denom)
        penalty = max(floor, 1.0 - strength * severity)
        penalties[symbol] = float(penalty)
    return penalties


def random_universe_summary(results_df: pd.DataFrame, category: str = "random_universe") -> Dict[str, float]:
    df = results_df[(results_df["category"] == category)].dropna(subset=["cagr", "sharpe", "max_drawdown"])
    if df.empty:
        return {"best_cagr": 0.0, "worst_cagr": 0.0, "median_cagr": 0.0, "spread": 0.0}
    return {
        "best_cagr": float(df["cagr"].max()),
        "worst_cagr": float(df["cagr"].min()),
        "median_cagr": float(df["cagr"].median()),
        "spread": float(df["cagr"].max() - df["cagr"].min()),
    }


def worst_random_universe_diagnostics(
    results_df: pd.DataFrame,
    category: str = "random_universe",
    top_n: int = 3,
) -> pd.DataFrame:
    cols = [
        "case",
        "selected_assets",
        "worst_asset_contributor",
        "worst_asset_pnl",
        "universe_avg_quality_score",
        "universe_avg_volume",
        "universe_avg_dollar_volume",
        "universe_avg_history_bars",
        "cagr",
        "sharpe",
        "max_drawdown",
        "universe_label",
        "error",
    ]
    available = [c for c in cols if c in results_df.columns]
    df = results_df[results_df["category"] == category].copy()
    if df.empty:
        return pd.DataFrame(columns=available)
    if "cagr" in df.columns:
        df = df.sort_values("cagr", ascending=True)
    return df.head(top_n)[available]


def rebalance_offset_diagnostics(results_df: pd.DataFrame) -> Dict[str, object]:
    df = results_df[results_df["category"] == "rebalance_offset"].dropna(subset=["cagr", "sharpe"])
    if df.empty:
        return {"worst_offset_case": None, "cagr_spread": 0.0, "timing_dependency": False}
    worst = df.sort_values("cagr", ascending=True).iloc[0]
    spread = float(df["cagr"].max() - df["cagr"].min())
    return {
        "worst_offset_case": str(worst["case"]),
        "worst_offset_cagr": float(worst["cagr"]),
        "worst_offset_sharpe": float(worst["sharpe"]),
        "cagr_spread": spread,
        "timing_dependency": bool(spread > 0.20),
    }


def parse_selected_assets(cell: object) -> List[str]:
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    txt = str(cell).strip()
    if not txt:
        return []
    return [x.strip() for x in txt.split(",") if x.strip()]


def quote_currency(symbol: str) -> str:
    s = str(symbol).strip().upper()
    if "(" in s and ")" in s and s.rfind("(") < s.rfind(")"):
        return s[s.rfind("(") + 1 : s.rfind(")")]
    return "UNKNOWN"


def quote_currency_summary(symbols: List[str]) -> Dict[str, object]:
    quotes = {quote_currency(s) for s in symbols if s}
    known = {q for q in quotes if q != "UNKNOWN"}
    mixed = len(known) > 1
    return {"quotes": sorted(list(quotes)), "mixed": mixed}


def compare_base_vs_random_universes(
    base_assets: List[str], random_df: pd.DataFrame, base_cagr: float
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    base_set = set(base_assets)
    for _, row in random_df.iterrows():
        assets = parse_selected_assets(row.get("selected_assets"))
        aset = set(assets)
        rows.append(
            {
                "case": row.get("case"),
                "selected_assets": ",".join(assets),
                "missing_from_random": ",".join(sorted(base_set - aset)),
                "added_in_random": ",".join(sorted(aset - base_set)),
                "cagr": row.get("cagr"),
                "cagr_vs_base": (float(row.get("cagr")) - float(base_cagr))
                if pd.notna(row.get("cagr"))
                else None,
            }
        )
    return pd.DataFrame(rows)


def run_universe_strategy_ablation(
    config: Config,
    market_data: Dict[str, pd.DataFrame],
    symbols: List[str],
    start_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Run targeted ablation on a fixed universe to identify sleeve failure.
    """
    universe: Dict[str, pd.DataFrame] = {}
    for s in symbols:
        if s not in market_data:
            continue
        df = market_data[s]
        if start_date is not None:
            df = df.loc[df.index >= start_date].copy()
        universe[s] = df
    if len(universe) < 2:
        return pd.DataFrame(
            [{"scenario": "invalid_universe", "error": "Need at least 2 symbols for ablation"}]
        )

    scenarios = [
        ("all_strategies", None, None),
        ("momentum_only", {"momentum"}, None),
        ("no_momentum", None, "momentum"),
        ("no_breakout", None, "breakout"),
        ("no_pair_trading", None, "pair_trading"),
        ("no_volatility", None, "volatility"),
        ("no_yield", None, "yield"),
    ]
    rows: List[Dict[str, object]] = []
    for name, only_set, drop_one in scenarios:
        cfg = deepcopy(config)
        enabled = list(cfg.strategy.enabled_strategies)
        if only_set is not None:
            enabled = [s for s in enabled if s in only_set]
        if drop_one is not None:
            enabled = [s for s in enabled if s != drop_one]
        if not enabled:
            rows.append({"scenario": name, "error": "No enabled strategies"})
            continue
        cfg.strategy.enabled_strategies = enabled
        try:
            res = BacktestEngine(cfg).run(universe)
            rows.append(
                {
                    "scenario": name,
                    "enabled_strategies": ",".join(enabled),
                    "total_return": float(res.total_return),
                    "cagr": float(res.cagr),
                    "sharpe": float(res.sharpe),
                    "max_drawdown": float(res.max_drawdown),
                }
            )
        except Exception as exc:
            rows.append({"scenario": name, "enabled_strategies": ",".join(enabled), "error": str(exc)})
    return pd.DataFrame(rows)


def classify_failure_mode(
    base_cagr: float,
    random_median_cagr: float,
    worst_row: Dict[str, object] | None,
    ablation_df: pd.DataFrame,
    quote_mixed: bool,
    min_hist_issue: bool = False,
) -> Tuple[str, str]:
    if quote_mixed:
        return (
            "QUOTE_CURRENCY_DATA_ISSUE",
            "Normalize quote currency or convert EUR assets into USD terms before combining.",
        )
    if min_hist_issue:
        return (
            "NEED_MORE_DATA_HISTORY",
            "Increase aligned history depth or exclude symbols with short histories.",
        )
    if not ablation_df.empty and "cagr" in ablation_df.columns:
        all_row = ablation_df[ablation_df["scenario"] == "all_strategies"]
        if not all_row.empty:
            all_cagr = float(all_row.iloc[0]["cagr"])
            better = ablation_df[ablation_df["cagr"] > (all_cagr + 0.20)]
            if not better.empty:
                return (
                    "STRATEGY_SLEEVE_FAILURE",
                    "Investigate sleeves whose removal materially improves worst-universe performance.",
                )
    if worst_row and isinstance(worst_row.get("per_asset_contribution_pnl"), dict):
        per_asset = worst_row["per_asset_contribution_pnl"]
        if per_asset:
            abs_total = sum(abs(float(v)) for v in per_asset.values())
            worst_val = min(float(v) for v in per_asset.values())
            if abs_total > 0 and abs(worst_val) / abs_total > 0.40:
                return (
                    "ASSET_CONTRIBUTION_FAILURE",
                    "Add asset-specific controls or temporary exclusion logic for dominant loss drivers.",
                )
    if (base_cagr - random_median_cagr) > 0.30:
        return (
            "UNIVERSE_COMPOSITION_DEPENDENCY",
            "Prioritize robustness on alternative universes before threshold tuning.",
        )
    return ("UNIVERSE_COMPOSITION_DEPENDENCY", "Run longer-window robustness with more diverse symbols.")


def _common_dates(market_data: Dict[str, pd.DataFrame]) -> List[pd.Timestamp]:
    common = None
    for df in market_data.values():
        common = df.index if common is None else common.intersection(df.index)
    if common is None:
        return []
    return list(common.sort_values())


def fragility_warning(results_df: pd.DataFrame, config: Config) -> bool:
    usable = results_df.dropna(subset=["cagr", "sharpe", "max_drawdown"]).copy()
    if usable.empty:
        return True

    cagr_range = float(usable["cagr"].max() - usable["cagr"].min())
    sharpe_range = float(usable["sharpe"].max() - usable["sharpe"].min())
    dd_range = float(usable["max_drawdown"].max() - usable["max_drawdown"].min())

    return (
        cagr_range > float(config.backtest.fragility_cagr_range_warn)
        or sharpe_range > float(config.backtest.fragility_sharpe_range_warn)
        or dd_range > float(config.backtest.fragility_drawdown_range_warn)
    )


def stability_summary(
    base_metrics: Dict[str, float],
    robustness_df: pd.DataFrame,
    config: Config,
) -> Dict[str, object]:
    usable = robustness_df.dropna(subset=["cagr", "sharpe", "max_drawdown"]).copy()
    if usable.empty:
        return {
            "robustness_score": 0.0,
            "warnings": ["WARNING: Strategy may be overfit or fragile"],
            "median_cagr": 0.0,
            "median_sharpe": 0.0,
            "median_max_drawdown": 0.0,
            "avg_sharpe": 0.0,
            "worst_drawdown": 0.0,
            "cagr_spread": 0.0,
            "weak_scenarios": 0,
            "scenario_count": 0,
        }

    median_cagr = float(usable["cagr"].median())
    median_sharpe = float(usable["sharpe"].median())
    median_dd = float(usable["max_drawdown"].median())
    avg_sharpe = float(usable["sharpe"].mean())
    worst_dd = float(usable["max_drawdown"].min())
    cagr_spread = float(usable["cagr"].max() - usable["cagr"].min())

    weak_mask = (usable["sharpe"] < config.backtest.fragility_min_scenario_sharpe) | (
        usable["max_drawdown"] < -config.backtest.fragility_worst_drawdown_warn
    )
    weak_count = int(weak_mask.sum())
    scenario_count = int(len(usable))
    weak_ratio = float(weak_count / scenario_count) if scenario_count > 0 else 1.0

    # 0-100 stability score (higher is better).
    sharpe_component = max(0.0, min(1.0, avg_sharpe / 2.0))
    drawdown_component = max(0.0, min(1.0, 1.0 - (abs(worst_dd) / 0.60)))
    spread_component = max(
        0.0,
        min(
            1.0,
            1.0 - (cagr_spread / max(config.backtest.fragility_cagr_range_warn, 1e-9)),
        ),
    )
    weak_component = max(0.0, min(1.0, 1.0 - weak_ratio))
    score = 100.0 * (
        0.35 * sharpe_component
        + 0.25 * drawdown_component
        + 0.20 * spread_component
        + 0.20 * weak_component
    )

    warnings: List[str] = []
    base_sharpe = float(base_metrics.get("sharpe", 0.0))
    base_cagr = float(base_metrics.get("cagr", 0.0))

    if worst_dd < -config.backtest.fragility_worst_drawdown_warn:
        warnings.append("WARNING: Worst-case robustness drawdown breached threshold.")
    if cagr_spread > config.backtest.fragility_cagr_range_warn:
        warnings.append("WARNING: CAGR spread across robustness tests is too wide.")
    if weak_ratio > config.backtest.fragility_weak_scenario_ratio_warn:
        warnings.append("WARNING: Too many weak robustness scenarios.")
    if (base_sharpe - median_sharpe) > config.backtest.fragility_base_vs_median_sharpe_gap_warn:
        warnings.append("WARNING: Base Sharpe is much higher than robustness median.")
    if (base_cagr - median_cagr) > config.backtest.fragility_base_vs_median_cagr_gap_warn:
        warnings.append("WARNING: Base CAGR is much higher than robustness median.")

    return {
        "robustness_score": float(score),
        "warnings": warnings,
        "median_cagr": median_cagr,
        "median_sharpe": median_sharpe,
        "median_max_drawdown": median_dd,
        "avg_sharpe": avg_sharpe,
        "worst_drawdown": worst_dd,
        "cagr_spread": cagr_spread,
        "weak_scenarios": weak_count,
        "scenario_count": scenario_count,
    }
