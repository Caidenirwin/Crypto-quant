"""
Entry point for the modular crypto quant trading system.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from fnmatch import fnmatch
from pathlib import Path
from typing import Dict

import pandas as pd

from backtest import BacktestEngine
from config import Config
from data import DataManager
from robustness import (
    cagr_dispersion_by_category,
    derive_asset_stability_penalties,
    fragility_warning,
    largest_cagr_spread_category,
    rebalance_offset_diagnostics,
    random_universe_summary,
    run_robustness_suite,
    stability_summary,
    worst_random_universe_diagnostics,
)


def _should_exclude_csv(path: Path, base: Path, excluded_patterns: list[str]) -> bool:
    rel = path.relative_to(base).as_posix()
    name = path.name
    for pat in excluded_patterns:
        p = str(pat).strip()
        if not p:
            continue
        if fnmatch(rel, p) or fnmatch(name, p):
            return True
    return False


def auto_discover_symbol_files(
    root: Path,
    search_roots: list[str] | None = None,
    excluded_patterns: list[str] | None = None,
) -> Dict[str, str]:
    """
    Discover CSV files in project root and map them to symbols using filename prefixes.
    Example filename: BTC-USD.csv -> symbol BTC
    """
    mapping: Dict[str, str] = {}
    roots = search_roots if search_roots else ["historic_data"]
    excluded = excluded_patterns if excluded_patterns else []
    for rel in roots:
        base = (root / rel).resolve()
        if not base.exists() or not base.is_dir():
            continue
        for p in base.rglob("*.csv"):
            if _should_exclude_csv(p, base, excluded):
                continue
            symbol = p.stem.split("-")[0].strip().upper().replace(" ", "_")
            if symbol:
                mapping[symbol] = str(p)
    return mapping


def build_symbol_file_map(config: Config, root: Path) -> Dict[str, str]:
    """
    Build symbol->file mapping from configurable sources.
    """
    mode = str(config.data.symbol_source_mode).strip().lower()
    roots = list(config.data.market_data_dirs) if config.data.market_data_dirs else list(config.data.symbol_search_roots)
    auto_map = (
        auto_discover_symbol_files(
            root,
            search_roots=roots,
            excluded_patterns=list(config.data.excluded_data_patterns),
        )
        if config.data.auto_detect_symbol_files
        else {}
    )
    cfg_map = dict(config.data.configured_symbol_files)

    if mode == "auto":
        return auto_map
    if mode == "config":
        return cfg_map
    if mode == "both":
        merged = dict(auto_map)
        merged.update(cfg_map)
        return merged
    raise ValueError("Invalid data.symbol_source_mode. Use one of: auto, config, both.")


def main() -> None:
    config = Config()
    data_manager = DataManager(config)

    root = Path(__file__).resolve().parent
    symbol_to_file = build_symbol_file_map(config, root)
    if not symbol_to_file:
        raise RuntimeError("No CSV files found. Add OHLCV CSV files to project root.")
    print(
        "Asset Source Mode: "
        f"{config.data.symbol_source_mode} "
        f"(configured={len(config.data.configured_symbol_files)}, auto_detect={config.data.auto_detect_symbol_files})"
    )

    loaded_data = data_manager.load_multi_csv(symbol_to_file)
    if len(loaded_data) < 2:
        raise RuntimeError("Not enough valid OHLCV symbols loaded.")

    def align_with_min_history(data: Dict[str, pd.DataFrame]) -> tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
        """
        Align symbols on common dates while dropping symbols that cannot satisfy
        minimum common-history requirements.
        """
        working = dict(data)
        dropped: Dict[str, str] = {}
        min_hist = int(config.data.min_history_days)
        while len(working) >= 2:
            common_index = None
            for df in working.values():
                common_index = df.index if common_index is None else common_index.intersection(df.index)
            if common_index is None or len(common_index) == 0:
                # Drop the shortest history symbol to recover overlap.
                shortest = min(working.items(), key=lambda kv: len(kv[1]))[0]
                dropped[shortest] = "NO_COMMON_DATE_OVERLAP"
                del working[shortest]
                continue

            too_short = []
            aligned: Dict[str, pd.DataFrame] = {}
            for sym, df in working.items():
                aligned_df = df.loc[common_index].copy().sort_index()
                if len(aligned_df) < min_hist:
                    too_short.append(sym)
                else:
                    aligned[sym] = aligned_df
            if not too_short:
                return aligned, dropped
            for sym in too_short:
                dropped[sym] = "INSUFFICIENT_COMMON_HISTORY"
                del working[sym]
        return {}, dropped

    quote_report = data_manager.quote_currency_report(list(loaded_data.keys()))
    print("Quote Currencies Detected")
    print(f"- Currencies: {', '.join(quote_report['quotes'])}")
    for q, syms in quote_report["by_quote"].items():
        print(f"- {q}: {', '.join(sorted(syms))}")
    if quote_report["unknown_symbols"]:
        print(f"- Unknown quote symbols: {', '.join(sorted(quote_report['unknown_symbols']))}")
    if quote_report["mixed_quote_warning"]:
        print("MIXED_QUOTE_CURRENCY_WARNING")

    filtered_data, include_reasons, excluded_symbols = data_manager.filter_by_allowed_quote_currency(
        loaded_data
    )
    print("\nQuote Filtering")
    print(
        f"- Allowed quote currency: {config.data.allowed_quote_currency}"
    )
    print(
        f"- Included symbols ({len(filtered_data)}): "
        f"{', '.join(sorted(filtered_data.keys())) if filtered_data else 'none'}"
    )
    if excluded_symbols:
        print("- Excluded symbols:")
        for s, reason in sorted(excluded_symbols.items()):
            print(f"  {s}: {reason}")
        print("- FX conversion placeholder: no FX data provided; excluded non-allowed quote symbols.")
    # Quality pre-filter: exclude low-quality symbols BEFORE alignment so that
    # symbols with thin volume or stale/short data don't restrict the common-date
    # intersection for the rest of the universe.
    pre_quality_data, pre_quality_dropped = data_manager.pre_filter_quality(filtered_data)
    if pre_quality_dropped:
        print("- Pre-filter quality exclusions (before alignment):")
        for s, reason in sorted(pre_quality_dropped.items()):
            print(f"  {s}: {reason}")
    print(
        f"- After quality pre-filter: {len(pre_quality_data)} symbols "
        f"(was {len(filtered_data)})"
    )

    market_data, dropped_common_history = align_with_min_history(pre_quality_data)
    if dropped_common_history:
        print("- Excluded during date-alignment:")
        for s, reason in sorted(dropped_common_history.items()):
            print(f"  {s}: {reason}")
    if len(market_data) < 2:
        raise RuntimeError("Not enough symbols remain after quote filtering.")

    # Eligible-universe diagnostics at the same selection date used by robustness random sampling.
    common = None
    for df in market_data.values():
        common = df.index if common is None else common.intersection(df.index)
    common_dates = list(common.sort_values()) if common is not None else []
    if len(common_dates) >= 2:
        select_idx = min(
            max(int(config.data.universe_quality_lookback_days) - 1, 0),
            len(common_dates) - 2,
        )
        selection_date = common_dates[select_idx]
    else:
        selection_date = common_dates[0] if common_dates else None

    print("\nEligible Universe Diagnostics")
    print(
        f"- All symbols after quote filtering ({len(market_data)}): "
        f"{', '.join(sorted(market_data.keys()))}"
    )
    if selection_date is not None:
        quality_snapshot = data_manager.universe_quality_snapshot(market_data, selection_date)
        passing = sorted(
            [s for s, m in quality_snapshot.items() if bool(m.get("eligible", False))]
        )
        failing = sorted([s for s in market_data.keys() if s not in set(passing)])
        selected_info = data_manager.select_universe_at_date(
            market_data=market_data,
            date=selection_date,
            target_size=int(config.data.universe_size),
        )
        actual_universe_size = int(len(selected_info.get("selected_symbols", set())))
        print(f"- Selection date: {selection_date.date()}")
        print(f"- Passing quality filters ({len(passing)}): {', '.join(passing) if passing else 'none'}")
        print(f"- Failing quality filters ({len(failing)}): {', '.join(failing) if failing else 'none'}")
        print(
            f"- Universe size tracking: total_loaded={len(loaded_data)}, "
            f"quote_filtered={len(market_data)}, passing_filters={len(passing)}, "
            f"target_universe_size={int(config.data.universe_size)}, "
            f"actual_universe_size={actual_universe_size}"
        )
        if len(passing) < 10:
            print("WARNING: UNIVERSE TOO SMALL FOR ROBUST STRATEGY")
        fail_reason_counts: Counter[str] = Counter()
        if failing:
            print("- Failed symbols with reasons:")
            for s in failing:
                reasons = quality_snapshot.get(s, {}).get("reasons", [])
                reason_txt = ",".join(reasons) if reasons else "NO_REASON_RECORDED"
                print(f"  {s}: {reason_txt}")
                for r in reasons:
                    fail_reason_counts[str(r)] += 1
        print(
            f"- Eligible count: {len(passing)} | "
            f"quality_random_universe_size: {config.backtest.quality_random_universe_size} | "
            f"quality_random_universe_min_size: {config.backtest.quality_random_universe_min_size}"
        )
        if len(passing) <= int(config.backtest.quality_random_universe_size):
            print("WARNING: Eligible pool is close to sampling size; diversity may be limited.")
        if fail_reason_counts:
            most = fail_reason_counts.most_common(3)
            print(
                "- Most restrictive filters: "
                + ", ".join(f"{k} ({v})" for k, v in most)
            )
            bottleneck = most[0][0]
            if "HISTORY" in bottleneck:
                print("- Relaxation diagnostic: history is the primary bottleneck.")
            elif "VOLUME" in bottleneck or "LIQUIDITY" in bottleneck:
                print("- Relaxation diagnostic: volume/liquidity is the primary bottleneck.")
            elif "VOLATILITY" in bottleneck:
                print("- Relaxation diagnostic: volatility cap is the primary bottleneck.")

        # Per-symbol asset inclusion diagnostics (history/volume/filter pass/fail).
        rows = []
        raw_quality_snapshot = data_manager.universe_quality_snapshot(loaded_data, selection_date)
        all_symbols = sorted(set(loaded_data.keys()))
        for sym in all_symbols:
            quote = data_manager.detect_quote_currency(sym)
            in_quote_filtered = sym in market_data
            in_active = sym in set(selected_info.get("selected_symbols", set()))
            snap = quality_snapshot.get(sym, raw_quality_snapshot.get(sym, {}))
            eligible = bool(snap.get("eligible", False)) if in_quote_filtered else False
            reasons = list(snap.get("reasons", [])) if in_quote_filtered else [excluded_symbols.get(sym, "EXCLUDED_PRE_FILTER")]
            reason_txt = "|".join(reasons) if reasons else "PASSED"
            first_date = loaded_data[sym].index.min().date() if sym in loaded_data and not loaded_data[sym].empty else None
            last_date = loaded_data[sym].index.max().date() if sym in loaded_data and not loaded_data[sym].empty else None
            rows.append(
                {
                    "symbol": sym,
                    "quote": quote,
                    "first_date": str(first_date) if first_date else "",
                    "last_date": str(last_date) if last_date else "",
                    "history_bars": int(snap.get("history_bars", len(loaded_data.get(sym, [])))),
                    "avg_volume": float(snap.get("avg_volume", 0.0)),
                    "avg_dollar_volume": float(snap.get("avg_dollar_volume", 0.0)),
                    "median_dollar_volume": float(snap.get("median_dollar_volume", 0.0)),
                    "quality_score": float(snap.get("quality_score", 0.0)),
                    "quote_filter_pass": bool(in_quote_filtered),
                    "quality_filter_pass": bool(eligible),
                    "included_in_active_universe": bool(in_active),
                    "exclusion_reason": reason_txt if not in_active else "INCLUDED",
                }
            )
        inclusion_df = pd.DataFrame(rows).sort_values(["included_in_active_universe", "quality_filter_pass", "symbol"], ascending=[False, False, True])
        inclusion_path = root / "asset_inclusion_diagnostics.csv"
        inclusion_df.to_csv(inclusion_path, index=False)
        print(f"- Asset inclusion diagnostics saved: {inclusion_path.name}")
        compact_cols = ["symbol", "history_bars", "avg_dollar_volume", "quality_filter_pass", "included_in_active_universe", "exclusion_reason"]
        print(inclusion_df[compact_cols].to_string(index=False))

    # Baseline (before robustness-aware stabilization controls).
    baseline_cfg = deepcopy(config)
    baseline_cfg.strategy.sleeve_stability_penalties = {}
    baseline_cfg.strategy.enforce_sleeve_caps = False
    baseline_cfg.strategy.sleeve_diversification_blend = 0.0
    baseline_cfg.risk.asset_stability_penalty_enabled = False
    baseline_cfg.risk.asset_stability_penalties = {}

    base_engine = BacktestEngine(baseline_cfg)
    base_result = base_engine.run(market_data)
    base_robustness_df = run_robustness_suite(baseline_cfg, market_data)

    # Diagnose largest source of CAGR dispersion.
    category_dispersion = cagr_dispersion_by_category(base_robustness_df)
    spread_driver = largest_cagr_spread_category(base_robustness_df)

    # Derive soft penalties from robustness sensitivity + sleeve concentration.
    asset_penalties = derive_asset_stability_penalties(base_robustness_df, config)
    sleeve_shares = base_result.diagnostics.get("strategy_contribution_proxy_share_abs", {})
    sleeve_penalties: Dict[str, float] = {}
    sleeve_trigger = float(config.strategy.sleeve_concentration_trigger)
    sleeve_strength = float(config.strategy.sleeve_penalty_strength)
    sleeve_floor = float(config.strategy.sleeve_penalty_min)
    for sleeve, share in sleeve_shares.items():
        share = float(share)
        if share <= sleeve_trigger:
            continue
        # Soft penalty only for concentrated sleeves.
        severity = min(1.0, (share - sleeve_trigger) / max(1.0 - sleeve_trigger, 1e-9))
        sleeve_penalties[sleeve] = max(sleeve_floor, 1.0 - sleeve_strength * severity)

    # Stabilized run (same strategies, safer concentration profile).
    config.risk.asset_stability_penalties = asset_penalties
    config.strategy.sleeve_stability_penalties = sleeve_penalties

    engine = BacktestEngine(config)
    result = engine.run(market_data)

    print("Backtest Results")
    print(f"Total Return: {result.total_return:.2%}")
    print(f"CAGR: {result.cagr:.2%}")
    print(f"Sharpe Ratio: {result.sharpe:.3f}")
    print(f"Max Drawdown: {result.max_drawdown:.2%}")
    print(f"Ending Equity: {result.equity_curve.iloc[-1]:,.2f}")

    if result.diagnostics:
        d = result.diagnostics
        print("\nDiagnostics")
        print(f"Trade Count: {d.get('trade_count', 0)}")
        print(f"Turnover: {d.get('turnover', 0.0):.3f}x")
        print(f"Avg Holding Period (days): {d.get('average_holding_period_days', 0.0):.2f}")
        print(f"Win Rate: {d.get('win_rate', 0.0):.2%}")
        print(f"Avg Gain: {d.get('average_gain', 0.0):,.2f}")
        print(f"Avg Loss: {d.get('average_loss', 0.0):,.2f}")
        print(f"Avg Gross Exposure: {d.get('average_exposure', 0.0):.2f}x")
        print(f"Fee Impact: {d.get('fee_impact', 0.0):,.2f}")
        print(f"Slippage Impact: {d.get('slippage_impact', 0.0):,.2f}")
        strat_proxy = d.get("strategy_contribution_proxy_pnl", {})
        if strat_proxy:
            print("Strategy Contribution (Proxy PnL):")
            for k, v in strat_proxy.items():
                print(f"  {k}: {v:,.2f}")
        print(
            "Yield Diagnostics: "
            f"yield_contribution={float(strat_proxy.get('yield', 0.0)):,.2f}, "
            f"yield_allocation_used={float(d.get('effective_strategy_weights', {}).get('yield', 0.0)):.2%}, "
            f"yield_enabled={d.get('yield_enabled', True)}, "
            f"yield_max_allocation={float(d.get('yield_max_allocation', 0.0)):.2%}"
        )
        print(
            "Yield Target Diagnostics: "
            f"preferred_targets={d.get('yield_preferred_targets', [])}, "
            f"fallback_to_quality_assets={config.strategy.yield_fallback_to_quality_assets}"
        )
        print(
            "Pair Selection Diagnostics: "
            f"selected_pairs={d.get('pair_selected_pairs', [])}, "
            f"pair_max_pairs={config.strategy.pair_max_pairs}, "
            f"pair_selection_lookback={config.strategy.pair_selection_lookback}"
        )

    if result.warnings:
        print("\nAudit Warnings")
        for w in result.warnings:
            print(f"- {w}")

    robustness_df = run_robustness_suite(config, market_data)
    if not robustness_df.empty:
        if not category_dispersion.empty:
            print("\nCAGR Dispersion by Robustness Category (Baseline)")
            print(
                category_dispersion[
                    ["category", "best_cagr", "worst_cagr", "median_cagr", "spread"]
                ].to_string(index=False)
            )
            print(
                "Largest CAGR Spread Driver: "
                f"{spread_driver['category']} "
                f"(best {spread_driver['best_cagr']:.2%}, "
                f"worst {spread_driver['worst_cagr']:.2%}, "
                f"median {spread_driver['median_cagr']:.2%}, "
                f"spread {spread_driver['spread']:.2%})"
            )

        print("\nDrop-One-Asset Results")
        drop_df = robustness_df[robustness_df["category"] == "drop_one"][
            ["case", "total_return", "cagr", "sharpe", "max_drawdown"]
        ]
        if not drop_df.empty:
            print(drop_df.to_string(index=False))

        print("\nRandomized-Universe Results")
        rnd_df = robustness_df[robustness_df["category"] == "random_universe"][
            [
                "case",
                "selected_assets",
                "total_return",
                "cagr",
                "sharpe",
                "max_drawdown",
                "universe_label",
                "yield_enabled_in_case",
            ]
        ]
        if not rnd_df.empty:
            print(rnd_df.to_string(index=False))
            if (rnd_df["universe_label"] == "LOW_QUALITY_UNIVERSE").any():
                print("WARNING: LOW_QUALITY_UNIVERSE detected in quality-filtered random tests.")
            uniq_q = int(rnd_df["selected_assets"].nunique()) if "selected_assets" in robustness_df.columns else 0
            req_q = int(config.backtest.quality_random_universe_trials)
            ok_q = int(rnd_df["case"].nunique())
            dup_q = max(0, req_q - ok_q)
            if "duplicate_trials_skipped" in robustness_df.columns:
                dup_vals = robustness_df[
                    robustness_df["category"] == "random_universe"
                ]["duplicate_trials_skipped"].dropna()
                if not dup_vals.empty:
                    dup_q = int(dup_vals.max())
            print(
                "Quality Random Unique Diagnostics: "
                f"requested_trials={req_q}, successful_unique_trials={ok_q}, "
                f"duplicate_trials_skipped={dup_q}, unique_universes={uniq_q}"
            )

        print("\nRandomized-Universe Raw Stress Results")
        rnd_raw_df = robustness_df[robustness_df["category"] == "random_universe_raw"][
            [
                "case",
                "selected_assets",
                "total_return",
                "cagr",
                "sharpe",
                "max_drawdown",
                "universe_label",
                "yield_enabled_in_case",
            ]
        ]
        if not rnd_raw_df.empty:
            print(rnd_raw_df.to_string(index=False))
            uniq_raw = int(rnd_raw_df["selected_assets"].nunique()) if "selected_assets" in robustness_df.columns else 0
            print(
                "Raw Random Unique Diagnostics: "
                f"requested_trials={config.backtest.random_universe_trials}, "
                f"successful_unique_trials={int(rnd_raw_df['case'].nunique())}, "
                f"unique_universes={uniq_raw}"
            )

        q_summary = random_universe_summary(robustness_df, "random_universe")
        raw_summary = random_universe_summary(robustness_df, "random_universe_raw")
        print("\nRandom Universe Comparison (Before vs After Quality Filter)")
        print(
            f"Raw Stress (before filter): best {raw_summary['best_cagr']:.2%}, "
            f"worst {raw_summary['worst_cagr']:.2%}, median {raw_summary['median_cagr']:.2%}, "
            f"spread {raw_summary['spread']:.2%}"
        )
        print(
            f"Quality Filtered (after): best {q_summary['best_cagr']:.2%}, "
            f"worst {q_summary['worst_cagr']:.2%}, median {q_summary['median_cagr']:.2%}, "
            f"spread {q_summary['spread']:.2%}"
        )

        worst_raw = worst_random_universe_diagnostics(robustness_df, "random_universe_raw", top_n=3)
        worst_quality = worst_random_universe_diagnostics(robustness_df, "random_universe", top_n=3)
        if not worst_quality.empty:
            print("\nWorst Quality-Filtered Random Universes")
            print(worst_quality.to_string(index=False))
        if not worst_raw.empty:
            print("\nWorst Raw Random Universes (lowest-quality failures)")
            print(worst_raw.to_string(index=False))

        print("\nParameter-Variation Results")
        par_df = robustness_df[robustness_df["category"] == "parameter_variation"][
            ["case", "total_return", "cagr", "sharpe", "max_drawdown"]
        ]
        if not par_df.empty:
            print(par_df.to_string(index=False))

        print("\nRebalance-Offset Results")
        rebal_df = robustness_df[robustness_df["category"] == "rebalance_offset"][
            ["case", "total_return", "cagr", "sharpe", "max_drawdown"]
        ]
        if not rebal_df.empty:
            print(rebal_df.to_string(index=False))
            rebal_diag = rebalance_offset_diagnostics(robustness_df)
            print(
                "Rebalance Offset Diagnostics: "
                f"worst={rebal_diag.get('worst_offset_case')} "
                f"(CAGR {float(rebal_diag.get('worst_offset_cagr', 0.0)):.2%}, "
                f"Sharpe {float(rebal_diag.get('worst_offset_sharpe', 0.0)):.3f}), "
                f"spread={float(rebal_diag.get('cagr_spread', 0.0)):.2%}, "
                f"timing_dependency={'YES' if rebal_diag.get('timing_dependency', False) else 'NO'}"
            )

        print("\nFee-Slippage Sensitivity Results")
        fee_df = robustness_df[robustness_df["category"] == "fee_slippage_sensitivity"][
            ["case", "total_return", "cagr", "sharpe", "max_drawdown"]
        ]
        if not fee_df.empty:
            print(fee_df.to_string(index=False))

        print("\nPair-Focused Robustness Comparison")
        pair_df = robustness_df[robustness_df["category"] == "pair_focus"][
            ["case", "total_return", "cagr", "sharpe", "max_drawdown", "pair_selected_pairs"]
        ]
        if not pair_df.empty:
            print(pair_df.to_string(index=False))

        base_metrics = {
            "cagr": result.cagr,
            "sharpe": result.sharpe,
            "max_drawdown": result.max_drawdown,
        }
        baseline_metrics = {
            "cagr": base_result.cagr,
            "sharpe": base_result.sharpe,
            "max_drawdown": base_result.max_drawdown,
        }
        baseline_summary = stability_summary(baseline_metrics, base_robustness_df, baseline_cfg)
        summary = stability_summary(base_metrics, robustness_df, config)
        print("\nStability Summary")
        print(f"Robustness Score: {summary['robustness_score']:.1f}/100")
        print(
            f"Base vs Robust Median CAGR: {result.cagr:.2%} vs {summary['median_cagr']:.2%}"
        )
        print(
            f"Base vs Robust Median Sharpe: {result.sharpe:.3f} vs {summary['median_sharpe']:.3f}"
        )
        print(
            "Base vs Robust Median Max DD: "
            f"{result.max_drawdown:.2%} vs {summary['median_max_drawdown']:.2%}"
        )
        print(
            f"Worst Robust Drawdown: {summary['worst_drawdown']:.2%} | "
            f"Robust CAGR Spread: {summary['cagr_spread']:.2%} | "
            f"Weak Scenarios: {summary['weak_scenarios']}/{summary['scenario_count']}"
        )
        print("\nStability Delta (Original vs New)")
        print(f"Original Base CAGR: {base_result.cagr:.2%}")
        print(f"New Base CAGR: {result.cagr:.2%}")
        print(f"Original Robust CAGR Spread: {baseline_summary['cagr_spread']:.2%}")
        print(f"New Robust CAGR Spread: {summary['cagr_spread']:.2%}")
        print(
            "Fragility Warning (Original vs New): "
            f"{'YES' if (fragility_warning(base_robustness_df, baseline_cfg) or baseline_summary['warnings']) else 'NO'}"
            f" -> {'YES' if (fragility_warning(robustness_df, config) or summary['warnings']) else 'NO'}"
        )

        if fragility_warning(robustness_df, config) or summary["warnings"]:
            print("\nWARNING: Strategy may be overfit or fragile")
            for w in summary["warnings"]:
                print(f"- {w}")
        print(
            "\nFragility Warning Status: "
            f"{'TRIGGERED' if (fragility_warning(robustness_df, config) or summary['warnings']) else 'NOT_TRIGGERED'}"
        )

        if "yield_enabled_in_case" in robustness_df.columns:
            any_disabled = bool((robustness_df["yield_enabled_in_case"] == False).any())
            print(
                "\nYield Robustness Setting: "
                f"yield_disable_in_random_universe_tests={config.backtest.yield_disable_in_random_universe_tests}, "
                f"effective_disabled_in_cases={any_disabled}"
            )

    # Yield vs no-yield comparison on filtered base universe.
    no_yield_cfg = deepcopy(config)
    no_yield_cfg.strategy.yield_enabled = False
    no_yield_result = BacktestEngine(no_yield_cfg).run(market_data)
    print("\nYield vs No-Yield (Base Universe)")
    print(
        f"With Yield: CAGR {result.cagr:.2%}, Sharpe {result.sharpe:.3f}, MaxDD {result.max_drawdown:.2%}"
    )
    print(
        f"No Yield: CAGR {no_yield_result.cagr:.2%}, "
        f"Sharpe {no_yield_result.sharpe:.3f}, MaxDD {no_yield_result.max_drawdown:.2%}"
    )

    if asset_penalties:
        print("\nApplied Asset Stability Penalties")
        for sym, p in sorted(asset_penalties.items()):
            if p < 0.999:
                print(f"- {sym}: {p:.3f}")
    if sleeve_penalties:
        print("\nApplied Sleeve Stability Penalties")
        for sleeve, p in sorted(sleeve_penalties.items()):
            if p < 0.999:
                print(f"- {sleeve}: {p:.3f}")

    top_asset_share = float(result.diagnostics.get("top_asset_contribution_share_abs", 0.0))
    if top_asset_share > config.backtest.fragility_asset_dominance_warn:
        print(
            "\nWARNING: Asset contribution is concentrated; one asset appears dominant."
        )


if __name__ == "__main__":
    main()
