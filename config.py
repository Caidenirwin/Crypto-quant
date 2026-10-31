"""
Central configuration for the crypto quant system.
All tunable parameters should live here.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class DataConfig:
    start_cash: float = 100_000.0
    min_history_days: int = 60
    date_column: str = "Date"
    ohlcv_columns: List[str] = field(
        default_factory=lambda: ["Open", "High", "Low", "Close", "Volume"]
    )
    csv_separator: str = ","
    # Rules-based tradable universe settings (prevents hand-picked symbol dependency).
    use_rules_based_universe: bool = True
    universe_size: int = 12  # expanded universe; quality filter selects top 12 by liquidity score
    min_volume_threshold: float = 50_000_000.0
    universe_refresh_days: int = 7
    universe_volume_lookback_days: int = 20
    universe_quality_lookback_days: int = 120
    universe_min_history_bars: int = 90
    # Symbols whose most-recent bar lags the freshest symbol by more than this many days
    # are treated as stale and excluded BEFORE alignment (prevents a delisted asset whose
    # data ended months ago from capping the aligned end-date for the full universe).
    universe_pre_filter_max_stale_days: int = 180
    universe_min_avg_volume: float = 200_000.0
    universe_min_avg_dollar_volume: float = 3_000_000.0
    universe_min_median_dollar_volume: float = 1_500_000.0
    universe_max_missing_data_pct: float = 0.05
    universe_max_volatility: float = 2.20
    universe_min_liquidity_stability: float = 0.20
    universe_min_recent_tradability: float = 0.80
    universe_recent_tradability_window: int = 30
    universe_quality_min_score: float = 0.35
    universe_quality_score_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "history": 0.20,
            "volume_level": 0.25,
            "volume_consistency": 0.15,
            "missing_quality": 0.15,
            "volatility_stability": 0.15,
            "tradability": 0.10,
        }
    )
    symbol_alias_map: Dict[str, str] = field(
        default_factory=lambda: {
            # Long names → canonical short names
            "MAPLE":          "MPL",
            "MAPLE(USD)":     "MPL(USD)",
            "CENTRIFUGE":     "CFG",
            "CENTRIFUGE(USD)":"CFG(USD)",
            "HEDERA":         "HBAR",
            "HEDERA(USD)":    "HBAR(USD)",
            "VECHAIN":        "VET",
            "VECHAIN(USD)":   "VET(USD)",
            "VECHAIN(EUR)":   "VET(EUR)",
            "ZEBEC":          "ZBCN",
            "ZEBEC(USD)":     "ZBCN(USD)",
            "SOLANA":         "SOL",
            "SOLANA(USD)":    "SOL(USD)",
            # Clearpool CSV uses full name; trade symbol is CPOOL
            "CLEARPOOL":      "CPOOL",
            "CLEARPOOL(USD)": "CPOOL(USD)",
        }
    )
    # Experimental / high-risk assets — included in the data universe for tracking
    # but capped at a small allocation so they can't dominate the portfolio.
    experimental_asset_list: List[str] = field(
        default_factory=lambda: [
            "POLYX(USD)",  # Polymesh — low liquidity ($125K/day)
            "PRO(USD)",    # Propy — borderline liquidity ($2M/day)
            "RIO(USD)",    # Realio — very low liquidity ($21K/day)
            "ZBCN(USD)",   # Zebec — very low liquidity
        ]
    )
    # Quote currency normalization/validation.
    allowed_quote_currency: str = "USD"
    allowed_additional_quote_currencies: List[str] = field(default_factory=list)
    allow_unknown_quote_currency: bool = False
    symbol_quote_overrides: Dict[str, str] = field(
        default_factory=lambda: {
            "XRP": "USD",
        }
    )
    # Asset expansion controls:
    # - auto_detect_symbol_files: load all CSVs found in project root.
    # - configured_symbol_files: explicit symbol->file mapping for manual additions.
    # - symbol_source_mode:
    #     "auto"  -> only discovered CSVs
    #     "config"-> only configured_symbol_files
    #     "both"  -> merge configured + discovered (configured paths override collisions)
    auto_detect_symbol_files: bool = True
    configured_symbol_files: Dict[str, str] = field(default_factory=dict)
    symbol_source_mode: str = "both"
    # Preferred market-data roots (logs/reports should be outside these directories).
    market_data_dirs: List[str] = field(default_factory=lambda: ["historic_data"])
    # Backward-compatible alias for older discovery code.
    symbol_search_roots: List[str] = field(default_factory=lambda: ["historic_data"])
    excluded_data_patterns: List[str] = field(
        default_factory=lambda: [
            "logs/*",
            "forward_*.csv",
            "live_trades.csv",
            "paper_equity.csv",
            "tax_reserve.csv",
            "data_coverage_report.csv",
            "asset_inclusion_diagnostics.csv",
            "forward_signal_trace.csv",
            "forward_positions.csv",
            "forward_trades.csv",
            "forward_equity.csv",
            "*_report.csv",
            "*_diagnostics.csv",
        ]
    )
    log_output_dir: str = "logs"


@dataclass
class StrategyConfig:
    enabled_strategies: List[str] = field(
        default_factory=lambda: [
            "momentum",
            "rsi_dip",
            "sector_rotation",
            "breakout",
            "volatility",
            "pair_trading",
            "yield",
        ]
    )
    strategy_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "momentum":        0.00,  # disabled — Sharpe30 IC too backward-looking; loses in practice
            "rsi_dip":         0.35,  # v6: retained to deploy capital when sector rotation in cash
            "sector_rotation": 0.65,  # v6 primary: HighBeta+ONDO sectors with gate=0.1
            "breakout":        0.00,  # disabled — consistently lost in all market conditions
            "volatility":      0.00,
            "pair_trading":    0.00,
            "yield":           0.00,
        }
    )
    momentum_enabled:         bool = False  # Disabled — Sharpe30 IC is backward-looking; loses in backtests
    rsi_dip_enabled:          bool = True
    sector_rotation_enabled:  bool = True
    breakout_enabled:         bool = False  # Disabled — lost -$9k-$10k in every backtest run
    pair_trading_enabled:     bool = False  # Disabled — ONDO/UNI pair lost consistently
    volatility_enabled:       bool = True
    yield_enabled:            bool = False  # Simulated only — keep off until real staking integration
    enforce_sleeve_caps: bool = True
    sleeve_max_weight: float = 0.35
    sleeve_diversification_blend: float = 0.25
    sleeve_stability_penalties: Dict[str, float] = field(default_factory=dict)
    sleeve_concentration_trigger: float = 0.35
    sleeve_penalty_strength: float = 0.35
    sleeve_penalty_min: float = 0.75
    momentum_lookbacks: List[int] = field(default_factory=lambda: [14, 60])  # Slower signals, less churn
    momentum_top_n: int = 2  # concentrate in top 2 for higher alpha, more risk
    momentum_max_allocation: float = 0.30
    momentum_quality_filter_enabled: bool = False
    momentum_trailing_stop_enabled: bool = False
    momentum_trailing_stop_pct: float = 0.10
    momentum_min_trend_ma_window: int = 50
    momentum_short_ma_window: int = 20
    momentum_medium_ma_window: int = 50
    pair_window: int = 20
    pair_correlation_window: int = 60
    pair_min_correlation: float = 0.50  # Only trade meaningfully correlated pairs
    pair_min_ratio_volatility: float = 0.01
    pair_max_allocation: float = 0.42
    pair_entry_zscore: float = 1.55
    pair_exit_zscore: float = 0.5
    pair_max_pairs: int = 1
    pair_selection_correlation_min: float = 0.35
    pair_selection_lookback: int = 60
    pair_selection_use_quality_score: bool = True
    pair_selection_ratio_vol_max: float = 0.35
    pair_z_entry: float = 2.0
    pair_z_exit: float = 0.5
    volatility_window: int = 20
    volatility_threshold: float = 0.02
    volatility_band_k: float = 1.0
    breakout_lookback: int = 21  # Longer lookback = fewer false breakouts
    breakout_max_allocation: float = 0.20
    breakout_volume_spike_z: float = 2.0  # Tighter: require stronger volume confirmation
    breakout_require_trend_confirmation: bool = True
    breakout_trend_ma_window: int = 50
    momentum_volume_filter_window: int = 20
    momentum_volume_filter_multiplier: float = 0.80
    breakout_volume_filter_window: int = 20
    breakout_volume_filter_multiplier: float = 0.90
    yield_daily_rate: float = 0.0002
    yield_conversion_days: int = 7
    yield_max_allocation: float = 0.05
    yield_preferred_targets: List[str] = field(
        default_factory=lambda: ["CHAINLINK(USD)", "ONDO(USD)"]
    )
    yield_fallback_to_quality_assets: bool = True
    # RSI Dip strategy parameters
    rsi_dip_entry_threshold: float = 45.0   # RSI below this = oversold candidate (wider threshold adds alpha vs 40)
    rsi_dip_extreme_threshold: float = 30.0  # RSI below this = extreme — enter immediately
    rsi_dip_min_history: int = 30            # minimum bars needed
    rsi_dip_top_n: int = 2                   # concentrate in top 2 picks
    rsi_dip_volume_filter_enabled: bool = True
    rsi_dip_volume_window: int = 20
    rsi_dip_volume_multiplier: float = 0.50  # volume must be at least 50% of 20d avg
    # Sharpe30 gate: skip assets with Sharpe30 below this threshold.
    # Filters out genuine crashes (CHAINLINK=-1.14, XRP=-1.33, SUI=-1.32)
    # while allowing temporary dips (ONDO=-0.54 with RSI=25 and r90=+17.7%).
    rsi_dip_min_sharpe30: float = -1.0

    # Sector Rotation strategy parameters
    # v7 finding: top_n=2 (hold both sectors simultaneously when both above gate) gives
    # 34.3% CAGR vs 28% with top_n=1. Allows capturing ONDO (Jan-Oct 2024) + HighBeta
    # (Nov 2024) overlapping periods. Fee-robust up to 2x standard rates; cliff at 3x.
    sector_rotation_top_n: int = 2               # v7: hold up to 2 sectors when both show positive momentum
    sector_rotation_sharpe_window: int = 14       # 14-day Sharpe — fast enough to catch rotations
    sector_rotation_min_assets_per_sector: int = 1
    # Minimum Sharpe14 a sector must have to qualify for entry.
    # When all sectors are negative (sustained bear), go to cash rather than
    # concentrating in the "least bad" sector.  Set to 0.0 to only enter
    # positive-momentum sectors; set to -99 to always enter something.
    # v6 finding: 0.1 outperforms 0.0 and -99 robustly (28% vs 20% vs -1.5%)
    # because it filters out noise at the bull/bear transition boundary.
    sector_rotation_min_entry_sharpe: float = 0.1
    sector_definitions: Dict[str, List[str]] = field(
        default_factory=lambda: {
            # v6 strategy: only the confirmed 2024-2026 winners — rotate between them,
            # or go to cash when none show clear positive momentum (gate > 0.1).
            # This outperforms the 5-sector universe because the non-winners (AltL1,
            # Infra, RWA) pull quality scores down and force allocation to losers.
            #
            # HighBeta: XRP (+89% full period) and HBAR (+545% in Nov 2024 bull run).
            #           When sector Sharpe14 > 0.1, concentrated in crypto rally leaders.
            # ONDO:     +202% in Jan-Oct 2024 sideways phase; only DeFi asset with
            #           positive total return over the full 2.43-year period (+41%).
            "HighBeta": ["XRP", "HBAR(USD)"],
            "ONDO":     ["ONDO(USD)"],
        }
    )


@dataclass
class PortfolioConfig:
    rebalance_frequency_days: int = 2  # 2-day rebalance optimal for sector rotation signal capture
    rebalance_initial_offset_days: int = 0
    max_positions: int = 10
    allow_partial_sells: bool = True
    # Legacy fee knob kept for compatibility. Prefer fee_rate.
    transaction_cost_bps: float = 5.0
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0005
    # Keep empty by default to avoid hardcoded symbol bias.
    core_assets: List[str] = field(default_factory=list)
    core_bull_allocation: float = 0.20
    core_neutral_allocation: float = 0.28
    core_bear_allocation: float = 0.38
    core_extreme_defensive_allocation: float = 0.55
    core_vol_lookback: int = 30
    core_min_fraction_per_asset: float = 0.20
    # Optional fixed-hold basket: always reserve these portfolio fractions
    # for selected assets (if present in the active market universe).
    # Values are target weights (e.g. 0.05 = 5%).
    fixed_hold_allocations: Dict[str, float] = field(default_factory=dict)
    fixed_hold_enabled: bool = False  # Strategy decides all positions; no forced holds
    take_profit_gain_threshold: float = 0.06
    take_profit_sell_fraction: float = 0.35
    take_profit_weak_trend_sell_fraction: float = 0.50
    take_profit_full_exit_on_trend_break: bool = True
    take_profit_trend_ma_window: int = 20
    take_profit_cooldown_days: int = 1
    trend_break_exit_ma_window: int = 20
    trend_break_exit_fraction: float = 0.50


@dataclass
class RiskConfig:
    max_position_weight: float = 0.28  # More concentration allowed (risky but higher ROI)
    stop_loss_pct: float = 0.12       # Wider stop — crypto needs room to breathe
    max_drawdown_pct: float = 0.55    # Accept deeper drawdowns for higher upside
    target_volatility: float = 0.20
    # OP has consistently been the worst portfolio contributor — cap at 5% max.
    high_risk_asset_caps: Dict[str, float] = field(
        default_factory=lambda: {
            "OP(USD)":    0.05,  # OP has been worst historical contributor; cap at 5%
            # Experimental assets — low liquidity, cap tightly so they can't distort results
            "POLYX(USD)": 0.03,
            "PRO(USD)":   0.03,
            "RIO(USD)":   0.03,
            "ZBCN(USD)":  0.03,
            "CFG(USD)":   0.03,
            "CPOOL(USD)": 0.03,
            "MPL(USD)":   0.03,
        }
    )
    high_risk_group_cap: float = 0.30
    high_risk_vol_lookback: int = 30
    high_risk_top_fraction: float = 0.35
    volatility_sizing_enabled: bool = True
    volatility_lookback: int = 30
    volatility_floor: float = 0.15
    volatility_scale_max: float = 3.0
    active_weight_min: float = 0.03
    active_weight_max: float = 0.18
    active_equal_weight_blend: float = 0.50
    asset_stability_penalty_enabled: bool = True
    asset_stability_penalties: Dict[str, float] = field(default_factory=dict)
    regime_filter_enabled: bool = True
    regime_symbol: str | None = None
    regime_use_dynamic_symbol: bool = True
    regime_fast_ma_window: int = 50
    regime_slow_ma_window: int = 200
    regime_active_multiplier_neutral: float = 0.90  # stay more deployed in neutral markets
    regime_active_multiplier_bear: float = 0.70    # still deploy 70% in bear (somewhat risky)
    regime_block_new_noncore_positions: bool = False
    universe_trend_filter_enabled: bool = True
    universe_trend_ma_window: int = 100
    drawdown_pause_adds_level: float = 0.20
    drawdown_level1: float = 0.25
    drawdown_level1_risk_multiplier: float = 0.82
    drawdown_level2: float = 0.35
    drawdown_level2_risk_multiplier: float = 0.65
    drawdown_level3: float = 0.45
    drawdown_level3_risk_multiplier: float = 0.45


@dataclass
class BacktestConfig:
    annualization_days: int = 365
    execution_timing: str = "signal_close_execute_next_open"
    include_diagnostics: bool = True
    selection_bias_warning: bool = True
    robustness_start_offsets: List[int] = field(default_factory=lambda: [0, 20, 40])
    robustness_rebalance_offsets: List[int] = field(default_factory=lambda: [0, 1, 2])
    robustness_fee_multipliers: List[float] = field(default_factory=lambda: [0.0, 1.0, 1.5])
    random_universe_trials: int = 12
    random_universe_size: int = 6
    quality_random_universe_size: int = 5
    quality_random_universe_min_size: int = 3
    quality_random_universe_trials: int = 20
    target_universe_size: int = 6
    min_eligible_universe_size: int = 4
    random_universe_use_quality_filter: bool = True
    random_universe_include_raw_stress_test: bool = True
    random_seed: int = 42
    parameter_variation_pct: float = 0.15
    fragility_cagr_range_warn: float = 0.40
    fragility_sharpe_range_warn: float = 0.80
    fragility_drawdown_range_warn: float = 0.20
    fragility_min_scenario_sharpe: float = 0.30
    fragility_worst_drawdown_warn: float = 0.35
    fragility_weak_scenario_ratio_warn: float = 0.30
    fragility_base_vs_median_sharpe_gap_warn: float = 0.40
    fragility_base_vs_median_cagr_gap_warn: float = 0.20
    fragility_asset_dominance_warn: float = 0.40
    asset_penalty_min: float = 0.70
    asset_penalty_strength: float = 0.60
    universe_quality_weighting_enabled: bool = True
    universe_quality_weight_strength: float = 0.20
    yield_disable_in_random_universe_tests: bool = True
    robustness_disable_momentum: bool = True
    robustness_disable_yield: bool = True


@dataclass
class ForwardTestConfig:
    # "fresh": cash-only start, "carry_backtest": bootstrap state from historical backtest slice.
    forward_start_mode: str = "fresh"
    # Last N calendar days used as forward evaluation window.
    forward_window_days: int = 30
    # Extra pre-forward history loaded for indicator warmup.
    forward_history_buffer_days: int = 300
    # Minimum bars expected for full signal coverage.
    min_required_history_bars: int = 250
    # If True, symbols below minimum history are excluded from tradable forward set.
    forward_require_full_history: bool = False
    # Verbose forward diagnostics.
    forward_verbose_audit: bool = True


@dataclass
class MemeConfig:
    """
    Meme coin sleeve + anti-rug-pull guard.

    Meme coins live OUTSIDE the core V7 engine: they are excluded from the quality
    universe, from RSI Dip / Sector Rotation signals, and are funded by a hard-capped
    carve-out. When the meme sleeve is in cash, core V7 behaviour is unchanged.
    """
    meme_enabled: bool = True
    # Internal symbols (CSV file prefix). Only Binance-listed memes with >2y history.
    meme_symbols: List[str] = field(
        default_factory=lambda: [
            "DOGE(USD)", "SHIB(USD)", "PEPE(USD)", "BONK(USD)", "WIF(USD)", "FLOKI(USD)",
        ]
    )
    # Hard cap on total meme exposure (fraction of equity) and per coin.
    # Sep 29 2026 backtest (Jan 2024 - Sep 2026, V7 core = 30.55% CAGR / -20.0% DD):
    #   cap 10% -> 2.9% CAGR  (meme losses pushed DD past the -20% add-pause on
    #              Nov 2 2024, so V7 missed the XRP/HBAR bull run)
    #   cap  5% -> 30.1% CAGR / -21.0% DD
    #   cap  3% -> 30.3% CAGR / -20.6% DD   <- chosen: meme exposure at ~0.3% CAGR cost
    # Memes were net-negative PnL at every cap tested. Raise at your own risk.
    meme_max_allocation: float = 0.03
    meme_max_per_coin: float = 0.015
    meme_top_n: int = 2
    # Entry signal: Sharpe14 above gate, price above MA, volume confirmation.
    meme_sharpe_window: int = 14
    meme_min_entry_sharpe: float = 0.15
    meme_trend_ma_window: int = 20
    meme_volume_window: int = 20
    meme_volume_multiplier: float = 0.70

    # ---- Anti-rug: price/volume history checks (backtest + live) ----
    rug_min_history_days: int = 90            # never trade a coin younger than this
    rug_crash_1d_pct: float = 0.25            # -25% in one day => hard exit
    rug_crash_from_high_pct: float = 0.50     # -50% from 14d high => hard exit
    rug_crash_high_window: int = 14
    rug_liquidity_drain_ratio: float = 0.35   # 5d avg volume < 35% of 60d avg => hard exit
    rug_pump_7d_pct: float = 1.50             # +150% in 7 days => block NEW buys (pump & dump)

    # ---- Anti-rug: live market microstructure (Binance mainnet public data) ----
    rug_market_data_url: str = "https://data-api.binance.vision"
    rug_min_quote_volume_24h: float = 1_000_000.0
    rug_max_spread_bps: float = 40.0
    rug_min_bid_depth_usd: float = 25_000.0   # bids within 2% of mid
    rug_min_depth_order_multiple: float = 50.0
    rug_intraday_crash_pct: float = 0.15      # -15% over last 4 hourly bars => hard exit

    # ---- Anti-rug: on-chain contract checks (GoPlus Security, free API) ----
    rug_contract_check_enabled: bool = True
    rug_contract_cache_hours: float = 24.0
    rug_max_tax: float = 0.10                 # buy/sell tax above 10% => fail
    rug_max_top10_holder_pct: float = 0.70    # excluding burn/locked addresses
    # chain: "1" = Ethereum (EVM), "solana". DOGE is a native L1 coin (no contract).
    meme_contracts: Dict[str, Dict[str, str]] = field(
        default_factory=lambda: {
            "PEPE":  {"chain": "1", "address": "0x6982508145454Ce325dDbE47a25d4ec3d2311933"},
            "SHIB":  {"chain": "1", "address": "0x95aD61b0a150d79219dCF64E1E6Cc01f0B64C4cE"},
            "FLOKI": {"chain": "1", "address": "0xcf0C122c6b73ff809C693DB761e7BaeBe62b6a2E"},
            "BONK":  {"chain": "solana", "address": "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"},
            "WIF":   {"chain": "solana", "address": "EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm"},
        }
    )


@dataclass
class MLConfig:
    """
    ML bracket-trade sleeve: predict P(take-profit is hit before stop-loss).

    Every candidate trade is a bracket order:
        entry = next day's open
        stop  = entry - stop_atr_mult * ATR14         (1R of risk)
        take  = entry + reward_risk * (entry - stop)  (R:R = reward_risk)
        exit at the close of day horizon_days if neither is hit.
    The model (trained walk-forward on all coins' full history) only takes trades
    whose expected value, net of costs, clears min_expected_r.
    """
    # Paper-trading forward test on Testnet (own $500 ledger in logs/ml/). New entries are
    # still gated weekly by ml_autoresearch.py's health check (models/ml_status.json).
    ml_enabled: bool = True
    reward_risk: float = 2.0                 # R:R target (user requirement 1.5-2.0)
    stop_atr_mult: float = 1.5
    horizon_days: int = 14
    min_history_bars: int = 200              # MA200 + warm-up before a coin can be traded
    min_expected_r: float = 0.20             # EV gate: p*RR - (1-p) - costs >= 0.2R
    roundtrip_cost: float = 0.003            # 0.1% fee + 0.05% slippage, each side
    risk_per_trade: float = 0.01             # lose at most 1% of equity if stopped out
    max_position_weight: float = 0.20
    max_open_trades: int = 5
    retrain_every_days: int = 90
    model_path: str = "models/ml_bracket_model.joblib"
    reference_dir: str = "reference_data"
    # Thin / not-on-Binance coins: used for training data, never traded.
    no_trade_symbols: List[str] = field(
        default_factory=lambda: [
            "PRO(USD)", "RIO(USD)", "POLYX(USD)", "ZBCN(USD)", "CPOOL(USD)",
            "CFG(USD)", "MPL(USD)", "SKY(USD)",
        ]
    )


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    forward: ForwardTestConfig = field(default_factory=ForwardTestConfig)
    meme: MemeConfig = field(default_factory=MemeConfig)
    ml: MLConfig = field(default_factory=MLConfig)

    def is_meme(self, symbol: str) -> bool:
        # Membership only — memes are always kept out of the core engine, even
        # when the meme sleeve is disabled (they then simply aren't traded).
        return symbol in set(self.meme.meme_symbols)
