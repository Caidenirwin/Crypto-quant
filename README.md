# Crypto-quant

**A Python crypto research and paper-trading system built to be honest about what works and what doesn't.**

> Educational project by Caiden Irwin (Carleton University). **Paper trading and Binance Spot Testnet only. Not financial advice.**

---

## Overview

Crypto-quant is a modular research system for crypto strategies. It runs ML and rule-based strategies through leak-free walk-forward tests and point-in-time (hindsight-free) backtests, and compares every result with simply holding BTC. It can also run strategies continuously as paper trades, with a Streamlit dashboard to monitor them.

The main lesson of the project: **most of the edge in a good-looking crypto backtest came from hindsight.** Once the hindsight was removed, the strategy that held up best was simple. It is a volatility-targeted BTC trend filter that cut the 2024–26 max drawdown from -53% (buy-and-hold) to -25%.

## Key features

- **ML and rule-based strategies.** Strategy sleeves live in `strategies/` (RSI dip, sector rotation, volatility, momentum, breakout, pair trading, meme momentum, yield), and the ML bracket model is in `ml_strategy.py`.
- **Leak-free walk-forward testing.** Folds are strictly chronological with a purge gap, and a fresh model is trained for each fold. This was checked independently in `audit_ml/`.
- **Point-in-time universe.** Each date uses the top 20 coins by trailing 30-day median dollar volume *as of that date*. Coins come from **593 Binance USDT pairs, including delisted ones** (LUNA, FTT, …), pulled from `data.binance.vision` by `universe_data.py`.
- **Realistic costs.** Honest backtests and the ML research use a 0.3% round trip (0.10% fee + 0.05% slippage per side).
- **BTC buy-and-hold benchmark** on every report.
- **Crash-safe paper trading.** `paper_trend.py` only processes completed daily candles and catches up after downtime. `paper_account.py` rebuilds its state from the trade logs.
- **Health gate.** The ML sleeve is automatically **paused** when recent out-of-sample evidence is missing or too thin (`ml_live.py`, `models/ml_status.json`).
- **Anti-rug filter.** `anti_rug.py` screens meme coins for coin age, crashes, liquidity drain and parabolic pumps.
- **Testnet-only execution.** `execution/binance_executor.py` forces ccxt sandbox mode and refuses to run if the endpoint URLs don't look like testnet.
- **Streamlit dashboard** showing equity, drawdown, positions and trades, plus a **Trend (paper)** tab.

## Results: honesty first

All numbers come from the project's own research outputs (linked below). Runs use 0.3% round-trip costs unless noted. All windows end on 2026-09-30, and "2024–26" starts on 2024-01-18.

### 1. Hindsight in coin selection ([`outputs/honest_baseline.md`](outputs/honest_baseline.md))

| Run | Window | CAGR | Sharpe | Max DD | Trades |
|---|---|---:|---:|---:|---:|
| V7 as-is (hand-picked coins, 0.2% costs) | 2024–26 | +30.2% | 1.21 | -20.6% | 675 |
| V7, honest point-in-time top-20 | 2024–26 | +2.0% | 0.21 | -21.4% | 164 |
| V7, honest point-in-time top-20 | 2020–26 | +3.0% | 0.40 | -22.7% | 125 |
| BTC buy-and-hold | 2024–26 | +29.8% | 0.79 | -53.0% | – |
| BTC buy-and-hold | 2020–26 | +43.8% | 0.91 | -76.6% | – |

Costs (about -1.2 pts) and the data source (about -1.0 pts) barely matter. **The coin list was worth about 26 points of CAGR.** That gap is hindsight in coin selection.

### 2. Simple baselines ([`outputs/simple_baselines.md`](outputs/simple_baselines.md))

Rule (e) holds BTC while it is above its 200-day SMA, sizes the position to a 40% volatility target, and only trades when the target weight moves by more than 10 points. Nothing was tuned.

| Rule | Period | CAGR | Sharpe | Max DD | Trades |
|---|---|---:|---:|---:|---:|
| BTC buy-and-hold | 2024–26 | +29.8% | 0.79 | -53.0% | 1 |
| BTC 200d trend + 40% vol target, 10-pt band | 2024–26 | +26.1% | 0.91 | -25.1% | 50 |
| BTC buy-and-hold | 2020–26 | +43.8% | 0.91 | -76.6% | 1 |
| BTC 200d trend + 40% vol target, 10-pt band | 2020–26 | +29.5% | 0.97 | -43.9% | 120 |

The rule gives up some return. In exchange, it has a higher Sharpe ratio and roughly half the drawdown. Replaying the `paper_trend.py` ledger reproduces these backtests exactly.

### 3. ML audit ([`audit_ml/REPORT.md`](audit_ml/REPORT.md))

- **Passed:** no lookahead or label leakage, a clean walk-forward setup, and live features that match the training features.
- **No statistically real edge:** after 34 configurations, the **deflated Sharpe is 0.49** (0.95 or higher would be significant). OOS AUC is about **0.59**, and on the rows the model actually trades, its probabilities are about **2× overconfident** (predicted 0.61 vs. actual 0.31).
- Because of this, the **ML sleeve is paused** until it is tested on a fresh holdout from November 2026 onward.

### 4. Fear & Greed study

The project's research found that the Fear & Greed index **lags BTC by about 1 day**. After controlling for BTC, it had **no predictive power** for 1-, 7- or 30-day coin returns. *(No standalone report for this study is committed yet. The weekly walk-forward results for the `fear_greed` ML variant are in [`outputs/research_reports/`](outputs/research_reports/).)*

**More docs:** [`docs/STRATEGY.md`](docs/STRATEGY.md) is the original rule-based strategy spec.

## Project structure

```
Crypto-quant/
├── main.py                  # Batch backtest entry point (V7 engine + robustness diagnostics)
├── backtest.py              # BacktestEngine: signal at close t, fill at open t+1
├── pit_backtest.py          # Honest point-in-time top-20 backtest -> outputs/honest_baseline.*
├── simple_baselines.py      # BTC hold / trend / vol-target baselines -> outputs/simple_baselines.*
├── universe_data.py         # Downloads every Binance USDT pair (incl. delisted) into universe_data/
├── update_data.py           # Refreshes market data for the live runner
├── paper_trend.py           # PAPER-ONLY BTC trend rule (e); own ledger in logs/paper_trend/
├── live_runner.py           # Paper/testnet runner, single cycle or --loop
├── forward_test.py          # Forward-testing engine
├── paper_account.py         # Virtual account rebuilt from trade logs
├── ml_strategy.py           # ML features, walk-forward, calibrated model
├── ml_research.py / ml_autoresearch.py   # ML experiments; weekly champion/challenger research
├── ml_live.py               # Live ML scoring behind the research health gate
├── anti_rug.py              # Anti-rug-pull guard for the meme sleeve
├── chart_patterns.py        # Chart patterns -> ML features (no lookahead)
├── news_feed.py             # Fear & Greed index + news headlines
├── indicators.py, data.py, portfolio.py, risk.py, robustness.py, tax_reserve.py, utils.py
├── config.py                # All configuration knobs (dataclasses)
├── dashboard.py             # Streamlit entry -> dashboard_app.py
├── strategies/              # Strategy sleeves (rsi_dip, sector_rotation, volatility, momentum, ...)
├── execution/               # binance_executor.py: Binance Spot Testnet executor (sandbox enforced)
├── audit_ml/                # 14 audit scripts (a01–a12), REPORT.md, results/
├── outputs/                 # Research outputs (honest baseline, simple baselines, ML walk-forward)
├── models/                  # Saved ML champion model + status/health file
├── reference_data/          # BTC(USD).csv, ETH(USD).csv, fear_greed.csv, and the original BTC sheet export
├── data/example_prices.csv  # Public-safe sample dataset
├── docs/STRATEGY.md         # Strategy spec
├── scripts/test_execution.py
├── tests/                   # pytest suite (anti-rug, chart patterns, ML persistence, profit split)
├── start_dashboard.ps1, start_live.ps1, run_bot_now.ps1
├── autopilot.ps1, install_autopilot.ps1, stop_autopilot.ps1, install_paper_trend.ps1
├── requirements.txt, .env.example, .streamlit/config.toml
```

These folders are **gitignored and regenerated locally**:

- `universe_data/` is rebuilt by `python universe_data.py`. The download is resumable and only fetches months that aren't on disk yet.
- `audit_ml/cache/` is created automatically by `audit_ml/_common.py` the first time you run an audit script.
- `logs/` is written by the runners.

## Quick start (Windows / PowerShell)

Requires Python 3.11+.

```powershell
git clone https://github.com/Caidenirwin/Crypto-quant.git
cd Crypto-quant
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env   # then add your Binance SPOT TESTNET keys
```

`.env` (gitignored, never commit it):

```env
BINANCE_API_KEY="YOUR_BINANCE_TESTNET_API_KEY_HERE"
BINANCE_SECRET="YOUR_BINANCE_TESTNET_SECRET_HERE"
```

> **Note:** some modules also import `scikit-learn`, `joblib`, `requests` and `yfinance`, and the tests need `pytest`. If an import fails, install those with `pip install`.

### Research and backtests

```powershell
python main.py                 # V7 batch backtest + diagnostics
python universe_data.py        # one-time: build the point-in-time universe (resumable)
python pit_backtest.py         # honest point-in-time backtest -> outputs/honest_baseline.md
python simple_baselines.py     # BTC baselines -> outputs/simple_baselines.md
python ml_autoresearch.py      # ML walk-forward research (--force to re-run)
cd audit_ml; python a01_leakage.py   # any audit script; results -> audit_ml/results/
```

### Dashboard

```powershell
.\start_dashboard.ps1          # opens http://localhost:8501 (read-only)
```

### Paper trend strategy (no orders, no API key needed)

```powershell
python paper_trend.py                                  # process new completed daily candles
python paper_trend.py --replay 2024-01-18 2026-09-30   # parity check vs the backtest
.\install_paper_trend.ps1      # register hourly Scheduled Task "CryptoQuantPaperTrend"
```

### Testnet runner

> ⚠️ **Use Binance Spot TESTNET keys only.** The executor forces sandbox mode, but never put real-exchange keys in `.env`.

```powershell
.\start_live.ps1               # loads .env, runs one live_runner.py cycle
python live_runner.py --loop --interval-minutes 60    # scheduled loop
.\run_bot_now.ps1 [-SkipData]  # refresh data and run one cycle now
.\install_autopilot.ps1        # Scheduled Task: data refresh + weekly ML research + hourly runner
.\stop_autopilot.ps1           # emergency stop (creates logs\emergency_stop.flag)
```

## Configuration

All settings are in `config.py`, grouped into dataclasses:

- **`DataConfig`**: starting cash, rules-based universe size and liquidity/quality filters, data folders.
- **`StrategyConfig`**: which sleeves are enabled. Momentum, breakout and pair trading are disabled because they lost money in backtests, and yield stays off until there is a real staking integration.
- **`PortfolioConfig`**: fees (`fee_rate`) and slippage (`slippage_rate`), each 0.05%.
- **`RiskConfig`**: `max_drawdown_pct`, plus `drawdown_pause_adds_level = 0.20`, which blocks new positions once drawdown passes -20%.
- **`MLConfig`**: EV gate and `roundtrip_cost = 0.003`.
- Also `BacktestConfig`, `ForwardTestConfig` and `MemeConfig`.

## Testing

```powershell
pytest tests/
```

## Roadmap

- **3-month paper comparison** of V7 vs. the BTC trend rule, running through about February 2027.
- **Calibration on fresh data**: fix the ML model's overconfident probabilities and re-test its edge on the November 2026+ holdout.
- **Altcoins only as a small satellite** around the BTC trend core.

## Security

The `.env` file, logs, paper-trading ledgers, private outputs, caches, `.claude/`, and any `secrets.toml` are gitignored. Only the `.env.example` template is committed.

## Disclaimer

This project is for educational and research purposes only. It is **not financial advice**. All trading is paper trading or Binance Spot Testnet. Backtests do not predict future results, and crypto assets are highly volatile.

## License

No license has been added yet. Consider adding one (for example MIT) so others know how they may reuse the code.

## Author

**Caiden Irwin**, Carleton University
