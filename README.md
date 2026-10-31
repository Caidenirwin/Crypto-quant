# Crypto Quant Trading System (Paper/Testnet)

Rule-based, modular crypto quant framework for research, backtesting, forward-testing, and Binance Spot Testnet paper execution.

## What This Project Does
- Loads OHLCV market data from CSV files.
- Builds a rules-based tradable universe.
- Runs multiple strategy sleeves (momentum, pair trading, volatility, breakout, yield).
- Applies portfolio/risk constraints.
- Backtests and forward-tests strategy behavior.
- Executes **paper/testnet-only** orders through Binance Spot Testnet.
- Provides a Streamlit monitoring dashboard.

## Features
- Modular architecture (`data.py`, `backtest.py`, `risk.py`, `portfolio.py`, `strategies/*`).
- Robustness and fragility diagnostics.
- Forward-testing with audit logs.
- Safe testnet execution layer (`execution/binance_executor.py`).
- Dashboard (`dashboard_app.py`) with equity, drawdown, positions, trades, and optimization diagnostics.
- Automatic Excel trade/equity backup (`logs/trading_backup.xlsx`) during runner cycles.

## Project Layout
- `main.py` - batch/backtest entry
- `live_runner.py` - scheduled paper/testnet runner
- `forward_test.py` - forward-testing engine
- `dashboard_app.py` - read-only monitoring dashboard
- `config.py` - all configuration knobs
- `strategies/` - strategy sleeves
- `execution/` - Binance Spot Testnet executor
- `data/example_prices.csv` - sample public-safe dataset
- `outputs/` - non-sensitive output staging

## Quick Start
1. Create virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Configure credentials in local `.env` (never commit):
   ```env
   BINANCE_API_KEY=YOUR_BINANCE_TESTNET_API_KEY_HERE
   BINANCE_SECRET=YOUR_BINANCE_TESTNET_SECRET_HERE
   ```
3. Run one paper cycle:
   ```bash
   python live_runner.py
   ```
4. Run scheduled paper loop (example: hourly):
   ```bash
   python live_runner.py --loop --interval-minutes 60
   ```
5. Launch dashboard:
   ```bash
   python -m streamlit run dashboard_app.py
   ```

## Example Output (Runner)
- Run number and timestamp
- Equity and drawdown
- Trades executed in cycle
- Tax reserve balance
- Next scheduled run time

## Tech Stack
- Python 3.11+
- pandas, numpy
- plotly + streamlit
- ccxt (Binance Spot Testnet)

## Security Notes
- This repository is configured to ignore secrets and private runtime data.
- `.env` is ignored.
- Runtime logs and private outputs are ignored.
- Only public-safe sample data is included by default.

## Disclaimer
This project is for educational purposes only. No financial advice.
