# Honest baseline for V7 (2026-10-31)

Produced by `python pit_backtest.py` (results: `outputs/honest_baseline.json`, log: `outputs/honest_baseline.log`).

**Method.** Same V7 engine code (`backtest.BacktestEngine`; validated: the point-in-time loop reproduces the
original engine's equity curve to the cent on the original setup). Only the coin choice changes:
- Universe = **top 20 coins by trailing 30-day median dollar volume as of each date** (market-cap proxy;
  free market-cap history needs a paid key), from **every Binance USDT pair ever listed**, incl. delisted
  coins (593 series from data.binance.vision; reused tickers such as LUNA split into separate coins).
- Removed hindsight inputs: XRP+HBAR / ONDO sectors (-> one sector per universe coin, same `top_n`),
  per-coin risk caps (OP 5% etc.), hand-picked meme list.
- Costs: **0.3% round trip** (0.10% fee + 0.05% slippage per side). Original V7 used 0.2%.
- Window ends 2026-09-30 (last full month in the archive).

| Run | Window | CAGR | Sharpe | Max DD | Trades |
|---|---|---|---|---|---|
| A V7 as-is (hand-picked, 0.2%) | 2024-01-18 .. 2026-09-30 | **+30.2%** | 1.21 | -20.6% | 675 |
| B V7 as-is, 0.3% costs | same | +29.0% | 1.17 | -21.0% | 675 |
| C hand-picked coins & sectors, Binance data, 0.3% | same | +28.0% | 1.06 | -24.4% | 688 |
| **D HONEST point-in-time top-20, 0.3%** | same | **+2.0%** | 0.21 | -21.4% | 164 |
| **E HONEST point-in-time top-20, 0.3%** | 2020-01-01 .. 2026-09-30 | **+3.0%** | 0.40 | -22.7% | 125 |
| BTC buy & hold | 2024-01-18 .. | +29.8% | 0.79 | -53.0% | |
| Equal-weight top-20, monthly | 2024-01-18 .. | -16.4% | 0.12 | -79.1% | |
| BTC buy & hold | 2020-01-01 .. | +43.8% | 0.91 | -76.6% | |
| Equal-weight top-20, monthly | 2020-01-01 .. | +4.4% | 0.48 | -92.9% | |

**Reading it.** Costs (A->B: -1.2 pts) and data source (B->C: -1.0 pts) barely matter. **The coin list is
worth ~26 points of CAGR**: V7's edge was choosing XRP/HBAR/ONDO in 2026, knowing how they did.

**Why D/E are so low - the -20% add-pause.** `risk.drawdown_pause_adds_level = 0.20` blocks new positions
once drawdown passes -20%. The honest runs cross it early and almost never recover:

| Run | Locked (DD <= -20%) | First lock | Trades after first lock |
|---|---|---|---|
| D | 91.4% of days | 2024-04-12 | 5 |
| E | 97.3% of days | 2020-03-08 (COVID crash) | 3 |

Diagnostic only (rule switched off, not a candidate strategy): D **-6.5% CAGR, -47.1% max DD**;
E **+18.6% CAGR, -50.9% max DD** - the trading logic itself loses in 2024-26 and trails BTC buy & hold
in 2020-26 with similar-or-worse risk per unit return.

**Bottom line.** Without hindsight, V7 does not beat holding BTC; its +30% backtest was coin selection.
The same -20% lock applies live (`live_runner` uses the same overlay): V7 live is at about -11% now.
