# Simple honest baselines (2026-10-31)

`python simple_baselines.py` — Binance archive data, 0.3% round-trip costs (0.15%/side on traded notional),
signal at close t -> fill at open t+1. Nothing tuned: SMA 200, vol target 40%, vol window 30 days
(= existing `risk.volatility_lookback`). (c): each coin uses its own 200-day SMA for its 50% half; holdings
re-sized only on signal flips. (d): re-sized daily. Windows end 2026-09-30; "2024-26" starts 2024-01-18.

| Period | Rule | CAGR | Sharpe | Max DD | Time in market | Avg exposure | Trades |
|---|---|---|---|---|---|---|---|
| 2020-26 | (a) BTC buy & hold | +43.8% | 0.91 | -76.6% | 100% | 100% | 1 |
| 2020-26 | (b) BTC > 200-SMA else cash | +34.6% | 0.91 | -64.4% | 58% | 58% | 57 |
| 2020-26 | (c) 50/50 BTC/ETH, own 200-SMA | **+47.1%** | **1.00** | -55.8% | 64% | 61% | 98 |
| 2020-26 | (d) (b) + 40% vol target | +28.5% | 0.95 | **-45.0%** | 58% | 46% | 1,142 |
| 2024-26 | (a) BTC buy & hold | **+29.8%** | 0.79 | -53.0% | 100% | 100% | 1 |
| 2024-26 | (b) BTC > 200-SMA else cash | +25.1% | 0.81 | -32.0% | 59% | 59% | 29 |
| 2024-26 | (c) 50/50 BTC/ETH, own 200-SMA | +22.5% | 0.73 | -30.9% | 60% | 54% | 50 |
| 2024-26 | (d) (b) + 40% vol target | +25.5% | **0.90** | **-25.1%** | 59% | 52% | 386 |

## (e) = (d) + 10-percentage-point rebalance band (fixed, not tuned)

Trade only when |target - actual drifted weight at the open| > 10 points.

| Period | Rule | CAGR | Sharpe | Max DD | Trades |
|---|---|---|---|---|---|
| 2020-26 | (d) re-size daily | +28.5% | 0.95 | -45.0% | 1,142 |
| 2020-26 | (e) 10-pt band | +29.5% | 0.97 | -43.9% | 120 |
| 2024-26 | (d) re-size daily | +25.5% | 0.90 | -25.1% | 386 |
| 2024-26 | (e) 10-pt band | +26.1% | 0.91 | -25.1% | 50 |

Edge case (signal says cash but a sub-10-pt residual is still held): 0 days in both periods.
(e) runs as a paper-only strategy: `paper_trend.py` (ledger in `logs/paper_trend/`); replaying the paper
ledger over both periods reproduces the backtest exactly (x1.8694 / x5.7112, 50 / 120 trades).

For comparison (outputs/honest_baseline.md): V7 honest point-in-time top-20 = +3.0% (2020-26) / +2.0% (2024-26);
V7 as-is with hindsight = +30.2% (2024-26).
