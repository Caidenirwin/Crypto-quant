# RWA Quant Trading System: Strategy Spec

## Objective

A rule-based quantitative trading system that:

- Maximizes crypto unit accumulation
- Focuses on real-world-asset (RWA) tokens
- Removes emotional decision-making
- Is fully automatable

## Asset Universe

**Core (long-term accumulation):** LINK (Chainlink), ONDO (Ondo Finance), MKR (Maker)

**Yield / credit:** MPL (Maple Finance), CFG (Centrifuge), GFI (Goldfinch), CPOOL (Clearpool)

**Growth / rotation:** RIO (Realio), PRO (Propy), POLYX (Polymesh)

Assets are only added through `config.py`.

## Strategy Modules

### 1. Momentum
- Entry (all required): 7-day return > +10%, 30-day return > +15%, volume > +20% vs 7-day average
- Allocation: top 2 assets by momentum, 40% total
- Exit: sell 25% at +25% profit; exit fully when momentum falls below threshold

### 2. Pair Trading
- Pairs: LINK/ONDO, MKR/CFG
- Entry: ratio deviates > 15% from its 30-day mean; sell the outperformer, buy the underperformer
- Exit: ratio reverts to mean

### 3. Volatility Farming
- Entry: asset trades within a +/-10% range over 14 days
- Buy in the bottom 20% of the range, sell in the top 20%

### 4. Breakout
- Entry (all required): breaks 14-day high, volume spike >= 30%, volatility expansion
- Exit: partial sell at +20%; full exit on loss of momentum

### 5. Yield
- 20% of portfolio in yield assets (MPL, CFG)
- Yield collected weekly and converted to LINK or ONDO

## Portfolio Allocation

| Category     | Allocation |
| ------------ | ---------- |
| Core         | 40%        |
| Momentum     | 20%        |
| Pair Trading | 15%        |
| Volatility   | 15%        |
| Yield        | 10%        |

## Risk Management

- Max position size: 20%
- Stop loss: -15% per position
- Max drawdown: -30%, then reduce exposure by 50%

## Execution Cycle (weekly)

1. Update market data
2. Rank assets
3. Apply strategies in order: momentum, pair trading, volatility, breakout
4. Rebalance portfolio
5. Allocate yield profits to core assets

## Rules

- No manual overrides
- Every trade follows the rules and is logged
- The goal is to accumulate more crypto, not chase price
