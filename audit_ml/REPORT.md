# ML pipeline audit — 2026-10-31

Scope: `ml_strategy.py`, `ml_research.py`, `ml_autoresearch.py`, `ml_live.py`, `models/`, `outputs/`.
No project code or config was modified. Every check is a runnable script in `audit_ml/`
(`cd audit_ml; python <script>`); raw outputs are in `audit_ml/results/`.
Champion audited: calibrated HGB, `baseline` features (37), RR 2.0, EV gate 0.2R — reproduced exactly
(2,053 gated OOS signals, +0.052R/signal, portfolio CAGR 10.9% 2021+).

| # | Check | Verdict | Evidence |
|---|---|---|---|
| 1 | Label leakage / shift test | **PASS** | `a01_leakage.py`: future rows deleted at 5 cut-offs (2021-06 … 2026-10) → 0 mismatches across all 69 features; no label column in `FEATURES` (ml_strategy.py:37-49). |
| 2 | Purge / embargo | **PASS (minor)** | `a02_purge.py`: calendar purge (ml_strategy.py:293,297) holds because crypto bars are daily without gaps — except **SKY(USD)** (181-day data gap) → 10 train rows in 2 folds reach the test block. Separately, inside `CalibratedModel.fit` (ml_strategy.py:261-264) ~252 base-model rows/fold have labels overlapping the calibration slice (no gap). |
| 3 | Walk-forward integrity | **PASS** | `a03_walkforward.py`: 24 strictly chronological folds, fresh model per fold, ≥17-day gap, no train row ≥ test start, rows date-sorted for the 80/20 split, no scalers/encoders/feature selection anywhere. |
| 4 | Survivorship | **FAIL** (bias, not look-ahead) | `a04_survivorship.py`: all 27 coins chosen in 2026 (files created Apr–Sep 2026); 25/27 still trading — collapsed coins never included. Cross-sectional features (ml_strategy.py:211-217) use only coins present that day (no look-ahead) but rest on **3–5 coins in 2018-2020** vs 25 now. |
| 5 | Data quality | **FAIL (moderate)** | `a05_data_quality.py`, `a05b_contamination.py`: data.py:341-347 only drops NaN — zeros/placeholders pass. **RIO** 7 zero-price rows in training; **OP** (tradable) 84 zero + 131 flat placeholder bars (2022-03/05) contaminate 138 (MA200) – 303 (365-day) training rows; ~1,100 contaminated rows total (2.5%). Stored Yahoo candles differ from a fresh download exactly on file-creation days (XRP 2026-04-20, ONDO 04-23, HBAR 04-24 — partial candles) and several files skip 2026-04-21/22. Non-USD files are correctly excluded (data.py:126-157) — none in the dataset. |
| 6 | Sanity baselines | **FAIL (economic edge not shown)** | `a06_baselines.py`: ranking is real — OOS AUC 0.586 vs shuffled-label 0.463/0.483/0.508. But gated avg R: champion +0.052R; **shuffled-label models +0.243R and +0.467R in 2 of 3 seeds** (small n, clustered); random same-n entries −0.013R (champion at 99th pct); trivial trend filter +0.038R on 9,693 trades. |
| 7 | Overfitting | **FAIL** | `a07_overfitting.py`: **34 configurations** evaluated on 2021-2026 (ml_research.py:136-221, ml_autoresearch.py:79-116 + RR/short sweeps). Champion Sharpe 0.75 ≈ expected max of 34 zero-skill trials (0.76); **Deflated Sharpe 0.49** (≥0.95 = significant). No untouched holdout exists. Pseudo-holdout (pick on pre-May data, test May–Oct 2026): picked config and current champion **both made 0 trades** → uninformative. |
| 8 | Calibration | **FAIL** | `a08_a09_calibration_decay.py`: top decile predicted 0.54 vs actual 0.29; **rows the EV gate trades: predicted 0.61 vs actual 0.31**. Brier 0.186 is *worse* than a constant base-rate forecast (0.177). The EV gate (ml_strategy.py:276-278) therefore rests on ~2× overstated probabilities. |
| 9 | Decay | **UNCLEAR (regime-bursty) + bug** | AUC by year 0.51/0.65/0.56/0.48/0.65/0.58 — no steady decay (rolling-90d mean 0.585 → 0.590). Gated trades come in bursts (Apr-May 2025 +0.6..+1.0R; Oct 2025 −1.0R; Jan 2026 371 trades −0.60R). **The "last 180d: 531 trades, −0.70R" is not recent**: ml_autoresearch.py:65 anchors the window on the last *signal* date, so it covers 2025-07-28…2026-01-24. **The champion has produced no gated signal since 2026-01-24.** Clustered (weekly) t-stat 1.82. |
| 10 | Costs | **PASS (thin)** | `a10_costs.py`: avg R +0.089 / +0.052 / +0.015 at 0% / 0.3% / 0.6% round trip; PF 1.15 / 1.09 / 1.02; CAGR 2021+ 12.9% / 10.9% / 9.0%; 2024+ 5.1% / 3.8% / 2.6%. Median stop = 8.6% of price, so 0.3% ≈ 0.035R/trade. |
| 11 | Live vs training features | **PASS normally / FAIL under data lag** | `a11_live_parity.py`: 703/703 values identical, model `feature_names_in_` == bundle features (same order), no scaler. `a11b_lag_skew.py`: when Yahoo coins lag Binance by a day (happened 2026-10-30), the 4 Binance-sourced coins (BONK/FLOKI/PEPE/SHIB) are scored on a 4-coin cross-section — breadth_ma50 0.04 → 0.25, ranks quantised to 0.25 steps; values training never saw. Minor: live drops NaN-feature coins (ml_live.py:236) while training keeps them (0.1% of rows). |
| 12 | Execution | **V7 Sep-29: not a bug. ML: FAIL (double-fire)** | `a12_execution.py`: Sep 29 = three separate processes (16:23 manual, 16:25 autopilot start, 16:40 manual) each stepping toward the ~$133 target under the $60 order cap: 60+60+14 = $134. `run_number` restarts at 0 per process (live_runner.py:1258) → three "run 1" rows. No overlap, so the cycle lock (live_runner.py:1144, added 2026-10-31) would not have changed it. **ML can fire twice:** `last_signal_date` is set in memory (ml_live.py:231) and state is only saved after `generate_entries` returns (ml_live.py:343); any exception after a fill (ml_live.py:292-308), e.g. `mirror_price` (ml_live.py:89/282) network error, loses the position and the signal date. Reproduced with a fake exchange: PENDLE bought twice for one signal, and the first position is orphaned (no stop/target management). |

## Is the ML working properly?

The plumbing is sound — no look-ahead, proper walk-forward, live features match training on normal days.
The **model is not demonstrably profitable**: a real but weak ranking signal (AUC ≈ 0.59) that does not
survive selection-bias correction (DSR 0.49), is barely better than a one-line trend rule, is badly
miscalibrated exactly where it trades, and has not issued a tradable signal in nine months.

## Top 3 problems (by impact)

1. **No statistically real trading edge after 34 trials** (#6, #7, #9): DSR 0.49; shuffled-label models matched or
   beat the gated avg R; champion ≈ trend filter. Everything since 2021 has been used for selection — there is
   no clean holdout.
2. **The decision layer reads wrong numbers** (#8, #9): probabilities ~2× too high where the EV gate trades, and
   the health gate's "recent 180 days" is really Jul-2025…Jan-2026 (ml_autoresearch.py:65).
3. **ml_live can double-buy and orphan positions** (#12): state is persisted only at the end of the cycle
   (ml_live.py:231, 341-343). Harmless while entries are paused; real money risk the moment they are not.

Next tier: survivorship-biased universe (#4), zero/placeholder prices and partial candles in training (#5),
cross-section skew under data lag (#11).

## What to fix first

1. **ml_live persistence** — save state (position + `last_signal_date`) immediately after every fill and before
   any further network call. Smallest change, removes a live-money failure mode.
2. **Health-gate window** — anchor `recent_cut` on today, not on the last signal; then the gate would correctly
   report "0 signals in 180 days".
3. **Freeze the ML config now and start a true holdout** (Nov 2026 onward, never used for selection); fix
   calibration (out-of-fold / gapped isotonic) and data hygiene (reject ≤0 / flat placeholder bars, re-download
   partial candles) before judging it again. Until then the paper sleeve should stay paused.
