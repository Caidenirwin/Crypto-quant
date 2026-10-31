"""
Sector Rotation Strategy.

Pattern (from factor IC analysis):
  - Sharpe30 IC = +0.099 (best single factor for 7-day forward returns)
  - DeFi sector consistently outperforms Layer1 during corrections
  - 30-day risk-adjusted return (Sharpe30) predicts 7-day forward returns
    better than raw 30-day return

Logic:
  1. Compute rolling 30-day Sharpe ratio for each available asset.
  2. Group assets into predefined sectors (from config.sector_definitions).
  3. Score each sector = average Sharpe30 of its available assets.
  4. Select the top-N scoring sectors.
  5. Equal-weight all assets within the selected sectors.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from config import Config
from utils import normalize_weights


class SectorRotationStrategy:
    def __init__(self, config: Config):
        self.config = config

    def generate_weights(
        self, market_data: Dict[str, pd.DataFrame], date: pd.Timestamp
    ) -> Dict[str, float]:
        window = self.config.strategy.sector_rotation_sharpe_window
        top_n  = self.config.strategy.sector_rotation_top_n
        min_assets = self.config.strategy.sector_rotation_min_assets_per_sector

        # Step 1: compute Sharpe30 for every available asset
        asset_sharpe: Dict[str, float] = {}
        for symbol, df in market_data.items():
            if date not in df.index:
                continue
            idx = df.index.get_loc(date)
            if isinstance(idx, slice):
                continue
            if idx < window:
                continue
            close = df["Close"].astype(float)
            r = close.pct_change()
            window_rets = r.iloc[idx - window + 1 : idx + 1].dropna()
            if len(window_rets) < window // 2:
                continue
            mu = float(window_rets.mean())
            sigma = float(window_rets.std())
            if sigma <= 0:
                continue
            asset_sharpe[symbol] = mu / sigma

        if not asset_sharpe:
            return {s: 0.0 for s in market_data}

        # Step 2: score each sector
        sector_defs: Dict[str, List[str]] = self.config.strategy.sector_definitions
        sector_scores: Dict[str, float] = {}
        sector_assets: Dict[str, List[str]] = {}

        for sector, members in sector_defs.items():
            available = [s for s in members if s in asset_sharpe]
            if len(available) < min_assets:
                continue
            sector_scores[sector] = sum(asset_sharpe[s] for s in available) / len(available)
            sector_assets[sector] = available

        if not sector_scores:
            return {s: 0.0 for s in market_data}

        # Step 3: select top-N sectors — but only if their Sharpe is above the minimum.
        # When ALL sectors are negative (sustained bear market), go to cash rather
        # than concentrating in the "least bad" sector.
        min_sharpe = self.config.strategy.sector_rotation_min_entry_sharpe
        ranked = sorted(sector_scores.items(), key=lambda x: x[1], reverse=True)
        selected_sectors = [sec for sec, sc in ranked[:top_n] if sc >= min_sharpe]

        # Step 4: equal-weight all assets in the selected sectors
        chosen: List[str] = []
        for sec in selected_sectors:
            chosen.extend(sector_assets[sec])

        raw = {s: 0.0 for s in market_data}
        for sym in chosen:
            if sym in raw:
                raw[sym] = 1.0

        return normalize_weights(raw)
