"""
Data loading and preprocessing for OHLCV crypto data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Set

import pandas as pd
from pandas.api.types import is_numeric_dtype

from config import Config
from utils import ensure_datetime_index


class DataManager:
    """
    Handles loading and validation of market data.
    Data format:
      - Dict[symbol, DataFrame]
      - DataFrame index: DatetimeIndex
      - Required columns: Open, High, Low, Close, Volume
    """

    def __init__(self, config: Config):
        self.config = config

    def load_symbol_csv(self, file_path: str, symbol: str) -> Dict[str, pd.DataFrame]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {file_path}")

        header_row = self._find_header_row(path)
        df = pd.read_csv(path, sep=self.config.data.csv_separator, skiprows=header_row)
        df = self._sanitize_raw_columns(df)
        df = ensure_datetime_index(df, self.config.data.date_column)
        df = self._coerce_ohlcv_numeric(df)
        df = self._drop_invalid_rows(df)
        self._validate_ohlcv(df, symbol)
        self._validate_min_history(df, symbol)
        return {symbol: df[self.config.data.ohlcv_columns].copy()}

    def load_multi_csv(self, symbol_to_file: Dict[str, str]) -> Dict[str, pd.DataFrame]:
        data: Dict[str, pd.DataFrame] = {}
        source_file: Dict[str, str] = {}
        for raw_symbol, file_path in symbol_to_file.items():
            symbol = self.normalize_symbol(raw_symbol)
            try:
                loaded = self.load_symbol_csv(file_path, symbol)[symbol]
                if symbol in data:
                    # Keep longer history if aliases map multiple files to same canonical symbol.
                    if len(loaded) > len(data[symbol]):
                        prev = source_file.get(symbol, "unknown")
                        print(
                            f"Alias collision for {symbol}: keeping {Path(file_path).name}, replacing {prev}"
                        )
                        data[symbol] = loaded
                        source_file[symbol] = Path(file_path).name
                    else:
                        print(
                            f"Alias collision for {symbol}: keeping {source_file.get(symbol, Path(file_path).name)}"
                        )
                else:
                    data[symbol] = loaded
                    source_file[symbol] = Path(file_path).name
            except Exception as exc:
                print(f"Skipping {symbol} ({Path(file_path).name}): {exc}")
        return data

    def align_on_common_dates(
        self,
        market_data: Dict[str, pd.DataFrame],
        drop_insufficient: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        if not market_data:
            return {}

        common_index = None
        for df in market_data.values():
            common_index = df.index if common_index is None else common_index.intersection(df.index)

        aligned: Dict[str, pd.DataFrame] = {}
        for symbol, df in market_data.items():
            aligned_df = df.loc[common_index].copy().sort_index()
            try:
                self._validate_min_history(aligned_df, symbol)
                aligned[symbol] = aligned_df
            except ValueError as exc:
                if drop_insufficient:
                    print(f"Dropping {symbol} after alignment: {exc}")
                else:
                    raise
        return aligned

    def symbols(self, market_data: Dict[str, pd.DataFrame]) -> List[str]:
        return sorted(market_data.keys())

    def detect_quote_currency(self, symbol: str) -> str:
        s = str(symbol).strip().upper()
        overrides = {k.upper(): v.upper() for k, v in self.config.data.symbol_quote_overrides.items()}
        if s in overrides:
            return overrides[s]
        if "(" in s and ")" in s and s.rfind("(") < s.rfind(")"):
            return s[s.rfind("(") + 1 : s.rfind(")")]
        return "UNKNOWN"

    def quote_currency_report(self, symbols: List[str]) -> Dict[str, object]:
        by_quote: Dict[str, List[str]] = {}
        unknown: List[str] = []
        for symbol in sorted(symbols):
            quote = self.detect_quote_currency(symbol)
            by_quote.setdefault(quote, []).append(symbol)
            if quote == "UNKNOWN":
                unknown.append(symbol)
        quotes = sorted(by_quote.keys())
        known = sorted([q for q in quotes if q != "UNKNOWN"])
        mixed = len(known) > 1
        return {
            "quotes": quotes,
            "by_quote": by_quote,
            "unknown_symbols": unknown,
            "mixed_quote_warning": mixed,
        }

    def filter_by_allowed_quote_currency(
        self, market_data: Dict[str, pd.DataFrame]
    ) -> tuple[Dict[str, pd.DataFrame], Dict[str, str], Dict[str, str]]:
        """
        Keep only allowed quote currencies. No synthetic conversion is performed.
        Placeholder for future FX conversion support.
        """
        allowed = {str(self.config.data.allowed_quote_currency).upper()}
        allowed.update(q.upper() for q in self.config.data.allowed_additional_quote_currencies)
        include_unknown = bool(self.config.data.allow_unknown_quote_currency)

        included: Dict[str, pd.DataFrame] = {}
        include_reason: Dict[str, str] = {}
        excluded: Dict[str, str] = {}

        for symbol, df in market_data.items():
            quote = self.detect_quote_currency(symbol)
            if quote in allowed:
                included[symbol] = df
                include_reason[symbol] = f"QUOTE_ALLOWED:{quote}"
                continue
            if quote == "UNKNOWN" and include_unknown:
                included[symbol] = df
                include_reason[symbol] = "QUOTE_UNKNOWN_ALLOWED_BY_CONFIG"
                continue
            if quote == "UNKNOWN":
                excluded[symbol] = "QUOTE_UNKNOWN_EXCLUDED"
            else:
                excluded[symbol] = (
                    f"QUOTE_{quote}_EXCLUDED_NO_FX_CONVERSION_DATA"
                )
        return included, include_reason, excluded

    def normalize_symbol(self, symbol: str) -> str:
        s = str(symbol).strip().upper()
        alias_map = {k.upper(): v.upper() for k, v in self.config.data.symbol_alias_map.items()}
        return alias_map.get(s, s)

    def liquidity_scores_at_date(
        self, market_data: Dict[str, pd.DataFrame], date: pd.Timestamp
    ) -> Dict[str, float]:
        """
        Compute trailing average dollar volume (Close * Volume) using only data available
        up to `date` (inclusive). This is used for rules-based universe selection.
        """
        scores: Dict[str, float] = {}
        lookback = self.config.data.universe_volume_lookback_days
        for symbol, df in market_data.items():
            if date not in df.index:
                continue
            idx = df.index.get_loc(date)
            if isinstance(idx, slice):
                continue
            if idx < 1:
                continue
            start = max(0, idx - lookback + 1)
            window = df.iloc[start : idx + 1]
            if window.empty:
                continue
            dollar_vol = (window["Close"].astype(float) * window["Volume"].astype(float)).mean()
            if pd.isna(dollar_vol):
                continue
            scores[symbol] = float(dollar_vol)
        return scores

    def universe_quality_snapshot(
        self, market_data: Dict[str, pd.DataFrame], date: pd.Timestamp
    ) -> Dict[str, Dict[str, object]]:
        """
        Per-symbol quality diagnostics using only bars up to `date` (inclusive).
        """
        out: Dict[str, Dict[str, object]] = {}
        for symbol, df in market_data.items():
            snap = self._quality_snapshot_for_symbol(df, date)
            if snap is None:
                continue
            out[symbol] = snap
        return out

    def pre_filter_quality(
        self, market_data: Dict[str, pd.DataFrame]
    ) -> tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
        """
        Quality-filter UNALIGNED raw data using each symbol's own most-recent history.

        Call this BEFORE align_on_common_dates so that low-quality symbols (those with
        very short history, stale data, or thin volume) cannot limit the common-date
        intersection for the rest of the universe.

        Also excludes symbols whose most-recent bar lags the freshest symbol by more than
        `universe_pre_filter_max_stale_days` — this prevents delisted assets whose data
        ended months ago from capping the aligned end-date for the rest of the universe.

        Returns (eligible_data, dropped_reasons) where dropped_reasons maps each
        excluded symbol to a short reason string.
        """
        eligible: Dict[str, pd.DataFrame] = {}
        dropped: Dict[str, str] = {}

        non_empty = {s: df for s, df in market_data.items() if not df.empty}
        if not non_empty:
            return eligible, dropped

        latest_across_all = max(df.index.max() for df in non_empty.values())
        stale_days = int(getattr(self.config.data, "universe_pre_filter_max_stale_days", 180))
        stale_cutoff = latest_across_all - pd.Timedelta(days=stale_days)

        for symbol, df in market_data.items():
            if df.empty:
                dropped[symbol] = "EMPTY_DATAFRAME"
                continue
            symbol_latest = df.index.max()
            if symbol_latest < stale_cutoff:
                dropped[symbol] = (
                    f"STALE_DATA (latest={symbol_latest.date()}, "
                    f"cutoff={stale_cutoff.date()})"
                )
                continue
            if self.config.is_meme(symbol):
                # Memes are screened by the anti-rug guard, not the core quality filter
                # (Close*Volume is ~0 for sub-cent tokens, which would wrongly drop them).
                if len(df) >= int(self.config.meme.rug_min_history_days):
                    eligible[symbol] = df
                else:
                    dropped[symbol] = f"MEME_TOO_YOUNG ({len(df)} bars)"
                continue
            date = symbol_latest
            snap = self._quality_snapshot_for_symbol(df, date)
            if snap is None:
                dropped[symbol] = "QUALITY_SNAPSHOT_FAILED"
                continue
            if bool(snap.get("eligible", False)):
                eligible[symbol] = df
            else:
                reasons = list(snap.get("reasons", []))
                dropped[symbol] = "|".join(reasons) if reasons else "QUALITY_FILTER_FAIL"
        return eligible, dropped

    def quality_eligible_symbols(
        self, market_data: Dict[str, pd.DataFrame], date: pd.Timestamp
    ) -> tuple[Set[str], Dict[str, Dict[str, object]]]:
        snapshot = self.universe_quality_snapshot(market_data, date)
        eligible = {s for s, m in snapshot.items() if bool(m.get("eligible", False))}
        return eligible, snapshot

    def select_universe_at_date(
        self,
        market_data: Dict[str, pd.DataFrame],
        date: pd.Timestamp,
        target_size: int,
    ) -> Dict[str, object]:
        """
        Rules-based universe selection with quality filtering + quality score ranking.
        """
        if not market_data:
            return {"selected_symbols": set(), "quality_snapshot": {}, "low_quality_universe": True}

        quality_snapshot = self.universe_quality_snapshot(market_data, date)
        eligible = {s: v for s, v in quality_snapshot.items() if bool(v.get("eligible", False))}
        low_quality_universe = len(eligible) < int(self.config.backtest.min_eligible_universe_size)

        ranked = sorted(
            eligible.items(),
            key=lambda x: (float(x[1].get("quality_score", 0.0)), float(x[1].get("avg_dollar_volume", 0.0))),
            reverse=True,
        )
        k = max(1, int(target_size))
        selected = {s for s, _ in ranked[:k]}

        # Fallback: if no symbols pass strict filters, rank by quality score anyway.
        if not selected:
            all_ranked = sorted(
                quality_snapshot.items(),
                key=lambda x: (
                    float(x[1].get("quality_score", 0.0)),
                    float(x[1].get("avg_dollar_volume", 0.0)),
                ),
                reverse=True,
            )
            selected = {s for s, _ in all_ranked[:k]}
            low_quality_universe = True

        return {
            "selected_symbols": selected,
            "quality_snapshot": quality_snapshot,
            "low_quality_universe": low_quality_universe,
            "eligible_count": len(eligible),
        }

    def _validate_ohlcv(self, df: pd.DataFrame, symbol: str) -> None:
        missing = [c for c in self.config.data.ohlcv_columns if c not in df.columns]
        if missing:
            raise ValueError(f"{symbol}: missing OHLCV columns: {missing}")

    def _validate_min_history(self, df: pd.DataFrame, symbol: str) -> None:
        if len(df) < self.config.data.min_history_days:
            raise ValueError(
                f"{symbol}: insufficient history ({len(df)} < {self.config.data.min_history_days})"
            )

    def _coerce_ohlcv_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in self.config.data.ohlcv_columns:
            if col not in out.columns:
                continue
            if not is_numeric_dtype(out[col]):
                out[col] = out[col].astype(str).str.replace(",", "", regex=False)
            out[col] = pd.to_numeric(out[col], errors="coerce")
        return out

    def _sanitize_raw_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out.columns = [str(c).strip() for c in out.columns]
        return out

    def _drop_invalid_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        required = list(self.config.data.ohlcv_columns)
        cols = [c for c in required if c in df.columns]
        if not cols:
            return df
        out = df.dropna(subset=cols, how="any").sort_index()
        return out

    def _find_header_row(self, path: Path) -> int:
        """
        Detect the first valid OHLCV header row.
        This handles CSV files with leading empty comma rows.
        """
        required = {"date", "open", "high", "low", "close"}
        with path.open("r", encoding="utf-8-sig", errors="ignore") as f:
            for i, line in enumerate(f):
                parts = [p.strip().strip('"').lower() for p in line.split(",")]
                if required.issubset(set(parts)):
                    return i
        raise ValueError("could not detect OHLCV header row")

    def _quality_snapshot_for_symbol(
        self, df: pd.DataFrame, date: pd.Timestamp
    ) -> Dict[str, object] | None:
        if date not in df.index:
            return None
        idx = df.index.get_loc(date)
        if isinstance(idx, slice):
            return None
        hist = df.iloc[: idx + 1].copy()
        if hist.empty:
            return None

        lookback = int(self.config.data.universe_quality_lookback_days)
        window = hist.iloc[max(0, len(hist) - lookback) :].copy()
        if window.empty:
            return None

        history_bars = int(len(hist))
        avg_volume = float(window["Volume"].astype(float).mean())
        dollar_vol_series = window["Close"].astype(float) * window["Volume"].astype(float)
        avg_dollar_volume = float(dollar_vol_series.mean())
        median_dollar_volume = float(dollar_vol_series.median())
        liquidity_stability = float(median_dollar_volume / max(avg_dollar_volume, 1e-9))

        expected_dates = pd.date_range(start=window.index.min(), end=window.index.max(), freq="D")
        missing_bars = max(0, int(len(expected_dates) - len(window.index)))
        missing_pct = float(missing_bars / max(len(expected_dates), 1))

        returns = window["Close"].astype(float).pct_change().dropna()
        annualized_vol = float(returns.std() * (365.0 ** 0.5)) if len(returns) > 1 else 0.0

        tradability_window = int(self.config.data.universe_recent_tradability_window)
        recent = hist.iloc[max(0, len(hist) - tradability_window) :].copy()
        tradable = (
            (recent["Open"].astype(float) > 0)
            & (recent["Close"].astype(float) > 0)
            & (recent["Volume"].astype(float) > 0)
        )
        recent_tradability = float(tradable.mean()) if len(tradable) > 0 else 0.0

        reasons: List[str] = []
        if history_bars < int(self.config.data.universe_min_history_bars):
            reasons.append("HISTORY_TOO_SHORT")
        if avg_volume < float(self.config.data.universe_min_avg_volume):
            reasons.append("AVG_VOLUME_TOO_LOW")
        if avg_dollar_volume < float(self.config.data.universe_min_avg_dollar_volume):
            reasons.append("AVG_DOLLAR_VOLUME_TOO_LOW")
        if median_dollar_volume < float(self.config.data.universe_min_median_dollar_volume):
            reasons.append("MEDIAN_DOLLAR_VOLUME_TOO_LOW")
        if missing_pct > float(self.config.data.universe_max_missing_data_pct):
            reasons.append("MISSING_DATA_TOO_HIGH")
        if annualized_vol > float(self.config.data.universe_max_volatility):
            reasons.append("VOLATILITY_TOO_HIGH")
        if liquidity_stability < float(self.config.data.universe_min_liquidity_stability):
            reasons.append("LIQUIDITY_UNSTABLE")
        if recent_tradability < float(self.config.data.universe_min_recent_tradability):
            reasons.append("RECENT_TRADABILITY_LOW")

        score = self._quality_score(
            history_bars=history_bars,
            avg_dollar_volume=avg_dollar_volume,
            liquidity_stability=liquidity_stability,
            missing_pct=missing_pct,
            annualized_vol=annualized_vol,
            recent_tradability=recent_tradability,
        )
        if score < float(self.config.data.universe_quality_min_score):
            reasons.append("QUALITY_SCORE_TOO_LOW")

        return {
            "history_bars": history_bars,
            "avg_volume": avg_volume,
            "avg_dollar_volume": avg_dollar_volume,
            "median_dollar_volume": median_dollar_volume,
            "liquidity_stability": liquidity_stability,
            "missing_pct": missing_pct,
            "annualized_vol": annualized_vol,
            "recent_tradability": recent_tradability,
            "quality_score": score,
            "eligible": len(reasons) == 0,
            "reasons": reasons,
        }

    def _quality_score(
        self,
        history_bars: int,
        avg_dollar_volume: float,
        liquidity_stability: float,
        missing_pct: float,
        annualized_vol: float,
        recent_tradability: float,
    ) -> float:
        w = self.config.data.universe_quality_score_weights
        history_target = max(float(self.config.data.universe_min_history_bars), 1.0)
        dollar_target = max(float(self.config.data.universe_min_avg_dollar_volume), 1.0)
        max_missing = max(float(self.config.data.universe_max_missing_data_pct), 1e-9)
        max_vol = max(float(self.config.data.universe_max_volatility), 1e-9)
        tradability_target = max(float(self.config.data.universe_min_recent_tradability), 1e-9)

        history_score = min(1.0, float(history_bars) / history_target)
        volume_level_score = min(1.0, float(avg_dollar_volume) / dollar_target)
        volume_consistency_score = min(1.0, max(0.0, float(liquidity_stability)))
        missing_quality_score = max(0.0, min(1.0, 1.0 - float(missing_pct) / max_missing))
        volatility_score = max(0.0, min(1.0, 1.0 - float(annualized_vol) / max_vol))
        tradability_score = min(1.0, float(recent_tradability) / tradability_target)

        score = (
            float(w.get("history", 0.0)) * history_score
            + float(w.get("volume_level", 0.0)) * volume_level_score
            + float(w.get("volume_consistency", 0.0)) * volume_consistency_score
            + float(w.get("missing_quality", 0.0)) * missing_quality_score
            + float(w.get("volatility_stability", 0.0)) * volatility_score
            + float(w.get("tradability", 0.0)) * tradability_score
        )
        return float(max(0.0, min(1.0, score)))
