"""
No-look-ahead test for the chart scanner. Run: python tests/test_chart_patterns.py

For several cut-off days t, pattern values computed from candles[:t] must equal
the values for day t computed from the FULL history. Any difference means a
pattern used future candles (which would make backtests lie).
Also prints how often each pattern fires on real data, as a sanity check.
"""

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chart_patterns import BINARY_PATTERNS, PATTERN_FEATURES, pattern_features  # noqa: E402
from config import Config  # noqa: E402
from data import DataManager  # noqa: E402
from ml_strategy import _atr  # noqa: E402


def main() -> int:
    cfg = Config()
    dm = DataManager(cfg)
    root = Path(__file__).resolve().parents[1]
    with redirect_stdout(io.StringIO()):
        df = dm.load_symbol_csv(str(root / "historic_data" / "XRP-CRYPTO - Sheet1.csv"), "XRP")["XRP"].sort_index()

    full = pattern_features(df, _atr(df))
    failures = 0
    for t in (400, 900, 1500, 2200, len(df) - 1):
        part_df = df.iloc[: t + 1]
        part = pattern_features(part_df, _atr(part_df))
        a = full.iloc[t][PATTERN_FEATURES].to_numpy(float)
        b = part.iloc[-1][PATTERN_FEATURES].to_numpy(float)
        same = np.isclose(a, b, equal_nan=True, rtol=1e-9, atol=1e-12)
        bad = [PATTERN_FEATURES[i] for i in np.where(~same)[0]]
        failures += bool(bad)
        print(f"{'PASS' if not bad else 'FAIL'}  cut-off {df.index[t].date()}  "
              f"{'no look-ahead' if not bad else 'LEAKS: ' + ', '.join(bad)}")

    print("\nHow often each pattern fires on XRP (sanity check):")
    for p in BINARY_PATTERNS:
        print(f"  {p:18s} {full[p].mean():6.1%}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
