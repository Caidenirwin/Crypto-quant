"""
Offline tests for the anti-rug guard (no network). Run: python tests/test_anti_rug.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anti_rug import AntiRugGuard, RugVerdict  # noqa: E402
from config import Config  # noqa: E402

GUARD = AntiRugGuard(Config())
IDX = pd.date_range("2025-01-01", periods=120)


def frame(close, vol=None) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    vol = np.full(len(close), 1e6) if vol is None else np.asarray(vol, dtype=float)
    return pd.DataFrame(
        {"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close, "Volume": vol},
        index=IDX[: len(close)],
    )


def history(df: pd.DataFrame) -> RugVerdict:
    return GUARD.history_check("X", df, df.index[-1])


def evm(res: dict) -> RugVerdict:
    v = RugVerdict("X")
    GUARD._check_evm(res, v)
    return v


def solana(res: dict) -> RugVerdict:
    v = RugVerdict("X")
    GUARD._check_solana(res, v)
    return v


RENOUNCED = "0x000000000000000000000000000000000000dead"
LIVE_OWNER = "0x1234567890abcdef1234567890abcdef12345678"

# (name, verdict, expect_allow_entry, expect_force_exit)
CASES = [
    ("healthy coin", history(frame(np.linspace(1, 1.2, 120))), True, False),
    ("too young (60d)", history(frame(np.linspace(1, 1.2, 60))), False, False),
    ("-40% in one day", history(frame(list(np.linspace(1, 1.2, 119)) + [0.72])), False, True),
    ("-60% from 14d high", history(frame(list(np.linspace(1, 1.2, 110)) + list(np.linspace(1.1, 0.45, 10)))), False, True),
    ("volume dried up", history(frame(np.linspace(1, 1.2, 120), [1e6] * 115 + [1e5] * 5)), False, True),
    ("+300% pump in 7 days", history(frame(list(np.full(113, 1.0)) + list(np.linspace(1.5, 4.0, 7)))), False, False),
    ("EVM honeypot", evm({"is_honeypot": "1", "owner_address": RENOUNCED}), False, True),
    ("EVM live owner can mint", evm({"is_mintable": "1", "owner_address": LIVE_OWNER}), False, False),
    ("EVM renounced + mintable", evm({"is_mintable": "1", "owner_address": RENOUNCED}), True, False),
    ("EVM 25% sell tax", evm({"sell_tax": "0.25", "owner_address": ""}), False, True),
    ("EVM whale holds 85%", evm({"owner_address": "", "holders": [
        {"address": "0xabc", "percent": "0.85", "is_locked": 0, "tag": ""}]}), False, False),
    ("EVM 85% burned (ok)", evm({"owner_address": "", "holders": [
        {"address": RENOUNCED, "percent": "0.85", "is_locked": 0, "tag": ""}]}), True, False),
    ("SOL freeze authority", solana({"freezable": {"status": "1"}}), False, True),
    ("SOL mint authority", solana({"mintable": {"status": "1"}}), False, True),
    ("SOL clean (BONK/WIF-like)", solana({"mintable": {"status": "0"}, "freezable": {"status": "0"},
                                          "transfer_fee": {}, "transfer_hook": []}), True, False),
]


def main() -> int:
    failures = 0
    for name, v, want_entry, want_exit in CASES:
        ok = v.allow_entry == want_entry and v.force_exit == want_exit
        failures += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {name:28s} entry={v.allow_entry!s:5s} exit={v.force_exit!s:5s} "
              f"{'; '.join(v.reasons)}")
    print(f"\n{len(CASES) - failures}/{len(CASES)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
