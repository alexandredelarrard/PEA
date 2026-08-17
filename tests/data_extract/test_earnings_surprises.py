"""
Earnings-surprise fetch planning. The incremental planner must only (re)fetch a
ticker when there is genuinely new data — otherwise ~30% of the S&P 500 were pulled
every run in the ~10-day gap before their next report (fixed-window shorter than the
~91-day quarterly cycle). It now gates on the forward earnings date yfinance returns.

test_plan_fetch_uses_forward_date — unseen -> full; future forward -> skip; passed
    forward -> re-pull; no-forward falls back to the staleness window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_extract.utils.fundamentals.fetch_earnings_surprises import _plan_fetch, _RECENT_LIMIT

_COLS = ["ticker", "earnings_date", "eps_estimate", "eps_actual", "surprise_pct"]


def _row(t, days_from_today, actual):
    d = pd.Timestamp.today().normalize() + pd.Timedelta(days=days_from_today)
    return {"ticker": t, "earnings_date": d, "eps_estimate": 1.0,
            "eps_actual": actual, "surprise_pct": np.nan}


def test_plan_fetch_uses_forward_date():
    FULL = 24
    existing = pd.DataFrame([
        _row("B", -60, 1.0), _row("B", +30, np.nan),   # reported 60d ago + forward 30d FUTURE -> skip
        _row("C", -120, 1.0), _row("C", -10, np.nan),  # forward 10d ago has PASSED -> re-pull
        _row("D", -120, 1.0),                          # no forward, 120d stale (>95) -> re-pull (fallback)
        _row("E", -30, 1.0),                           # no forward, only 30d old -> skip (fallback)
    ], columns=_COLS)

    plan = dict(_plan_fetch(["A", "B", "C", "D", "E"], existing, full_limit=FULL,
                            refetch_window_days=95))

    assert plan.get("A") == FULL, "unseen ticker -> full pull"
    assert "B" not in plan, "next earnings still in the future -> skip"
    assert plan.get("C") == _RECENT_LIMIT, "forward date passed -> re-pull"
    assert plan.get("D") == _RECENT_LIMIT, "no forward date + stale beyond window -> re-pull"
    assert "E" not in plan, "no forward date but recent -> skip"

    print("\n=== SANITY CHECK: earnings plan gates on forward date ===")
    print(f"  A(unseen)->full({plan['A']}); B(forward future)->skip; C(forward passed)->pull; "
          f"D(no-fwd stale)->pull; E(no-fwd recent)->skip")
    print("  -> tickers waiting on a future report are no longer re-pulled every run. Validated.")
