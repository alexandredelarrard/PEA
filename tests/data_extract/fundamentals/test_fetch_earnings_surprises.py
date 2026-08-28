"""Tests for the incremental earnings-surprise extraction
(src/data_extract/fetch_earnings_surprises.py).

The network download is not exercised here; what matters is the INCREMENTAL
plan: unseen tickers get a full pull, stale tickers (no reported quarter within
the refetch window) get a small top-up pull, and up-to-date tickers are skipped.
"""
from __future__ import annotations

import types

import numpy as np
import pandas as pd

from src.data_extract.utils.fundamentals import fetch_earnings_surprises as surprises
from src.data_extract.utils.fundamentals.fetch_earnings_surprises import (
    _plan_fetch, _RECENT_LIMIT)


def test_plan_fetch_incremental():
    today = pd.Timestamp.today().normalize()
    existing = pd.DataFrame({
        "ticker": ["FRESH", "FRESH", "STALE", "NOACTUAL"],
        "earnings_date": [today - pd.Timedelta(days=10),   # FRESH: reported recently
                          today - pd.Timedelta(days=100),
                          today - pd.Timedelta(days=200),   # STALE: last report old
                          today + pd.Timedelta(days=20)],   # only a FUTURE estimate
        "eps_estimate": [1.0, 1.0, 1.0, 2.0],
        "eps_actual": [1.1, 1.0, 0.9, np.nan],              # NOACTUAL has no reported row
        "surprise_pct": [10.0, 0.0, -10.0, np.nan],
    })
    tickers = ["FRESH", "STALE", "NOACTUAL", "NEW"]
    plan = dict(_plan_fetch(tickers, existing, full_limit=44, refetch_window_days=80))

    assert "FRESH" not in plan, "recently-reported ticker must be skipped"
    assert plan["STALE"] == _RECENT_LIMIT, "stale ticker -> small top-up pull"
    assert plan["NEW"] == 44, "unseen ticker -> full pull"
    # NOACTUAL has only a forward estimate (no reported quarter) -> treated as unseen
    assert plan["NOACTUAL"] == 44

    print("\n=== SANITY CHECK: incremental fetch plan ===")
    print(f"  FRESH skipped; STALE={plan.get('STALE')} (top-up); "
          f"NEW={plan.get('NEW')} (full); NOACTUAL={plan.get('NOACTUAL')} (full).")
    print("  Only what remains to extract is fetched -> no redundant re-downloads.")


def test_plan_fetch_no_existing_pulls_all():
    plan = dict(_plan_fetch(["A", "B", "C"], None, full_limit=20, refetch_window_days=80))
    assert plan == {"A": 20, "B": 20, "C": 20}
    print("\n=== SANITY CHECK: cold start ===")
    print("  no history file -> every ticker gets a full pull.")


def test_the_no_data_branch_returns_after_recording_exactly_one_run(monkeypatch):
    """The `not parts` branch: no cache and nothing fetched.

    It recorded the run and then FELL THROUGH to `new["ticker"].nunique()` on a column-less
    frame -- `KeyError: 'ticker'` every single time -- so the double `record_run` the branch
    also carried was never observed. Both are the missing `return`.
    """
    calls: list[tuple] = []
    saved: list = []

    context = types.SimpleNamespace(
        store=types.SimpleNamespace(load=lambda *a, **k: None,
                                    save=lambda table, df: saved.append((table, df))),
        log=types.SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None),
        config=types.SimpleNamespace(data_extract=types.SimpleNamespace(years_history=15)))

    monkeypatch.setattr(surprises, "record_run",
                        lambda ctx, table, n_tickers, rows, **k: calls.append(
                            (table.name, n_tickers, rows)))
    monkeypatch.setattr(surprises, "_download_one", lambda tkr, limit: None)

    # No exception is the assertion: this raised KeyError('ticker') before the `return`.
    surprises.fetch_earnings_surprises(context, ["AAPL", "MSFT"], pause=0.0)

    assert len(calls) == 1, f"expected exactly one recorded run, got {calls}"
    assert calls[0][2] == 0, "an empty run must record rows_added=0"
    assert saved == [], "nothing to save -- the empty upsert is skipped with the crash"

    print(f"\n=== SANITY CHECK: earnings-surprises empty branch ===")
    print(f"  no cache + no Yahoo calendar -> returned cleanly, "
          f"record_run called {len(calls)} time(s): {calls}")
    print("  -> One run recorded with rows_added=0; the KeyError('ticker') fall-through is "
          "gone.")
