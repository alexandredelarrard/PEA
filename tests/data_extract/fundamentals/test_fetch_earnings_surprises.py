"""Tests for the incremental earnings-surprise extraction
(src/data_extract/fetch_earnings_surprises.py).

The network download is not exercised here; what matters is the INCREMENTAL
plan: unseen tickers get a full pull, stale tickers (no reported quarter within
the refetch window) get a small top-up pull, and up-to-date tickers are skipped.
"""

from __future__ import annotations

import types
from typing import Any

import numpy as np
import pandas as pd

from src.data_extract.utils.fundamentals import fetch_earnings_surprises as surprises
from src.data_extract.utils.fundamentals.fetch_earnings_surprises import _RECENT_LIMIT, _plan_fetch, _resume_dates
from src.data_store.schema import Tables


def test_plan_fetch_incremental(sqlite_store):
    today = pd.Timestamp.today().normalize()
    existing = pd.DataFrame(
        {
            "ticker": ["FRESH", "FRESH", "STALE", "NOACTUAL"],
            "earnings_date": [
                today - pd.Timedelta(days=10),  # FRESH: reported recently
                today - pd.Timedelta(days=100),
                today - pd.Timedelta(days=200),  # STALE: last report old
                today + pd.Timedelta(days=20),
            ],  # only a FUTURE estimate
            "eps_estimate": [1.0, 1.0, 1.0, 2.0],
            "eps_actual": [1.1, 1.0, 0.9, np.nan],  # NOACTUAL has no reported row
            "surprise_pct": [10.0, 0.0, -10.0, np.nan],
        }
    )
    tickers = ["FRESH", "STALE", "NOACTUAL", "NEW"]
    sqlite_store.save(Tables.earnings_surprises, existing.assign(earnings_date=existing["earnings_date"].dt.strftime("%Y-%m-%d")))
    context: Any = types.SimpleNamespace(store=sqlite_store)
    last_reported, next_expected = _resume_dates(context)
    plan = dict(_plan_fetch(tickers, last_reported, next_expected, full_limit=44, refetch_window_days=80))

    assert "FRESH" not in plan, "recently-reported ticker must be skipped"
    assert plan["STALE"] == _RECENT_LIMIT, "stale ticker -> small top-up pull"
    assert plan["NEW"] == 44, "unseen ticker -> full pull"
    # NOACTUAL has only a forward estimate (no reported quarter) -> treated as unseen
    assert plan["NOACTUAL"] == 44

    print("\n=== SANITY CHECK: incremental fetch plan ===")
    print(f"  FRESH skipped; STALE={plan.get('STALE')} (top-up); NEW={plan.get('NEW')} (full); NOACTUAL={plan.get('NOACTUAL')} (full).")
    print("  Only what remains to extract is fetched -> no redundant re-downloads.")


def test_plan_fetch_no_existing_pulls_all():
    plan = dict(_plan_fetch(["A", "B", "C"], {}, {}, full_limit=20, refetch_window_days=80))
    assert plan == {"A": 20, "B": 20, "C": 20}
    print("\n=== SANITY CHECK: cold start ===")
    print("  no stored row -> every ticker gets a full pull.")


def test_a_run_that_fetches_nothing_returns_without_saving(sqlite_store, monkeypatch):
    """No stored row and no Yahoo calendar: the plan reads the empty table through `key_stats`,
    nothing is saved and the run returns (no manifest is written)."""
    infos: list[str] = []
    context: Any = types.SimpleNamespace(
        store=sqlite_store,
        log=types.SimpleNamespace(info=lambda msg, *a: infos.append(msg % a if a else msg), warning=lambda *a, **k: None),
    )
    monkeypatch.setattr(surprises, "_download_one", lambda tkr, limit: None)

    surprises.fetch_earnings_surprises(context, ["AAPL", "MSFT"], years_history=15, pause=0.0)

    assert not sqlite_store.exists(Tables.earnings_surprises)
    assert infos[0] == "Earnings surprises: 2/2 tickers to fetch (0 already current)" and infos[-1] == "Earnings surprises: nothing new fetched."
    print("\n=== SANITY CHECK: earnings-surprises empty run ===")
    print(f"  cold table + no Yahoo calendar -> planned 2 full pulls, saved nothing, returned: {infos[-1]!r}")
