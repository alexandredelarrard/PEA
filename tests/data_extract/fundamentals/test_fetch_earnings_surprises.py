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


class _CappedYahooTicker:
    """Stands in for `yf.Ticker` over a fixed quarterly history ending 2026-07-30. Like yfinance 1.5,
    it refuses a limit above 100, serves `limit` rows from `offset` (newest first), returns None past
    the end, and caches by `limit` alone per instance (a reused instance ignores a new offset)."""

    calls: list[tuple[int, int]] = []
    history: pd.DatetimeIndex = pd.date_range(end="2026-07-30", periods=150, freq="91D", tz="America/New_York")

    def __init__(self, ticker: str) -> None:
        self._ticker = ticker
        self._cache: dict[int, pd.DataFrame | None] = {}

    def get_earnings_dates(self, limit: int = 12, offset: int = 0) -> pd.DataFrame | None:
        type(self).calls.append((limit, offset))
        if limit > 100:
            raise ValueError("Yahoo caps limit at 100")
        if limit in self._cache:
            return self._cache[limit]
        newest_first = type(self).history[::-1][offset : offset + limit]
        page = None
        if len(newest_first):
            page = pd.DataFrame({"EPS Estimate": 1.0, "Reported EPS": 1.1, "Surprise(%)": 10.0}, index=pd.Index(newest_first, name="Earnings Date"))
        self._cache[limit] = page
        return page


def test_a_new_ticker_full_pull_stays_within_the_yahoo_cap(sqlite_store, monkeypatch):
    """A never-seen ticker under the live `years_history` (31 -> 128 quarters) is asked for at most
    Yahoo's 100 rows, so its history is saved instead of the call failing every night."""
    warnings: list[str] = []
    context: Any = types.SimpleNamespace(
        store=sqlite_store,
        log=types.SimpleNamespace(info=lambda *a, **k: None, warning=lambda msg, *a: warnings.append(msg % a if a else msg)),
    )
    _CappedYahooTicker.calls = []
    monkeypatch.setattr(surprises.yf, "Ticker", _CappedYahooTicker)

    surprises.fetch_earnings_surprises(context, ["HOG"], years_history=31, pause=0.0)

    # the first 100-row page already reaches the 2002-10-01 cutoff, so no second page is asked for
    assert _CappedYahooTicker.calls == [(100, 0)], f"full pull must ask for Yahoo's maximum page, asked {_CappedYahooTicker.calls}"
    assert not any("failed" in w for w in warnings), warnings
    saved = sqlite_store.load(Tables.earnings_surprises)
    assert set(saved["ticker"]) == {"HOG"} and len(saved) > 0
    assert pd.to_datetime(saved["earnings_date"]).min() >= pd.Timestamp(surprises.MIGRATION_DATE)
    print("\n=== SANITY CHECK: new-ticker full pull under the Yahoo cap ===")
    print(f"  years_history=31 (128 quarters wanted) -> calls (limit, offset)={_CappedYahooTicker.calls}, no failure,")
    print(f"  saved {len(saved)} HOG rows from {pd.to_datetime(saved['earnings_date']).min().date()} (cutoff {surprises.MIGRATION_DATE}).")


def test_a_full_pull_longer_than_one_yahoo_page_is_read_page_by_page(sqlite_store, monkeypatch):
    """When the wanted history (31 years -> 128 quarters) after the cutoff is longer than Yahoo's
    100-row page, the full pull reads the next page from offset 100 on a fresh Ticker, so the
    oldest quarters are not silently lost. The cutoff is moved to 1990 to make the history long."""
    context: Any = types.SimpleNamespace(store=sqlite_store, log=types.SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None))
    _CappedYahooTicker.calls = []
    monkeypatch.setattr(surprises.yf, "Ticker", _CappedYahooTicker)
    monkeypatch.setattr(surprises, "MIGRATION_DATE", "1990-01-01")

    surprises.fetch_earnings_surprises(context, ["HOG"], years_history=31, pause=0.0)

    saved = pd.to_datetime(sqlite_store.load(Tables.earnings_surprises)["earnings_date"]).sort_values()
    wanted = _CappedYahooTicker.history[-128:].tz_localize(None).normalize()
    assert len(saved) == 128 and saved.iloc[0] == wanted[0], (
        f"saved {len(saved)} rows from {saved.iloc[0].date()}, wanted 128 from {wanted[0].date()}"
    )
    assert [offset for _, offset in _CappedYahooTicker.calls] == [0, 100], _CappedYahooTicker.calls
    print("\n=== SANITY CHECK: full pull across two Yahoo pages ===")
    print(f"  calls (limit, offset)={_CappedYahooTicker.calls}; saved {len(saved)} quarters from {saved.iloc[0].date()} to {saved.iloc[-1].date()},")
    print("  the full 128 wanted, none duplicated, instead of the newest 100 only.")
