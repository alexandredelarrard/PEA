"""Sharadar resume windows and failed pages, offline (`sqlite_store`, a fake `sharadar_get`).

SF1 resumes per ticker from its own last filing date minus the contract overlap (95 days); a new
ticker gets the whole window; the market-wide tables resume from their last date minus 7 days. A page
that still fails after the retrying GET fails the request whole: the ticker is counted as failed and
nothing partial is saved.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from src.data_extract.utils.fundamentals_sharadar import client as client_mod
from src.data_extract.utils.fundamentals_sharadar import fetch_sharadar as fetch_mod
from src.data_extract.utils.fundamentals_sharadar.client import SharadarRequestError
from src.data_store.schema import Tables
from tests.data_extract.edgar_fixtures import fake_context

_AS_OF = pd.Timestamp("2026-09-30")


class FakeSharadar:
    """`sharadar_get` stand-in: records each request's ticker and `date.gte`; raises for `failing` tickers."""

    def __init__(self, failing: set[str] | None = None) -> None:
        self.failing = failing or set()
        self.requests: list[tuple[str | None, str]] = []

    def __call__(self, context: Any, table: str, /, **filters: Any) -> pd.DataFrame:
        ticker = filters.get("ticker")
        self.requests.append((ticker, filters["date.gte"]))
        if ticker in self.failing:
            raise SharadarRequestError(f"Sharadar {table}: page at offset 0 failed ({ticker})")
        return pd.DataFrame()


def _context(tmp_path, sqlite_store, tickers: list[str]) -> Any:
    ctx = fake_context(tmp_path, sqlite_store, tickers, sharadar_request_pace=0.0)
    ctx.log.debug = lambda msg, *a: None
    sqlite_store.save(
        Tables.sharadar_tickers,
        pd.DataFrame({"table": "SF1", "permaticker": range(len(tickers)), "ticker": tickers, "currency": "USD", "isdelisted": "N"}),
    )
    return ctx


def _sf1(ticker: str, dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ticker,
            "dimension": "ARQ",
            "calendardate": pd.to_datetime(dates),
            "datekey": pd.to_datetime(dates),
            "reportperiod": pd.to_datetime(dates),
            "date": pd.to_datetime(dates),
            "revenue": 1.0,
        }
    )


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeSharadar:
    stub = FakeSharadar()
    monkeypatch.setattr(fetch_mod, "sharadar_get", stub)
    monkeypatch.setattr(fetch_mod, "sleep_pace", lambda *a, **k: None)
    return stub


def test_sf1_resumes_each_ticker_from_its_last_date_minus_the_overlap(tmp_path, sqlite_store, fake):
    ctx = _context(tmp_path, sqlite_store, ["AAA", "LAG", "OLD"])
    sqlite_store.save(Tables.sharadar_fundamentals, pd.concat([_sf1("AAA", ["2026-08-01"]), _sf1("LAG", ["2025-11-01"])]))

    fetch_mod.fetch_sharadar_fundamentals(ctx, ["AAA", "LAG", "OLD"], years_history=10, as_of=_AS_OF)

    since = {ticker: day for ticker, day in fake.requests}
    assert since == {"AAA": "2026-04-28", "LAG": "2025-07-29", "OLD": "2026-04-28"}
    assert len(fake.requests) == 9  # three as-reported dimensions per ticker
    print("\n=== SANITY CHECK: SF1 per-ticker window ===")
    print(f"  AAA last 2026-08-01 -> {since['AAA']}; LAG last 2025-11-01 -> {since['LAG']} (own date - 95 d);")
    print(f"  OLD has no row and is not new -> the table frontier - 95 d ({since['OLD']}), per D1a.")


def test_a_failed_sf1_page_fails_the_ticker_and_is_named(tmp_path, sqlite_store, fake):
    ctx = _context(tmp_path, sqlite_store, ["AAA", "BAD"])
    sqlite_store.save(Tables.sharadar_fundamentals, pd.concat([_sf1("AAA", ["2026-08-01"]), _sf1("BAD", ["2026-08-01"])]))
    fake.failing = {"BAD"}

    fetch_mod.fetch_sharadar_fundamentals(ctx, ["AAA", "BAD"], years_history=10, as_of=_AS_OF)

    assert any("1 ticker(s) failed and are retried next run -> BAD" in w for w in ctx.warnings)
    assert sqlite_store.max_date_by(Tables.sharadar_fundamentals, "ticker")["BAD"] == pd.Timestamp("2026-08-01")
    print("\n=== SANITY CHECK: SF1 failed page ===")
    print("  BAD's request raised -> counted as failed (not 'up to date'), named in the coverage warning, frontier unchanged.")


def test_a_page_that_fails_mid_pagination_returns_nothing_partial(monkeypatch: pytest.MonkeyPatch):
    pages = iter(["ticker,date\nAAA,2026-01-01\nAAA,2026-02-01\n", None])
    monkeypatch.setattr(client_mod, "_PAGE_LIMIT", 2)
    monkeypatch.setattr(client_mod, "_api_key", lambda: "key")
    monkeypatch.setattr(client_mod, "_page", lambda context, url, params: next(pages))

    with pytest.raises(SharadarRequestError, match="offset 2"):
        get: Any = client_mod.sharadar_get
        get(None, "fundamentals", ticker="AAA", **{"date.gte": "2026-01-01"})
    print("\n=== SANITY CHECK: no partial page ===")
    print("  page 1 served 2 rows, page 2 failed -> SharadarRequestError, not the 2-row prefix.")


def test_the_market_wide_tables_resume_from_their_last_date_minus_seven_days(tmp_path, sqlite_store, fake):
    ctx = _context(tmp_path, sqlite_store, ["AAA"])
    sqlite_store.save(
        Tables.sharadar_actions, pd.DataFrame({"date": [pd.Timestamp("2026-09-29")], "ticker": "AAA", "action": "split", "contraticker": "N/A"})
    )

    fetch_mod.fetch_sharadar_actions(ctx, years_history=10, as_of=_AS_OF)
    fetch_mod.fetch_sharadar_sp500(ctx)
    fetch_mod.fetch_sharadar_actions(ctx, years_history=10, as_of=_AS_OF, full=True)

    assert [day for _, day in fake.requests] == ["2026-09-22", fetch_mod.SHARADAR_SP500_FIRST_DATE, "2016-09-30"]
    print("\n=== SANITY CHECK: market-wide windows ===")
    print("  actions last 2026-09-29 -> 2026-09-22; cold sp500 -> 1990-01-01; actions --full -> as_of - 10y.")
