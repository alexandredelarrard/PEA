"""Sharadar resume windows and failed pages, offline (`sqlite_store`, a fake `sharadar_get`).

SF1 resumes per ticker from its own last filing date minus the contract overlap (95 days); a new
ticker gets the whole window; the market-wide tables resume from their last date minus 7 days. A page
that still fails after the retrying GET fails the request whole: the ticker is counted as failed and
nothing partial is saved.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

from src.data_extract.utils.fundamentals_sharadar import client as client_mod
from src.data_extract.utils.fundamentals_sharadar import fetch_sharadar as fetch_mod
from src.data_extract.utils.fundamentals_sharadar.client import SharadarRequestError
from src.data_store.schema import Resume, Tables
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
    monkeypatch.setattr(client_mod, "PAGE_LIMIT", 2)
    monkeypatch.setattr(client_mod, "_api_key", lambda: "key")
    monkeypatch.setattr(client_mod, "_page", lambda context, url, params: next(pages))

    with pytest.raises(SharadarRequestError, match="offset 2"):
        get: Any = client_mod.sharadar_get
        get(None, "fundamentals", ticker="AAA", **{"date.gte": "2026-01-01"})
    print("\n=== SANITY CHECK: no partial page ===")
    print("  page 1 served 2 rows, page 2 failed -> SharadarRequestError, not the 2-row prefix.")


def test_the_membership_table_resumes_from_its_last_date_minus_seven_days(tmp_path, sqlite_store, fake):
    ctx = _context(tmp_path, sqlite_store, ["AAA"])

    fetch_mod.fetch_sharadar_sp500(ctx)
    sqlite_store.save(Tables.sharadar_sp500, pd.DataFrame({"date": [pd.Timestamp("2026-09-29")], "ticker": "AAA", "action": "added"}))
    fetch_mod.fetch_sharadar_sp500(ctx)

    assert [day for _, day in fake.requests] == [cast(Resume, Tables.sharadar_sp500.resume).source_start, "2026-09-22"]
    print("\n=== SANITY CHECK: membership window ===")
    print("  cold sp500 -> 1990-01-01; last 2026-09-29 -> 2026-09-22.")


def _actions(rows: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    """A Sharadar `actions` response body: `(date, ticker, action, contraticker)` rows, values as the API sends them."""
    return pd.DataFrame(
        {
            "date": [r[0] for r in rows],
            "action": [r[2] for r in rows],
            "ticker": [r[1] for r in rows],
            "name": "N",
            "value": ["1.0" for _ in rows],
            "contraticker": [r[3] for r in rows],
            "contraname": "",
        }
    )


def test_actions_are_pulled_whole_and_replace_the_table(tmp_path, sqlite_store, monkeypatch: pytest.MonkeyPatch):
    """PD-5: every run re-pulls the whole window and replaces the table, so a stored hole closes and a stale row goes."""
    ctx = _context(tmp_path, sqlite_store, ["AAA"])
    stored_before = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-05-01", "2026-09-29"]),
            "action": ["split", "dividend"],
            "ticker": ["OLD", "AAA"],
            "name": "N",
            "value": [2.0, 0.5],
            "contraticker": "N/A",
            "contraname": "",
        }
    )
    sqlite_store.save(Tables.sharadar_actions, stored_before)
    served = _actions(
        [
            ("2022-06-09", "META", "tickerchangefrom", "FB"),
            ("2022-06-09", "FB", "tickerchangeto", "META"),
            ("2022-06-09", "FB", "tickerchangeto", "META"),
            ("2026-09-29", "AAA", "dividend", "N/A"),
            ("2024-03-04", "BRK.B", "split", "N/A"),
        ]
    )
    windows: list[tuple[str, str]] = []

    def serve(context: Any, table: str, /, **filters: Any) -> pd.DataFrame:
        windows.append((filters["date.gte"], filters["date.lte"]))
        return served.loc[served["date"].between(filters["date.gte"], filters["date.lte"])].reset_index(drop=True)

    monkeypatch.setattr(fetch_mod, "sharadar_get", serve)

    written = fetch_mod.fetch_sharadar_actions(ctx, years_history=10, as_of=_AS_OF)
    stored = sqlite_store.load(Tables.sharadar_actions)

    ordered = sorted(windows)
    assert ordered[0][0] == "2016-09-30" and ordered[-1][1] == "2026-09-30", "the whole window, not the stored max date minus the overlap"
    assert all(pd.Timestamp(a[1]) + pd.Timedelta(days=1) == pd.Timestamp(b[0]) for a, b in zip(ordered, ordered[1:], strict=False)), "windows tile"
    assert len(windows) == 41 and ordered[1][0] == "2016-10-01", "quarterly windows from the floor to the run date"
    assert written == 4 and len(stored) == 4, "one duplicated PK row dropped"
    assert "OLD" not in set(stored["ticker"]), "a row Sharadar no longer serves is gone"
    assert {"META", "FB", "AAA", "BRK-B"} == set(stored["ticker"]), "share-class spelling mapped to the repo's"
    assert stored.loc[stored["ticker"].eq("META"), "contraticker"].tolist() == ["FB"]

    monkeypatch.setattr(fetch_mod, "sharadar_get", FakeSharadar(failing={None}))
    assert fetch_mod.fetch_sharadar_actions(ctx, years_history=10, as_of=_AS_OF) == 0
    assert len(sqlite_store.load(Tables.sharadar_actions)) == 4, "a failed pull leaves the table unchanged"
    print("\n=== SANITY CHECK: sharadar_actions full replace (PD-5) ===")
    print(
        f"  {len(windows)} tiling windows 2016-09-30..2026-09-30 (as_of - 10y); 5 served rows -> 4 stored (1 duplicate PK); stale OLD split removed;"
    )
    print("  FB->META tickerchange rows land; BRK.B -> BRK-B; a failed pull writes nothing and keeps the 4 rows. Validated.")


def test_a_full_page_halves_its_window_and_a_truncated_pull_never_replaces(tmp_path, sqlite_store, monkeypatch: pytest.MonkeyPatch):
    """The API caps offset paging per query, so a window whose page is full is split, never paged; a pull far smaller than
    the stored table is refused."""
    ctx = _context(tmp_path, sqlite_store, ["AAA"])
    days = pd.date_range("2026-07-01", "2026-09-30", freq="D").strftime("%Y-%m-%d")
    served = _actions([(day, f"T{i}", "dividend", "N/A") for i, day in enumerate(days)])
    windows: list[tuple[str, str]] = []

    def serve(context: Any, table: str, /, **filters: Any) -> pd.DataFrame:
        windows.append((filters["date.gte"], filters["date.lte"]))
        return served.loc[served["date"].between(filters["date.gte"], filters["date.lte"])].reset_index(drop=True)

    monkeypatch.setattr(fetch_mod, "sharadar_get", serve)
    monkeypatch.setattr(fetch_mod, "PAGE_LIMIT", 30)

    written = fetch_mod.fetch_sharadar_actions(ctx, years_history=1, as_of=_AS_OF)

    quarter = [w for w in windows if w[0] >= "2026-07-01"]
    split = sorted(w for w in quarter if len(served[served["date"].between(*w)]) < 30)
    assert written == len(days) == 92 and len(sqlite_store.load(Tables.sharadar_actions)) == 92
    assert sum(len(served[served["date"].between(*w)]) for w in split) == 92, "the kept windows hold every row once"
    assert split[0][0] == "2026-07-01" and split[-1][1] == "2026-09-30" and len(quarter) == len(split) * 2 - 1, "a binary split of the quarter"
    monkeypatch.setattr(
        fetch_mod, "sharadar_get", lambda context, table, /, **f: served.iloc[:5].loc[served.iloc[:5]["date"].between(f["date.gte"], f["date.lte"])]
    )
    assert fetch_mod.fetch_sharadar_actions(ctx, years_history=1, as_of=_AS_OF) == 0
    assert len(sqlite_store.load(Tables.sharadar_actions)) == 92, "5 rows pulled vs 92 stored: the table is kept"
    print("\n=== SANITY CHECK: windowed actions pull ===")
    print(
        f"  page limit 30: the 92-row 2026Q3 window split into {len(split)} windows, 92 rows written; a 5-row pull vs 92 stored -> refused. Validated."
    )


def test_membership_requests_follow_the_registered_source_start(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sqlite_store: Any, fake: FakeSharadar
) -> None:
    resume = replace(cast(Resume, Tables.sharadar_sp500.resume), source_start="2001-04-01")
    monkeypatch.setattr(Tables, "sharadar_sp500", replace(Tables.sharadar_sp500, resume=resume))
    ctx = _context(tmp_path, sqlite_store, ["AAA"])
    fetch_mod.fetch_sharadar_sp500(ctx)
    sqlite_store.save(Tables.sharadar_sp500, pd.DataFrame({"date": [pd.Timestamp("2026-09-29")], "ticker": "AAA", "action": "added"}))
    fetch_mod.fetch_sharadar_sp500(ctx)
    fetch_mod.fetch_sharadar_sp500(ctx, full=True)
    assert [day for _, day in fake.requests] == ["2001-04-01", "2026-09-22", "2001-04-01"]
    print("SANITY: changing the membership contract moves cold/full requests to 2001-04-01 and preserves the seven-day incremental overlap.")
