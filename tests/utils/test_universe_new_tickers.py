"""`sp500_tickers.added_on`: the new-ticker window and the scraper's stamping rule."""

from __future__ import annotations

import datetime as dt
import logging
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from src.data_extract.utils.prices import fetch_tickers
from src.data_store.schema import Tables
from src.utils.universe import new_tickers
from tests.conftest import FakeStore

_AS_OF = pd.Timestamp("2026-10-03")

_PAGE = """
<table>
  <tr><th>Symbol</th><th>Security</th><th>GICS Sector</th><th>GICS Sub-Industry</th><th>CIK</th></tr>
  <tr><td>AAPL</td><td>Apple Inc.</td><td>Information Technology</td><td>Technology Hardware, Storage &amp; Peripherals</td><td>320193</td></tr>
  <tr><td>MSFT</td><td>Microsoft</td><td>Information Technology</td><td>Systems Software</td><td>789019</td></tr>
  <tr><td>NEWCO</td><td>New Company</td><td>Industrials</td><td>Industrial Machinery &amp; Supplies &amp; Components</td><td>1999999</td></tr>
</table>
"""


def _ctx(store: Any) -> Any:
    return SimpleNamespace(store=store, log=logging.getLogger("test.universe_new_tickers"))


@pytest.fixture
def stub_page(monkeypatch: pytest.MonkeyPatch) -> None:
    response = SimpleNamespace(text=_PAGE, raise_for_status=lambda: None)
    monkeypatch.setattr(fetch_tickers.requests, "get", lambda *args, **kwargs: response)


def _universe(added_on: list[object]) -> pd.DataFrame:
    return pd.DataFrame({"ticker": ["OLD", "MID", "NEW", "NUL"], "cik": ["1", "2", "3", "4"], "added_on": pd.to_datetime(pd.Series(added_on))})


def test_new_tickers_window(sqlite_store) -> None:
    sqlite_store.save(Tables.sp500_tickers, _universe(["2000-01-01", "2026-09-26", "2026-10-02", None]))

    week = new_tickers(sqlite_store, 7, _AS_OF)
    quarter = new_tickers(sqlite_store, 95, _AS_OF)
    assert week == {"NEW"}  # 2026-09-26 is exactly 7 days back: outside the window
    assert quarter == {"MID", "NEW"}
    assert new_tickers(sqlite_store, 7, pd.Timestamp("2026-10-10")) == set()

    print("\n=== SANITY CHECK: new-ticker window ===")
    print(f"  as_of {_AS_OF.date()}: 7-day overlap -> {sorted(week)}, 95-day -> {sorted(quarter)}; NULL and 2000-01-01 are established. Validated.")


def test_new_tickers_without_the_column_or_table(sqlite_store) -> None:
    assert new_tickers(sqlite_store, 7, _AS_OF) == set()
    sqlite_store.save(Tables.sp500_tickers, pd.DataFrame({"ticker": ["AAPL"], "cik": ["320193"]}))
    assert new_tickers(sqlite_store, 7, _AS_OF) == set()

    print("\n=== SANITY CHECK: pre-migration universe ===")
    print("  absent table and a table without added_on -> no new ticker. Validated.")


def test_scraper_keeps_stored_added_on_and_stamps_new_tickers(sqlite_store, stub_page) -> None:
    sqlite_store.save(
        Tables.sp500_tickers,
        pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT"],
                "name": ["Apple", "Microsoft Corp"],
                "sector": ["Information Technology"] * 2,
                "industry_group": [None, None],
                "sub_industry": [None, None],
                "cik": ["0000320193", "0000789019"],
                "added_on": pd.to_datetime(pd.Series(["2000-01-01", None])),
            }
        ),
    )

    fetch_tickers.get_sp500_tickers(_ctx(sqlite_store), as_of=_AS_OF)

    df = sqlite_store.load(Tables.sp500_tickers, columns=["ticker", "added_on", "name"])
    assert df is not None
    added = dict(zip(df["ticker"], df["added_on"], strict=True))
    assert added["AAPL"] == dt.date(2000, 1, 1)
    assert added["MSFT"] is None
    assert added["NEWCO"] == _AS_OF.date()
    assert set(df["name"].dropna()) == {"Apple Inc.", "Microsoft", "New Company"}
    assert new_tickers(sqlite_store, 7, _AS_OF) == {"NEWCO"}

    fetch_tickers.get_sp500_tickers(_ctx(sqlite_store), as_of=_AS_OF + pd.Timedelta(days=30))
    again = sqlite_store.load(Tables.sp500_tickers, columns=["ticker", "added_on"])
    assert again is not None
    assert dict(zip(again["ticker"], again["added_on"], strict=True))["NEWCO"] == _AS_OF.date()

    print("\n=== SANITY CHECK: scraper stamping ===")
    print(f"  AAPL keeps 2000-01-01, MSFT keeps NULL, NEWCO stamped {_AS_OF.date()} and keeps it on a later re-scrape. Validated.")


def test_scraper_does_not_add_the_column_before_migration(stub_page) -> None:
    store = FakeStore({Tables.sp500_tickers: pd.DataFrame({"ticker": ["AAPL"], "cik": ["0000320193"]})})

    fetch_tickers.get_sp500_tickers(_ctx(store), as_of=_AS_OF)

    saved = store.saved_frames(Tables.sp500_tickers)
    assert len(saved) == 1 and "added_on" not in saved[0].columns
    assert set(saved[0]["ticker"]) == {"AAPL", "MSFT", "NEWCO"}

    print("\n=== SANITY CHECK: no implicit migration ===")
    print("  a stored table without added_on gets a frame without it: the save cannot ALTER the table or stamp 500 tickers as new. Validated.")
