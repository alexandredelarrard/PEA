"""Ticker -> CIK, name and GICS come from `context.universe` (the `tickers_universe` rows); `load_cik_mapping` is gone.

test_universe_rows_are_the_requested_universe_rows — `universe_rows` keeps the 10-digit CIK, the name and GICS.
test_fundamentals_gics_come_from_the_universe — the XBRL regime GICS dict equals the universe rows.
"""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace
from typing import Any

from src.data_extract.utils.edgar_extract.edgar_driver import universe_rows
from src.data_extract.utils.fundamentals.fetch_fundamentals_sec import fundamentals_fetch
from tests.support.universe import seed_universe

ROWS = {
    "XOM": {"name": "ExxonMobil", "cik": "0000034088", "sector": "Energy", "industry_group": "Energy", "sub_industry": "Integrated Oil & Gas"},
    "AAPL": {
        "name": "Apple",
        "cik": "0000320193",
        "sector": "Information Technology",
        "industry_group": "Technology Hardware & Equipment",
        "sub_industry": "Technology Hardware, Storage & Peripherals",
    },
    "JPM": {"name": "JPMorgan", "cik": "0000019617", "sector": "Financials", "industry_group": "Banks", "sub_industry": "Diversified Banks"},
}


def test_universe_rows_are_the_requested_universe_rows(sqlite_store):
    context: Any = SimpleNamespace(universe=seed_universe(sqlite_store, list(ROWS), rows=ROWS))

    rows = universe_rows(context, ["XOM", "AAPL"])
    by_ticker = {r["ticker"]: r for _, r in rows.iterrows()}

    assert set(by_ticker) == {"XOM", "AAPL"}
    assert by_ticker["AAPL"]["cik"] == "0000320193" and by_ticker["XOM"]["name"] == "ExxonMobil"
    assert {"name", "sector", "industry_group", "sub_industry"} <= set(rows.columns)
    assert importlib.util.find_spec("src.data_extract.utils.edgar_extract.sec_utils") is None
    print("\n=== SANITY CHECK: universe rows ===")
    print(f"  {sorted(by_ticker)} from context.universe; CIK {by_ticker['AAPL']['cik']}, name and GICS kept; sec_utils is gone. Validated.")


def test_fundamentals_gics_come_from_the_universe(sqlite_store):
    context: Any = SimpleNamespace(universe=seed_universe(sqlite_store, list(ROWS), rows=ROWS), config_dir="./configs")

    gics = fundamentals_fetch(context).parse.keywords["gics_by_ticker"]

    levels = ("sector", "industry_group", "sub_industry")
    assert gics == {ticker: {level: row[level] for level in levels} for ticker, row in ROWS.items()}
    print("\n=== SANITY CHECK: XBRL regime GICS ===")
    print(f"  {len(gics)} tickers, three GICS levels each, equal to the tickers_universe rows. Validated.")
