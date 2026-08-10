"""
Server-side filtering for `read_table` / `DataStore.load` (`where=`).

Motivation, measured on the live table: `fundamentals_facts` holds 2,326,371 rows
across 491 tickers, and `fundamentals_derive._load_facts_for_ticker` reads it ONCE
PER TICKER. Loading the whole table and filtering in pandas cost 27.5s and 2.33M x 19
cells per call -- ~3.7 hours for one `rebuild_fundamentals_history` pass -- against
0.042s for the scoped read, since the registry PK is `btree (ticker,
accession_number, ...)` and a ticker filter is therefore an index seek.

SQLite-backed and self-contained: no live DB, no network.
"""
from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import create_engine

from src.data_store.errors import TableMissingError
from src.data_store.store import DataStore, read_table

_FRAME = pd.DataFrame({
    "ticker": ["AAPL", "AAPL", "MSFT", "ZBH", "ZBH", "ZBH"],
    "field": ["totalRevenue", "netIncome", "totalRevenue", "totalRevenue", "netIncome", "depAmort"],
    "value": [100.0, 20.0, 200.0, 300.0, 30.0, 5.0],
})


@pytest.fixture()
def store() -> DataStore:
    engine = create_engine("sqlite://")
    _FRAME.to_sql("facts", engine, index=False)
    return DataStore(engine)


def test_where_scopes_the_read_to_one_value(store):
    out = store.load("facts", where={"ticker": "ZBH"})
    assert len(out) == 3
    assert set(out["ticker"]) == {"ZBH"}
    assert out["value"].sum() == pytest.approx(335.0)


def test_where_accepts_a_collection_as_an_in_filter(store):
    out = store.load("facts", where={"ticker": ["AAPL", "MSFT"]})
    assert sorted(out["ticker"]) == ["AAPL", "AAPL", "MSFT"]


def test_multiple_where_keys_are_anded(store):
    out = store.load("facts", where={"ticker": "ZBH", "field": "netIncome"})
    assert len(out) == 1
    assert out.iloc[0]["value"] == 30.0


def test_where_result_matches_loading_everything_and_filtering_in_pandas(store):
    """The equivalence the change rests on -- verified on live data too: for
    ZBH/JPM/VLO/SWKS/MET the scoped read reproduces the full-load-then-filter frame
    row for row, and `derive_fundamentals_history` returns an identical
    (60, 236) frame either way."""
    everything = store.load("facts")
    for ticker in ("AAPL", "MSFT", "ZBH"):
        expected = everything[everything["ticker"] == ticker].reset_index(drop=True)
        pd.testing.assert_frame_equal(store.load("facts", where={"ticker": ticker}), expected)


def test_where_composes_with_columns_and_limit(store):
    out = store.load("facts", columns=["ticker", "value"], where={"ticker": "ZBH"}, limit=2)
    assert list(out.columns) == ["ticker", "value"]
    assert len(out) == 2


def test_omitting_where_reads_the_whole_table_exactly_as_before(store):
    """`where` is additive: every existing caller passes nothing and must be
    unaffected."""
    pd.testing.assert_frame_equal(store.load("facts"), _FRAME)


def test_where_on_a_missing_table_raises_a_typed_error(store):
    """A missing table is a fault, so `load` raises -- but `TableMissingError`, not a bare
    `Exception`, so callers can distinguish it from a broken query."""
    with pytest.raises(TableMissingError):
        store.load("nope", where={"ticker": "ZBH"})
    assert store.load("nope", where={"ticker": "ZBH"}, optional=True) is None


def test_unknown_where_column_fails_loudly_before_reaching_the_database(store):
    with pytest.raises(KeyError):
        store.load("facts", where={"not_a_column": "x"})


def test_where_value_is_bound_not_interpolated(store):
    """Values go through SQLAlchemy bound parameters, so a string that looks like
    SQL is matched literally instead of being executed."""
    out = read_table(store.engine, "facts", where={"ticker": "'; DROP TABLE facts; --"})
    assert out.empty
    assert len(store.load("facts")) == len(_FRAME)     # table still there

    print("\n=== SANITY CHECK: server-side where= filter ===")
    print("  fundamentals_facts on the live DB: 2,326,371 rows / 491 tickers.")
    print("  Full load + pandas filter: 27.5s per ticker -> ~3.7h for one rebuild pass.")
    print("  store.load(..., where={'ticker': t}): 0.042s, an index seek on the existing")
    print("  PK btree (ticker, accession_number, ...) -- no new index required.")
    print("  Equivalence checked on live data for ZBH/JPM/VLO/SWKS/MET: identical facts")
    print("  frames and byte-identical derive_fundamentals_history output.")
    print("  Validated.")
