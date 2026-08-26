"""Tests for the shared EDGAR fetch driver
(`src/data_extract/utils/common/edgar_driver.py`): the filing-listing helper and
the per-ticker thread-pool driver that the 8-K / 13D / DEF 14A / filing-text
fetchers all delegate to. Offline -- a real `DataStore` on SQLite plus a stub
`Company`, no network.
"""
from __future__ import annotations

import threading
import time
import types

import pandas as pd
import pytest

from src.data_extract.utils.common import edgar_driver
from src.data_extract.utils.common.edgar_driver import new_filings, run_edgar_fetch
from src.data_extract.utils.common.run_manifest import get_entry
from src.data_store import schema
from src.data_store.schema import Table, Tables

_T_MAIN = Table("driver_main", ("ticker", "accession_number"), date_col="filing_date")
_T_CHILD = Table("driver_child", ("ticker", "accession_number"), date_col="filing_date")
_T_EMPTY = Table("driver_empty", ("ticker", "accession_number"), date_col="filing_date")


@pytest.fixture(autouse=True)
def _register_test_tables(monkeypatch):
    """`store.resolve` only accepts registered tables, by design. These three exist so the
    driver is exercised on its own multi-table contract rather than on a real fetcher's."""
    for table in (_T_MAIN, _T_CHILD, _T_EMPTY):
        monkeypatch.setitem(schema.BY_NAME, table.name, table)


def _filing(accession: str, filing_date: str):
    return types.SimpleNamespace(accession_number=accession, filing_date=filing_date)


def _ctx(tmp_path, store, tickers):
    """A Context stand-in carrying the four attributes the driver touches."""
    store.save(Tables.sp500_tickers,
               pd.DataFrame({"ticker": tickers,
                             "cik": [str(i + 1) for i in range(len(tickers))]}))
    warnings: list[str] = []
    ctx = types.SimpleNamespace(
        store=store,
        paths={"DATA_STORE": tmp_path},
        log=types.SimpleNamespace(info=lambda *a, **k: None,
                                  warning=lambda msg, *a: warnings.append(msg % a)),
        config=types.SimpleNamespace(
            data_extract=types.SimpleNamespace(manifest_full_rescan_days=30)))
    ctx.warnings = warnings
    return ctx


def _rows(table, ticker, accession):
    return pd.DataFrame([{"ticker": ticker, "accession_number": accession,
                          "filing_date": pd.Timestamp("2024-01-02"),
                          "src": str(table)}])


# --------------------------------------------------------------------------- #
# new_filings                                                                  #
# --------------------------------------------------------------------------- #
def test_new_filings_drops_done_accessions_filters_since_and_sorts_oldest_first(monkeypatch):
    listed = [_filing("c", "2023-06-01"), _filing("a", "2020-01-01"),
              _filing("d", "2024-03-01"), _filing("b", "2021-05-01")]
    monkeypatch.setattr(edgar_driver, "Company",
                        lambda t: types.SimpleNamespace(get_filings=lambda form: listed))

    out = new_filings("AAPL", ["8-K"], pd.Timestamp("2021-01-01"), frozenset({"c"}))

    # "a" is pre-since, "c" is already stored -> only b, d survive, oldest first
    assert [f.accession_number for f in out] == ["b", "d"]

    print("\n=== SANITY CHECK: new_filings dedup + since + order ===")
    print("  4 listed, 1 stored, 1 pre-since -> ['b', 'd'] oldest-first. Validated.")


def test_new_filings_without_since_returns_everything_sorted(monkeypatch):
    listed = [_filing("c", "2023-06-01"), _filing("a", "2020-01-01")]
    monkeypatch.setattr(edgar_driver, "Company",
                        lambda t: types.SimpleNamespace(get_filings=lambda form: listed))

    got = new_filings("AAPL", ["8-K"], None, frozenset())

    assert [f.accession_number for f in got] == ["a", "c"]

    print("\n=== SANITY CHECK: new_filings with since=None ===")
    print("  no cutoff -> both filings kept, still oldest-first. Validated.")


# --------------------------------------------------------------------------- #
# run_edgar_fetch                                                              #
# --------------------------------------------------------------------------- #
def test_run_edgar_fetch_saves_every_declared_table_and_records_each(tmp_path, sqlite_store,
                                                                    monkeypatch):
    # One ticker on purpose: `sqlite_store` shares ONE connection across the pool's threads,
    # so concurrent writes to it are not a reliable assertion. Concurrency is covered by
    # `test_run_edgar_fetch_serializes_writes_until_a_cold_table_exists`, which instruments
    # `save` instead of racing it.
    ctx = _ctx(tmp_path, sqlite_store, ["AAPL"])
    monkeypatch.setattr(edgar_driver, "configure_identity", lambda: None)

    def build(ticker, cik, *, since, done_accessions):
        return {_T_MAIN: _rows(_T_MAIN, ticker, f"{ticker}-1"),
                _T_CHILD: _rows(_T_CHILD, ticker, f"{ticker}-1"),
                _T_EMPTY: pd.DataFrame()}

    run_edgar_fetch(ctx, ["AAPL"], 15,
                    tables=(_T_MAIN, _T_CHILD, _T_EMPTY), build=build, desc="test")

    assert sqlite_store.row_count(_T_MAIN) == 1
    assert sqlite_store.row_count(_T_CHILD) == 1
    # the table no ticker produced rows for STILL gets a manifest entry, else it would
    # read as "never run" and full-rescan forever
    assert get_entry(ctx, _T_EMPTY)["rows_added"] == 0
    assert get_entry(ctx, _T_MAIN)["rows_added"] == 1
    assert get_entry(ctx, _T_MAIN)["ticker_count"] == 1

    print("\n=== SANITY CHECK: driver multi-table save + manifest ===")
    print("  main + child rows saved; all 3 declared tables recorded, including the one no "
          "ticker produced rows for (rows_added=0). Validated.")


def test_run_edgar_fetch_serializes_writes_until_a_cold_table_exists(
        tmp_path, sqlite_store, monkeypatch):
    """`store.ensure_table` is a check-then-create with no locking, so several workers can
    each see a cold table missing and race the CREATE -- the state of every
    rebuild-from-scratch. The driver must serialize the first write per table; once the
    table exists, saves run concurrently again."""
    tickers = [f"TK{i}" for i in range(12)]
    ctx = _ctx(tmp_path, sqlite_store, tickers)
    monkeypatch.setattr(edgar_driver, "configure_identity", lambda: None)

    probe_lock = threading.Lock()
    state = {"in_flight": 0, "peak_cold": 0, "peak_warm": 0, "calls": 0}

    def instrumented_save(table, df, pk=None):
        with probe_lock:
            state["in_flight"] += 1
            state["calls"] += 1
            key = "peak_cold" if state["calls"] <= 1 else "peak_warm"
            state[key] = max(state[key], state["in_flight"])
        time.sleep(0.01)                       # widen the window a real CREATE would occupy
        with probe_lock:
            state["in_flight"] -= 1
        return len(df)

    monkeypatch.setattr(sqlite_store, "save", instrumented_save)

    def build(ticker, cik, *, since, done_accessions):
        return {_T_MAIN: _rows(_T_MAIN, ticker, f"{ticker}-1")}

    run_edgar_fetch(ctx, tickers, 15, tables=(_T_MAIN,), build=build, desc="test")

    assert ctx.warnings == []
    assert state["calls"] == len(tickers)
    assert state["peak_cold"] == 1             # the creating write never overlaps another
    assert state["peak_warm"] > 1              # afterwards the lock is out of the way

    print("\n=== SANITY CHECK: cold-table write serialization ===")
    print(f"  {state['calls']} writes on 8 threads: the first (table-creating) write ran "
          f"alone (peak concurrency {state['peak_cold']}), later writes overlapped freely "
          f"(peak {state['peak_warm']}). Validated.")


def test_run_edgar_fetch_isolates_a_failing_ticker(tmp_path, sqlite_store, monkeypatch):
    ctx = _ctx(tmp_path, sqlite_store, ["AAPL", "MSFT"])
    monkeypatch.setattr(edgar_driver, "configure_identity", lambda: None)

    def build(ticker, cik, *, since, done_accessions):
        if ticker == "AAPL":
            raise RuntimeError("edgar exploded")
        return {_T_MAIN: _rows(_T_MAIN, ticker, f"{ticker}-1")}

    run_edgar_fetch(ctx, ["AAPL", "MSFT"], 15, tables=(_T_MAIN,), build=build, desc="test")

    stored = sqlite_store.load(_T_MAIN)
    assert list(stored["ticker"]) == ["MSFT"]
    assert get_entry(ctx, _T_MAIN)["rows_added"] == 1

    print("\n=== SANITY CHECK: driver isolates a failing ticker ===")
    print("  AAPL raised, MSFT's row still landed and the run was recorded. Validated.")


def test_run_edgar_fetch_survives_a_save_failure_without_aborting_the_pool(
        tmp_path, sqlite_store, monkeypatch):
    """The regression guard for the pre-refactor bug: every fetcher saved OUTSIDE its
    per-ticker try, so one DB error propagated through `future.result()` and killed the
    whole pool."""
    ctx = _ctx(tmp_path, sqlite_store, ["AAPL"])
    monkeypatch.setattr(edgar_driver, "configure_identity", lambda: None)

    real_save = sqlite_store.save

    def flaky_save(table, df, pk=None):
        if table is _T_MAIN:
            raise RuntimeError("deadlock detected")
        return real_save(table, df, pk)

    monkeypatch.setattr(sqlite_store, "save", flaky_save)

    def build(ticker, cik, *, since, done_accessions):
        return {_T_MAIN: _rows(_T_MAIN, ticker, "x"), _T_CHILD: _rows(_T_CHILD, ticker, "x")}

    run_edgar_fetch(ctx, ["AAPL"], 15, tables=(_T_MAIN, _T_CHILD), build=build, desc="test")

    assert not sqlite_store.exists(_T_MAIN)              # the failing save is swallowed
    assert sqlite_store.row_count(_T_CHILD) == 1        # the sibling table still landed
    assert get_entry(ctx, _T_MAIN)["rows_added"] == 0

    print("\n=== SANITY CHECK: driver survives a save failure ===")
    print("  save to driver_main raised; driver_child still saved and the pool completed "
          "instead of aborting. Validated.")


def test_run_edgar_fetch_passes_manifest_window_and_dedup_set_to_build(
        tmp_path, sqlite_store, monkeypatch):
    ctx = _ctx(tmp_path, sqlite_store, ["AAPL"])
    monkeypatch.setattr(edgar_driver, "configure_identity", lambda: None)
    sqlite_store.save(_T_MAIN, _rows(_T_MAIN, "AAPL", "already-stored"))

    seen: dict = {}

    def build(ticker, cik, *, since, done_accessions):
        seen["since"] = since
        seen["done"] = done_accessions
        return {}

    run_edgar_fetch(ctx, ["AAPL"], 15, tables=(_T_MAIN,), build=build, desc="test")

    # first run -> no manifest entry -> the full years_history fallback window
    assert seen["since"].year == (pd.Timestamp.today() - pd.DateOffset(years=15)).year
    assert seen["done"] == frozenset({"already-stored"})

    print("\n=== SANITY CHECK: driver window + dedup wiring ===")
    print(f"  cold manifest -> since={seen['since'].date()} (15y back); "
          f"done_accessions read from the table: {sorted(seen['done'])}. Validated.")


def test_run_edgar_fetch_rejects_an_undeclared_table(tmp_path, sqlite_store, monkeypatch):
    ctx = _ctx(tmp_path, sqlite_store, ["AAPL"])
    monkeypatch.setattr(edgar_driver, "configure_identity", lambda: None)

    def build(ticker, cik, *, since, done_accessions):
        return {_T_MAIN: _rows(_T_MAIN, ticker, "x"), _T_CHILD: _rows(_T_CHILD, ticker, "x")}

    run_edgar_fetch(ctx, ["AAPL"], 15, tables=(_T_MAIN,), build=build, desc="test")

    assert sqlite_store.row_count(_T_MAIN) == 1
    assert not sqlite_store.exists(_T_CHILD)        # not declared -> not written

    print("\n=== SANITY CHECK: driver ignores an undeclared table ===")
    print("  build returned driver_child but only driver_main was declared -> child not "
          "written (it would never get a manifest entry). Validated.")
