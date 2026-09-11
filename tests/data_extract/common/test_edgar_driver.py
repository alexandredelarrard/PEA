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
from src.data_extract.utils.common.registrant import Registrant, Segment
from src.data_extract.utils.common.run_manifest import get_entry
from src.data_extract.utils.common.sec_utils import CIK_MAPPING_COLS
from src.data_store import schema
from src.data_store.schema import Table, Tables
from tests.data_extract.fake_context import extract_config

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
    """A Context stand-in carrying the four attributes the driver touches.

    `sp500_tickers` is seeded with EVERY column `load_cik_mapping` projects, not just the two
    the driver itself reads: the projection is server-side, so a column missing from this
    fixture fails as a `KeyError` inside the SELECT rather than as a missing value."""
    store.save(Tables.sp500_tickers,
               pd.DataFrame({col: [str(i + 1) if col == "cik" else f"{col}-{t}"
                                   for i, t in enumerate(tickers)]
                             for col in CIK_MAPPING_COLS} | {"ticker": tickers}))
    warnings: list[str] = []
    ctx = types.SimpleNamespace(
        store=store,
        paths={"DATA_STORE": tmp_path},
        log=types.SimpleNamespace(info=lambda *a, **k: None,
                                  warning=lambda msg, *a: warnings.append(msg % a)),
        config=extract_config(data_extract={"manifest_full_rescan_days": 30}),
        ensure_edgar_identity=lambda: None)
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
    monkeypatch.setattr("edgar.Company",
                        lambda t: types.SimpleNamespace(get_filings=lambda form: listed))

    out = new_filings("AAPL", ["8-K"], pd.Timestamp("2021-01-01"), frozenset({"c"}))

    # "a" is pre-since, "c" is already stored -> only b, d survive, oldest first
    assert [f.accession_number for f in out] == ["b", "d"]

    print("\n=== SANITY CHECK: new_filings dedup + since + order ===")
    print("  4 listed, 1 stored, 1 pre-since -> ['b', 'd'] oldest-first. Validated.")


def test_new_filings_without_since_returns_everything_sorted(monkeypatch):
    listed = [_filing("c", "2023-06-01"), _filing("a", "2020-01-01")]
    monkeypatch.setattr("edgar.Company",
                        lambda t: types.SimpleNamespace(get_filings=lambda form: listed))

    got = new_filings("AAPL", ["8-K"], None, frozenset())

    assert [f.accession_number for f in got] == ["a", "c"]

    print("\n=== SANITY CHECK: new_filings with since=None ===")
    print("  no cutoff -> both filings kept, still oldest-first. Validated.")


# --------------------------------------------------------------------------- #
# new_filings across a registrant cutover                                      #
# --------------------------------------------------------------------------- #
#: XOM's real shape, measured 2026-09-09. `Company("XOM")` resolves to the PREDECESSOR (EDGAR
#: has not remapped the ticker), so the successor's filings are invisible without the register.
_CUTOVER_DATE = pd.Timestamp("2026-07-01")


def _xom_cutover():
    """XOM's real two-segment chain, as a `Registrant`."""
    return Registrant(ticker="XOM", kind="reorganisation", segments=(
        Segment(cik="0000034088", valid_from=None, valid_to=_CUTOVER_DATE,
                evidence="test fixture"),
        Segment(cik="0002115436", valid_from=_CUTOVER_DATE, valid_to=None,
                evidence="test fixture")))


def _patch_registrants(monkeypatch, by_cik: dict, cutover=None):
    """`Company(x)` -> that registrant's filings, and the register -> `cutover` or empty.

    ⚠ PATCHES `edgar.Company`, NOT `edgar_driver.Company`. `new_filings` is now a wrapper over
    `registrant.resolve_registrant_filings`, which imports `Company` at call time from the
    lowest layer -- so patching the driver's own name silently patches nothing and every one
    of these tests reaches the live SEC API instead. That failure mode is loud (an
    `IdentityNotSetError` from inside a retry wrapper) but it reads as a network fault rather
    than a stale patch point, so it is called out here.

    Keys are what the resolver passes: the TICKER string for the ticker-resolved lookup, and
    an `int` CIK for each segment.
    """
    monkeypatch.setattr("edgar.Company",
                        lambda x: types.SimpleNamespace(
                            get_filings=lambda form: by_cik.get(x, [])))
    monkeypatch.setattr("src.data_extract.utils.common.registrant.load_registrants",
                        lambda *a, **k: ({"XOM": cutover} if cutover else {}))


def test_new_filings_unions_the_successor_registrant(monkeypatch):
    """The defect this fixes: three real XOM 8-Ks reached no table at all, because the ticker
    resolves to the predecessor and nothing walked the successor."""
    _patch_registrants(monkeypatch, {
        "XOM": [_filing("pred-old", "2026-05-01")],
        2115436: [_filing("suc-1", "2026-07-07"), _filing("suc-2", "2026-08-28")],
    }, cutover=_xom_cutover())

    out = new_filings("XOM", ["8-K"], None, frozenset())

    assert [f.accession_number for f in out] == ["pred-old", "suc-1", "suc-2"]

    print("\n=== SANITY CHECK: cutover union recovers the successor's filings ===")
    print("  successor-only accessions ['suc-1', 'suc-2'] now reachable. Validated.")


def test_new_filings_keeps_a_predecessor_filing_dated_after_the_cutover(monkeypatch):
    """⚠ THE ANTI-REGRESSION TEST, and the reason this path UNIONS where `cutover_filings`
    SPLITS. XOM's SCHEDULE 13G of 2026-08-07 is filed under the PREDECESSOR, five weeks after
    the 2026-07-01 boundary. A dated split would discard it -- a filing already in the
    database -- so applying the fundamentals rule to the event pipelines loses data."""
    _patch_registrants(monkeypatch, {
        "XOM": [_filing("pred-late", "2026-08-07")],
        2115436: [_filing("suc-1", "2026-07-07")],
    }, cutover=_xom_cutover())

    out = new_filings("XOM", ["SCHEDULE 13G"], None, frozenset())
    kept = [f.accession_number for f in out]

    assert "pred-late" in kept, "a dated split would have dropped this"
    assert pd.Timestamp(out[-1].filing_date) > _CUTOVER_DATE
    assert kept == ["suc-1", "pred-late"]

    print("\n=== SANITY CHECK: predecessor filings after the cutover survive ===")
    print("  'pred-late' (2026-08-07, past a 2026-07-01 boundary) kept. Validated.")


def test_new_filings_takes_a_co_indexed_document_once(monkeypatch):
    """XOM's 2026-08-03 10-Q carries ONE accession indexed under BOTH CIKs -- which is why
    fundamentals stayed clean while `sec_8k` lost filings. It must not arrive twice."""
    shared = _filing("0000034088-26-000093", "2026-08-03")
    _patch_registrants(monkeypatch, {
        "XOM": [shared], 34088: [shared], 2115436: [shared],
    }, cutover=_xom_cutover())

    out = new_filings("XOM", ["10-Q"], None, frozenset())

    assert [f.accession_number for f in out] == ["0000034088-26-000093"]

    print("\n=== SANITY CHECK: a co-indexed accession is taken once ===")
    print("  same accession under both registrants -> 1 filing. Validated.")


def test_new_filings_survives_an_unresolvable_cutover_cik(monkeypatch):
    """A dead CIK in the register must cost that registrant's filings, not the whole walk."""
    def _company(x):
        if x == 2115436:
            raise ValueError("no such company")
        return types.SimpleNamespace(get_filings=lambda form: [_filing("pred", "2026-05-01")])

    monkeypatch.setattr("edgar.Company", _company)
    monkeypatch.setattr("src.data_extract.utils.common.registrant.load_registrants",
                        lambda *a, **k: {"XOM": _xom_cutover()})

    out = new_filings("XOM", ["8-K"], None, frozenset())

    assert [f.accession_number for f in out] == ["pred"]

    print("\n=== SANITY CHECK: a dead cutover CIK does not kill the walk ===")
    print("  successor unresolvable -> predecessor's filing still returned. Validated.")


def test_def14a_forms_covers_contested_proxies_but_not_revised_ones():
    """The contested proxy REPLACES the annual DEF 14A, so 42 ticker-years across 37 tickers
    were invisible to every governance feature. DEFR14A is excluded on measurement, not taste:
    220 of its 232 ticker-years already hold the DEF 14A, so it is a duplicate far more often
    than a recovery."""
    from src.constants.constants import DEF14A_FORMS

    assert "DEFC14A" in DEF14A_FORMS
    assert "DEFR14A" not in DEF14A_FORMS
    for not_an_annual_meeting in ("DEFM14A", "DEFS14A", "DEFN14A"):
        assert not_an_annual_meeting not in DEF14A_FORMS

    print("\n=== SANITY CHECK: DEF14A_FORMS scope ===")
    print(f"  {DEF14A_FORMS} -- contested in, revised and non-annual out. Validated.")


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


def test_run_edgar_fetch_reraises_a_programming_error_instead_of_warning(
        tmp_path, sqlite_store, monkeypatch):
    """The contrast with `..._isolates_a_failing_ticker` above, and the reason that test's
    `RuntimeError` is not a `NameError`.

    A `NameError` in `xbrl_linkbase.statement_arcs` was logged as "fundamentals: NEM failed"
    by the very handler that isolates a bad ticker, so three tickers lost every fact they had
    while the run reported success for 10.6 h. A defect in this repo is not a bad ticker: it
    will hit every remaining ticker too, so the pool must abort and the run must fail.
    """
    ctx = _ctx(tmp_path, sqlite_store, ["AAPL", "MSFT"])

    def build(ticker, cik, *, since, done_accessions):
        if ticker == "AAPL":
            raise NameError("name 'cols' is not defined")
        return {_T_MAIN: _rows(_T_MAIN, ticker, f"{ticker}-1")}

    with pytest.raises(NameError, match="cols"):
        run_edgar_fetch(ctx, ["AAPL", "MSFT"], 15, tables=(_T_MAIN,), build=build,
                        desc="test")

    assert ctx.warnings == []                            # not downgraded to a warning
    assert get_entry(ctx, _T_MAIN) is None               # nothing recorded -> the run retries

    print("\n=== SANITY CHECK: driver re-raises a programming error ===")
    print(f"  build raised NameError -> run_edgar_fetch propagated it; "
          f"warnings logged: {len(ctx.warnings)}, manifest entry: "
          f"{get_entry(ctx, _T_MAIN)}.")
    print("  -> A repo defect now FAILS the run; a bad ticker (RuntimeError, test above) "
          "is still isolated.")


def test_run_edgar_fetch_survives_a_save_failure_without_aborting_the_pool(
        tmp_path, sqlite_store, monkeypatch):
    """The regression guard for the pre-refactor bug: every fetcher saved OUTSIDE its
    per-ticker try, so one DB error propagated through `future.result()` and killed the
    whole pool."""
    ctx = _ctx(tmp_path, sqlite_store, ["AAPL"])

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

    def build(ticker, cik, *, since, done_accessions):
        return {_T_MAIN: _rows(_T_MAIN, ticker, "x"), _T_CHILD: _rows(_T_CHILD, ticker, "x")}

    run_edgar_fetch(ctx, ["AAPL"], 15, tables=(_T_MAIN,), build=build, desc="test")

    assert sqlite_store.row_count(_T_MAIN) == 1
    assert not sqlite_store.exists(_T_CHILD)        # not declared -> not written

    print("\n=== SANITY CHECK: driver ignores an undeclared table ===")
    print("  build returned driver_child but only driver_main was declared -> child not "
          "written (it would never get a manifest entry). Validated.")
