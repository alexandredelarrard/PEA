"""Tests for the shared EDGAR fetch driver
(`src/data_extract/utils/common/edgar_driver.py`): the scope listing it resolves through, the
lineage-driven relist, the guard count and the per-ticker thread-pool driver that the 8-K / 13D /
DEF 14A / filing-text fetchers all delegate to. Offline -- a real `DataStore` on SQLite, a real
`Identity` over dated lineage rows and a stub `Company`, no network.
"""

from __future__ import annotations

import types
from typing import Any, cast

import pandas as pd
import pytest

from src.data_extract.utils.common import edgar_driver, frame_sanitize
from src.data_extract.utils.common.edgar_driver import (
    EdgarFetch,
    EdgarScope,
    FilingStamp,
    IncompleteEdgarRunError,
    build_filing_rows,
    run_edgar_fetch,
)
from src.data_extract.utils.common.identity import FilingScope
from src.data_extract.utils.common.registrant import resolve_registrant_filings
from src.data_extract.utils.common.run_manifest import get_entry as _get_entry
from src.data_extract.utils.common.run_manifest import record_run
from src.data_extract.utils.common.sec_utils import CIK_MAPPING_COLS
from src.data_store import schema
from src.data_store.schema import Table, Tables
from tests.data_extract.common.scope_fixtures import SENTINEL, dated_identity, patch_company
from tests.data_extract.fake_context import extract_config

_T_MAIN = Table("driver_main", ("ticker", "accession_number"), date_col="filing_date")
_T_CHILD = Table("driver_child", ("ticker", "accession_number"), date_col="filing_date")
_T_EMPTY = Table("driver_empty", ("ticker", "accession_number"), date_col="filing_date")
get_entry: Any = _get_entry


@pytest.fixture(autouse=True)
def _register_test_tables(monkeypatch):
    """`store.resolve` only accepts registered tables, by design. These three exist so the
    driver is exercised on its own multi-table contract rather than on a real fetcher's."""
    for table in (_T_MAIN, _T_CHILD, _T_EMPTY):
        monkeypatch.setitem(schema.BY_NAME, table.name, table)


#: The `_ctx` roster (AAPL = CIK 1, MSFT = CIK 2), each one open window.
_ROSTER_IDENTITY = dated_identity(
    [("AAPL", "0000000001", "cik_window", SENTINEL, None), ("MSFT", "0000000002", "cik_window", SENTINEL, None)],
    {"AAPL": "0000000001", "MSFT": "0000000002"},
)


@pytest.fixture(autouse=True)
def _identity(monkeypatch):
    """Every driver run reads the identity layer; the default is the `_ctx` roster with no scope change."""
    monkeypatch.setattr(edgar_driver, "load_identity", lambda context: _ROSTER_IDENTITY)


def _filing(accession: str, filing_date: str):
    return types.SimpleNamespace(accession_number=accession, filing_date=filing_date)


def _ctx(tmp_path, store, tickers) -> Any:
    """A Context stand-in carrying the four attributes the driver touches.

    `sp500_tickers` is seeded with EVERY column `load_cik_mapping` projects, not just the two
    the driver itself reads: the projection is server-side, so a column missing from this
    fixture fails as a `KeyError` inside the SELECT rather than as a missing value."""
    store.save(
        Tables.sp500_tickers,
        pd.DataFrame(
            {col: [str(i + 1) if col == "cik" else f"{col}-{t}" for i, t in enumerate(tickers)] for col in CIK_MAPPING_COLS} | {"ticker": tickers}
        ),
    )
    warnings: list[str] = []
    infos: list[str] = []
    ctx = types.SimpleNamespace(
        store=store,
        paths={"DATA_STORE": tmp_path},
        log=types.SimpleNamespace(info=lambda msg, *a, **k: infos.append(msg % a), warning=lambda msg, *a: warnings.append(msg % a)),
        config=extract_config(data_extract={"manifest_full_rescan_days": 30}),
        ensure_edgar_identity=lambda: None,
        config_dir=tmp_path,
    )
    ctx.warnings = warnings
    ctx.infos = infos
    return ctx


def _fetch(tables, build, desc="test", *, minimum_since=None, listing_since=None, since_by_ticker=None, done_where=None) -> EdgarFetch:
    return EdgarFetch(
        desc=desc,
        tables=tables,
        build=build,
        minimum_since=minimum_since,
        listing_since=listing_since,
        since_by_ticker=since_by_ticker,
        done_where=done_where,
    )


def _rows(table, ticker, accession):
    return pd.DataFrame([{"ticker": ticker, "accession_number": accession, "filing_date": pd.Timestamp("2024-01-02"), "src": str(table)}])


# --------------------------------------------------------------------------- #
# resolve_registrant_filings over a roster-only scope                          #
# --------------------------------------------------------------------------- #
def test_registrant_filings_drops_done_accessions_filters_since_and_sorts_oldest_first(monkeypatch):
    listed = [_filing("c", "2023-06-01"), _filing("a", "2020-01-01"), _filing("d", "2024-03-01"), _filing("b", "2021-05-01")]
    monkeypatch.setattr("edgar.Company", lambda cik: types.SimpleNamespace(get_filings=lambda form: listed))

    out = resolve_registrant_filings(
        FilingScope.roster_only("AAPL", "320193"), ["8-K"], since=pd.Timestamp("2021-01-01"), done_accessions=frozenset({"c"})
    )

    # "a" is pre-since, "c" is already stored -> only b, d survive, oldest first
    assert [f.accession_number for f in out] == ["b", "d"]

    print("\n=== SANITY CHECK: registrant filings dedup + since + order ===")
    print("  4 listed, 1 stored, 1 pre-since -> ['b', 'd'] oldest-first. Validated.")


def test_registrant_filings_without_since_returns_everything_sorted(monkeypatch):
    listed = [_filing("c", "2023-06-01"), _filing("a", "2020-01-01")]
    monkeypatch.setattr("edgar.Company", lambda cik: types.SimpleNamespace(get_filings=lambda form: listed))

    got = resolve_registrant_filings(FilingScope.roster_only("AAPL", "320193"), ["8-K"], since=None, done_accessions=frozenset())

    assert [f.accession_number for f in got] == ["a", "c"]

    print("\n=== SANITY CHECK: registrant filings with since=None ===")
    print("  no cutoff -> both filings kept, still oldest-first. Validated.")


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
def test_run_edgar_fetch_saves_every_declared_table_and_records_each(tmp_path, sqlite_store, monkeypatch):
    # One ticker on purpose: `sqlite_store` shares ONE connection across the pool's threads,
    # so concurrent writes to it are not a reliable assertion. The cold-table CREATE race is
    # covered at the store level by `tests/data_store/test_ensure_table_lock.py`.
    ctx = _ctx(tmp_path, sqlite_store, ["AAPL"])

    def build(ticker, cik, *, since, done_accessions, scope):
        return {_T_MAIN: _rows(_T_MAIN, ticker, f"{ticker}-1"), _T_CHILD: _rows(_T_CHILD, ticker, f"{ticker}-1"), _T_EMPTY: pd.DataFrame()}

    run_edgar_fetch(ctx, ["AAPL"], 15, _fetch((_T_MAIN, _T_CHILD, _T_EMPTY), build))

    assert sqlite_store.row_count(_T_MAIN) == 1
    assert sqlite_store.row_count(_T_CHILD) == 1
    # the table no ticker produced rows for STILL gets a manifest entry, else it would
    # read as "never run" and full-rescan forever
    assert get_entry(ctx, _T_EMPTY)["rows_added"] == 0
    assert get_entry(ctx, _T_MAIN)["rows_added"] == 1
    assert get_entry(ctx, _T_MAIN)["ticker_count"] == 1

    print("\n=== SANITY CHECK: driver multi-table save + manifest ===")
    print("  main + child rows saved; all 3 declared tables recorded, including the one no ticker produced rows for (rows_added=0). Validated.")


def test_a_failing_ticker_is_isolated_but_the_run_does_not_record_a_partial_success(tmp_path, sqlite_store, monkeypatch):
    ctx = _ctx(tmp_path, sqlite_store, ["AAPL", "MSFT"])

    def build(ticker, cik, *, since, done_accessions, scope):
        if ticker == "AAPL":
            raise RuntimeError("discovery page failed")
        return {_T_MAIN: _rows(_T_MAIN, ticker, f"{ticker}-1")}

    with pytest.raises(IncompleteEdgarRunError, match="no run manifest was advanced"):
        run_edgar_fetch(ctx, ["AAPL", "MSFT"], 15, _fetch((_T_MAIN,), build, "schedule test"))

    assert list(sqlite_store.load(_T_MAIN)["ticker"]) == ["MSFT"]
    assert get_entry(ctx, _T_MAIN) is None
    print("\n=== SANITY CHECK: incomplete run ===")
    print("  AAPL raised; MSFT's row still landed (pool not aborted), but the manifest did not advance")
    print("  OK: partial discovery cannot masquerade as a complete empty history")


def test_completeness_sensitive_success_marks_a_trustworthy_frontier(tmp_path, sqlite_store):
    ctx = _ctx(tmp_path, sqlite_store, ["AAPL"])

    def build(ticker, cik, *, since, done_accessions, scope):
        return {_T_MAIN: _rows(_T_MAIN, ticker, f"{ticker}-1")}

    run_edgar_fetch(ctx, ["AAPL"], 15, _fetch((_T_MAIN,), build, "schedule test"))

    entry = get_entry(ctx, _T_MAIN)
    assert entry is not None and entry.get("coverage_complete") is True
    assert entry.get("tickers") == ["AAPL"]
    print("\n=== SANITY CHECK: complete schedule frontier ===")
    print("  every ticker discovered and saved -> coverage_complete=true in the manifest")
    print("  OK: aggregation can distinguish this run from a legacy or partial walk")


def test_run_edgar_fetch_reraises_a_programming_error_instead_of_warning(tmp_path, sqlite_store, monkeypatch):
    """The contrast with `..._isolates_a_failing_ticker` above, and the reason that test's
    `RuntimeError` is not a `NameError`.

    A `NameError` in `xbrl_linkbase.statement_arcs` was logged as "fundamentals: NEM failed"
    by the very handler that isolates a bad ticker, so three tickers lost every fact they had
    while the run reported success for 10.6 h. A defect in this repo is not a bad ticker: it
    will hit every remaining ticker too, so the pool must abort and the run must fail.
    """
    ctx = _ctx(tmp_path, sqlite_store, ["AAPL", "MSFT"])

    def build(ticker, cik, *, since, done_accessions, scope):
        if ticker == "AAPL":
            raise NameError("name 'cols' is not defined")
        return {_T_MAIN: _rows(_T_MAIN, ticker, f"{ticker}-1")}

    with pytest.raises(NameError, match="cols"):
        run_edgar_fetch(ctx, ["AAPL", "MSFT"], 15, _fetch((_T_MAIN,), build))

    assert ctx.warnings == []  # not downgraded to a warning
    assert get_entry(ctx, _T_MAIN) is None  # nothing recorded -> the run retries

    print("\n=== SANITY CHECK: driver re-raises a programming error ===")
    print(
        f"  build raised NameError -> run_edgar_fetch propagated it; warnings logged: {len(ctx.warnings)}, manifest entry: {get_entry(ctx, _T_MAIN)}."
    )
    print("  -> A repo defect now FAILS the run; a bad ticker (RuntimeError, test above) is still isolated.")


def test_run_edgar_fetch_survives_a_save_failure_without_aborting_the_pool(tmp_path, sqlite_store, monkeypatch):
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

    def build(ticker, cik, *, since, done_accessions, scope):
        return {_T_MAIN: _rows(_T_MAIN, ticker, "x"), _T_CHILD: _rows(_T_CHILD, ticker, "x")}

    with pytest.raises(IncompleteEdgarRunError, match="no run manifest was advanced"):
        run_edgar_fetch(ctx, ["AAPL"], 15, _fetch((_T_MAIN, _T_CHILD), build))

    assert not sqlite_store.exists(_T_MAIN)  # the failing save is caught per table
    assert sqlite_store.row_count(_T_CHILD) == 1  # the sibling table still landed
    assert get_entry(ctx, _T_MAIN) is None and get_entry(ctx, _T_CHILD) is None

    print("\n=== SANITY CHECK: driver survives a save failure ===")
    print("  save to driver_main raised; driver_child still saved, the pool completed, and no manifest entry advanced. Validated.")


def test_completeness_sensitive_run_rejects_a_save_failure(tmp_path, sqlite_store, monkeypatch):
    ctx = _ctx(tmp_path, sqlite_store, ["AAPL"])

    def failed_save(table, df, pk=None):
        raise RuntimeError("deadlock detected")

    monkeypatch.setattr(sqlite_store, "save", failed_save)

    def build(ticker, cik, *, since, done_accessions, scope):
        return {_T_MAIN: _rows(_T_MAIN, ticker, "x")}

    with pytest.raises(IncompleteEdgarRunError, match="no run manifest was advanced"):
        run_edgar_fetch(ctx, ["AAPL"], 15, _fetch((_T_MAIN,), build, "schedule test"))

    assert not sqlite_store.exists(_T_MAIN)
    assert get_entry(ctx, _T_MAIN) is None
    print("\n=== SANITY CHECK: schedule persistence failure ===")
    print("  discovery succeeded but persistence failed; the completeness frontier was withheld")
    print("  OK: a storage failure cannot turn an unknown Schedule history into an observed zero")


def test_listing_since_overrides_full_and_the_manifest_window(tmp_path, sqlite_store):
    """`listing_since` is the window itself: it wins over `full=True` and over a complete manifest
    entry, applies to every ticker, and the run is recorded as an incremental (not full) rescan."""
    ctx = _ctx(tmp_path, sqlite_store, ["AAPL"])
    prior = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
    record_run(ctx, _T_MAIN, 1, 0, is_full_rescan=True, run_date=prior - pd.Timedelta(days=3), coverage_complete=True, tickers=["AAPL"])
    seen: list[pd.Timestamp] = []

    def build(ticker, cik, *, since, done_accessions, scope):
        seen.append(since)
        return {}

    listing = pd.Timestamp("2026-04-29 13:45")
    run_edgar_fetch(ctx, ["AAPL"], 15, _fetch((_T_MAIN,), build, listing_since=listing), full=True)
    run_edgar_fetch(ctx, ["AAPL"], 15, _fetch((_T_MAIN,), build, listing_since=listing))

    assert seen == [pd.Timestamp("2026-04-29"), pd.Timestamp("2026-04-29")]
    assert get_entry(ctx, _T_MAIN)["last_full_rescan_date"] == (prior - pd.Timedelta(days=3)).strftime("%Y-%m-%d"), "not a full rescan"
    assert get_entry(ctx, _T_MAIN)["coverage_complete"] is True
    print("\n=== SANITY CHECK: listing_since ===")
    print("  full=True and a complete manifest both yield since=2026-04-29 (normalised); the run is not recorded as a full rescan. Validated.")


def test_done_where_limits_the_dedup_set_to_matching_rows(tmp_path, sqlite_store):
    """`done_where` filters the stored-accession read, so rows outside it (another source) are re-listed."""
    ctx = _ctx(tmp_path, sqlite_store, ["AAPL"])
    sqlite_store.save(
        _T_MAIN, pd.concat([_rows(_T_MAIN, "AAPL", "from-edgar").assign(source="edgar"), _rows(_T_MAIN, "AAPL", "from-zip").assign(source="zip")])
    )
    seen: dict[str, frozenset[str]] = {}

    def build(ticker, cik, *, since, done_accessions, scope):
        seen[str(len(seen))] = done_accessions
        return {}

    run_edgar_fetch(ctx, ["AAPL"], 15, _fetch((_T_MAIN,), build, done_where={"source": "edgar"}))
    run_edgar_fetch(ctx, ["AAPL"], 15, _fetch((_T_MAIN,), build))

    assert seen["0"] == frozenset({"from-edgar"})
    assert seen["1"] == frozenset({"from-edgar", "from-zip"}), "no done_where keeps the old all-rows dedup"
    print("\n=== SANITY CHECK: done_where ===")
    print("  done_where={'source': 'edgar'} -> only 'from-edgar' is skipped; without it both stored accessions are. Validated.")


def test_run_edgar_fetch_passes_manifest_window_and_dedup_set_to_build(tmp_path, sqlite_store, monkeypatch):
    ctx = _ctx(tmp_path, sqlite_store, ["AAPL"])
    sqlite_store.save(_T_MAIN, _rows(_T_MAIN, "AAPL", "already-stored"))

    seen: dict = {}

    def build(ticker, cik, *, since, done_accessions, scope):
        seen["since"] = since
        seen["done"] = done_accessions
        return {}

    run_edgar_fetch(ctx, ["AAPL"], 15, _fetch((_T_MAIN,), build))

    # first run -> no manifest entry -> the full years_history fallback window
    assert seen["since"].year == (pd.Timestamp.today() - pd.DateOffset(years=15)).year
    assert seen["done"] == frozenset({"already-stored"})

    print("\n=== SANITY CHECK: driver window + dedup wiring ===")
    print(f"  cold manifest -> since={seen['since'].date()} (15y back); done_accessions read from the table: {sorted(seen['done'])}. Validated.")


def test_a_lineage_scope_change_relists_only_that_ticker_same_day_included(tmp_path, sqlite_store, monkeypatch):
    """P12: a ticker whose `scope_changed_at` is at or after the table's `last_run_date` relists the
    full window; the same day counts (a lineage rebuilt after this morning's run). Others stay incremental."""
    ctx = _ctx(tmp_path, sqlite_store, ["AAPL", "MSFT"])
    last_run = pd.Timestamp.today().normalize()
    record_run(ctx, _T_MAIN, ticker_count=2, rows_added=0, is_full_rescan=True, run_date=last_run, coverage_complete=True, tickers=["AAPL", "MSFT"])
    identity = dated_identity(
        [("AAPL", "0000000001", "cik_window", SENTINEL, None), ("MSFT", "0000000002", "cik_window", SENTINEL, None)],
        {"AAPL": "0000000001", "MSFT": "0000000002"},
        changed_at={"AAPL": last_run - pd.Timedelta(seconds=1), "MSFT": last_run},
    )
    monkeypatch.setattr(edgar_driver, "load_identity", lambda context: identity)
    seen: dict[str, pd.Timestamp] = {}

    def build(ticker, cik, *, since, done_accessions, scope):
        seen[ticker] = since
        return {}

    run_edgar_fetch(ctx, ["AAPL", "MSFT"], 15, _fetch((_T_MAIN,), build, "relist test"))

    assert seen["AAPL"] == last_run
    assert seen["MSFT"].year == (pd.Timestamp.today() - pd.DateOffset(years=15)).year
    assert "relist test: 1 ticker lineage scope(s) changed -> full-window relist: MSFT" in ctx.infos
    print("\n=== SANITY CHECK: lineage-driven relist ===")
    print(f"  MSFT scope_changed_at == last_run_date ({last_run.date()}) -> full window; AAPL changed one second earlier -> incremental")


def test_per_ticker_starts_keep_the_lineage_relist(tmp_path, sqlite_store, monkeypatch):
    """`since_by_ticker` gives each unchanged ticker its own start inside `listing_since`; a ticker whose
    scope changed relists from the `minimum_since`-floored window, unless its own start is earlier."""
    ctx = _ctx(tmp_path, sqlite_store, ["AAPL", "MSFT", "NVDA"])
    last_run = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
    record_run(ctx, _T_MAIN, 3, 0, is_full_rescan=True, run_date=last_run, coverage_complete=True, tickers=["AAPL", "MSFT", "NVDA"])
    identity = dated_identity(
        [(t, f"000000000{i}", "cik_window", SENTINEL, None) for i, t in enumerate(["AAPL", "MSFT", "NVDA"], start=1)],
        {"AAPL": "0000000001", "MSFT": "0000000002", "NVDA": "0000000003"},
        changed_at={"AAPL": last_run - pd.Timedelta(days=5), "MSFT": last_run, "NVDA": last_run},
    )
    monkeypatch.setattr(edgar_driver, "load_identity", lambda context: identity)
    seen: dict[str, pd.Timestamp] = {}

    def build(ticker, cik, *, since, done_accessions, scope):
        seen[ticker] = since
        return {}

    floor = pd.Timestamp("2026-07-01")
    starts = {"AAPL": pd.Timestamp("2026-09-20"), "MSFT": pd.Timestamp("2026-09-25"), "NVDA": pd.Timestamp("2026-03-02")}
    fetch = _fetch((_T_MAIN,), build, minimum_since=floor, listing_since=min(starts.values()), since_by_ticker=starts)
    run_edgar_fetch(ctx, ["AAPL", "MSFT", "NVDA"], 15, fetch)

    assert seen == {"AAPL": starts["AAPL"], "MSFT": floor, "NVDA": starts["NVDA"]}
    print("\n=== SANITY CHECK: per-ticker starts + lineage relist ===")
    print(
        f"  AAPL unchanged -> own start {starts['AAPL'].date()}; MSFT changed -> floor {floor.date()}; NVDA changed but own start earlier -> {starts['NVDA'].date()}. Validated."
    )


def test_an_unchanged_lineage_relists_nothing(tmp_path, sqlite_store, monkeypatch):
    """AC-032: with every `scope_changed_at` before the last run, every ticker keeps the manifest window."""
    ctx = _ctx(tmp_path, sqlite_store, ["AAPL", "MSFT"])
    last_run = pd.Timestamp.today().normalize() - pd.Timedelta(days=2)
    record_run(ctx, _T_MAIN, ticker_count=2, rows_added=0, is_full_rescan=True, run_date=last_run, coverage_complete=True, tickers=["AAPL", "MSFT"])
    seen: dict[str, pd.Timestamp] = {}

    def build(ticker, cik, *, since, done_accessions, scope):
        seen[ticker] = since
        return {}

    run_edgar_fetch(ctx, ["AAPL", "MSFT"], 15, _fetch((_T_MAIN,), build))
    assert seen == {"AAPL": last_run, "MSFT": last_run}
    print("\n=== SANITY CHECK: unchanged lineage ===")
    print(f"  both tickers listed from the manifest window {last_run.date()}; no relist")


def test_run_edgar_fetch_rejects_an_undeclared_table(tmp_path, sqlite_store, monkeypatch):
    ctx = _ctx(tmp_path, sqlite_store, ["AAPL"])

    def build(ticker, cik, *, since, done_accessions, scope):
        return {_T_MAIN: _rows(_T_MAIN, ticker, "x"), _T_CHILD: _rows(_T_CHILD, ticker, "x")}

    with pytest.raises(IncompleteEdgarRunError):
        run_edgar_fetch(ctx, ["AAPL"], 15, _fetch((_T_MAIN,), build))

    assert sqlite_store.row_count(_T_MAIN) == 1
    assert not sqlite_store.exists(_T_CHILD)  # not declared -> not written
    assert get_entry(ctx, _T_MAIN) is None  # a misdeclared fetch never advances the manifest

    print("\n=== SANITY CHECK: driver ignores an undeclared table ===")
    print("  build returned driver_child but only driver_main was declared -> child not written (it would never get a manifest entry). Validated.")


def test_the_walk_lists_lineage_ciks_counts_guard_skips_and_writes_no_identity_fingerprint(tmp_path, sqlite_store, monkeypatch):
    """The 8-K walk lists every lineage CIK of the ticker (never the register file, never a symbol), the
    summary counts guard skips for the table (AC-010), and the manifest gets no EDGAR fingerprint (P11)."""
    from src.data_extract.utils.institutionals.fetch_8k_edgar import SEC_8K_FETCH

    identity = dated_identity(
        [("AAPL", "0009999991", "cik_window", SENTINEL, "2020-01-01"), ("AAPL", "0000000001", "cik_window", "2020-01-01", None)],
        {"AAPL": "0000000001"},
    )
    monkeypatch.setattr(edgar_driver, "load_identity", lambda context: identity)
    foreign = types.SimpleNamespace(accession_number="foreign-1", form="8-K", filing_date="2024-01-02", cik=825313, items="")
    built = patch_company(monkeypatch, {1: [foreign]})
    ctx = _ctx(tmp_path, sqlite_store, ["AAPL"])

    run_edgar_fetch(ctx, ["AAPL"], 15, SEC_8K_FETCH)

    assert sorted(built) == [1, 9999991]
    entry = get_entry(ctx, Tables.sec_8k)
    assert "identity_scope_fingerprints" not in entry and entry["coverage_complete"] is True
    summary = [line for line in ctx.infos if "guard skipped" in line]
    assert summary and summary[0].endswith("guard skipped 1 filing(s) outside a filing scope for 'sec_8k'")
    assert not sqlite_store.exists(Tables.sec_8k)
    print("\n=== SANITY CHECK: lineage walk, guard count, no fingerprint ===")
    print(f"  Company() built for {sorted(built)}; {summary[0]}")


# --------------------------------------------------------------------------- #
# build_filing_rows                                                            #
# --------------------------------------------------------------------------- #
def test_build_filing_rows_stamps_once_dedups_on_pk_and_coerces_after_dedup(monkeypatch):
    listed = [
        types.SimpleNamespace(accession_number="a-1", form="8-K", filing_date="2024-01-02", cik="320193"),
        types.SimpleNamespace(accession_number="a-2", form="8-K/A", filing_date="2024-02-02", cik=None),
    ]
    seen_listing: dict = {}

    def fake_listing(scope, forms, *, since, done_accessions, stats):
        seen_listing.update(ticker=scope.ticker, ciks=scope.event_ciks, forms=list(forms), since=since, done=done_accessions)
        return listed

    stamps: list[str] = []
    real_of = FilingStamp.of.__func__

    def counting_of(cls, filing, roster_cik):
        stamps.append(filing.accession_number)
        return real_of(cls, filing, roster_cik)

    coerced_lengths: list[int] = []
    real_to_numeric = pd.to_numeric

    def spy_to_numeric(values, **kwargs):
        coerced_lengths.append(len(values))
        return real_to_numeric(values, **kwargs)

    monkeypatch.setattr(edgar_driver, "resolve_registrant_filings", fake_listing)
    monkeypatch.setattr(FilingStamp, "of", classmethod(counting_of))
    monkeypatch.setattr(frame_sanitize.pd, "to_numeric", spy_to_numeric)

    def row_fn(ticker, stamp):
        # two rows on one PK per filing: the LAST must survive the dedup
        return [
            {"ticker": ticker, "accession_number": stamp.accession_number, "cik": stamp.cik, "value": "not a number"},
            {"ticker": ticker, "accession_number": stamp.accession_number, "cik": stamp.cik, "value": "7"},
        ]

    out = build_filing_rows(
        "AAPL",
        "0000320193",
        since=pd.Timestamp("2024-01-01"),
        done_accessions=frozenset({"x"}),
        scope=EdgarScope(),
        forms=["8-K"],
        table=_T_MAIN,
        columns=["ticker", "accession_number", "cik", "value"],
        row_fn=row_fn,
        numeric=("value",),
    )[_T_MAIN]

    assert stamps == ["a-1", "a-2"]  # one stamp per filing
    assert seen_listing == {
        "ticker": "AAPL",
        "ciks": ("0000320193",),
        "forms": ["8-K"],
        "since": pd.Timestamp("2024-01-01"),
        "done": frozenset({"x"}),
    }
    assert list(out["accession_number"]) == ["a-1", "a-2"]  # 4 rows -> 2 after the PK dedup
    assert list(out["value"]) == [7.0, 7.0]  # the last row won, then was coerced
    assert coerced_lengths == [2]  # coercion saw only the de-duplicated rows
    assert list(out["cik"]) == ["0000320193", "0000320193"]  # filer CIK, roster fallback when absent
    print("\n=== SANITY CHECK: build_filing_rows ===")
    print(
        f"  2 filings -> {len(stamps)} stamps; 4 rows -> {len(out)} after PK dedup; to_numeric ran on {coerced_lengths[0]} rows (after dedup). Validated."
    )


def _guard_fetches() -> list[tuple[str, Any]]:
    from src.data_extract.utils.fundamentals.fetch_fundamentals_sec import build_ticker_fundamentals
    from src.data_extract.utils.institutionals.fetch_8k_edgar import SEC_8K_FETCH
    from src.data_extract.utils.structure.fetch_def14a_edgar import DEF14A_EDGAR_FETCH
    from src.data_extract.utils.structure.fetch_filing_text import FILING_TEXT_FETCH

    return [
        ("8-K", SEC_8K_FETCH.build),
        ("filing text", FILING_TEXT_FETCH.build),
        ("DEF 14A ECD", DEF14A_EDGAR_FETCH.build),
        ("facts", lambda *args, **kwargs: build_ticker_fundamentals(*args, catalogue=cast(Any, None), gics_by_ticker={}, **kwargs)),
    ]


@pytest.mark.parametrize("label", ["8-K", "filing text", "DEF 14A ECD", "facts"])
def test_every_listing_fetcher_skips_and_counts_a_foreign_filing(monkeypatch, label):
    """AC-008: the listing guard sits under every per-ticker EDGAR fetcher; a foreign filing produces no row and is counted."""
    build = dict(_guard_fetches())[label]
    foreign = types.SimpleNamespace(
        accession_number="foreign-1", form="10-K", filing_date="2025-02-01", cik=825313, items="", primary_document="x.htm"
    )
    patch_company(monkeypatch, {1: [foreign]})
    scope = EdgarScope(_ROSTER_IDENTITY)
    frames = build("AAPL", "0000000001", since=None, done_accessions=frozenset(), scope=scope)
    assert all(frame.empty for frame in frames.values())
    assert scope.guard.skipped == 1
    print(f"\n=== SANITY CHECK: {label} guard ===")
    print("  the CIK-1 listing returned a CIK-825313 filing -> 0 rows, 1 guard skip, nothing raised")
