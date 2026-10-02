"""Tests for the shared EDGAR fetch driver
(`src/data_extract/utils/common/edgar_driver.py`): the registrant filing listing it
resolves through and the per-ticker thread-pool driver that the 8-K / 13D / DEF 14A /
filing-text fetchers all delegate to. Offline -- a real `DataStore` on SQLite plus a stub
`Company`, no network.
"""

from __future__ import annotations

import types
from typing import Any

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
from src.data_extract.utils.common.registrant import Registrant, Segment, resolve_registrant_filings
from src.data_extract.utils.common.run_manifest import get_entry as _get_entry
from src.data_extract.utils.common.run_manifest import record_run
from src.data_extract.utils.common.sec_utils import CIK_MAPPING_COLS
from src.data_store import schema
from src.data_store.schema import Table, Tables
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
    ctx = types.SimpleNamespace(
        store=store,
        paths={"DATA_STORE": tmp_path},
        log=types.SimpleNamespace(info=lambda *a, **k: None, warning=lambda msg, *a: warnings.append(msg % a)),
        config=extract_config(data_extract={"manifest_full_rescan_days": 30}),
        ensure_edgar_identity=lambda: None,
        config_dir=tmp_path,
    )
    ctx.warnings = warnings
    return ctx


def _fetch(tables, build, desc="test", *, require_complete=False, identity_aware=False, completion_table=None) -> EdgarFetch:
    """An `EdgarFetch` with the driver's plain defaults (not completeness-sensitive, not
    identity-aware) unless the test sets them."""
    return EdgarFetch(
        desc=desc, tables=tables, build=build, require_complete=require_complete, identity_aware=identity_aware, completion_table=completion_table
    )


def _rows(table, ticker, accession):
    return pd.DataFrame([{"ticker": ticker, "accession_number": accession, "filing_date": pd.Timestamp("2024-01-02"), "src": str(table)}])


# --------------------------------------------------------------------------- #
# resolve_registrant_filings                                                   #
# --------------------------------------------------------------------------- #
def test_registrant_filings_drops_done_accessions_filters_since_and_sorts_oldest_first(monkeypatch):
    listed = [_filing("c", "2023-06-01"), _filing("a", "2020-01-01"), _filing("d", "2024-03-01"), _filing("b", "2021-05-01")]
    monkeypatch.setattr("edgar.Company", lambda t: types.SimpleNamespace(get_filings=lambda form: listed))

    out = resolve_registrant_filings("AAPL", ["8-K"], since=pd.Timestamp("2021-01-01"), done_accessions=frozenset({"c"}), registrants={})

    # "a" is pre-since, "c" is already stored -> only b, d survive, oldest first
    assert [f.accession_number for f in out] == ["b", "d"]

    print("\n=== SANITY CHECK: registrant filings dedup + since + order ===")
    print("  4 listed, 1 stored, 1 pre-since -> ['b', 'd'] oldest-first. Validated.")


def test_registrant_filings_without_since_returns_everything_sorted(monkeypatch):
    listed = [_filing("c", "2023-06-01"), _filing("a", "2020-01-01")]
    monkeypatch.setattr("edgar.Company", lambda t: types.SimpleNamespace(get_filings=lambda form: listed))

    got = resolve_registrant_filings("AAPL", ["8-K"], since=None, done_accessions=frozenset(), registrants={})

    assert [f.accession_number for f in got] == ["a", "c"]

    print("\n=== SANITY CHECK: registrant filings with since=None ===")
    print("  no cutoff -> both filings kept, still oldest-first. Validated.")


# --------------------------------------------------------------------------- #
# resolve_registrant_filings across a registrant cutover                       #
# --------------------------------------------------------------------------- #
#: XOM's real shape, measured 2026-09-09. `Company("XOM")` resolves to the PREDECESSOR (EDGAR
#: has not remapped the ticker), so the successor's filings are invisible without the register.
_CUTOVER_DATE = pd.Timestamp("2026-07-01")


def _xom_cutover():
    """XOM's real two-segment chain, as a `Registrant`."""
    return Registrant(
        ticker="XOM",
        kind="reorganisation",
        segments=(
            Segment(cik="0000034088", valid_from=None, valid_to=_CUTOVER_DATE, evidence="test fixture"),
            Segment(cik="0002115436", valid_from=_CUTOVER_DATE, valid_to=None, evidence="test fixture"),
        ),
    )


def _patch_registrants(monkeypatch, by_cik: dict, cutover=None) -> dict[str, Registrant]:
    """`Company(x)` -> that registrant's filings; returns the register holding `cutover`, or empty.

    ⚠ PATCHES `edgar.Company`: `registrant.resolve_registrant_filings` calls `edgar.Company` at
    call time, so patching a module-local name would patch nothing and reach the live SEC API.

    Keys are what the resolver passes: the TICKER string for the ticker-resolved lookup, and
    an `int` CIK for each segment.
    """
    monkeypatch.setattr("edgar.Company", lambda x: types.SimpleNamespace(get_filings=lambda form: by_cik.get(x, [])))
    return {"XOM": cutover} if cutover else {}


def test_registrant_filings_unions_the_successor_registrant(monkeypatch):
    """The defect this fixes: three real XOM 8-Ks reached no table at all, because the ticker
    resolves to the predecessor and nothing walked the successor."""
    registrants = _patch_registrants(
        monkeypatch,
        {
            "XOM": [_filing("pred-old", "2026-05-01")],
            2115436: [_filing("suc-1", "2026-07-07"), _filing("suc-2", "2026-08-28")],
        },
        cutover=_xom_cutover(),
    )

    out = resolve_registrant_filings("XOM", ["8-K"], since=None, done_accessions=frozenset(), registrants=registrants)

    assert [f.accession_number for f in out] == ["pred-old", "suc-1", "suc-2"]

    print("\n=== SANITY CHECK: cutover union recovers the successor's filings ===")
    print("  successor-only accessions ['suc-1', 'suc-2'] now reachable. Validated.")


def test_registrant_filings_keeps_a_predecessor_filing_dated_after_the_cutover(monkeypatch):
    """⚠ THE ANTI-REGRESSION TEST, and the reason this path UNIONS where `cutover_filings`
    SPLITS. XOM's SCHEDULE 13G of 2026-08-07 is filed under the PREDECESSOR, five weeks after
    the 2026-07-01 boundary. A dated split would discard it -- a filing already in the
    database -- so applying the fundamentals rule to the event pipelines loses data."""
    registrants = _patch_registrants(
        monkeypatch,
        {
            "XOM": [_filing("pred-late", "2026-08-07")],
            2115436: [_filing("suc-1", "2026-07-07")],
        },
        cutover=_xom_cutover(),
    )

    out = resolve_registrant_filings("XOM", ["SCHEDULE 13G"], since=None, done_accessions=frozenset(), registrants=registrants)
    kept = [f.accession_number for f in out]

    assert "pred-late" in kept, "a dated split would have dropped this"
    assert pd.Timestamp(out[-1].filing_date) > _CUTOVER_DATE
    assert kept == ["suc-1", "pred-late"]

    print("\n=== SANITY CHECK: predecessor filings after the cutover survive ===")
    print("  'pred-late' (2026-08-07, past a 2026-07-01 boundary) kept. Validated.")


def test_registrant_filings_takes_a_co_indexed_document_once(monkeypatch):
    """XOM's 2026-08-03 10-Q carries ONE accession indexed under BOTH CIKs -- which is why
    fundamentals stayed clean while `sec_8k` lost filings. It must not arrive twice."""
    shared = _filing("0000034088-26-000093", "2026-08-03")
    registrants = _patch_registrants(
        monkeypatch,
        {
            "XOM": [shared],
            34088: [shared],
            2115436: [shared],
        },
        cutover=_xom_cutover(),
    )

    out = resolve_registrant_filings("XOM", ["10-Q"], since=None, done_accessions=frozenset(), registrants=registrants)

    assert [f.accession_number for f in out] == ["0000034088-26-000093"]

    print("\n=== SANITY CHECK: a co-indexed accession is taken once ===")
    print("  same accession under both registrants -> 1 filing. Validated.")


def test_registrant_filings_survives_an_unresolvable_cutover_cik(monkeypatch):
    """A dead CIK in the register must cost that registrant's filings, not the whole walk."""

    def _company(x):
        if x == 2115436:
            raise ValueError("no such company")
        return types.SimpleNamespace(get_filings=lambda form: [_filing("pred", "2026-05-01")])

    monkeypatch.setattr("edgar.Company", _company)
    registrants = {"XOM": _xom_cutover()}

    out = resolve_registrant_filings("XOM", ["8-K"], since=None, done_accessions=frozenset(), registrants=registrants)

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


def test_run_edgar_fetch_isolates_a_failing_ticker(tmp_path, sqlite_store, monkeypatch):
    ctx = _ctx(tmp_path, sqlite_store, ["AAPL", "MSFT"])

    def build(ticker, cik, *, since, done_accessions, scope):
        if ticker == "AAPL":
            raise RuntimeError("edgar exploded")
        return {_T_MAIN: _rows(_T_MAIN, ticker, f"{ticker}-1")}

    run_edgar_fetch(ctx, ["AAPL", "MSFT"], 15, _fetch((_T_MAIN,), build))

    stored = sqlite_store.load(_T_MAIN)
    assert list(stored["ticker"]) == ["MSFT"]
    assert get_entry(ctx, _T_MAIN)["rows_added"] == 1

    print("\n=== SANITY CHECK: driver isolates a failing ticker ===")
    print("  AAPL raised, MSFT's row still landed and the run was recorded. Validated.")


def test_completeness_sensitive_run_does_not_record_a_partial_success(tmp_path, sqlite_store, monkeypatch):
    ctx = _ctx(tmp_path, sqlite_store, ["AAPL", "MSFT"])

    def build(ticker, cik, *, since, done_accessions, scope):
        if ticker == "AAPL":
            raise RuntimeError("discovery page failed")
        return {_T_MAIN: _rows(_T_MAIN, ticker, f"{ticker}-1")}

    with pytest.raises(IncompleteEdgarRunError, match="no run manifest was advanced"):
        run_edgar_fetch(
            ctx,
            ["AAPL", "MSFT"],
            15,
            _fetch((_T_MAIN,), build, "schedule test", require_complete=True),
        )

    assert sqlite_store.row_count(_T_MAIN) == 1
    assert get_entry(ctx, _T_MAIN) is None
    print("\n=== SANITY CHECK: incomplete schedule run ===")
    print("  one ticker failed; successful rows remain, but the manifest did not advance")
    print("  OK: partial discovery cannot masquerade as a complete empty history")


def test_completeness_sensitive_success_marks_a_trustworthy_frontier(tmp_path, sqlite_store):
    ctx = _ctx(tmp_path, sqlite_store, ["AAPL"])

    def build(ticker, cik, *, since, done_accessions, scope):
        return {_T_MAIN: _rows(_T_MAIN, ticker, f"{ticker}-1")}

    run_edgar_fetch(
        ctx,
        ["AAPL"],
        15,
        _fetch((_T_MAIN,), build, "schedule test", require_complete=True),
    )

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

    run_edgar_fetch(ctx, ["AAPL"], 15, _fetch((_T_MAIN, _T_CHILD), build))

    assert not sqlite_store.exists(_T_MAIN)  # the failing save is swallowed
    assert sqlite_store.row_count(_T_CHILD) == 1  # the sibling table still landed
    assert get_entry(ctx, _T_MAIN)["rows_added"] == 0

    print("\n=== SANITY CHECK: driver survives a save failure ===")
    print("  save to driver_main raised; driver_child still saved and the pool completed instead of aborting. Validated.")


def test_completeness_sensitive_run_rejects_a_save_failure(tmp_path, sqlite_store, monkeypatch):
    ctx = _ctx(tmp_path, sqlite_store, ["AAPL"])

    def failed_save(table, df, pk=None):
        raise RuntimeError("deadlock detected")

    monkeypatch.setattr(sqlite_store, "save", failed_save)

    def build(ticker, cik, *, since, done_accessions, scope):
        return {_T_MAIN: _rows(_T_MAIN, ticker, "x")}

    with pytest.raises(IncompleteEdgarRunError, match="no run manifest was advanced"):
        run_edgar_fetch(
            ctx,
            ["AAPL"],
            15,
            _fetch((_T_MAIN,), build, "schedule test", require_complete=True),
        )

    assert not sqlite_store.exists(_T_MAIN)
    assert get_entry(ctx, _T_MAIN) is None
    print("\n=== SANITY CHECK: schedule persistence failure ===")
    print("  discovery succeeded but persistence failed; the completeness frontier was withheld")
    print("  OK: a storage failure cannot turn an unknown Schedule history into an observed zero")


def test_completion_table_is_not_saved_after_an_earlier_save_failure(tmp_path, sqlite_store, monkeypatch):
    ctx = _ctx(tmp_path, sqlite_store, ["AAPL"])
    real_save = sqlite_store.save

    def flaky_save(table, df, pk=None):
        if table is _T_MAIN:
            raise RuntimeError("deadlock detected")
        return real_save(table, df, pk)

    monkeypatch.setattr(sqlite_store, "save", flaky_save)

    def build(ticker, cik, *, since, done_accessions, scope):
        return {
            _T_MAIN: _rows(_T_MAIN, ticker, "x"),
            _T_EMPTY: _rows(_T_EMPTY, ticker, "coverage"),
        }

    run_edgar_fetch(
        ctx,
        ["AAPL"],
        15,
        _fetch((_T_MAIN, _T_EMPTY), build, completion_table=_T_EMPTY),
    )

    assert not sqlite_store.exists(_T_MAIN)
    assert not sqlite_store.exists(_T_EMPTY)
    assert any("coverage not advanced" in warning for warning in ctx.warnings)
    print("\n=== SANITY CHECK: explicit coverage commits last ===")
    print("  the transaction save failed, so the completion row was withheld and the ticker remains visibly stale. Validated.")


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


def test_identity_scope_change_rewinds_only_the_changed_ticker(tmp_path, sqlite_store, monkeypatch):
    ctx = _ctx(tmp_path, sqlite_store, ["AAPL", "MSFT"])
    ctx.config_dir = tmp_path
    prior_date = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
    record_run(
        ctx,
        _T_MAIN,
        ticker_count=2,
        rows_added=0,
        is_full_rescan=True,
        run_date=prior_date,
        identity_scope_fingerprints={"AAPL": "same", "MSFT": "old"},
        tickers=["AAPL", "MSFT"],
    )
    identity = types.SimpleNamespace(filing_scope=lambda ticker: types.SimpleNamespace(ticker=ticker))
    monkeypatch.setattr("src.data_extract.utils.common.edgar_driver.load_identity", lambda context: identity)
    monkeypatch.setattr("src.data_extract.utils.common.edgar_driver.load_registrants", lambda config_dir: {})
    monkeypatch.setattr(
        "src.data_extract.utils.common.edgar_driver.identity_scope_fingerprint",
        lambda scope, entry: {"AAPL": "same", "MSFT": "new"}[scope.ticker],
    )
    seen: dict[str, pd.Timestamp] = {}

    def build(ticker, cik, *, since, done_accessions, scope):
        seen[ticker] = since
        return {}

    run_edgar_fetch(
        ctx,
        ["AAPL", "MSFT"],
        15,
        _fetch((_T_MAIN,), build, "identity test", identity_aware=True),
    )

    assert seen["AAPL"] == prior_date
    assert seen["MSFT"].year == (pd.Timestamp.today() - pd.DateOffset(years=15)).year
    assert get_entry(ctx, _T_MAIN)["identity_scope_fingerprints"] == {"AAPL": "same", "MSFT": "new"}
    assert get_entry(ctx, _T_MAIN)["tickers"] == ["AAPL", "MSFT"]
    print("\nSANITY: unchanged AAPL stayed incremental while only changed-scope MSFT relisted the full window.")


def test_run_edgar_fetch_rejects_an_undeclared_table(tmp_path, sqlite_store, monkeypatch):
    ctx = _ctx(tmp_path, sqlite_store, ["AAPL"])

    def build(ticker, cik, *, since, done_accessions, scope):
        return {_T_MAIN: _rows(_T_MAIN, ticker, "x"), _T_CHILD: _rows(_T_CHILD, ticker, "x")}

    run_edgar_fetch(ctx, ["AAPL"], 15, _fetch((_T_MAIN,), build))

    assert sqlite_store.row_count(_T_MAIN) == 1
    assert not sqlite_store.exists(_T_CHILD)  # not declared -> not written

    print("\n=== SANITY CHECK: driver ignores an undeclared table ===")
    print("  build returned driver_child but only driver_main was declared -> child not written (it would never get a manifest entry). Validated.")


def _identity_for_one_ticker(ticker: str, cik: str):
    """A real `Identity` for one roster ticker that owns its own symbol."""
    from src.data_extract.utils.common.identity import build_identity

    return build_identity(
        lineage=pd.DataFrame([{"cik": cik, "entity_id": f"E{cik}", "source": "roster"}]),
        tenure=pd.DataFrame(
            [{"symbol": ticker, "issuer_cik": cik, "valid_from": pd.Timestamp("2000-01-01"), "valid_to": None, "n_filings": 10, "source": "form345"}]
        ),
        roster=pd.DataFrame([{"ticker": ticker, "cik": cik}]),
    )


def test_run_edgar_fetch_honours_the_cli_config_dir_in_fingerprint_and_walk(tmp_path, sqlite_store, monkeypatch):
    """`-c` must reach BOTH the identity-scope fingerprint and the filing walk: a register entry
    that exists only in the run's config dir adds its predecessor CIK to the 8-K walk."""
    import json

    from src.data_extract.utils.institutionals.fetch_8k_edgar import SEC_8K_FETCH

    roster_cik, predecessor = "0000000001", "0009999991"
    chain_dir, empty_dir = tmp_path / "cfg_chain", tmp_path / "cfg_empty"
    (chain_dir / "sec").mkdir(parents=True)
    empty_dir.mkdir()
    blob = {
        "AAPL": {
            "kind": "reorganisation",
            "segments": [
                {"cik": predecessor, "valid_to": "2020-01-01", "evidence": "test-only predecessor"},
                {"cik": roster_cik, "valid_from": "2020-01-01", "evidence": "test-only successor"},
            ],
        }
    }
    (chain_dir / "sec" / "registrant_cutover.json").write_text(json.dumps(blob), encoding="utf-8")
    identity = _identity_for_one_ticker("AAPL", roster_cik)
    monkeypatch.setattr("src.data_extract.utils.common.edgar_driver.load_identity", lambda context: identity)
    constructed: list[object] = []

    def _company(key):
        constructed.append(key)
        return types.SimpleNamespace(get_filings=lambda form: [])

    monkeypatch.setattr("edgar.Company", _company)
    fingerprints: dict[str, str] = {}
    for label, config_dir in (("empty", empty_dir), ("chain", chain_dir)):
        ctx = _ctx(tmp_path, sqlite_store, ["AAPL"])
        ctx.config_dir = config_dir
        constructed.clear()
        run_edgar_fetch(ctx, ["AAPL"], 15, SEC_8K_FETCH)
        fingerprints[label] = get_entry(ctx, Tables.sec_8k)["identity_scope_fingerprints"]["AAPL"]
        walked = list(constructed)

    assert fingerprints["chain"] != fingerprints["empty"], "the fingerprint ignored the run's config dir"
    assert int(predecessor) in walked, f"the walk ignored the run's config dir: Company() built for {walked}"
    print("\n=== SANITY CHECK: -c reaches fingerprint and walk ===")
    print(f"  fingerprint changed with the temp register ({fingerprints['empty'][:8]} -> {fingerprints['chain'][:8]});")
    print(f"  8-K walk built Company() for {walked}: the predecessor {predecessor} from the temp register was walked.")


# --------------------------------------------------------------------------- #
# build_filing_rows                                                            #
# --------------------------------------------------------------------------- #
def test_build_filing_rows_stamps_once_dedups_on_pk_and_coerces_after_dedup(monkeypatch):
    listed = [
        types.SimpleNamespace(accession_number="a-1", form="8-K", filing_date="2024-01-02", cik="320193"),
        types.SimpleNamespace(accession_number="a-2", form="8-K/A", filing_date="2024-02-02", cik=None),
    ]
    seen_listing: dict = {}

    def fake_listing(ticker, forms, *, since, done_accessions, registrants, identity):
        seen_listing.update(ticker=ticker, forms=list(forms), since=since, done=done_accessions, identity=identity)
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
        scope=EdgarScope(None, {}),
        forms=["8-K"],
        table=_T_MAIN,
        columns=["ticker", "accession_number", "cik", "value"],
        row_fn=row_fn,
        numeric=("value",),
    )[_T_MAIN]

    assert stamps == ["a-1", "a-2"]  # one stamp per filing
    assert seen_listing == {"ticker": "AAPL", "forms": ["8-K"], "since": pd.Timestamp("2024-01-01"), "done": frozenset({"x"}), "identity": None}
    assert list(out["accession_number"]) == ["a-1", "a-2"]  # 4 rows -> 2 after the PK dedup
    assert list(out["value"]) == [7.0, 7.0]  # the last row won, then was coerced
    assert coerced_lengths == [2]  # coercion saw only the de-duplicated rows
    assert list(out["cik"]) == ["0000320193", "0000320193"]  # filer CIK, roster fallback when absent
    print("\n=== SANITY CHECK: build_filing_rows ===")
    print(
        f"  2 filings -> {len(stamps)} stamps; 4 rows -> {len(out)} after PK dedup; to_numeric ran on {coerced_lengths[0]} rows (after dedup). Validated."
    )
