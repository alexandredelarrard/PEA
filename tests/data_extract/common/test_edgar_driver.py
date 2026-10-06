"""Tests for the shared EDGAR fetch driver (`src/data_extract/utils/common/edgar_driver.py`) and the
`Company`-based registrant listing (`resolve_registrant_filings`) the LLM fetchers still use.

The driver tests run offline on a real `DataStore` (SQLite), a seeded local EDGAR index and fake
filings: per-filing units, empty-filing markers, in-task retry rounds and exit 0 (AC-011 check 4,
AC-012).
"""

from __future__ import annotations

import dataclasses
import types
from typing import Any

import pandas as pd
import pytest

from src.data_extract.utils.common import edgar_driver, frame_sanitize
from src.data_extract.utils.common.edgar_driver import EdgarFetch, EdgarScope, FilingStamp, parse_filing_rows, run_edgar_fetch
from src.data_extract.utils.common.identity import FilingScope
from src.data_extract.utils.common.registrant import resolve_registrant_filings
from src.data_extract.utils.common.sec_io import ParseFailureError, TransientReadError
from src.data_store import schema
from src.data_store.schema import RESUME_DOCUMENTS, Resume, Table
from tests.data_extract.common.scope_fixtures import SENTINEL, dated_identity
from tests.data_extract.edgar_fixtures import fake_context, fake_filing, seed_index

_T_DOC = Table(
    "driver_doc",
    ("ticker", "accession_number", "item"),
    date_col="filing_date",
    resume=Resume(RESUME_DOCUMENTS, "ticker", "filing_date", 7, forms=("8-K",)),
    empty_marker=("item", "_empty"),
)
_T_CHILD = Table("driver_child", ("ticker", "accession_number"), date_col="filing_date")
_AS_OF = pd.Timestamp("2026-09-30")


@pytest.fixture(autouse=True)
def _register_test_tables(monkeypatch):
    """`store.resolve` only accepts registered tables, by design; these exercise the driver on its own contract."""
    for table in (_T_DOC, _T_CHILD):
        monkeypatch.setitem(schema.BY_NAME, table.name, table)


@pytest.fixture
def waits(monkeypatch) -> list[float]:
    """The retry-round waits, recorded instead of slept."""
    recorded: list[float] = []
    monkeypatch.setattr(edgar_driver, "_sleep", recorded.append)
    return recorded


def _filing(accession: str, filing_date: str):
    return types.SimpleNamespace(accession_number=accession, filing_date=filing_date)


def _accession(i: int) -> str:
    return f"0000000001-26-{i:06d}"


def _seed(ctx, n: int, *, cik: int = 1, start: str = "2026-01-02") -> list[str]:
    """`n` 8-K index rows under `cik`, one per day from `start`; their accessions."""
    days = pd.date_range(start, periods=n, freq="D")
    rows = [(cik, "Fixture Co", "8-K", day.date().isoformat(), _accession(i)) for i, day in enumerate(days)]
    seed_index(ctx, rows)
    return [row[4] for row in rows]


class _Source:
    """Per-accession behaviour of the fake SEC: `fail_reads` (transient reads left), `kinds` (`parse`,
    `empty`, `bug`), else one row in each table; `reads` records every read."""

    def __init__(self) -> None:
        self.fail_reads: dict[str, int] = {}
        self.kinds: dict[str, str] = {}
        self.reads: list[str] = []

    def parse(self, ticker: str, cik: str, stamp: FilingStamp, scope: EdgarScope) -> dict[Table, pd.DataFrame]:
        accession = stamp.accession_number
        self.reads.append(accession)
        if self.fail_reads.get(accession, 0) > 0:
            self.fail_reads[accession] -= 1
            raise TransientReadError(f"{accession}: HTTP 503")
        kind = self.kinds.get(accession, "rows")
        if kind == "parse":
            raise ParseFailureError(f"{accession}: unparseable")
        if kind == "empty":
            return {_T_DOC: pd.DataFrame(columns=["ticker", "accession_number", "item", "filing_date"])}
        if kind == "bug":
            raise NameError("name 'cols' is not defined")
        row = {"ticker": ticker, "accession_number": accession, "item": "8.01", "filing_date": stamp.filed.normalize()}
        child = {k: row[k] for k in ("ticker", "accession_number", "filing_date")}
        return {_T_CHILD: pd.DataFrame([child]), _T_DOC: pd.DataFrame([row])}


def _fetch(source: _Source) -> EdgarFetch:
    return EdgarFetch(desc="driver test", tables=(_T_CHILD, _T_DOC), forms=("8-K",), parse=source.parse, done_table=_T_DOC, identity_aware=False)


def _run(ctx, source: _Source, tickers: list[str], as_of: pd.Timestamp = _AS_OF, **kwargs: Any):
    return run_edgar_fetch(ctx, tickers, 15, _fetch(source), as_of=as_of, refresh_index=False, max_workers=1, **kwargs)


def _stored(store, table: Table = _T_DOC) -> set[str]:
    return set(store.distinct(table, "accession_number"))


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
# run_edgar_fetch: per-filing units, markers, retry rounds, exit 0             #
# --------------------------------------------------------------------------- #
def test_failed_documents_are_saved_partially_and_exit_zero(tmp_path, sqlite_store, waits):
    """AC-011 check 4, second half: 200 listed, 70 keep failing -> 130 saved, no raise, the 70 named."""
    ctx = fake_context(tmp_path, sqlite_store, ["AAA"])
    accessions = _seed(ctx, 200)
    source = _Source()
    failing = accessions[::2][:70]
    source.fail_reads = dict.fromkeys(failing, 10**6)

    summary = _run(ctx, source, ["AAA"])

    assert _stored(sqlite_store) == set(accessions) - set(failing)
    assert sorted(summary.missing["AAA"]) == sorted(failing)
    assert waits == [60.0, 120.0, 240.0]  # three in-task rounds, growing waits
    line = next(w for w in ctx.warnings if "AAA read 130/200" in w)
    assert all(a in line for a in failing[:20]) and "(+50 more)" in line
    print("\n=== SANITY CHECK: partial save, exit 0 ===")
    print(f"  200 listed, 70 always 503 -> {len(_stored(sqlite_store))} saved after 3 rounds (waits {waits}); no exception.")
    print(f"  coverage line: {line[:110]}...")


def test_failing_documents_recover_inside_the_retry_rounds(tmp_path, sqlite_store, waits):
    """AC-011 check 4, first half: 70 of 200 fail on the first pass, then succeed -> 200 stored after one round."""
    ctx = fake_context(tmp_path, sqlite_store, ["AAA"])
    accessions = _seed(ctx, 200)
    source = _Source()
    source.fail_reads = dict.fromkeys(accessions[:70], 1)

    summary = _run(ctx, source, ["AAA"])

    assert _stored(sqlite_store) == set(accessions)
    assert summary.missing == {} and waits == [60.0]
    assert len(source.reads) == 270  # 200 on the first pass, then only the 70 failed ones
    print("\n=== SANITY CHECK: retry rounds recover ===")
    print(f"  70/200 failed once -> round 1 re-read only those 70 -> {len(_stored(sqlite_store))} stored; {len(source.reads)} reads.")


def test_gradual_recovery_lists_only_what_is_missing_each_night(tmp_path, sqlite_store, waits):
    """AC-012 gradual: night 1 saves 130 of 200, night 2's work list is 70 and saves 60, night 3's is 10."""
    ctx = fake_context(tmp_path, sqlite_store, ["AAA"])
    accessions = _seed(ctx, 200)
    source = _Source()
    failing = accessions[:70]
    source.fail_reads = dict.fromkeys(failing, 10**6)
    _run(ctx, source, ["AAA"])

    source.fail_reads = dict.fromkeys(failing[:10], 10**6)
    night2 = _run(ctx, source, ["AAA"], as_of=_AS_OF + pd.Timedelta(days=1))
    night3 = _run(ctx, source, ["AAA"], as_of=_AS_OF + pd.Timedelta(days=2))

    assert night2.work.size == 70 and night3.work.size == 10
    assert night3.missing["AAA"] == failing[:10]
    assert len([a for a in source.reads if a not in failing]) == 130  # a stored filing is never read again
    print("\n=== SANITY CHECK: gradual recovery ===")
    print(f"  night 1: 130 stored; night 2 work list {night2.work.size}; night 3 work list {night3.work.size}; each stored filing read once.")


def test_a_long_outage_heals_on_the_first_good_night(tmp_path, sqlite_store, waits):
    """AC-012 long outage: 10 nights of 503 (past the 7-day overlap), then night 11 reads every filing."""
    ctx = fake_context(tmp_path, sqlite_store, ["AAA"])
    accessions = _seed(ctx, 40, start="2026-08-01")
    source = _Source()
    for night in range(10):
        source.fail_reads = dict.fromkeys(accessions, 10**6)
        summary = _run(ctx, source, ["AAA"], as_of=_AS_OF + pd.Timedelta(days=night))
        assert summary.work.size == 40 and not _stored(sqlite_store)
    source.fail_reads = {}

    night11 = _run(ctx, source, ["AAA"], as_of=_AS_OF + pd.Timedelta(days=10))

    assert night11.work.size == 40 and _stored(sqlite_store) == set(accessions)
    print("\n=== SANITY CHECK: long outage ===")
    print(f"  10 nights of 503 stored nothing and raised nothing; night 11 read all {len(_stored(sqlite_store))} filings from 2026-08-01 on.")


def test_an_empty_or_unparseable_filing_is_one_marker_never_listed_again(tmp_path, sqlite_store, waits):
    ctx = fake_context(tmp_path, sqlite_store, ["AAA"])
    accessions = _seed(ctx, 3)
    source = _Source()
    source.kinds = {accessions[0]: "empty", accessions[1]: "parse"}

    summary = _run(ctx, source, ["AAA"])
    again = _run(ctx, source, ["AAA"], as_of=_AS_OF + pd.Timedelta(days=1))

    shown = sqlite_store.load(_T_DOC, markers=True)
    markers = shown[shown["item"] == "_empty"]
    assert sorted(markers["accession_number"]) == sorted(accessions[:2])
    assert markers["form"].tolist() == ["8-K", "8-K"] and set(markers["ticker"]) == {"AAA"}
    assert sqlite_store.load(_T_DOC)["accession_number"].tolist() == [accessions[2]]
    assert summary.outcomes["AAA"].markers == 2 and again.work.size == 0
    print("\n=== SANITY CHECK: empty-filing markers ===")
    print(f"  empty + unparseable -> {len(markers)} marker rows (item '_empty'), hidden from load; the next run lists {again.work.size}.")


def test_done_table_is_saved_last_so_a_failed_save_leaves_the_filing_listed(tmp_path, sqlite_store, waits, monkeypatch):
    ctx = fake_context(tmp_path, sqlite_store, ["AAA"])
    accessions = _seed(ctx, 2)
    order: list[str] = []
    crash = [True]
    real_save = sqlite_store.save

    def save(table, df, pk=None):
        if str(table).startswith("driver_"):
            order.append(str(table))
        if table is _T_CHILD and crash[0]:
            raise RuntimeError("deadlock detected")
        return real_save(table, df, pk)

    monkeypatch.setattr(sqlite_store, "save", save)
    first = _run(ctx, _Source(), ["AAA"])
    assert not sqlite_store.exists(_T_DOC)  # every child save crashed (rounds included) -> the done table was withheld
    crash[0] = False
    second = _run(ctx, _Source(), ["AAA"], as_of=_AS_OF + pd.Timedelta(days=1))

    assert first.missing["AAA"] == accessions and second.work.size == 2
    assert _stored(sqlite_store) == set(accessions)
    assert set(order[:-2]) == {"driver_child"} and order[-2:] == ["driver_child", "driver_doc"]
    print("\n=== SANITY CHECK: done table last ===")
    print(f"  {len(order) - 2} crashed child saves never reached the done table; the next run saved {order[-2:]} in that order.")


def test_a_programming_error_fails_the_run(tmp_path, sqlite_store, waits):
    ctx = fake_context(tmp_path, sqlite_store, ["AAA"])
    accessions = _seed(ctx, 2)
    source = _Source()
    source.kinds = {accessions[0]: "bug"}

    with pytest.raises(NameError, match="cols"):
        _run(ctx, source, ["AAA"])
    print("\n=== SANITY CHECK: programming error ===")
    print("  a NameError inside parse propagated: the driver's only non-zero exit.")


def test_the_document_cap_reads_the_newest_first_and_logs_the_uncapped_size(tmp_path, sqlite_store, waits, caplog):
    ctx = fake_context(tmp_path, sqlite_store, ["AAA"], max_documents_per_run=5)
    accessions = _seed(ctx, 12)

    with caplog.at_level("ERROR"):
        summary = _run(ctx, _Source(), ["AAA"])
    uncapped = _run(ctx, _Source(), ["AAA"], as_of=_AS_OF + pd.Timedelta(days=1), no_cap=True)

    assert summary.work.uncapped == 12 and summary.work.size == 5
    assert _stored(sqlite_store) == set(accessions)
    assert any("12 document(s) exceeds the per-run cap of 5" in r.getMessage() for r in caplog.records)
    assert uncapped.work.size == 7 and set(summary.work.units["AAA"]["accession"]) == set(accessions[-5:])
    print("\n=== SANITY CHECK: per-run cap ===")
    print(f"  12 listed, cap 5 -> the 5 newest read and an ERROR logged; --no-cap read the other {uncapped.work.size}.")


def test_full_rereads_stored_documents(tmp_path, sqlite_store, waits):
    ctx = fake_context(tmp_path, sqlite_store, ["AAA"])
    _seed(ctx, 4)
    source = _Source()
    _run(ctx, source, ["AAA"])

    full = _run(ctx, source, ["AAA"], as_of=_AS_OF + pd.Timedelta(days=1), full=True)

    assert full.work.size == 4 and len(source.reads) == 8
    print("\n=== SANITY CHECK: -F ===")
    print(f"  4 stored; -F lists and re-reads all {full.work.size} (stored accessions ignored).")


def test_a_lineage_expansion_lists_the_new_cik_on_the_next_run_with_no_run_state(tmp_path, sqlite_store, waits, monkeypatch):
    """AC-030: once the lineage gives AAA a predecessor CIK, the next run lists that CIK's filings. The work list is
    the index over the ticker's scope minus the stored accessions, so no record of a previous run is read."""
    ctx = fake_context(tmp_path, sqlite_store, ["AAA"])
    seed_index(ctx, [(1, "AAA Inc", "8-K", "2026-03-02", _accession(1)), (9999991, "Old AAA", "8-K", "2019-03-01", "0009999991-19-000001")])
    before = dated_identity([("AAA", "0000000001", "cik_window", SENTINEL, None)], {"AAA": "0000000001"})
    after = dated_identity(
        [("AAA", "0009999991", "cik_window", SENTINEL, "2020-01-01"), ("AAA", "0000000001", "cik_window", "2020-01-01", None)],
        {"AAA": "0000000001"},
    )
    fetch = dataclasses.replace(_fetch(_Source()), identity_aware=True)
    monkeypatch.setattr(edgar_driver, "load_identity", lambda context: before)
    plain = run_edgar_fetch(ctx, ["AAA"], 15, fetch, as_of=_AS_OF, refresh_index=False, max_workers=1)
    monkeypatch.setattr(edgar_driver, "load_identity", lambda context: after)

    expanded = run_edgar_fetch(ctx, ["AAA"], 15, fetch, as_of=_AS_OF + pd.Timedelta(days=1), refresh_index=False, max_workers=1)

    assert plain.work.units["AAA"]["accession"].tolist() == [_accession(1)]
    assert expanded.work.units["AAA"]["accession"].tolist() == ["0009999991-19-000001"]
    print("\n=== SANITY CHECK: a lineage expansion relists from the DB ===")
    print("  run 1 (roster CIK only) read the 2026 8-K; after the predecessor window lands, run 2 lists only its 2019 8-K. Validated.")


def test_parse_filing_rows_dedups_on_pk_and_coerces_after_dedup(monkeypatch):
    coerced_lengths: list[int] = []
    real_to_numeric = pd.to_numeric

    def spy_to_numeric(values, **kwargs):
        coerced_lengths.append(len(values))
        return real_to_numeric(values, **kwargs)

    monkeypatch.setattr(frame_sanitize.pd, "to_numeric", spy_to_numeric)

    def row_fn(ticker, stamp):
        # two rows on one PK: the LAST must survive the dedup
        return [
            {"ticker": ticker, "accession_number": stamp.accession_number, "item": "8.01", "cik": stamp.cik, "value": "not a number"},
            {"ticker": ticker, "accession_number": stamp.accession_number, "item": "8.01", "cik": stamp.cik, "value": "7"},
        ]

    stamp = FilingStamp.of(fake_filing("a-1", 320193, "2024-01-02"), "0000320193")
    columns = ["ticker", "accession_number", "item", "cik", "value"]
    out = parse_filing_rows("AAPL", "0000320193", stamp, EdgarScope(), table=_T_DOC, columns=columns, row_fn=row_fn, numeric=("value",))[_T_DOC]

    assert list(out["value"]) == [7.0] and coerced_lengths == [1]
    assert list(out["cik"]) == ["0000320193"]
    print("\n=== SANITY CHECK: parse_filing_rows ===")
    print(f"  2 rows on one PK -> {len(out)} after dedup; to_numeric ran on {coerced_lengths[0]} row (after dedup). Validated.")
