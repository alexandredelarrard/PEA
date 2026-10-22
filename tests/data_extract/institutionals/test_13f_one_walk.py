"""
test_13f_one_walk.py (tests/data_extract/institutionals/test_13f_one_walk.py)
------------------------------------------------------------------------------
`fetch_13f` feeds both 13F tables from one parse, and `fetch_13f_managers` is a per-CIK catch-up
from the stored `filing_date` frontier. Fake filings (no network) on a real SQLite `DataStore`;
known-truth infotables, plus the pre-P7 `_holdings_frame` + `_resolve_tickers` logic as the
oracle for `sec13f_hr`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from xml.etree import ElementTree

import edgar.exceptions as edgar_exceptions
import httpx
import pandas as pd
import pyarrow as pa
import pytest

from src.data_aggregate.utils.common.availability import public_books
from src.data_extract.utils.common import parallel_fetch
from src.data_extract.utils.common.edgar_driver import FilingStamp
from src.data_extract.utils.institutionals import fetch_13f as f13
from src.data_extract.utils.institutionals import fetch_13f_managers as f13m
from src.data_store.schema import Tables
from src.utils.string import pad_cik

ROSTER = "0001067983"
OTHER = "0000999999"
UNIVERSE = ["AAPL", "MSFT"]
CMAP = pd.DataFrame({"cusip": ["037833100", "594918104", "G0450A105"], "ticker": ["AAPL", "MSFT", "XNOTSP"]})
_LOG = logging.getLogger("test_13f_one_walk")


def _line(cusip: str, name: str, value: float, shares: int, putcall: str = "") -> dict[str, Any]:
    return {
        "CUSIP": cusip,
        "NAMEOFISSUER": name,
        "TITLEOFCLASS": "COM",
        "VALUE": value,
        "SSHPRNAMT": shares,
        "SSHPRNAMTTYPE": "SH",
        "PUTCALL": putcall,
    }


@dataclass
class _FakeFiling:
    """The attributes `FilingStamp.of` and `_read_filing` touch; `reads` records every `obj()`."""

    cik: str
    filing_date: str
    period_of_report: str | None
    lines: list[dict[str, Any]]
    form: str = "13F-HR"
    primary_document: str | None = None
    reads: list[str] = field(default_factory=list)
    fail_reads: int = 0  # the first N `obj()` calls raise, like a transient SEC 429
    accession: str | None = None  # a real accession's prefix is the filer agent's CIK
    fail_error: type[Exception] = ConnectionError  # what a transient failure raises
    unparseable: bool = False  # every `obj()` raises a deterministic parse error

    @property
    def accession_number(self) -> str:
        return self.accession or f"{self.cik}-{self.filing_date}-{self.form}"

    def obj(self) -> Any:
        self.reads.append(self.accession_number)
        if self.unparseable:
            raise ValueError("unparseable information table")
        if self.fail_reads > 0:
            self.fail_reads -= 1
            raise self.fail_error("429 Too Many Requests")
        root = ElementTree.Element("informationTable")
        for line in self.lines:
            entry = ElementTree.SubElement(root, "infoTable")
            for source, tag in (
                ("CUSIP", "cusip"),
                ("NAMEOFISSUER", "nameOfIssuer"),
                ("TITLEOFCLASS", "titleOfClass"),
                ("VALUE", "value"),
                ("PUTCALL", "putCall"),
            ):
                ElementTree.SubElement(entry, tag).text = str(line[source])
            amount = ElementTree.SubElement(entry, "shrsOrPrnAmt")
            ElementTree.SubElement(amount, "sshPrnamt").text = str(line["SSHPRNAMT"])
            ElementTree.SubElement(amount, "sshPrnamtType").text = str(line["SSHPRNAMTTYPE"])
        return SimpleNamespace(infotable=pd.DataFrame(self.lines), infotable_xml=ElementTree.tostring(root, encoding="unicode"))

    def xml(self) -> str:
        return (
            "<edgarSubmission><isAmendment>"
            + str(self.form.endswith("/A")).lower()
            + "</isAmendment><amendmentType>RESTATEMENT</amendmentType><amendmentNo>1</amendmentNo>"
            "<reportCalendarOrQuarter>" + str(self.period_of_report) + "</reportCalendarOrQuarter>"
            "<reportType>13F HOLDINGS REPORT</reportType><isConfidentialOmitted>false</isConfidentialOmitted>"
            "<tableEntryTotal>"
            + str(len(self.lines))
            + "</tableEntryTotal><tableValueTotal>"
            + str(sum(row["VALUE"] for row in self.lines))
            + "</tableValueTotal></edgarSubmission>"
        )


def _ctx(store: Any) -> Any:
    return SimpleNamespace(store=store, log=_LOG, ensure_edgar_identity=lambda: None)


def _seed_roster(store: Any, ciks: list[str]) -> None:
    rows = [
        {
            "snapshot_date": date(2026, 1, 1),
            "dataroma_code": f"M{i}",
            "manager_name": f"Manager {i}",
            "cik": cik,
            "resolution": "edgar",
            "source_url": "https://example.invalid/",
        }
        for i, cik in enumerate(ciks)
    ]
    store.replace(Tables.superinvestor_roster.name, pd.DataFrame(rows))


def _patch_walk(monkeypatch: pytest.MonkeyPatch, filings: list[_FakeFiling]) -> None:
    """`_listing` returns the fakes in the given order behind the edgartools `Filings` surface
    `fetch_13f` touches: a pyarrow `data` index, `len`, iteration, construction from an index."""
    by_accession = {f.accession_number: f for f in filings}

    class _FakeFilings:
        def __init__(self, filing_index: pa.Table) -> None:
            self.data = filing_index

        def __len__(self) -> int:
            return self.data.num_rows

        def __iter__(self) -> Any:
            return (by_accession[a] for a in self.data.column("accession_number").to_pylist())

    index = pa.table(
        {
            "form": [f.form for f in filings],
            "accession_number": [f.accession_number for f in filings],
            "filing_date": [date.fromisoformat(f.filing_date) for f in filings],
        }
    )
    monkeypatch.setattr(f13, "Filings", _FakeFilings)
    monkeypatch.setattr(f13, "_listing", lambda forms, since, until: _FakeFilings(index))
    monkeypatch.setattr(f13, "build_cusip_ticker_map", lambda context, cusips, **kwargs: CMAP)
    monkeypatch.setattr(f13, "_sleep", lambda wait: None)


def _stored(store: Any, table: Any) -> pd.DataFrame:
    df = store.load(table)
    for col in ("period", "filing_date"):
        df[col] = pd.to_datetime(df[col])
    return df


def _resolved(store: Any, table: Any, as_of: str = "2026-06-15") -> pd.DataFrame:
    manifests = store.load(table, where={"cusip": "_empty", "parse_status": "complete"}, markers=True)
    book, _ = public_books(manifests, _stored(store, table), as_of, pd.bdate_range("2024-01-01", "2027-01-01"))
    return book


def _old_hr_rows(filing: _FakeFiling) -> pd.DataFrame:
    """The pre-P7 `sec13f_hr` path, verbatim: `_holdings_frame` then `_resolve_tickers`."""
    typed = f13._classify_holdings(pd.DataFrame(filing.lines)).dropna(subset=["cusip"])
    out = typed.groupby("cusip", as_index=False).sum(numeric_only=True)
    out["cik"] = pad_cik(filing.cik)
    out["period"] = pd.Timestamp(filing.period_of_report)
    out["filing_date"] = pd.Timestamp(filing.filing_date)
    out = out.merge(CMAP, on="cusip", how="inner")
    return out[out["ticker"].isin(set(UNIVERSE))][f13._HR_COLS]


def _roster_original() -> _FakeFiling:
    return _FakeFiling(
        ROSTER,
        "2026-05-10",
        "2026-03-31",
        [
            _line("037833100", "APPLE INC", 600_000.0, 3_000),
            _line("037833100", "APPLE INC", 400_000.0, 2_000),
            _line("G0450A105", "NON SP500", 123_000.0, 1_500),
        ],
    )


def _roster_amendment() -> _FakeFiling:
    return _FakeFiling(ROSTER, "2026-06-01", "2026-03-31", [_line("037833100", "APPLE INC", 1_200_000.0, 6_000)], form="13F-HR/A")


def _other_filer() -> _FakeFiling:
    return _FakeFiling(
        OTHER, "2026-05-12", "2026-03-31", [_line("594918104", "MICROSOFT", 250_000.0, 700), _line("037833100", "APPLE INC", 50_000.0, 100)]
    )


def test_fetch_13f_writes_hr_as_before_and_books_only_for_roster_ciks(sqlite_store, monkeypatch):
    filings = [_other_filer(), _roster_original()]  # edgartools order: newest filing first
    _seed_roster(sqlite_store, [ROSTER])
    _patch_walk(monkeypatch, filings)

    f13.fetch_13f(_ctx(sqlite_store), tickers=UNIVERSE, years_history=15, save_every=600)

    hr = _stored(sqlite_store, Tables.sec13f_hr)
    expected = pd.concat([_old_hr_rows(f) for f in filings], ignore_index=True)
    keys = ["cik", "period", "ticker", "cusip"]
    got = hr.loc[hr.ticker.isin(UNIVERSE), f13._HR_COLS].sort_values(keys).reset_index(drop=True)
    want = expected.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(got, want, check_dtype=False)

    book = _stored(sqlite_store, Tables.sec13f_manager_holdings)
    assert set(book["cik"]) == {ROSTER}
    assert set(book["cusip"]) == {"037833100", "G0450A105"}  # the non-S&P500 CUSIP stays in the book
    aapl = book[book["cusip"] == "037833100"].iloc[0]
    assert aapl["value_usd"] == 1_000_000.0 and aapl["shares"] == 5_000
    print("\n=== SANITY: one 13F walk, two tables ===")
    print(
        f"  sec13f_hr {len(hr)} full-source rows; mapped universe slice == pre-P7 path; manager book {len(book)} rows, roster CIK only, non-S&P500 kept. Validated."
    )


@pytest.mark.parametrize(("order", "save_every"), [("newest_first", 600), ("newest_first", 1), ("oldest_first", 1)])
def test_amendment_wins_in_both_tables_whatever_the_listing_order(sqlite_store, monkeypatch, order, save_every):
    filings = [_roster_amendment(), _other_filer(), _roster_original()]  # edgartools order: newest filing first
    if order == "oldest_first":
        filings.reverse()
    _seed_roster(sqlite_store, [ROSTER])
    _patch_walk(monkeypatch, filings)

    f13.fetch_13f(_ctx(sqlite_store), tickers=UNIVERSE, years_history=15, save_every=save_every)

    for table in (Tables.sec13f_hr, Tables.sec13f_manager_holdings):
        raw = _stored(sqlite_store, table)
        roster = raw.loc[raw.cik.eq(ROSTER)]
        assert len(roster) == 3
        assert set(roster.accession_number) == {_roster_original().accession_number, _roster_amendment().accession_number}
        before = _resolved(sqlite_store, table, "2026-05-20")
        after = _resolved(sqlite_store, table)
        assert set(before.loc[before.cik.eq(ROSTER), "cusip"]) == {"037833100", "G0450A105"}
        latest = after.loc[after.cik.eq(ROSTER)]
        assert len(latest) == 1 and latest.iloc[0].value_usd == 1_200_000
    assert len(_stored(sqlite_store, Tables.sec13f_hr)) == 5
    print(
        f"SANITY: both tables retain original and restatement ({order}, save_every={save_every}); omissions become exits only after amendment publication."
    )


@pytest.mark.parametrize(("save_every", "n_lookups"), [(2, 1), (1, 2)])
def test_cusips_are_mapped_once_per_batch_of_books_saved_in_read_order(sqlite_store, monkeypatch, save_every, n_lookups):
    """The unseen CUSIPs of `save_every` books are looked up in one call, then the books are saved oldest
    first; a book whose CUSIPs are all known makes no lookup, and the tail batch is flushed."""
    filings = [_roster_amendment(), _other_filer(), _roster_original()]
    _seed_roster(sqlite_store, [ROSTER])
    _patch_walk(monkeypatch, filings)
    lookups: list[list[str]] = []
    saved: list[str] = []
    real_save = f13._Saver.save_packet

    def _lookup(context: Any, cusips: list[str], *, cached: pd.DataFrame | None = None, **kwargs: Any) -> pd.DataFrame:
        lookups.append(cusips)
        return pd.concat([cached, CMAP[CMAP["cusip"].isin(cusips)]], ignore_index=True)

    monkeypatch.setattr(f13, "build_cusip_ticker_map", _lookup)
    monkeypatch.setattr(
        f13._Saver, "save_packet", lambda self, manifest, book: saved.append(manifest.iloc[0].accession_number) or real_save(self, manifest, book)
    )

    f13.fetch_13f(_ctx(sqlite_store), tickers=UNIVERSE, years_history=15, save_every=save_every)

    assert len(lookups) == n_lookups
    assert sorted(c for batch in lookups for c in batch) == sorted({line["CUSIP"] for f in filings for line in f.lines})
    assert saved == [f.accession_number for f in sorted(filings, key=lambda f: f.filing_date)]
    assert len(_stored(sqlite_store, Tables.sec13f_hr)) == 5
    print(f"SANITY: save_every={save_every} -> {len(lookups)} CUSIP lookup(s) {lookups}, books saved oldest first. Validated.")


def test_crash_mid_walk_leaves_the_watermark_at_the_oldest_saved_batch(sqlite_store, monkeypatch):
    filings = [_roster_amendment(), _other_filer(), _roster_original()]  # newest first
    _seed_roster(sqlite_store, [ROSTER])
    _patch_walk(monkeypatch, filings)
    real_save, calls = f13._Saver.save_packet, []

    def _save_then_crash(*args: Any, **kwargs: Any) -> None:
        calls.append(1)
        if len(calls) == 2:
            raise ConnectionError("store went away")
        real_save(*args, **kwargs)

    monkeypatch.setattr(f13._Saver, "save_packet", _save_then_crash)

    with pytest.raises(ConnectionError):
        f13.fetch_13f(_ctx(sqlite_store), tickers=UNIVERSE, years_history=15, save_every=1)

    watermark = pd.Timestamp(sqlite_store.max_date(Tables.sec13f_hr, "filing_date"))
    assert watermark == pd.Timestamp("2026-05-10"), watermark
    print("\n=== SANITY: crash mid-walk ===")
    print(f"  2nd batch raised; stored max(filing_date) = {watermark:%Y-%m-%d} = the OLDEST filing, so the next run")
    print("  resumes before every unsaved filing. Validated.")


def test_empty_roster_still_writes_hr_and_warns(sqlite_store, monkeypatch, caplog):
    _patch_walk(monkeypatch, [_roster_original(), _other_filer()])

    with caplog.at_level(logging.WARNING):
        f13.fetch_13f(_ctx(sqlite_store), tickers=UNIVERSE, years_history=15)

    assert len(sqlite_store.load(Tables.sec13f_hr)) == 4
    assert not sqlite_store.exists(Tables.sec13f_manager_holdings.name)
    assert any("holds no CIK" in r.getMessage() for r in caplog.records)
    print("\n=== SANITY: empty roster ===")
    print("  sec13f_hr written (4 full-source rows), no manager book, one warning. Validated.")


def _save_book_rows(store: Any, rows: list[tuple[str, str, str]]) -> None:
    """Store one AAPL manager-book row per `(cik, period, filing_date)`."""
    store.save(
        Tables.sec13f_manager_holdings,
        pd.DataFrame(
            [
                {
                    "cik": cik,
                    "period": pd.Timestamp(period),
                    "filing_date": pd.Timestamp(filed),
                    "cusip": "037833100",
                    "issuer_name": "APPLE INC",
                    "title_of_class": "COM",
                    "position_type": "common",
                    "shares": 1.0,
                    "value_usd": 100.0,
                    "call_shares": 0.0,
                    "call_value": 0.0,
                    "put_shares": 0.0,
                    "put_value": 0.0,
                    "debt_prn": 0.0,
                    "debt_value": 0.0,
                    "other_value": 0.0,
                }
                for cik, period, filed in rows
            ]
        ),
    )

    for cik, period, filed in rows:
        filing = _FakeFiling(cik, filed, period, [_line("037833100", "APPLE INC", 1000.0, 10)])
        packet = f13.read_filing_vintage(FilingStamp.of(filing, cik))
        assert not isinstance(packet, f13.ReadFailure)
        store.save_13f_filing(*packet)


def _patch_company(monkeypatch: pytest.MonkeyPatch, listings: dict[str, list[_FakeFiling]], down: set[str] | None = None) -> None:
    """`Company(cik).get_filings` returns `listings[cik]`; a CIK in `down` raises on listing."""
    down = set() if down is None else down

    class _FakeCompany:
        def __init__(self, cik: str) -> None:
            if cik in down:
                raise ConnectionError("listing throttled")
            self.cik = cik

        def get_filings(self, form: Any) -> list[_FakeFiling]:
            return listings[self.cik]

    monkeypatch.setattr("edgar.Company", _FakeCompany)  # `sec_io.company` builds it
    monkeypatch.setattr(parallel_fetch, "DEFAULT_WORKERS", 1)


def _stored_periods(store: Any, cik: str) -> list[str]:
    df = store.load(Tables.sec13f_manager_holdings, columns=["cik", "period"], where={"cik": cik}, optional=True)
    return [] if df is None else sorted(pd.to_datetime(df["period"]).dt.strftime("%Y-%m-%d").unique())


def _three_quarters(cik: str, fail_reads_middle: int = 0) -> list[_FakeFiling]:
    aapl = [_line("037833100", "APPLE INC", 1_000.0, 10)]
    return [  # edgartools order: newest first
        _FakeFiling(cik, "2026-05-15", "2026-03-31", aapl),
        _FakeFiling(cik, "2026-02-14", "2025-12-31", aapl, fail_reads=fail_reads_middle),
        _FakeFiling(cik, "2025-11-14", "2025-09-30", aapl),
    ]


def test_a_failed_read_is_retried_next_run_even_when_later_filings_saved(sqlite_store, monkeypatch, caplog):
    listing = _three_quarters(ROSTER, fail_reads_middle=1)
    _seed_roster(sqlite_store, [ROSTER])
    _patch_company(monkeypatch, {ROSTER: listing})

    with caplog.at_level(logging.WARNING):
        f13m.fetch_13f_managers(_ctx(sqlite_store), years_history=15)
    after_run1 = _stored_periods(sqlite_store, ROSTER)
    for f in listing:
        f.reads.clear()
    f13m.fetch_13f_managers(_ctx(sqlite_store), years_history=15)

    assert after_run1 == ["2025-09-30", "2026-03-31"]  # the later filing saved despite the failure
    assert _stored_periods(sqlite_store, ROSTER) == ["2025-09-30", "2025-12-31", "2026-03-31"]
    assert [f.filing_date for f in listing if f.reads] == ["2026-02-14"]  # run 2 re-reads only the gap
    assert any("read(s) failed" in r.getMessage() for r in caplog.records)  # the failure is reported, not silent
    print("\n=== SANITY: transient read failure ===")
    print(f"  run 1 stored {after_run1} and warned; run 2 read only the failed 2026-02-14 filing and filled 2025-12-31. Validated.")


def test_a_new_cik_whose_newest_filing_was_written_inline_still_backfills(sqlite_store, monkeypatch):
    new_cik = "0000000077"
    listing = _three_quarters(new_cik)
    _seed_roster(sqlite_store, [new_cik])
    _patch_company(monkeypatch, {new_cik: listing}, down={new_cik})
    f13m.fetch_13f_managers(_ctx(sqlite_store), years_history=15)  # day 1: listing throttled
    assert _stored_periods(sqlite_store, new_cik) == []
    _save_book_rows(sqlite_store, [(new_cik, "2026-03-31", "2026-05-15")])  # day 2: `fetch_13f` writes the newest inline
    _patch_company(monkeypatch, {new_cik: listing})

    f13m.fetch_13f_managers(_ctx(sqlite_store), years_history=15)

    assert _stored_periods(sqlite_store, new_cik) == ["2025-09-30", "2025-12-31", "2026-03-31"]
    assert sorted(f.filing_date for f in listing if f.reads) == ["2025-11-14", "2026-02-14"]  # the inline filing is not re-read
    print("\n=== SANITY: new roster CIK back-fill ===")
    print("  day-1 listing failed, day-2 inline write of 2026-03-31; the catch-up still back-filled 2025-09-30 and 2025-12-31. Validated.")


def test_stored_accession_and_verified_empty_book_are_never_reread(sqlite_store, monkeypatch, caplog):
    aapl = [_line("037833100", "APPLE INC", 1_000.0, 10)]
    stored = _FakeFiling(ROSTER, "2026-05-15", "2026-03-31", aapl)
    empty = _FakeFiling(ROSTER, "2026-08-14", "2026-06-30", [])
    _seed_roster(sqlite_store, [ROSTER])
    _save_book_rows(sqlite_store, [(ROSTER, "2026-03-31", "2026-05-15")])
    _patch_company(monkeypatch, {ROSTER: [empty, stored]})

    with caplog.at_level(logging.WARNING):
        saved = [f13m.fetch_13f_managers(_ctx(sqlite_store), years_history=15) for _ in range(2)]

    assert stored.reads == []  # the old frontier re-read the newest stored filing on every run
    assert len(empty.reads) == 1 and saved == [0, 0]  # read once, stored as a complete empty marker, never re-read
    assert not any("read(s) failed" in r.getMessage() for r in caplog.records)  # empty is not a failure
    print("\n=== SANITY: done-set cost ===")
    print(
        f"  stored filing reads {len(stored.reads)}; empty-info-table filing read once ({len(empty.reads)} over 2 runs), marked complete, not a failure. Validated."
    )


def test_catch_up_reads_every_unstored_filing_and_warns_only_for_bookless_ciks(sqlite_store, monkeypatch, caplog):
    with_book, no_book, failing, idle = "0000000001", "0000000002", "0000000003", "0000000004"
    _seed_roster(sqlite_store, [with_book, no_book, failing, idle])
    _save_book_rows(sqlite_store, [(with_book, "2026-03-31", "2026-05-15"), (idle, "2026-06-30", "2026-08-14")])
    sqlite_store.save(
        Tables.sec13f_hr,
        pd.DataFrame(
            [{"cik": "3", "period": pd.Timestamp("2025-12-31"), "ticker": "AAPL", "cusip": "037833100", "filing_date": pd.Timestamp("2026-02-10")}]
        ),
    )
    aapl = [_line("037833100", "APPLE INC", 1_000.0, 10)]
    listings = {
        with_book: [
            _FakeFiling(with_book, "2026-08-14", "2026-06-30", aapl),
            _FakeFiling(with_book, "2026-05-15", "2026-03-31", aapl),
            _FakeFiling(with_book, "2026-02-14", "2025-12-31", aapl),
        ],
        no_book: [
            _FakeFiling(no_book, "2026-05-15", "2026-03-31", aapl),
            _FakeFiling(no_book, "2016-02-14", "2015-12-31", aapl),
            _FakeFiling(no_book, "2009-02-14", "2008-12-31", aapl),  # period older than years_history
            _FakeFiling(no_book, "2026-01-10", None, aapl),  # null period
        ],
        idle: [_FakeFiling(idle, "2026-08-14", "2026-06-30", [])],
    }
    _patch_company(monkeypatch, listings, down={failing})

    with caplog.at_level(logging.WARNING):
        saved = f13m.fetch_13f_managers(_ctx(sqlite_store), years_history=15)

    read = {f.filing_date for f in listings[with_book] if f.reads}
    assert read == {"2026-08-14", "2026-02-14"}  # the stored 2026-05-15 filing is skipped, the older unstored one is not
    read_nf = {f.filing_date for f in listings[no_book] if f.reads}
    assert read_nf == {"2026-05-15", "2016-02-14"}  # whole window, minus old and null periods
    assert not listings[idle][0].reads
    assert saved == 4
    empty_warnings = [r.getMessage() for r in caplog.records if "produced NO rows" in r.getMessage()]
    assert len(empty_warnings) == 1
    assert "1/4 roster CIK(s)" in empty_warnings[0] and failing in empty_warnings[0] and idle not in empty_warnings[0]
    print("\n=== SANITY: 13F manager catch-up ===")
    print(f"  CIK with a book read {sorted(read)}; book-less read {sorted(read_nf)}; listing failure -> 0 rows;")
    print(f"  NO-rows warning names only the book-less failing CIK: {empty_warnings[0][-40:]!r}. Validated.")


def test_catch_up_keeps_the_last_filed_amendment_from_a_newest_first_listing(sqlite_store, monkeypatch):
    listing = [_roster_amendment(), _roster_original()]  # edgartools order: newest filing first
    _seed_roster(sqlite_store, [ROSTER])

    class _FakeCompany:
        def __init__(self, cik: str) -> None:
            self.cik = cik

        def get_filings(self, form: Any) -> list[_FakeFiling]:
            return list(listing)

    sent: list[pd.DataFrame] = []
    real_save_book = sqlite_store.save_13f_filing

    def _spy_save_book(manifest: pd.DataFrame, book: pd.DataFrame, **kwargs: Any) -> int:
        sent.append(book.copy())
        return real_save_book(manifest, book, **kwargs)

    monkeypatch.setattr("edgar.Company", _FakeCompany)  # `sec_io.company` builds it
    monkeypatch.setattr(sqlite_store, "save_13f_filing", _spy_save_book)
    monkeypatch.setattr(parallel_fetch, "DEFAULT_WORKERS", 1)

    saved = f13m.fetch_13f_managers(_ctx(sqlite_store), years_history=15)

    df_sent = pd.concat(sent, ignore_index=True)
    assert not df_sent.duplicated(subset=["cik", "accession_number", "cusip"]).any()
    assert saved == 3
    assert len(_stored(sqlite_store, Tables.sec13f_manager_holdings)) == 3
    latest = _resolved(sqlite_store, Tables.sec13f_manager_holdings)
    assert len(latest) == 1 and latest.iloc[0].value_usd == 1_200_000
    print("SANITY: catch-up preserves all source accessions; original two-position book resolves to the one-position replacement only when public.")


def _same_day_pair() -> list[_FakeFiling]:
    """A 13F-HR and its 13F-HR/A filed the same day by two filer agents: the original's accession
    prefix sorts AFTER the amendment's, so an (filed, accession) order puts the original last."""
    original = _roster_original()
    original.accession = "0001234567-26-000001"
    amendment = _FakeFiling(ROSTER, "2026-05-10", "2026-03-31", [_line("037833100", "APPLE INC", 1_200_000.0, 6_000)], form="13F-HR/A")
    amendment.accession = "0000950123-26-000009"
    return [amendment, original]


@pytest.mark.parametrize("path", ["walk", "catch_up"])
def test_a_same_day_amendment_wins_over_an_original_with_a_higher_accession(sqlite_store, monkeypatch, path):
    filings = _same_day_pair()
    _seed_roster(sqlite_store, [ROSTER])
    if path == "walk":
        _patch_walk(monkeypatch, filings)
        f13.fetch_13f(_ctx(sqlite_store), tickers=UNIVERSE, years_history=15, save_every=600)
    else:
        _patch_company(monkeypatch, {ROSTER: filings})
        f13m.fetch_13f_managers(_ctx(sqlite_store), years_history=15)

    raw = _stored(sqlite_store, Tables.sec13f_manager_holdings)
    assert len(raw.loc[raw.cusip.eq("037833100")]) == 2
    latest = _resolved(sqlite_store, Tables.sec13f_manager_holdings)
    assert latest.loc[latest.cusip.eq("037833100"), "value_usd"].iloc[0] == 1_200_000
    print(f"SANITY: same-day amendment wins by source semantics ({path}); both accessions remain stored despite inverse accession order.")


def _book_by_period(store: Any, cik: str) -> dict[str, tuple[float, str]]:
    """`{period: (AAPL value_usd, filing_date)}` of the CIK's stored book."""
    df = store.load(Tables.sec13f_manager_holdings, columns=["period", "value_usd", "filing_date"], where={"cik": cik}, optional=True)
    if df is None:
        return {}
    return {
        f"{pd.Timestamp(p):%Y-%m-%d}": (float(v), f"{pd.Timestamp(d):%Y-%m-%d}")
        for p, v, d in zip(df["period"], df["value_usd"], df["filing_date"], strict=True)
    }


def test_a_failed_same_day_amendment_of_a_stored_period_is_retried(sqlite_store, monkeypatch):
    aapl = [_line("037833100", "APPLE INC", 1_000.0, 10)]
    original = _FakeFiling(ROSTER, "2025-11-14", "2025-09-30", aapl, accession="0000000001-25-000001")
    new_quarter = _FakeFiling(ROSTER, "2026-02-14", "2025-12-31", aapl, accession="0000000001-26-000002")
    amendment = _FakeFiling(
        ROSTER,
        "2026-02-14",
        "2025-09-30",
        [_line("037833100", "APPLE INC", 9_000.0, 90)],
        form="13F-HR/A",
        accession="0000000001-26-000003",
        fail_reads=1,
    )
    _seed_roster(sqlite_store, [ROSTER])
    _patch_company(monkeypatch, {ROSTER: [original]})
    f13m.fetch_13f_managers(_ctx(sqlite_store), years_history=15)  # run 0: the P1 original is stored
    _patch_company(monkeypatch, {ROSTER: [amendment, new_quarter, original]})

    f13m.fetch_13f_managers(_ctx(sqlite_store), years_history=15)  # run 1: the P1 amendment fails, P2 saves the same day
    after_run1 = _book_by_period(sqlite_store, ROSTER)
    for f in (original, new_quarter, amendment):
        f.reads.clear()
    f13m.fetch_13f_managers(_ctx(sqlite_store), years_history=15)  # run 2
    after_run2 = _book_by_period(sqlite_store, ROSTER)

    assert after_run1 == {"2025-09-30": (1_000.0, "2025-11-14"), "2025-12-31": (1_000.0, "2026-02-14")}
    assert amendment.reads == ["0000000001-26-000003"] and not original.reads and not new_quarter.reads  # only the failed filing is re-read
    assert after_run2["2025-09-30"] == (9_000.0, "2026-02-14")  # the amendment supersedes the original
    print("\n=== SANITY: same-day failed amendment (F-010) ===")
    print(
        f"  P1 period and the 2026-02-14 date were each stored by OTHER filings; run 2 re-read only the amendment and stored {after_run2['2025-09-30']}. Validated."
    )


@pytest.mark.parametrize("unreadable", ["original", "amendment"])
def test_a_parse_failure_skips_only_its_filing_and_the_quarter_saves(sqlite_store, monkeypatch, caplog, unreadable):
    original = _FakeFiling(ROSTER, "2026-05-15", "2026-03-31", [_line("037833100", "APPLE INC", 1_000.0, 10)], accession="0000000001-26-000001")
    amendment = _FakeFiling(
        ROSTER, "2026-06-01", "2026-03-31", [_line("037833100", "APPLE INC", 2_000.0, 20)], form="13F-HR/A", accession="0000000001-26-000002"
    )
    broken, readable = (original, amendment) if unreadable == "original" else (amendment, original)
    broken.unparseable = True
    _seed_roster(sqlite_store, [ROSTER])
    _patch_company(monkeypatch, {ROSTER: [amendment, original]})

    with caplog.at_level(logging.WARNING):
        f13m.fetch_13f_managers(_ctx(sqlite_store), years_history=15)

    book = _book_by_period(sqlite_store, ROSTER)
    readable_value = 2_000.0 if readable is amendment else 1_000.0
    assert book == {"2026-03-31": (readable_value, readable.filing_date)}  # the readable sibling saved in run 1
    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1 and all(s in errors[0] for s in (ROSTER, broken.accession_number, "2026-03-31")), errors
    print(f"\n=== SANITY: deterministic parse failure of the {unreadable} (F-011) ===")
    print(
        f"  the period saved from the readable {('amendment' if readable is amendment else 'original')} {book['2026-03-31']}; one ERROR: {errors[0][:110]!r}. Validated."
    )


# `sec_io` retries a "429" up to its 3 attempts, so that failure must last 3 reads to outlive the
# policy; a network error edgartools already retried fails at once.
@pytest.mark.parametrize(("fail_error", "fail_reads"), [(RuntimeError, 3), (ConnectionError, 1), (TimeoutError, 1)])
def test_a_transient_failure_preserves_original_and_retries_only_amendment(sqlite_store, monkeypatch, caplog, fail_error, fail_reads):
    original = _FakeFiling(ROSTER, "2026-05-15", "2026-03-31", [_line("037833100", "APPLE INC", 1_000.0, 10)], accession="0000000001-26-000001")
    amendment = _FakeFiling(
        ROSTER,
        "2026-06-01",
        "2026-03-31",
        [_line("037833100", "APPLE INC", 2_000.0, 20)],
        form="13F-HR/A",
        accession="0000000001-26-000002",
        fail_reads=fail_reads,
        fail_error=fail_error,
    )
    other_quarter = _FakeFiling(ROSTER, "2026-02-14", "2025-12-31", [_line("037833100", "APPLE INC", 500.0, 5)], accession="0000000001-26-000000")
    _seed_roster(sqlite_store, [ROSTER])
    _patch_company(monkeypatch, {ROSTER: [amendment, original, other_quarter]})

    with caplog.at_level(logging.WARNING):
        f13m.fetch_13f_managers(_ctx(sqlite_store), years_history=15)
    after_run1 = _book_by_period(sqlite_store, ROSTER)
    f13m.fetch_13f_managers(_ctx(sqlite_store), years_history=15)
    after_run2 = _book_by_period(sqlite_store, ROSTER)

    assert after_run1 == {
        "2025-12-31": (500.0, "2026-02-14"),
        "2026-03-31": (1_000.0, "2026-05-15"),
    }  # the verified original stays public; only the unread amendment is retried
    assert after_run2["2026-03-31"] == (2_000.0, "2026-06-01")  # filled next run, the amendment winning
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING and amendment.accession_number in r.getMessage()]
    assert warnings and all(s in warnings[0] for s in (ROSTER, "2026-03-31")), warnings
    assert not any(r.levelno >= logging.ERROR for r in caplog.records)
    print(f"\n=== SANITY: transient read failure ({fail_error.__name__}) preserves readable source versions (F-011) ===")
    print(
        f"  run 1 stored {sorted(after_run1)} only; run 2 filled 2026-03-31 with {after_run2['2026-03-31']}; WARNING: {warnings[0][:100]!r}. Validated."
    )


@pytest.mark.parametrize(
    ("error", "transient"),
    [
        (RuntimeError("429 Too Many Requests"), True),
        (ConnectionResetError("connection reset by peer"), True),
        (TimeoutError("timed out"), True),
        (httpx.ConnectError("boom"), True),
        (httpx.RemoteProtocolError("peer closed connection"), True),
        (edgar_exceptions.TooManyRequestsError("https://www.sec.gov/x"), True),
        (edgar_exceptions.TransportError("Could not reach https://www.sec.gov/x", url="https://www.sec.gov/x"), True),
        (edgar_exceptions.TransportError("HTTP 404 from https://www.sec.gov/x", url="https://www.sec.gov/x", status_code=404), False),
        (ValueError("unparseable information table"), False),
        (KeyError("infotable"), False),
    ],
)
def test_read_filing_reports_the_failure_kind(error, transient):
    class _Raising(_FakeFiling):
        def obj(self) -> Any:
            raise error

    stamp = FilingStamp.of(_Raising(ROSTER, "2026-05-15", "2026-03-31", [], accession="0000000001-26-000001"), ROSTER)

    failure = f13._read_filing(stamp)

    assert isinstance(failure, f13.ReadFailure) and failure.transient is transient, failure
    assert type(error).__name__ in failure.reason
    print(f"\n=== SANITY: {type(error).__name__} -> {'transient' if transient else 'deterministic'} ({failure.reason[:60]!r}). Validated.")


def test_ordinary_walk_preserves_original_and_amendment_accessions(sqlite_store, monkeypatch):
    filings = [_roster_amendment(), _roster_original(), _other_filer()]
    _seed_roster(sqlite_store, [ROSTER])
    _patch_walk(monkeypatch, filings)
    f13.fetch_13f(_ctx(sqlite_store), tickers=UNIVERSE, years_history=15)
    manifests = sqlite_store.load(Tables.sec13f_hr, where={"cusip": "_empty", "parse_status": "complete"}, markers=True)
    versions = sqlite_store.load(Tables.sec13f_hr)
    assert len(manifests) == 3
    assert set(manifests.amendment_type) == {"original", "restatement"}
    original = versions.loc[versions.accession_number.eq(_roster_original().accession_number)]
    assert original.loc[original.cusip.eq("037833100"), "shares"].iloc[0] == 5000
    before = versions.copy()
    f13.fetch_13f(_ctx(sqlite_store), tickers=UNIVERSE, years_history=15)
    pd.testing.assert_frame_equal(before, sqlite_store.load(Tables.sec13f_hr))
    assert sum(len(f.reads) for f in filings) == 3
    print(
        "SANITY: ordinary extraction retains all three complete accessions; originals survive amendments; completed-accession replay skips parsing."
    )


def _legacy_cover(entry_total: str | None, value_total: str | None) -> str:
    """A pre-XML 13F-HR text submission with a Summary Page and no information-table lines."""
    lines = ["<DOCUMENT>", "<TYPE>13F-HR", "<TEXT>", "                    FORM 13F", "[X] 13F HOLDINGS REPORT", "Report Summary:"]
    lines.append("Number of Other Included Managers: 0")
    if entry_total is not None:
        lines.append(f"Form 13F Information Table Entry Total: {entry_total}")
    if value_total is not None:
        lines.append(f"Form 13F Information Table Value Total: ${value_total}")
    return "\n".join([*lines, "</TEXT>", "</DOCUMENT>"])


@dataclass
class _TextFiling(_FakeFiling):
    """A pre-XML filing: no information-table XML, the text submission instead."""

    raw: str = ""

    def obj(self) -> Any:
        self.reads.append(self.accession_number)
        return SimpleNamespace(infotable=pd.DataFrame(), infotable_xml=None, infotable_txt=self.raw)

    def full_text_submission(self) -> str:
        return self.raw


@dataclass
class _DeclaredXmlFiling(_FakeFiling):
    """An XML filing whose cover declares `entries` rows worth `value`, whatever its lines."""

    entries: int = 0
    value: float = 0.0

    def xml(self) -> str:
        cover = super().xml()
        cover = cover.replace(f"<tableEntryTotal>{len(self.lines)}<", f"<tableEntryTotal>{self.entries}<")
        return cover.replace(f"<tableValueTotal>{sum(r['VALUE'] for r in self.lines)}<", f"<tableValueTotal>{self.value:g}<")


@dataclass
class _NoInfoTableXmlFiling(_DeclaredXmlFiling):
    """An XML-era filing with a cover XML but no information-table attachment; the full submission
    carries only the cover XML, which the legacy text regexes cannot read."""

    def obj(self) -> Any:
        self.reads.append(self.accession_number)
        return SimpleNamespace(infotable=None, infotable_xml=None, infotable_txt=None, primary_form_information=object(), total_holdings=self.entries)

    def full_text_submission(self) -> str:
        return "<SEC-DOCUMENT>\n<TYPE>13F-HR\n<XML>\n" + self.xml() + "\n</XML>\n</SEC-DOCUMENT>"


_ZERO_TOTAL_KINDS = ("synthetic_table_total_0_0", "synthetic_bare_zero_row", "synthetic_total_outside_table")


def _zero_total_raw(kind: str, body: str | None = None) -> str:
    """SYNTHETIC: the real 0001082481-05-000006 0/0 cover and table layout, its 67 holdings replaced by
    a zero total row (`body`, when given, replaces that row)."""
    raw = (Path(__file__).parent / "fixtures" / "legacy_13f" / "0001082481-05-000006.sgml").read_text(encoding="utf-8")
    head, tail = raw[: raw.index("Abbott Laboratories")], raw[raw.index("</TABLE>") :]
    zero = " " * 38 + "0       0\n"
    if body is not None:
        return head + body + tail
    if kind == "synthetic_table_total_0_0":
        return head + "TOTAL" + zero[5:] + tail
    if kind == "synthetic_bare_zero_row":
        return head + zero + tail
    return head + tail.replace("</TABLE>\n", "</TABLE>\nTotal" + zero[5:], 1)


def _empty_cover_filing(kind: str) -> _FakeFiling:
    accession = "0000000001-01-000123"
    if kind == "xml_0_0":
        return _FakeFiling(ROSTER, "2026-05-15", "2026-03-31", [], accession=accession)
    if kind == "xml_no_information_table":
        return _NoInfoTableXmlFiling(ROSTER, "2026-05-15", "2026-03-31", [], accession=accession)
    if kind in _ZERO_TOTAL_KINDS:
        return _TextFiling(ROSTER, "2005-10-31", "2005-09-30", [], accession=accession, raw=_zero_total_raw(kind))
    covers = {"text_count_0": ("0", "0"), "text_count_0_no_total": ("0", None), "text_total_0_no_count": (None, "0")}
    return _TextFiling(ROSTER, "2001-05-15", "2001-03-31", [], accession=accession, raw=_legacy_cover(*covers[kind]))


def _markers(store: Any, table: Any, accession: str) -> pd.DataFrame:
    df = store.load(table, where={"cusip": "_empty", "parse_status": "complete"}, markers=True, optional=True)
    return pd.DataFrame() if df is None else df.loc[df.accession_number.eq(accession)]


@pytest.mark.parametrize(
    "kind", ["xml_0_0", "xml_no_information_table", "text_count_0", "text_count_0_no_total", "text_total_0_no_count", *_ZERO_TOTAL_KINDS]
)
def test_empty_cover_is_silent_and_marked_once(sqlite_store, monkeypatch, caplog, kind):
    """`synthetic_*`: SYNTHETIC zero total rows (no CUSIP, every amount 0) under a real 0/0 cover."""
    filing = _empty_cover_filing(kind)
    _seed_roster(sqlite_store, [ROSTER])
    _patch_walk(monkeypatch, [filing])
    real_obj, obj_calls = f13.filing_obj, []

    def _spy_obj(f: Any, *, sgml_xml_only: bool = False) -> Any:
        obj_calls.append(f.accession_number)
        return real_obj(f, sgml_xml_only=sgml_xml_only)

    monkeypatch.setattr(f13, "filing_obj", _spy_obj)
    saver = f13._Saver(_ctx(sqlite_store), set(UNIVERSE), {ROSTER}, save_every=1)

    with caplog.at_level(logging.WARNING):
        held = f13._first_pass([filing], saver)
        first_reads = len(obj_calls)
        held += f13._first_pass([filing], saver)  # the next walk over the same listing

    loud = [f"{r.levelname}: {r.getMessage()}" for r in caplog.records if r.levelno >= logging.WARNING]
    assert not loud and not held, (loud, held)
    counts = {}
    for table in (Tables.sec13f_hr, Tables.sec13f_manager_holdings):
        marker = _markers(sqlite_store, table, filing.accession_number)
        assert len(marker) == 1 and int(marker.iloc[0].source_row_count) == 0 and int(marker.iloc[0].declared_row_count) == 0, marker
        positions = sqlite_store.load(table, optional=True)
        assert positions is None or not positions.accession_number.eq(filing.accession_number).any()
        counts[table.name] = len(marker)
    assert first_reads == 1 and len(obj_calls) == 1, obj_calls  # the second walk never reads it
    print(
        f"SANITY: {'SYNTHETIC ' if kind in _ZERO_TOTAL_KINDS else ''}empty cover ({kind}) -> markers {counts}, 0 WARNING/ERROR, filing_obj calls {first_reads} then 0; recorded once, never refetched."
    )


def _positive_total_empty_cover_filing(kind: str) -> _FakeFiling:
    """A cover declaring 0 entries worth 1,234 (an all-exits book) with no holding lines."""
    accession = "0000000001-01-000321"
    if kind == "text":
        return _TextFiling(ROSTER, "2001-05-15", "2001-03-31", [], accession=accession, raw=_legacy_cover("0", "1,234"))
    if kind == "xml":
        return _DeclaredXmlFiling(ROSTER, "2026-05-15", "2026-03-31", [], accession=accession, entries=0, value=1_234.0)
    return _NoInfoTableXmlFiling(ROSTER, "2026-05-15", "2026-03-31", [], accession=accession, entries=0, value=1_234.0)


@pytest.mark.parametrize("kind", ["text", "xml", "xml_no_information_table"])
def test_zero_count_cover_with_a_positive_total_is_marked_once_keeping_the_total(sqlite_store, monkeypatch, caplog, kind):
    filing = _positive_total_empty_cover_filing(kind)
    _seed_roster(sqlite_store, [ROSTER])
    _patch_walk(monkeypatch, [filing])
    real_obj, obj_calls = f13.filing_obj, []

    def _spy_obj(f: Any, *, sgml_xml_only: bool = False) -> Any:
        obj_calls.append(f.accession_number)
        return real_obj(f, sgml_xml_only=sgml_xml_only)

    monkeypatch.setattr(f13, "filing_obj", _spy_obj)
    saver = f13._Saver(_ctx(sqlite_store), set(UNIVERSE), {ROSTER}, save_every=1)

    with caplog.at_level(logging.WARNING):
        held = f13._first_pass([filing], saver)
        first_reads = len(obj_calls)
        held += f13._first_pass([filing], saver)

    loud = [f"{r.levelname}: {r.getMessage()}" for r in caplog.records if r.levelno >= logging.WARNING]
    assert not loud and not held, (loud, held)
    totals = {}
    for table in (Tables.sec13f_hr, Tables.sec13f_manager_holdings):
        marker = _markers(sqlite_store, table, filing.accession_number)
        assert len(marker) == 1, marker
        row = marker.iloc[0]
        assert int(row.source_row_count) == 0 and int(row.declared_row_count) == 0 and float(row.source_value_total) == 0.0, row
        assert float(row.declared_value_total) == 1234.0, row.declared_value_total
        positions = sqlite_store.load(table, optional=True)
        assert positions is None or not positions.accession_number.eq(filing.accession_number).any()
        totals[table.name] = float(row.declared_value_total)
    assert first_reads == 1 and len(obj_calls) == 1, obj_calls
    print(
        f"SANITY: 0-entry cover worth 1,234 ({kind}) -> one complete marker per table, declared_value_total {totals} as filed, "
        f"0 WARNING/ERROR, filing_obj calls {first_reads} then 0; an all-exits book, never refetched."
    )


def test_zero_count_positive_total_cover_over_a_holding_line_fails():
    """SYNTHETIC: a 0-entry cover worth 1,234 over one real-looking holding line is a contradiction."""
    row = "Abbott Laboratories   Common  002824100       946   26775      0      0None  None    26775None       0\n"
    raw = _zero_total_raw("", row).replace("Value Total:\t0", "Value Total:\t1,234")
    assert "Value Total:\t1,234" in raw
    filing = _TextFiling(ROSTER, "2005-10-31", "2005-09-30", [], accession="0000000001-05-000790", raw=raw)
    failure = f13.read_filing_vintage(FilingStamp.of(filing, ""))
    assert isinstance(failure, f13.ReadFailure) and "cover declares an empty book" in failure.reason, failure
    print(f"SANITY: SYNTHETIC 0 / 1,234 cover over one 002824100 line -> ReadFailure {failure.reason[:80]!r}; the holding still wins.")


@pytest.mark.parametrize("kind", ["text", "xml", "xml_no_information_table"])
def test_nonzero_cover_zero_lines_fails_loudly(sqlite_store, monkeypatch, caplog, kind):
    accession = "0000000001-01-000456"
    if kind == "text":
        filing: _FakeFiling = _TextFiling(ROSTER, "2001-05-15", "2001-03-31", [], accession=accession, raw=_legacy_cover("2", "1,500"))
    elif kind == "xml":
        filing = _DeclaredXmlFiling(ROSTER, "2026-05-15", "2026-03-31", [], accession=accession, entries=2, value=1_500.0)
    else:
        filing = _NoInfoTableXmlFiling(ROSTER, "2026-05-15", "2026-03-31", [], accession=accession, entries=2, value=1_500.0)
    _seed_roster(sqlite_store, [ROSTER])
    _patch_walk(monkeypatch, [filing])

    failure = f13.read_filing_vintage(FilingStamp.of(filing, ""))
    with caplog.at_level(logging.WARNING):
        f13.fetch_13f(_ctx(sqlite_store), tickers=UNIVERSE, years_history=30, as_of=pd.Timestamp("2026-06-30"))

    assert isinstance(failure, f13.ReadFailure) and not failure.transient, failure
    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR and accession in r.getMessage()]
    assert len(errors) == 1, errors
    for table in (Tables.sec13f_hr, Tables.sec13f_manager_holdings):
        assert _markers(sqlite_store, table, accession).empty
    print(f"SANITY: non-zero cover with zero lines ({kind}) -> ReadFailure {failure.reason[:70]!r}, one ERROR, nothing saved.")


def test_empty_cover_over_real_holding_rows_fails_loudly(sqlite_store, monkeypatch, caplog):
    """Real 0001082481-05-000006: its cover declares 0 entries / 0 value over 67 typed holding lines."""
    accession = "0001082481-05-000006"
    raw = (Path(__file__).parent / "fixtures" / "legacy_13f" / f"{accession}.sgml").read_text(encoding="utf-8")
    filing = _TextFiling(ROSTER, "2005-10-31", "2005-09-30", [], accession=accession, raw=raw)
    _seed_roster(sqlite_store, [ROSTER])
    _patch_walk(monkeypatch, [filing])

    failure = f13.read_filing_vintage(FilingStamp.of(filing, ""))
    with caplog.at_level(logging.WARNING):
        f13.fetch_13f(_ctx(sqlite_store), tickers=UNIVERSE, years_history=30, as_of=pd.Timestamp("2026-06-30"))

    assert isinstance(failure, f13.ReadFailure) and not failure.transient, failure
    assert "cover declares an empty book" in failure.reason and "67" in failure.reason, failure.reason
    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR and accession in r.getMessage()]
    assert len(errors) == 1, errors
    for table in (Tables.sec13f_hr, Tables.sec13f_manager_holdings):
        assert _markers(sqlite_store, table, accession).empty
        stored = sqlite_store.load(table, optional=True)
        assert stored is None or not stored.accession_number.eq(accession).any()
    print(f"SANITY: real {accession} (0/0 cover, 67 holding lines) -> ReadFailure {failure.reason[:90]!r}, one ERROR, nothing saved.")


def test_empty_cover_over_a_zero_valued_cusip_line_fails():
    """SYNTHETIC: a valid CUSIP contradicts a 0/0 cover even when its amounts are all zero."""
    row = "Abbott Laboratories   Common  002824100         0       0      0      0None  None        0None       0\n"
    filing = _TextFiling(ROSTER, "2005-10-31", "2005-09-30", [], accession="0000000001-05-000789", raw=_zero_total_raw("", row))
    failure = f13.read_filing_vintage(FilingStamp.of(filing, ""))
    assert isinstance(failure, f13.ReadFailure) and "cover declares an empty book" in failure.reason, failure
    print(f"SANITY: SYNTHETIC 0/0 cover over one zero-valued 002824100 line -> ReadFailure {failure.reason[:80]!r}; a CUSIP is a holding.")


def test_catch_up_original_gap_cannot_overwrite_a_completed_amendment(sqlite_store, monkeypatch, tmp_path):
    ctx = _ctx(sqlite_store)
    original = _FakeFiling(ROSTER, "2024-08-14", "2024-06-30", [_line("037833100", "APPLE INC", 1000, 100)])
    amendment = _FakeFiling(ROSTER, "2024-10-17", "2024-06-30", [_line("037833100", "APPLE INC", 1500, 150)], form="13F-HR/A")
    stamp = FilingStamp.of(amendment, ROSTER)
    manifest, book = f13.read_filing_vintage(stamp)
    sqlite_store.save_13f_filing(manifest, book)
    amendment.reads.clear()
    _patch_company(monkeypatch, {ROSTER: [original, amendment]})
    f13m._catch_up_cik("truth", ROSTER, context=ctx, since=pd.Timestamp("2024-01-01"))
    stored = _stored(sqlite_store, Tables.sec13f_manager_holdings)
    assert stored.set_index("accession_number").shares.to_dict() == {amendment.accession_number: 150, original.accession_number: 100}
    assert len(sqlite_store.load(Tables.sec13f_manager_holdings, where={"cusip": "_empty", "parse_status": "complete"}, markers=True)) == 2
    assert len(original.reads) == 1 and amendment.reads == []
    print("SANITY: accession catch-up restores missing original without overwriting the already-preserved amended150 version.")


def _spy_done_reads(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str, int]]:
    """Every `stored_accessions` read `fetch_13f` issues, as (table, scope, number of keys)."""
    calls: list[tuple[str, str, int]] = []
    real = f13.stored_accessions

    def _spy(context: Any, table: Any, keys: Any, scope: str, where: dict[str, object] | None = None) -> dict[str, set[str]]:
        calls.append((table.name, scope, len(keys)))
        return real(context, table, keys, scope, where=where)

    monkeypatch.setattr(f13, "stored_accessions", _spy)
    return calls


def _walk_filing(cik: str, day: int, accession: str, lines: list[dict[str, Any]] | None = None) -> _FakeFiling:
    lines = [_line("037833100", "APPLE INC", 1_000.0 * day, 10 * day)] if lines is None else lines
    return _FakeFiling(cik, f"2026-05-{day:02d}", "2026-03-31", lines, accession=accession)


def _store_in(store: Any, filing: _FakeFiling, tables: tuple[Any, ...]) -> None:
    """Preserve `filing` as complete in only the given destinations (as an earlier walk may have)."""
    manifest, book = f13.read_filing_vintage(FilingStamp.of(filing, ""))
    filing.reads.clear()
    for table in tables:
        packet = book.assign(ticker=Tables.sec13f_hr.empty_marker[1]) if table == Tables.sec13f_hr else book
        store.save_13f_filing(manifest, packet, table=table)


def test_first_pass_reads_each_done_set_once_per_destination(sqlite_store, monkeypatch):
    """N = 18 filings of M = 6 CIKs, 2 of them roster CIKs: one done read per destination, not M + 2."""
    monkeypatch.setattr(f13, "build_cusip_ticker_map", lambda context, cusips, **kwargs: CMAP)
    ciks = [f"{i:010d}" for i in range(1, 7)]
    filings = [_walk_filing(cik, 10 + q, f"{cik}-26-{q:06d}") for cik in ciks for q in range(3)]
    calls = _spy_done_reads(monkeypatch)
    saver = f13._Saver(_ctx(sqlite_store), set(UNIVERSE), set(ciks[:2]), save_every=1)

    held = f13._first_pass(filings, saver)

    assert not held and all(len(f.reads) == 1 for f in filings)
    destinations = sorted((table, scope) for table, scope, _ in calls)
    assert destinations == [(Tables.sec13f_hr.name, "table"), (Tables.sec13f_manager_holdings.name, "table")], calls
    calls.clear()
    f13._first_pass(filings, f13._Saver(_ctx(sqlite_store), set(UNIVERSE), set(), save_every=1))
    assert [(table, scope) for table, scope, _ in calls] == [(Tables.sec13f_hr.name, "table")], calls
    assert all(len(f.reads) == 1 for f in filings)  # the second walk reads nothing
    print(
        f"SANITY: {len(filings)} filings of {len(ciks)} CIKs (2 roster) -> 1 sec13f_hr + 1 manager done read, not {len(ciks)} + 2; "
        "no roster CIK -> 1 read; a re-walk reads 0 filings."
    )


def test_first_pass_done_semantics_match_the_per_cik_sets(sqlite_store, monkeypatch):
    """Done = complete in sec13f_hr and, for a roster CIK, in the manager table; markers count, failures
    do not; an accession listed under two filers is done per filer."""
    monkeypatch.setattr(f13, "build_cusip_ticker_map", lambda context, cusips, **kwargs: CMAP)
    hr, mgr = Tables.sec13f_hr, Tables.sec13f_manager_holdings
    cases = {
        "roster_hr_only": (_walk_filing(ROSTER, 10, "0000000001-26-000010"), (hr,), True),
        "roster_both": (_walk_filing(ROSTER, 11, "0000000001-26-000011"), (hr, mgr), False),
        "other_hr": (_walk_filing(OTHER, 12, "0000000001-26-000012"), (hr,), False),
        "other_manager_only": (_walk_filing(OTHER, 13, "0000000001-26-000013"), (mgr,), True),
        "other_empty_marker": (_walk_filing(OTHER, 14, "0000000001-26-000014", lines=[]), (hr,), False),
        "never_stored": (_walk_filing(OTHER, 15, "0000000001-26-000015"), (), True),
        "joint_stored_filer": (_walk_filing(OTHER, 16, "0000000001-26-000016"), (hr,), False),
        "joint_unstored_filer": (_walk_filing("0000888888", 16, "0000000001-26-000016"), (), True),
    }
    for filing, tables, _ in cases.values():
        _store_in(sqlite_store, filing, tables)
    failed = _FakeFiling(OTHER, "2026-05-17", "2026-03-31", [], accession="0000000001-26-000017", unparseable=True)
    listing = [filing for filing, _, _ in cases.values()] + [failed]
    calls = _spy_done_reads(monkeypatch)
    saver = f13._Saver(_ctx(sqlite_store), set(UNIVERSE), {ROSTER}, save_every=1)

    f13._first_pass(listing, saver)
    f13._first_pass(listing, saver)

    reread = {name: len(filing.reads) for name, (filing, _, _) in cases.items()}
    assert reread == {name: int(expected) for name, (_, _, expected) in cases.items()}, reread
    assert len(failed.reads) == 2  # a failure is never done
    hr_reads = [(scope, n) for table, scope, n in calls if table == hr.name]
    assert hr_reads == [("table", 1), ("key", 2)] * 2, calls  # the per-filer read covers only the joint accession's 2 filers
    print(
        f"SANITY: re-read once {sorted(k for k, v in reread.items() if v)}; skipped {sorted(k for k, v in reread.items() if not v)}; "
        "the failure is re-read on both walks; per walk 1 table-wide read per destination + 1 per-filer read for the joint accession."
    )


def test_filing_vintage_builds_the_canonical_checksum_only_without_a_source_checksum(monkeypatch):
    """Spies: the canonical json (`sort_keys`) and its row records (`to_dict('records')`)."""
    real_dumps, real_to_dict, canonical = f13.json.dumps, pd.DataFrame.to_dict, []

    def _spy_dumps(obj: Any, **kwargs: Any) -> str:
        if kwargs.get("sort_keys"):
            canonical.append("json")
        return real_dumps(obj, **kwargs)

    def _spy_to_dict(self: pd.DataFrame, orient: str = "dict", *args: Any, **kwargs: Any) -> Any:
        if orient == "records":
            canonical.append("records")
        return real_to_dict(self, orient, *args, **kwargs)

    monkeypatch.setattr(f13, "json", SimpleNamespace(dumps=_spy_dumps))
    monkeypatch.setattr(pd.DataFrame, "to_dict", _spy_to_dict)
    info = pd.DataFrame([_line("037833100", "APPLE INC", 600.0, 3), _line("594918104", "MICROSOFT", 400.0, 2)])
    cover = {"cik": ROSTER, "accession_number": "0000000001-26-000001", "period": "2026-03-31", "filing_date": "2026-05-15"}
    cover |= {"amendment_type": "original", "source": "test", "declared_rows": 2, "declared_value": 1000.0}

    supplied_manifest, supplied_book = f13.filing_vintage(info, source_checksum="a" * 64, **cover)
    assert canonical == [] and supplied_manifest.iloc[0].source_checksum == "a" * 64
    built_manifest, built_book = f13.filing_vintage(info, **cover)
    assert canonical == ["records", "json"] and len(built_manifest.iloc[0].source_checksum) == 64
    pd.testing.assert_frame_equal(supplied_book, built_book)
    print("SANITY: a supplied source_checksum builds no canonical json (0 calls); without one it is built once; books identical.")


@dataclass
class _CountingTextFiling(_TextFiling):
    """A text filing whose full submission is `full`; counts how often it is fetched."""

    full: str = ""
    full_calls: int = 0

    def full_text_submission(self) -> str:
        self.full_calls += 1
        return self.full


@pytest.mark.parametrize(
    ("case", "marker", "value_total", "fetches"),
    [
        ("bracket", "[X]", "0", 0),
        ("paren", "(X)", "0", 0),
        ("brace", "{x}", "0", 0),
        ("no_total", "[X]", None, 1),
        ("no_checked_report", "[ ]", "0", 1),
    ],
)
def test_text_path_fetches_the_full_submission_only_without_cover_and_checked_report(case, marker, value_total, fetches):
    raw = _legacy_cover("0", value_total).replace("[X] 13F HOLDINGS REPORT", f"{marker} 13F HOLDINGS REPORT")
    full = _legacy_cover("0", "0")
    filing = _CountingTextFiling(ROSTER, "2001-05-15", "2001-03-31", [], accession="0000000001-01-000321", raw=raw, full=full)

    result = f13.read_filing_vintage(FilingStamp.of(filing, ""))

    assert filing.full_calls == fetches, (case, filing.full_calls)
    assert not isinstance(result, f13.ReadFailure), result
    used = full if fetches else raw
    assert result[0].iloc[0].source_checksum == f13.hashlib.sha256(used.encode()).hexdigest()
    print(
        f"SANITY: infotable_txt {case} ({marker!r}, value total {value_total!r}) -> {filing.full_calls} full-submission request(s); "
        f"vintage built from the {'full submission' if fetches else 'information-table text'}."
    )
