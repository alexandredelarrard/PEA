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
from types import SimpleNamespace
from typing import Any

import edgar.exceptions as edgar_exceptions
import httpx
import pandas as pd
import pyarrow as pa
import pytest

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
        return SimpleNamespace(infotable=pd.DataFrame(self.lines))


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
    """`get_filings` returns the fakes in the given order behind the edgartools `Filings` surface
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
    monkeypatch.setattr(f13, "get_filings", lambda **kwargs: _FakeFilings(index))
    monkeypatch.setattr(f13, "build_cusip_ticker_map", lambda context, cusips: CMAP)
    monkeypatch.setattr(f13, "record_run", lambda *args, **kwargs: None)


def _stored(store: Any, table: Any) -> pd.DataFrame:
    df = store.load(table)
    for col in ("period", "filing_date"):
        df[col] = pd.to_datetime(df[col])
    return df


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

    f13.fetch_13f(_ctx(sqlite_store), tickers=UNIVERSE, save_every=600)

    hr = _stored(sqlite_store, Tables.sec13f_hr)
    expected = pd.concat([_old_hr_rows(f) for f in filings], ignore_index=True)
    keys = ["cik", "period", "ticker", "cusip"]
    got = hr[f13._HR_COLS].sort_values(keys).reset_index(drop=True)
    want = expected.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(got, want, check_dtype=False)

    book = _stored(sqlite_store, Tables.sec13f_manager_holdings)
    assert set(book["cik"]) == {ROSTER}
    assert set(book["cusip"]) == {"037833100", "G0450A105"}  # the non-S&P500 CUSIP stays in the book
    aapl = book[book["cusip"] == "037833100"].iloc[0]
    assert aapl["value_usd"] == 1_000_000.0 and aapl["shares"] == 5_000
    print("\n=== SANITY: one 13F walk, two tables ===")
    print(f"  sec13f_hr {len(hr)} rows == pre-P7 path; manager book {len(book)} rows, roster CIK only, non-S&P500 kept. Validated.")


@pytest.mark.parametrize(("order", "save_every"), [("newest_first", 600), ("newest_first", 1), ("oldest_first", 1)])
def test_amendment_wins_in_both_tables_whatever_the_listing_order(sqlite_store, monkeypatch, order, save_every):
    filings = [_roster_amendment(), _other_filer(), _roster_original()]  # edgartools order: newest filing first
    if order == "oldest_first":
        filings.reverse()
    _seed_roster(sqlite_store, [ROSTER])
    _patch_walk(monkeypatch, filings)

    f13.fetch_13f(_ctx(sqlite_store), tickers=UNIVERSE, save_every=save_every)

    book = _stored(sqlite_store, Tables.sec13f_manager_holdings).set_index("cusip")
    assert len(book) == 2
    assert book.loc["037833100", "value_usd"] == 1_200_000.0
    assert book.loc["037833100", "filing_date"] == pd.Timestamp("2026-06-01")
    assert book.loc["G0450A105", "filing_date"] == pd.Timestamp("2026-05-10")  # only the original carries it
    hr = _stored(sqlite_store, Tables.sec13f_hr)
    roster_aapl = hr[(hr["cik"] == ROSTER) & (hr["ticker"] == "AAPL")]
    assert len(roster_aapl) == 1 and len(hr) == 3
    assert roster_aapl["value_usd"].iloc[0] == 1_200_000.0 and roster_aapl["filing_date"].iloc[0] == pd.Timestamp("2026-06-01")
    print(f"\n=== SANITY: amendment wins in both tables ({order}, save_every={save_every}) ===")
    print(f"  sec13f_hr and the manager book carry the 2026-06-01 13F-HR/A for AAPL (value {roster_aapl['value_usd'].iloc[0]:,.0f});")
    print("  the CUSIP only the original reported survives in the book. Validated.")


def test_crash_mid_walk_leaves_the_watermark_at_the_oldest_saved_batch(sqlite_store, monkeypatch):
    filings = [_roster_amendment(), _other_filer(), _roster_original()]  # newest first
    _seed_roster(sqlite_store, [ROSTER])
    _patch_walk(monkeypatch, filings)
    real_save, calls = f13._save_batch, []

    def _save_then_crash(*args: Any, **kwargs: Any) -> None:
        calls.append(1)
        if len(calls) == 2:
            raise ConnectionError("store went away")
        real_save(*args, **kwargs)

    monkeypatch.setattr(f13, "_save_batch", _save_then_crash)

    with pytest.raises(ConnectionError):
        f13.fetch_13f(_ctx(sqlite_store), tickers=UNIVERSE, save_every=1)

    watermark = pd.Timestamp(sqlite_store.max_date(Tables.sec13f_hr, "filing_date"))
    assert watermark == pd.Timestamp("2026-05-10"), watermark
    print("\n=== SANITY: crash mid-walk ===")
    print(f"  2nd batch raised; stored max(filing_date) = {watermark:%Y-%m-%d} = the OLDEST filing, so the next run")
    print("  resumes before every unsaved filing. Validated.")


def test_empty_roster_still_writes_hr_and_warns(sqlite_store, monkeypatch, caplog):
    _patch_walk(monkeypatch, [_roster_original(), _other_filer()])

    with caplog.at_level(logging.WARNING):
        f13.fetch_13f(_ctx(sqlite_store), tickers=UNIVERSE)

    assert len(sqlite_store.load(Tables.sec13f_hr)) == 3
    assert not sqlite_store.exists(Tables.sec13f_manager_holdings.name)
    assert any("holds no CIK" in r.getMessage() for r in caplog.records)
    print("\n=== SANITY: empty roster ===")
    print("  sec13f_hr written (3 rows), no manager book, one warning. Validated.")


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
    monkeypatch.setattr(f13m, "record_run", lambda *args, **kwargs: None)
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


def test_a_stored_filing_is_never_reread_and_an_empty_info_table_is_rechecked_once_per_run(sqlite_store, monkeypatch, caplog):
    aapl = [_line("037833100", "APPLE INC", 1_000.0, 10)]
    stored = _FakeFiling(ROSTER, "2026-05-15", "2026-03-31", aapl)
    empty = _FakeFiling(ROSTER, "2026-08-14", "2026-06-30", [])
    _seed_roster(sqlite_store, [ROSTER])
    _save_book_rows(sqlite_store, [(ROSTER, "2026-03-31", "2026-05-15")])
    _patch_company(monkeypatch, {ROSTER: [empty, stored]})

    with caplog.at_level(logging.WARNING):
        saved = [f13m.fetch_13f_managers(_ctx(sqlite_store), years_history=15) for _ in range(2)]

    assert stored.reads == []  # the old frontier re-read the newest stored filing on every run
    assert len(empty.reads) == 2 and saved == [0, 0]  # one cheap re-check per run, nothing saved, no loop
    assert not any("read(s) failed" in r.getMessage() for r in caplog.records)  # empty is not a failure
    print("\n=== SANITY: done-set cost ===")
    print(
        f"  stored filing reads {len(stored.reads)}; empty-info-table filing read once per run ({len(empty.reads)} over 2 runs), not a failure. Validated."
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
    real_save_book = f13m._save_book

    def _spy_save_book(context: Any, book: pd.DataFrame) -> tuple[int, int]:
        sent.append(book.copy())
        return real_save_book(context, book)

    monkeypatch.setattr("edgar.Company", _FakeCompany)  # `sec_io.company` builds it
    monkeypatch.setattr(f13m, "_save_book", _spy_save_book)
    monkeypatch.setattr(f13m, "record_run", lambda *args, **kwargs: None)
    monkeypatch.setattr(parallel_fetch, "DEFAULT_WORKERS", 1)

    saved = f13m.fetch_13f_managers(_ctx(sqlite_store), years_history=15)

    df_sent = pd.concat(sent, ignore_index=True)
    assert not df_sent.duplicated(subset=f13._BOOK_KEY).any(), df_sent[f13._BOOK_KEY]  # Postgres rejects a repeated PK in one upsert
    assert saved == 2  # AAPL from the amendment + the CUSIP only the original reported
    df_book = _stored(sqlite_store, Tables.sec13f_manager_holdings).set_index("cusip")
    assert df_book.loc["037833100", "value_usd"] == 1_200_000.0
    assert df_book.loc["037833100", "filing_date"] == pd.Timestamp("2026-06-01")
    assert df_book.loc["G0450A105", "filing_date"] == pd.Timestamp("2026-05-10")
    print("\n=== SANITY: catch-up keeps the amendment ===")
    print(f"  newest-first listing, original + 13F-HR/A for one CUSIP: {len(df_sent)} rows sent, one per PK;")
    print(f"  AAPL stored {df_book.loc['037833100', 'value_usd']:,.0f} filed 2026-06-01 (the 13F-HR/A). Validated.")


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
        f13.fetch_13f(_ctx(sqlite_store), tickers=UNIVERSE, save_every=600)
    else:
        _patch_company(monkeypatch, {ROSTER: filings})
        f13m.fetch_13f_managers(_ctx(sqlite_store), years_history=15)

    book = _stored(sqlite_store, Tables.sec13f_manager_holdings).set_index("cusip")
    assert book.loc["037833100", "value_usd"] == 1_200_000.0, book
    print(f"\n=== SANITY: same-day original vs amendment ({path}) ===")
    print(f"  original accession sorts after the 13F-HR/A; stored AAPL value {book.loc['037833100', 'value_usd']:,.0f} = the amendment. Validated.")


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
def test_a_transient_failure_holds_the_quarter_back_and_fills_it_next_run(sqlite_store, monkeypatch, caplog, fail_error, fail_reads):
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

    assert after_run1 == {"2025-12-31": (500.0, "2026-02-14")}  # the quarter is held back whole; the other quarter saves
    assert after_run2["2026-03-31"] == (2_000.0, "2026-06-01")  # filled next run, the amendment winning
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING and amendment.accession_number in r.getMessage()]
    assert warnings and all(s in warnings[0] for s in (ROSTER, "2026-03-31")), warnings
    assert not any(r.levelno >= logging.ERROR for r in caplog.records)
    print(f"\n=== SANITY: transient read failure ({fail_error.__name__}) holds the quarter back (F-011) ===")
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
