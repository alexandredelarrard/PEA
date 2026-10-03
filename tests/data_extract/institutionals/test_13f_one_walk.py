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

import pandas as pd
import pyarrow as pa
import pytest

from src.data_extract.utils.common import parallel_fetch
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

    @property
    def accession_number(self) -> str:
        return f"{self.cik}-{self.filing_date}-{self.form}"

    def obj(self) -> Any:
        self.reads.append(self.accession_number)
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


def test_catch_up_reads_from_each_frontier_and_warns_only_for_frontierless_ciks(sqlite_store, monkeypatch, caplog):
    with_frontier, no_frontier, failing, idle = "0000000001", "0000000002", "0000000003", "0000000004"
    _seed_roster(sqlite_store, [with_frontier, no_frontier, failing, idle])
    stored = [(with_frontier, "2026-05-15"), (idle, "2026-08-14")]
    sqlite_store.save(
        Tables.sec13f_manager_holdings,
        pd.DataFrame(
            [
                {
                    "cik": cik,
                    "period": pd.Timestamp("2026-03-31"),
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
                for cik, filed in stored
            ]
        ),
    )
    sqlite_store.save(
        Tables.sec13f_hr,
        pd.DataFrame(
            [{"cik": "3", "period": pd.Timestamp("2025-12-31"), "ticker": "AAPL", "cusip": "037833100", "filing_date": pd.Timestamp("2026-02-10")}]
        ),
    )
    aapl = [_line("037833100", "APPLE INC", 1_000.0, 10)]
    listings = {
        with_frontier: [
            _FakeFiling(with_frontier, "2026-08-14", "2026-06-30", aapl),
            _FakeFiling(with_frontier, "2026-05-15", "2026-03-31", aapl),
            _FakeFiling(with_frontier, "2026-02-14", "2025-12-31", aapl),
        ],
        no_frontier: [
            _FakeFiling(no_frontier, "2026-05-15", "2026-03-31", aapl),
            _FakeFiling(no_frontier, "2016-02-14", "2015-12-31", aapl),
            _FakeFiling(no_frontier, "2009-02-14", "2008-12-31", aapl),  # period older than years_history
            _FakeFiling(no_frontier, "2026-01-10", None, aapl),  # null period
        ],
        idle: [_FakeFiling(idle, "2026-08-14", "2026-06-30", [])],
    }

    class _FakeCompany:
        def __init__(self, cik: str) -> None:
            if cik == failing:
                raise ConnectionError("listing throttled")
            self.cik = cik

        def get_filings(self, form: Any) -> list[_FakeFiling]:
            return listings[self.cik]

    monkeypatch.setattr(f13m, "Company", _FakeCompany)
    monkeypatch.setattr(f13m, "record_run", lambda *args, **kwargs: None)
    monkeypatch.setattr(parallel_fetch, "DEFAULT_WORKERS", 1)

    with caplog.at_level(logging.WARNING):
        saved = f13m.fetch_13f_managers(_ctx(sqlite_store), years_history=15)

    read = {f.filing_date for f in listings[with_frontier] if f.reads}
    assert read == {"2026-08-14", "2026-05-15"}  # nothing filed before the 2026-05-15 frontier
    read_nf = {f.filing_date for f in listings[no_frontier] if f.reads}
    assert read_nf == {"2026-05-15", "2016-02-14"}  # whole window, minus old and null periods
    assert saved == 4
    empty_warnings = [r.getMessage() for r in caplog.records if "produced NO rows" in r.getMessage()]
    assert len(empty_warnings) == 1
    assert "1/4 roster CIK(s)" in empty_warnings[0] and failing in empty_warnings[0] and idle not in empty_warnings[0]
    print("\n=== SANITY: 13F manager catch-up ===")
    print(f"  frontier CIK read {sorted(read)}; frontier-less read {sorted(read_nf)}; listing failure -> 0 rows;")
    print(f"  NO-rows warning names only the frontier-less failing CIK: {empty_warnings[0][-40:]!r}. Validated.")


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

    monkeypatch.setattr(f13m, "Company", _FakeCompany)
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
