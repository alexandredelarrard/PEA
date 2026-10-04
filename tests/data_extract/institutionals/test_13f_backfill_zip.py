"""
test_13f_backfill_zip.py (tests/data_extract/institutionals/test_13f_backfill_zip.py)
-------------------------------------------------------------------------------------
The new-ticker 13F backfill from SEC 13F data-set ZIPs (M4 case 2), on a real SQLite `DataStore` with
small fixture ZIPs written into the cache (no download, no SEC request). The oracle for values is the
nightly path itself: each filing's info table parsed by edgartools' `ThirteenF.infotable` (which
decides the filing's $-thousands unit) and built by `fetch_13f._book_frame`.
"""

from __future__ import annotations

import time
import zipfile
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from edgar.thirteenf.models import ThirteenF

from src.data_extract.utils.common import bulk_cache
from src.data_extract.utils.common.sec_utils import CIK_MAPPING_COLS
from src.data_extract.utils.institutionals import fetch_13f as f13
from src.data_extract.utils.institutionals import fetch_13f_backfill as fb
from src.data_store.schema import Tables
from tests.data_extract.edgar_fixtures import fake_context
from tests.data_extract.institutionals.test_13f_one_walk import ROSTER, _seed_roster

_AS_OF = pd.Timestamp("2026-09-30")
_NEW, _NEW_CUSIP = "NEWCO", "111111111"
_MAP = {"037833100": "AAPL", "594918104": "MSFT", _NEW_CUSIP: _NEW, "G0450A105": "XNOTSP"}

# (cik, accession, form, filed, period, [(cusip, issuer, value as filed, shares, putcall)])
_A1 = (
    "1",
    "0000000001-23-000001",
    "13F-HR",
    "2023-05-15",
    "2023-03-31",
    [(_NEW_CUSIP, "NEWCO", 1_500, 10_000, ""), ("037833100", "APPLE INC", 700, 2_000, ""), (_NEW_CUSIP, "NEWCO", 50, 100, "Put")],
)
_A2 = (
    "1067983",
    "0001067983-23-000002",
    "13F-HR",
    "2023-05-15",
    "2023-03-31",
    [(_NEW_CUSIP, "NEWCO", 3_000_000, 20_000, ""), ("594918104", "MICROSOFT", 700_000, 2_000, "")],
)
_A3 = ("3", "0000000003-23-000003", "13F-NT", "2023-05-16", "2023-03-31", [])
_B1 = ("1", "0000000001-24-000004", "13F-HR", "2024-08-14", "2024-06-30", [(_NEW_CUSIP, "NEWCO", 4_000_000, 25_000, "")])
_DATA_SETS = {"2023q2": [_A1, _A2, _A3], "01jun2024-31aug2024": [_B1]}


def _dmy(day: str) -> str:
    return pd.Timestamp(day).strftime("%d-%b-%Y").upper()


def _write_zip(path: Any, filings: list[tuple]) -> None:
    """A data-set ZIP with the SEC member names and upper-case headers."""
    sub = pd.DataFrame(
        [
            {"ACCESSION_NUMBER": a, "FILING_DATE": _dmy(f), "SUBMISSIONTYPE": form, "CIK": c, "PERIODOFREPORT": _dmy(p)}
            for c, a, form, f, p, _ in filings
        ]
    )
    cover = pd.DataFrame([{"ACCESSION_NUMBER": a, "REPORTCALENDARORQUARTER": _dmy(p), "ISAMENDMENT": "N"} for _, a, _, _, p, _ in filings])
    info = pd.DataFrame(
        [
            {
                "ACCESSION_NUMBER": a,
                "INFOTABLE_SK": i,
                "NAMEOFISSUER": name,
                "TITLEOFCLASS": "COM",
                "CUSIP": cusip,
                "VALUE": value,
                "SSHPRNAMT": shares,
                "SSHPRNAMTTYPE": "SH",
                "PUTCALL": putcall,
                "INVESTMENTDISCRETION": "SOLE",
            }
            for _, a, _, _, _, lines in filings
            for i, (cusip, name, value, shares, putcall) in enumerate(lines)
        ]
    )
    with zipfile.ZipFile(path, "w") as archive:
        for name, frame in (("SUBMISSION.tsv", sub), ("COVERPAGE.tsv", cover), ("INFOTABLE.tsv", info)):
            archive.writestr(name, frame.to_csv(sep="\t", index=False))


def _xml(lines: list[tuple]) -> str:
    rows = "".join(
        f"<infoTable><nameOfIssuer>{name}</nameOfIssuer><titleOfClass>COM</titleOfClass><cusip>{cusip}</cusip><value>{value}</value>"
        f"<shrsOrPrnAmt><sshPrnamt>{shares}</sshPrnamt><sshPrnamtType>SH</sshPrnamtType></shrsOrPrnAmt>"
        + (f"<putCall>{putcall}</putCall>" if putcall else "")
        + "<investmentDiscretion>SOLE</investmentDiscretion></infoTable>"
        for cusip, name, value, shares, putcall in lines
    )
    return f'<informationTable xmlns="http://www.sec.gov/edgar/document/thirteenf/informationtable">{rows}</informationTable>'


def _edgartools_book(filing: tuple) -> pd.DataFrame:
    """The nightly path for one filing: edgartools' `ThirteenF.infotable` (no primary document, so the
    unit falls back on the report period exactly as for a data-set row), then `_book_frame`."""
    cik, _, form, filed, period, lines = filing
    thirteen_f = ThirteenF(SimpleNamespace(form=form, period_of_report=period, xml=lambda: None))
    thirteen_f.__dict__["infotable_xml"] = _xml(lines)
    thirteen_f.__dict__["_schema_version"] = None
    infotable = thirteen_f.infotable
    assert infotable is not None
    return f13._book_frame(cik, filed, period, infotable)


def _ctx(tmp_path: Any, store: Any, added_on: dict[str, str]) -> Any:
    tickers = ["AAPL", "MSFT", _NEW]
    ctx = fake_context(tmp_path, store, tickers, redundant_ticks=[])
    store.drop(Tables.sp500_tickers)
    store.save(
        Tables.sp500_tickers,
        pd.DataFrame(
            {col: ["x"] * len(tickers) for col in CIK_MAPPING_COLS}
            | {"ticker": tickers, "cik": ["1", "2", "3"], "added_on": [pd.Timestamp(added_on.get(t, "2000-01-01")) for t in tickers]}
        ),
    )
    store.save(Tables.cusip_ticker_map, pd.DataFrame({"cusip": list(_MAP), "ticker": list(_MAP.values())}))
    _seed_roster(store, [ROSTER])
    cache = tmp_path / str(ctx.config.local.paths.sec_13f_datasets)
    cache.mkdir(parents=True, exist_ok=True)
    for name, filings in _DATA_SETS.items():
        _write_zip(cache / f"{name}_form13f.zip", filings)
    return ctx


def _offline(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The landing page lists the two fixture data sets; a download or an unexpected GET fails the test."""
    pages: list[str] = []

    def _landing(context: Any, url: str, **kwargs: Any) -> Any:
        pages.append(url)
        links = "".join(f'<a href="/files/structureddata/data/form-13f-data-sets/{n}_form13f.zip">{n}</a>' for n in _DATA_SETS)
        return SimpleNamespace(text=f"<html>{links}</html>")

    def _no_download(*args: Any, **kwargs: Any) -> int:
        raise AssertionError("the backfill must read the cached ZIPs, not download")

    monkeypatch.setattr(fb, "sec_get", _landing)
    monkeypatch.setattr(bulk_cache, "download", _no_download)
    return pages


def _stored_hr(store: Any, ticker: str) -> pd.DataFrame:
    df = store.load(Tables.sec13f_hr, where={"ticker": ticker}, optional=True)
    if df is None:
        return pd.DataFrame(columns=f13._HR_COLS)
    return df.assign(period=pd.to_datetime(df["period"]), filing_date=pd.to_datetime(df["filing_date"]))


def _save_oracle_book(store: Any, filing: tuple, scale: float = 1.0) -> None:
    book = _edgartools_book(filing)
    value_cols = ["value_usd", "call_value", "put_value", "debt_value", "other_value"]
    store.save(Tables.sec13f_manager_holdings, book.assign(**{c: book[c] * scale for c in value_cols}))


def test_values_match_the_edgartools_rows_per_filing_unit(tmp_path, sqlite_store, monkeypatch):
    ctx = _ctx(tmp_path, sqlite_store, added_on={_NEW: "2026-09-28"})
    _offline(monkeypatch)
    _save_oracle_book(sqlite_store, _A2)  # the roster manager's stored book (unit check reference)

    saved = fb.fetch_13f_backfill(ctx, tickers=None, as_of=_AS_OF)

    oracle = pd.concat([_edgartools_book(f) for f in (_A1, _A2, _B1)], ignore_index=True)
    oracle = oracle[oracle["cusip"] == _NEW_CUSIP]
    got = _stored_hr(sqlite_store, _NEW)
    keys = ["cik", "period", "cusip"]
    both = got.merge(oracle, on=keys, suffixes=("", "_oracle"))
    assert saved == len(got) == len(oracle) == len(both) == 3
    for col in ("value_usd", "shares", "put_value", "put_shares"):
        assert ((both[col] - both[f"{col}_oracle"]).abs() <= 0.005 * both[f"{col}_oracle"].abs()).all(), both[[*keys, col, f"{col}_oracle"]]
    a1 = both[both["cik"] == "0000000001"].sort_values("period").iloc[0]
    assert a1["value_usd"] == 1_500_000.0 and a1["put_value"] == 50_000.0  # the $-thousands filing scaled x1000
    a2 = both[both["cik"] == ROSTER].iloc[0]
    assert a2["value_usd"] == 3_000_000.0  # the whole-dollar filing of the same period left as filed
    assert _stored_hr(sqlite_store, "AAPL").empty and _stored_hr(sqlite_store, "MSFT").empty  # M4-2: other tickers untouched
    print("\n=== SANITY: 13F data-set backfill vs edgartools rows (D13) ===")
    print(f"  {len(both)} NEWCO rows equal the edgartools-parsed rows within 0.5%: the 2023-03-31 $-thousands filing")
    print("  scaled to 1,500,000 (put 50,000), the whole-dollar one kept at 3,000,000; AAPL/MSFT rows never written. Validated.")


def test_resume_starts_from_the_tickers_own_minimum_period_newest_first(tmp_path, sqlite_store, monkeypatch):
    ctx = _ctx(tmp_path, sqlite_store, added_on={_NEW: "2026-09-28"})
    _offline(monkeypatch)
    sqlite_store.save(
        Tables.sec13f_hr,
        pd.DataFrame(
            [
                {
                    "cik": "0000000005",
                    "period": pd.Timestamp("2024-03-31"),
                    "ticker": _NEW,
                    "cusip": _NEW_CUSIP,
                    "filing_date": pd.Timestamp("2024-05-10"),
                }
            ]
        ),
    )
    read: list[str] = []
    real = fb._read_data_set
    monkeypatch.setattr(fb, "_read_data_set", lambda path, *a: read.append(path.name) or real(path, *a))

    fb.fetch_13f_backfill(ctx, tickers=None, as_of=_AS_OF)
    points = fb.resume_points(ctx, [_NEW])

    assert read == ["2023q2_form13f.zip"]  # the 2024-06-01 window starts after NEWCO's stored 2024-03-31
    assert points == {_NEW: pd.Timestamp("2023-03-31")}  # the resume point moved back to the data set's period
    work = fb.backfill_work([fb.DataSet.parse(n) for n in _DATA_SETS], {_NEW: None}, full=False)
    assert [d.name for d, _ in work] == ["01jun2024-31aug2024", "2023q2"]  # newest first
    print("\n=== SANITY: backfill resume point ===")
    print(f"  stored min(period) 2024-03-31 -> only {read} read; the new minimum {points[_NEW]:%Y-%m-%d} is the next resume point;")
    print(f"  a rowless ticker reads {[d.name for d, _ in work]} (newest first). Validated.")


def test_no_new_ticker_reads_and_downloads_nothing(tmp_path, sqlite_store, monkeypatch):
    ctx = _ctx(tmp_path, sqlite_store, added_on={_NEW: "2026-09-01"})  # added 29 days ago: past the 7-day window
    pages = _offline(monkeypatch)

    started = time.perf_counter()
    saved = fb.fetch_13f_backfill(ctx, tickers=None, as_of=_AS_OF)
    elapsed = time.perf_counter() - started

    assert saved == 0 and pages == [] and elapsed < 1.0
    assert sqlite_store.load(Tables.sec13f_hr, optional=True) is None
    print("\n=== SANITY: backfill no-op ===")
    print(f"  no ticker added inside 7 days: no landing page, no download, nothing saved, {elapsed:.3f}s. Validated.")


def test_a_stored_row_filed_later_is_kept(tmp_path, sqlite_store, monkeypatch):
    ctx = _ctx(tmp_path, sqlite_store, added_on={_NEW: "2026-09-28"})
    _offline(monkeypatch)
    later = {
        "cik": "0000000001",
        "period": pd.Timestamp("2024-06-30"),
        "ticker": _NEW,
        "cusip": _NEW_CUSIP,
        "filing_date": pd.Timestamp("2024-09-02"),
    }
    sqlite_store.save(Tables.sec13f_hr, pd.DataFrame([later | {"value_usd": 9.0}]))

    fb.fetch_13f_backfill(ctx, tickers=None, as_of=_AS_OF, full=True)

    got = _stored_hr(sqlite_store, _NEW).set_index(["cik", "period"])
    assert got.loc[("0000000001", pd.Timestamp("2024-06-30")), "value_usd"] == 9.0  # the 2024-08-14 data-set row did not overwrite it
    assert got.loc[("0000000001", pd.Timestamp("2023-03-31")), "value_usd"] == 1_500_000.0
    print("\n=== SANITY: backfill never overwrites a later filing ===")
    print("  the stored 2024-09-02 amendment row kept value 9; the data set's 2024-08-14 original was dropped. Validated.")


def test_a_unit_flip_against_the_stored_roster_book_aborts_the_data_set(tmp_path, sqlite_store, monkeypatch, caplog):
    ctx = _ctx(tmp_path, sqlite_store, added_on={_NEW: "2026-09-28"})
    _offline(monkeypatch)
    _save_oracle_book(sqlite_store, _A2, scale=1000.0)  # stored book 1000x the data-set filing

    with caplog.at_level("ERROR"):
        fb.fetch_13f_backfill(ctx, tickers=None, as_of=_AS_OF)

    got = _stored_hr(sqlite_store, _NEW)
    assert sorted(got["period"].dt.strftime("%Y-%m-%d")) == ["2024-06-30"]  # 2023q2 not saved; the 2024 data set has no roster filing
    errors = [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]
    assert len(errors) == 1 and "2023q2" in errors[0] and ROSTER in errors[0], errors
    print("\n=== SANITY: backfill unit check ===")
    print(f"  the roster filing's data-set total is 1/1000 of its stored book -> {errors[0][:90]!r}; 2023q2 not saved. Validated.")


@pytest.mark.parametrize(("tickers", "full", "saved_tickers"), [(["AAPL"], False, []), (["AAPL"], True, ["AAPL"]), ([_NEW], False, [_NEW])])
def test_scoped_runs_follow_the_archive_rules(tmp_path, sqlite_store, monkeypatch, tickers, full, saved_tickers):
    ctx = _ctx(tmp_path, sqlite_store, added_on={_NEW: "2026-09-28"})
    _offline(monkeypatch)

    fb.fetch_13f_backfill(ctx, tickers=tickers, as_of=_AS_OF, full=full)

    df = sqlite_store.load(Tables.sec13f_hr, columns=["ticker"], optional=True)
    assert (sorted(set(df["ticker"])) if df is not None else []) == saved_tickers
    print(f"\n=== SANITY: backfill -t {tickers} {'-F' if full else ''} ===")
    print(f"  saved tickers {saved_tickers}: -t narrows the new tickers; an established ticker needs -F. Validated.")
