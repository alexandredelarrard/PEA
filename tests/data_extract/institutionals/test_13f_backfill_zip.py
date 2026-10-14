"""
test_13f_backfill_zip.py (tests/data_extract/institutionals/test_13f_backfill_zip.py)
-------------------------------------------------------------------------------------
The new-ticker 13F backfill from SEC 13F data-set ZIPs (M4 case 2), on a real SQLite `DataStore` with
small fixture ZIPs written into the cache (no download, no SEC request). The oracle for values is the
nightly path itself: each filing's info table parsed by edgartools' `ThirteenF.infotable` (which
decides the filing's $-thousands unit) and built by `fetch_13f._book_frame`.
"""

from __future__ import annotations

import logging
import time
import zipfile
from io import BytesIO
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from edgar.thirteenf.models import ThirteenF

from src.data_aggregate.utils.institutionals.inputs import load_source
from src.data_extract.utils.common import bulk_cache
from src.data_extract.utils.common.sec_utils import CIK_MAPPING_COLS
from src.data_extract.utils.institutionals import fetch_13f as f13
from src.data_extract.utils.institutionals import fetch_13f_backfill as fb
from src.data_extract.utils.institutionals.fetch_superinvestors import activity_evidence
from src.data_store.schema import Tables
from tests.data_extract.edgar_fixtures import fake_context
from tests.data_extract.institutionals.test_13f_one_walk import ROSTER, _seed_roster

_AS_OF = pd.Timestamp("2026-09-30")
_NEW, _NEW_CUSIP = "NEWCO", "111111111"
_GAP_START = pd.Timestamp("2024-09-01")  # the day after the newest fixture data set ends
_SENTINEL = "_empty"
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


@pytest.mark.parametrize("name", ["2013q2", "01jun2026-31aug2026"])
def test_data_set_urls_keep_both_sec_locations(name: str) -> None:
    data_set = fb.DataSet.parse(name)
    assert data_set.urls == (
        f"https://www.sec.gov/files/structureddata/data/form-13f-data-sets/{name}_form13f.zip",
        f"https://www.sec.gov/files/datastandardsinnovation/data/form-13f-data-sets/{name}_form13f.zip",
    )
    assert data_set.url == data_set.urls[0]
    print(f"SANITY: {name} keeps structureddata first and datastandardsinnovation as fallback; the existing primary URL is unchanged.")


def _write_zip(path: Any, filings: list[tuple], *, confidential: str | None = "N", omit_confidential: bool = False) -> None:
    """A data-set ZIP with the SEC member names and upper-case headers."""
    sub = pd.DataFrame(
        [
            {"ACCESSION_NUMBER": a, "FILING_DATE": _dmy(f), "SUBMISSIONTYPE": form, "CIK": c, "PERIODOFREPORT": _dmy(p)}
            for c, a, form, f, p, _ in filings
        ]
    )
    cover = pd.DataFrame(
        [
            {
                "ACCESSION_NUMBER": a,
                "REPORTCALENDARORQUARTER": _dmy(p),
                "ISAMENDMENT": "Y" if form.endswith("/A") else "N",
                "AMENDMENTTYPE": "RESTATEMENT" if form.endswith("/A") else "",
                "AMENDMENTNO": 1 if form.endswith("/A") else 0,
                "REPORTTYPE": "13F HOLDINGS REPORT",
            }
            for _, a, form, _, p, _ in filings
        ]
    )
    summary = pd.DataFrame(
        [
            {
                "ACCESSION_NUMBER": a,
                "TABLEENTRYTOTAL": len(lines),
                "TABLEVALUETOTAL": sum(line[2] for line in lines),
                "ISCONFIDENTIALOMITTED": confidential,
            }
            for _, a, _, _, _, lines in filings
        ]
    )
    if omit_confidential:
        summary = summary.drop(columns="ISCONFIDENTIALOMITTED")
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
        for name, frame in (("SUBMISSION.tsv", sub), ("COVERPAGE.tsv", cover), ("SUMMARYPAGE.tsv", summary), ("INFOTABLE.tsv", info)):
            archive.writestr(name, frame.to_csv(sep="\t", index=False))


@pytest.mark.parametrize(("flag", "restricted"), [(None, False), ("N", False), ("false", False), ("Y", True), ("true", True)])
@pytest.mark.parametrize("omit_column", [False, True])
def test_optional_zip_confidentiality_requires_affirmative_evidence(tmp_path: Any, flag: str | None, restricted: bool, omit_column: bool) -> None:
    path = tmp_path / "source.zip"
    _write_zip(path, [_B1], confidential=flag, omit_confidential=omit_column)
    packets = fb.cached_filing_vintages(path, [_B1[1]])
    assert len(packets) == 1
    manifest, holdings = packets[0]
    assert bool(manifest.restricted.iloc[0]) is (restricted and not omit_column)
    assert holdings.shares.iloc[0] == 25000 and holdings.value_usd.iloc[0] == 4000000
    print(f"SANITY: ZIP flag {flag!r}, absent column={omit_column}; only affirmative metadata restricts a complete book.")


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
    _patch_gap_walk(monkeypatch)
    return pages


def _patch_gap_walk(monkeypatch: pytest.MonkeyPatch, *, save: bool = False, cut: bool = False, unread: int = 0) -> list[dict[str, Any]]:
    """`fetch_13f` as the gap walk sees it: each call is recorded; with `save` (or `cut`) it stores one
    row filed 2025-01-10 (inside the gap) for every ticker it was given, like a real walk over that
    window. `cut` then raises, as an interrupted walk does; the call returns `unread`, the filings a
    transient failure left unread."""
    walks: list[dict[str, Any]] = []

    def _walk(context: Any, **kwargs: Any) -> int:
        walks.append(kwargs)
        if save or cut:
            row = {"cik": "0000000007", "period": pd.Timestamp("2024-12-31"), "cusip": _NEW_CUSIP, "filing_date": pd.Timestamp("2025-01-10")}
            context.store.save(Tables.sec13f_hr, pd.DataFrame([row | {"ticker": ticker} for ticker in kwargs["tickers"]]))
        if cut:
            raise RuntimeError("EDGAR walk cut off")
        return unread

    monkeypatch.setattr(fb, "fetch_13f", _walk, raising=False)
    return walks


def _markers(store: Any) -> pd.DataFrame:
    """The stored gap-walk marker rows of `sec13f_hr` (empty frame when none)."""
    df = store.load(Tables.sec13f_hr, where={"cusip": _SENTINEL, "cik": _SENTINEL}, markers=True, optional=True)
    return pd.DataFrame(columns=f13._HR_COLS) if df is None else df


def _stored_hr(store: Any, ticker: str) -> pd.DataFrame:
    df = store.load(Tables.sec13f_hr, where={"ticker": ticker}, optional=True)
    if df is None:
        return pd.DataFrame(columns=f13._HR_COLS)
    return df.assign(period=pd.to_datetime(df["period"]), filing_date=pd.to_datetime(df["filing_date"]))


def _save_oracle_book(store: Any, filing: tuple, scale: float = 1.0) -> None:
    book = _edgartools_book(filing)
    value_cols = ["value_usd", "call_value", "put_value", "debt_value", "other_value"]
    store.save(Tables.sec13f_manager_holdings, book.assign(**{c: book[c] * scale for c in value_cols}))


def test_zip_preserves_raw_units_and_complete_source_books(tmp_path, sqlite_store, monkeypatch):
    ctx = _ctx(tmp_path, sqlite_store, added_on={_NEW: "2026-09-28"})
    _offline(monkeypatch)
    saved = fb.fetch_13f_backfill(ctx, tickers=None, as_of=_AS_OF)
    got = _stored_hr(sqlite_store, _NEW)
    assert saved == 5 and len(got) == 3
    a1 = got.loc[got.cik.eq("0000000001") & got.period.eq(pd.Timestamp("2023-03-31"))].iloc[0]
    assert a1.raw_value_usd == a1.value_usd == 1500
    assert a1.raw_put_value == a1.put_value == 50
    assert a1.shares == 10000 and a1.put_shares == 100
    a2 = got.loc[got.cik.eq(ROSTER)].iloc[0]
    assert a2.value_usd == 3000000 and a2.shares == 20000
    assert not _stored_hr(sqlite_store, "AAPL").empty
    assert not _stored_hr(sqlite_store, "MSFT").empty
    print("SANITY: complete ZIP books keep source quantities/raw values; filing-date unit evidence stays distinct from later P1 market corrections.")


def test_earliest_stored_period_does_not_hide_missing_source_filings(tmp_path, sqlite_store, monkeypatch):
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
    real = fb._archive_scope
    monkeypatch.setattr(fb, "_archive_scope", lambda path, *a: read.append(path.name) or real(path, *a))

    fb.fetch_13f_backfill(ctx, tickers=None, as_of=_AS_OF)
    assert read == ["01jun2024-31aug2024_form13f.zip", "2023q2_form13f.zip"]
    assert _B1[1] in set(sqlite_store.load(Tables.sec13f_hr).accession_number)
    work = fb.backfill_work([fb.DataSet.parse(n) for n in _DATA_SETS], [_NEW])
    assert [d.name for d, _ in work] == ["01jun2024-31aug2024", "2023q2"]
    print("SANITY: an earlier unverified row cannot hide a missing later filing; all source windows remain discoverable.")


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

    got = _stored_hr(sqlite_store, _NEW)
    legacy = got.loc[got.accession_number.eq("_legacy")]
    assert len(legacy) == 1 and legacy.iloc[0].value_usd == 9
    original = got.loc[got.accession_number.eq(_B1[1])]
    assert len(original) == 1 and original.iloc[0].value_usd == 4000000
    assert got.loc[got.accession_number.eq(_A1[1]), "value_usd"].iloc[0] == 1500
    print("SANITY: historical source replay preserves the later legacy row and separately restores the original accession.")


def test_legacy_unit_error_cannot_block_verified_source_packets(tmp_path, sqlite_store, monkeypatch, caplog):
    ctx = _ctx(tmp_path, sqlite_store, added_on={_NEW: "2026-09-28"})
    _offline(monkeypatch)
    _save_oracle_book(sqlite_store, _A2, scale=1000)
    with caplog.at_level("ERROR"):
        fb.fetch_13f_backfill(ctx, tickers=None, as_of=_AS_OF)
    got = _stored_hr(sqlite_store, _NEW)
    assert len(got) == 3
    assert got.loc[got.accession_number.eq(_A2[1]), "value_usd"].iloc[0] == 3000000
    assert not any(r.levelno >= logging.ERROR for r in caplog.records)
    print("SANITY: a previously mis-scaled legacy book cannot veto reconciled source accessions; P1 determines economic scaling.")


@pytest.mark.parametrize(("tickers", "full", "saved_tickers"), [(["AAPL"], False, []), (["AAPL"], True, ["AAPL"]), ([_NEW], False, [_NEW])])
def test_scoped_runs_follow_the_archive_rules(tmp_path, sqlite_store, monkeypatch, tickers, full, saved_tickers):
    ctx = _ctx(tmp_path, sqlite_store, added_on={_NEW: "2026-09-28"})
    _offline(monkeypatch)

    fb.fetch_13f_backfill(ctx, tickers=tickers, as_of=_AS_OF, full=full)

    df = sqlite_store.load(Tables.sec13f_hr, columns=["ticker"], optional=True)
    if not saved_tickers:
        assert df is None
    else:
        assert set(saved_tickers) <= set(df.ticker)
        assert {"AAPL", "MSFT", _NEW} <= set(df.ticker)
    print(f"\n=== SANITY: backfill -t {tickers} {'-F' if full else ''} ===")
    print(f"  requested tickers {saved_tickers}: manager discovery is scoped; complete source portfolios remain stored. Validated.")


def test_the_gap_after_the_newest_data_set_is_walked_once_per_ticker(tmp_path, sqlite_store, monkeypatch):
    ctx = _ctx(tmp_path, sqlite_store, added_on={_NEW: "2026-09-28"})
    _offline(monkeypatch)
    walks = _patch_gap_walk(monkeypatch, save=True)

    fb.fetch_13f_backfill(ctx, tickers=None, as_of=_AS_OF)
    fb.fetch_13f_backfill(ctx, tickers=None, as_of=_AS_OF + pd.Timedelta(days=1))  # night 2, NEWCO still new

    assert len(walks) == 1, walks  # the completed walk's marker marks it done
    assert walks[0]["tickers"] == [_NEW] and walks[0]["filing_window"] == ("2024-09-01", "2026-09-28")
    assert set(walks[0]["all_tickers"]) == {"AAPL", "MSFT", _NEW}, "The gap walk reuses the complete universe, not its new-name subset."
    print("\n=== SANITY: 13F gap walk ===")
    print(f"  NEWCO joined 2026-09-28, newest data set ends 2024-08-31 -> one EDGAR walk {walks[0]['filing_window']};")
    print("  night 2 finds the walk's NEWCO marker and walks nothing. Validated.")


def test_the_gap_starts_after_the_newest_data_set_read_not_the_newest_listed(tmp_path, sqlite_store, monkeypatch):
    ctx = _ctx(tmp_path, sqlite_store, added_on={_NEW: "2026-09-28"})
    _offline(monkeypatch)
    walks = _patch_gap_walk(monkeypatch)
    listed = [*_DATA_SETS, "01sep2024-30nov2024"]  # the newest is listed but its ZIP answers HTTP 404
    links = "".join(f'<a href="/files/structureddata/data/form-13f-data-sets/{n}_form13f.zip">{n}</a>' for n in listed)
    monkeypatch.setattr(fb, "sec_get", lambda *a, **k: SimpleNamespace(text=f"<html>{links}</html>"))
    asked: list[str] = []
    monkeypatch.setattr(bulk_cache, "download", lambda context, url, path, **k: asked.append(url) or 404)

    fb.fetch_13f_backfill(ctx, tickers=None, as_of=_AS_OF)

    assert asked == list(fb.DataSet.parse("01sep2024-30nov2024").urls)
    assert len(walks) == 1 and walks[0]["filing_window"] == ("2024-09-01", "2026-09-28"), walks
    print("\n=== SANITY: 13F gap start after a missing data set ===")
    print(f"  01sep2024-30nov2024 listed but HTTP 404 -> the walk starts after 01jun2024-31aug2024: {walks[0]['filing_window']}. Validated.")


def test_rows_from_the_nightly_overlap_do_not_mark_the_gap_done(tmp_path, sqlite_store, monkeypatch):
    ctx = _ctx(tmp_path, sqlite_store, added_on={_NEW: "2026-09-28"})
    _offline(monkeypatch)
    walks = _patch_gap_walk(monkeypatch)
    nightly = {
        "cik": "0000000008",
        "period": pd.Timestamp("2026-06-30"),
        "ticker": _NEW,
        "cusip": _NEW_CUSIP,
        "filing_date": pd.Timestamp("2026-09-25"),
    }
    sqlite_store.save(Tables.sec13f_hr, pd.DataFrame([nightly]))  # what the join night's walk can write

    fb.fetch_13f_backfill(ctx, tickers=None, as_of=_AS_OF)

    assert len(walks) == 1
    print("\n=== SANITY: gap done test ignores the nightly overlap ===")
    print("  a NEWCO row filed 2026-09-25 (within 7 days of joining) is the nightly walk's; the gap is still walked. Validated.")


def test_rows_a_stale_nightly_walk_wrote_do_not_mark_the_gap_done(tmp_path, sqlite_store, monkeypatch):
    ctx = _ctx(tmp_path, sqlite_store, added_on={_NEW: "2026-09-28"})
    _offline(monkeypatch)
    walks = _patch_gap_walk(monkeypatch)
    # The join night's walk ran from a stale frontier (stage D: HOG rows filed 09-22/09-23, joined 10-06):
    # its rows sit inside [since, added_on - 7d), yet nothing walked 2024-09-01..2026-09-14.
    stale = {"cik": "0000000008", "period": pd.Timestamp("2026-06-30"), "ticker": _NEW, "cusip": _NEW_CUSIP}
    sqlite_store.save(
        Tables.sec13f_hr,
        pd.DataFrame([stale | {"filing_date": pd.Timestamp("2026-09-15")}, stale | {"cik": "0000000009", "filing_date": pd.Timestamp("2026-09-16")}]),
    )

    fb.fetch_13f_backfill(ctx, tickers=None, as_of=_AS_OF)

    assert len(walks) == 1, walks
    print("\n=== SANITY: gap done test vs a stale-frontier nightly walk ===")
    print("  NEWCO rows filed 2026-09-15/16 came from the nightly walk, not the gap walk; the gap is still walked. Validated.")


def test_a_completed_gap_walk_marks_each_walked_ticker_once_and_the_next_night_skips(tmp_path, sqlite_store, monkeypatch):
    ctx = _ctx(tmp_path, sqlite_store, added_on={_NEW: "2026-09-28", "MSFT": "2026-09-27"})
    _offline(monkeypatch)
    walks = _patch_gap_walk(monkeypatch)

    fb.fetch_13f_backfill(ctx, tickers=None, as_of=_AS_OF)
    fb.fetch_13f_backfill(ctx, tickers=None, as_of=_AS_OF + pd.Timedelta(days=1))  # night 2, both still new

    markers = _markers(sqlite_store).sort_values("ticker")
    shown = sqlite_store.load(Tables.sec13f_hr, where={"ticker": ["MSFT", _NEW]})
    assert len(walks) == 1 and walks[0]["tickers"] == ["MSFT", _NEW], walks
    assert markers["ticker"].tolist() == ["MSFT", _NEW]
    assert set(markers["cik"]) == {_SENTINEL}
    assert (pd.to_datetime(markers["period"]) == _GAP_START).all() and (pd.to_datetime(markers["filing_date"]) == _GAP_START).all()
    assert markers[["shares", "value_usd", "put_value"]].isna().all().all()
    assert _SENTINEL not in set(shown["cusip"]) and _SENTINEL not in set(shown["cik"])
    print("\n=== SANITY: a completed gap walk writes one marker per walked ticker ===")
    print(f"  one walk for MSFT+NEWCO -> markers {markers['ticker'].tolist()} (cik/cusip '_empty', period = filing_date = 2024-09-01,")
    print(f"  values NULL); night 2 walks nothing; a consumer read of their {len(shown)} rows shows no marker. Validated.")


def test_a_cut_off_gap_walk_writes_no_marker_and_is_walked_again(tmp_path, sqlite_store, monkeypatch):
    ctx = _ctx(tmp_path, sqlite_store, added_on={_NEW: "2026-09-28"})
    _offline(monkeypatch)
    _patch_gap_walk(monkeypatch, cut=True)  # saves a NEWCO row filed 2025-01-10, then fails

    with pytest.raises(RuntimeError, match="cut off"):
        fb.fetch_13f_backfill(ctx, tickers=None, as_of=_AS_OF)
    after_cut = _markers(sqlite_store)
    walks = _patch_gap_walk(monkeypatch)
    fb.fetch_13f_backfill(ctx, tickers=None, as_of=_AS_OF + pd.Timedelta(days=1))

    assert after_cut.empty
    assert len(walks) == 1 and walks[0]["filing_window"] == ("2024-09-01", "2026-09-28"), walks
    assert _markers(sqlite_store)["ticker"].tolist() == [_NEW]
    print("\n=== SANITY: a cut-off gap walk ===")
    print("  night 1's walk saved a NEWCO row inside the gap, then failed: no marker; night 2 walks the whole gap again")
    print("  and only then writes NEWCO's marker. Validated.")


def test_a_gap_walk_that_leaves_filings_unread_writes_no_marker(tmp_path, sqlite_store, monkeypatch, caplog):
    ctx = _ctx(tmp_path, sqlite_store, added_on={_NEW: "2026-09-28"})
    _offline(monkeypatch)
    _patch_gap_walk(monkeypatch, save=True, unread=2)  # two filings still failing transiently after the retry rounds

    with caplog.at_level("WARNING"):
        fb.fetch_13f_backfill(ctx, tickers=None, as_of=_AS_OF)
    after_night_1 = _markers(sqlite_store)
    walks = _patch_gap_walk(monkeypatch)
    fb.fetch_13f_backfill(ctx, tickers=None, as_of=_AS_OF + pd.Timedelta(days=1))

    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING" and "unread" in r.getMessage()]
    assert after_night_1.empty and len(warnings) == 1, warnings
    assert len(walks) == 1
    print("\n=== SANITY: a gap walk with unread filings ===")
    print(f"  2 filings left unread -> no marker, {warnings[0][:80]!r}...; night 2 walks the gap again. Validated.")


def _seed_frontier_rows(store: Any) -> None:
    """A data-set row (period 2023-03-31) and a nightly row (filed 2026-09-25) for NEWCO; AAPL has none."""
    row = {"cik": "0000000001", "ticker": _NEW, "cusip": _NEW_CUSIP, "shares": 10, "value_usd": 1.0}
    store.save(
        Tables.sec13f_hr,
        pd.DataFrame(
            [
                row | {"period": pd.Timestamp("2023-03-31"), "filing_date": pd.Timestamp("2023-05-15")},
                row | {"period": pd.Timestamp("2026-06-30"), "filing_date": pd.Timestamp("2026-09-25")},
            ]
        ),
    )


def test_gap_markers_move_neither_the_nightly_frontier_nor_the_zip_resume_point(tmp_path, sqlite_store, monkeypatch):
    ctx = _ctx(tmp_path, sqlite_store, added_on={_NEW: "2026-09-28"})
    _seed_frontier_rows(sqlite_store)
    data_sets = [fb.DataSet.parse(n) for n in _DATA_SETS]

    def _frontiers() -> tuple[Any, ...]:
        points = fb.backfill_work(data_sets, [_NEW, "AAPL"])
        return sqlite_store.max_date(Tables.sec13f_hr, "filing_date"), f13._resolve_window(ctx, 15, None, _AS_OF, False), points

    before = _frontiers()
    saved = fb.mark_gap_walked(ctx, [_NEW, "AAPL"], _GAP_START)
    after = _frontiers()

    assert saved == 2
    assert after[0] == before[0] == pd.Timestamp("2026-09-25")  # the nightly frontier max(filing_date)
    assert after[1] == before[1]  # the nightly walk window
    assert after[2] == before[2]
    print("\n=== SANITY: gap markers and the 13F frontiers ===")
    print(f"  max(filing_date) {after[0]:%Y-%m-%d} and nightly window {after[1][0]:%Y-%m-%d}:{after[1][1]:%Y-%m-%d} unchanged;")
    print("  gap markers do not alter archive discovery or certify source accessions. Validated.")


def test_every_sec13f_hr_consumer_read_hides_the_gap_marker(tmp_path, sqlite_store, monkeypatch):
    ctx = _ctx(tmp_path, sqlite_store, added_on={_NEW: "2026-09-28"})
    _offline(monkeypatch)
    _save_oracle_book(sqlite_store, _A2)
    fb.fetch_13f_backfill(ctx, tickers=None, as_of=_AS_OF)  # the data-set rows, then the walk's NEWCO marker
    cache = tmp_path / str(ctx.config.local.paths.sec_13f_datasets)
    tables = fb._read_data_set(cache / "2023q2_form13f.zip", frozenset({_NEW_CUSIP}), {ROSTER})
    assert tables is not None
    book = fb.data_set_book(*tables)
    universe = ["AAPL", "MSFT", _NEW]
    log = logging.getLogger("test")

    def _reads() -> dict[str, Any]:
        return {
            "load": sqlite_store.load(Tables.sec13f_hr),
            "load projected": sqlite_store.load(Tables.sec13f_hr, project=True),
            "iter_load": pd.concat(sqlite_store.iter_load(Tables.sec13f_hr, project=True), ignore_index=True),
            "aggregation load_source": load_source(sqlite_store, log, Tables.sec13f_hr, universe),
            "unit check": fb.unit_check(ctx, book, {ROSTER}),
            "superinvestor activity": activity_evidence(ctx, [ROSTER, "1"]),
        }

    with_marker = _reads()
    n_markers = len(_markers(sqlite_store))
    sqlite_store.delete(Tables.sec13f_hr, {"cusip": _SENTINEL, "cik": _SENTINEL})
    without = _reads()

    assert n_markers == 1
    for name, value in with_marker.items():
        if isinstance(value, pd.DataFrame):
            pd.testing.assert_frame_equal(value.reset_index(drop=True), without[name].reset_index(drop=True), check_exact=True)
            assert "cusip" not in value.columns or _SENTINEL not in set(value["cusip"]), name
        else:
            assert value == without[name], name
    print("\n=== SANITY: sec13f_hr consumers vs the gap marker ===")
    print(f"  with NEWCO's marker stored, {len(with_marker)} reads ({', '.join(with_marker)})")
    print("  return exactly what they return without it. Validated.")


def test_the_cik_padding_guard_keeps_the_marker_sentinel(sqlite_store):
    ctx: Any = SimpleNamespace(store=sqlite_store)
    real = {"cik": "7", "period": pd.Timestamp("2026-06-30"), "filing_date": pd.Timestamp("2026-08-14"), "ticker": _NEW, "cusip": _NEW_CUSIP}

    f13.save_hr(ctx, pd.DataFrame([real]))
    fb.mark_gap_walked(ctx, [_NEW], _GAP_START)

    stored = sqlite_store.load(Tables.sec13f_hr, markers=True)
    assert sorted(zip(stored["cik"], stored["cusip"], strict=True)) == [("0000000007", _NEW_CUSIP), (_SENTINEL, _SENTINEL)]
    print("\n=== SANITY: 13F CIK padding vs the gap marker ===")
    print(f"  save_hr pads cik '7' to '0000000007' and keeps the marker's cik {_SENTINEL!r} (not padded to ''). Validated.")


def test_an_established_ticker_has_no_gap(tmp_path, sqlite_store, monkeypatch):
    ctx = _ctx(tmp_path, sqlite_store, added_on={_NEW: "2026-09-28"})
    _offline(monkeypatch)
    walks = _patch_gap_walk(monkeypatch)

    fb.fetch_13f_backfill(ctx, tickers=["AAPL"], as_of=_AS_OF, full=True)

    assert walks == []  # AAPL joined 2000-01-01, before the newest data set ends
    print("\n=== SANITY: no gap walk for an established ticker ===")
    print("  -t AAPL -F reads the data sets only; AAPL joined long before 2024-08-31, so no EDGAR walk. Validated.")


def test_a_ticker_added_after_as_of_is_not_backfilled_yet(tmp_path, sqlite_store, monkeypatch):
    ctx = _ctx(tmp_path, sqlite_store, added_on={_NEW: "2026-10-03"})  # three days after the run date
    pages = _offline(monkeypatch)
    walks = _patch_gap_walk(monkeypatch)

    nightly = fb.fetch_13f_backfill(ctx, tickers=None, as_of=_AS_OF)
    assert nightly == 0 and pages == [] and walks == []
    assert sqlite_store.load(Tables.sec13f_hr, optional=True) is None

    fb.fetch_13f_backfill(ctx, tickers=[_NEW], as_of=_AS_OF, full=True)
    assert walks == []  # under -F the data sets are read, but a ticker that has not joined yet has no gap to walk
    print("\n=== SANITY: 13F backfill and a future added_on ===")
    print("  NEWCO added 2026-10-03, run date 2026-09-30: the night reads nothing; -t NEWCO -F reads the data sets, no gap walk. Validated.")


def test_gap_tickers_without_added_on_column_is_empty(tmp_path: Any, sqlite_store: Any) -> None:
    tickers = ["AAPL", _NEW]
    ctx = fake_context(tmp_path, sqlite_store, tickers, redundant_ticks=[])
    sqlite_store.drop(Tables.sp500_tickers)
    sqlite_store.save(
        Tables.sp500_tickers, pd.DataFrame({col: ["x"] * len(tickers) for col in CIK_MAPPING_COLS} | {"ticker": tickers, "cik": ["1", "3"]})
    )
    assert "added_on" not in sqlite_store.columns(Tables.sp500_tickers)

    since = pd.Timestamp("2024-09-01")
    nightly = fb.gap_tickers(ctx, tickers, since, _AS_OF, full=False)
    forced = fb.gap_tickers(ctx, tickers, since, _AS_OF, full=True)

    assert nightly == {} and forced == {}
    print("\n=== SANITY: 13F gap tickers before the added_on migration ===")
    print(f"  sp500_tickers without added_on -> gap_tickers {nightly} nightly and {forced} under -F, no KeyError. Validated.")


def test_normal_zip_backfill_preserves_whole_original_and_amended_books(tmp_path, sqlite_store, monkeypatch):
    ctx = _ctx(tmp_path, sqlite_store, added_on={_NEW: "2026-09-28"})
    _offline(monkeypatch)
    amendment = ("1", "0000000001-23-000009", "13F-HR/A", "2023-06-01", "2023-03-31", [(_NEW_CUSIP, "NEWCO", 1600, 11000, "")])
    cache = tmp_path / str(ctx.config.local.paths.sec_13f_datasets)
    _write_zip(cache / "2023q2_form13f.zip", [_A1, amendment])
    fb.fetch_13f_backfill(ctx, tickers=None, as_of=_AS_OF)
    manifests = sqlite_store.load(
        Tables.sec13f_hr,
        columns=["accession_number", "amendment_type", "parsed_row_count"],
        where={"cik": "0000000001", "cusip": "_empty", "parse_status": "complete"},
        markers=True,
    )
    assert set(manifests.accession_number) == {_A1[1], amendment[1], _B1[1]}
    vintages = sqlite_store.load(Tables.sec13f_hr, columns=["accession_number", "cusip", "shares"], where={"cik": "0000000001"})
    original = vintages.loc[vintages.accession_number.eq(_A1[1])]
    assert set(original.cusip) == {_NEW_CUSIP, "037833100"}, "keep the whole book before filtering its NEWCO leg"
    assert original.set_index("cusip").loc[_NEW_CUSIP, "shares"] == 10000
    assert set(vintages.loc[vintages.accession_number.eq(amendment[1])].cusip) == {_NEW_CUSIP}
    first = vintages.sort_values(["accession_number", "cusip"]).reset_index(drop=True)
    fb.fetch_13f_backfill(ctx, tickers=[_NEW], as_of=_AS_OF, full=True)
    again = sqlite_store.load(Tables.sec13f_hr, columns=["accession_number", "cusip", "shares"], where={"cik": "0000000001"})
    pd.testing.assert_frame_equal(first, again.sort_values(["accession_number", "cusip"]).reset_index(drop=True))
    print("SANITY: ordinary cached-ZIP backfill retains original whole book, amended omission and immutable repeat saves.")


def test_invalid_zip_accession_does_not_hide_another_complete_packet(tmp_path):
    path = tmp_path / "source.zip"
    _write_zip(path, [_A1, _A2])
    with zipfile.ZipFile(path) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    summary = pd.read_csv(BytesIO(entries["SUMMARYPAGE.tsv"]), sep="\t")
    summary.loc[summary.ACCESSION_NUMBER.eq(_A1[1]), "TABLEENTRYTOTAL"] = 99
    entries["SUMMARYPAGE.tsv"] = summary.to_csv(sep="\t", index=False).encode()
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    failures = []
    packets = fb.cached_filing_vintages(path, [_A1[1], _A2[1]], on_invalid=lambda accession, error: failures.append((accession, str(error))))
    assert len(packets) == 1 and packets[0][0].iloc[0].accession_number == _A2[1]
    assert len(failures) == 1 and failures[0][0] == _A1[1]
    print("SANITY: invalid source packet is explicitly rejected; independent complete packet remains recoverable.")


def test_zip_scope_preserves_earlier_baseline_and_later_omission(tmp_path, sqlite_store, monkeypatch):
    ctx = _ctx(tmp_path, sqlite_store, added_on={_NEW: "2026-09-28"})
    _offline(monkeypatch)
    cache = tmp_path / str(ctx.config.local.paths.sec_13f_datasets)
    earlier = ("42", "baseline", "13F-HR", "2023-05-15", "2023-03-31", [("037833100", "APPLE", 500, 5, "")])
    first = ("42", "holding", "13F-HR", "2024-08-14", "2024-06-30", [(_NEW_CUSIP, "NEWCO", 1000, 10, "")])
    amended = ("42", "omission", "13F-HR/A", "2024-08-20", "2024-06-30", [("037833100", "APPLE", 500, 5, "")])
    _write_zip(cache / "2023q2_form13f.zip", [earlier])
    _write_zip(cache / "01jun2024-31aug2024_form13f.zip", [first, amended])
    fb.fetch_13f_backfill(ctx, tickers=[_NEW], as_of=_AS_OF, full=True)
    raw = sqlite_store.load(Tables.sec13f_hr, where={"cik": "0000000042"})
    assert set(raw.accession_number) == {"baseline", "holding", "omission"}
    print("SANITY: manager union across ZIPs preserves an earlier baseline and the later amendment that omits the last requested ticker.")


def test_zip_and_xml_equivalent_packet_replays_once(tmp_path, sqlite_store):
    path = tmp_path / "replay.zip"
    _write_zip(path, [_A1])
    packet = fb.cached_filing_vintages(path, [_A1[1]])[0]
    sqlite_store.save_13f_filing(*packet)
    cover = (
        "<edgarSubmission><isAmendment>false</isAmendment><amendmentNo>0</amendmentNo>"
        "<reportCalendarOrQuarter>2023-03-31</reportCalendarOrQuarter>"
        "<reportType>13F HOLDINGS REPORT</reportType><isConfidentialOmitted>false</isConfidentialOmitted>"
        "<tableEntryTotal>3</tableEntryTotal><tableValueTotal>2250</tableValueTotal></edgarSubmission>"
    )
    xml = _xml(_A1[5]).replace("<titleOfClass>COM</titleOfClass>", "<titleOfClass>  COM  </titleOfClass>")
    parsed = f13.xml_filing_vintage(cover, xml, cik=_A1[0], accession_number=_A1[1], period=_A1[4], filing_date=_A1[3], source="edgar")
    assert sqlite_store.save_13f_filing(*parsed) == 0
    assert len(sqlite_store.load(Tables.sec13f_manager_holdings)) == 2
    print("SANITY: actual ZIP and XML parsers produce equivalent accession content despite source whitespace/transport differences.")


def test_original_xml_can_omit_false_amendment_flag():
    cover = (
        "<edgarSubmission><submissionType>13F-HR</submissionType>"
        "<reportCalendarOrQuarter>2023-03-31</reportCalendarOrQuarter>"
        "<reportType>13F HOLDINGS REPORT</reportType><isConfidentialOmitted>false</isConfidentialOmitted>"
        "<tableEntryTotal>3</tableEntryTotal><tableValueTotal>2250</tableValueTotal></edgarSubmission>"
    )
    kwargs = dict(cik=_A1[0], accession_number=_A1[1], period=_A1[4], filing_date=_A1[3], source="edgar")
    manifest, rows = f13.xml_filing_vintage(cover, _xml(_A1[5]), expected_form="13F-HR", **kwargs)
    assert manifest.iloc[0].amendment_type == "original" and len(rows) == 2
    with pytest.raises(ValueError, match="disagrees"):
        f13.xml_filing_vintage(cover, _xml(_A1[5]), expected_form="13F-HR/A", **kwargs)
    with pytest.raises(ValueError, match="missing or invalid"):
        f13.xml_filing_vintage(cover.replace("<submissionType>13F-HR</submissionType>", ""), _xml(_A1[5]), **kwargs)
    for form, flag in (("13F-HR/A", "false"), ("13F-HR", "true")):
        conflicting = cover.replace(
            "<submissionType>13F-HR</submissionType>", f"<submissionType>{form}</submissionType><isAmendment>{flag}</isAmendment>"
        )
        with pytest.raises(ValueError, match="disagrees with cover form"):
            f13.xml_filing_vintage(conflicting, _xml(_A1[5]), **kwargs)
    print(
        "SANITY: missing false amendment checkbox is accepted only for an explicitly original SEC cover; unknown/conflicting form remains rejected."
    )


def test_backfill_reuses_one_map_load_across_archives(tmp_path: Any, sqlite_store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _ctx(tmp_path, sqlite_store, added_on={_NEW: "2026-09-28"})
    _offline(monkeypatch)
    original = sqlite_store.load
    map_loads = []

    def counted(table: Any, *args: Any, **kwargs: Any) -> Any:
        if table == Tables.cusip_ticker_map:
            map_loads.append(table)
        return original(table, *args, **kwargs)

    monkeypatch.setattr(sqlite_store, "load", counted)
    fb.fetch_13f_backfill(ctx, tickers=[_NEW], as_of=_AS_OF, full=True)
    assert len(map_loads) == 1
    assert sqlite_store.load(Tables.sec13f_hr) is not None
    assert set(sqlite_store.load(Tables.sec13f_manager_holdings).accession_number) == {_A2[1]}
    print("SANITY: the actual two-archive backfill saver loads the map once and writes full broad and roster books.")


def test_completed_backfill_skips_decode_but_keeps_discovery(tmp_path: Any, sqlite_store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _ctx(tmp_path, sqlite_store, added_on={_NEW: "2026-09-28"})
    _offline(monkeypatch)
    fb.fetch_13f_backfill(ctx, tickers=[_NEW], as_of=_AS_OF, full=True)
    discovered = []
    original = fb._archive_scope

    def scope(path: Any, *args: Any) -> Any:
        discovered.append(path.name)
        return original(path, *args)

    monkeypatch.setattr(fb, "_archive_scope", scope)
    monkeypatch.setattr(fb, "cached_filing_vintages", lambda *args, **kwargs: pytest.fail("All required destinations already complete"))
    assert fb.fetch_13f_backfill(ctx, tickers=[_NEW], as_of=_AS_OF) == 0
    assert discovered == ["01jun2024-31aug2024_form13f.zip", "2023q2_form13f.zip"]
    print("SANITY: completed ordinary rerun still discovers possible holders but performs no second decode/map/save pass.")


def test_missing_roster_destination_remains_pending(tmp_path: Any, sqlite_store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _ctx(tmp_path, sqlite_store, added_on={_NEW: "2026-09-28"})
    _offline(monkeypatch)
    fb.fetch_13f_backfill(ctx, tickers=[_NEW], as_of=_AS_OF, full=True)
    sqlite_store.delete(Tables.sec13f_manager_holdings, where={"accession_number": _A2[1]})
    original = fb.cached_filing_vintages
    decoded = []

    def decode(path: Any, accessions: list[str], **kwargs: Any) -> Any:
        decoded.extend(accessions)
        return original(path, accessions, **kwargs)

    monkeypatch.setattr(fb, "cached_filing_vintages", decode)
    fb.fetch_13f_backfill(ctx, tickers=[_NEW], as_of=_AS_OF)
    assert decoded == [_A2[1]]
    assert set(sqlite_store.load(Tables.sec13f_manager_holdings).accession_number) == {_A2[1]}
    print("SANITY: broad completion cannot hide a missing roster destination; only that accession is decoded and repaired.")


def test_full_backfill_rechecks_completed_sources(tmp_path: Any, sqlite_store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _ctx(tmp_path, sqlite_store, added_on={_NEW: "2026-09-28"})
    _offline(monkeypatch)
    fb.fetch_13f_backfill(ctx, tickers=[_NEW], as_of=_AS_OF, full=True)
    original = fb.cached_filing_vintages
    decoded = []

    def decode(path: Any, accessions: list[str], **kwargs: Any) -> Any:
        decoded.extend(accessions)
        return original(path, accessions, **kwargs)

    monkeypatch.setattr(fb, "cached_filing_vintages", decode)
    assert fb.fetch_13f_backfill(ctx, tickers=[_NEW], as_of=_AS_OF, full=True) == 0
    assert set(decoded) == {_A1[1], _A2[1], _B1[1]}
    print("SANITY: explicit -F re-decodes completed originals and checks immutable replay rather than becoming an unchecked no-op.")


def test_cusip_scope_keeps_unmapped_stored_holder_for_omission(tmp_path: Any, sqlite_store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _ctx(tmp_path, sqlite_store, added_on={_NEW: "2026-09-28"})
    _offline(monkeypatch)
    original = ("42", "original-unmapped", "13F-HR", "2024-08-14", "2024-06-30", [(_NEW_CUSIP, "NEWCO", 1000, 10, "")])
    source = tmp_path / "original.zip"
    _write_zip(source, [original])
    manifest, positions = fb.cached_filing_vintages(source, [original[1]])[0]
    sqlite_store.save_13f_filing(manifest, positions.assign(ticker="_empty"), table=Tables.sec13f_hr)
    omission = ("42", "later-omission", "13F-HR/A", "2024-08-20", "2024-06-30", [("037833100", "APPLE", 500, 5, "")])
    cache = tmp_path / str(ctx.config.local.paths.sec_13f_datasets)
    for name in _DATA_SETS:
        _write_zip(cache / f"{name}_form13f.zip", [omission])
    fb.fetch_13f_backfill(ctx, tickers=[_NEW], as_of=_AS_OF)
    got = sqlite_store.load(Tables.sec13f_hr, where={"cik": "0000000042"})
    assert set(got.accession_number) == {"original-unmapped", "later-omission"}
    assert got.loc[got.accession_number.eq("original-unmapped"), "ticker"].eq("_empty").all()
    print("SANITY: CUSIP-aware stored scope recovers a later omission despite an unmapped stored ticker, without rewriting the original.")
