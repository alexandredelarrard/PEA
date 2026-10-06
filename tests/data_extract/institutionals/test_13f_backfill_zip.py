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
    df = store.load(Tables.sec13f_hr, where={"cusip": _SENTINEL}, markers=True, optional=True)
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


def test_the_gap_after_the_newest_data_set_is_walked_once_per_ticker(tmp_path, sqlite_store, monkeypatch):
    ctx = _ctx(tmp_path, sqlite_store, added_on={_NEW: "2026-09-28"})
    _offline(monkeypatch)
    walks = _patch_gap_walk(monkeypatch, save=True)

    fb.fetch_13f_backfill(ctx, tickers=None, as_of=_AS_OF)
    fb.fetch_13f_backfill(ctx, tickers=None, as_of=_AS_OF + pd.Timedelta(days=1))  # night 2, NEWCO still new

    assert len(walks) == 1, walks  # the completed walk's marker marks it done
    assert walks[0]["tickers"] == [_NEW] and walks[0]["filing_window"] == ("2024-09-01", "2026-09-28")
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

    assert [u.rsplit("/", 1)[-1] for u in asked] == ["01sep2024-30nov2024_form13f.zip"]
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
        points = fb.resume_points(ctx, [_NEW, "AAPL"])
        return sqlite_store.max_date(Tables.sec13f_hr, "filing_date"), f13._resolve_window(ctx, 15, None, _AS_OF, False), points

    before = _frontiers()
    saved = fb.mark_gap_walked(ctx, [_NEW, "AAPL"], _GAP_START)
    after = _frontiers()

    assert saved == 2
    assert after[0] == before[0] == pd.Timestamp("2026-09-25")  # the nightly frontier max(filing_date)
    assert after[1] == before[1]  # the nightly walk window
    assert after[2][_NEW] == before[2][_NEW] == pd.Timestamp("2023-03-31")  # the ZIP resume point min(period)
    # A ticker with no holding row: its resume point becomes the gap start, which reads the same data sets.
    assert before[2]["AAPL"] is None and after[2]["AAPL"] == _GAP_START
    assert fb.backfill_work(data_sets, after[2], False) == fb.backfill_work(data_sets, before[2], False)
    print("\n=== SANITY: gap markers and the 13F frontiers ===")
    print(f"  max(filing_date) {after[0]:%Y-%m-%d} and the nightly window {after[1][0]:%Y-%m-%d}:{after[1][1]:%Y-%m-%d} unchanged;")
    print(f"  NEWCO min(period) stays {after[2][_NEW]:%Y-%m-%d}; a rowless AAPL reads the same data sets either way. Validated.")


def test_every_sec13f_hr_consumer_read_hides_the_gap_marker(tmp_path, sqlite_store, monkeypatch):
    ctx = _ctx(tmp_path, sqlite_store, added_on={_NEW: "2026-09-28"})
    _offline(monkeypatch)
    _save_oracle_book(sqlite_store, _A2)
    fb.fetch_13f_backfill(ctx, tickers=None, as_of=_AS_OF)  # the data-set rows, then the walk's NEWCO marker
    cache = tmp_path / str(ctx.config.local.paths.sec_13f_datasets)
    cmap = sqlite_store.load(Tables.cusip_ticker_map)
    tables = fb._read_data_set(cache / "2023q2_form13f.zip", frozenset({_NEW_CUSIP}), {ROSTER})
    assert tables is not None
    book = fb.data_set_book(*tables)
    candidates = f13._resolve_tickers(book, cmap, {_NEW})
    universe = ["AAPL", "MSFT", _NEW]
    log = logging.getLogger("test")

    def _reads() -> dict[str, Any]:
        return {
            "load": sqlite_store.load(Tables.sec13f_hr),
            "load projected": sqlite_store.load(Tables.sec13f_hr, project=True),
            "iter_load": pd.concat(sqlite_store.iter_load(Tables.sec13f_hr, project=True), ignore_index=True),
            "aggregation load_source": load_source(sqlite_store, log, Tables.sec13f_hr, universe),
            "superseded check": fb._drop_superseded(ctx, candidates),
            "unit check": fb.unit_check(ctx, book, {ROSTER}),
            "superinvestor activity": activity_evidence(ctx, [ROSTER, "1"]),
        }

    with_marker = _reads()
    n_markers = len(_markers(sqlite_store))
    sqlite_store.delete(Tables.sec13f_hr, {"cusip": _SENTINEL})
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


def test_gap_tickers_without_added_on_column_is_empty(tmp_path: Any, sqlite_store: Any) -> None:
    tickers = ["AAPL", _NEW]
    ctx = fake_context(tmp_path, sqlite_store, tickers, redundant_ticks=[])
    sqlite_store.drop(Tables.sp500_tickers)
    sqlite_store.save(
        Tables.sp500_tickers, pd.DataFrame({col: ["x"] * len(tickers) for col in CIK_MAPPING_COLS} | {"ticker": tickers, "cik": ["1", "3"]})
    )
    assert "added_on" not in sqlite_store.columns(Tables.sp500_tickers)

    since = pd.Timestamp("2024-09-01")
    nightly = fb.gap_tickers(ctx, tickers, since, full=False)
    forced = fb.gap_tickers(ctx, tickers, since, full=True)

    assert nightly == {} and forced == {}
    print("\n=== SANITY: 13F gap tickers before the added_on migration ===")
    print(f"  sp500_tickers without added_on -> gap_tickers {nightly} nightly and {forced} under -F, no KeyError. Validated.")
