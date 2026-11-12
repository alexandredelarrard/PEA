"""
test_13f_zips.py (tests/data_extract/institutionals/test_13f_zips.py)
---------------------------------------------------------------------
The 13F data-set ZIP readers and the `zips-ingest --only 13f` ingest (`fetch_13f_zips`) on a real SQLite store and
small fixture ZIPs written to `tmp_path` (no download, no SEC request).
"""

from __future__ import annotations

import logging
import zipfile
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from edgar.thirteenf.models import ThirteenF

from src.data_extract.utils.institutionals import fetch_13f as f13
from src.data_extract.utils.institutionals import fetch_13f_zips as z13
from src.data_extract.utils.institutionals.fetch_13f import save_hr_filing
from src.data_extract.utils.institutionals.fetch_13f_zips import cached_filing_vintages
from src.data_store.schema import Tables
from tests.data_extract.edgar_fixtures import fake_context

# (cik, accession, form, filed, period, [(cusip, issuer, value as filed, shares, putcall)])
_FILING = ("1", "0000000001-24-000004", "13F-HR", "2024-08-14", "2024-06-30", [("111111111", "NEWCO", 4_000_000, 25_000, "")])


def _dmy(day: str) -> str:
    return pd.Timestamp(day).strftime("%d-%b-%Y").upper()


def write_filings_zip(path: Any, filings: list[tuple], *, confidential: str | None = "N", omit_confidential: bool = False) -> None:
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
    write_filings_zip(path, [_FILING], confidential=flag, omit_confidential=omit_column)
    packets = cached_filing_vintages(path, [_FILING[1]])
    assert len(packets) == 1
    manifest, holdings = packets[0]
    assert bool(manifest.restricted.iloc[0]) is (restricted and not omit_column)
    assert holdings.shares.iloc[0] == 25000 and holdings.value_usd.iloc[0] == 4000000
    print(f"SANITY: ZIP flag {flag!r}, absent column={omit_column}; only affirmative metadata restricts a complete book.")


# `ingest_13f_datasets` (`zips-ingest --only 13f`): fixture data sets in the cache of a real SQLite store.

_NEW, _NEW_CUSIP = "NEWCO", "111111111"
_MAP = {"037833100": "AAPL", "594918104": "MSFT", _NEW_CUSIP: _NEW}
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
_B1 = _FILING
_DATA_SETS = {"2023q2": [_A1, _A2, _A3], "01jun2024-31aug2024": [_B1]}


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
    """The nightly path for one filing: edgartools' `ThirteenF.infotable`, then `_book_frame`."""
    cik, _, form, filed, period, lines = filing
    thirteen_f = ThirteenF(SimpleNamespace(form=form, period_of_report=period, xml=lambda: None))
    thirteen_f.__dict__["infotable_xml"] = _xml(lines)
    thirteen_f.__dict__["_schema_version"] = None
    infotable = thirteen_f.infotable
    assert infotable is not None
    return f13._book_frame(cik, filed, period, infotable)


def _cache(ctx: Any) -> Path:
    return Path(ctx.paths["DATA_STORE"]) / str(ctx.config.local.paths.sec_13f_datasets)


def _ctx(tmp_path: Any, store: Any, data_sets: dict[str, list[tuple]] | None = None) -> Any:
    """A context over `store` with the CUSIP map seeded and `data_sets` written into the 13F ZIP cache."""
    ctx = fake_context(tmp_path, store, ["AAPL", "MSFT", _NEW])
    store.save(Tables.cusip_ticker_map, pd.DataFrame({"cusip": list(_MAP), "ticker": list(_MAP.values())}))
    cache = _cache(ctx)
    cache.mkdir(parents=True, exist_ok=True)
    for name, filings in (_DATA_SETS if data_sets is None else data_sets).items():
        write_filings_zip(cache / f"{name}_form13f.zip", filings)
    return ctx


def _set_summary(path: Path, accession: str, column: str, value: object) -> None:
    """Rewrite one SUMMARYPAGE cell of a fixture ZIP (an invalid packet: the cover no longer reconciles)."""
    with zipfile.ZipFile(path) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    summary = pd.read_csv(BytesIO(entries["SUMMARYPAGE.tsv"]), sep="\t")
    summary.loc[summary.ACCESSION_NUMBER.eq(accession), column] = value
    entries["SUMMARYPAGE.tsv"] = summary.to_csv(sep="\t", index=False).encode()
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in entries.items():
            archive.writestr(name, content)


def _markers(store: Any, **where: object) -> pd.DataFrame:
    df = store.load(Tables.sec13f_hr, where={"cusip": "_empty", **where}, markers=True, optional=True)
    return pd.DataFrame(columns=["accession_number", "parse_status", "period"]) if df is None else df


def _positions(store: Any, **where: object) -> pd.DataFrame:
    df = store.load(Tables.sec13f_hr, where=where or None, optional=True)
    if df is None:
        return pd.DataFrame(columns=["accession_number", "cik", "period", "cusip", "ticker", "shares"])
    return df.assign(period=pd.to_datetime(df["period"]), filing_date=pd.to_datetime(df["filing_date"]))


def _spy_decode(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Record the accessions of every `cached_filing_vintages` call the ingest makes."""
    calls: list[list[str]] = []
    real = z13.cached_filing_vintages

    def decode(path: Any, accessions: list[str], **kwargs: Any) -> Any:
        calls.append(list(accessions))
        return real(path, accessions, **kwargs)

    monkeypatch.setattr(z13, "cached_filing_vintages", decode)
    return calls


def _snapshot(store: Any) -> pd.DataFrame:
    df = store.load(Tables.sec13f_hr, markers=True)
    columns = ["cik", "period", "accession_number", "ticker", "cusip", "shares", "value_usd", "parse_status", "content_checksum"]
    return df[columns].astype(str).sort_values(columns[:5]).reset_index(drop=True)


def test_ingest_stores_complete_books_through_the_walk_writer(tmp_path: Any, filing_store: Any) -> None:
    ctx = _ctx(tmp_path, filing_store)
    saved = z13.ingest_13f_datasets(ctx, full=False)

    markers = _markers(filing_store).set_index("accession_number")
    got = _positions(filing_store)
    assert saved == 5 and len(got) == 5
    assert set(markers.index) == {_A1[1], _A2[1], _B1[1]}, "every 13F-HR(/A) accession; the 13F-NT is not a holdings report"
    assert markers.parse_status.eq("complete").all() and markers.content_checksum.notna().all()
    assert markers.loc[_A1[1], "source"] == "2023q2_form13f.zip" and markers.loc[_B1[1], "source"] == "01jun2024-31aug2024_form13f.zip"
    assert int(markers.loc[_A1[1], "parsed_row_count"]) == 2, "A1's two NEWCO lines are one CUSIP row, as the walk groups them"
    assert set(got.ticker) == {"AAPL", "MSFT", _NEW}
    assert set(f13.completed_accessions(ctx, ["complete"])) == {_A1[1], _A2[1], _B1[1]}
    print("\n=== SANITY CHECK: 13F ZIP ingest writes through the walk writer ===")
    print(f"  2 windows -> {saved} position rows + 3 completion markers (source = ZIP filename); the 13F-NT skipped. Validated.")


def test_ingest_keeps_original_and_amended_books(tmp_path: Any, filing_store: Any) -> None:
    amendment = ("1", "0000000001-23-000009", "13F-HR/A", "2023-06-01", "2023-03-31", [(_NEW_CUSIP, "NEWCO", 1600, 11000, "")])
    ctx = _ctx(tmp_path, filing_store, {"2023q2": [_A1, amendment], "01jun2024-31aug2024": [_B1]})
    z13.ingest_13f_datasets(ctx, full=False)

    manifests = _markers(filing_store, cik="0000000001")
    assert set(manifests.accession_number) == {_A1[1], amendment[1], _B1[1]}
    assert set(manifests.loc[manifests.accession_number.eq(amendment[1]), "amendment_type"]) == {"restatement"}
    vintages = _positions(filing_store, cik="0000000001")
    original = vintages.loc[vintages.accession_number.eq(_A1[1])]
    assert set(original.cusip) == {_NEW_CUSIP, "037833100"}, "the whole original book is kept"
    assert original.set_index("cusip").loc[_NEW_CUSIP, "shares"] == 10000
    assert set(vintages.loc[vintages.accession_number.eq(amendment[1])].cusip) == {_NEW_CUSIP}, "the amendment omits AAPL"
    first = _snapshot(filing_store)
    assert z13.ingest_13f_datasets(ctx, full=True) == 0
    pd.testing.assert_frame_equal(first, _snapshot(filing_store))
    print("\n=== SANITY CHECK: original + amended 13F books ===")
    print("  the original whole book and the amendment's omission are separate accessions; a -F replay changes nothing. Validated.")


def test_ingest_raw_units_follow_per_filing_evidence(tmp_path: Any, filing_store: Any) -> None:
    ctx = _ctx(tmp_path, filing_store)
    z13.ingest_13f_datasets(ctx, full=False)

    got = _positions(filing_store, ticker=_NEW)
    a1 = got.loc[got.cik.eq("0000000001") & got.period.eq(pd.Timestamp("2023-03-31"))].iloc[0]
    # One share row at 1500 / 10,000 sh: only the few-rows implied price decides ($150/sh at x1000, $0.15 at x1).
    assert a1.raw_value_usd == 1500 and a1.value_usd == 1_500_000
    assert a1.raw_put_value == 50 and a1.put_value == 50_000
    assert a1.shares == 10000 and a1.put_shares == 100
    a2 = got.loc[got.cik.eq("0001067983")].iloc[0]
    assert a2.value_usd == 3_000_000 and a2.raw_value_usd == 3_000_000 and a2.shares == 20000
    print("\n=== SANITY CHECK: 13F ZIP units ===")
    print("  A1 raw 1,500 -> $1.5M (x1000 by implied price), A2 raw $3M kept at x1: units from per-filing evidence. Validated.")


def test_batch_cusips_mapped_in_one_pass(tmp_path: Any, filing_store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    filings = [
        (str(10 + i), f"00000000{10 + i}-24-000001", "13F-HR", "2024-08-14", "2024-06-30", [(f"99999999{i}", "X", 1000, 10, "")]) for i in range(3)
    ]
    ctx = _ctx(tmp_path, filing_store, {"01jun2024-31aug2024": filings})
    monkeypatch.setattr(z13, "_BATCH_ACCESSIONS", 2)
    lookups: list[list[str]] = []

    def fake_map(context: Any, cusips: list[str], *args: Any, cached: Any = None, include_misses: bool = False) -> pd.DataFrame:
        lookups.append(sorted(cusips))
        known = pd.DataFrame(columns=["cusip", "ticker"]) if cached is None else cached
        return pd.concat([known, pd.DataFrame({"cusip": cusips, "ticker": [None] * len(cusips)})], ignore_index=True).drop_duplicates("cusip")

    monkeypatch.setattr(f13, "build_cusip_ticker_map", fake_map)
    z13.ingest_13f_datasets(ctx, full=False)

    assert [lookup for lookup in lookups if lookup] == [["999999990", "999999991"], ["999999992"]], lookups
    assert len(_markers(filing_store)) == 3 and _positions(filing_store).ticker.eq("_empty").all()
    print("\n=== SANITY CHECK: one CUSIP-map pass per batch ===")
    print(f"  3 accessions, batch 2 -> lookups {lookups}: one OpenFIGI pass per batch, never one per packet. Validated.")


def test_stored_accession_is_not_reread(tmp_path: Any, filing_store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _ctx(tmp_path, filing_store)
    walk_packet = z13.cached_filing_vintages(_cache(ctx) / "2023q2_form13f.zip", [_A1[1]])[0]
    save_hr_filing(filing_store, *walk_packet, None)  # the walk stored A1
    decoded = _spy_decode(monkeypatch)

    z13.ingest_13f_datasets(ctx, full=False)

    assert decoded == [[_A2[1]], [_B1[1]]], decoded
    assert set(_markers(filing_store).accession_number) == {_A1[1], _A2[1], _B1[1]}
    print("\n=== SANITY CHECK: a stored accession is not decoded again ===")
    print(f"  A1 stored by the walk -> decoded {decoded}; the ingest only completes what the table lacks. Validated.")


def test_window_without_pending_reads_no_infotable(tmp_path: Any, filing_store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _ctx(tmp_path, filing_store)
    z13.ingest_13f_datasets(ctx, full=False)
    members: list[list[str]] = []
    real = z13.read_zip_tables

    def read(path: Any, specs: Any, **kwargs: Any) -> Any:
        members.append(sorted(specs))
        return real(path, specs, **kwargs)

    monkeypatch.setattr(z13, "read_zip_tables", read)
    z13.ingest_13f_datasets(ctx, full=False)

    assert members == [["COVERPAGE.tsv", "SUBMISSION.tsv"]] * 2, members
    print("\n=== SANITY CHECK: a fully stored window ===")
    print(f"  2 stored windows -> member reads {members}: SUBMISSION + COVERPAGE only, no INFOTABLE scan. Validated.")


def test_second_run_saves_nothing(tmp_path: Any, filing_store: Any) -> None:
    ctx = _ctx(tmp_path, filing_store)
    first = z13.ingest_13f_datasets(ctx, full=False)
    before = _snapshot(filing_store)
    second = z13.ingest_13f_datasets(ctx, full=False)

    assert first == 5 and second == 0
    pd.testing.assert_frame_equal(before, _snapshot(filing_store))
    print("\n=== SANITY CHECK: 13F ZIP ingest is idempotent ===")
    print(f"  run 1 saved {first} rows, run 2 saved {second}; the table is unchanged. Validated.")


def test_full_rechecks_by_checksum(tmp_path: Any, filing_store: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    ctx = _ctx(tmp_path, filing_store)
    z13.ingest_13f_datasets(ctx, full=False)
    before = _snapshot(filing_store)
    decoded = _spy_decode(monkeypatch)

    assert z13.ingest_13f_datasets(ctx, full=True) == 0
    assert sorted(a for call in decoded for a in call) == sorted([_A1[1], _A2[1], _B1[1]])
    pd.testing.assert_frame_equal(before, _snapshot(filing_store))

    filing_store.update(Tables.sec13f_hr, {"content_checksum": "tampered"}, {"accession_number": _A2[1], "cusip": "_empty"})
    with caplog.at_level(logging.ERROR):
        assert z13.ingest_13f_datasets(ctx, full=True) == 0
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 1 and _A2[1] in errors[0] and "checksum" in errors[0], errors
    print("\n=== SANITY CHECK: -F re-verifies stored accessions ===")
    print(f"  -F decodes all 3 stored accessions and saves 0 rows; a tampered A2 checksum is logged ({errors[0][:60]!r}...). Validated.")


def test_full_never_duplicates_a_walk_packet_under_another_period(tmp_path: Any, filing_store: Any, caplog: pytest.LogCaptureFixture) -> None:
    ctx = _ctx(tmp_path, filing_store)
    shifted = tmp_path / "walk.zip"
    write_filings_zip(shifted, [(*_A2[:4], "2022-12-31", _A2[5])])
    save_hr_filing(filing_store, *z13.cached_filing_vintages(shifted, [_A2[1]])[0], None)  # the walk's copy under 2022-12-31

    with caplog.at_level(logging.WARNING):
        z13.ingest_13f_datasets(ctx, full=True)

    homes = _markers(filing_store, accession_number=_A2[1])
    assert pd.to_datetime(homes.period).tolist() == [pd.Timestamp("2022-12-31")], "A2 is never written under a second period"
    assert _positions(filing_store, accession_number=_A2[1]).period.eq(pd.Timestamp("2022-12-31")).all()
    assert any(_A2[1] in r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)
    assert {_A1[1], _B1[1]} <= set(_markers(filing_store).accession_number)
    print("\n=== SANITY CHECK: -F and a walk packet stored under another (cik, period) ===")
    print("  A2 stored under 2022-12-31 by the walk; the ZIP's 2023-03-31 copy is logged and not written. Validated.")


def test_invalid_or_conflicting_packet_is_logged_and_others_saved(tmp_path: Any, filing_store: Any, caplog: pytest.LogCaptureFixture) -> None:
    ctx = _ctx(tmp_path, filing_store)
    _set_summary(_cache(ctx) / "2023q2_form13f.zip", _A1[1], "TABLEENTRYTOTAL", 99)  # A1 no longer reconciles
    conflicting = tmp_path / "conflict.zip"
    write_filings_zip(conflicting, [(*_A2[:5], [(_NEW_CUSIP, "NEWCO", 3_000_000, 21_000, ""), _A2[5][1]])])
    save_hr_filing(filing_store, *z13.cached_filing_vintages(conflicting, [_A2[1]])[0], None)  # another A2 content

    with caplog.at_level(logging.ERROR):
        saved = z13.ingest_13f_datasets(ctx, full=True)

    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert any(_A1[1] in e for e in errors) and any(_A2[1] in e and "checksum" in e for e in errors), errors
    assert saved == 1 and set(_positions(filing_store, accession_number=_B1[1]).ticker) == {_NEW}
    assert _positions(filing_store, accession_number=_A2[1]).shares.max() == 21_000, "the stored A2 is never replaced"
    print("\n=== SANITY CHECK: invalid and conflicting 13F packets ===")
    print(f"  A1 invalid and A2 checksum conflict logged ({len(errors)} errors); B1 in the next window still saved. Validated.")


def test_invalid_marker_is_written_and_skipped_next_run(tmp_path: Any, filing_store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _ctx(tmp_path, filing_store)
    _set_summary(_cache(ctx) / "2023q2_form13f.zip", _A1[1], "TABLEENTRYTOTAL", 99)
    z13.ingest_13f_datasets(ctx, full=False)

    marker = _markers(filing_store, accession_number=_A1[1])
    assert len(marker) == 1 and marker.iloc[0].parse_status == "invalid" and marker.iloc[0].ticker == "_empty"
    assert marker.iloc[0].source == "2023q2_form13f.zip" and str(marker.iloc[0].value_unit_evidence)
    assert pd.Timestamp(marker.iloc[0].period) == pd.Timestamp("2023-03-31") and marker.iloc[0].cik == "0000000001"
    assert _A1[1] not in set(_positions(filing_store).accession_number), "a consumer read never sees the invalid marker"
    assert _A1[1] not in f13.completed_accessions(ctx, ["complete"]), "the walk still treats A1 as not stored"
    decoded = _spy_decode(monkeypatch)
    assert z13.ingest_13f_datasets(ctx, full=False) == 0 and decoded == []
    print("\n=== SANITY CHECK: D7 invalid marker ===")
    print(f"  A1 invalid -> one 'invalid' marker (reason {str(marker.iloc[0].value_unit_evidence)[:50]!r}); the next run decodes nothing. Validated.")


def test_valid_packet_replaces_an_invalid_marker(tmp_path: Any, filing_store: Any) -> None:
    ctx = _ctx(tmp_path, filing_store)
    path = _cache(ctx) / "2023q2_form13f.zip"
    _set_summary(path, _A1[1], "TABLEENTRYTOTAL", 99)
    z13.ingest_13f_datasets(ctx, full=False)
    write_filings_zip(path, _DATA_SETS["2023q2"])  # the source corrected: A1 reconciles again

    saved = z13.ingest_13f_datasets(ctx, full=True)
    replay = save_hr_filing(filing_store, *z13.cached_filing_vintages(path, [_A1[1]])[0], None)

    marker = _markers(filing_store, accession_number=_A1[1])
    assert saved == 2 and replay == 0
    assert marker.parse_status.tolist() == ["complete"], "exactly one marker, now complete"
    assert len(_positions(filing_store, accession_number=_A1[1])) == 2
    print("\n=== SANITY CHECK: a valid packet replaces an invalid marker ===")
    print("  -F re-decodes invalid A1; save_hr_filing swaps its marker for the complete one (2 rows); a walk replay saves 0. Validated.")


def test_pending_beyond_one_batch_are_all_saved(tmp_path: Any, filing_store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _ctx(tmp_path, filing_store)
    monkeypatch.setattr(z13, "_BATCH_ACCESSIONS", 1)
    decoded = _spy_decode(monkeypatch)

    saved = z13.ingest_13f_datasets(ctx, full=False)

    assert decoded == [[_A1[1]], [_A2[1]], [_B1[1]]], decoded
    assert saved == 5 and set(_markers(filing_store).accession_number) == {_A1[1], _A2[1], _B1[1]}
    print("\n=== SANITY CHECK: batches ===")
    print(f"  batch size 1 -> {len(decoded)} decode calls; every pending accession saved ({saved} rows). Validated.")


def test_ingest_keeps_a_later_legacy_row(tmp_path: Any, filing_store: Any) -> None:
    ctx = _ctx(tmp_path, filing_store)
    later = {
        "cik": "0000000001",
        "period": pd.Timestamp("2024-06-30"),
        "ticker": _NEW,
        "cusip": _NEW_CUSIP,
        "filing_date": pd.Timestamp("2024-09-02"),
    }
    filing_store.save(Tables.sec13f_hr, pd.DataFrame([later | {"value_usd": 9.0}]).assign(accession_number="_legacy"))

    z13.ingest_13f_datasets(ctx, full=True)

    got = _positions(filing_store, ticker=_NEW)
    legacy = got.loc[got.accession_number.eq("_legacy")]
    assert len(legacy) == 1 and legacy.iloc[0].value_usd == 9
    assert got.loc[got.accession_number.eq(_B1[1]), "value_usd"].tolist() == [4_000_000]
    print("\n=== SANITY CHECK: a later legacy row ===")
    print("  the ZIP replay keeps the legacy row and stores the original accession beside it. Validated.")


def test_legacy_unit_error_cannot_block_verified_source_packets(tmp_path: Any, filing_store: Any, caplog: pytest.LogCaptureFixture) -> None:
    ctx = _ctx(tmp_path, filing_store)
    book = _edgartools_book(_A2)
    value_cols = ["value_usd", "call_value", "put_value", "debt_value", "other_value"]
    scaled = book.assign(**{c: book[c] * 1000 for c in value_cols})
    filing_store.save(Tables.sec13f_hr, scaled.assign(accession_number="_legacy", ticker=book["cusip"].map(_MAP).fillna("_empty")))

    with caplog.at_level(logging.ERROR):
        z13.ingest_13f_datasets(ctx, full=False)

    got = _positions(filing_store, ticker=_NEW)
    assert len(got.loc[got.accession_number.ne("_legacy")]) == 3
    assert got.loc[got.accession_number.eq(_A2[1]), "value_usd"].iloc[0] == 3_000_000
    assert not any(r.levelno >= logging.ERROR for r in caplog.records)
    print("\n=== SANITY CHECK: a mis-scaled legacy book ===")
    print("  a x1000 legacy A2 book cannot veto the reconciled source accessions; no error logged. Validated.")


def test_invalid_zip_accession_does_not_hide_another_complete_packet(tmp_path: Any) -> None:
    path = tmp_path / "source.zip"
    write_filings_zip(path, [_A1, _A2])
    _set_summary(path, _A1[1], "TABLEENTRYTOTAL", 99)
    failures: list[tuple[str, str]] = []
    packets = cached_filing_vintages(path, [_A1[1], _A2[1]], on_invalid=lambda accession, error: failures.append((accession, str(error))))
    assert len(packets) == 1 and packets[0][0].iloc[0].accession_number == _A2[1]
    assert len(failures) == 1 and failures[0][0] == _A1[1]
    print("SANITY: an invalid source packet is rejected; the independent complete packet remains recoverable.")


def test_ingest_preserves_earlier_baseline_and_later_omission(tmp_path: Any, filing_store: Any) -> None:
    earlier = ("42", "baseline", "13F-HR", "2023-05-15", "2023-03-31", [("037833100", "APPLE", 500, 5, "")])
    first = ("42", "holding", "13F-HR", "2024-08-14", "2024-06-30", [(_NEW_CUSIP, "NEWCO", 1000, 10, "")])
    amended = ("42", "omission", "13F-HR/A", "2024-08-20", "2024-06-30", [("037833100", "APPLE", 500, 5, "")])
    ctx = _ctx(tmp_path, filing_store, {"2023q2": [earlier], "01jun2024-31aug2024": [first, amended]})

    z13.ingest_13f_datasets(ctx, full=False)

    assert set(_positions(filing_store, cik="0000000042").accession_number) == {"baseline", "holding", "omission"}
    print("SANITY: the ingest preserves an earlier baseline and the later amendment that omits a holding.")


def test_zip_and_xml_equivalent_packet_replays_once(tmp_path: Any, filing_store: Any) -> None:
    path = tmp_path / "replay.zip"
    write_filings_zip(path, [_A1])
    packet = cached_filing_vintages(path, [_A1[1]])[0]
    save_hr_filing(filing_store, *packet, None)
    cover = (
        "<edgarSubmission><isAmendment>false</isAmendment><amendmentNo>0</amendmentNo>"
        "<reportCalendarOrQuarter>2023-03-31</reportCalendarOrQuarter>"
        "<reportType>13F HOLDINGS REPORT</reportType><isConfidentialOmitted>false</isConfidentialOmitted>"
        "<tableEntryTotal>3</tableEntryTotal><tableValueTotal>2250</tableValueTotal></edgarSubmission>"
    )
    xml = _xml(_A1[5]).replace("<titleOfClass>COM</titleOfClass>", "<titleOfClass>  COM  </titleOfClass>")
    parsed = f13.xml_filing_vintage(cover, xml, cik=_A1[0], accession_number=_A1[1], period=_A1[4], filing_date=_A1[3], source="edgar")
    assert save_hr_filing(filing_store, *parsed, None) == 0
    assert len(filing_store.load(Tables.sec13f_hr)) == 2
    print("SANITY: ZIP and XML parsers produce equivalent accession content despite source whitespace/transport differences.")


def test_original_xml_can_omit_false_amendment_flag() -> None:
    cover = (
        "<edgarSubmission><submissionType>13F-HR</submissionType>"
        "<reportCalendarOrQuarter>2023-03-31</reportCalendarOrQuarter>"
        "<reportType>13F HOLDINGS REPORT</reportType><isConfidentialOmitted>false</isConfidentialOmitted>"
        "<tableEntryTotal>3</tableEntryTotal><tableValueTotal>2250</tableValueTotal></edgarSubmission>"
    )
    kwargs: dict[str, Any] = dict(cik=_A1[0], accession_number=_A1[1], period=_A1[4], filing_date=_A1[3], source="edgar")
    manifest, rows = f13.xml_filing_vintage(cover, _xml(_A1[5]), expected_form="13F-HR", **kwargs)
    assert manifest.iloc[0].amendment_type == "original" and len(rows) == 2
    with pytest.raises(ValueError, match="disagrees"):
        f13.xml_filing_vintage(cover, _xml(_A1[5]), expected_form="13F-HR/A", **kwargs)
    with pytest.raises(ValueError, match="missing or invalid"):
        f13.xml_filing_vintage(cover.replace("<submissionType>13F-HR</submissionType>", ""), _xml(_A1[5]), **kwargs)
    for form, flag in (("13F-HR/A", "false"), ("13F-HR", "true")):
        submission = f"<submissionType>{form}</submissionType><isAmendment>{flag}</isAmendment>"
        conflicting = cover.replace("<submissionType>13F-HR</submissionType>", submission)
        with pytest.raises(ValueError, match="disagrees with cover form"):
            f13.xml_filing_vintage(conflicting, _xml(_A1[5]), **kwargs)
    print("SANITY: a missing false amendment checkbox is accepted only for an explicitly original SEC cover.")


def test_ingest_reuses_one_map_load_across_windows(tmp_path: Any, filing_store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _ctx(tmp_path, filing_store)
    original = filing_store.load
    map_loads: list[Any] = []

    def counted(table: Any, *args: Any, **kwargs: Any) -> Any:
        if table == Tables.cusip_ticker_map:
            map_loads.append(table)
        return original(table, *args, **kwargs)

    monkeypatch.setattr(filing_store, "load", counted)
    z13.ingest_13f_datasets(ctx, full=False)
    assert len(map_loads) == 1
    assert {_A2[1], _B1[1]} <= set(filing_store.load(Tables.sec13f_hr).accession_number)
    print("SANITY: the two-window ingest loads the CUSIP map once and writes every book to sec13f_hr.")
