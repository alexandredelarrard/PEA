"""Known-truth unit declarations, and the per-filing evidence (implied price first) that decides the cell unit."""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import edgar.thirteenf.parsers.infotable_xml as xml_parser
import pandas as pd
import pyarrow as pa
import pytest

from src.constants.constants import SEC_13F_SOURCE_ACCOUNTED
from src.data_extract.utils.common.edgar_driver import FilingStamp
from src.data_extract.utils.institutionals import fetch_13f as f13
from src.data_extract.utils.institutionals.fetch_13f import _book_frame
from src.data_extract.utils.institutionals.fetch_13f_backfill import data_set_book
from src.data_extract.utils.institutionals.legacy_13f_fallback import parse_legacy_information_table
from src.data_extract.utils.institutionals.value_units import (
    FEW_ROWS_PRICE,
    NO_EVIDENCE,
    caption_factor,
    normalize_information_values,
    resolve_value_units,
)


@pytest.mark.parametrize(
    ("filed", "schema", "caption", "factor"),
    [
        ("2022-11-14", "X0102", None, 1000),
        ("2023-01-03", "X0202", None, 1),
        ("2025-05-14", None, None, 1),
        ("1995-05-15", None, "<TABLE>NAME CUSIP VALUE (x$1000) SHARES\n</TABLE>", 1000),
        (None, None, "<TABLE>NAME CUSIP VALUE (x$10000) SHARES\n</TABLE>", 10000),
    ],
)
def test_source_declarations_select_units(filed: str | None, schema: str | None, caption: str | None, factor: float) -> None:
    units = resolve_value_units(filed, schema_version=schema, information_table_text=caption)
    assert units.factor == factor
    raw = pd.DataFrame({"VALUE": [145462], "SSHPRNAMT": [3362515]})
    normalized = normalize_information_values(raw, units)
    assert normalized.VALUE.iloc[0] == 145462 * factor
    assert normalized.raw_value.iloc[0] == 145462
    assert normalized.SSHPRNAMT.iloc[0] == 3362515
    assert raw.VALUE.iloc[0] == 145462
    print(f"\nSANITY: source factor {factor:g} applied once; raw value and shares retained.")


@pytest.mark.parametrize(
    ("filed", "schema", "text"),
    [
        ("2025-05-14", "X0202", "<TABLE>VALUE (x$1000) CUSIP SHARES\n</TABLE>"),
        ("2022-05-15", "X0102", "<TABLE>VALUE (in dollars) CUSIP SHARES\n</TABLE>"),
        (None, None, None),
        ("2025-05-14", "X9999", None),
    ],
)
def test_conflicting_or_unknown_units_withhold_dollars_and_keep_shares(filed: str | None, schema: str | None, text: str | None) -> None:
    units = resolve_value_units(filed, schema_version=schema, information_table_text=text)
    assert units.factor is None
    frame = normalize_information_values(pd.DataFrame({"Value": [123], "SharesPrnAmount": [456]}), units)
    assert pd.isna(frame.Value.iloc[0]) and frame.raw_value.iloc[0] == 123
    assert frame.SharesPrnAmount.iloc[0] == 456
    print("\nSANITY: unknown/conflicting units preserve raw values and shares while withholding USD.")


def test_cover_billion_does_not_set_information_table_units() -> None:
    text = "<TABLE>Form 13F Information Table Value Total: $45.9 Billion</TABLE>\n<TABLE>NAME CUSIP VALUE (x$1000) SHARES\n</TABLE>"
    units = resolve_value_units("2012-02-14", information_table_text=text)
    assert units.factor == 1000 and units.stated_factor == 1000
    print("\nSANITY: Billion cover total cannot turn the holding VALUE column into billions.")


def test_information_table_captions_are_checked_on_every_page() -> None:
    text = "<TABLE>VALUE (x$1000) CUSIP SHARES\n</TABLE><TABLE>VALUE (in dollars) CUSIP SHARES\n</TABLE>"
    assert resolve_value_units("2022-02-14", information_table_text=text).factor is None
    print("\nSANITY: conflicting table captions withhold dollars instead of silently selecting one.")


def test_normalized_values_cannot_be_scaled_a_second_time() -> None:
    units = resolve_value_units("2012-02-14")
    once = normalize_information_values(pd.DataFrame({"VALUE": [123]}), units)
    with pytest.raises(ValueError, match="already normalized"):
        normalize_information_values(once, units)
    print("\nSANITY: attempting a second normalization fails before multiplying values again.")


def test_post_cutover_amendment_uses_filing_date_in_zip_preview() -> None:
    sub = pd.DataFrame(
        [
            dict(
                accession="0000000001-25-000001",
                cik="0000000001",
                period=pd.Timestamp("2020-12-31"),
                report_period=pd.Timestamp("2020-12-31"),
                filing_date=pd.Timestamp("2025-05-14"),
                is_amendment=True,
            )
        ]
    )
    raw = pd.DataFrame([dict(ACCESSION_NUMBER=sub.accession.iloc[0], CUSIP="037833100", VALUE=2100, SSHPRNAMT=10, SSHPRNAMTTYPE="SH", PUTCALL="")])
    book = data_set_book(sub, raw, source_values=True)
    assert book.value_usd.iloc[0] == 2100 and book.shares.iloc[0] == 10
    assert book.accession.iloc[0] == sub.accession.iloc[0]
    print("\nSANITY: old-quarter amendment filed in 2025 retains $2,100 and 10 shares in the accession-preserving preview.")


def test_legacy_source_parser_accepts_explicit_dollar_factor() -> None:
    raw = "Form 13F Information Table Entry Total: 1\nForm 13F Information Table Value Total: $145462 (dollars)\n"
    raw += (
        "<TABLE>\nNAME OF ISSUER CLASS CUSIP VALUE (in dollars) SHARES\n<S> <C> <C> <C> <C>\nAflac Inc  COM 001055102 145462 3362 SH SOLE\n</TABLE>"
    )
    parsed = parse_legacy_information_table(raw, value_factor=1)
    assert parsed.VALUE.iloc[0] == 145462 and parsed.SSHPRNAMT.iloc[0] == 3362
    print("\nSANITY: a dollar-declared text holding is not multiplied by the legacy default.")


def test_every_typed_leg_preserves_unknown_dollars() -> None:
    units = resolve_value_units(None)
    raw = pd.DataFrame(
        [
            dict(CUSIP="037833100", VALUE=123, SSHPRNAMT=456, SSHPRNAMTTYPE=kind, PUTCALL=option)
            for kind, option in [("SH", ""), ("SH", "CALL"), ("SH", "PUT"), ("PRN", ""), ("OTHER", "")]
        ]
    )
    book = _book_frame("1", "2025-05-14", "2025-03-31", normalize_information_values(raw, units))
    assert book.shares.iloc[0] == book.call_shares.iloc[0] == book.put_shares.iloc[0] == book.debt_prn.iloc[0] == 456
    assert book[["value_usd", "call_value", "put_value", "debt_value", "other_value"]].isna().all().all()
    print("\nSANITY: all five unknown value legs remain missing after mixed-CUSIP grouping; valid amounts survive.")


@pytest.mark.parametrize(("schema", "factor"), [("X0102", 1000), ("X0202", 1)])
def test_xml_preview_uses_raw_parser_and_never_heuristic_values(monkeypatch: pytest.MonkeyPatch, schema: str, factor: float) -> None:
    monkeypatch.setattr(xml_parser, "cusip_ticker_mapping", lambda **_: SimpleNamespace(Ticker=pd.Series({"037833100": "AAPL"})))
    xml = "<informationTable><infoTable><nameOfIssuer>APPLE INC</nameOfIssuer><titleOfClass>COM</titleOfClass>"
    xml += "<cusip>037833100</cusip><value>2100</value><shrsOrPrnAmt><sshPrnamt>10</sshPrnamt><sshPrnamtType>SH</sshPrnamtType>"
    xml += "</shrsOrPrnAmt><investmentDiscretion>SOLE</investmentDiscretion></infoTable></informationTable>"

    class Report:
        _schema_version = schema
        infotable_xml = xml
        infotable_txt = None

        @property
        def infotable(self) -> pd.DataFrame:
            raise AssertionError("heuristic-normalized infotable must never be read")

    report = Report()
    filing = SimpleNamespace(
        cik="1",
        filing_date="2022-05-14" if schema == "X0102" else "2025-05-14",
        period_of_report="2020-12-31",
        form="13F-HR/A",
        accession_number="0000000001-25-000001",
        obj=lambda: report,
    )
    book = f13._read_filing(FilingStamp.of(filing, "1"), source_values=True)
    assert isinstance(book, pd.DataFrame), book
    assert book.value_usd.iloc[0] == 2100 * factor and book.shares.iloc[0] == 10
    assert book.attrs["value_units"].factor == factor
    print(f"\nSANITY: raw XML oracle $2100 x {factor:g}, 10 shares; heuristic output bypassed.")


def test_bad_xml_is_not_accepted_as_a_partial_book(monkeypatch: pytest.MonkeyPatch) -> None:
    report = SimpleNamespace(_schema_version="X0202", infotable_xml="<informationTable><infoTable>", infotable_txt=None)
    filing = SimpleNamespace(
        cik="1", filing_date="2025-05-14", period_of_report="2025-03-31", form="13F-HR", accession_number="0000000001-25-000001", obj=lambda: report
    )
    result = f13._read_filing(FilingStamp.of(filing, "1"), source_values=True)
    assert isinstance(result, f13.ReadFailure) and not result.transient
    print("\nSANITY: malformed XML cannot recover into a complete empty/partial source book.")


@pytest.mark.parametrize(("flag", "restricted"), [(None, False), ("", False), ("false", False), ("0", False), ("true", True), ("1", True)])
def test_optional_xml_confidentiality_requires_affirmative_evidence(flag: str | None, restricted: bool) -> None:
    checkbox = "" if flag is None else f"<isConfidentialOmitted>{flag}</isConfidentialOmitted>"
    primary = (
        "<edgarSubmission><schemaVersion>X0202</schemaVersion><submissionType>13F-HR</submissionType>"
        "<reportCalendarOrQuarter>2024-06-30</reportCalendarOrQuarter><reportType>13F HOLDINGS REPORT</reportType>"
        f"<tableEntryTotal>1</tableEntryTotal><tableValueTotal>2000</tableValueTotal>{checkbox}</edgarSubmission>"
    )
    info = (
        "<informationTable><infoTable><nameOfIssuer>AFLAC INC</nameOfIssuer><titleOfClass>COM</titleOfClass>"
        "<cusip>001055102</cusip><value>2000</value><shrsOrPrnAmt><sshPrnamt>100</sshPrnamt>"
        "<sshPrnamtType>SH</sshPrnamtType></shrsOrPrnAmt></infoTable></informationTable>"
    )
    manifest, holdings = f13.xml_filing_vintage(
        primary,
        info,
        cik="1",
        accession_number="optional-confidentiality",
        period="2024-06-30",
        filing_date="2024-08-14",
        source="known_truth",
    )
    assert bool(manifest.restricted.iloc[0]) is restricted
    assert holdings.shares.iloc[0] == 100 and holdings.value_usd.iloc[0] == 2000
    print(f"SANITY: optional XML flag {flag!r} -> restricted={restricted}; reported holdings preserved.")


def test_thousands_of_dollars_is_one_caption() -> None:
    raw = "<TABLE>NAME CUSIP VALUE (in thousands of dollars) SHARES\n</TABLE>"
    assert resolve_value_units("2012-02-14", information_table_text=raw).factor == 1000
    print("\nSANITY: thousands-of-dollars is one unit; the dollars noun is not a contradiction.")


@pytest.mark.parametrize("caption", ["Value Total: 145.462 (in thousands)", "Value Total x1000: 145.462"])
def test_cover_thousands_does_not_rescale_dollar_holdings(caption: str) -> None:
    from_text = "Form 13F Information Table Entry Total: 1\nForm 13F Information Table " + caption + "\n"
    from_text += (
        "<TABLE>\nNAME OF ISSUER CLASS CUSIP VALUE (in dollars) SHARES\n<S> <C> <C> <C> <C>\nAflac Inc  COM 001055102 145462 3362 SH SOLE\n</TABLE>"
    )
    assert parse_legacy_information_table(from_text, value_factor=1).VALUE.iloc[0] == 145462
    print("\nSANITY: independent cover display in thousands checks correctly against dollar-valued holdings.")


@pytest.mark.parametrize("source", ["xml", "text"])
@pytest.mark.parametrize("clock", ["utc", "offset", "library_naive", "missing", "nat"])
def test_ordinary_reader_retains_acceptance_clock(
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    clock: str,
) -> None:
    """CompanyFiling's Arrow-stripped timestamp is UTC; unavailable clocks remain dates."""
    accepted = {
        "utc": pd.Timestamp("2025-05-14T13:30:00Z"),
        "offset": pd.Timestamp("2025-05-14T09:30:00-04:00"),
        "library_naive": pa.array([pd.Timestamp("2025-05-14T13:30:00Z")], type=pa.timestamp("us"))[0].as_py(),
        "missing": None,
        "nat": pd.NaT,
    }[clock]
    primary = (
        "<edgarSubmission><schemaVersion>X0202</schemaVersion><submissionType>13F-HR</submissionType>"
        "<reportCalendarOrQuarter>2025-03-31</reportCalendarOrQuarter><isAmendment>false</isAmendment>"
        "<reportType>13F HOLDINGS REPORT</reportType><isConfidentialOmitted>false</isConfidentialOmitted>"
        "<tableEntryTotal>1</tableEntryTotal><tableValueTotal>145462</tableValueTotal></edgarSubmission>"
    )
    information = (
        "<informationTable><infoTable><nameOfIssuer>AFLAC INC</nameOfIssuer><titleOfClass>COM</titleOfClass>"
        "<cusip>001055102</cusip><value>145462</value><shrsOrPrnAmt><sshPrnamt>3362</sshPrnamt>"
        "<sshPrnamtType>SH</sshPrnamtType></shrsOrPrnAmt><investmentDiscretion>SOLE</investmentDiscretion>"
        "</infoTable></informationTable>"
    )
    raw = (
        "[X] 13F HOLDINGS REPORT\nForm 13F Information Table Entry Total: 1\n"
        "Form 13F Information Table Value Total: $145462 (dollars)\n"
        "<TABLE>\nNAME OF ISSUER CLASS CUSIP VALUE (in dollars) SHARES\n<S> <C> <C> <C> <C>\n"
        "Aflac Inc  COM 001055102 145462 3362 SH SOLE\n</TABLE>"
    )
    report = SimpleNamespace(infotable_xml=information if source == "xml" else None, infotable_txt=raw)
    filing = SimpleNamespace(
        cik="1",
        filing_date="2025-05-14",
        period_of_report="2025-03-31",
        form="13F-HR",
        accession_number="0000000001-25-000001",
        xml=lambda: primary,
        full_text_submission=lambda: raw,
    )
    if clock != "missing":
        filing.acceptance_datetime = accepted
    monkeypatch.setattr(f13, "filing_obj", lambda _: report)
    monkeypatch.setattr(f13, "sec_call", lambda fn, *args, **kwargs: fn(*args))
    result = f13.read_filing_vintage(FilingStamp.of(filing, "1"))
    assert isinstance(result, tuple), result
    manifest, book = result
    marker = manifest.iloc[0]
    assert book.shares.iloc[0] == 3362 and book.value_usd.iloc[0] == 145462
    if clock in {"missing", "nat"}:
        assert pd.isna(marker.accepted_at) and marker.clock_precision == "date"
        assert marker.clock_source == "SEC filing date" and marker.clock_timezone == "America/New_York"
    else:
        assert marker.accepted_at == pd.Timestamp("2025-05-14T13:30:00")
        assert marker.clock_precision == "timestamp" and marker.clock_source == "SEC acceptance"
        assert marker.clock_timezone == "UTC"
    print(f"\nSANITY: ordinary {source} reader preserves {clock} acceptance evidence without changing shares/dollars.")


# ---- per-filing unit evidence: implied price, stated unit, cover ratio, x1 fallback ------------------

_LEGACY_SOURCES = Path(__file__).parent / "fixtures" / "legacy_13f"
_EVIDENCE_LOGGER = "src.data_extract.utils.institutionals"


def _legacy(accession: str) -> tuple[str, FilingStamp]:
    """A real SEC legacy submission and the stamp its own header declares."""
    raw = (_LEGACY_SOURCES / f"{accession}.sgml").read_text(encoding="utf-8")

    def header(label: str) -> str:
        match = re.search(rf"{label}:\s*(\S+)", raw)
        assert match, label
        return match.group(1)

    filing = SimpleNamespace(
        cik=header("CENTRAL INDEX KEY"),
        filing_date=header("FILED AS OF DATE"),
        period_of_report=header("CONFORMED PERIOD OF REPORT"),
        form="13F-HR",
        accession_number=accession,
    )
    return raw, FilingStamp.of(filing, filing.cik)


def _messages(caplog: pytest.LogCaptureFixture) -> str:
    return "\n".join(record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING)


@pytest.mark.parametrize(
    ("accession", "factor", "cusip", "band", "shape"),
    [
        ("0000315498-99-000011", 1.0, "92047K107", (15, 25), "ValueVision; (x$1000) caption over dollar cells, cover truncated to 1,976,772,00"),
        ("0001082327-13-000006", 1.0, "204166102", (70, 95), "Commvault; dollar cells, cover in dollars (ratio 1); CVLT near $82 on 2013-03-28"),
        ("0001193125-05-224912", 1.0, "30231G102", (55, 70), "Exxon; dollar cells, no cover; XOM near $63.5 on 2005-09-30"),
        ("0001356407-13-000002", 1.0, "058498106", (40, 55), "Ball; dollar cells, no cover; BLL near $47.6 on 2013-03-28"),
        ("0000812295-05-000069", 1000.0, "91324P102", (50, 62), "UnitedHealth; (X$1,000PRN caption, $1000s cells; UNH near $56 on 2005-09-30"),
        ("0000903944-05-000011", 1000.0, "370442717", (20, 30), "GM $25-par preferred; $1000s cells, no cover"),
        ("0001387131-13-001589", 1000.0, "78462F103", (140, 170), "SPY; $1000s cells, cover/sum ratio 16.36; SPY near $156.7 on 2013-03-28"),
    ],
)
def test_implied_price_decides_the_cell_unit_on_real_sources(
    caplog: pytest.LogCaptureFixture, accession: str, factor: float, cusip: str, band: tuple[float, float], shape: str
) -> None:
    raw, stamp = _legacy(accession)
    with caplog.at_level(logging.WARNING, logger=_EVIDENCE_LOGGER):
        manifest, rows = f13.text_filing_vintage(raw, stamp)
    row = manifest.iloc[0]
    assert row.value_unit_factor == factor and row.value_unit_evidence == SEC_13F_SOURCE_ACCOUNTED + "implied_price", row.value_unit_evidence
    assert (rows.value_usd == rows.raw_value_usd * factor).all()
    holding = rows.set_index("cusip").loc[cusip]
    price = holding.value_usd / holding.shares
    assert band[0] < price < band[1], price
    common = rows[(rows.position_type == "common") & (rows.shares > 0)]
    median = float((common.value_usd / common.shares).median())
    assert 1 <= median <= 1000, median
    print(f"SANITY: real {accession} -> x{factor:g} by implied price (median ${median:.2f}/sh); {cusip} ${price:.2f}/sh ({shape}).")


def test_truncated_cover_and_wrong_caption_warn_and_keep_the_filing(caplog: pytest.LogCaptureFixture) -> None:
    raw, stamp = _legacy("0000315498-99-000011")
    with caplog.at_level(logging.WARNING, logger=_EVIDENCE_LOGGER):
        manifest, rows = f13.text_filing_vintage(raw, stamp)
    row, messages = manifest.iloc[0], _messages(caplog)
    assert row.declared_value_total == 197_677_200 and row.source_value_total == 1_976_772_000
    assert row.parse_status == "complete" and len(rows) == row.parsed_row_count and row.stated_unit_factor == 1000
    assert "stated_caption says x1000" in messages and "picks x1" in messages, messages
    value_vision = rows.set_index("cusip").loc["92047K107"]
    assert value_vision.raw_value_usd == 2_810_000 and value_vision.value_usd == 2_810_000 and value_vision.shares == 141_400
    print(
        "SANITY: real 0000315498-99-000011 keeps its truncated cover 197,677,200 as filed beside the 1,976,772,000 cell sum; "
        f"the (x$1000) caption is overruled with a WARNING; ValueVision ${value_vision.value_usd / value_vision.shares:.2f}/sh, not $19,873."
    )


def _two_row_text(caption: str, cover: str) -> str:
    """SYNTHETIC: Aflac and Amgen, too few share rows for the implied price to decide."""
    header = f"ISSUER                 TYPE           CUSIP     {caption} SHS      INVEST"
    return "\n".join(
        [
            "[X] 13F HOLDINGS REPORT",
            "Form 13F Information Table Entry Total: 2",
            f"Form 13F Information Table Value Total: {cover}",
            "<TABLE>",
            header,
            "<S>                            <C>            <C>       <C>      <C> <C>",
            "Aflac, Inc.             Com         001055102 145,462  3,362,515   DEFINED      1   SOLE",
            "Amgen Inc.              Com         031162100  12,000    100,000   DEFINED      1   SOLE",
            "</TABLE>",
        ]
    )


@pytest.mark.parametrize(
    ("caption", "cover", "factor", "evidence"),
    [
        ("VALUE(x$1000)", "$157,462", 1000.0, "stated_caption"),
        ("VALUE(dollars)", "$157,462", 1.0, "stated_caption"),
        ("VALUE   ", "$157,462 (in thousands)", 1000.0, "stated_cover"),
        ("VALUE   ", "$157,462,000", 1000.0, "cover_total_ratio"),
        ("VALUE   ", "$157", 1.0, "cover_total_ratio"),
        ("VALUE   ", "$157,462", 1000.0, FEW_ROWS_PRICE),
        ("VALUE(x$1000)", "$157 (thousands)", 1000.0, FEW_ROWS_PRICE),
    ],
)
def test_too_few_share_rows_fall_back_to_stated_unit_then_cover_ratio_then_x1(
    caplog: pytest.LogCaptureFixture, caption: str, cover: str, factor: float, evidence: str
) -> None:
    raw = _two_row_text(caption, cover)
    filing = SimpleNamespace(cik="1", filing_date="2010-05-13", period_of_report="2010-03-31", form="13F-HR", accession_number="few")
    with caplog.at_level(logging.WARNING, logger=_EVIDENCE_LOGGER):
        manifest, rows = f13.text_filing_vintage(raw, FilingStamp.of(filing, "1"))
    row = manifest.iloc[0]
    assert row.value_unit_factor == factor and row.value_unit_evidence == SEC_13F_SOURCE_ACCOUNTED + evidence, row.value_unit_evidence
    assert sorted(rows.raw_value_usd) == [12000.0, 145462.0] and (rows.value_usd == rows.raw_value_usd * factor).all()
    if evidence == FEW_ROWS_PRICE:
        assert "from the implied price over only 2 share row(s)" in _messages(caplog)
    print(f"SANITY: SYNTHETIC 2-row book, caption {caption.strip()!r}, cover {cover!r} -> x{factor:g} ({evidence}); never the 2010 date default.")


def test_one_row_real_source_uses_its_stated_caption() -> None:
    raw, stamp = _legacy("0000902664-13-002273")
    manifest, rows = f13.text_filing_vintage(raw, stamp)
    row = manifest.iloc[0]
    assert row.value_unit_factor == 1000 and row.value_unit_evidence == SEC_13F_SOURCE_ACCOUNTED + "stated_caption"
    price = rows.value_usd.iloc[0] / rows.shares.iloc[0]
    assert 6 < price < 11, price
    print(
        f"SANITY: real 0000902664-13-002273 (1 share row) -> x1000 from its (x$1000) caption; Harbinger ${price:.2f}/sh (HRG near $8.3 in 2013-03)."
    )


def test_caption_multiplier_reads_its_whole_digit_group() -> None:
    assert caption_factor("<TABLE>\nCUSIP      VALUE   SHARES OR\n     (X$1,000PRN AMOUN\n</TABLE>") == 1000
    assert caption_factor("<TABLE>\nNAME CUSIP VALUE (x$1000) SHARES\n</TABLE>") == 1000
    assert caption_factor("<TABLE>\nNAME CUSIP VALUE ($1,000) SHARES\n</TABLE>") == 1000
    assert math.isnan(caption_factor("<TABLE>VALUE (x$1000) CUSIP\n</TABLE><TABLE>VALUE (in dollars) CUSIP\n</TABLE>"))
    print("SANITY: the (X$1,000PRN AMOUN caption reads x1000, not x1; two different captions are a conflict (NaN).")


def _xml(schema: str | None, cells: list[tuple[str, int, int]], cover_total: float) -> tuple[str, str]:
    """SYNTHETIC cover and information-table XML; each cell is (cusip, VALUE, shares)."""
    version = f"<schemaVersion>{schema}</schemaVersion>" if schema else ""
    primary = (
        f"<edgarSubmission>{version}<submissionType>13F-HR</submissionType>"
        "<reportCalendarOrQuarter>2016-03-31</reportCalendarOrQuarter><reportType>13F HOLDINGS REPORT</reportType>"
        f"<tableEntryTotal>{len(cells)}</tableEntryTotal><tableValueTotal>{cover_total:g}</tableValueTotal></edgarSubmission>"
    )
    rows = "".join(
        f"<infoTable><nameOfIssuer>ISSUER {cusip}</nameOfIssuer><titleOfClass>COM</titleOfClass><cusip>{cusip}</cusip><value>{value}</value>"
        f"<shrsOrPrnAmt><sshPrnamt>{shares}</sshPrnamt><sshPrnamtType>SH</sshPrnamtType></shrsOrPrnAmt></infoTable>"
        for cusip, value, shares in cells
    )
    return primary, f"<informationTable>{rows}</informationTable>"


_CUSIPS = ["037833100", "594918104", "001055102", "031162100", "88579Y101", "30231G102"]


@pytest.mark.parametrize(
    ("schema", "cells_per_share", "rows", "factor", "evidence", "shape"),
    [
        ("X0102", 50.0, 6, 1.0, "implied_price", "2016 schema prescribes $1000s but the cells are dollars"),
        ("X0102", 0.05, 6, 1000.0, "implied_price", "2016 schema and $1000s cells agree"),
        ("X0202", 50.0, 6, 1.0, "implied_price", "dollar schema and dollar cells agree"),
        ("X0202", 0.05, 2, 1.0, "stated_schema", "2 rows: the dollar schema decides"),
        ("X0102", 50.0, 2, 1000.0, "stated_schema", "2 rows: the $1000s schema decides"),
        (None, 0.05, 2, 1000.0, FEW_ROWS_PRICE, "no schema, 2 rows, ratio 1: the few-rows implied price decides"),
        (None, 5000.0, 2, 1.0, NO_EVIDENCE, "no schema, ratio 1, $5000/sh fits no unit: x1 with a WARNING"),
    ],
)
def test_xml_unit_comes_from_evidence(
    caplog: pytest.LogCaptureFixture, schema: str | None, cells_per_share: float, rows: int, factor: float, evidence: str, shape: str
) -> None:
    cells = [(cusip, int(1000 * cells_per_share * (index + 1)), 1000 * (index + 1)) for index, cusip in enumerate(_CUSIPS[:rows])]
    primary, info = _xml(schema, cells, sum(value for _, value, _ in cells))
    with caplog.at_level(logging.WARNING, logger=_EVIDENCE_LOGGER):
        manifest, book = f13.xml_filing_vintage(
            primary, info, cik="1", accession_number="xml-units", period="2016-03-31", filing_date="2016-05-13", source="known_truth"
        )
    row = manifest.iloc[0]
    assert row.value_unit_factor == factor and row.value_unit_evidence == evidence, (row.value_unit_factor, row.value_unit_evidence)
    assert (book.value_usd == book.raw_value_usd * factor).all()
    if schema == "X0102" and factor == 1.0 and evidence == "implied_price":
        assert "stated_schema says x1000" in _messages(caplog)
    if evidence == NO_EVIDENCE:
        assert "cells stored as dollars (x1)" in _messages(caplog)
    print(f"SANITY: SYNTHETIC XML {schema} {rows} rows at {cells_per_share:g} cell/share -> x{factor:g} ({evidence}): {shape}.")


def test_value_total_mismatch_is_saved_with_a_warning(sqlite_store: Any, caplog: pytest.LogCaptureFixture) -> None:
    cells = [(cusip, 50_000 * (index + 1), 1000 * (index + 1)) for index, cusip in enumerate(_CUSIPS)]
    primary, info = _xml("X0202", cells, 777.0)
    with caplog.at_level(logging.WARNING, logger=_EVIDENCE_LOGGER):
        manifest, book = f13.xml_filing_vintage(
            primary, info, cik="1", accession_number="bad-cover", period="2016-03-31", filing_date="2024-05-13", source="known_truth"
        )
    assert "the cover is a reference only, filing kept" in _messages(caplog)
    assert manifest.iloc[0].declared_value_total == 777.0 and manifest.iloc[0].source_value_total == 1_050_000
    assert sqlite_store.save_13f_filing(manifest, book) == len(cells)
    print("SANITY: XML cover total 777 against a 1,050,000 VALUE column -> WARNING, filing kept and saved, cover stored as filed.")
