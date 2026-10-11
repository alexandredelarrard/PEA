"""Known-truth unit declarations; no market prices or library normalization oracle."""

from __future__ import annotations

from types import SimpleNamespace

import edgar.thirteenf.parsers.infotable_xml as xml_parser
import pandas as pd
import pyarrow as pa
import pytest

from src.data_extract.utils.common.edgar_driver import FilingStamp
from src.data_extract.utils.institutionals import fetch_13f as f13
from src.data_extract.utils.institutionals.fetch_13f import _book_frame
from src.data_extract.utils.institutionals.fetch_13f_backfill import data_set_book
from src.data_extract.utils.institutionals.legacy_13f_fallback import parse_legacy_information_table
from src.data_extract.utils.institutionals.value_units import normalize_information_values, resolve_value_units


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
