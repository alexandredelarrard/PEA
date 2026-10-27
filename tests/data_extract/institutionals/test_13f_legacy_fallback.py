"""
test_13f_legacy_fallback.py (tests/data_extract/institutionals/test_13f_legacy_fallback.py)
--------------------------------------------------------------------------------------------
Known-truth SEC text lines for the source-checked legacy 13F parser, and the `_read_filing` hook:
clean XML is untouched, a malformed text parse is replaced by verified source rows, an
unverifiable text book is a deterministic read failure, and no F-001 garbage shape is returned.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pandas as pd
import pytest
from edgar import Filing

from src.data_extract.utils.common.edgar_driver import FilingStamp
from src.data_extract.utils.institutionals import fetch_13f as f13
from src.data_extract.utils.institutionals.fetch_13f import save_13f_filing
from src.data_extract.utils.institutionals.legacy_13f_fallback import (
    _blank_cusip_value,
    _holding_rows,
    _source_rows,
    needs_legacy_fallback,
    parse_legacy_information_table,
)
from src.data_extract.utils.institutionals.value_units import FEW_ROWS_PRICE, NO_EVIDENCE
from src.utils.institutional_vintages import public_books

_HEADER = "ISSUER                 TYPE           CUSIP     VALUE     SHS      INVEST"
_AFLAC = "Aflac, Inc.             Com         001055102 145,462  3,362,515   DEFINED      1   SOLE"
_AMGEN = "Amgen Inc.              Com         031162100  12,000    100,000   DEFINED      1   SOLE"


def _table(header: str, *rows: str) -> str:
    return "\n".join(
        [
            f"Form 13F Information Table Entry Total: {len(rows)}",
            "<TABLE>",
            header,
            "<S>                            <C>            <C>       <C>      <C> <C>",
            *rows,
            "</TABLE>",
        ]
    )


def _edgar_line(cusip: str, issuer: str, value: float, shares: float) -> dict[str, Any]:
    """One row in EdgarTools' text-parse column names (value already in dollars)."""
    return {"Issuer": issuer, "Class": "COM", "Cusip": cusip, "Value": value, "SharesPrnAmount": shares, "Type": "Shares", "PutCall": ""}


class _Report:
    """The `ThirteenF` surface `_read_filing` reads; counts every `infotable_txt` access. `download_error`
    is raised by the information-table download (the `infotable_xml` access)."""

    def __init__(
        self,
        infotable: pd.DataFrame | None,
        txt: str | None = None,
        xml: str | None = None,
        error: Exception | None = None,
        download_error: Exception | None = None,
    ) -> None:
        self._infotable, self._txt, self._xml, self._error, self._download_error = infotable, txt, xml, error, download_error
        self.txt_reads = 0

    @property
    def infotable(self) -> pd.DataFrame | None:
        if self._error is not None:
            raise self._error
        return self._infotable

    @property
    def infotable_xml(self) -> str | None:
        if self._download_error is not None:
            raise self._download_error
        return self._xml

    @property
    def infotable_txt(self) -> str | None:
        self.txt_reads += 1
        return self._txt


@dataclass
class _Filing:
    report: _Report
    cik: str = "0001036325"
    filing_date: str = "2010-05-13"
    period_of_report: str = "2010-03-31"
    form: str = "13F-HR"
    accession_number: str = "0001036325-10-000001"

    def obj(self) -> _Report:
        return self.report


def _read(report: _Report) -> pd.DataFrame | f13.ReadFailure:
    filing = _Filing(report)
    return f13._read_filing(FilingStamp.of(filing, filing.cik))


def test_legacy_edgar_preserves_accession_and_reconciled_book(filing_store: Any) -> None:
    raw = "[X] 13F HOLDINGS REPORT\nForm 13F Information Table Value Total: $157,462\n" + _table(_HEADER, _AFLAC, _AMGEN)
    filing = _Filing(_Report(None, txt=raw))
    packet = f13.read_filing_vintage(FilingStamp.of(filing, filing.cik))
    assert not isinstance(packet, f13.ReadFailure), packet
    manifest, holdings = packet
    assert manifest.iloc[0].accession_number == filing.accession_number
    # Two share rows, no caption and a ratio-1 cover: only the few-rows implied price ($43 and $120 at x1000) decides.
    assert len(holdings) == 2 and holdings.value_usd.sum() == 157462000
    assert manifest.iloc[0].value_unit_evidence.endswith(FEW_ROWS_PRICE)
    save_13f_filing(filing_store, manifest, holdings)
    assert len(filing_store.load(f13.Tables.sec13f_manager_holdings)) == 2
    amendment = _Filing(_Report(None, txt=raw), form="13F-HR/A", accession_number="missing-type")
    failed = f13.read_filing_vintage(FilingStamp.of(amendment, amendment.cik))
    assert isinstance(failed, f13.ReadFailure) and "amendment type" in failed.reason
    amendment.report._txt = "[X] is a restatement\n" + raw
    parsed = f13.read_filing_vintage(FilingStamp.of(amendment, amendment.cik))
    assert not isinstance(parsed, f13.ReadFailure) and parsed[0].iloc[0].amendment_type == "restatement"
    print("SANITY: legacy EDGAR source totals, raw values, accession/date survive; unsupported amendment type cannot complete a book.")


def _forbidden(name: str) -> Any:
    """A stand-in that fails the test if `name` is ever called."""

    def _raise(*_: Any) -> Any:
        raise AssertionError(f"{name} called")

    return _raise


def _f001_garbage(book: pd.DataFrame) -> int:
    """F-001's garbage definition on a stored-shape book."""
    zero_sh = (book["position_type"] == "common") & (book["shares"] == 0) & (book["value_usd"] > 0)
    text = book["issuer_name"].fillna("").str.contains(r"\bSH\b.*\bSOLE\b|\d{1,3},\d{3} SH", regex=True)
    return int((zero_sh | (book["value_usd"] > 1e12) | (book["shares"] < 0) | (book["value_usd"] < 0) | text).sum())


# ---- parser: known-truth SEC text lines ---------------------------------------------------------


def test_cusip_and_amounts_from_named_aflac_source_line() -> None:
    rows = parse_legacy_information_table(_table(_HEADER, _AFLAC))
    assert len(rows) == 1
    assert rows.iloc[0]["CUSIP"] == "001055102"
    assert rows.iloc[0]["VALUE"] == 145_462_000
    assert rows.iloc[0]["SSHPRNAMT"] == 3_362_515
    print("\n=== SANITY: Aflac source CUSIP, value ($1000 x 145,462) and shares recovered exactly. Validated.")


def test_colonless_sec_entry_count_rejects_short_book() -> None:
    raw = _table(_HEADER, _AFLAC).replace("Entry Total: 1", "Entry Total\t\t357")  # SEC 0001040273-01-500009's header form
    with pytest.raises(ValueError, match="1 holding lines, cover declares 357"):
        parse_legacy_information_table(raw)
    print("\n=== SANITY: the SEC's colonless 357-entry header rejects a one-row legacy book. Validated.")


def test_intact_legacy_text_without_summary_count_keeps_source_rows() -> None:
    raw = _table(_HEADER, _AFLAC).replace("Form 13F Information Table Entry Total: 1\n", "")
    assert needs_legacy_fallback(raw, pd.DataFrame([{"CUSIP": "001055102"}]))
    rows = parse_legacy_information_table(raw)
    assert len(rows) == 1 and rows.iloc[0].SSHPRNAMT == 3362515
    assert rows.attrs["source_accounted"]
    print("SANITY: closed and fully accounted source retains its holdings without inventing a cover count.")


def test_no_summary_count_keeps_an_edgartools_book_that_matches_the_source() -> None:
    """Greenhaven/Tiger shape: the filing is the table alone. Two independent parses agreeing is the check."""
    raw = _table(_HEADER, _AFLAC, _AMGEN).replace("Form 13F Information Table Entry Total: 2\n", "")
    info = pd.DataFrame(
        [_edgar_line("001055102", "Aflac, Inc.", 145_462_000, 3_362_515), _edgar_line("031162100", "Amgen Inc.", 12_000_000, 100_000)]
    )
    assert not needs_legacy_fallback(raw, info)
    out = _read(_Report(info, txt=raw))
    assert isinstance(out, pd.DataFrame) and set(out["cusip"]) == {"001055102", "031162100"}
    print("\n=== SANITY: no cover count, EdgarTools == source parse row for row -> 2-row book kept. Validated.")


@pytest.mark.parametrize("field,value", [("SharesPrnAmount", 3_362_514), ("Value", 145_461_999)])
def test_no_summary_count_repairs_edgartools_when_source_facts_disagree(field: str, value: Any) -> None:
    raw = _table(_HEADER, _AFLAC).replace("Form 13F Information Table Entry Total: 1\n", "")
    line = _edgar_line("001055102", "Aflac, Inc.", 145_462_000, 3_362_515) | {field: value}
    info = pd.DataFrame([line])
    assert needs_legacy_fallback(raw, info)
    out = _read(_Report(info, txt=raw))
    assert isinstance(out, pd.DataFrame)
    assert out.iloc[0].shares == 3362515 and out.iloc[0].value_usd == 145462000
    print(f"SANITY: missing cover count and {field} off by one are repaired from the intact source row.")


def test_repeated_cusip_continuation_keeps_both_source_rows() -> None:
    raw = _table(
        "        NAME OF ISSUER          TITLE OF CLASS    CUSIP   (x$1000) PRN AMT  PRN CALL DSCRETN",
        "Abbott Laboratories-When Issue COM              002824126    82233  2618878 SH       Sole",
        "                                                            162687  5181109 SH       Defined 01",
    )
    rows = parse_legacy_information_table(raw)
    assert list(rows["CUSIP"]) == ["002824126", "002824126"]
    assert list(rows["VALUE"]) == [82_233_000, 162_687_000]
    assert list(rows["SSHPRNAMT"]) == [2_618_878, 5_181_109]
    print("\n=== SANITY: a CUSIP-less continuation line stays a second source holding of the same CUSIP. Validated.")


def test_multiline_header_and_separator_do_not_become_issuer() -> None:
    raw = "\n".join(
        [
            "Form 13F Information Table Entry Total: 2",
            "<TABLE>",
            "<S>                                 <C>       <C>       <C>",
            "                                    TITLE OF             VALUE   SHARES/",
            "NAME OF ISSUER                       CLASS      CUSIP   (x$1000) PRN AMT",
            "--------------                      --------  --------- -------- -------",
            "3M Company                          COM       88579y101    2676    30000 SH        SOLE",
            "                                                          15505   173800 SH        DEFINED",
            "</TABLE>",
        ]
    )
    rows = parse_legacy_information_table(raw)
    assert list(rows["NAMEOFISSUER"]) == ["3M Company", "3M Company"]
    assert list(rows["VALUE"]) == [2_676_000, 15_505_000]
    print("\n=== SANITY: multiline headers and separators never enter the first issuer name. Validated.")


def test_summary_value_discrepancy_uses_approved_shared_tolerance(caplog: pytest.LogCaptureFixture) -> None:
    aflac = "Aflac, Inc.             Com         001055102 100,000  2,000,000   DEFINED      1   SOLE"
    raw = _table(_HEADER, aflac) + "\nForm 13F Information Table Value Total: 100,500"
    with caplog.at_level(logging.WARNING):
        assert parse_legacy_information_table(raw).iloc[0].VALUE == 100000000
        assert not caplog.records
        assert parse_legacy_information_table(raw.replace("100,500", "150,000")).iloc[0].VALUE == 100000000
    assert any("value total differs" in record.getMessage() for record in caplog.records)
    print("SANITY: modest cover discrepancy is silent; a material mismatch keeps the book with a WARNING (the cover is a reference).")


def test_eight_character_source_cusip_remains_literal_without_identity_evidence() -> None:
    raw = _table(
        "        NAME OF ISSUER         TITLE OF CLASS     CUSIP   (x$1000) PRN AMT  PRN CALL DSCRETN",
        "American Express                COM             25816109     69700  1204627 SH       SOLE",
    )
    rows = parse_legacy_information_table(raw)
    assert list(rows["CUSIP"]) == ["25816109"]
    assert list(rows["VALUE"]) == [69_700_000]
    print("SANITY: a short source identifier is preserved; a checksum alone does not prove a leading-zero interpretation.")


def test_source_column_order_and_false_fragment_rejected() -> None:
    raw = _table(
        "SECURITY DESCRIPTION             CLASS           CUSIP       SHARES      VALUE     (A)    (B)",
        "3 Com Corp                         COM         885535104      100000        817    X",
    )
    rows = parse_legacy_information_table(raw)
    assert rows.iloc[0]["CUSIP"] == "885535104"
    assert rows.iloc[0]["SSHPRNAMT"] == 100_000
    assert rows.iloc[0]["VALUE"] == 817_000
    print("\n=== SANITY: a SHARES-before-VALUE header keeps value and shares in their source roles. Validated.")


def test_shifted_harris_cusip_and_attached_share_type() -> None:
    # SEC 0000813917-11-000099: the displayed header is wider than the rows.
    raw = (
        _table(
            "        NAME OF ISSUER          TITLE OF CLASS    CUSIP   (x$1000) PRN AMT  PRN CALL DSCRETN",
            "3M               COM       88579Y101    55045 766746.00SH       SOLE                  766746.00",
        )
        + "\nForm 13F Information Table Value Total: $55,045 (in thousands)"
    )
    out = _read(_Report(None, txt=raw, error=ValueError("shifted columns")))
    assert isinstance(out, pd.DataFrame)
    row = out.iloc[0]
    assert (row["issuer_name"], row["cusip"], row["shares"], row["value_usd"]) == ("3M", "88579Y101", 766_746, 55_045_000)
    print("\n=== SANITY: Harris 3M retains the source issuer, CUSIP, 766,746 shares and $55,045,000. Validated.")


@pytest.mark.parametrize(
    ("line", "cusip", "amount", "kind"),
    [
        ("D ENERGIZER HLDGS INC    COM 29266R108 18991 1040607.84SH DEFINED 2,4,5 1039441.83 1166.00", "29266R108", 1_040_607.84, "SH"),
        ("UNITED STATES CELLULAR CORP  NOTE 6/1 911684AA6 3,687 10,385,000PRN SOLE", "911684AA6", 10_385_000, "PRN"),
        ("Key 3 Media  com 49326R104 538 117200sh sole 0 117200", "49326R104", 117_200, "SH"),
    ],
)
def test_complete_attached_amount_tokens(line: str, cusip: str, amount: float, kind: str) -> None:
    # Exact amount tokens in SEC 0000813917-00-000053, 0000949509-02-000013,
    # and 0001112520-02-000005; shortened whitespace makes column shifts explicit.
    rows = parse_legacy_information_table(_table(_HEADER, line))
    assert rows.iloc[0]["CUSIP"] == cusip
    assert rows.iloc[0]["SSHPRNAMT"] == amount
    assert rows.iloc[0]["SSHPRNAMTTYPE"] == kind
    print(f"\n=== SANITY: complete attached {kind} amount is {amount}, without partial numeric matching. Validated.")


def test_explicit_billion_cover_uses_displayed_precision(caplog: pytest.LogCaptureFixture) -> None:
    # SEC 0001036325-12-000004: $45.864729bn rounds to its $45.9bn cover.
    raw = _table(_HEADER, _AFLAC.replace("145,462", "45,864,729"))
    raw += "\nForm 13F Information Table Value Total: $45.9 Billion"
    assert parse_legacy_information_table(raw).iloc[0]["VALUE"] == 45_864_729_000
    with caplog.at_level(logging.WARNING):
        assert parse_legacy_information_table(raw.replace("45,864,729", "45,840,000")).iloc[0]["VALUE"] == 45_840_000_000
    assert any("value total differs" in record.getMessage() for record in caplog.records)
    print("\n=== SANITY: explicit Billion precision admits $45.864729bn silently and keeps $45.84bn with a WARNING. Validated.")


def test_wrapped_pabrai_value_and_class_do_not_use_voting_or_date() -> None:
    # SEC 0001546927-13-000077: value is on the next line; SH amount and
    # voting authority are on the first. The warrant date is a class fragment.
    raw = "\n".join(
        [
            "Form 13F Information Table Entry Total: 2",
            "Form 13F Information Table Value Total: 161,275.81 (thousands)",
            "<TABLE>",
            "       Name of Issuer         Title of Class   CUSIP   (x$1000)  Prn Amt  Prn Call Discretion Managers   Sole",
            "<S>                           <C>            <C>       <C>      <C>       <C> <C>  <C>        <C>      <C>",
            "BANK OF AMERICA CORPORATION   COM            060505104          7,502,000 SH          SOLE             7,502,000",
            "                                                       91,374.36",
            "GENERAL MTRS CO               *W EXP         37045V126          5,928,876 SH          SOLE             5,928,876",
            "                              07/10/201                69,901.45",
            "</TABLE>",
        ]
    )
    rows = parse_legacy_information_table(raw).set_index("CUSIP")
    assert rows.loc["060505104", "VALUE"] == 91_374_360
    assert rows.loc["060505104", "SSHPRNAMT"] == 7_502_000
    assert rows.loc["37045V126", "VALUE"] == 69_901_450
    assert rows.loc["37045V126", "SSHPRNAMT"] == 5_928_876
    assert rows.loc["37045V126", "NAMEOFISSUER"] == "GENERAL MTRS CO"
    assert rows.loc["37045V126", "TITLEOFCLASS"] == "*W EXP 07/10/201"
    print("\n=== SANITY: wrapped values and warrant class are associated with their holdings; dates/votes are excluded. Validated.")


def test_headerless_table_page_inherits_only_compatible_markers() -> None:
    # SEC 0001193125-12-439529: second page has identical column markers.
    markers = "<S>                              <C>              <C>       <C>       <C>     <C> <C>  <C>"
    raw = "\n".join(
        [
            "Form 13F Information Table Entry Total: 2",
            "Form 13F Information Table Value Total: $60,303 (thousands)",
            "<TABLE>",
            "NAME OF ISSUER                   -TITLE OF CLASS- --CUSIP--   x$1000  PRN AMT PRN CALL DSCRETN",
            markers,
            "D APPLE INC                      COM              037833100    35406    53074 SH       SOLE                  47837        0     5237",
            "</TABLE><PAGE><TABLE>",
            markers,
            "D COCA-COLA CO                   COM              191216100    24897   656384 SH       SOLE                 595730        0    60654",
            "</TABLE>",
        ]
    )
    rows = parse_legacy_information_table(raw)
    assert list(rows["CUSIP"]) == ["037833100", "191216100"]
    assert list(rows["VALUE"]) == [35_406_000, 24_897_000]
    assert list(rows["SSHPRNAMT"]) == [53_074, 656_384]
    with pytest.raises(ValueError):
        parse_legacy_information_table(raw.replace("</TABLE><PAGE><TABLE>\n" + markers, "</TABLE><PAGE><TABLE>\n<S> <C> <C>"))
    print("\n=== SANITY: compatible marker-only page retains both holdings; incompatible markers cannot certify a book. Validated.")


@pytest.mark.parametrize("amount", ["117200oops", "10,38,500", "1.2.3", "1e6"])
def test_malformed_amount_cannot_fall_back_to_voting_numbers(amount: str) -> None:
    raw = _table(_HEADER, f"Key 3 Media  com 49326R104 538 {amount} SH SOLE 0 117200")
    with pytest.raises(ValueError):
        parse_legacy_information_table(raw)
    print(f"\n=== SANITY: unsupported amount {amount!r} is rejected before voting fields can replace it. Validated.")


# ---- `_read_filing` hook ------------------------------------------------------------------------


@pytest.mark.parametrize("edgar_parsed", [False, True])
def test_invalid_logical_line_cannot_be_hidden_by_duplicate_grouping(edgar_parsed: bool) -> None:
    zero = _AFLAC.replace("3,362,515", "-1")
    info = pd.DataFrame([_edgar_line("001055102", "Aflac, Inc.", 145_462_000, shares) for shares in [-1, 3_362_515]]) if edgar_parsed else None
    out = _read(_Report(info, txt=_table(_HEADER, zero, _AFLAC)))
    assert isinstance(out, f13.ReadFailure) and not out.transient
    print("\n=== SANITY: a negative source quantity cannot disappear into a positive duplicate-CUSIP aggregate. Validated.")


def test_invalid_shifted_cusip_cannot_become_a_prior_holding_continuation() -> None:
    raw = _table(_HEADER, _AFLAC, _AMGEN.replace("031162100", "031162101"))
    with pytest.raises(ValueError):
        parse_legacy_information_table(raw)
    print("\n=== SANITY: invalid named source CUSIP is rejected even after a valid prior holding. Validated.")


def test_clean_xml_is_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    """(a) An XML-era filing never reads the text table nor calls the fallback."""
    for name in ("parse_legacy_information_table", "needs_legacy_fallback"):
        monkeypatch.setattr(f13, name, _forbidden(name), raising=False)
    info = pd.DataFrame([_edgar_line("001055102", "Aflac", 145_462_000, 3_362_515)])
    report = _Report(info, txt="unused", xml="<informationTable/>")
    out = _read(report)
    expected = f13._book_frame("0001036325", "2010-05-13", "2010-03-31", info)
    assert isinstance(out, pd.DataFrame)
    pd.testing.assert_frame_equal(out, expected)
    assert report.txt_reads == 0
    print("\n=== SANITY: clean XML -> the EdgarTools book unchanged, 0 text reads, 0 fallback calls. Validated.")


def test_clean_text_parse_skips_the_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """A text book EdgarTools parsed in full (count and CUSIPs match the source) is not reparsed."""
    monkeypatch.setattr(f13, "parse_legacy_information_table", _forbidden("parse_legacy_information_table"), raising=False)
    info = pd.DataFrame([_edgar_line("001055102", "Aflac, Inc.", 145_462_000, 3_362_515)])
    out = _read(_Report(info, txt=_table(_HEADER, _AFLAC)))
    assert isinstance(out, pd.DataFrame)
    pd.testing.assert_frame_equal(out, f13._book_frame("0001036325", "2010-05-13", "2010-03-31", info))
    print("\n=== SANITY: a source-consistent EdgarTools text parse is kept as is, with no reparse. Validated.")


@pytest.mark.parametrize("field,value", [("SharesPrnAmount", 3_362_514), ("Value", 145_461_999), ("PutCall", "CALL")])
def test_matching_cover_and_cusip_cannot_accept_wrong_holding_facts(field: str, value: Any) -> None:
    raw = _table(_HEADER, _AFLAC) + "\nForm 13F Information Table Value Total: 145,462"
    parsed = _edgar_line("001055102", "Aflac, Inc.", 145_462_000, 3_362_515)
    parsed[field] = value
    out = _read(_Report(pd.DataFrame([parsed]), txt=raw))
    assert isinstance(out, pd.DataFrame)
    row = out.iloc[0]
    assert (row["shares"], row["value_usd"], row["call_shares"], row["debt_prn"]) == (3_362_515, 145_462_000, 0, 0)
    print(f"\n=== SANITY: matching count, CUSIP and cover cannot admit source-discrepant {field}; exact Aflac facts recovered. Validated.")


@pytest.mark.parametrize("parsed_type", ["Shares", "Principal", "PRN"])
def test_oaktree_source_principal_is_never_common_shares(parsed_type: str, monkeypatch: pytest.MonkeyPatch) -> None:
    # SEC 0000949509-03-000018: source SHA256 e4804579bd49b7a0a3cb7160d0a022b1226d3c8a969da7c496da596dfc9f352c.
    raw = (
        _table(
            "NAME OF ISSUER                TITLE OF CLASS    CUSIP      (X$1000)   PRN AMT     PRN DSCRETN",
            "QUANTA SVCS INC               NOTE 4.000% 7/0   74762EAA0  52,772     108,870,000 PRN  SOLE",
        )
        + "\nForm 13F Information Table Value Total: $52,772"
    )
    parsed = _edgar_line("74762EAA0", "QUANTA SVCS INC", 52_772_000, 108_870_000)
    parsed["Type"] = parsed_type
    if parsed_type != "Shares":
        monkeypatch.setattr(f13, "parse_legacy_information_table", _forbidden("parse_legacy_information_table"))
    out = _read(_Report(pd.DataFrame([parsed]), txt=raw))
    assert isinstance(out, pd.DataFrame)
    row = out.iloc[0]
    assert (row["shares"], row["value_usd"], row["debt_prn"], row["debt_value"]) == (0, 0, 108_870_000, 52_772_000)
    print(
        f"\n=== SANITY: Oaktree's 108,870,000 PRN remains debt for EdgarTools type {parsed_type!r}; clean principal aliases avoid fallback. Validated."
    )


@pytest.mark.parametrize("xml_names,parsed_type", [(False, ""), (False, "SH"), (False, "Shares"), (True, "sh"), (True, None)])
def test_clean_reordered_duplicate_lines_preserve_field_aliases(xml_names: bool, parsed_type: str | None, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = _table(_HEADER, _AFLAC, _AFLAC.replace("145,462  3,362,515", "12,000  100,000"))
    info = pd.DataFrame(
        [_edgar_line("001055102", "Aflac, Inc.", 12_000_000, 100_000), _edgar_line("001055102", "Aflac, Inc.", 145_462_000, 3_362_515)]
    )
    info["Type"] = parsed_type
    if xml_names:
        info = info.rename(columns={"SharesPrnAmount": "SSHPRNAMT", "Type": "SSHPRNAMTTYPE"})
    info.columns = info.columns.str.swapcase()
    monkeypatch.setattr(f13, "parse_legacy_information_table", _forbidden("parse_legacy_information_table"))
    out = _read(_Report(info, txt=raw))
    assert isinstance(out, pd.DataFrame) and len(out) == 1
    assert (out.iloc[0]["shares"], out.iloc[0]["value_usd"]) == (3_462_515, 157_462_000)
    print("\n=== SANITY: reordered duplicate-CUSIP lines and case-insensitive XML/EdgarTools share aliases remain clean. Validated.")


def test_malformed_text_parse_is_replaced_by_verified_source_rows() -> None:
    """(b) EdgarTools merges Amgen into a false CUSIP and drops a line; the source text is verifiable."""
    info = pd.DataFrame([_edgar_line("COM031162", "Aflac, Inc. Com 001055102 145,462", 12_000_000, 100_000)])
    out = _read(_Report(info, txt=_table(_HEADER, _AFLAC, _AMGEN)))
    assert isinstance(out, pd.DataFrame)
    got = out.set_index("cusip")
    assert set(got.index) == {"001055102", "031162100"}
    assert got.loc["001055102", "value_usd"] == 145_462_000 and got.loc["001055102", "shares"] == 3_362_515
    assert got.loc["031162100", "value_usd"] == 12_000_000 and got.loc["031162100", "issuer_name"] == "Amgen Inc."
    assert _f001_garbage(out) == 0
    print("\n=== SANITY: a short, misaligned text parse -> 2 source-verified rows with exact value and shares. Validated.")


def test_edgartools_text_exception_uses_the_verified_source() -> None:
    out = _read(_Report(None, txt=_table(_HEADER, _AFLAC), error=ValueError("misaligned text table")))
    assert isinstance(out, pd.DataFrame) and list(out["cusip"]) == ["001055102"]
    print("\n=== SANITY: an EdgarTools text-parse exception falls back to the verified source row. Validated.")


def test_transient_error_stays_transient_without_reading_text() -> None:
    """A throttle on the information-table download is retried by `sec_io` and stays transient (held back
    by the low watermark); the text table is never read and no fallback is attempted."""
    report = _Report(None, txt=_table(_HEADER, _AFLAC), download_error=ConnectionError("429 Too Many Requests"))
    out = _read(report)
    assert isinstance(out, f13.ReadFailure) and out.transient and "TransientReadError" in out.reason
    assert report.txt_reads == 0
    print("\n=== SANITY: a throttle on the infotable download stays a transient failure; no fallback attempted. Validated.")


def test_a_parse_error_quoting_429_is_not_transient() -> None:
    """EdgarTools' own text-parse error quoting ',429' is parsed outside `sec_io`, so it never reads as a throttle."""
    raw = _table(_HEADER, _AFLAC).replace("Entry Total: 1", "Entry Total: 460,429")
    out = _read(_Report(None, txt=raw, error=ValueError("bad row 460,429 Too Many")))
    assert isinstance(out, f13.ReadFailure) and not out.transient
    print(f"\n=== SANITY: an EdgarTools ValueError quoting ',429' -> ReadFailure(transient=False): {out.reason[:70]!r}. Validated.")


def test_unverifiable_text_is_a_parse_failure_not_stored() -> None:
    """(c) The text declares 357 entries but carries one line: no rows, a deterministic failure."""
    raw = _table(_HEADER, _AFLAC).replace("Entry Total: 1", "Entry Total: 357")
    info = pd.DataFrame([_edgar_line("001055102", "Aflac", 145_462_000, 3_362_515)])
    out = _read(_Report(info, txt=raw))
    assert isinstance(out, f13.ReadFailure), "an unverifiable book must not be returned"
    assert not out.transient
    assert "unverifiable" in out.reason and "357" in out.reason
    print(f"\n=== SANITY: unverifiable legacy text -> ReadFailure(transient=False): {out.reason[:90]!r}. Validated.")


def test_unverifiable_reason_quoting_429_is_not_transient() -> None:
    """Torray 2011-06-30 shape: the reason quotes a source number ending in ",429", which the
    rate-limit text matcher reads as an HTTP 429; the failure must stay deterministic."""
    raw = _table(_HEADER, _AFLAC).replace("Entry Total: 1", "Entry Total: 460,429")
    info = pd.DataFrame([_edgar_line("001055102", "Aflac", 145_462_000, 3_362_515)])
    out = _read(_Report(info, txt=raw))
    assert isinstance(out, f13.ReadFailure) and "460,429" in out.reason
    assert not out.transient, "a source-verification failure is never a throttle"
    print(f"\n=== SANITY: reason {out.reason[-40:]!r} quotes ',429' yet stays ReadFailure(transient=False). Validated.")


def test_zero_share_huge_value_parse_is_replaced() -> None:
    """(d) F-001 Davis shape: right CUSIPs and count, but shares 0 and a $3.3e17 value."""
    info = pd.DataFrame(
        [_edgar_line("001055102", "Aflac, Inc.", 327_403_410_520_673_000, 0), _edgar_line("031162100", "Amgen Inc.", 12_000_000, 100_000)]
    )
    out = _read(_Report(info, txt=_table(_HEADER, _AFLAC, _AMGEN)))
    assert isinstance(out, pd.DataFrame)
    assert _f001_garbage(out) == 0
    assert out.set_index("cusip").loc["001055102", "value_usd"] == 145_462_000
    print("\n=== SANITY: a zero-share $3.3e17 row is replaced by the source's 3,362,515 sh / $145.5m. Validated.")


def test_unsplit_issuer_text_parse_is_replaced() -> None:
    """(d) F-001 Davis 2005 shape: a whole info-table line merged into `issuer_name`."""
    info = pd.DataFrame([_edgar_line("001055102", "Alltel Corp Common Stock 20039103 451 8,215 SH SOLE", 145_462_000, 3_362_515)])
    out = _read(_Report(info, txt=_table(_HEADER, _AFLAC)))
    assert isinstance(out, pd.DataFrame)
    assert _f001_garbage(out) == 0 and list(out["issuer_name"]) == ["Aflac, Inc."]
    print("\n=== SANITY: an issuer name carrying unsplit table text is replaced by the source issuer. Validated.")


def test_value_total_disagreement_is_replaced() -> None:
    """(d) Count and CUSIPs match the source but shares were read into the value column: the
    Summary Page value total ($145,462k) exposes it."""
    raw = _table(_HEADER, _AFLAC) + "\nForm 13F Information Table Value Total: 145,462"
    info = pd.DataFrame([_edgar_line("001055102", "Aflac, Inc.", 3_362_515_000, 3_362_515)])
    out = _read(_Report(info, txt=raw))
    assert isinstance(out, pd.DataFrame)
    assert out.set_index("cusip").loc["001055102", "value_usd"] == 145_462_000
    print("\n=== SANITY: a 23x value-column misread is replaced by the source row matching the SEC value total. Validated.")


def test_garbage_in_the_source_itself_is_a_parse_failure() -> None:
    """(d) A source line valued with zero shares verifies on totals but is still never returned."""
    zero = "Aflac, Inc.             Com         001055102 145,462          0   DEFINED      1   SOLE"
    info = pd.DataFrame([_edgar_line("001055102", "Aflac, Inc.", 145_462_000, 0)])
    out = _read(_Report(info, txt=_table(_HEADER, zero)))
    assert isinstance(out, f13.ReadFailure) and not out.transient
    print(f"\n=== SANITY: a zero-share valued source row -> deterministic failure, no rows: {out.reason[:80]!r}. Validated.")


def test_legacy_separate_cover_and_zero_quantity_reach_p1(monkeypatch: pytest.MonkeyPatch) -> None:
    table = _table(_HEADER, _AFLAC.replace("3,362,515", "0"))
    raw = "[X] 13F HOLDINGS REPORT\nForm 13F Information Table Value Total: $145,462\n" + table
    filing = _Filing(_Report(None, txt=table))
    monkeypatch.setattr(filing, "full_text_submission", lambda: raw, raising=False)
    packet = f13.read_filing_vintage(FilingStamp.of(filing, filing.cik))
    assert not isinstance(packet, f13.ReadFailure), packet
    manifest, rows = packet
    assert manifest.iloc[0].source_row_count == 1
    # One zero-share row and a ratio-1 cover carry no unit evidence: the cell is stored as dollars.
    assert rows.iloc[0].shares == 0 and rows.iloc[0].value_usd == 145462
    assert manifest.iloc[0].value_unit_evidence.endswith(NO_EVIDENCE)
    print("SANITY: separate legacy cover/table reconciles; zero quantity and positive source value survive for inverse P1 price fallback.")


@pytest.mark.parametrize("missing", ["count", "value", "both"])
def test_intact_legacy_missing_summary_is_truthful_and_replayable(filing_store: Any, missing: str) -> None:
    raw = "Form 13F Information Table Value Total: $145,462\n" + _table(_HEADER, _AFLAC)
    if missing in {"count", "both"}:
        raw = raw.replace("Form 13F Information Table Entry Total: 1", "")
    if missing in {"value", "both"}:
        raw = raw.replace("Form 13F Information Table Value Total: $145,462", "")
    filing = _Filing(_Report(None, txt=raw))
    packet = f13.text_filing_vintage(raw, FilingStamp.of(filing, filing.cik))
    manifest, rows = packet
    assert pd.isna(manifest.iloc[0].declared_row_count) == (missing in {"count", "both"})
    assert pd.isna(manifest.iloc[0].declared_value_total) == (missing in {"value", "both"})
    assert manifest.iloc[0].source_row_count == 1
    assert rows.iloc[0].shares == 3362515 and rows.iloc[0].raw_value_usd == 145462
    for table in (f13.Tables.sec13f_hr, f13.Tables.sec13f_manager_holdings):
        positions = rows.assign(ticker="AFL") if table == f13.Tables.sec13f_hr else rows
        assert save_13f_filing(filing_store, manifest, positions, table=table) == 1
        assert save_13f_filing(filing_store, manifest, positions, table=table) == 0
        marker = filing_store.load(table, where={"cusip": "_empty"}, markers=True)
        assert pd.isna(marker.iloc[0].declared_row_count) == (missing in {"count", "both"})
    print(f"SANITY: intact source with missing {missing} reaches both existing tables without invented declarations; replay is idempotent.")


@pytest.mark.parametrize("damage", ["unclosed-table", "unsupported-page", "hidden-row"])
def test_missing_cover_cannot_certify_truncated_or_unaccounted_source(damage: str) -> None:
    raw = _table(_HEADER, _AFLAC).replace("Form 13F Information Table Entry Total: 1", "")
    if damage == "unclosed-table":
        raw += "\n<TABLE>\n" + _HEADER + "\n" + _AMGEN
    elif damage == "unsupported-page":
        raw += "\n<TABLE>\n<S> <C> <C>\n" + _AMGEN + "\n</TABLE>"
    else:
        raw = raw.replace("</TABLE>", _AMGEN.replace(" 12,000", " <UNKNOWN>12,000") + "\n</TABLE>")
    filing = _Filing(_Report(None, txt=raw))
    with pytest.raises(ValueError):
        f13.text_filing_vintage(raw, FilingStamp.of(filing, filing.cik))
    print(f"SANITY: missing summary does not waive {damage}; failed packet cannot reach either writer.")


def test_summary_variants_cover_scale_voting_and_footer() -> None:
    raw = "Form 13F Table Value Total (x$1000): 145,462\n" + _table(_HEADER, _AFLAC)
    raw = raw.replace("Information Table Entry", "Table Entry")
    raw = raw.replace(
        "</TABLE>",
        "                                                                                SOLE 1 200\nGRAND TOTALS 145,462 3,362,515\n</TABLE>",
    )
    filing = _Filing(_Report(None, txt=raw))
    manifest, rows = f13.text_filing_vintage(raw, FilingStamp.of(filing, filing.cik))
    assert manifest.iloc[0].declared_row_count == 1 and manifest.iloc[0].declared_value_total == 145462
    assert len(rows) == 1 and rows.iloc[0].value_usd == 145462000
    usd_cover = raw.replace("Form 13F Table Value Total (x$1000): 145,462", "Form 13F Table Value Total: 145,462,000")
    assert f13.text_filing_vintage(usd_cover, FilingStamp.of(filing, filing.cik))[1].iloc[0].value_usd == 145462000
    print("SANITY: abbreviated summary and unit-before-number are recognized; cover scale is independent; votes/footer never add holdings.")


def test_source_currency_separators_and_class_boundary() -> None:
    raw = _table(
        _HEADER,
        "ACADIA RLTY TR  COM SH BEN INT004239109  $122  6797 SH SOLE 6797 0 0",
        "ACCREDO HEALTH INC  COM 00437P-10-7 $ 46,567 1,025,710 SH SOLE 1,025,710",
    )
    rows = parse_legacy_information_table(raw)
    assert rows.CUSIP.tolist() == ["004239109", "00437P107"]
    assert rows.VALUE.tolist() == [122000, 46567000]
    assert rows.SSHPRNAMT.tolist() == [6797, 1025710]
    print("SANITY: actual dollar signs, standard CUSIP separators and a joined class/CUSIP boundary retain exact source economics.")


def test_plain_text_information_table_and_report_summary_footer() -> None:
    raw = (
        "<DOCUMENT>\n<TEXT>\n"
        + _table(_HEADER, _AFLAC).replace("<TABLE>\n", "").replace("</TABLE>", "S REPORT SUMMARY 1 DATA RECORDS 145462 0 OTHER MANAGERS")
        + "\n</TEXT>\n</DOCUMENT>"
    )
    rows = parse_legacy_information_table(raw)
    assert len(rows) == 1 and rows.iloc[0].SSHPRNAMT == 3362515
    with pytest.raises(ValueError):
        parse_legacy_information_table(raw.replace("</TEXT>", ""))
    print("SANITY: a complete text document need not use TABLE tags; report footer is excluded and cut text is rejected.")


def test_numeric_only_voting_subdivisions_do_not_create_cusips() -> None:
    raw = "\n".join(
        [
            "Form 13F Information Table Entry Total: 1",
            "<TABLE>",
            "NAME OF ISSUER                 TITLE OF CLASS    CUSIP   VALUE    SHARES   SH PRN DSCRETN",
            "<S>                            <C>              <C>     <C>      <C>      <C>    <C>",
            "AMGEN INC                      COM              031162100 12000   12345678 SH DEFINED",
            "                                                                                DEFINED 3   12345678 0 0",
            "</TABLE>",
        ]
    )
    rows = parse_legacy_information_table(raw)
    assert len(rows) == 1 and rows.iloc[0].CUSIP == "031162100"
    assert rows.iloc[0].SSHPRNAMT == 12345678
    print("SANITY: an eight-digit voting quantity cannot become a source identifier or a second holding.")


def test_source_subtotals_and_real_economic_continuations() -> None:
    raw = _table(
        _HEADER,
        _AFLAC,
        "                                                1000 100 SH DEFINED 1 100 0 0",
        "                                                146462 3362615",
        _AMGEN,
        "                                                12000 100000",
    ).replace("Entry Total: 5", "Entry Total: 3")
    rows = parse_legacy_information_table(raw)
    assert rows.SSHPRNAMT.tolist() == [3362515, 100, 100000]
    assert rows.VALUE.tolist() == [145462000, 1000000, 12000000]
    assert len(rows) == 3
    no_count = raw.replace("Form 13F Information Table Entry Total: 3", "")
    with pytest.raises(ValueError, match="Ambiguous unlabelled"):
        parse_legacy_information_table(no_count)
    print("SANITY: repeated exact subtotals are removed with independent count proof; the economic continuation is preserved.")


def test_adjacent_value_quantity_requires_independent_voting_proof() -> None:
    line = "LABORATORY CORP AMER HLDGS  NOTE 9/1 50540RAC6 118352158280000 PRN SOLE 158280000 0 0"
    row = parse_legacy_information_table(_table(_HEADER, line)).iloc[0]
    assert row.VALUE == 118352000 and row.SSHPRNAMT == 158280000 and row.SSHPRNAMTTYPE == "PRN"
    with pytest.raises(ValueError):
        parse_legacy_information_table(_table(_HEADER, line.replace("SOLE 158280000", "SOLE 150000000")))
    print("SANITY: touching value/principal digits are recovered only from a matching three-cell voting quantity.")


@pytest.mark.parametrize("flag", ["[X]", "{X}", "( X )"])
def test_legacy_checked_report_scope_variants(flag: str) -> None:
    raw = flag + " 13 F COMBINATION REPORT\n" + _table(_HEADER, _AFLAC)
    filing = _Filing(_Report(None, txt=raw))
    manifest, _ = f13.text_filing_vintage(raw, FilingStamp.of(filing, filing.cik))
    assert manifest.iloc[0].report_type == "13F COMBINATION REPORT"
    print("SANITY: legacy checkbox punctuation never upgrades an explicitly partial report into full exit support.")


@pytest.mark.parametrize("separator", ["  ", "\t"])
def test_etf_class_digits_cannot_become_cusip_or_value(separator: str) -> None:
    line = separator.join(["ISHARES TR", "RUSSELL1000VAL", "464287598", "6278", "91200 SH SOLE 91200 0 0"])
    row = parse_legacy_information_table(_table(_HEADER, line)).iloc[0]
    assert row.CUSIP == "464287598"
    assert row.VALUE == 6278000 and row.SSHPRNAMT == 91200
    print("SANITY: digits within the ETF class do not shift CUSIP, value or quantity columns.")


def test_subtotals_between_manager_groups_reset_the_source_sum() -> None:
    raw = _table(
        _HEADER,
        _AFLAC,
        "                                                1000 100 SH DEFINED 1 100 0 0",
        "                                                146462 3362615",
        "                                                2000 200 SH DEFINED 2 200 0 0",
        "                                                2000 200",
    ).replace("Entry Total: 5", "Entry Total: 3")
    rows = parse_legacy_information_table(raw)
    assert rows.SSHPRNAMT.tolist() == [3362515, 100, 200]
    assert rows.VALUE.tolist() == [145462000, 1000000, 2000000]
    print("SANITY: each manager-group subtotal is checked against only its own preceding economic rows.")


def test_subtotal_independent_rounding_keeps_exact_share_proof() -> None:
    raw = _table(
        _HEADER,
        "AMDOCS LTD  ORD G02602103 550 19818 SH DEFINED 9 19818 0 0",
        " 226 8151 SH DEFINED 3 8055 96 0",
        " 4082 147200 SH DEFINED 85 147200 0 0",
        " 4857 175169",
    ).replace("Entry Total: 4", "Entry Total: 3")
    rows = parse_legacy_information_table(raw)
    assert rows.VALUE.tolist() == [550000, 226000, 4082000]
    assert rows.SSHPRNAMT.sum() == 175169
    wrong_quantity = parse_legacy_information_table(raw.replace("4857 175169", "4857 175168").replace("Entry Total: 3", "Entry Total: 4"))
    assert len(wrong_quantity) == 4
    print("SANITY: independently rounded VALUE subtotals require exact quantity and independent cover count; source values stay unchanged.")


def test_missing_declarations_require_source_accounting_at_writer(filing_store: Any) -> None:
    raw = _table(_HEADER, _AFLAC).replace("Form 13F Information Table Entry Total: 1", "")
    filing = _Filing(_Report(None, txt=raw))
    manifest, rows = f13.text_filing_vintage(raw, FilingStamp.of(filing, filing.cik))
    unaccounted = manifest.assign(value_unit_evidence="filing_date_default")
    for table in (f13.Tables.sec13f_hr, f13.Tables.sec13f_manager_holdings):
        positions = rows.assign(ticker="AFL") if table == f13.Tables.sec13f_hr else rows
        with pytest.raises(ValueError, match="does not reconcile"):
            save_13f_filing(filing_store, unaccounted, positions, table=table)
        assert filing_store.row_count(table) == 0
    print("SANITY: nullable cover fields alone cannot waive source validation or create completion markers.")


def test_missing_declarations_replay_null_types_and_preserve_conflicts(filing_store: Any) -> None:
    raw = _table(_HEADER, _AFLAC).replace("Form 13F Information Table Entry Total: 1", "")
    filing = _Filing(_Report(None, txt=raw))
    manifest, rows = f13.text_filing_vintage(raw, FilingStamp.of(filing, filing.cik))
    for table in (f13.Tables.sec13f_hr, f13.Tables.sec13f_manager_holdings):
        positions = rows.assign(ticker="AFL") if table == f13.Tables.sec13f_hr else rows
        assert save_13f_filing(filing_store, manifest, positions, table=table) == 1
        for missing in (None, float("nan"), pd.NA):
            replay = manifest.assign(declared_row_count=missing, declared_value_total=missing)
            assert save_13f_filing(filing_store, replay, positions, table=table) == 0
        with pytest.raises(ValueError, match="conflict"):
            save_13f_filing(filing_store, manifest.assign(declared_row_count=1), positions, table=table)
    print("SANITY: SQL null, NaN and pandas NA replay identically; adding an observed declaration conflicts with immutable source metadata.")


def test_private_identifier_and_inferred_scope_reach_public_books(filing_store: Any) -> None:
    raw = _table(_HEADER, "NAMIC INSURANCE COMPANY INC  COM 62989*105 21 93 SH SOLE 93 0 0")
    filing = _Filing(_Report(None, txt=raw))
    manifest, rows = f13.text_filing_vintage(raw, FilingStamp.of(filing, filing.cik))
    assert rows.iloc[0].cusip == "62989*105" and rows.iloc[0].shares == 93
    assert manifest.iloc[0].report_type == f13.SEC_13F_INFERRED_HOLDINGS_REPORT
    calendar = pd.bdate_range("2010-03-01", "2010-06-01")
    for table in (f13.Tables.sec13f_hr, f13.Tables.sec13f_manager_holdings):
        positions = rows.assign(ticker=table.empty_marker[1]) if table == f13.Tables.sec13f_hr else rows
        save_13f_filing(filing_store, manifest, positions, table=table)
        saved = filing_store.load(table)
        markers = filing_store.load(table, where={"cusip": table.empty_marker[1]}, markers=True)
        book, states = public_books(markers, saved, "2010-06-01", calendar)
        assert book.iloc[0].cusip == "62989*105" and book.iloc[0].shares == 93
        assert states.zero_eligible.all() and states.comparison_eligible.all()
    print("SANITY: source private CUSIP stays literal through both stores; inferred full-report scope supports ordinary daily comparison and exits.")


@pytest.mark.parametrize(
    "label",
    [
        "FORM 13F DATE 03-29-13 NORTHERN TRUST CORPORATION PAGE 2",
        "COLUMN 1 COLUMN 2 COLUMN 3 COLUMN 4 COLUMN 5 COLUMN 6 COLUMN 7 COLUMN 8",
        "FINAL TOTALS FOR 2 RECORDS 157462",
    ],
)
def test_page_labels_and_final_totals_are_not_economic_rows(label: str) -> None:
    raw = _table(_HEADER, _AFLAC, label, _AMGEN).replace("Entry Total: 3", "Entry Total: 2")
    rows = parse_legacy_information_table(raw)
    assert rows.CUSIP.tolist() == ["001055102", "031162100"]
    assert rows.VALUE.tolist() == [145462000, 12000000]
    assert rows.SSHPRNAMT.tolist() == [3362515, 100000]
    print("SANITY: page headings, numbered column labels and final totals cannot become CUSIP continuations; both original economic rows survive.")


@pytest.mark.parametrize("count", [":\n    2", ":\n\n\t2", "\n2"])
def test_next_line_entry_count_is_preserved_in_both_destinations(count: str, filing_store: Any) -> None:
    raw = _table(_HEADER, _AFLAC, _AMGEN).replace("Entry Total: 2", "Entry Total" + count)
    filing = _Filing(_Report(None, txt=raw))
    manifest, rows = f13.text_filing_vintage(raw, FilingStamp.of(filing, filing.cik))
    assert manifest.iloc[0].declared_row_count == 2
    assert manifest.iloc[0].source_row_count == 2
    for table in (f13.Tables.sec13f_hr, f13.Tables.sec13f_manager_holdings):
        positions = rows.assign(ticker=["AFL", "AMGN"]) if table == f13.Tables.sec13f_hr else rows
        assert save_13f_filing(filing_store, manifest, positions, table=table) == 2
        marker = filing_store.load(table, where={"cusip": "_empty"}, markers=True)
        assert marker.iloc[0].declared_row_count == 2
        assert save_13f_filing(filing_store, manifest, positions, table=table) == 0
    print("SANITY: a numeric count on the next line remains an observed declaration in both destinations; replay preserves all economic facts.")


def test_empty_entry_count_cannot_borrow_the_next_summary_field() -> None:
    raw = _table(_HEADER, _AFLAC, _AMGEN).replace("Entry Total: 2", "Entry Total:\nForm 13F Information Table Value Total: $157,462")
    filing = _Filing(_Report(None, txt=raw))
    manifest, rows = f13.text_filing_vintage(raw, FilingStamp.of(filing, filing.cik))
    assert pd.isna(manifest.iloc[0].declared_row_count)
    assert manifest.iloc[0].declared_value_total == 157462
    assert rows.shares.tolist() == [3362515, 100000]
    print("SANITY: an actually blank count remains missing; the following value label cannot supply its number.")


def test_spaced_etf_class_cannot_shift_the_header_aligned_cusip() -> None:
    # Exact class/column layout from SEC 0001157523-13-002697.
    header = "          NAME OF ISSUER              TITLE OF     CUSIP    VALUE     SHRS OR   SH/ PUT/ INVEST-"
    line = "ISHARES RUSSELL 2000                 ISHRR2000   464287655    19399      205633 SH       Sole     NONE         2621           203012"
    row = parse_legacy_information_table(_table(header, line)).iloc[0]
    assert row.CUSIP == "464287655"
    assert row.TITLEOFCLASS == "ISHRR2000"
    assert row.VALUE == 19399000 and row.SSHPRNAMT == 205633
    print("SANITY: the header-aligned CUSIP wins over a numeric ETF class; reported value and quantity do not shift columns.")


def test_short_source_cusip_cannot_be_replaced_by_a_valid_numeric_value() -> None:
    header = "ISSUER                 TYPE       CUSIP      VALUE       SHS"
    line = "PRIVATE HOLDING        COM        46626V20   464287655   19399 SH SOLE 19399 0 0"
    row = parse_legacy_information_table(_table(header, line)).iloc[0]
    assert row.CUSIP == "46626V20"
    assert row.VALUE == 464287655000 and row.SSHPRNAMT == 19399
    print("SANITY: a checksum-valid monetary token cannot replace the literal short source identifier.")


@pytest.mark.parametrize(
    "metadata",
    [
        " /s/ Lisa Leonardo  121 E. Seneca Street, Ithaca, NY  14851        04/18/2013",
        "Phone:     212 856 4722",
        "Phone:     (415) 247 5300",
        "F o r m  1 3 F",
        "SK 01864 0005 1382043",
        "Phone:     44 20 7144 1000",
        "PHONE                                   212 508 3920",
        "BUSINESS PHONE: 310 734 1200",
        "Phone:     +44 (0) 1534 723825",
        "Column\tColumn\tColumn\tColumn\tColumn\tColumn\tColumn\tColumn\n1\t\t2\t3\t4\t5\t6\t7\t8",
        "             COLUMN 1                 COLUMN 2      COLUMN 3     4           COLUMN 5        COLUMN 6  COLUMN 7     COLUMN 8",
    ],
)
def test_signer_address_is_not_an_unconsumed_holding(metadata: str) -> None:
    # Exact outside-table metadata from the independently reviewed SEC sources.
    raw = metadata + "\n" + _table(_HEADER, _AFLAC)
    rows = parse_legacy_information_table(raw)
    assert len(rows) == 1 and rows.iloc[0].CUSIP == "001055102" and rows.iloc[0].SSHPRNAMT == 3362515
    print("SANITY: signature, contact and form metadata cannot become economic rows; exact holdings remain complete.")


@pytest.mark.parametrize(
    "location",
    [
        "before-table",
        "after-table",
        "before-header",
        "before-marker",
        "after-table-missing-cusip",
        "before-marker-missing-cusip",
        "after-table-untyped-missing-cusip",
        "before-marker-untyped-missing-cusip",
    ],
)
def test_unconsumed_holdings_cannot_complete_either_destination(location: str, filing_store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = _table(_HEADER, _AFLAC).replace("Form 13F Information Table Entry Total: 1", "")
    extra = _AMGEN
    if location.endswith("missing-cusip"):
        extra = extra.replace("031162100", "         ")
        if "untyped" not in location:
            extra = extra.replace("100,000", "100,000 SH")
    if location == "before-table":
        raw = _HEADER + "\n" + extra + "\n" + raw
    elif location.startswith("after-table"):
        raw += "\n" + _HEADER + "\n" + extra
    elif location == "before-header":
        raw = raw.replace("<TABLE>\n", "<TABLE>\n" + extra + "\n")
    else:
        raw = raw.replace(_HEADER + "\n<S>", _HEADER + "\n" + extra + "\n<S>")
    raw = "<DOCUMENT>\n<TEXT>\n" + raw + "\n</TEXT>\n</DOCUMENT>"
    filing = _Filing(_Report(None, txt=raw))
    monkeypatch.setattr(filing, "full_text_submission", lambda: raw, raising=False)
    packet = f13.read_filing_vintage(FilingStamp.of(filing, filing.cik))
    assert isinstance(packet, f13.ReadFailure) and not packet.transient
    saver = f13._Saver(SimpleNamespace(store=filing_store), {"AFL", "AMGN"}, {filing.cik}, 600)
    saver.add(f13._Read(FilingStamp.of(filing, filing.cik), packet))
    for table in (f13.Tables.sec13f_hr, f13.Tables.sec13f_manager_holdings):
        assert filing_store.row_count(table) == 0
    print(f"SANITY: economic rows {location} cannot self-certify a surviving subset or produce a completion marker in either destination.")


@pytest.mark.parametrize(
    "label", ["GRAND TOTALS 145462000 3362515", "FINAL TOTALS FOR 1 RECORDS 145462000", "S REPORT SUMMARY 1 DATA RECORDS 145462000 0 OTHER MANAGERS"]
)
def test_genuine_summary_outside_table_does_not_hide_or_add_positions(label: str) -> None:
    raw = _table(_HEADER, _AFLAC).replace("Form 13F Information Table Entry Total: 1", "")
    raw += "\n" + label
    filing = _Filing(_Report(None, txt=raw))
    manifest, rows = f13.text_filing_vintage(raw, FilingStamp.of(filing, filing.cik))
    assert manifest.iloc[0].source_row_count == 1 and pd.isna(manifest.iloc[0].declared_row_count)
    assert rows.iloc[0].cusip == "001055102" and rows.iloc[0].shares == 3362515
    print("SANITY: explicit summary/footer outside the tagged table remains non-economic; the intact missing-count book retains its exact holding.")


@pytest.mark.parametrize("title", ["Ord", "CL A", "Common", "Com New", "Custom Class", ""])
@pytest.mark.parametrize("spacing", [" ", "  "])
@pytest.mark.parametrize("location", ["outside_table", "before_marker"])
def test_missing_identifier_does_not_hide_other_supported_security_classes(title: str, spacing: str, location: str) -> None:
    extra = f"Amgen Inc.{spacing}{title}{spacing}12,000 100,000"
    raw = _table(_HEADER, _AFLAC).replace("Form 13F Information Table Entry Total: 1", "")
    raw = raw + "\n" + extra if location == "outside_table" else raw.replace("<S>", extra + "\n<S>")
    with pytest.raises(ValueError, match="Unaccounted"):
        parse_legacy_information_table(raw)
    print("SANITY: compact and fixed-column security classes remain economic evidence without a CUSIP or explicit quantity type.")


@pytest.mark.parametrize("title", ["Ord", "CL A", "Common", "Com New", "Custom Class", ""])
def test_compact_class_keeps_the_identified_source_holding(title: str) -> None:
    raw = _table(_HEADER, f"Amgen Inc. {title} 031162100 12,000 100,000 SH")
    rows = parse_legacy_information_table(raw)
    assert rows.CUSIP.tolist() == ["031162100"] and rows.VALUE.tolist() == [12000000] and rows.SSHPRNAMT.tolist() == [100000]
    assert rows.SSHPRNAMTTYPE.tolist() == ["SH"]
    print("SANITY: arbitrary compact class wording preserves exact CUSIP/value/quantity/type facts.")


@pytest.mark.parametrize("location", ["outside_table", "before_marker"])
@pytest.mark.parametrize("extra", ["12,000 100,000", "1 2 3 4 5 6 7 8"])
def test_untyped_numeric_continuation_cannot_disappear(location: str, extra: str) -> None:
    raw = _table(_HEADER, _AFLAC).replace("Form 13F Information Table Entry Total: 1", "")
    raw = raw + "\n" + extra if location == "outside_table" else raw.replace("<S>", extra + "\n<S>")
    with pytest.raises(ValueError, match="Unaccounted"):
        parse_legacy_information_table(raw)
    print("SANITY: a CUSIP-less untyped numeric continuation cannot certify a surviving one-row subset.")


@pytest.mark.parametrize("extra,value,quantity", [("12,000 100,000", 12000000, 100000), ("1 2 3 4 5 6 7 8", 1000, 2)])
def test_intact_untyped_numeric_continuation_keeps_both_economic_rows(extra: str, value: int, quantity: int) -> None:
    rows = parse_legacy_information_table(_table(_HEADER, _AFLAC, extra))
    assert rows.CUSIP.tolist() == ["001055102", "001055102"]
    assert rows.VALUE.tolist() == [145462000, value] and rows.SSHPRNAMT.tolist() == [3362515, quantity]
    print("SANITY: the same bare numeric geometry is an economic continuation inside a supported table; both exact allocations survive.")


@pytest.mark.parametrize("location", ["before_marker", "inside_table"])
def test_split_column_caption_is_metadata_only_with_its_label_row(location: str) -> None:
    caption = "Column\tColumn\tColumn\tColumn\tColumn\tColumn\tColumn\tColumn\n1\t\t2\t3\t4\t5\t6\t7\t8"
    raw = _table(_HEADER, _AFLAC)
    raw = raw.replace("<S>", caption + "\n<S>") if location == "before_marker" else raw.replace("</TABLE>", caption + "\n</TABLE>")
    rows = parse_legacy_information_table(raw)
    assert rows.CUSIP.tolist() == ["001055102"] and rows.VALUE.tolist() == [145462000] and rows.SSHPRNAMT.tolist() == [3362515]
    print("SANITY: paired column labels/numbers form one metadata caption; a bare numeric continuation remains an economic row.")


# ---- cover-total / VALUE-sum ratio rule ----------------------------------------------------------

_LEGACY_SOURCES = Path(__file__).parent / "fixtures" / "legacy_13f"
_VALUE_COLUMNS = list(f13._VALUE_BY_TYPE.values())


def _cached_source(accession: str) -> tuple[str, FilingStamp]:
    """A real SEC legacy submission and the stamp its own header declares."""
    raw = (_LEGACY_SOURCES / f"{accession}.sgml").read_text(encoding="utf-8")

    def header(label: str) -> str:
        match = re.search(rf"{label}:\s*(\S+)", raw)
        assert match, label
        return match.group(1)

    filing = _Filing(
        _Report(None, txt=raw),
        cik=header("CENTRAL INDEX KEY"),
        filing_date=header("FILED AS OF DATE"),
        period_of_report=header("CONFORMED PERIOD OF REPORT"),
        accession_number=accession,
    )
    return raw, FilingStamp.of(filing, filing.cik)


def _assert_values_unchanged(manifest: pd.DataFrame, rows: pd.DataFrame, source: list[dict[str, Any]]) -> float:
    """Raw cells survive per CUSIP as typed; USD columns move only by the filing's ValueUnits factor. Returns cover/sum."""
    expected = pd.DataFrame(source).groupby("CUSIP")["VALUE"].sum().astype(float)
    stored = rows.set_index("cusip")[["raw_" + column for column in _VALUE_COLUMNS]].sum(axis=1).sort_index()
    pd.testing.assert_series_equal(stored, expected, check_names=False, check_index_type=False, rtol=1e-12)
    factor = manifest.iloc[0].value_unit_factor
    for column in _VALUE_COLUMNS:
        assert (rows[column] == rows["raw_" + column] * factor).all(), column
    assert math.isclose(manifest.iloc[0].source_value_total, sum(row["VALUE"] for row in source), rel_tol=1e-12)
    return float(manifest.iloc[0].declared_value_total / manifest.iloc[0].source_value_total)


_SYNTHETIC_BOOK = [{"CUSIP": "001055102", "VALUE": 145462.0}, {"CUSIP": "031162100", "VALUE": 12000.0}]


@pytest.mark.parametrize(
    ("accession", "expected", "shape"),
    [
        ("0001144204-13-029112", 1.0, "cover and cells both in $1000s"),
        ("0001056868-13-000004", 1000.0, "cover in dollars, cells in $1000s"),
        ("0001568235-13-000005", 0.001, "cells in dollars, cover in $1000s"),
        ("0000950159-05-001336", 1.0, "one row above the cover count, no row equals the total"),
        ("0000902664-13-002273", 1.0, "single holding equal to the total, count matches"),
    ],
)
def test_ratio_rule_cached_sources(accession: str, expected: float, shape: str) -> None:
    raw, stamp = _cached_source(accession)
    source = _holding_rows(raw, _source_rows(raw), 1.0)[0]
    manifest, rows = f13.text_filing_vintage(raw, stamp)
    ratio = _assert_values_unchanged(manifest, rows, source)
    assert manifest.iloc[0].source_row_count == len(source)
    assert math.isclose(ratio, expected, rel_tol=0.03), ratio
    factor = manifest.iloc[0].value_unit_factor
    print(
        f"SANITY: real {accession} ({shape}) accepted at cover/sum ratio {ratio:.6g}; all {len(source)} rows kept, raw values unchanged, USD only by ValueUnits x{factor:g}."
    )


def test_ratio_16_36_cached_source_is_kept_with_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    raw, stamp = _cached_source("0001387131-13-001589")
    source = _holding_rows(raw, _source_rows(raw), 1.0)[0]
    ratio = 29161518 / sum(row["VALUE"] for row in source)
    with caplog.at_level(logging.WARNING):
        manifest, rows = f13.text_filing_vintage(raw, stamp)
        assert len(parse_legacy_information_table(raw)) == len(source)
    messages = [record.getMessage() for record in caplog.records]
    assert any("the cover is a reference only, filing kept" in message for message in messages)
    assert any("differs materially" in message for message in messages)
    assert manifest.iloc[0].declared_value_total == 29161518 and manifest.iloc[0].value_unit_factor == 1000
    print(
        f"SANITY: real 0001387131-13-001589 at cover/sum ratio {ratio:.4g} is kept with a WARNING on the live and the verified path; "
        f"cover stored as filed, {len(rows)} rows, units x1000 by implied price."
    )


def _synthetic_double_counted(groups: int) -> str:
    """SYNTHETIC: `groups` x (Aflac, Amgen) and an unlabelled table-total line read as a continuation."""
    value, shares = 157462 * groups, (3362515 + 100000) * groups
    raw = _table(_HEADER, *([_AFLAC, _AMGEN] * groups), " " * 45 + f"{value:,}  {shares:,}")
    raw = raw.replace(f"Entry Total: {2 * groups + 1}", f"Entry Total: {2 * groups}")
    return f"[X] 13F HOLDINGS REPORT\nForm 13F Information Table Value Total: ${value:,}\n" + raw


@pytest.mark.parametrize("groups", [1, 10])
def test_synthetic_ratio_2_total_line_is_dropped_without_changing_values(groups: int) -> None:
    raw = _synthetic_double_counted(groups)
    parsed = _holding_rows(raw, _source_rows(raw), 1.0)[0]
    assert len(parsed) == 2 * groups + 1 and sum(row["VALUE"] for row in parsed) == 2 * 157462 * groups
    for verify_values in (True, False):
        rows = parse_legacy_information_table(raw, value_factor=1.0, verify_values=verify_values)
        assert rows.VALUE.tolist() == [145462, 12000] * groups and rows.SSHPRNAMT.tolist() == [3362515, 100000] * groups
    filing = _Filing(_Report(None, txt=raw))
    manifest, rows = f13.text_filing_vintage(raw, FilingStamp.of(filing, filing.cik))
    ratio = _assert_values_unchanged(manifest, rows, _SYNTHETIC_BOOK * groups)
    assert manifest.iloc[0].source_row_count == manifest.iloc[0].declared_row_count == 2 * groups and ratio == 1.0
    print(
        f"SANITY: SYNTHETIC ratio-2 book ({2 * groups}+1 rows, ratio 2 before the drop) keeps its {2 * groups} declared rows with unchanged values."
    )


def test_synthetic_ratio_2_with_two_rows_equal_to_the_total_is_rejected() -> None:
    raw = _table(_HEADER, _AFLAC, _AMGEN, " " * 45 + "145,462  3,362,515").replace("Entry Total: 3", "Entry Total: 2")
    raw = "[X] 13F HOLDINGS REPORT\nForm 13F Information Table Value Total: $145,462\n" + raw
    assert [row["VALUE"] for row in _holding_rows(raw, _source_rows(raw), 1.0)[0]] == [145462, 12000, 145462]
    with pytest.raises(ValueError, match="does not reconcile"):
        parse_legacy_information_table(raw, value_factor=1.0, verify_values=False)
    print("SANITY: SYNTHETIC extra row is kept when two rows equal the cover total, so the count mismatch still rejects the book.")


@pytest.mark.parametrize(("ratio", "aligned"), [(2.0, False), (0.5, False), (3.0, False), (1.5, False), (10.0, True), (100.0, True), (1000.0, True)])
def test_synthetic_cover_ratios_without_a_total_row(caplog: pytest.LogCaptureFixture, ratio: float, aligned: bool) -> None:
    raw = f"[X] 13F HOLDINGS REPORT\nForm 13F Information Table Value Total: {int(157462 * ratio):,}\n" + _table(_HEADER, _AFLAC, _AMGEN)
    filing = _Filing(_Report(None, txt=raw))
    with caplog.at_level(logging.WARNING):
        manifest, rows = f13.text_filing_vintage(raw, FilingStamp.of(filing, filing.cik))
    assert _assert_values_unchanged(manifest, rows, _SYNTHETIC_BOOK) == ratio
    warned = any("the cover is a reference only, filing kept" in record.getMessage() for record in caplog.records)
    assert warned is not aligned
    verdict = "aligned (power of ten, A-1)" if aligned else "kept with a WARNING"
    print(f"SANITY: SYNTHETIC cover/sum ratio {ratio:g} with rows == declared is {verdict}; no row is dropped.")


@pytest.mark.parametrize(
    ("accession", "cusip", "factor", "band", "shape"),
    [
        ("0001568235-13-000005", "166764100", 1.0, (100.0, 140.0), "Chevron; cells in dollars, cover in $1000s; CVX closed near $119 on 2013-03-28"),
        ("0001056868-13-000004", "031162100", 1000.0, (90.0, 115.0), "Amgen; cover in dollars, cells (x$1000); AMGN closed near $102 on 2013-03-28"),
        ("0001144204-13-029112", "143658300", 1000.0, (30.0, 40.0), "Carnival; cover and cells (x1000); CCL closed near $34 on 2013-03-28"),
    ],
)
def test_cover_ratio_sets_the_cell_unit_on_real_sources(accession: str, cusip: str, factor: float, band: tuple[float, float], shape: str) -> None:
    raw, stamp = _cached_source(accession)
    source = _holding_rows(raw, _source_rows(raw), 1.0)[0]
    manifest, rows = f13.text_filing_vintage(raw, stamp)
    ratio = _assert_values_unchanged(manifest, rows, source)
    assert manifest.iloc[0].value_unit_factor == factor, (ratio, manifest.iloc[0].value_unit_factor)
    holding = rows.set_index("cusip").loc[cusip]
    price = holding.value_usd / holding.shares
    assert band[0] < price < band[1], price
    print(
        f"SANITY: real {accession} at cover/sum ratio {ratio:.6g} stores cells x{factor:g}; {cusip} value_usd {holding.value_usd:,.0f} / "
        f"{holding.shares:,.0f} sh = ${price:.2f} ({shape})."
    )


@pytest.mark.parametrize(
    ("cover", "header", "factor", "shape"),
    [
        ("$157,462", _HEADER, 1000.0, "ratio 1 and 2 share rows: the few-rows implied price decides"),
        ("$1,574,620", _HEADER, 1000.0, "ratio 10 is no unit evidence: the few-rows implied price decides"),
        ("$157,462,000", _HEADER, 1000.0, "ratio 1000, cover in dollars over cells in $1000s"),
        ("$157,462,000,000", _HEADER, 1e6, "ratio 1e6, cover in dollars over cells in $millions"),
        ("$157 (thousands)", _HEADER, 1.0, "ratio 0.001, cover in $1000s over cells in dollars"),
        ("$157.462 Million", _HEADER, 1000.0, "Million cover over cells in $1000s"),
        ("$0.157462 Million", _HEADER, 1.0, "Million cover over cells in dollars"),
        (
            "$157 (thousands)",
            _HEADER.replace("VALUE    ", "VALUE(x$1000)"),
            1000.0,
            "stated cover against an (x$1000) caption: conflict, the few-rows price decides",
        ),
    ],
)
def test_synthetic_cover_ratio_sets_the_cell_unit(cover: str, header: str, factor: float | None, shape: str) -> None:
    raw = f"[X] 13F HOLDINGS REPORT\nForm 13F Information Table Value Total: {cover}\n" + _table(header, _AFLAC, _AMGEN)
    filing = _Filing(_Report(None, txt=raw))
    manifest, rows = f13.text_filing_vintage(raw, FilingStamp.of(filing, filing.cik))
    assert sorted(rows.raw_value_usd.tolist()) == [12000.0, 145462.0]
    stored = manifest.iloc[0].value_unit_factor
    if factor is None:
        assert math.isnan(stored) and rows.value_usd.isna().all(), (stored, rows.value_usd.tolist())
    else:
        assert stored == factor and (rows.value_usd == rows.raw_value_usd * factor).all(), stored
    print(f"SANITY: SYNTHETIC cover {cover!r} ({shape}) -> cell factor {stored:g}; raw cells unchanged.")


@pytest.mark.parametrize(
    ("cover_line", "factor", "shape"),
    [
        ("Value Total: $157.462 (in millions)", 1000.0, "(in millions) after the number, cells in $1000s"),
        ("Value Total (in millions): $157.462", 1000.0, "(in millions) in the caption, cells in $1000s"),
        ("Value Total: $0.157462 (in millions)", 1.0, "(in millions) after the number, cells in dollars"),
        ("Value Total: $157.462 Million", 1000.0, "Million word, cells in $1000s"),
        ("Value Total: $0.157462 in billions", 1000.0, "in billions, cells in $1000s"),
        ("Value Total: $0.000157462 in billions", 1.0, "in billions, cells in dollars"),
        ("Value Total: $157,462 (in thousands)", 1000.0, "(in thousands), cells in $1000s"),
        ("Value Total: $157 (in thousands)", 1.0, "(in thousands), cells in dollars"),
        ("Value Total: $157,462 (000s)", 1000.0, "(000s), cells in $1000s"),
        ("Value Total: $157,462,000 dollars", 1000.0, "dollars, cells in $1000s"),
        ("Value Total: $157", 1.0, "unlabelled ratio 0.001"),
        ("Value Total: $157,462,000", 1000.0, "unlabelled ratio 1000"),
        ("Value Total: $157,462", 1000.0, "unlabelled ratio 1: the few-rows implied price decides"),
        ("Value Total: $157,462,000,000 (in millions)", 1000.0, "millions label at ratio 1e6 implies cells x1e9: the few-rows price decides"),
    ],
)
def test_stated_cover_unit_wins_over_the_cover_ratio(cover_line: str, factor: float | None, shape: str) -> None:
    raw = f"[X] 13F HOLDINGS REPORT\nForm 13F Information Table {cover_line}\n" + _table(_HEADER, _AFLAC, _AMGEN)
    filing = _Filing(_Report(None, txt=raw))
    manifest, rows = f13.text_filing_vintage(raw, FilingStamp.of(filing, filing.cik))
    assert sorted(rows.raw_value_usd.tolist()) == [12000.0, 145462.0]
    aflac = rows.set_index("cusip").loc["001055102"]
    stored = manifest.iloc[0].value_unit_factor
    if factor is None:
        assert math.isnan(stored) and math.isnan(aflac.value_usd), (stored, aflac.value_usd)
        assert manifest.iloc[0].value_unit_evidence.endswith("conflicting_cover_total"), manifest.iloc[0].value_unit_evidence
    else:
        assert stored == factor and aflac.value_usd == 145462.0 * factor, (stored, aflac.value_usd)
    print(f"SANITY: SYNTHETIC {cover_line!r} ({shape}) -> cell factor {stored:g}; Aflac value_usd {aflac.value_usd:,.0f} from raw 145,462.")


@pytest.mark.parametrize(
    ("accession", "outcome"),
    [
        ("0000950159-05-001336", "ok"),
        ("0001082481-05-000006", "cover declares an empty book"),
    ],
)
def test_legacy_read_makes_no_homepage_index_request(monkeypatch: pytest.MonkeyPatch, accession: str, outcome: str) -> None:
    """Real 2005 submissions as EdgarTools `Filing`s: no primary XML, so `Filing.xml()` would fetch the
    `-index.html` homepage, which EDGAR answers 403. The read must come from the submission alone."""
    requested: list[str] = []

    def _blocked(_client: httpx.Client, request: httpx.Request, **_kw: Any) -> httpx.Response:
        requested.append(str(request.url))
        raise RuntimeError(f"network blocked: {request.url}")

    monkeypatch.setenv("EDGAR_IDENTITY", "Test Runner test@example.com")  # so a request reaches the blocked send
    monkeypatch.setattr(httpx.Client, "send", _blocked)
    filing = Filing.from_sgml(_LEGACY_SOURCES / f"{accession}.sgml")
    assert not filing.sgml().xml()

    packet = f13.read_filing_vintage(FilingStamp.of(filing, ""))

    assert not [url for url in requested if url.endswith("-index.html")], requested
    assert requested == [], requested
    if outcome == "ok":
        assert not isinstance(packet, f13.ReadFailure), packet
        manifest, holdings = packet
        assert manifest.iloc[0].parse_status == "complete" and len(holdings) > 0
        result = f"complete, {len(holdings)} holdings"
    else:
        assert isinstance(packet, f13.ReadFailure) and not packet.transient and outcome in packet.reason, packet
        result = f"deterministic failure {packet.reason[:60]!r}"
    print(f"SANITY: real legacy {accession} read from its SGML alone -> {result}; 0 -index.html requests, 0 network sends.")


def _sample_source(accession: str, cik: str, filed: str, period: str) -> tuple[str, FilingStamp]:
    """A real SEC information-table text as `read_filing_vintage` hands it over, with the sample's own stamp."""
    raw = (_LEGACY_SOURCES / f"{accession}.sgml").read_text(encoding="utf-8")
    filing = _Filing(_Report(None, txt=raw), cik=cik, filing_date=filed, period_of_report=period, accession_number=accession)
    return raw, FilingStamp.of(filing, filing.cik)


@pytest.mark.parametrize(
    ("accession", "cik", "filed", "period", "rows", "raw_total", "checks", "shape"),
    [
        (
            "0000919574-08-002711",
            "0001387674",
            "2008-05-02",
            "2008-03-31",
            52,
            19012,
            {"037833100": (140.0, 147.0), "20854P109": (66.0, 72.0)},
            "a 'SK 03653 0001 874432' law-firm document number below the table; AAPL near $143.50 and CNX near $69 on 2008-03-31",
        ),
        (
            "0001193125-12-233096",
            "0001166309",
            "2012-05-15",
            "2012-03-31",
            51,
            1441565,
            {"40412C101": (23.0, 26.5), "911363109": (40.0, 46.0)},
            "a headerless second page whose <C> markers sit one column right; HCA near $24.7 and URI near $42.9 on 2012-03-30",
        ),
    ],
)
def test_sample_legacy_sources_parse_and_reconcile(
    accession: str, cik: str, filed: str, period: str, rows: int, raw_total: int, checks: dict[str, tuple[float, float]], shape: str
) -> None:
    raw, stamp = _sample_source(accession, cik, filed, period)
    manifest, book = f13.text_filing_vintage(raw, stamp)
    header = manifest.iloc[0]
    assert header.source_row_count == header.declared_row_count == rows, (header.source_row_count, header.declared_row_count)
    assert header.source_value_total == header.declared_value_total == raw_total, (header.source_value_total, header.declared_value_total)
    assert header.value_unit_factor == 1000.0
    prices = {}
    for cusip, (low, high) in checks.items():
        holding = book.set_index("cusip").loc[cusip]
        prices[cusip] = holding.value_usd / holding.shares
        assert low < prices[cusip] < high, (cusip, prices[cusip])
    quoted = ", ".join(f"{cusip} ${price:.2f}" for cusip, price in prices.items())
    print(f"SANITY: real {accession} ({shape}) parses {rows}/{rows} rows, raw total {raw_total:,} equals the cover; {quoted}.")


@pytest.mark.parametrize(
    ("accession", "cik", "filed", "period", "reason", "shape"),
    [
        ("0000892712-01-500072", "0001123570", "2001-05-14", "2001-03-31", "Invalid source CUSIP '999000020'", "Flextronics under an internal code"),
        (
            "0001039565-01-500004",
            "0001039565",
            "2001-04-27",
            "2001-03-31",
            "Invalid source CUSIP '022609107'",
            "American Home Products typo of 026609107",
        ),
        (
            "0001137065-01-500009",
            "0001137065",
            "2001-04-26",
            "1999-03-31",
            "Holding line without a source CUSIP: Blake Barrows & Brown          COM                             348      208 SH       SOLE"
            "                               ; no cover count and total to reconcile the book without 2 such line(s)",
            "2 blank CUSIP cells (Blake Barrows & Brown, Promus Hotel) and no cover count or total to reconcile without them",
        ),
        (
            "0000950144-08-003876",
            "0001409775",
            "2008-05-09",
            "2008-03-31",
            "Invalid source CUSIP '869004995'",
            "its 4 blank CUSIP cells are set aside; the next line, a present CUSIP failing its check digit, still fails",
        ),
        (
            "0000948985-12-000005",
            "0000948985",
            "2012-05-15",
            "2012-03-31",
            "Unreadable source value/shares for 168628105: Chief Consolidated Mining",
            "a '?' VALUE cell under its own CUSIP, not Chevron's 166764100 above it",
        ),
    ],
)
def test_sample_legacy_sources_keep_failing_with_the_right_reason(accession: str, cik: str, filed: str, period: str, reason: str, shape: str) -> None:
    raw, stamp = _sample_source(accession, cik, filed, period)
    with pytest.raises(ValueError, match=re.escape(reason)):
        f13.text_filing_vintage(raw, stamp)
    print(f"SANITY: real {accession} ({shape}) still fails loudly: {reason!r}.")


_CREF = "0001007399-13-000002"


def test_blank_cusip_line_is_excluded_when_the_rest_reconciles(caplog: pytest.LogCaptureFixture) -> None:
    """Real 2013Q1 book: a TIAA-CREF annuity account line has no CUSIP. The cover counts it (300 lines);
    the other 299 lines and their VALUE sum reconcile with the cover, so the book is stored without it."""
    raw, stamp = _cached_source(_CREF)
    with caplog.at_level(logging.WARNING, logger=f13.__name__):
        manifest, book = f13.text_filing_vintage(raw, stamp)
    header = manifest.iloc[0]
    assert (header.source_row_count, header.declared_row_count) == (299, 300)
    assert (header.source_value_total, header.declared_value_total, header.value_unit_factor) == (654519.0, 654689.0, 1000.0)
    assert len(book) == 281 and book.raw_value_usd.sum() == 654519.0
    assert not book.issuer_name.str.contains("CREF").any()
    warnings = [r.getMessage() for r in caplog.records if "without a source CUSIP" in r.getMessage()]
    assert warnings == [
        f"13F {_CREF}: excluded 1 holding line(s) without a source CUSIP, VALUE 169.0 in source cells; the other 299 row(s) reconcile with the cover"
    ]
    positions = book.set_index("cusip")
    prices = {cusip: positions.loc[cusip].value_usd / positions.loc[cusip].shares for cusip in ("88579Y101", "002824100")}
    assert 100.0 < prices["88579Y101"] < 112.0 and 33.0 < prices["002824100"] < 38.0, prices
    print(
        f"SANITY: real {_CREF} stores 299/300 lines (281 positions), raw total 654,519 vs cover 654,689 (excluded 169 + 1 rounding); "
        f"one WARNING names 1 line, VALUE 169; MMM ${prices['88579Y101']:.2f} (closed near $106.3), "
        f"ABT ${prices['002824100']:.2f} (near $35.3) on 2013-03-28."
    )


def test_synthetic_blank_cusip_value_that_breaks_the_cover_still_fails() -> None:
    """SYNTHETIC from the real CREF source: the blank line's VALUE is raised to 169,000 and the cover with it,
    so the 299 kept lines hold 79% of the cover total, outside the existing source band."""
    raw, stamp = _cached_source(_CREF)
    line = next(line for line in raw.splitlines() if line.startswith("CREF Growth"))
    raw = raw.replace(line, line.replace("     169    15157", "  169000    15157")).replace("Value Total: $ 654689", "Value Total: $ 823520")
    with pytest.raises(ValueError, match=r"^Holding line without a source CUSIP: CREF Growth .*does not reconcile: 13F source VALUE total"):
        f13.text_filing_vintage(raw, stamp)
    print("SANITY: SYNTHETIC CREF line at 169,000 of an 823,520 cover leaves 654,519 (79%) kept, so the book still fails loudly.")


@pytest.mark.parametrize(
    ("line", "value", "shape"),
    [
        (
            "Altium FX LTD             COMMON STOCK                    5     12000   SH        SOLE       01, 02, 04     12000      0      0",
            5,
            "empty cell",
        ),
        (
            "MetLife, Inc.             COMMON STOCK   (blank)       1475    24,477   SH        SOLE       01, 02, 03     24477      0      0",
            1475,
            "(blank)",
        ),
        (
            "   Growth                 COMMON STOCK   (blank)        928    12,837   SH        SOLE       01, 02, 03     12837      0      0",
            928,
            "wrapped name",
        ),
        (
            "Target Corporation        COMMON STOCK   8.7612E+110   2278    44,958   SH        SOLE       01, 02, 03     44958      0      0",
            None,
            "garbled CUSIP",
        ),
        ("Column Totals                            4,018,843 84,153,226                                  72,239,393    -", None, "column totals"),
        ("ABBOTT LABS              Common Stock      2824100       97,520.00      2,300      2,300     -       ALL      2,300", None, "short CUSIP"),
    ],
)
def test_blank_cusip_cell_is_told_from_other_cusipless_lines(line: str, value: int | None, shape: str) -> None:
    header = "    NAME OF ISSUER      TITLE OF CLASS     CUSIP    (X1000)  PRIN AMT  PRIN  CALL DISCRETION  MANAGERS     SOLE   SHARED  NONE"
    found = _blank_cusip_value(header, line)
    assert found == value, (found, value)
    print(f"SANITY: {shape} line -> blank-CUSIP VALUE {found}; only an empty or (blank) cell with VALUE, shares and SH/PRN is set aside.")


def test_altium_blank_lines_are_set_aside_and_its_invalid_cusip_still_fails() -> None:
    """Real 2008Q1 book (cover 917 = every line, blanks included): the 4 blank-CUSIP lines are recognised;
    the first present-but-invalid CUSIP, 869004995, still fails the book."""
    raw, stamp = _sample_source("0000950144-08-003876", "0001409775", "2008-05-09", "2008-03-31")
    header = next(line for line in raw.splitlines() if "CUSIP" in line and "ISSUER" in line)
    blanks = [_blank_cusip_value(header, line) for line in raw.splitlines() if "(blank)" in line or line.startswith("Altium FX")]
    assert blanks == [5, 928, 1475, 1263], blanks
    with pytest.raises(ValueError, match=re.escape("Invalid source CUSIP '869004995'")):
        f13.text_filing_vintage(raw, stamp)
    print("SANITY: real 0000950144-08-003876 sets aside 4 blank-CUSIP lines (VALUE 3,671 of a 1,962,677 cover) but still fails on 869004995.")


@pytest.mark.parametrize(
    ("accession", "cik", "filed", "form", "period", "lines", "checks", "shape"),
    [
        (
            "0001087767-08-000003",
            "0001087767",
            "2008-05-14",
            "13F-HR",
            "2008-03-31",
            66,
            {"291011104": (49.0, 54.0), "30231G102": (82.0, 87.0)},
            "voting cells wrapped onto their own 'X 42,540 2,106' line; EMR near $51.5 and XOM near $84.6 on 2008-03-31",
        ),
        (
            "0000950123-01-001193",
            "0000020171",
            "2001-02-13",
            "13F-HR",
            "2000-12-31",
            192,
            {"369550108": (75.0, 81.0), "071813109": (85.0, 91.0)},
            "a '01  28 - 6866  Chubb Asset Managers' other-manager table before the holdings; GD near $78 and BAX near $88.3 on 2000-12-29",
        ),
        (
            "0001384042-10-000008",
            "0001384042",
            "2010-11-12",
            "13F-HR",
            "2010-09-30",
            68,
            {"693718108": (46.0, 50.0), "87612E106": (51.0, 56.0)},
            "an 'ITEM 1  ITEM 2 ... ITEM 8' caption row inside the table; PCAR near $48.2 and TGT near $53.4 on 2010-09-30",
        ),
        (
            "0001015308-09-000004",
            "0001015308",
            "2009-10-30",
            "13F-HR",
            "2009-09-30",
            428,
            {"020002101": (29.0, 32.0), "46625H100": (42.0, 46.0)},
            "a 'LINE COUNT: 428  $ 7,100,040' total row at the table foot; ALL near $30.6 and JPM near $43.8 on 2009-09-30",
        ),
        (
            "0001169232-06-000117",
            "0001135307",
            "2006-01-13",
            "13F-HR/A",
            "2005-12-31",
            20,
            {"747525103": (41.0, 45.0), "615369105": (59.0, 64.0)},
            "spaced CUSIPs like '025816 10 9' and a '|X|' amendment box; QCOM near $43.1 and MCO near $61.4 on 2005-12-30",
        ),
        (
            "0001144204-07-024525",
            "0001283718",
            "2007-05-11",
            "13F-HR/A",
            "2007-03-31",
            57,
            {"30231G102": (73.0, 78.0), "060505104": (49.0, 53.0)},
            "a '|X| is a restatement.' amendment box; XOM near $75.5 and BAC near $51.0 on 2007-03-30",
        ),
    ],
)
def test_yearly_sample_legacy_sources_parse_and_reconcile(
    accession: str, cik: str, filed: str, form: str, period: str, lines: int, checks: dict[str, tuple[float, float]], shape: str
) -> None:
    """Real books the yearly sample could not read: each source line is a holding, the count equals the cover and
    two positions sit at their period-end close at the chosen x1000."""
    raw = (_LEGACY_SOURCES / f"{accession}.sgml").read_text(encoding="utf-8")
    filing = _Filing(_Report(None, txt=raw), cik=cik, filing_date=filed, period_of_report=period, form=form, accession_number=accession)
    manifest, book = f13.text_filing_vintage(raw, FilingStamp.of(filing, filing.cik))
    header = manifest.iloc[0]
    assert header.source_row_count == header.declared_row_count == lines, (header.source_row_count, header.declared_row_count)
    assert header.value_unit_factor == 1000.0
    prices = {}
    for cusip, (low, high) in checks.items():
        holding = book[book.cusip == cusip]
        prices[cusip] = holding.value_usd.sum() / holding.shares.sum()
        assert low < prices[cusip] < high, (cusip, prices[cusip])
    quoted = ", ".join(f"{cusip} ${price:.2f}" for cusip, price in prices.items())
    print(f"SANITY: real {accession} ({shape}) parses {lines}/{lines} lines into {len(book)} positions at x1000; {quoted}.")
