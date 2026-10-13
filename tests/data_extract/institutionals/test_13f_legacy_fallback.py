"""
test_13f_legacy_fallback.py (tests/data_extract/institutionals/test_13f_legacy_fallback.py)
--------------------------------------------------------------------------------------------
Known-truth SEC text lines for the source-checked legacy 13F parser, and the `_read_filing` hook:
clean XML is untouched, a malformed text parse is replaced by verified source rows, an
unverifiable text book is a deterministic read failure, and no F-001 garbage shape is returned.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from src.data_extract.utils.common.edgar_driver import FilingStamp
from src.data_extract.utils.institutionals import fetch_13f as f13
from src.data_extract.utils.institutionals.legacy_13f_fallback import needs_legacy_fallback, parse_legacy_information_table
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


def test_legacy_edgar_preserves_accession_and_reconciled_book(sqlite_store: Any) -> None:
    raw = "[X] 13F HOLDINGS REPORT\nForm 13F Information Table Value Total: $157,462\n" + _table(_HEADER, _AFLAC, _AMGEN)
    filing = _Filing(_Report(None, txt=raw))
    packet = f13.read_filing_vintage(FilingStamp.of(filing, filing.cik))
    assert not isinstance(packet, f13.ReadFailure), packet
    manifest, holdings = packet
    assert manifest.iloc[0].accession_number == filing.accession_number
    assert len(holdings) == 2 and holdings.value_usd.sum() == 157462000
    sqlite_store.save_13f_filing(manifest, holdings)
    assert len(sqlite_store.load(f13.Tables.sec13f_manager_holdings)) == 2
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


def test_summary_value_discrepancy_uses_approved_shared_tolerance() -> None:
    aflac = "Aflac, Inc.             Com         001055102 100,000  2,000,000   DEFINED      1   SOLE"
    raw = _table(_HEADER, aflac) + "\nForm 13F Information Table Value Total: 100,500"
    assert parse_legacy_information_table(raw).iloc[0].VALUE == 100000000
    with pytest.raises(ValueError, match="value total differs"):
        parse_legacy_information_table(raw.replace("100,500", "150,000"))
    print("SANITY: modest cover discrepancy is accepted under the shared rule; an unexplained material mismatch still fails.")


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


def test_explicit_billion_cover_uses_displayed_precision() -> None:
    # SEC 0001036325-12-000004: $45.864729bn rounds to its $45.9bn cover.
    raw = _table(_HEADER, _AFLAC.replace("145,462", "45,864,729"))
    raw += "\nForm 13F Information Table Value Total: $45.9 Billion"
    assert parse_legacy_information_table(raw).iloc[0]["VALUE"] == 45_864_729_000
    with pytest.raises(ValueError, match="value total differs"):
        parse_legacy_information_table(raw.replace("45,864,729", "45,840,000"))
    print("\n=== SANITY: explicit Billion precision admits $45.864729bn and rejects $45.84bn. Validated.")


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
    assert rows.iloc[0].shares == 0 and rows.iloc[0].value_usd == 145462000
    print("SANITY: separate legacy cover/table reconciles; zero quantity and positive source value survive for inverse P1 price fallback.")


@pytest.mark.parametrize("missing", ["count", "value", "both"])
def test_intact_legacy_missing_summary_is_truthful_and_replayable(sqlite_store: Any, missing: str) -> None:
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
        assert sqlite_store.save_13f_filing(manifest, positions, table=table) == 1
        assert sqlite_store.save_13f_filing(manifest, positions, table=table) == 0
        marker = sqlite_store.load(table, where={"cusip": "_empty"}, markers=True)
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


def test_missing_declarations_require_source_accounting_at_writer(sqlite_store: Any) -> None:
    raw = _table(_HEADER, _AFLAC).replace("Form 13F Information Table Entry Total: 1", "")
    filing = _Filing(_Report(None, txt=raw))
    manifest, rows = f13.text_filing_vintage(raw, FilingStamp.of(filing, filing.cik))
    unaccounted = manifest.assign(value_unit_evidence="filing_date_default")
    for table in (f13.Tables.sec13f_hr, f13.Tables.sec13f_manager_holdings):
        positions = rows.assign(ticker="AFL") if table == f13.Tables.sec13f_hr else rows
        with pytest.raises(ValueError, match="does not reconcile"):
            sqlite_store.save_13f_filing(unaccounted, positions, table=table)
        assert not sqlite_store.exists(table)
    print("SANITY: nullable cover fields alone cannot waive source validation or create completion markers.")


def test_missing_declarations_replay_null_types_and_preserve_conflicts(sqlite_store: Any) -> None:
    raw = _table(_HEADER, _AFLAC).replace("Form 13F Information Table Entry Total: 1", "")
    filing = _Filing(_Report(None, txt=raw))
    manifest, rows = f13.text_filing_vintage(raw, FilingStamp.of(filing, filing.cik))
    for table in (f13.Tables.sec13f_hr, f13.Tables.sec13f_manager_holdings):
        positions = rows.assign(ticker="AFL") if table == f13.Tables.sec13f_hr else rows
        assert sqlite_store.save_13f_filing(manifest, positions, table=table) == 1
        for missing in (None, float("nan"), pd.NA):
            replay = manifest.assign(declared_row_count=missing, declared_value_total=missing)
            assert sqlite_store.save_13f_filing(replay, positions, table=table) == 0
        with pytest.raises(ValueError, match="conflict"):
            sqlite_store.save_13f_filing(manifest.assign(declared_row_count=1), positions, table=table)
    print("SANITY: SQL null, NaN and pandas NA replay identically; adding an observed declaration conflicts with immutable source metadata.")


def test_private_identifier_and_inferred_scope_reach_public_books(sqlite_store: Any) -> None:
    raw = _table(_HEADER, "NAMIC INSURANCE COMPANY INC  COM 62989*105 21 93 SH SOLE 93 0 0")
    filing = _Filing(_Report(None, txt=raw))
    manifest, rows = f13.text_filing_vintage(raw, FilingStamp.of(filing, filing.cik))
    assert rows.iloc[0].cusip == "62989*105" and rows.iloc[0].shares == 93
    assert manifest.iloc[0].report_type == f13.SEC_13F_INFERRED_HOLDINGS_REPORT
    calendar = pd.bdate_range("2010-03-01", "2010-06-01")
    for table in (f13.Tables.sec13f_hr, f13.Tables.sec13f_manager_holdings):
        positions = rows.assign(ticker=table.empty_marker[1]) if table == f13.Tables.sec13f_hr else rows
        sqlite_store.save_13f_filing(manifest, positions, table=table)
        saved = sqlite_store.load(table)
        markers = sqlite_store.load(table, where={"cusip": table.empty_marker[1]}, markers=True)
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
def test_next_line_entry_count_is_preserved_in_both_destinations(count: str, sqlite_store: Any) -> None:
    raw = _table(_HEADER, _AFLAC, _AMGEN).replace("Entry Total: 2", "Entry Total" + count)
    filing = _Filing(_Report(None, txt=raw))
    manifest, rows = f13.text_filing_vintage(raw, FilingStamp.of(filing, filing.cik))
    assert manifest.iloc[0].declared_row_count == 2
    assert manifest.iloc[0].source_row_count == 2
    for table in (f13.Tables.sec13f_hr, f13.Tables.sec13f_manager_holdings):
        positions = rows.assign(ticker=["AFL", "AMGN"]) if table == f13.Tables.sec13f_hr else rows
        assert sqlite_store.save_13f_filing(manifest, positions, table=table) == 2
        marker = sqlite_store.load(table, where={"cusip": "_empty"}, markers=True)
        assert marker.iloc[0].declared_row_count == 2
        assert sqlite_store.save_13f_filing(manifest, positions, table=table) == 0
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
def test_unconsumed_holdings_cannot_complete_either_destination(location: str, sqlite_store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
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
    saver = f13._Saver(SimpleNamespace(store=sqlite_store), {"AFL", "AMGN"}, {filing.cik}, 600)
    saver.add(f13._Read(FilingStamp.of(filing, filing.cik), packet))
    for table in (f13.Tables.sec13f_hr, f13.Tables.sec13f_manager_holdings):
        assert not sqlite_store.exists(table)
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


@pytest.mark.parametrize("title", ["Ord", "CL A"])
@pytest.mark.parametrize("spacing", [" ", "  "])
@pytest.mark.parametrize("location", ["outside_table", "before_marker"])
def test_missing_identifier_does_not_hide_other_supported_security_classes(title: str, spacing: str, location: str) -> None:
    extra = f"Amgen Inc.{spacing}{title}{spacing}12,000 100,000"
    raw = _table(_HEADER, _AFLAC).replace("Form 13F Information Table Entry Total: 1", "")
    raw = raw + "\n" + extra if location == "outside_table" else raw.replace("<S>", extra + "\n<S>")
    with pytest.raises(ValueError, match="Unaccounted"):
        parse_legacy_information_table(raw)
    print("SANITY: compact and fixed-column security classes remain economic evidence without a CUSIP or explicit quantity type.")


@pytest.mark.parametrize("title", ["Ord", "CL A"])
def test_compact_class_keeps_the_identified_source_holding(title: str) -> None:
    raw = _table(_HEADER, f"Amgen Inc. {title} 031162100 12,000 100,000 SH")
    rows = parse_legacy_information_table(raw)
    assert rows.CUSIP.tolist() == ["031162100"] and rows.VALUE.tolist() == [12000000] and rows.SSHPRNAMT.tolist() == [100000]
    assert rows.NAMEOFISSUER.tolist() == ["Amgen Inc."] and rows.TITLEOFCLASS.tolist() == [title]
    print("SANITY: compact source class belongs to its issuer and preserves exact CUSIP/value/quantity facts.")
