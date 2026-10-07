"""
test_13f_legacy_fallback.py (tests/data_extract/institutionals/test_13f_legacy_fallback.py)
--------------------------------------------------------------------------------------------
Known-truth SEC text lines for the source-checked legacy 13F parser, and the `_read_filing` hook:
clean XML is untouched, a malformed text parse is replaced by verified source rows, an
unverifiable text book is a deterministic read failure, and no F-001 garbage shape is returned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import pytest

from src.data_extract.utils.common.edgar_driver import FilingStamp
from src.data_extract.utils.institutionals import fetch_13f as f13
from src.data_extract.utils.institutionals.legacy_13f_fallback import needs_legacy_fallback, parse_legacy_information_table

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


def test_legacy_text_without_summary_count_fails_closed() -> None:
    raw = _table(_HEADER, _AFLAC).replace("Form 13F Information Table Entry Total: 1\n", "")
    assert needs_legacy_fallback(raw, pd.DataFrame([{"CUSIP": "001055102"}]))
    with pytest.raises(ValueError, match="entry count is missing"):
        parse_legacy_information_table(raw)
    print("\n=== SANITY: a text book with no SEC Summary Page count cannot be certified. Validated.")


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


def test_summary_value_discrepancy_rejects_plausible_count() -> None:
    aflac = "Aflac, Inc.             Com         001055102 100,000  2,000,000   DEFINED      1   SOLE"
    raw = _table(_HEADER, aflac) + "\nForm 13F Information Table Value Total: 100,500"
    with pytest.raises(ValueError, match="value total differs"):
        parse_legacy_information_table(raw)
    print("\n=== SANITY: a matching entry count cannot admit a book 0.5% off its SEC value total. Validated.")


def test_eight_character_source_cusip_restores_only_valid_leading_zero() -> None:
    raw = _table(
        "        NAME OF ISSUER         TITLE OF CLASS     CUSIP   (x$1000) PRN AMT  PRN CALL DSCRETN",
        "American Express                COM             25816109     69700  1204627 SH       SOLE",
    )
    rows = parse_legacy_information_table(raw)
    assert list(rows["CUSIP"]) == ["025816109"]
    assert list(rows["VALUE"]) == [69_700_000]
    print("\n=== SANITY: an omitted leading CUSIP zero is restored only when the check digit verifies. Validated.")


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
