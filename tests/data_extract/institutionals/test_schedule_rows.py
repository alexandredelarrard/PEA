"""
test_schedule_rows.py (tests/data_extract/institutionals/test_schedule_rows.py)
------------------------------------------------------------------------------
The shared Schedule 13D/13G row builder and its issuer-guarded walk, driven through both form
specs with known-truth fake filings. Each test pins one rule the two forms share or one stored
divergence they must keep (13D keeps `cusip=''`, has no header CIK backfill, parses its event
date with `pd.Timestamp`; 13G normalises blanks to None and backfills CIKs from the header).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from src.data_extract.utils.common.edgar_driver import EdgarScope, FilingStamp
from src.data_extract.utils.institutionals import fetch_13d_edgar
from src.data_extract.utils.institutionals.fetch_13d_edgar import SCHEDULE_13D, SEC_13D_FETCH
from src.data_extract.utils.institutionals.fetch_13g_edgar import SCHEDULE_13G, SEC_13G_FETCH
from src.data_extract.utils.institutionals.schedule_rows import SCHEDULE_NUMERIC_COLS, schedule_filing_rows
from src.data_store.schema import Tables

_NUMERIC = SCHEDULE_NUMERIC_COLS
_RESOLVE = "src.data_extract.utils.institutionals.schedule_rows.resolve_schedule_subject_filings"
_SPEC = {"13D": SCHEDULE_13D, "13G": SCHEDULE_13G}
_FETCH = {"13D": SEC_13D_FETCH, "13G": SEC_13G_FETCH}
_TABLE = {"13D": Tables.sec_13d, "13G": Tables.sec_13g}
_SCOPE = EdgarScope(None, {})


def _rows(family: str, filing: Any) -> list[dict]:
    """One filing's rows through the form's row builder, fed a stamp as the walk feeds it."""
    return schedule_filing_rows(FilingStamp.of(filing, ""), _SPEC[family])


def _patch(monkeypatch: pytest.MonkeyPatch, filings: list[Any]) -> None:
    """Serve `filings` as the schedule listing of either form (both walk the shared resolver)."""
    monkeypatch.setattr(_RESOLVE, lambda ticker, subject_ciks, forms, since, done_accessions: filings)


def _build(family: str, ticker: str = "AAPL", cik: str = "0000320193") -> dict:
    return _FETCH[family].build(ticker, cik, since=None, done_accessions=frozenset(), scope=_SCOPE)


def _rp(name: str = "Icahn Carl C", cik: str = "", *, no_cik: bool = False, comment: str | None = None, values: tuple = (0, 0, 0, 0, 0, 0.0)) -> Any:
    return SimpleNamespace(
        name=name,
        cik=cik,
        no_cik=no_cik,
        citizenship="",
        type_of_reporting_person="",
        member_of_group=None,
        comment=comment,
        **dict(zip(_NUMERIC, values, strict=True)),
    )


def _filing(
    *,
    structured: bool,
    persons: list[Any],
    accession: str = "0001-26-000001",
    form: str = "SCHEDULE 13D",
    issuer_cik: str = "0000320193",
    cusip: str = "",
    date_of_event: Any = None,
    event_date: Any = None,
    filers: list[tuple[str, str]] | None = None,
    attachments: list[Any] | None = None,
) -> Any:
    obj = SimpleNamespace(
        has_structured_data=structured,
        is_amendment=form.endswith("/A"),
        amendment_number=None,
        issuer_info=SimpleNamespace(cik=issuer_cik, name="Apple Inc."),
        security_info=SimpleNamespace(cusip=cusip),
        items=None,
        date_of_event=date_of_event,
        event_date=event_date,
        rule_designation="Rule 13d-1(c)",
        reporting_persons=persons,
    )
    header = SimpleNamespace(filers=[SimpleNamespace(company_information=SimpleNamespace(cik=c, name=n)) for c, n in (filers or [])])
    filing = SimpleNamespace(
        accession_number=accession,
        form=form,
        filing_date="2026-03-02",
        primary_document="doc.htm",
        document=None,
        cik=issuer_cik,
        ticker=None,
        header=header,
        attachments=attachments or [],
        text=lambda: "",
    )
    filing.obj = lambda: obj
    return filing


def test_structured_and_unstructured_numerics_for_both_forms():
    values = (100, 0, 100, 0, 100, 5.5)
    for family in ("13D", "13G"):
        trusted = _rows(family, _filing(structured=True, persons=[_rp(values=values)]))[0]
        distrusted = _rows(family, _filing(structured=False, persons=[_rp(values=values)]))[0]
        assert [trusted[c] for c in _NUMERIC] == [100.0, 0.0, 100.0, 0.0, 100.0, 5.5]
        assert all(pd.isna(distrusted[c]) for c in _NUMERIC)
        assert trusted["has_structured_data"] == 1.0 and distrusted["has_structured_data"] == 0.0
    print("\n=== SANITY: schedule numerics ===")
    print("  structured -> 6 disclosed values kept; unstructured -> 6 NaN, on 13D and 13G alike. Validated.")


def test_no_reporting_person_yields_one_nan_fallback_row():
    for family in ("13D", "13G"):
        rows = _rows(family, _filing(structured=True, persons=[]))
        assert len(rows) == 1 and rows[0]["rp_seq"] == 0
        assert rows[0]["reporting_person_name"] is None and rows[0]["reporting_person_cik"] is None
        assert all(isinstance(rows[0][c], float) and pd.isna(rows[0][c]) for c in _NUMERIC)
    assert "item4_purpose_of_transaction" in _rows("13D", _filing(structured=True, persons=[]))[0]
    assert _rows("13G", _filing(structured=True, persons=[]))[0]["rule_designation"] == "Rule 13d-1(c)"
    print("\n=== SANITY: fallback row ===")
    print("  zero parsed persons -> exactly one rp_seq=0 row with float NaN numerics per form. Validated.")


def test_13g_backfills_reporting_person_cik_from_the_header():
    filers = [("0000000111", "ALPHA CAPITAL LLC"), ("0000000222", "BETA ADVISERS LP")]
    exact = _rows("13G", _filing(structured=True, persons=[_rp("Beta Advisers LP"), _rp("Alpha Capital LLC")], filers=filers))
    prefix = _rows("13G", _filing(structured=True, persons=[_rp("Beta Advisers"), _rp("Alpha Capital")], filers=filers))
    positional = _rows("13G", _filing(structured=True, persons=[_rp("Fund One"), _rp("Fund Two")], filers=filers))
    no_cik = _rows("13G", _filing(structured=True, persons=[_rp("Jane Doe", no_cik=True)], filers=[("0000000999", "JANE DOE")]))
    declined = _rows("13G", _filing(structured=True, persons=[_rp("Unrelated Fund")], filers=filers))
    own = _rows("13G", _filing(structured=True, persons=[_rp("Alpha Capital", cik="0000000555")], filers=filers))
    assert [r["reporting_person_cik"] for r in exact] == ["0000000222", "0000000111"]
    assert [r["reporting_person_cik"] for r in prefix] == ["0000000222", "0000000111"]
    assert [r["reporting_person_cik"] for r in positional] == ["0000000111", "0000000222"]
    assert no_cik[0]["reporting_person_cik"] is None
    assert declined[0]["reporting_person_cik"] is None
    assert own[0]["reporting_person_cik"] == "0000000555"
    print("\n=== SANITY: 13G header CIK backfill ===")
    print("  exact, prefix and same-length positional matches fill the CIK; no_cik and an unmatched name stay None. Validated.")


def test_13d_keeps_its_own_cik_without_header_backfill():
    filers = [("0000000111", "ALPHA CAPITAL LLC")]
    blank = _rows("13D", _filing(structured=True, persons=[_rp("Alpha Capital LLC", cik="")], filers=filers))[0]
    no_cik = _rows("13D", _filing(structured=True, persons=[_rp("Doe Jane", cik="9999999999", no_cik=True)]))[0]
    own = _rows("13D", _filing(structured=True, persons=[_rp("Alpha Capital LLC", cik="0000000555")], filers=filers))[0]
    assert blank["reporting_person_cik"] == ""
    assert no_cik["reporting_person_cik"] is None
    assert own["reporting_person_cik"] == "0000000555"
    print("\n=== SANITY: 13D reporting-person CIK ===")
    print("  13D stores the parsed CIK as-is ('' stays ''), nulls only on no_cik, never reads the header. Validated.")


def test_13d_placeholder_numerics_are_nulled_but_13g_keeps_them():
    placeholder = _rp(comment="Rows 7-13: see Item 5")
    disposal = _rp(comment=None)
    row_13d = _rows("13D", _filing(structured=True, persons=[placeholder]))[0]
    row_13d_disposal = _rows("13D", _filing(structured=True, persons=[disposal]))[0]
    row_13g = _rows("13G", _filing(structured=True, persons=[placeholder]))[0]
    assert all(pd.isna(row_13d[c]) for c in _NUMERIC)
    assert [row_13d_disposal[c] for c in _NUMERIC] == [0.0] * 6
    assert [row_13g[c] for c in _NUMERIC] == [0.0] * 6
    assert row_13d["reporting_person_comment"] == "Rows 7-13: see Item 5"
    print("\n=== SANITY: 13D placeholder numerics ===")
    print("  13D comment + all-zero -> NaN; zero without comment -> 0.0; 13G has no placeholder rule. Validated.")


def test_blank_values_kept_on_13d_and_nulled_on_13g():
    row_13d = _rows("13D", _filing(structured=True, persons=[_rp(name="")], cusip=""))[0]
    row_13g = _rows("13G", _filing(structured=True, persons=[_rp(name="")], cusip=""))[0]
    assert row_13d["cusip"] == "" and row_13d["reporting_person_name"] == ""
    assert row_13g["cusip"] is None and row_13g["reporting_person_name"] is None
    real = _rows("13G", _filing(structured=True, persons=[_rp()], cusip="037833100"))[0]
    assert real["cusip"] == "037833100"
    print("\n=== SANITY: blank normalisation ===")
    print("  13D stores cusip='' and name='' verbatim; 13G stores None for both; real values pass. Validated.")


def test_event_date_parsing_differs_by_form():
    def event(family: str, **kw: Any) -> Any:
        return _rows(family, _filing(structured=True, persons=[_rp()], **kw))[0]["date_of_event"]

    assert event("13D", date_of_event="2024-05-13") == pd.Timestamp("2024-05-13")
    assert event("13D", date_of_event=None, event_date="2024-06-01") == pd.Timestamp("2024-06-01")
    assert event("13D", date_of_event="") is None
    with pytest.raises(ValueError):
        event("13D", date_of_event="not a date")
    assert event("13G", date_of_event="01/02/2026") == pd.Timestamp("2026-01-02")
    assert event("13G", date_of_event="2026-03-31") == pd.Timestamp("2026-03-31")
    assert event("13G", date_of_event="not a date") is None
    assert event("13G", date_of_event="", event_date="2024-06-01") is None
    print("\n=== SANITY: event dates ===")
    print("  13D: date_of_event or event_date via pd.Timestamp (garbage raises); 13G: month-first, garbage -> None. Validated.")


def test_walk_skips_filer_side_filings_and_stamps_the_ticker(monkeypatch: pytest.MonkeyPatch):
    for family in ("13D", "13G"):
        own = _filing(structured=True, persons=[_rp(), _rp("Second")], accession="0001-own")
        filer_side = _filing(structured=True, persons=[_rp()], accession="0001-other", issuer_cik="0001199004")
        unknown = _filing(structured=True, persons=[_rp()], accession="0001-unknown", issuer_cik="")
        _patch(monkeypatch, [own, filer_side, unknown])
        frame = _build(family)[_TABLE[family]]
        assert list(frame["accession_number"]) == ["0001-own", "0001-own", "0001-unknown"]
        assert set(frame["ticker"]) == {"AAPL"}
        assert list(frame["rp_seq"]) == [0, 1, 0]
    print("\n=== SANITY: issuer/filer guard ===")
    print("  per form: 3 listed -> own (2 persons) + unknown-issuer kept, filer-side dropped, ticker stamped. Validated.")


def test_parse_failures_keep_their_messages(monkeypatch: pytest.MonkeyPatch):
    def broken() -> Any:
        raise ValueError("broken schedule")

    for family, label in (("13D", "SC 13D"), ("13G", "SC 13G")):
        filing = _filing(structured=True, persons=[_rp()], accession="0001-broken")
        filing.obj = broken
        _patch(monkeypatch, [filing])
        with pytest.raises(RuntimeError, match=f"^{label} accession 0001-broken could not be parsed$"):
            _build(family)
    bad_event = _filing(structured=True, persons=[_rp()], accession="0001-garbage", date_of_event="not a date")
    _patch(monkeypatch, [bad_event])
    with pytest.raises(RuntimeError, match="^SC 13D accession 0001-garbage could not be parsed$"):
        _build("13D")
    exploding = SimpleNamespace(is_html=lambda: True)
    _patch(monkeypatch, [_filing(structured=True, persons=[_rp()], accession="0001-txn", attachments=[exploding])])
    monkeypatch.setattr(fetch_13d_edgar, "BeautifulSoup", lambda *a, **k: (_ for _ in ()).throw(ValueError("bad html")))
    exploding.content = "<table><tr><td>Trade Date</td></tr></table>"
    with pytest.raises(RuntimeError, match="^SC 13D accession 0001-txn transaction exhibit could not be parsed$"):
        _build("13D")
    print("\n=== SANITY: parse failure messages ===")
    print("  SC 13D / SC 13G parse and 13D exhibit failures raise RuntimeError with the accession named. Validated.")


def test_13d_transaction_rows_carry_the_issuer_stamp(monkeypatch: pytest.MonkeyPatch):
    html = """
    <table>
      <tr><td>Trade Date</td><td>Buy/Sell</td><td>Quantity</td><td>Price</td></tr>
      <tr><td>May 1, 2024</td><td>Sell</td><td>1,000</td><td>$</td><td>12.50</td></tr>
      <tr><td>May 2, 2024</td><td>Buy</td><td>500</td><td>$</td><td>11.00</td></tr>
    </table>
    """
    attachment = SimpleNamespace(is_html=lambda: True, content=html)
    filing = _filing(structured=True, persons=[_rp("Icahn Carl C")], accession="0001-txn", issuer_cik="320193", attachments=[attachment])
    _patch(monkeypatch, [filing])
    txn = _build("13D")[Tables.sec_13d_transactions]
    assert list(txn["trade_seq"]) == [0, 1]
    assert set(txn["ticker"]) == {"AAPL"} and set(txn["cik"]) == {"320193"}
    assert set(txn["reporting_person_name"]) == {"Icahn Carl C"}
    assert list(txn["quantity"]) == [1000.0, 500.0]
    assert set(txn["filing_date"]) == {pd.Timestamp("2026-03-02")}
    print("\n=== SANITY: 13D transaction stamp ===")
    print("  2 exhibit trades -> trade_seq 0,1 stamped with ticker, issuer CIK, accession, filing date, sole person. Validated.")
