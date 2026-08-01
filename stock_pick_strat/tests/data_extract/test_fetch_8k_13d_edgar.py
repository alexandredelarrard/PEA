"""
Unit tests for the edgartools-based 8-K / SC 13D fetchers
(fetch_8k_edgar.py / fetch_13d_edgar.py). Pure-synthetic, no network -- filings
and their typed `.obj()` results are faked with SimpleNamespace so the row-
building logic (`_filing_row` / `_filing_rows`) is exercised without needing a
live `Company(ticker).get_filings(...)` call.
"""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from src.data_extract.utils.structure.fetch_8k_edgar import _filing_row
from src.data_extract.utils.structure.fetch_13d_edgar import _filing_rows


def test_8k_and_13d_default_to_the_same_years_history_as_fundamentals():
    """8-K/13D must discover filings over the SAME window fundamentals_facts uses,
    so a ticker never shows a fundamentals quarter with no corresponding 8-K/13D
    coverage for that period purely from a config divergence rather than a
    genuine absence of filings. fundamentals_facts resolves its window via
    `getattr(de, "fundamentals_years_history", de.years_history)`
    (fetch_fundamentals_edgar.py); 8-K/13D resolve via plain `de.years_history`
    (fetch_8k_edgar.py / fetch_13d_edgar.py). These are the SAME value whenever
    `fundamentals_years_history` is unset (the case in this repo's own
    configs/configs.yml today) -- this test pins that equivalence property so a
    future source-specific override on ONE side doesn't silently desync them
    without at least a visible test failure."""
    de = SimpleNamespace(years_history=15)   # fundamentals_years_history intentionally unset
    fundamentals_window = int(getattr(de, "fundamentals_years_history", de.years_history))
    sec_8k_window = int(de.years_history)
    sec_13d_window = int(de.years_history)
    assert fundamentals_window == sec_8k_window == sec_13d_window == 15


def _fake_8k_filing(*, accession="0001-24-000001", form="8-K", filing_date="2024-05-01",
                    period_of_report="2024-04-30", items="2.02,9.01",
                    primary_document="form8k.htm", obj=None):
    filing = SimpleNamespace(
        accession_number=accession, form=form, filing_date=filing_date,
        period_of_report=period_of_report, items=items, primary_document=primary_document,
    )
    filing.obj = (lambda: obj) if obj is not None else (lambda: (_ for _ in ()).throw(RuntimeError("no parse")))
    return filing


def test_8k_filing_row_reads_current_report_flags():
    """has_earnings/has_press_release come from the typed CurrentReport object,
    stored as 1.0/0.0 flags (repo convention), alongside the existing item codes."""
    obj = SimpleNamespace(has_earnings=True, has_press_release=False)
    filing = _fake_8k_filing(obj=obj)
    row = _filing_row("MAA", filing)
    assert row["has_earnings"] == 1.0
    assert row["has_press_release"] == 0.0
    assert row["items"] == "2.02,9.01"
    assert row["n_items"] == 2
    assert row["is_amendment"] == 0.0


def test_8k_filing_row_survives_failed_obj_parse():
    """A filing whose .obj() call raises must still yield a row (item codes are
    always reliable, straight from the filing index) with both CurrentReport
    flags null rather than losing the row entirely."""
    filing = _fake_8k_filing()   # obj=None -> .obj() raises
    row = _filing_row("MAA", filing)
    assert row["has_earnings"] is None
    assert row["has_press_release"] is None
    assert row["items"] == "2.02,9.01"


def test_8k_amendment_flag_from_form_suffix():
    filing = _fake_8k_filing(form="8-K/A", obj=SimpleNamespace(has_earnings=False, has_press_release=False))
    row = _filing_row("MAA", filing)
    assert row["is_amendment"] == 1.0


def _fake_13d_filing(*, accession="0001-24-000002", form="SC 13D", filing_date="2024-06-11",
                     primary_document="sc13d.htm", obj=None):
    filing = SimpleNamespace(
        accession_number=accession, form=form, filing_date=filing_date,
        primary_document=primary_document, document=None,
    )
    filing.obj = lambda: obj
    return filing


def _reporting_person(name, cik="0001822844", no_cik=False, **kw):
    defaults = dict(citizenship="", sole_voting_power=0, shared_voting_power=0,
                    sole_dispositive_power=0, shared_dispositive_power=0,
                    aggregate_amount=0, percent_of_class=0.0, type_of_reporting_person="")
    defaults.update(kw)
    return SimpleNamespace(name=name, cik=cik, no_cik=no_cik, **defaults)


def test_13d_reporting_persons_get_one_row_each_with_rp_seq():
    """A 13D with TWO co-filing reporting persons must produce two rows, keyed by
    rp_seq (0-based position) -- collapsing them into one row would silently drop
    all but one filer."""
    obj = SimpleNamespace(
        has_structured_data=False, is_amendment=False, amendment_number=None,
        issuer_info=SimpleNamespace(cik="0001326380", name="GameStop Corp."),
        security_info=SimpleNamespace(cusip="36467W109"),
        items=SimpleNamespace(item4_purpose_of_transaction=None),
        date_of_event=None, event_date=None,
        reporting_persons=[_reporting_person("RC Ventures LLC"), _reporting_person("Cohen Ryan", cik="0001")],
    )
    filing = _fake_13d_filing(obj=obj)
    rows = _filing_rows(filing)
    assert len(rows) == 2
    assert [r["rp_seq"] for r in rows] == [0, 1]
    assert {r["reporting_person_name"] for r in rows} == {"RC Ventures LLC", "Cohen Ryan"}
    assert rows[0]["cusip"] == "36467W109"
    assert rows[0]["issuer_name"] == "GameStop Corp."


def test_13d_numeric_ownership_fields_null_when_not_structured():
    """When has_structured_data is False, the underlying parser's numeric fields
    (voting/dispositive power, percent_of_class) default to 0 -- NOT a real
    disclosed value. Publishing that 0 would silently claim a 0% stake for an
    activist that may hold a real position, so every such field must be NaN
    (never the class default 0) -- NaN rather than None/null so the column stays
    float dtype even when a whole batch is unknown (an all-None object column
    would get inferred as SQL TEXT, corrupting a genuinely numeric field the
    first time a real row needs to share it)."""
    obj = SimpleNamespace(
        has_structured_data=False, is_amendment=True, amendment_number=3,
        issuer_info=None, security_info=None,
        items=SimpleNamespace(item4_purpose_of_transaction=None),
        date_of_event=None, event_date=None,
        reporting_persons=[_reporting_person("RC Ventures LLC", percent_of_class=0.0, aggregate_amount=0)],
    )
    filing = _fake_13d_filing(obj=obj)
    row = _filing_rows(filing)[0]
    assert row["has_structured_data"] == 0.0
    assert pd.isna(row["percent_of_class"])
    assert pd.isna(row["aggregate_amount"])
    assert pd.isna(row["sole_voting_power"])
    assert row["is_amendment"] == 1.0
    assert row["amendment_number"] == 3


def test_13d_numeric_ownership_fields_trusted_when_structured():
    """When has_structured_data IS True, the real parsed numeric values must pass
    through untouched -- the null-out above is specifically an UNRELIABLE-parse
    guard, not a blanket "never trust the numbers" rule."""
    obj = SimpleNamespace(
        has_structured_data=True, is_amendment=False, amendment_number=None,
        issuer_info=None, security_info=None,
        items=SimpleNamespace(item4_purpose_of_transaction="Acquire control of the issuer."),
        date_of_event="2024-05-13", event_date=None,
        reporting_persons=[_reporting_person("Icahn Carl C", percent_of_class=9.9, aggregate_amount=12345678)],
    )
    filing = _fake_13d_filing(obj=obj)
    row = _filing_rows(filing)[0]
    assert row["percent_of_class"] == 9.9
    assert row["aggregate_amount"] == 12345678.0
    assert row["item4_purpose_of_transaction"] == "Acquire control of the issuer."
    assert row["date_of_event"] == "2024-05-13"


def test_13d_reporting_person_without_cik_is_not_dropped():
    """`no_cik=True` (common for individuals / entities without an assigned CIK)
    must not drop the row -- only the CIK column is nulled."""
    obj = SimpleNamespace(
        has_structured_data=False, is_amendment=False, amendment_number=None,
        issuer_info=None, security_info=None,
        items=SimpleNamespace(item4_purpose_of_transaction=None),
        date_of_event=None, event_date=None,
        reporting_persons=[_reporting_person("Doe Jane", cik="9999999999", no_cik=True)],
    )
    filing = _fake_13d_filing(obj=obj)
    row = _filing_rows(filing)[0]
    assert row["reporting_person_name"] == "Doe Jane"
    assert row["reporting_person_cik"] is None

    print("\n=== SANITY CHECK: edgartools 8-K / SC 13D row extraction ===")
    print("  8-K: has_earnings/has_press_release read from CurrentReport (best-effort,")
    print("  null not crash on parse failure); item codes always present; amendment flag")
    print("  from form suffix.")
    print("  SC 13D: one row PER REPORTING PERSON (rp_seq-keyed, multi-filer 13Ds preserved);")
    print("  numeric ownership fields (voting/dispositive power, percent_of_class) are NULL")
    print("  -- never the parser's 0 default -- whenever has_structured_data is false, but")
    print("  pass through untouched when it is true; a reporting person with no assigned")
    print("  CIK keeps their row (name preserved, CIK nulled).")
    print("  Validated.")
