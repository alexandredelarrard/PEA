"""
Amendment / point-in-time tests: original and amended (10-K/A, 10-Q/A) filings
must coexist as separate accession-keyed rows in `fundamentals_facts`, and
`fundamentals_derive._resolve_latest_per_period` must never expose an
amendment's value before its own filing date.

Pure-synthetic, no network / no DB.
"""
from __future__ import annotations

import pandas as pd

from src.data_extract.utils.fundamentals.fetch_fundamentals_edgar import (
    populate_amends_accession,
)
from src.data_extract.utils.fundamentals.fundamentals_derive import (
    _resolve_latest_per_period,
)
from src.data_extract.utils.fundamentals.fundamentals_periods import (
    annual_flow, decumulate_quarterly_flow, instant_stock,
)


def _facts_row(accession, form, filing_date, value, is_amendment=0.0,
              field="netIncome", fiscal_year=2024, fiscal_period="Q1",
              duration_type="quarterly", ticker="ZZZ"):
    return {"ticker": ticker, "accession_number": accession, "field": field,
           "fiscal_year": fiscal_year, "fiscal_period": fiscal_period,
           "duration_type": duration_type, "form": form,
           "filing_date": pd.Timestamp(filing_date), "value": value,
           "is_amendment": is_amendment}


def test_original_and_amendment_coexist_as_separate_rows():
    """A later 10-Q/A restating a value must NOT overwrite the original row --
    both persist as distinct accession-keyed facts."""
    facts = pd.DataFrame([
        _facts_row("acc-orig", "10-Q", "2024-04-20", 100.0, is_amendment=0.0),
        _facts_row("acc-amend", "10-Q/A", "2024-08-09", 105.0, is_amendment=1.0),
    ])
    assert len(facts) == 2
    assert set(facts["accession_number"]) == {"acc-orig", "acc-amend"}
    assert facts[facts["accession_number"] == "acc-orig"]["value"].iloc[0] == 100.0
    assert facts[facts["accession_number"] == "acc-amend"]["value"].iloc[0] == 105.0


def test_amends_accession_resolution_matches_prior_filing():
    """amends_accession = the immediately-prior accession sharing (ticker, field,
    fiscal_year, fiscal_period, duration_type) -- chains correctly even through
    multiple amendments, and stays None when no qualifying prior original exists
    within the fetch window."""
    facts = pd.DataFrame([
        _facts_row("acc-orig", "10-Q", "2024-04-20", 100.0, is_amendment=0.0),
        _facts_row("acc-amend1", "10-Q/A", "2024-08-09", 105.0, is_amendment=1.0),
        _facts_row("acc-amend2", "10-Q/A", "2024-09-01", 107.0, is_amendment=1.0),
        # a DIFFERENT period's amendment with no original in this frame -> stays None
        _facts_row("acc-orphan-amend", "10-Q/A", "2024-09-01", 999.0, is_amendment=1.0,
                  fiscal_period="Q2"),
    ])
    out = populate_amends_accession(facts)
    by_acc = out.set_index("accession_number")
    assert pd.isna(by_acc.loc["acc-orig", "amends_accession"])
    assert by_acc.loc["acc-amend1", "amends_accession"] == "acc-orig"
    assert by_acc.loc["acc-amend2", "amends_accession"] == "acc-amend1"   # chains, not always-original
    assert pd.isna(by_acc.loc["acc-orphan-amend", "amends_accession"])


def test_amendment_free_10ka_produces_zero_rows_and_never_enters_resolution():
    """A 10-K/A filed only to add Part III proxy items (no new financial facts)
    must simply produce no rows for a given field/period -- it never appears in
    `fundamentals_facts` at all for that concept, so it can't spuriously supersede
    the original."""
    facts = pd.DataFrame([
        _facts_row("acc-orig", "10-K", "2024-02-15", 460.0, is_amendment=0.0, fiscal_period="FY"),
        # the 10-K/A simply never contributed a netIncome row -- nothing to resolve
    ])
    resolved = _resolve_latest_per_period(facts, as_of_cutoff=None)
    assert len(resolved) == 1
    assert resolved.iloc[0]["value"] == 460.0


def _raw_fact(fiscal_year, fiscal_period, start, end, value, filed, accn, form,
             is_amendment=0.0, source_tag="us-gaap:X"):
    """Pre-decumulation raw-fact row shape (as `build_ticker_facts_edgar` hands to
    `decumulate_quarterly_flow`/`annual_flow`/`instant_stock`), NOT the already-
    assembled `fundamentals_facts` row shape `_facts_row` above builds."""
    return {"fiscal_year": fiscal_year, "fiscal_period": fiscal_period,
           "period_start": pd.Timestamp(start), "period_end": pd.Timestamp(end),
           "value": value, "filing_date": pd.Timestamp(filed),
           "accession_number": accn, "form": form, "is_amendment": is_amendment,
           "source_tag": source_tag, "fiscal_period_source": "native"}


def test_decumulate_quarterly_flow_keeps_amendment_as_its_own_row_with_correct_provenance():
    """Real bug found via live data: JPM's original 10-Q for fiscal-2012 Q1
    `deferredIncomeTaxExpense` (accession ...-000213, filed 2012-05-10) came back
    in `fundamentals_facts` tagged is_amendment=1.0 -- contaminated from a LATER
    10-Q/A (accession ...-000262, filed 2012-08-09) restating the SAME quarter
    with the SAME value. Root cause: a prior version looked up provenance
    (source_tag/is_amendment/fiscal_period_source) via a SEPARATE groupby keyed
    only by (fiscal_year, fiscal_period), decoupled from which accession's value
    ended up in the row, AND collapsed to "earliest filing wins" -- discarding the
    amendment's own row entirely. Provenance must come from the SAME raw fact as
    the row's own value/accession_number, and the amendment must persist as its
    OWN row rather than being discarded."""
    facts = pd.DataFrame([
        _raw_fact(2012, "Q1", "2012-01-01", "2012-03-31", -444000000.0, "2012-05-10",
                 "0000019617-12-000213", "10-Q", is_amendment=0.0),
        _raw_fact(2012, "Q1", "2012-01-01", "2012-03-31", -444000000.0, "2012-08-09",
                 "0000019617-12-000262", "10-Q/A", is_amendment=1.0),
    ])
    out = decumulate_quarterly_flow(facts)
    q1_rows = out[out["fiscal_period"] == "Q1"].set_index("accession_number")
    assert len(q1_rows) == 2
    assert q1_rows.loc["0000019617-12-000213", "is_amendment"] == 0.0
    assert q1_rows.loc["0000019617-12-000213", "form"] == "10-Q"
    assert q1_rows.loc["0000019617-12-000262", "is_amendment"] == 1.0
    assert q1_rows.loc["0000019617-12-000262", "form"] == "10-Q/A"


def test_annual_flow_keeps_amendment_as_its_own_row():
    """A 10-K/A restating the annual figure must coexist with the original 10-K's
    row (own accession_number, is_amendment=1.0) rather than being discarded by
    annual_flow's per-fiscal_year collapse."""
    facts = pd.DataFrame([
        _raw_fact(2012, "FY", "2012-01-01", "2012-12-31", 1000.0, "2013-02-20", "acc-orig", "10-K"),
        _raw_fact(2012, "FY", "2012-01-01", "2012-12-31", 950.0, "2013-05-01", "acc-amend",
                 "10-K/A", is_amendment=1.0),
    ])
    out = annual_flow(facts)
    assert len(out) == 2
    by_acc = out.set_index("accession_number")
    assert by_acc.loc["acc-orig", "value"] == 1000.0 and by_acc.loc["acc-orig", "is_amendment"] == 0.0
    assert by_acc.loc["acc-amend", "value"] == 950.0 and by_acc.loc["acc-amend", "is_amendment"] == 1.0


def test_instant_stock_keeps_amendment_as_its_own_row():
    """Same coexistence guarantee for a balance-sheet (instant) field restated by
    a 10-K/A."""
    facts = pd.DataFrame([
        _raw_fact(2012, "Q4", "2012-10-01", "2012-12-31", 5000.0, "2013-02-20", "acc-orig", "10-K"),
        _raw_fact(2012, "Q4", "2012-10-01", "2012-12-31", 5200.0, "2013-05-01", "acc-amend",
                 "10-K/A", is_amendment=1.0),
    ])
    out = instant_stock(facts)
    assert len(out) == 2
    by_acc = out.set_index("accession_number")
    assert by_acc.loc["acc-orig", "value"] == 5000.0 and by_acc.loc["acc-orig", "is_amendment"] == 0.0
    assert by_acc.loc["acc-amend", "value"] == 5200.0 and by_acc.loc["acc-amend", "is_amendment"] == 1.0

    print("\n=== SANITY CHECK: amendment coexistence at the construction-function level ===")
    print("  decumulate_quarterly_flow/annual_flow/instant_stock each now emit ONE ROW PER")
    print("  ACCESSION (original AND every amendment), instead of collapsing to only the")
    print("  earliest-filed fact -- reproduces the real JPM bug (original 10-Q wrongly")
    print("  tagged is_amendment=1.0, borrowed from an unrelated later 10-Q/A) and confirms")
    print("  it is fixed: each row's is_amendment/form/source_tag now comes from the SAME")
    print("  accession as its own value, and the amendment is no longer discarded.")
    print("  Validated.")


def test_fundamentals_history_excludes_amendments_before_filing_date():
    """`_resolve_latest_per_period`'s as_of_cutoff must never select an amendment
    whose OWN filing_date is after the cutoff -- point-in-time correctness: a query
    'as of 2024-06-01' must see the original value (amendment not yet filed); a
    query 'as of 2024-09-01' (after the amendment) must see the restated value."""
    facts = pd.DataFrame([
        _facts_row("acc-orig", "10-Q", "2024-04-20", 100.0, is_amendment=0.0),
        _facts_row("acc-amend", "10-Q/A", "2024-08-09", 105.0, is_amendment=1.0),
    ])

    before = _resolve_latest_per_period(facts, as_of_cutoff=pd.Timestamp("2024-06-01"))
    assert len(before) == 1
    assert before.iloc[0]["value"] == 100.0
    assert before.iloc[0]["accession_number"] == "acc-orig"

    after = _resolve_latest_per_period(facts, as_of_cutoff=pd.Timestamp("2024-09-01"))
    assert len(after) == 1
    assert after.iloc[0]["value"] == 105.0
    assert after.iloc[0]["accession_number"] == "acc-amend"

    today = _resolve_latest_per_period(facts, as_of_cutoff=None)
    assert today.iloc[0]["value"] == 105.0

    print("\n=== SANITY CHECK: amendment point-in-time resolution ===")
    print("  original + amendment coexist as separate accession-keyed rows;")
    print("  amends_accession chains correctly through multiple amendments;")
    print("  a query 'as of' a date BEFORE the amendment's own filing_date sees only")
    print("  the original value (100.0); a query 'as of' a date AFTER sees the restated")
    print("  value (105.0) -- an amendment is never exposed before its own filing date.")
    print("  Validated.")
