"""Resume contracts in `schema.py` agree with the extraction code: forms and source starts."""

from __future__ import annotations

import pandas as pd

from src.data_extract.utils.common.registrant import FORM_POLICY
from src.data_store.schema import RESUME_DOCUMENTS, Table, Tables, resume_tables


def _start(table: Table) -> pd.Timestamp:
    assert table.resume is not None and table.resume.source_start is not None, table.name
    return pd.Timestamp(table.resume.source_start)


def test_document_forms_are_in_the_form_policy() -> None:
    documents = [t for t in resume_tables() if t.resume is not None and t.resume.mode == RESUME_DOCUMENTS]
    unknown = {
        t.name: sorted(set(t.resume.forms) - set(FORM_POLICY)) for t in documents if t.resume is not None and set(t.resume.forms) - set(FORM_POLICY)
    }
    assert not unknown, f"document forms missing from registrant.FORM_POLICY: {unknown}"
    index_diffed = sorted(t.name for t in documents if t.resume is not None and t.resume.forms)
    assert len(index_diffed) == 9, index_diffed

    print("\n=== SANITY CHECK: document forms ===")
    print(f"  {len(index_diffed)} index-diffed tables, every declared form is a FORM_POLICY form: {index_diffed}. Validated.")


def test_registered_source_starts_preserve_the_source_boundaries() -> None:
    expected = (
        (Tables.pension_facts, "2009-01-01"),
        (Tables.notes_num, "2009-01-01"),
        (Tables.notes_text, "2009-01-01"),
        (Tables.insider_transactions, "2006-01-01"),
        (Tables.sec_fails_to_deliver, "2009-07-01"),
        (Tables.sec_fails_to_deliver_security, "2009-07-01"),
        (Tables.earnings_call_sections, "2005-01-01"),
        (Tables.earnings_surprises, "2002-10-01"),
        (Tables.def14a_edgar, "2022-12-16"),
        (Tables.sharadar_sp500, "1990-01-01"),
        (Tables.superinvestor_roster, "2012-01-01"),
        (Tables.sec13f_hr, "1999-07-01"),
    )
    for table, start in expected:
        assert _start(table) == pd.Timestamp(start), table.name
    assert Tables.sec13f_hr.resume is not None and Tables.sec13f_hr.resume.coverage_start == "2013-06-30"
    for table in (Tables.pension_facts, Tables.notes_num, Tables.notes_text):
        assert table.resume is not None and table.resume.observed_from == "2026-09-01", table.name
    indexed = [t for t in resume_tables() if t.resume is not None and t.resume.forms]
    assert all(t.resume is not None and t.resume.source_start is not None for t in indexed)
    assert min(_start(t) for t in indexed) == pd.Timestamp("1993-01-01")

    print("\n=== SANITY CHECK: source starts ===")
    print(
        "  12 source boundaries preserved; EDGAR starts in 1993Q1; 13F broad coverage is distinct from its 1999 history; archive clocks turn observed in September 2026. Validated."
    )
