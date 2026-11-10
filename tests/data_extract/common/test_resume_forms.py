"""Resume contracts in `schema.py` agree with the extraction code: forms and source starts."""

from __future__ import annotations

from dataclasses import replace
from typing import cast

import pandas as pd
import pytest

from src.data_extract.utils.common.registrant import FORM_POLICY, Combine
from src.data_extract.utils.institutionals import fetch_superinvestors as roster
from src.data_extract.utils.zips import fetch_fails_to_deliver as ftd
from src.data_store.schema import RESUME_DOCUMENTS, Resume, Table, Tables, resume_tables


def _start(table: Table) -> pd.Timestamp:
    assert table.resume is not None and table.resume.source_start is not None, table.name
    return pd.Timestamp(table.resume.source_start)


def test_document_forms_are_in_the_form_policy() -> None:
    documents = [t for t in resume_tables() if t.resume is not None and t.resume.mode == RESUME_DOCUMENTS]
    unknown = {
        t.name: sorted(set(t.resume.forms) - set(FORM_POLICY)) for t in documents if t.resume is not None and set(t.resume.forms) - set(FORM_POLICY)
    }
    assert not unknown, f"document forms missing from registrant.FORM_POLICY: {unknown}"
    index_diffed = {t.name: set(t.resume.forms) for t in documents if t.resume is not None and t.resume.forms}
    proxy_forms = {"DEF 14A", "DEF 14C", "DEFC14A"}
    employee_forms = {"10-K", "10-K/A", "10-K405", "10-K405/A"}
    assert index_diffed == {
        Tables.fundamentals_employees.name: employee_forms,
        Tables.sec_employees_text.name: employee_forms,
        Tables.sec_def14a_text.name: proxy_forms,
        Tables.sec_fundamentals_facts.name: {"10-K", "10-Q"},
        Tables.insider_transactions.name: {"3", "4", "5", "3/A", "4/A", "5/A"},
        Tables.sec_13d.name: {"SC 13D", "SC 13D/A", "SCHEDULE 13D", "SCHEDULE 13D/A"},
        Tables.sec_13g.name: {"SC 13G", "SC 13G/A", "SCHEDULE 13G", "SCHEDULE 13G/A"},
        Tables.def14a_llm.name: proxy_forms,
        Tables.def14a_directors.name: proxy_forms,
        Tables.def14a_executive_comp.name: proxy_forms,
        Tables.def14a_director_comp.name: proxy_forms,
        Tables.def14a_ownership.name: proxy_forms,
        Tables.sec_8k.name: {"8-K", "8-K/A"},
        Tables.filing_risk_text.name: {"10-K", "10-Q"},
    }

    for table in (Tables.sec_employees_text, Tables.sec_def14a_text):
        resume = cast(Resume, table.resume)
        assert (resume.key, resume.frontier_col, resume.overlap_days, resume.source_start) == ("ticker", "filing_date", 95, "1993-01-01")
        assert all(FORM_POLICY[form] is Combine.SPLIT for form in resume.forms), table.name

    print("\n=== SANITY CHECK: document forms ===")
    print(
        f"  {len(index_diffed)} index-diffed tables retain their exact forms, including all four proxy children: {sorted(index_diffed)}. Validated."
    )
    print("  both EDGAR text tables: key ticker, frontier filing_date, 95-day overlap, 1993 start, SPLIT; employees include 10-K405/A.")


def test_changed_source_dates_reach_archive_and_activity_planning(monkeypatch: pytest.MonkeyPatch) -> None:
    table = Tables.sec_fails_to_deliver_security
    resume = replace(cast(Resume, table.resume), source_start="2011-06-16")
    monkeypatch.setattr(Tables, "sec_fails_to_deliver_security", replace(table, resume=resume))
    periods = ftd.ftd_periods(50, pd.Timestamp("2011-07-01"))
    assert periods == ["201106b", "201107a", "201107b"]
    table = Tables.sec13f_hr
    resume = replace(cast(Resume, table.resume), coverage_start="2014-09-30")
    monkeypatch.setattr(Tables, "sec13f_hr", replace(table, resume=resume))
    end = roster.activity_window("2012-03-07").end
    assert end is not None and end.isoformat() == "2014-09-30"
    print("SANITY: registry changes move the first FTD half-month and the pre-XML activity window without fetcher constants.")


def test_registered_source_starts_preserve_the_source_boundaries() -> None:
    expected = (
        (Tables.notes_num, "2009-01-01"),
        (Tables.notes_text, "2009-01-01"),
        (Tables.insider_transactions, "2006-01-01"),
        (Tables.sec_fails_to_deliver, "2009-07-01"),
        (Tables.sec_fails_to_deliver_security, "2009-07-01"),
        (Tables.earnings_call_sections, "2005-01-01"),
        (Tables.earnings_surprises, "2002-10-01"),
        (Tables.sharadar_sp500, "1990-01-01"),
        (Tables.superinvestor_roster, "2012-01-01"),
        (Tables.sec13f_hr, "1999-07-01"),
    )
    for table, start in expected:
        assert _start(table) == pd.Timestamp(start), table.name
    assert Tables.sec13f_hr.resume is not None and Tables.sec13f_hr.resume.coverage_start == "2013-06-30"
    for table in (Tables.notes_num, Tables.notes_text):
        assert table.resume is not None and table.resume.observed_from == "2026-09-01", table.name
    indexed = [t for t in resume_tables() if t.resume is not None and t.resume.mode == RESUME_DOCUMENTS and t.resume.forms]
    assert all(t.resume is not None and t.resume.source_start is not None for t in indexed)
    assert min(_start(t) for t in indexed) == pd.Timestamp("1993-01-01")

    print("\n=== SANITY CHECK: source starts ===")
    print(
        "  11 source boundaries preserved; EDGAR starts in 1993Q1; 13F broad coverage is distinct from its 1999 history; archive clocks turn observed in September 2026. Validated."
    )
