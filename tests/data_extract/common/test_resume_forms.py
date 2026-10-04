"""Resume contracts in `schema.py` agree with the extraction code: forms and source starts."""

from __future__ import annotations

import pandas as pd

from src.constants.constants import SEC_INSIDER_FIRST_YEAR
from src.data_extract.utils.behavioral.fetch_earnings_call_transcripts import HISTORY_START
from src.data_extract.utils.common.registrant import FORM_POLICY
from src.data_extract.utils.fundamentals.fetch_financial_notes import SEC_FINNOTES_FIRST_YEAR
from src.data_extract.utils.fundamentals.fetch_financial_statements import SEC_FINSTMT_FIRST_YEAR
from src.data_extract.utils.institutionals.fetch_fails_to_deliver import SEC_FTD_FIRST_YEAR
from src.data_extract.utils.structure.fetch_def14a_edgar import _PVP_EFFECTIVE
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


def test_source_starts_match_the_fetchers() -> None:
    assert _start(Tables.pension_facts).year == SEC_FINSTMT_FIRST_YEAR
    assert _start(Tables.notes_num).year == _start(Tables.notes_text).year == SEC_FINNOTES_FIRST_YEAR
    assert _start(Tables.insider_transactions).year == SEC_INSIDER_FIRST_YEAR
    assert _start(Tables.sec_fails_to_deliver).year == SEC_FTD_FIRST_YEAR
    assert _start(Tables.earnings_call_sections) == pd.Timestamp(HISTORY_START)
    assert _start(Tables.def14a_edgar) == _PVP_EFFECTIVE

    print("\n=== SANITY CHECK: source starts ===")
    print("  pension, notes, insider bulk, FTD, earnings calls and DEF 14A ECD starts equal the fetchers' own constants. Validated.")
