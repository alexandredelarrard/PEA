"""
fetch_def14a_edgar.py (src/data_extract/utils/structure/fetch_def14a_edgar.py)
--------------------------------------------------------------------------------
`sec_def14a`: the Pay-versus-Performance / ECD inline-XBRL block of a proxy, and
nothing else. One row per `(ticker, accession_number)`, zero LLM cost.

These are facts the FILER tagged and computed, so reading them deterministically
beats any extraction. Everything a proxy says in PROSE -- the compensation
tables, director fees, beneficial ownership, audit fees, the CEO pay ratio,
board voting recommendations -- now belongs entirely to `fetch_def14a_llm.py`.
The HTML-parsed block that used to live here, plus its four child tables, was
deleted: edgartools' proxy HTML parser returns values that are silently WRONG
rather than absent (a missed "(in thousands)" header, a hardcoded 0.5 standing
in for a "*" percent, three value-inventing pay-ratio repairs), and the defects
are ticker-persistent, so they do not average out.
"""

from __future__ import annotations

from functools import partial

import pandas as pd

from src.constants.constants import DEF14A_FORMS
from src.data_extract.utils.common.edgar_driver import EdgarFetch, FilingStamp, build_filing_rows
from src.data_extract.utils.structure.def14a.ecd import ecd_facts, ecd_row, has_ecd_block
from src.data_extract.utils.structure.def14a.validate import repair_main_row
from src.data_store.schema import Tables

_MAIN_COLS = [
    "ticker",
    "cik",
    "accession_number",
    "form",
    "filing_date",
    "period_of_report",
    "company_name",
    "has_individual_executive_data",
    # The fiscal year the PVP facts describe. Not derivable from `period_of_report`, which for a
    # proxy is the MEETING date -- without this column nothing says which year `peo_total_comp`
    # belongs to, and the PVP table carries five.
    "ecd_period_end",
    "peo_name",
    "peo_total_comp",
    "peo_actually_paid_comp",
    "n_peos",
    "peo_names_all",
    "neo_avg_total_comp",
    "neo_avg_actually_paid_comp",
    "total_shareholder_return",
    "peer_group_tsr",
    "net_income",
    "company_selected_measure_name",
    "company_selected_measure_value",
    "insider_trading_policy_adopted",
    "award_timing_mnpi_considered",
    "award_dates_predetermined",
    "mnpi_disclosure_timed_for_comp_value",
]

#: `sec_def14a`'s numeric columns, coerced after the PK dedup.
_NUMERIC_COLS = tuple(
    c
    for c in _MAIN_COLS
    if c
    not in (
        "ticker",
        "cik",
        "accession_number",
        "form",
        "filing_date",
        "period_of_report",
        "ecd_period_end",
        "company_name",
        "peo_name",
        "peo_names_all",
        "company_selected_measure_name",
    )
)

#: Cover-page registrant name. Read from XBRL when tagged, else from the filing index --
#: edgartools reads this concept with NO index fallback, and DEF 14A has no mandatory
#: cover-page iXBRL requirement, so it is None on every pre-2023 filing and `__str__`
#: substitutes the literal "Unknown Company".
_REGISTRANT = "dei:EntityRegistrantName"


def _company_name(facts: pd.DataFrame, filing) -> str | None:
    sub = facts[facts["concept"].astype(str) == _REGISTRANT]
    for v in sub.get("value", pd.Series(dtype=object)):
        if isinstance(v, str) and v.strip():
            return v.strip()
    name = getattr(filing, "company", None)
    return name.strip() if isinstance(name, str) and name.strip() else None


def _filing_row(ticker: str, stamp: FilingStamp) -> list[dict]:
    """One ECD row for a tagged filing, none for a filing with no `ecd:` facts.

    Reads `filing.xbrl()` facts directly, so DEF 14A and DEF 14C (absent from edgartools'
    `PROXY_FORMS` dispatch) are treated alike.
    """
    facts = ecd_facts(stamp.filing)
    if not has_ecd_block(facts):
        return []  # pre-402(v) fiscal year -- correct behaviour, no row
    assert facts is not None
    row = ecd_row(facts)
    row.update(
        ticker=ticker,
        # The filer's own CIK, not the roster's: a registrant reorganisation then shows up as
        # two CIKs either side of a date instead of hiding behind one stamped value.
        cik=stamp.cik,
        accession_number=stamp.accession_number,
        form=str(stamp.form),
        filing_date=stamp.filed.normalize(),
        # From the filing index, like every sibling fetcher (guarded: the raw property can raise).
        period_of_report=stamp.period_of_report,
        company_name=_company_name(facts, stamp.filing),
    )
    return [repair_main_row(row)]


DEF14A_EDGAR_FETCH = EdgarFetch(
    desc="DEF 14A (ECD XBRL)",
    tables=(Tables.def14a_edgar,),
    build=partial(build_filing_rows, forms=DEF14A_FORMS, table=Tables.def14a_edgar, columns=_MAIN_COLS, row_fn=_filing_row, numeric=_NUMERIC_COLS),
)
