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

REGULATORY SCOPE, which is why a row is not written for every filing
--------------------------------------------------------------------
Item 402(v) applies to fiscal years ending on or after 2022-12-16. A proxy
covering an earlier year carries no `ecd:` facts at all, so it gets NO ROW --
`has_xbrl` was dropped with the HTML block because a table that only holds
tagged filings makes it degenerate. The inventory of "which proxies exist" is
`def14a_llm`'s job, and it covers all of them.

THE DIMENSION FILTER
--------------------
`ProxyStatement`'s accessors filter on `concept ==` only and take `.iloc[0]`, so
on a co-PEO year document order decides which executive survives -- BA's 2025
proxy silently drops one of Ortberg / Calhoun. `def14a_ecd.py` reads the facts
frame directly and resolves the dimensions; see its module docstring for the two
incompatible tagging styles filers use and why the axis filter is conditional.

`peo_actually_paid_comp` is NEGATIVE on real filings (NKE 2025: -10,924,243) --
Compensation Actually Paid subtracts prior-year unvested fair value. There is no
sign flip and no `abs()` anywhere on this path.
"""

from __future__ import annotations

import pandas as pd

from src.constants.constants import DEF14A_FORMS
from src.context import Context
from src.data_extract.utils.common.edgar_driver import new_filings, run_edgar_fetch
from src.data_extract.utils.structure.def14a.ecd import ecd_facts, ecd_row, has_ecd_block
from src.data_extract.utils.structure.def14a.validate import repair_main_row
from src.data_store.schema import Table, Tables

_MAIN_COLS = [
    "ticker", "cik", "accession_number", "form", "filing_date", "period_of_report",
    "company_name", "has_individual_executive_data",
    # The fiscal year the PVP facts describe. Not derivable from `period_of_report`, which for a
    # proxy is the MEETING date -- without this column nothing says which year `peo_total_comp`
    # belongs to, and the PVP table carries five.
    "ecd_period_end",
    "peo_name", "peo_total_comp", "peo_actually_paid_comp", "n_peos", "peo_names_all",
    "neo_avg_total_comp", "neo_avg_actually_paid_comp",
    "total_shareholder_return", "peer_group_tsr", "net_income",
    "company_selected_measure_name", "company_selected_measure_value",
    "insider_trading_policy_adopted", "award_timing_mnpi_considered",
    "award_dates_predetermined", "mnpi_disclosure_timed_for_comp_value",
]

#: Destination table -> its numeric columns. Doubles as the table list handed to the driver, so
#: the two can never disagree.
_NUMERIC_COLS: dict[Table, list[str]] = {
    Tables.def14a_edgar: [c for c in _MAIN_COLS if c not in (
        "ticker", "cik", "accession_number", "form", "filing_date", "period_of_report",
        "ecd_period_end", "company_name", "peo_name", "peo_names_all",
        "company_selected_measure_name",
    )],
}

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


def build_ticker_def14a_edgar(ticker: str, cik: str, *, since: pd.Timestamp | None = None,
                              done_accessions: frozenset[str] = frozenset(),
                              ) -> dict[Table, pd.DataFrame]:
    """One ECD row per tagged filing. A filing with no `ecd:` facts yields nothing.

    `filing.xbrl()` replaces the old `filing.obj()`: the typed `ProxyStatement` was only needed
    for the HTML block, and DEF 14C is not in edgartools' `PROXY_FORMS` dispatch at all, so the
    old `hasattr(proxy, "voting_proposals")` guard silently skipped every DEF 14C. Going
    straight to the facts frame treats both forms alike.
    """
    rows: list[dict] = []
    for f in new_filings(ticker, DEF14A_FORMS, since, done_accessions):
        facts = ecd_facts(f)
        if not has_ecd_block(facts):
            continue                       # pre-402(v) fiscal year -- correct behaviour, no row
        row = ecd_row(facts)
        row.update(
            ticker=ticker, cik=cik, accession_number=f.accession_number,
            form=str(f.form), filing_date=pd.Timestamp(f.filing_date).normalize(),
            # From the filing index, like every sibling fetcher. `ProxyStatement.fiscal_year_end`
            # was the previous source and never once resolved -- 0 of 329 stored rows had it.
            period_of_report=f.period_of_report,
            company_name=_company_name(facts, f),
        )
        rows.append(repair_main_row(row))

    df = pd.DataFrame(rows, columns=_MAIN_COLS)
    # De-dup on the PK FIRST (an upsert touching one PK row twice is an error in Postgres),
    # then coerce -- coercing first would do the work on rows about to be dropped.
    table = Tables.def14a_edgar
    return {table: _coerce_numeric(df.drop_duplicates(subset=list(table.pk), keep="last"),
                                   _NUMERIC_COLS[table])}


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def fetch_def14a_edgar(context: Context, tickers: list[str], years_history: int) -> None:
    run_edgar_fetch(context, tickers, years_history, tables=tuple(_NUMERIC_COLS),
                    build=build_ticker_def14a_edgar, desc="DEF 14A (ECD XBRL)")
