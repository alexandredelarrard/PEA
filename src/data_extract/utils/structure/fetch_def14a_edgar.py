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

import pandas as pd

from src.constants.constants import DEF14A_FORMS
from src.context import Context
from src.data_extract.utils.common.edgar_driver import new_filings, run_edgar_fetch
from src.data_extract.utils.common.identity import Identity
from src.data_extract.utils.structure.def14a.ecd import ecd_facts, ecd_row, has_ecd_block
from src.data_extract.utils.structure.def14a.validate import repair_main_row
from src.data_store.schema import Table, Tables

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

#: Destination table -> its numeric columns. Doubles as the table list handed to the driver, so
#: the two can never disagree.
_NUMERIC_COLS: dict[Table, list[str]] = {
    Tables.def14a_edgar: [
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
    ],
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


def build_ticker_def14a_edgar(
    ticker: str,
    cik: str,
    *,
    since: pd.Timestamp | None = None,
    done_accessions: frozenset[str] = frozenset(),
    identity: Identity | None = None,
    symbol_tenure: pd.DataFrame | None = None,
    roster_cik: str | None = None,
) -> dict[Table, pd.DataFrame]:
    """One ECD row per tagged filing. A filing with no `ecd:` facts yields nothing.

    `filing.xbrl()` replaces the old `filing.obj()`: the typed `ProxyStatement` was only needed
    for the HTML block, and DEF 14C is not in edgartools' `PROXY_FORMS` dispatch at all, so the
    old `hasattr(proxy, "voting_proposals")` guard silently skipped every DEF 14C. Going
    straight to the facts frame treats both forms alike.
    """
    rows: list[dict] = []
    for f in new_filings(
        ticker,
        DEF14A_FORMS,
        since,
        done_accessions,
        identity=identity,
        symbol_tenure=symbol_tenure,
        roster_cik=roster_cik,
    ):
        facts = ecd_facts(f)
        if not has_ecd_block(facts):
            continue  # pre-402(v) fiscal year -- correct behaviour, no row
        assert facts is not None
        row = ecd_row(facts)
        row.update(
            # ⚠ THE CIK COMES OFF THE FILING, NOT OFF THE ROSTER. `new_filings` resolves by
            # TICKER, so stamping the roster's `cik` recorded a value that need not be the one
            # that filed: measured 2026-09-09, 521 XOM `sec_8k` rows carried CIK 2115436
            # (ExxonMobil Holdings Corp, 29 filings, first on 2026-07-01) against filings going
            # back to 1996 that were actually the predecessor's, CIK 34088.
            #
            # This does NOT fix resolution -- you can only read a CIK off filings you already
            # have, and a wrong roster CIK yields none to read (that is what the cutover
            # register in `def14a/fetch.py` is for). What it fixes is OBSERVABILITY: with the
            # filer's own CIK stored, a reorganisation shows up immediately as two CIKs either
            # side of a date instead of hiding behind a uniformly-stamped column.
            ticker=ticker,
            cik=str(getattr(f, "cik", cik) or cik).zfill(10),
            accession_number=f.accession_number,
            form=str(f.form),
            filing_date=pd.Timestamp(f.filing_date).normalize(),
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
    return {table: _coerce_numeric(df.drop_duplicates(subset=list(table.pk), keep="last"), _NUMERIC_COLS[table])}


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def fetch_def14a_edgar(context: Context, tickers: list[str], years_history: int) -> None:
    run_edgar_fetch(
        context,
        tickers,
        years_history,
        tables=tuple(_NUMERIC_COLS),
        build=build_ticker_def14a_edgar,
        identity_aware=True,
        desc="DEF 14A (ECD XBRL)",
        require_complete=True,
    )
