"""The stored tables that keep a filer CIK per row, and the removal record a foreign filer leaves.

Shared by `identity-propagate` (which purges) and the identity validator (which recomputes the same
pending removals through the store); a row is foreign when its filer CIK is not a CIK of its ticker's entity.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from src.data_store.schema import Table, Tables

#: One row per (table, ticker, filer CIK) a propagation removes; `cik` is empty for a symbol tape.
REMOVAL_COLUMNS = ("table", "ticker", "cik", "first_filed", "last_filed", "keys", "rows")


@dataclass(frozen=True)
class FilerTable:
    """A table whose rows carry the filer CIK: the purge reads `ticker`, `cik_col`, `date_col` and deletes by `key_col`."""

    table: Table
    cik_col: str
    date_col: str
    key_col: str


#: Every stored table that keeps a filer CIK per row. `def14a_llm` and `fundamentals_employees` keep none.
PURGE_TABLES: tuple[FilerTable, ...] = (
    FilerTable(Tables.sec_8k, "cik", "filing_date", "accession_number"),
    FilerTable(Tables.sec_8k_votes, "cik", "filing_date", "accession_number"),
    FilerTable(Tables.sec_13d, "cik", "filing_date", "accession_number"),
    FilerTable(Tables.sec_13d_transactions, "cik", "filing_date", "accession_number"),
    FilerTable(Tables.sec_13g, "cik", "filing_date", "accession_number"),
    FilerTable(Tables.insider_transactions_live, "issuer_cik", "filing_date", "accession_number"),
    FilerTable(Tables.insider_transactions, "issuer_cik", "filing_date", "accession_number"),
    FilerTable(Tables.fundamentals_facts, "cik", "filing_date", "accession_number"),
    FilerTable(Tables.filing_risk_text, "cik", "filed", "accession_number"),
    FilerTable(Tables.def14a_edgar, "cik", "filing_date", "accession_number"),
    FilerTable(Tables.def14a_directors, "cik", "as_of", "accession_number"),
    FilerTable(Tables.def14a_executive_comp, "cik", "as_of", "accession_number"),
    FilerTable(Tables.def14a_director_comp, "cik", "as_of", "accession_number"),
    FilerTable(Tables.def14a_ownership, "cik", "as_of", "accession_number"),
    FilerTable(Tables.notes_num, "cik", "filed", "adsh"),
    FilerTable(Tables.notes_text, "cik", "filed", "adsh"),
    FilerTable(Tables.pension_facts, "cik", "filed", "adsh"),
)
PURGE_TABLES_BY_NAME: Mapping[str, FilerTable] = {spec.table.name: spec for spec in PURGE_TABLES}


def filing_window(dates: pd.Series) -> tuple[str, str]:
    """First and last date as ISO strings ('' when none parse)."""
    stamps = pd.to_datetime(dates, errors="coerce").dropna()
    if stamps.empty:
        return "", ""
    return str(stamps.min().date()), str(stamps.max().date())


def removal_records(table: str, foreign: pd.DataFrame, spec: FilerTable) -> list[dict]:
    """One `REMOVAL_COLUMNS` record per (ticker, filer CIK) of `foreign` rows."""
    records = []
    for (ticker, cik), group in foreign.groupby(["ticker", spec.cik_col], sort=True):
        first, last = filing_window(group[spec.date_col])
        records.append(
            {
                "table": table,
                "ticker": str(ticker),
                "cik": str(cik),
                "first_filed": first,
                "last_filed": last,
                "keys": int(group[spec.key_col].nunique()),
                "rows": len(group),
            }
        )
    return records
