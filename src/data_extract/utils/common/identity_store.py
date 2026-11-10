"""Load the source identity cohort and commit one identity table (or symbol-tenure partition) atomically."""

from __future__ import annotations

import pandas as pd

from src.data_store.schema import Table, Tables
from src.data_store.store import DataStore


def load_source_identity(store: DataStore) -> dict[Table, pd.DataFrame]:
    """Load the closed three-table identity projection for source classification and guarded writes."""
    cohort: dict[Table, pd.DataFrame] = {}
    for table in (Tables.universe_cik_symbol, Tables.symbol_tenure, Tables.universe_cusip):
        df_rows = store.load(table, project=True, optional=True)
        cohort[table] = df_rows if df_rows is not None else pd.DataFrame()
    return cohort


def save_identity(store: DataStore, table: Table, df_rows: pd.DataFrame, where: dict[str, object] | None = None) -> int:
    """Save one identity table, or replace one symbol-tenure partition (`where`), in one transaction."""
    if table not in (Tables.universe_cik_symbol, Tables.universe_cusip, Tables.symbol_tenure):
        raise ValueError("Only the lineage, master and tenure writers use the identity commit")
    if where == {} or table != Tables.symbol_tenure and where is not None:
        raise ValueError("Only a symbol-tenure partition can carry a replacement scope")
    if not set(table.pk).issubset(df_rows) or df_rows.duplicated(list(table.pk)).any():
        raise ValueError("Identity replacement needs unique declared keys")
    with store.transaction() as bound:
        if where is None:
            return bound.replace(table, df_rows)
        bound.delete(table, where)
        return bound.save(table, df_rows, ignore_all_conflicts=True)
