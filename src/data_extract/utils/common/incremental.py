"""Resume helpers shared by the per-entity fetchers: what is already stored before a request is spent.

`stored_values` returns the keys already stored (`SELECT DISTINCT`); `matches_stored` lets a full
re-derivation skip an unchanged replace; `resume_since` gives one shared per-ticker frontier from a
single `store.max_date_by` aggregate, never a table read.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from src.context import Context
from src.data_store.schema import Table

__all__ = ["matches_stored", "stored_values"]


def stored_values(context: Context, tables: Table | str | Sequence[Table | str], column: str) -> frozenset[str]:
    """Distinct non-null `column` values already stored across `tables`, as strings.

    A table that is absent or lacks `column` contributes nothing, so a first run starts empty.
    """
    names = [tables] if isinstance(tables, Table | str) else tables
    values: set[str] = set()
    for table in names:
        if column in context.store.columns(table):
            values.update(str(value) for value in context.store.distinct(table, column))
    return frozenset(values)


def matches_stored(existing: pd.DataFrame | None, frame: pd.DataFrame, table: Table) -> bool:
    """True when `frame` holds exactly the stored rows of a fully derived `table`.

    Both sides are compared sorted on the PK, date columns parsed and every value as a nullable string.
    """
    if existing is None or len(existing) != len(frame) or set(existing.columns) != set(frame.columns):
        return False
    columns = list(frame.columns)
    return _canonical_rows(existing, columns, table).equals(_canonical_rows(frame, columns, table))


def _canonical_rows(df: pd.DataFrame, columns: list[str], table: Table) -> pd.DataFrame:
    """`columns` of `df` as nullable strings (date columns parsed first), sorted on the table key."""
    df_canonical = df[columns].copy()
    for column in table.date_type_cols:
        df_canonical[column] = pd.to_datetime(df_canonical[column], errors="coerce")
    return df_canonical.astype("string").sort_values(list(table.pk), kind="mergesort", ignore_index=True)
