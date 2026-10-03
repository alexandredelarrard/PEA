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

__all__ = ["matches_stored", "resume_since", "stored_values"]


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


def resume_since(
    context: Context,
    table: Table | str,
    tickers: list[str],
    years_history: int,
    ticker_col: str = "ticker",
    date_col: str = "date",
    include_missing: bool = True,
) -> pd.Timestamp:
    """Earliest per-ticker last-stored date across `tickers`, never earlier than `years_history` back.

    The caller re-fetches every ticker from this one date and relies on the upsert to no-op current ones.
    `include_missing=True` lets a ticker with no row pull the window to the full history; pass False
    where absence is permanent (e.g. a dividend non-payer)."""

    history_start = pd.Timestamp.today().normalize() - pd.DateOffset(years=years_history)
    last_by_ticker = context.store.max_date_by(table, ticker_col, date_col)
    if not last_by_ticker:
        return history_start
    if include_missing:
        stored = [last_by_ticker.get(t, history_start) for t in tickers]
    else:
        stored = [last_by_ticker[t] for t in tickers if t in last_by_ticker]
    return max(min(stored, default=history_start), history_start)
