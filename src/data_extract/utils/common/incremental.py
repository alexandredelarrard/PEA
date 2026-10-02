"""
incremental.py  (src/data_extract/utils/common/incremental.py)
--------------------------------------------------------------
Resume helpers shared by the per-entity fetchers -- the "what do we already have?"
read that every incremental fetcher does before it spends a request.

`stored_values` answers "which keys are already stored" (accessions, bulk periods) across one or
more tables with `SELECT DISTINCT`; `matches_stored` tells a full re-derivation that its frame
equals the stored table, so the replace can be skipped.

`resume_since` generalizes the per-ticker `groupby(...)[date_col].max()` idiom that
several fetchers (dividends, earnings surprises, filing text) already
duplicate ad hoc: the oldest per-ticker last-extracted date across a batch, so a
caller can re-fetch every ticker forward from ONE shared date and let the upsert
no-op whichever tickers were already current. It resolves that date with a single
`SELECT key, MAX(date) GROUP BY key` (`store.max_date_by`) and never loads the table
-- grouping `prices` (~1.8M rows) in pandas just to read one date was seconds and
hundreds of MB per run.

NOT here: a whole-market source, where the frontier is one date for everyone rather
than one per entity. `short_interest`'s RegSHO day-files each carry the entire market,
so it resumes straight off `store.max_date` -- a per-ticker frontier there would only
re-download days already held.

NOT here either: the three `_is_up_to_date` functions. They share a name but not a
meaning (business-day price freshness vs per-ticker DB coverage vs universe-size
meta), so merging them would invent an abstraction that does not exist.
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

    Both sides are compared sorted on the table's primary key, with date columns parsed (a DATE
    column reads back as `datetime.date`) and every value as a nullable string, so a stored
    table read back from either backend compares equal to the frame that wrote it.
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
    """Oldest date any of `tickers` still needs (re)fetching from: the earliest
    per-ticker last-extracted date, never earlier than `years_history` back.

    Generic across any fetcher keyed by (ticker_col, date_col) -- the caller
    re-fetches every ticker forward from this ONE shared date in a single batch
    and relies on the upsert to no-op a ticker that was already current. Costs one
    grouped aggregate query (`store.max_date_by`), not a table read.

    `include_missing=True`: a ticker with no stored row pulls the window back to the
    full `years_history` -- correct for `prices`, where an unseen ticker genuinely
    needs its whole history, and self-correcting once it has rows. Pass **False**
    where absence is legitimate and PERMANENT: `dividends` never gets a row for a
    non-payer, so counting those would pin every run to the full window forever."""

    history_start = pd.Timestamp.today().normalize() - pd.DateOffset(years=years_history)
    last_by_ticker = context.store.max_date_by(table, ticker_col, date_col)
    if not last_by_ticker:
        return history_start
    if include_missing:
        stored = [last_by_ticker.get(t, history_start) for t in tickers]
    else:
        stored = [last_by_ticker[t] for t in tickers if t in last_by_ticker]
    # clamp: a ticker stale beyond the window must not widen it past `years_history`
    return max(min(stored, default=history_start), history_start)
