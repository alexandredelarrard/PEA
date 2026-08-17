"""
incremental.py  (src/data_extract/utils/common/incremental.py)
--------------------------------------------------------------
Resume helpers shared by the per-entity fetchers -- the "what do we already have?"
read that every incremental fetcher does before it spends a request.

`load_existing` replaces five byte-identical private copies (google_trends,
wiki_pageviews, dividends, short_interest, employees_history), which differed only in
the table name. The normalisation matters and is easy to get wrong in a copy: dates
must be `.normalize()`d, because the resume logic compares a stored timestamp against
a fetched one and a stray time component silently re-downloads a day that is already
there.

NOT here: the three `_is_up_to_date` functions. They share a name but not a meaning
(business-day price freshness vs per-ticker DB coverage vs universe-size meta), so
merging them would invent an abstraction that does not exist.
"""
from __future__ import annotations

import pandas as pd

from src.context import Context

__all__ = ["load_existing"]


def load_existing(context: Context, table: str,
                  date_col: str | None = "date") -> pd.DataFrame | None:
    """A fetcher's already-stored rows, or None when there is nothing to resume from.

    None (not an empty frame) is the contract the callers rely on to branch between
    "full history" and "incremental". `date_col` is normalised to midnight so date
    comparisons against freshly fetched rows are exact; pass `date_col=None` for a
    table keyed by something else (employees_history is keyed by filing date)."""
    df = context.store.load(table, optional=True)
    if df is None:
        return None
    if date_col is not None and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()
    return df
