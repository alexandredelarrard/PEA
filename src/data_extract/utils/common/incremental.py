"""
incremental.py  (src/data_extract/utils/common/incremental.py)
--------------------------------------------------------------
Resume helpers shared by the per-entity fetchers -- the "what do we already have?"
read that every incremental fetcher does before it spends a request.

`load_existing` replaces five byte-identical private copies, which differed only in
the table name; google_trends and wiki_pageviews still use it (the others now resolve
their frontier without reading the table -- see below). The normalisation matters and
is easy to get wrong in a copy: dates must be `.normalize()`d, because the resume
logic compares a stored timestamp against a fetched one and a stray time component
silently re-downloads a day that is already there.

`resume_since` generalizes the per-ticker `groupby(...)[date_col].max()` idiom that
several fetchers (dividends, wiki pageviews, earnings surprises, filing text) already
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

import pandas as pd

from src.context import Context

__all__ = ["load_existing", "resume_since"]


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


def resume_since(
    context: Context,
    table: str,
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
