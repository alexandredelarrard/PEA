"""Completeness-frontier resolution for institutional source families."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import pandas as pd

from src.data_store.schema import Table, Tables
from src.data_store.store import DataStore


def _normalized_timestamp(value: Any) -> pd.Timestamp:
    return cast(pd.Timestamp, pd.Timestamp(value)).normalize()


def schedule_complete_through(store: DataStore, table: Table, last_session: pd.Timestamp | None) -> pd.Timestamp | None:
    """The "absence means zero through date D" frontier of an EDGAR event table (13D, 13G and
    insider), read from `table` alone.

    D is `last_session` when the table's latest `filing_date` (markers included) lies within its
    resume overlap of that session, else the latest `filing_date`. None for an absent or empty table.
    """
    latest = store.max_date(table, "filing_date")
    if latest is None or last_session is None or table.resume is None:
        return latest
    last = _normalized_timestamp(last_session)
    fresh = latest >= last - pd.Timedelta(days=table.resume.overlap_days)
    return max(latest, last) if fresh else latest


def insider_inventory_boundaries(store: DataStore, tickers: Sequence[str]) -> pd.Series:
    """An issuer is complete only when every stored row and marker shares one fetched proof."""
    result = pd.Series(pd.NaT, index=pd.Index(sorted(set(tickers)), name="ticker"), dtype="datetime64[ns]")
    table = Tables.insider_transactions
    if not {"inventory_checked_through", "fetched_at", "ticker"}.issubset(store.columns(table)):
        return result
    df_summary = store.summarize(
        table,
        "ticker",
        {
            "first": ("min", "inventory_checked_through"),
            "last": ("max", "inventory_checked_through"),
            "checked": ("count", "inventory_checked_through"),
            "fetched": ("count", "fetched_at"),
            "rows": ("count", None),
        },
        where={"ticker": list(result.index)},
    )
    complete = (
        df_summary["first"].eq(df_summary["last"]) & df_summary["checked"].eq(df_summary["rows"]) & df_summary["fetched"].eq(df_summary["rows"])
    )
    for ticker, boundary in df_summary.loc[complete, ["ticker", "first"]].itertuples(index=False, name=None):
        result.loc[ticker] = pd.Timestamp(boundary)
    return result
