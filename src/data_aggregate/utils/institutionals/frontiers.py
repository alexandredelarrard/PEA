"""Completeness-frontier resolution for institutional source families."""

from __future__ import annotations

from typing import Any, cast

import pandas as pd

from src.data_store.schema import Table
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
