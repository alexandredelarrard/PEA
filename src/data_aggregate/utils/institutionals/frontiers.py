"""Completeness-frontier resolution for institutional source families."""

from __future__ import annotations

from typing import Any, cast

import pandas as pd

from src.context import Context
from src.data_store.schema import Table
from src.utils.data_policy import source_read_window
from src.utils.freshness import max_age_days


def _normalized_timestamp(value: Any) -> pd.Timestamp:
    return cast(pd.Timestamp, pd.Timestamp(value)).normalize()


def schedule_complete_through(context: Context, table: Table, last_session: pd.Timestamp | None) -> pd.Timestamp | None:
    """The "absence means zero through date D" frontier of an EDGAR event table (13D and 13G),
    read from `table` alone.

    D is `last_session` when the table's latest `filing_date` (markers included) lies within its
    configured freshness tolerance of that session, else the latest `filing_date`. None for an absent or empty table.
    """
    retained = source_read_window(context.config, table, context.run.as_of)
    if retained is None:
        return None
    latest = context.store.max_date(table, "filing_date", where={"filing_date": retained})
    if latest is None or last_session is None:
        return latest
    last = _normalized_timestamp(last_session)
    fresh = latest >= last - pd.Timedelta(days=max_age_days(context.config, table))
    return max(latest, last) if fresh else latest
