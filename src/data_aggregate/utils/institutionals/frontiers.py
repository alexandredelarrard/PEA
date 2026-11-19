"""Completeness-frontier resolution for institutional source families."""

from __future__ import annotations

import pandas as pd

from src.context import Context
from src.data_store.schema import Table
from src.utils.data_policy import source_read_window


def schedule_complete_through(context: Context, table: Table, last_session: pd.Timestamp | None) -> pd.Timestamp | None:
    """The "absence means zero through date D" frontier of an EDGAR event table (13D and 13G),
    read from `table` alone.

    Only the latest stored filing/marker proves completion; source age cannot invent a later observed zero.
    """
    retained = source_read_window(context.config, table, context.run.as_of)
    if retained is None:
        return None
    latest = context.store.max_date(table, "filing_date", where={"filing_date": retained})
    return latest if latest is None or last_session is None else min(latest, pd.Timestamp(last_session).normalize())
