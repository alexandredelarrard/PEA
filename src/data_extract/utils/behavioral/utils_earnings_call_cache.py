"""Atomic cache policy for transcript-section replacements."""

from __future__ import annotations

import pandas as pd

from src.context import Context
from src.data_store.schema import Tables


def invalidate_earnings_call_derivatives(context: Context, calls: pd.DataFrame) -> int:
    """Delete derived rows for every transcript call about to be replaced."""
    if calls.empty or not hasattr(context.store, "delete"):
        return 0
    invalidated = 0
    keys = calls[["ticker", "quarter"]].drop_duplicates()
    for ticker, group in keys.groupby("ticker", sort=False):
        where = {"ticker": str(ticker), "quarter": group["quarter"].astype(str).tolist()}
        invalidated += context.store.delete(Tables.earnings_call_sentiment, where)
        invalidated += context.store.delete(Tables.earning_calls_embedding, where)
    return invalidated


def save_earnings_call_sections(context: Context, rows: pd.DataFrame) -> int:
    """Invalidate stale derivatives, then upsert their replacement source sections."""
    if rows.empty:
        return 0
    invalidate_earnings_call_derivatives(context, rows)
    return context.store.save(Tables.earnings_call_sections, rows)
