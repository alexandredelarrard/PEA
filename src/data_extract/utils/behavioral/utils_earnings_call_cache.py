"""Invalidate saved model outputs when source transcripts are replaced or removed."""

from __future__ import annotations

import pandas as pd

from src.context import Context
from src.data_store.schema import Tables


def invalidate_earnings_call_derivatives(context: Context, calls: pd.DataFrame) -> int:
    """Delete derived rows for every transcript call about to be replaced."""
    if calls.empty:
        return 0
    invalidated = 0
    keys = calls[["ticker", "quarter"]].drop_duplicates()
    for ticker, group in keys.groupby("ticker", sort=False):
        where = {"ticker": str(ticker), "quarter": group["quarter"].astype(str).tolist()}
        invalidated += context.store.delete(Tables.earnings_call_sentiment, where)
        invalidated += context.store.delete(Tables.earning_calls_embedding, where)
    return invalidated
