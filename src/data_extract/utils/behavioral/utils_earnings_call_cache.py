"""Invalidate saved model outputs when source transcripts are replaced or removed."""

from __future__ import annotations

import pandas as pd

from src.data_store.schema import Tables
from src.data_store.store import DataStore


def invalidate_earnings_call_derivatives(store: DataStore, calls: pd.DataFrame) -> int:
    """Delete derived rows for every transcript call about to be replaced."""
    if calls.empty:
        return 0
    invalidated = 0
    keys = calls[["ticker", "quarter"]].drop_duplicates()
    for ticker, group in keys.groupby("ticker", sort=False):
        where = {"ticker": str(ticker), "quarter": group["quarter"].astype(str).tolist()}
        invalidated += store.delete(Tables.earnings_call_sentiment, where)
        invalidated += store.delete(Tables.earning_calls_embedding, where)
    return invalidated
