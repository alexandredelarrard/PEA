"""Derived-cache invalidation and pending-refresh markers for re-issued transcript calls."""

from __future__ import annotations

import pandas as pd

from src.constants.constants import EARNINGS_CALL_SCORED_TAGS, EARNINGS_CALL_SENTIMENT_INVALID_PENDING_MODEL
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


def pending_refresh_markers(calls: pd.DataFrame) -> pd.DataFrame:
    """`invalid-pending` sentiment rows for `calls` (`ticker`, `quarter`, `as_of`).

    The marker survives the extraction/aggregation process boundary: the text step reads
    the earliest pending `as_of` as the start of the cube tail it must refresh, and
    acknowledges the marker only after the cube write succeeds.
    """
    return pd.DataFrame(
        [
            {
                "ticker": ticker,
                "quarter": quarter,
                "tag": tag,
                "as_of": as_of,
                "sent_pos": None,
                "sent_neg": None,
                "sent_neu": None,
                "n_words": None,
                "uncertainty_ratio": None,
                "model": EARNINGS_CALL_SENTIMENT_INVALID_PENDING_MODEL,
            }
            for ticker, quarter, as_of in calls[["ticker", "quarter", "as_of"]].itertuples(index=False)
            for tag in EARNINGS_CALL_SCORED_TAGS
        ]
    )
