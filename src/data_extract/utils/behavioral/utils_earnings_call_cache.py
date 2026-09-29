"""Atomic cache policy for transcript-section replacements."""

from __future__ import annotations

import pandas as pd

from src.constants.constants import EARNINGS_CALL_SCORED_TAGS, EARNINGS_CALL_SENTIMENT_INVALID_PENDING_MODEL
from src.context import Context
from src.data_store.schema import Tables
from src.utils.text_metrics import assess_earnings_call_sections


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
    """Invalidate stale derivatives, upsert source, and flag invalid replacements.

    The pending cache marker survives the extraction/aggregation process boundary.  The
    text step acknowledges it only after using its ``as_of`` to refresh the persisted
    historical tail, so a formerly valid call cannot leave stale non-null cube rows.
    """
    if rows.empty:
        return 0
    rows = rows.copy()
    previous_dates: dict[tuple[str, str], object] = {}
    if hasattr(context.store, "columns") and "as_of" in context.store.columns(Tables.earnings_call_sections):
        for ticker, group in rows[["ticker", "quarter"]].drop_duplicates().groupby("ticker", sort=False):
            previous = context.store.load(
                Tables.earnings_call_sections,
                ["ticker", "quarter", "as_of"],
                where={"ticker": str(ticker), "quarter": group["quarter"].astype(str).tolist()},
                optional=True,
            )
            if previous is None:
                continue
            for (old_ticker, old_quarter), call in previous.groupby(["ticker", "quarter"], sort=False):
                dates = pd.to_datetime(call["as_of"], errors="coerce").dropna()
                if not dates.empty:
                    previous_dates[(str(old_ticker), str(old_quarter))] = dates.iloc[0]
    if "as_of" not in rows:
        rows["as_of"] = pd.NaT
    missing_date = pd.to_datetime(rows["as_of"], errors="coerce").isna()
    recovered_dates = pd.to_datetime(
        [previous_dates.get((str(ticker), str(quarter))) for ticker, quarter in rows.loc[missing_date, ["ticker", "quarter"]].to_numpy()]
    )
    rows.loc[missing_date, "as_of"] = recovered_dates
    invalidate_earnings_call_derivatives(context, rows)
    saved = context.store.save(Tables.earnings_call_sections, rows)
    markers = []
    for (ticker, quarter), call in rows.groupby(["ticker", "quarter"], sort=False):
        quality = assess_earnings_call_sections(dict(zip(call["tag"].astype(str), call["text"], strict=False)))
        if quality.valid:
            continue
        as_of = pd.to_datetime(call["as_of"], errors="coerce").dropna() if "as_of" in call else pd.Series(dtype="datetime64[ns]")
        for tag in EARNINGS_CALL_SCORED_TAGS:
            markers.append(
                {
                    "ticker": ticker,
                    "quarter": quarter,
                    "tag": tag,
                    "as_of": as_of.iloc[0] if not as_of.empty else None,
                    "sent_pos": None,
                    "sent_neg": None,
                    "sent_neu": None,
                    "n_words": None,
                    "uncertainty_ratio": None,
                    "model": EARNINGS_CALL_SENTIMENT_INVALID_PENDING_MODEL,
                }
            )
    if markers:
        context.store.save(Tables.earnings_call_sentiment, pd.DataFrame(markers))
    return saved
