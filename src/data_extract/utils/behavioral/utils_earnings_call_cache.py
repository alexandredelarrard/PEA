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
    invalid = []
    for (ticker, quarter), call in rows.groupby(["ticker", "quarter"], sort=False):
        quality = assess_earnings_call_sections(dict(zip(call["tag"].astype(str), call["text"], strict=False)))
        if quality.valid:
            continue
        as_of = pd.to_datetime(call["as_of"], errors="coerce").dropna() if "as_of" in call else pd.Series(dtype="datetime64[ns]")
        invalid.append({"ticker": ticker, "quarter": quarter, "as_of": as_of.iloc[0] if not as_of.empty else None})
    if invalid:
        context.store.save(Tables.earnings_call_sentiment, pending_refresh_markers(pd.DataFrame(invalid)))
    return saved
