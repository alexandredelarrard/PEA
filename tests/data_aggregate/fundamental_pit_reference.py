"""Explicit legacy capital inputs for feature regression, separate from PIT correctness tests."""

from __future__ import annotations

import pandas as pd

from src.data_aggregate.utils.common.pit import fundamentals_to_daily
from src.data_aggregate.utils.price.frames import PriceFrames
from src.utils.price.adjustments import ShareBasis


def known_no_share_events() -> ShareBasis:
    """Explicit event-free basis for known-truth feature fixtures."""
    df_events = pd.DataFrame({"ticker": pd.Series(dtype=str), "date": pd.Series(dtype="datetime64[ns]"), "value": pd.Series(dtype=float)})
    return ShareBasis(df_events, df_events, df_events, {})


def legacy_capital_frames(
    history: pd.DataFrame | None, close: pd.DataFrame | None, level_factor: pd.DataFrame | None = None, max_age_days: int | None = 185
) -> PriceFrames | None:
    """Supply the former compatible price/count pair as explicit regression inputs.

    This fixture preserves old feature comparisons; source-backed price tests establish
    the actual historical quote, quantity units and issuer-cap coverage separately.
    """
    if close is None:
        return None
    df_counts = (
        pd.DataFrame(index=close.index)
        if history is None
        else fundamentals_to_daily(history, "sharesOutstanding", close.index, max_age_days=max_age_days)
    )
    df_counts = df_counts.reindex(columns=close.columns.intersection(df_counts.columns))
    df_price = close if level_factor is None else close * level_factor.reindex_like(close)
    df_cap = (df_price[df_counts.columns] * df_counts).where(lambda values: values > 0)
    return PriceFrames(
        trading_index=pd.DatetimeIndex(close.index),
        universe=tuple(map(str, close.columns)),
        peers={},
        close_split=close,
        level_factor=level_factor,
        close_pit=df_price,
        shares_outstanding_pit=df_counts,
        mcap_pit=df_cap,
    )
