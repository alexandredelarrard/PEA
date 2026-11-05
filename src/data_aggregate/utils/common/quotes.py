"""As-of quote lookups shared by the insider and 13F value checks."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd


def period_close(close: pd.DataFrame, periods: pd.Series, max_age_days: int) -> pd.Series:
    """Last observed valuation-day quote, bounded by age and never from the future."""
    wanted = pd.DatetimeIndex(sorted(periods.dropna().unique()))
    df_quotes = close.sort_index().where(np.isfinite(close) & close.gt(0))
    union = df_quotes.index.union(wanted)
    df_asof = df_quotes.reindex(union).ffill().reindex(wanted)
    for ticker in df_quotes:
        observed = pd.Series(df_quotes.index, index=df_quotes.index).where(df_quotes[ticker].notna())
        last = observed.reindex(union).ffill().reindex(wanted)
        df_asof[ticker] = df_asof[ticker].where((pd.Series(wanted, index=wanted) - last).le(pd.Timedelta(days=max_age_days)))
    result = cast(pd.Series, df_asof.stack(future_stack=True))
    result.index = result.index.set_names(["period", "ticker"])
    return result
