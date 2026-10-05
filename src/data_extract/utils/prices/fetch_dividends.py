"""
fetch_dividends.py  (src/data_extract/utils/prices/fetch_dividends.py)
------------------------------------------------------------------------
Row shaping for `prices_dividends`: cash dividends keyed on the EX-date, one row per session
(0 when none). The download is `fetch_prices.fetch_prices_and_actions`, which shares one
yfinance `actions=True` response between prices, dividends and splits.
"""

from __future__ import annotations

from typing import cast

import pandas as pd

_COLUMNS = ["date", "ticker", "dividends"]


def _extract_dividends(long_prices: pd.DataFrame | None) -> pd.DataFrame:
    """Keep 0 dividends as a value, since they are informative anyway. Increase table size,
    but more stable to refresh and merge.

    Empty in, empty out: `download_ohlcv` returns a column-less frame when every chunk
    failed, and a total yfinance outage must no-op rather than KeyError."""

    if long_prices is None or long_prices.empty or "dividends" not in long_prices.columns:
        return pd.DataFrame(columns=_COLUMNS)

    d = cast(pd.DataFrame, long_prices[_COLUMNS].copy())
    d["dividends"] = pd.to_numeric(d["dividends"], errors="coerce")
    d["date"] = pd.to_datetime(d["date"], format="%Y-%m-%d")
    return d.reset_index(drop=True)
