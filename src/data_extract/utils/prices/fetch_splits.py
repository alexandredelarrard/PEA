"""
fetch_splits.py  (src/data_extract/utils/prices/fetch_splits.py)
------------------------------------------------------------------------
Row shaping for `prices_splits`: the non-zero share-split EX-dates of a yfinance `actions=True`
response. The download is `fetch_prices.fetch_prices_and_actions`.

A second split source next to `sharadar_actions`, which misses events yfinance carries;
`field_map.split_events` takes their union. Not a market-cap input: `close_split` and
`sharesbas` carry the same retroactive restatement.
"""

from __future__ import annotations

from typing import cast

import pandas as pd

_COLUMNS = ["date", "ticker", "ratio"]
#: yfinance names the column `Stock Splits`, which `_normalize_prices` lowercases.
_RAW_COLUMN = "stock splits"


def _extract_splits(long_prices: pd.DataFrame | None) -> pd.DataFrame:
    """The non-zero split events in a `download_ohlcv` response -> `[date, ticker, ratio]`.

    ⚠ Only NON-ZERO rows are kept, which is the one deliberate difference from
    `fetch_dividends`. There a stored 0 is informative and makes the refresh idempotent; here
    a zero split is meaningless and would turn a few-thousand-row event table into a 3.2M-row
    copy of the price grid.

    Empty in, empty out: `download_ohlcv` returns a column-less frame when every chunk
    failed, and a total yfinance outage must no-op rather than KeyError."""
    if long_prices is None or long_prices.empty or _RAW_COLUMN not in long_prices.columns:
        return pd.DataFrame(columns=_COLUMNS)

    s = cast(pd.DataFrame, long_prices[["date", "ticker", _RAW_COLUMN]]).rename(columns={_RAW_COLUMN: "ratio"})
    ratio = cast(pd.Series, pd.to_numeric(s["ratio"], errors="coerce"))
    s["ratio"] = ratio
    s = cast(pd.DataFrame, s[ratio.notna() & (ratio != 0.0)])
    if s.empty:
        return pd.DataFrame(columns=_COLUMNS)
    s["date"] = pd.to_datetime(s["date"], format="%Y-%m-%d")
    return cast(pd.DataFrame, s[_COLUMNS]).drop_duplicates(subset=["ticker", "date"]).reset_index(drop=True)
