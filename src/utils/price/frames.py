"""Shared Yahoo price reshaping and total-return derivation."""

from __future__ import annotations

from typing import cast

import pandas as pd


def extract_field(df: pd.DataFrame, field: str) -> pd.DataFrame:
    """
    Return a wide DataFrame (index=dates, columns=tickers) for a single price
    field (e.g. 'Close' or 'Open'), regardless of the exact yfinance layout.

    Handles three common shapes:
      1. MultiIndex columns with field on level 0: ('Close','AAPL')
      2. MultiIndex columns with field on level 1: ('AAPL','Close')
      3. Already-wide single-field frame (index=dates, columns=tickers)
    """
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0)
        lvl1 = df.columns.get_level_values(1)
        if field in set(lvl0):
            out = df.xs(field, axis=1, level=0)
        elif field in set(lvl1):
            out = df.xs(field, axis=1, level=1)
        else:
            raise KeyError(f"Field '{field}' not found in MultiIndex columns.")
    else:
        # Assume it is already a wide single-field frame.
        out = df.copy()

    out = out.sort_index()
    out.index = pd.to_datetime(out.index)

    # Drop tickers that are entirely empty.
    keep = cast(pd.Series, out.notna().any(axis=0))
    out = out.loc[:, keep.to_numpy(dtype=bool)]

    return cast(pd.DataFrame, out.astype("float64"))


def daily_returns(close_total: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns from a wide price matrix. First row is NaN.

    ⚠ Expects the TOTAL-RETURN series (`close_total`), not `close_split`. The maths is the
    same either way, which is exactly the hazard: fed the split-adjusted series this silently
    returns PRICE returns, and every consumer -- momentum, vol, betas, the labels -- is then
    wrong by every dividend ever paid. The signature says which basis it wants; the column
    names say which one you have."""
    return close_total.pct_change(fill_method=None)


def prices_long_to_multiindex(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the long-format prices frame (date, ticker, open, close_split, close_total, ...)
    into a yfinance-style MultiIndex frame: ('CloseSplit', ticker), ('CloseTotal', ticker).

    ⚠ `CloseSplit` and `CloseTotal`, never a bare `Close`. The two bases must stay
    distinguishable all the way to the consumer that picks one -- see the `prices` block in
    sql/schema.sql.
    """

    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"], format="%Y-%m-%d")
    colmap = {c: c.lower() for c in prices.columns}
    prices = prices.rename(columns=colmap)

    fields = {"CloseSplit": "close_split", "CloseTotal": "close_total", "Open": "open"}
    for cap, low in (("High", "high"), ("Low", "low"), ("Volume", "volume")):
        if low in prices.columns:
            fields[cap] = low

    wide = {cap: prices.pivot(index="date", columns="ticker", values=low) for cap, low in fields.items() if low in prices.columns}

    return pd.concat(wide, axis=1)
