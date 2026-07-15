"""
data_utils.py
-------------
Normalize a raw yfinance download into clean wide matrices (dates x tickers)
and compute daily simple returns.

A multi-ticker yfinance download typically has MultiIndex columns like:
    ('Close', 'AAPL'), ('Open', 'AAPL'), ('Close', 'MSFT'), ...
This module flattens that into per-field wide frames.
"""

from __future__ import annotations
import pandas as pd


def extract_field(df: pd.DataFrame, field: str = "Close") -> pd.DataFrame:
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
    out = out.dropna(axis=1, how="all")
    return out.astype("float64")


def daily_returns(close: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns from a wide close matrix. First row is NaN."""
    return close.pct_change(fill_method=None)


def prices_long_to_multiindex(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the long-format prices parquet (date, ticker, open, close, ...)
    into a yfinance-style MultiIndex frame: ('Close', ticker), ('Open', ticker).
    """
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    colmap = {c: c.lower() for c in prices.columns}
    prices = prices.rename(columns=colmap)

    fields = {"Close": "close", "Open": "open"}
    for cap, low in (("High", "high"), ("Low", "low"), ("Volume", "volume")):
        if low in prices.columns:
            fields[cap] = low
    wide = {cap: prices.pivot(index="date", columns="ticker", values=low)
            for cap, low in fields.items()}
    return pd.concat(wide, axis=1)
