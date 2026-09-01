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
import logging 

logger = logging.getLogger(__name__)


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
    prices["date"] = pd.to_datetime(prices["date"], format="%Y-%m-%d")
    colmap = {c: c.lower() for c in prices.columns}
    prices = prices.rename(columns=colmap)

    fields = {"Close": "close", "Open": "open"}
    for cap, low in (("High", "high"), ("Low", "low"), ("Volume", "volume")):
        if low in prices.columns:
            fields[cap] = low

    wide = {cap: prices.pivot(index="date", columns="ticker", values=low)
            for cap, low in fields.items() if low in prices.columns}
    
    return pd.concat(wide, axis=1)


def get_trading_days(close, market: pd.Series, market_name: str = "market") -> pd.Series:
    """The trading calendar: a boolean over `close.index`, True where the MARKET traded.

    Takes the market series, not a column name. It used to be `close[market_ticker]`, back
    when SPY was stored inside `prices` alongside the equities; the market series now lives in
    `prices_macro`, so the caller reads it and hands it in. Same definition, so the calendar
    is unchanged -- and the quorum denominator is now right, since `close` no longer carries
    four non-equity columns it was counting as stocks."""
    quorum = 0.5 * close.shape[1]
    trading_days = market.reindex(close.index).notna()
    stock_cov = close.notna().sum(axis=1)
    holes = close.index[(~trading_days) & (stock_cov >= quorum)]
    if len(holes):
        logger.warning(
            "%s (market series) missing on %d date(s) where >=50%% of stocks "
            "trade (%s .. %s) -> these dates are dropped for the ENTIRE universe. "
            "Re-run `data_extract macro` to backfill %s.", market_name, len(holes),
            holes.min().date(), holes.max().date(), market_name)

    return trading_days

def _sub(f, universe):
    return f[[c for c in universe if c in f.columns]] if f is not None else None
