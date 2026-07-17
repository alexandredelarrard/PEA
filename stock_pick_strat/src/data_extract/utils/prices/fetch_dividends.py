"""
fetch_dividends.py  (src/data_extract/utils/fetch_dividends.py)
---------------------------------------------------------------
Cash-dividend history per ticker via yfinance (free, full history). Saved as a
long parquet [date, ticker, dividend] keyed on the EX-date (known/paid at that
date, so point-in-time and backtestable). Dividends are otherwise invisible to
the model because prices are auto-adjusted.

Re-runs are incremental: only ex-dates after each ticker's cached max are pulled.
Network access is isolated in `_ticker_dividends`; the parse step
(`_series_to_long`) is pure and unit-tested.
"""
from __future__ import annotations

import time

import pandas as pd
import yfinance as yf
from tqdm import tqdm
import logging

from src.context import Context

logger = logging.getLogger(__name__)


def _series_to_long(dividends: pd.Series, ticker: str) -> pd.DataFrame:
    """Convert a yfinance dividends Series (index=ex-date, value=cash/share) into
    long rows [date, ticker, dividend]. Drops non-positive/na entries. Pure."""
    if dividends is None or len(dividends) == 0:
        logger.warning(f"No dividends found for {ticker}")
        return pd.DataFrame(columns=["date", "ticker", "dividend"])
    s = pd.Series(dividends).dropna()
    s = s[s > 0]
    if s.empty:
        return pd.DataFrame(columns=["date", "ticker", "dividend"])
    out = pd.DataFrame({
        "date": pd.to_datetime(s.index).tz_localize(None).normalize(),
        "ticker": ticker,
        "dividend": s.to_numpy(dtype="float64"),
    })
    return out.reset_index(drop=True)


def _ticker_dividends(ticker: str) -> pd.Series:
    """Network call, isolated for testability/mocking. Returns the ex-date series."""
    return yf.Ticker(ticker).dividends


def _load_existing(context: Context) -> pd.DataFrame | None:
    df = context.store.load("dividends")
    if df.empty:
        return None
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


def fetch_dividends(context: Context, tickers: list[str], pause: float = 0.3,
                    refetch_window_days: int = 80) -> pd.DataFrame:
    """Download (incrementally) cash-dividend history for `tickers` and cache it.

    Incremental: a ticker whose most recent cached ex-date is within
    `refetch_window_days` of today is considered CURRENT and is NOT re-downloaded
    (dividends are ~quarterly, so no new one is due yet). Tickers with no cached
    dividend (never paid, or new) are always checked. yfinance `.dividends` has no
    date-range API, so we skip current tickers rather than request a sub-range,
    and only the ex-dates after the cached max are appended.
    """
    existing = _load_existing(context)
    last_by_ticker = ({} if existing is None
                      else existing.groupby("ticker")["date"].max().to_dict())
    today = pd.Timestamp.today().normalize()

    new_frames: list[pd.DataFrame] = []
    skipped = 0

    ticks = [ticker for ticker in tickers if ticker not in context.config.data_extract.other_tickers]
    for tkr in tqdm(ticks, desc="Downloading dividends"):
        cutoff = last_by_ticker.get(tkr)
        if cutoff is not None and (today - cutoff).days <= refetch_window_days:
            skipped += 1                       # already current -> no re-download
            continue
        try:
            long = _series_to_long(_ticker_dividends(tkr), tkr)
        except Exception as e:  # one bad ticker must not abort the whole run
            logger.error(f"Dividends fetch failed for {tkr}: {e}")
            continue
        if long.empty:
            continue
        if cutoff is not None:
            long = long[long["date"] > cutoff]   # append only new ex-dates
        if not long.empty:
            new_frames.append(long)
        time.sleep(pause)
    logger.info(f"Dividends: {skipped}/{len(tickers)} tickers already current (skipped).")

    parts = [df for df in (existing, *new_frames) if df is not None and not df.empty]
    if not parts:
        logger.info("No dividend data available.")
        return pd.DataFrame(columns=["date", "ticker", "dividend"])

    out = (pd.concat(parts, ignore_index=True)
           .drop_duplicates(subset=["ticker", "date"], keep="last")
           .sort_values(["ticker", "date"])
           .reset_index(drop=True))
    # persist only the newly-fetched ex-dates; the DB merges on (ticker, date)
    new = pd.concat(new_frames, ignore_index=True) if new_frames else pd.DataFrame()
    if not new.empty:
        context.store.save("dividends", new)
    print(f"Saved {len(new)} new dividend rows to DB table 'dividends'")
    return out
