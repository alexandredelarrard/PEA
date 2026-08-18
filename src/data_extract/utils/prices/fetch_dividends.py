"""
fetch_dividends.py  (src/data_extract/utils/prices/fetch_dividends.py)
------------------------------------------------------------------------
Cash-dividend history keyed on the EX-date (known/paid at that date, so
point-in-time and backtestable). Dividends are otherwise invisible to the model
because prices are auto-adjusted.

Its OWN fetcher, not a side effect of the price pull: it resumes from the
`dividends` table's own per-ticker max ex-date, a much sparser frontier than the
daily price one (ex-dates are quarterly). It reuses `download_ohlcv` because the
yfinance `actions=True` response already carries the ex-dates next to the OHLCV.

Never-payers are the trap here: a ticker that pays nothing will NEVER have a row,
so the resume scan ignores tickers absent from the table (`include_missing=False`).
Counting them as "needs full history" would pin every run to the whole
`years_history` window, forever.
"""
from __future__ import annotations

import logging

import pandas as pd

from src.context import Context
from src.data_store.schema import Tables
from src.data_extract.utils.common.incremental import resume_since
from src.data_extract.utils.common.run_manifest import record_run
from src.data_extract.utils.prices.fetch_prices import download_ohlcv

logger = logging.getLogger(__name__)


def _extract_dividends(long_prices: pd.DataFrame | None) -> pd.DataFrame:
    """Pull the (raw, pre-adjust) cash dividends out of a normalized price frame
    that was downloaded with actions=True -> long [date, ticker, dividend], only
    the nonzero ex-dates. Pure. Empty if no dividends column."""
    cols = {"date", "ticker", "dividends"}
    if long_prices is None or long_prices.empty or not cols.issubset(long_prices.columns):
        return pd.DataFrame(columns=["date", "ticker", "dividend"])
    d = long_prices[["date", "ticker", "dividends"]].copy()
    d["dividends"] = pd.to_numeric(d["dividends"], errors="coerce")
    d = d[d["dividends"] > 0].rename(columns={"dividends": "dividend"})
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    return d.reset_index(drop=True)


def fetch_dividends(
    context: Context,
    tickers: list[str],
    years_history: int,
    chunk_size: int = 50,
    pause: float = 2.0,
) -> pd.DataFrame:
    """Download cash dividends for `tickers` and upsert the nonzero ex-dates into
    the `dividends` table, returning the freshly-extracted rows.

    Call with the EQUITY universe only: the market/macro tickers (SPY, ^VIX, FX,
    commodities) pay nothing and would just cost a download."""
    if not tickers:
        logger.warning(f"No tickers given — nothing to fetch into '{Tables.dividends}'")
        return pd.DataFrame(columns=["date", "ticker", "dividend"])

    today = pd.Timestamp.today().normalize()
    since = resume_since(context, Tables.dividends, tickers, years_history,
                         include_missing=False)

    logger.info(f"Downloading dividends for {len(tickers)} tickers since {since.date()}")
    df_downloaded = download_ohlcv(tickers, since, today, chunk_size, pause,
                                   desc="Downloading dividends")
    df_dividends = _extract_dividends(df_downloaded)

    # upsert on (ticker, date) — the DB merges with any prior ex-dates
    n = context.store.save(Tables.dividends, df_dividends)
    logger.info(f"Saved {n} dividend rows to DB table '{Tables.dividends}'")
    record_run(context, Tables.dividends, len(tickers), n)

    return df_dividends
