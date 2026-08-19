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


_COLUMNS = ["date", "ticker", "dividends"]


def _extract_dividends(long_prices: pd.DataFrame | None) -> pd.DataFrame:
    """ Keep 0 dividends as a value, since they are informative anyway. Increase table size,
    but more stable to refresh and merge.

    Empty in, empty out: `download_ohlcv` returns a column-less frame when every chunk
    failed, and a total yfinance outage must no-op rather than KeyError."""

    if long_prices is None or long_prices.empty or "dividends" not in long_prices.columns:
        return pd.DataFrame(columns=_COLUMNS)

    d = long_prices[_COLUMNS].copy()
    d["dividends"] = pd.to_numeric(d["dividends"], errors="coerce")
    d["date"] = pd.to_datetime(d["date"], format="%Y-%m-%d")
    return d.reset_index(drop=True)


def fetch_dividends(
    context: Context,
    tickers: list[str],
    years_history: int,
    chunk_size: int = 50,
    pause: float = 2.0,
) -> None:
    """Download cash dividends for `tickers` and upsert them into the `dividends`
    table (column `dividends`), returning the freshly-extracted rows. Zero-dividend
    days are kept: a 0 is itself informative, and storing it makes the incremental
    refresh idempotent rather than dependent on which bars happened to be nonzero.

    Call with the EQUITY universe only: the market/macro tickers (SPY, ^VIX, FX,
    commodities) pay nothing and would just cost a download."""

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
