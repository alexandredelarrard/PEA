"""
fetch_splits.py  (src/data_extract/utils/prices/fetch_splits.py)
------------------------------------------------------------------------
Share-split EX-DATES from yfinance, upserted into `prices_splits`.

Its OWN fetcher, not a side effect of the price pull, for the same reason `fetch_dividends`
is: a split frontier is sparse where the price one is daily. It reuses
`download_ohlcv(..., actions=True)` because the `Stock Splits` column is already in that
response -- `fetch_dividends` downloads it and throws it away.

WHY a second split source at all. `sharadar_actions` is fresh but holed: it misses GOOGL
2022-07-18 x20, NVDA 2021-07-20 x4, TSLA 2022-08-25 x3, AVGO 2024-07-15 x10, CMG 2024-06-26
x50, ANET 2024-12-04 x4, BKNG 2026-04-06 x25, MNST 2023-03-28 and 2026-08-11 x2, and AMCR
2026-01-15 x0.2 -- and it carries at least one false positive (SJM 2002-05-30 x0.945, a
merger share-issuance factor yfinance does not report). yfinance has every missing event, so
the two sources cross-validate: agreement means genuine, and a Sharadar-only NON-INTEGER
factor is the false-positive signature. `field_map.split_events` applies that union rule.

⚠ NOT a market-cap input. After the basis fix `close_split` and `sharesbas` both carry the
same retroactive split restatement, so the factor cancels identically and market cap never
reads a split event. This list has exactly three consumers: `sharesOutstandingPit`, the
split-triggered price re-pull in `fetch_prices`, and the prices validator.
"""

from __future__ import annotations

import logging
import pandas as pd

from src.context import Context
from src.data_store.schema import Tables
from src.data_extract.utils.common.run_manifest import record_run
from src.data_extract.utils.prices.fetch_prices import download_ohlcv

logger = logging.getLogger(__name__)

_COLUMNS = ["date", "ticker", "ratio"]
#: How far back a routine run looks. Splits are ANNOUNCED weeks ahead and this runs nightly,
#: so a year is enormous headroom for catching a new one.
#:
#: ⚠ NOT `resume_since`, and that is the whole point. `resume_since` takes the MINIMUM
#: per-ticker frontier, which is right for a daily series and catastrophic for a sparse event
#: one: most of the 343 tickers with splits last split decades ago, so the minimum is ~1998
#: and every nightly run would re-pull 31 years x 491 tickers. The frontier that matters here
#: is the TABLE'S OWN max date -- the last split anyone had -- because a new split lands after
#: it by definition. Use `--full` for the from-scratch backfill.
INCREMENTAL_LOOKBACK_YEARS = 1
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

    s = long_prices[["date", "ticker", _RAW_COLUMN]].rename(columns={_RAW_COLUMN: "ratio"})
    s["ratio"] = pd.to_numeric(s["ratio"], errors="coerce")
    s = s[s["ratio"].notna() & (s["ratio"] != 0.0)]
    if s.empty:
        return pd.DataFrame(columns=_COLUMNS)
    s["date"] = pd.to_datetime(s["date"], format="%Y-%m-%d")
    return s[_COLUMNS].drop_duplicates(subset=["ticker", "date"]).reset_index(drop=True)


def fetch_splits(
    context: Context,
    tickers: list[str],
    years_history: int,
    chunk_size: int = 50,
    pause: float = 2.0,
    full: bool = False,
) -> None:
    """Download share-split ex-dates for `tickers` and upsert them into `prices_splits`.

    A routine run looks back from `min(table max date, today - 1y)`, i.e. AT LEAST a year --
    see `INCREMENTAL_LOOKBACK_YEARS` for why `resume_since` is the wrong frontier here.
    `full=True` re-pulls the whole `years_history` window, which is how the table is first
    populated: on a cold table there is no frontier to resume from at all.

    Call with the EQUITY universe only: the macro series never split."""
    today = pd.Timestamp.today().normalize()
    if full:
        since = today - pd.DateOffset(years=years_history)
    else:
        last = context.store.max_date(Tables.prices_splits)
        since = (today - pd.DateOffset(years=INCREMENTAL_LOOKBACK_YEARS) if last is None
                 else min(pd.Timestamp(last),
                          today - pd.DateOffset(years=INCREMENTAL_LOOKBACK_YEARS)))

    logger.info("Downloading splits for %d tickers since %s", len(tickers), since.date())
    df_downloaded = download_ohlcv(tickers, since, today, chunk_size, pause,
                                   desc="Downloading splits",
                                   auto_adjust=False, actions=True)
    df_splits = _extract_splits(df_downloaded)

    if df_splits.empty:
        logger.warning("no split events returned -- leaving DB table '%s' untouched",
                       Tables.prices_splits)
        record_run(context, Tables.prices_splits, len(tickers), 0)
        return

    n = context.store.save(Tables.prices_splits, df_splits)
    logger.info("Saved %d split rows to DB table '%s' (%d distinct tickers)",
                n, Tables.prices_splits, df_splits["ticker"].nunique())
    record_run(context, Tables.prices_splits, len(tickers), n)
