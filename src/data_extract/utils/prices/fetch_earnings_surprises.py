"""Historical earnings surprises (consensus EPS estimate vs reported EPS) per quarter -> `earnings_surprises`.

Source is `yfinance.get_earnings_dates()`. The next, not-yet-reported date arrives as a forward row with a NaN
actual; the upsert on `(ticker, earnings_date)` overwrites it once the actual is reported. Yahoo files a company's
history under its current symbol. Routine candidate checks and explicit history rebuilds use the same configured bounds.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

from src.context import Context
from src.data_extract.utils.common.rate_limit import call_with_retries
from src.data_extract.utils.common.resume import bounded_span, planner_bounds, source_plan
from src.data_store.schema import Tables
from src.utils.data_policy import overlap_column, window_policy
from src.utils.window import Window

_RENAME = {
    "EPS Estimate": "eps_estimate",
    "Reported EPS": "eps_actual",
    "Surprise(%)": "surprise_pct",
}
_COLUMNS = ["ticker", "earnings_date", "eps_estimate", "eps_actual", "surprise_pct"]

# yfinance's `get_earnings_dates` refuses a larger limit ("Yahoo caps limit at 100"), so a longer pull is paged
_YAHOO_MAX_LIMIT = 100


def _download_page(ticker: str, limit: int, offset: int) -> pd.DataFrame | None:
    """One page of a ticker's earnings dates, `offset` rows from the newest, normalized to `_COLUMNS`; None if empty.

    A fresh `yf.Ticker` per page, because yfinance caches earnings dates by `limit` alone and a reused
    instance would return its first page again.
    """
    raw = call_with_retries(
        lambda: yf.Ticker(ticker).get_earnings_dates(limit=limit, offset=offset), retries=3, base_wait=10.0, label=f"earnings {ticker}"
    )
    if raw is None or raw.empty:
        return None

    df = raw.reset_index()
    date_col = "Earnings Date" if "Earnings Date" in df.columns else df.columns[0]
    df = df.rename(columns={date_col: "earnings_date", **_RENAME})
    for c in ("eps_estimate", "eps_actual", "surprise_pct"):
        if c not in df.columns:
            df[c] = np.nan
    df["ticker"] = ticker
    df["earnings_date"] = pd.to_datetime(df["earnings_date"], utc=True).dt.tz_localize(None).dt.normalize()
    return df[_COLUMNS]


def _download_one(ticker: str, window: Window) -> pd.DataFrame | None:
    """Read bounded quarterly history in provider-sized pages, without a moving year/count cap."""
    pages: list[pd.DataFrame] = []
    offset = 0
    oldest: pd.Timestamp | None = None
    while True:
        page = _download_page(ticker, _YAHOO_MAX_LIMIT, offset)
        if page is None:
            break
        edge = pd.Timestamp(page["earnings_date"].min())
        if oldest is not None and edge >= oldest:
            raise ValueError(f"{ticker}: earnings pagination did not advance beyond {oldest.date()}")
        pages.append(page)
        if edge <= window.start or len(page) < _YAHOO_MAX_LIMIT:
            break
        oldest, offset = edge, offset + _YAHOO_MAX_LIMIT
    if not pages:
        return None
    frame = pd.concat(pages, ignore_index=True).drop_duplicates(subset=["ticker", "earnings_date"])
    return frame.loc[window.mask(frame["earnings_date"])].reset_index(drop=True)


def fetch_earnings_surprises(context: Context, tickers: list[str]) -> None:
    """Recheck configured publication candidates; cold/full keys read their fixed history."""
    # Load scoped resume observations before preparing candidate work.
    column = overlap_column(context.config, Tables.earnings_surprises)
    policy = window_policy(context.config, Tables.earnings_surprises, context.run.as_of)
    if policy is None:
        return
    df_stats = context.store.key_stats(Tables.earnings_surprises, "ticker", column, where={"ticker": tickers, column: policy.requested})
    frontier = dict(zip(df_stats["key"], df_stats["last"], strict=True))
    bounds, symbols = planner_bounds(context), context.universe.trading_symbols
    work: list[tuple[str, Window]] = []
    for ticker in tickers:
        plan = source_plan(context, Tables.earnings_surprises, frontier.get(ticker), (("ticker", (ticker,)),))
        if plan is None:
            continue
        end = bounds.get(ticker, (None, None))[1]
        window = bounded_span({ticker: (None, end)}, ticker, *plan.write)
        if window is not None:
            work.append((ticker, window))
    context.log.info("Earnings surprises: %d/%d tickers with configured candidate windows", len(work), len(tickers))

    # Provider-sized pages and both Window bounds are independent of stored forward estimates.
    written, empty, failed = 0, [], []
    for ticker, window in tqdm(work, desc="Fetching earnings-surprise history"):
        try:
            frame = _download_one(symbols.get(ticker, ticker), window)
        except (AttributeError, TypeError, NameError, KeyError, ImportError):
            raise
        except Exception as error:
            context.log.warning("%s: earnings history failed (%s)", ticker, error)
            failed.append(ticker)
            continue
        if frame is None or frame.empty:
            empty.append(ticker)
            continue
        frame = frame.loc[window.mask(frame["earnings_date"]), _COLUMNS].assign(ticker=ticker)
        written += context.store.save(Tables.earnings_surprises, frame)
        time.sleep(0.3)

    # Upserts preserve source observations outside the selected publication window.
    context.log.info("Earnings surprises: %d rows saved; %d empty and %d failed keys", written, len(empty), len(failed))
    if empty or failed:
        context.log.warning("Earnings source unavailable: empty=%s; failed=%s", empty, failed)
