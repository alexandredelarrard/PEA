"""The publication clock of the quarterly cluster features and their expansion onto trading days.

A quarter is published once, on the first trading day on or after its filing deadline (45 days, rolled to a trading day) plus the buffer. From then
until the next quarter is published, and at most `max_age_days` calendar days, its frame is the value of every ticker it holds a row for.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from src.utils.institutional_vintages import availability_date


def filing_deadlines(periods: pd.DatetimeIndex, trading_index: pd.DatetimeIndex) -> pd.Series:
    """Report period -> the last day a filing counts for it: the 45-day deadline rolled to the next trading day, NaT past the calendar.

    The one deadline of a quarter: the books keep the filings made on or before it and the publication clock starts from it.
    """
    return availability_date(periods, trading_index, settle_trading_days=0)


def publication_dates(periods: pd.DatetimeIndex, trading_index: pd.DatetimeIndex, buffer_days: int) -> pd.Series:
    """Report period -> publication day, indexed by `periods`: the filing-deadline trading day plus `buffer_days`, snapped forward onto a trading day.

    A period whose publication day falls after the last trading day is NaT.
    """
    # compute: the shared deadline day, then the buffer on the calendar
    deadline = filing_deadlines(periods, trading_index)
    grid = pd.DatetimeIndex(trading_index).normalize().unique().sort_values()
    target = (deadline + pd.Timedelta(days=buffer_days)).to_numpy()
    position = grid.searchsorted(target, side="left")

    # post-process: the trading day at the position, NaT past the end
    inside = deadline.notna().to_numpy() & (position < len(grid))
    published = np.where(inside, grid.to_numpy()[np.clip(position, 0, len(grid) - 1)], np.datetime64("NaT"))
    return pd.Series(published, index=deadline.index, dtype="datetime64[ns]")


def expand_daily(quarters: Mapping[pd.Timestamp, pd.DataFrame], trading_index: pd.DatetimeIndex, max_age_days: int) -> pd.DataFrame:
    """The daily long frame (`date`, `ticker`, the quarters' feature columns) of `quarters`, keyed by publication day and indexed by ticker.

    On a trading day only the newest quarter published on or before it counts, and only while at most `max_age_days` calendar days old; a
    ticker the quarter has no row for has no row (a value is never carried over a newer publication).
    """
    # clean: the trading days, the quarters oldest first and the day each is replaced
    days = pd.DatetimeIndex(trading_index).normalize().unique().sort_values()
    starts = sorted(quarters)
    stops = [*starts[1:], pd.Timestamp.max]
    horizon = pd.Timedelta(days=max_age_days)

    # compute: each quarter repeated over the days it is the published one
    blocks = []
    for start, stop in zip(starts, stops, strict=True):
        span = days[(days >= start) & (days < stop) & (days <= start + horizon)]
        frame = quarters[start]
        if span.empty or frame.empty:
            continue
        block = pd.DataFrame(np.tile(frame.to_numpy(dtype="float32"), (len(span), 1)), columns=frame.columns)
        block.insert(0, "ticker", np.tile(frame.index.to_numpy(dtype=object), len(span)))
        block.insert(0, "date", np.repeat(span.to_numpy(), len(frame)))
        blocks.append(block)

    # post-process: one frame
    if blocks:
        return pd.concat(blocks, ignore_index=True)
    columns = next(iter(quarters.values())).columns if quarters else pd.Index([])
    return pd.DataFrame(columns=["date", "ticker", *columns])
