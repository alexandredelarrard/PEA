"""The publication clock of the quarterly cluster features and their expansion onto trading days.

A quarter is published once, on the first trading day on or after its filing deadline (45 days, rolled to a trading day) plus the `sec13f_hr`
publication lag of `data.yml`. From then until the next quarter is published, and at most `max_age_days` calendar days, its frame is the value of
every ticker it holds a row for.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.data_store.schema import Tables
from src.utils.publication import point_in_time, roll_to_session

#: Form 13F-HR is due 45 calendar days after the report quarter end (SEC rule; a weekend or holiday deadline rolls to the next session).
FILING_DEADLINE_DAYS = 45


def _sessions(trading_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(trading_index).normalize().unique().sort_values()


def filing_deadlines(periods: pd.DatetimeIndex, trading_index: pd.DatetimeIndex) -> pd.Series:
    """Report period -> the last day a filing counts for it: the 45-day deadline rolled to the next trading day, NaT past the calendar.

    The one deadline of a quarter: the books keep the filings made on or before it and the publication clock starts from it.
    """
    index = pd.DatetimeIndex(periods).normalize()
    days = pd.Series(index + pd.Timedelta(days=FILING_DEADLINE_DAYS), index=index)
    return roll_to_session(days, _sessions(trading_index))


def publication_dates(periods: pd.DatetimeIndex, trading_index: pd.DatetimeIndex, config: DictConfig) -> pd.Series:
    """Report period -> publication day, indexed by `periods`: the filing-deadline trading day taken as a `sec13f_hr` filing date and put through
    its configured `pit` (the lag in calendar days, then the first session on or after).

    A period whose publication day falls after the last trading day is NaT.
    """
    # compute: the shared deadline day as the source clock of the table, then its configured publication policy
    deadline = filing_deadlines(periods, trading_index)
    source = pd.DataFrame({"period": deadline.index, "filing_date": deadline.to_numpy()}).dropna()
    prepared = point_in_time(source, config, Tables.sec13f_hr, _sessions(trading_index))

    # post-process: one day per period, NaT where the policy leaves none
    return pd.Series(prepared["as_of"].to_numpy(), index=pd.DatetimeIndex(prepared["period"])).reindex(deadline.index).astype("datetime64[ns]")


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
