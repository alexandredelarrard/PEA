"""Configured source publication clocks, with one calendar buffer and market-session roll."""

from __future__ import annotations

import logging
import re
from datetime import date
from numbers import Number
from typing import Any, cast

import pandas as pd
from omegaconf import DictConfig
from pandas._libs.tslibs.nattype import NaTType

from src.constants.constants import MARKET_TIMEZONE
from src.data_store.schema import Table
from src.utils.data_policy import publication_policy

logger = logging.getLogger(__name__)


def period_end(tag: str) -> date:
    """Calendar end of a quarterly, monthly or semi-monthly published archive unit."""
    match = re.fullmatch(r"(\d{4})(?:(\d{2})([ab])|_(\d{2})|[qQ]([1-4]))", tag)
    if match is None:
        raise ValueError(f"Invalid archive period: {tag!r}")
    year, half_month, half, month, quarter = match.groups()
    number = int(half_month or month or int(quarter) * 3)
    end = pd.Period(year=int(year), month=number, freq="M").end_time.date()
    return end.replace(day=15) if half == "a" else end


def _market_day(value: object) -> pd.Timestamp | NaTType:
    try:
        if isinstance(value, Number):
            return pd.NaT
        # Raw source objects retain pandas' coercion and malformed-value handling.
        if isinstance(value, str) and re.fullmatch(r"\d{4}(?:\d{2}[ab]|_\d{2}|[qQ][1-4])", value):
            value = period_end(value)
        stamp = pd.Timestamp(cast(Any, value))
        if pd.isna(stamp):
            return pd.NaT
        if stamp.tz is not None:
            stamp = stamp.tz_convert(MARKET_TIMEZONE).tz_localize(None)
        return stamp.normalize().as_unit("ns")
    except (ValueError, TypeError, OverflowError):
        return pd.NaT


def market_days(values: pd.Series) -> pd.Series:
    """Date-only source fields keep their day; aware instants use the exchange-local day."""
    mapping = {value: _market_day(value) for value in values.dropna().unique()}
    return pd.to_datetime(values.map(mapping), errors="coerce").dt.as_unit("ns")


def roll_to_session(days: pd.Series, sessions: pd.DatetimeIndex) -> pd.Series:
    """First retained real session on/after each day; an unavailable future session stays unknown."""
    if sessions.tz is not None or sessions.hasnans or sessions.has_duplicates or not sessions.is_monotonic_increasing:
        raise ValueError("Publication requires an ordered unique market calendar")
    result = pd.Series(pd.NaT, index=days.index, dtype="datetime64[ns]")
    if sessions.empty:
        return result
    positions = sessions.searchsorted(pd.DatetimeIndex(days), side="left")
    valid = days.notna() & (positions < len(sessions))
    result.loc[valid] = sessions.take(positions[valid])
    return result


def point_in_time(df_source: pd.DataFrame, config: DictConfig, table: Table, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """Apply the declared source column plus calendar lag once, retaining original evidence."""
    policy = publication_policy(config, table)
    clock = str(policy.column)
    if clock not in df_source:
        raise ValueError(f"{table.name}: configured source column {clock!r} is missing")
    original = f"_source_{clock}"
    if original in df_source:
        raise ValueError(f"{table.name}: source publication has already been applied")
    raw_dates = market_days(df_source[clock])
    invalid = int(raw_dates.isna().sum())
    if invalid:
        logger.warning("%s: excluded %s invalid/missing source date values in %s", table.name, invalid, clock)
    days = raw_dates + pd.Timedelta(days=int(policy.lag_days))
    dates = roll_to_session(days, sessions)
    eligible = dates.notna()
    df_ready = df_source.loc[eligible].copy()
    df_ready[original] = df_ready[clock]
    if clock != "as_of" and "as_of" in df_ready:
        df_ready["_source_as_of"] = df_ready["as_of"]
    df_ready["as_of"] = dates.loc[eligible].array
    return df_ready
