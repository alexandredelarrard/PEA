"""Configured source publication clocks, with one calendar buffer and market-session roll."""

from __future__ import annotations

from datetime import date

import pandas as pd
from omegaconf import DictConfig

from src.constants.constants import MARKET_TIMEZONE
from src.data_store.schema import Table
from src.utils.data_policy import publication_policy
from src.utils.window import Offset


def period_end(tag: str) -> date:
    """Calendar end of a quarterly, monthly or semi-monthly published archive unit."""
    if len(tag) == 7 and tag[-1] in "ab":
        end = pd.Period(year=int(tag[:4]), month=int(tag[4:6]), freq="M").end_time.date()
        return end.replace(day=15) if tag[-1] == "a" else end
    month = int(tag[-2:]) if "_" in tag else int(tag[-1]) * 3
    return pd.Period(year=int(tag[:4]), month=month, freq="M").end_time.date()


def _market_day(value: object) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(value)
    except (ValueError, TypeError, OverflowError):
        return pd.NaT
    if pd.isna(stamp):
        return pd.NaT
    if stamp.tz is not None:
        stamp = stamp.tz_convert(MARKET_TIMEZONE).tz_localize(None)
    return stamp.normalize()


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


def _lag(policy: DictConfig) -> pd.Timedelta:
    offset = Offset(policy.value, policy.unit)
    if offset.unit != "calendar_days":
        raise ValueError("Publication buffers must use calendar days")
    return pd.Timedelta(days=offset.value)


def archive_release_date(policy: DictConfig, period: str) -> pd.Timestamp:
    """Configured notes release estimate, separate from any later observed availability."""
    end = pd.Timestamp(period_end(period))
    if "_" in period:
        return end + pd.offsets.MonthBegin(1) + pd.Timedelta(days=int(policy.monthly_release_day) - 1)
    return end + _lag(policy.quarterly_lag)


def _notes_dates(policy: DictConfig, rows: pd.DataFrame) -> pd.Series:
    releases = {period: archive_release_date(policy, str(period)) for period in rows["period"].dropna().unique()}
    estimated = pd.to_datetime(rows["period"].map(releases))
    filed = market_days(rows[str(policy.clock)]) + _lag(policy.filing_lag)
    return pd.concat([estimated, filed], axis=1).max(axis=1)


def publication_dates(config: DictConfig, table: Table, rows: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.Series:
    """Resolve raw source clocks and inherited policy once; processing time is never a source date."""
    policy = publication_policy(config, table)
    if policy.mode == "notes_release":
        days = _notes_dates(policy, rows)
    elif policy.mode == "half_month_end":
        days = pd.to_datetime(rows[str(policy.clock)].astype(str).map(period_end)) + _lag(policy.lag)
    elif policy.mode in {"direct", "completed_session", "dated_snapshot"}:
        days = market_days(rows[str(policy.clock)])
    else:
        raise ValueError(f"{table.name}: no source publication clock for {policy.mode}")
    later = [market_days(rows[str(column)]) for column in policy.get("later_clocks", []) if column in rows]
    if later:
        days = pd.concat([days, *later], axis=1).max(axis=1)
    if policy.mode == "direct":
        days = days + _lag(policy.lag)
    return roll_to_session(days, sessions)
