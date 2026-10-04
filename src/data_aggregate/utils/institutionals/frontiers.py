"""Completeness-frontier resolution for institutional source families."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, cast

import pandas as pd

from src.data_store.schema import Table, Tables
from src.data_store.store import DataStore


def _normalized_timestamp(value: Any) -> pd.Timestamp:
    return cast(pd.Timestamp, pd.Timestamp(value)).normalize()


def insider_live_complete_through(
    store: DataStore,
    log: logging.Logger,
    universe: Sequence[str],
) -> pd.Timestamp | None:
    """Return the minimum successful EDGAR scan across the requested universe."""
    expected = set(map(str, universe))
    coverage = store.load(
        Tables.insider_transactions_live_coverage,
        columns=("ticker", "complete_through"),
        where={"ticker": sorted(expected)},
        optional=True,
    )
    if coverage is None or coverage.empty:
        return None
    coverage = coverage.dropna(subset=["ticker", "complete_through"])
    covered = set(coverage["ticker"].astype(str))
    missing = expected - covered
    if missing:
        log.warning(
            "insider EDGAR coverage is missing %d/%d universe ticker(s); live rows are loaded provisionally but cannot advance the family frontier",
            len(missing),
            len(expected),
        )
        return None
    latest_by_ticker = cast(pd.Series, coverage.groupby("ticker")["complete_through"].max())
    per_ticker = pd.to_datetime(
        latest_by_ticker,
        errors="coerce",
    )
    value: Any = per_ticker.min()
    return _normalized_timestamp(value) if pd.notna(value) else None


def insider_complete_through(
    bulk_complete_through: object,
    insider: pd.DataFrame,
    live_complete_through: pd.Timestamp | None = None,
) -> pd.Timestamp | None:
    """Return the latest valid inclusive frontier across bulk and live sources."""
    candidates: list[pd.Timestamp] = []
    if bulk_complete_through is not None:
        try:
            candidates.append(_normalized_timestamp(bulk_complete_through))
        except (TypeError, ValueError):
            pass
    if live_complete_through is not None and pd.notna(live_complete_through):
        candidates.append(_normalized_timestamp(live_complete_through))
    if candidates:
        return max(candidates)
    return None


def schedule_complete_through(store: DataStore, table: Table, last_session: pd.Timestamp | None) -> pd.Timestamp | None:
    """The 13D/13G "absence means zero through date D" frontier, read from `table` alone.

    D is `last_session` when the table's latest `filing_date` (markers included) lies within its
    resume overlap of that session, else the latest `filing_date`. None for an absent or empty table.
    """
    latest = store.max_date(table, "filing_date")
    if latest is None or last_session is None or table.resume is None:
        return latest
    last = _normalized_timestamp(last_session)
    fresh = latest >= last - pd.Timedelta(days=table.resume.overlap_days)
    return max(latest, last) if fresh else latest
