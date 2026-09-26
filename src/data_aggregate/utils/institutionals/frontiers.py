"""Completeness-frontier resolution for institutional source families."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pandas as pd

from src.context import Context
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

    filing_dates = insider["filing_date"] if "filing_date" in insider else pd.Series(dtype="datetime64[ns]")
    latest_filing: Any = pd.to_datetime(filing_dates, errors="coerce").max()
    return _normalized_timestamp(latest_filing) if pd.notna(latest_filing) else None


def schedule_complete_through(
    context: Context,
    log: logging.Logger,
    table: Table,
    *,
    expected_ticker_count: int,
) -> pd.Timestamp | None:
    """Trust only a complete manifest frontier for the analysis universe."""
    path = Path(context.paths["DATA_STORE"]) / Path(context.config.local.filename.extraction)
    try:
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        entry: dict[str, Any] = payload.get(table.name) or {}
    except (OSError, ValueError, TypeError):
        log.warning("%s has no readable extraction manifest frontier", table.name)
        return None
    if entry.get("coverage_complete") is not True:
        log.warning(
            "%s manifest predates completeness-sensitive subject discovery; known events remain usable, but absence cannot be emitted as zero",
            table.name,
        )
        return None
    if int(entry.get("ticker_count", -1)) != expected_ticker_count:
        log.warning(
            "%s manifest covers %s ticker(s), analysis universe has %s; zero semantics disabled",
            table.name,
            entry.get("ticker_count"),
            expected_ticker_count,
        )
        return None
    try:
        return _normalized_timestamp(entry["last_run_date"])
    except (KeyError, TypeError, ValueError):
        log.warning("%s completeness manifest has no valid last_run_date", table.name)
        return None
