"""Incremental public-book iteration for 13F feature histories."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd

from src.constants.constants import F13_SETTLE_TRADING_DAYS
from src.utils.institutional_vintages import prepare_book_vintages, resolve_prepared_books


def _publication_schedule(df_manifest: pd.DataFrame, trading_index: pd.DatetimeIndex, expiry_quarters: int | None) -> tuple[dict, pd.DatetimeIndex]:
    """Schedule public filings and the first trading session after each expiry."""
    publications = df_manifest.groupby("_day", sort=True).indices
    days = pd.DatetimeIndex(sorted(publications))
    if expiry_quarters is not None:
        expiry = days + pd.DateOffset(months=3 * expiry_quarters)
        grid = pd.DatetimeIndex(trading_index).normalize().drop_duplicates().sort_values()
        locations = grid.searchsorted(expiry, side="right")
        days = days.union(grid[locations[locations < len(grid)]]).unique().sort_values()
    return publications, days


def _history_floor(active: Any, history_quarters: int | None) -> pd.Timestamp:
    """Keep the same report-quarter dependency floor, including an unlimited history."""
    return (active.to_period("Q") - history_quarters).end_time.normalize() if pd.notna(active) and history_quarters is not None else pd.Timestamp.min


def _reconcile_publication(
    df_manifest: pd.DataFrame,
    df_holdings: pd.DataFrame,
    filing_groups: dict,
    position_indices: dict,
    publications: dict,
    cache: dict[tuple[str, pd.Timestamp], tuple[pd.DataFrame, pd.DataFrame]],
    active: Any,
    day: pd.Timestamp,
    history_quarters: int | None,
) -> tuple[dict[tuple[str, pd.Timestamp], tuple[pd.DataFrame, pd.DataFrame]], Any]:
    """Resolve changed books and retain the dependency quarters of the latest known report."""
    df_events = df_manifest.iloc[publications.get(day, np.array([], dtype=int))]
    floor = _history_floor(active, history_quarters)
    for key in df_events[["cik", "period"]].drop_duplicates().itertuples(index=False, name=None):
        if key[1] < floor:
            continue
        df_versions = df_manifest.iloc[filing_groups[key]]
        indices = np.concatenate([position_indices.get(accession, np.array([], dtype=int)) for accession in df_versions.accession_number])
        cache[key] = resolve_prepared_books(df_versions, df_holdings.iloc[indices], day)
    usable = [key[1] for key, (_, df_state) in cache.items() if df_state.book_known.any()]
    if usable:
        active = max(usable)
        floor = _history_floor(active, history_quarters)
    cache = {key: value for key, value in cache.items() if key[1] >= floor}
    return cache, active


def _cached_snapshot(cache: dict[tuple[str, pd.Timestamp], tuple[pd.DataFrame, pd.DataFrame]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine retained books and their states in the cache's existing insertion order."""
    df_books = pd.concat([value[0] for value in cache.values()], ignore_index=True)
    df_states = pd.concat([value[1] for value in cache.values()], ignore_index=True)
    return df_books, df_states


def iter_public_books(
    manifests: pd.DataFrame,
    holdings: pd.DataFrame,
    trading_index: pd.DatetimeIndex,
    *,
    settle_days: int = F13_SETTLE_TRADING_DAYS,
    history_quarters: int | None = 1,
    expiry_quarters: int | None = None,
) -> Iterator[tuple[pd.Timestamp, pd.DataFrame, pd.DataFrame]]:
    """Schedule and yield reconciled public-book snapshots without cleaning inside the loop."""
    df_manifest = prepare_book_vintages(manifests, holdings, trading_index, settle_days)
    df_manifest["_day"] = df_manifest["public_at"]
    df_manifest = df_manifest.loc[df_manifest["_day"].notna()]
    filing_groups = df_manifest.groupby(["cik", "period"], sort=False).indices
    position_indices = holdings.groupby("accession_number", sort=False).indices
    publications, days = _publication_schedule(df_manifest, trading_index, expiry_quarters)
    cache: dict[tuple[str, pd.Timestamp], tuple[pd.DataFrame, pd.DataFrame]] = {}
    active = pd.NaT
    for day in days:
        cache, active = _reconcile_publication(
            df_manifest, holdings, filing_groups, position_indices, publications, cache, active, day, history_quarters
        )
        if cache:
            df_books, df_states = _cached_snapshot(cache)
            yield pd.Timestamp(day), df_books, df_states
