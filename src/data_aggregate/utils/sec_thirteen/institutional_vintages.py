"""Incremental public-book iteration for 13F feature histories."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.constants.constants import F13_SETTLE_TRADING_DAYS
from src.utils.institutional_vintages import prepare_book_vintages, resolve_prepared_books


@dataclass(frozen=True, slots=True)
class _PublicationSources:
    """Prepared source frames and their fixed filing, position and publication indices."""

    df_manifest: pd.DataFrame
    df_holdings: pd.DataFrame
    filing_groups: Mapping[tuple[str, pd.Timestamp], NDArray[np.intp]]
    position_indices: Mapping[str, NDArray[np.intp]]
    publications: Mapping[pd.Timestamp, NDArray[np.intp]]


def _publication_schedule(df_manifest: pd.DataFrame) -> tuple[dict, pd.DatetimeIndex]:
    """Schedule public filings in chronological order."""
    publications = cast(dict[pd.Timestamp, Any], df_manifest.groupby("_day", sort=True).indices)
    days = pd.DatetimeIndex(sorted(publications))
    return publications, days


def _history_floor(active: Any, history_quarters: int | None) -> pd.Timestamp:
    """Keep the same report-quarter dependency floor, including an unlimited history."""
    return (active.to_period("Q") - history_quarters).end_time.normalize() if pd.notna(active) and history_quarters is not None else pd.Timestamp.min


def _reconcile_publication(
    sources: _PublicationSources,
    cache: dict[tuple[str, pd.Timestamp], tuple[pd.DataFrame, pd.DataFrame]],
    active: Any,
    day: pd.Timestamp,
    history_quarters: int | None,
) -> tuple[dict[tuple[str, pd.Timestamp], tuple[pd.DataFrame, pd.DataFrame]], Any, list[tuple[str, pd.Timestamp]]]:
    """Resolve changed books and retain the dependency quarters of the latest known report."""
    df_events = sources.df_manifest.iloc[sources.publications.get(day, np.array([], dtype=int))]
    floor = _history_floor(active, history_quarters)
    changed = []
    for key in df_events[["cik", "period"]].drop_duplicates().itertuples(index=False, name=None):
        if key[1] < floor:
            continue
        df_versions = sources.df_manifest.iloc[sources.filing_groups[key]]
        indices = np.concatenate([sources.position_indices.get(accession, np.array([], dtype=int)) for accession in df_versions.accession_number])
        cache[key] = resolve_prepared_books(df_versions, sources.df_holdings.iloc[indices], day)
        changed.append(key)

    # Unlimited archives retain every resolved key; their consumer ignores the active watermark.
    if history_quarters is None:
        return cache, active, changed

    usable = [key[1] for key, (_, df_state) in cache.items() if df_state.book_known.any()]
    if usable:
        active = max(usable)
        floor = _history_floor(active, history_quarters)
    cache = {key: value for key, value in cache.items() if key[1] >= floor}
    return cache, active, changed


def _cached_snapshot(cache: dict[tuple[str, pd.Timestamp], tuple[pd.DataFrame, pd.DataFrame]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine retained books and their states in the cache's existing insertion order."""
    df_books = pd.concat([value[0] for value in cache.values()], ignore_index=True)
    df_states = pd.concat([value[1] for value in cache.values()], ignore_index=True)
    return df_books, df_states


def _iter_public_book_cache(
    manifests: pd.DataFrame,
    holdings: pd.DataFrame,
    *,
    history_quarters: int | None = 1,
) -> Iterator[
    tuple[
        pd.Timestamp,
        dict[tuple[str, pd.Timestamp], tuple[pd.DataFrame, pd.DataFrame]],
        dict[tuple[str, pd.Timestamp], tuple[pd.DataFrame, pd.DataFrame]],
    ]
]:
    """Schedule prepared book replacements, retaining unchanged book objects."""
    df_manifest = manifests.copy()
    df_manifest["_day"] = df_manifest["public_at"]
    df_manifest = df_manifest.loc[df_manifest["_day"].notna()]
    filing_groups = df_manifest.groupby(["cik", "period"], sort=False).indices
    position_indices = holdings.groupby("accession_number", sort=False).indices
    publications, days = _publication_schedule(df_manifest)
    sources = _PublicationSources(
        df_manifest,
        holdings,
        cast(Mapping[tuple[str, pd.Timestamp], NDArray[np.intp]], filing_groups),
        cast(Mapping[str, NDArray[np.intp]], position_indices),
        publications,
    )
    cache: dict[tuple[str, pd.Timestamp], tuple[pd.DataFrame, pd.DataFrame]] = {}
    active = pd.NaT
    for day in days:
        cache, active, changed = _reconcile_publication(sources, cache, active, day, history_quarters)
        if cache:
            yield pd.Timestamp(day), cache, {key: cache[key] for key in changed if key in cache}


def iter_public_books(
    manifests: pd.DataFrame,
    holdings: pd.DataFrame,
    trading_index: pd.DatetimeIndex,
    *,
    settle_days: int = F13_SETTLE_TRADING_DAYS,
    history_quarters: int | None = 1,
) -> Iterator[tuple[pd.Timestamp, pd.DataFrame, pd.DataFrame]]:
    """Adapt raw vintages to the existing full reconciled-book snapshot interface."""
    df_manifest = prepare_book_vintages(manifests, holdings, trading_index, settle_days)
    for day, cache, _ in _iter_public_book_cache(df_manifest, holdings, history_quarters=history_quarters):
        df_books, df_states = _cached_snapshot(cache)
        yield day, df_books, df_states
