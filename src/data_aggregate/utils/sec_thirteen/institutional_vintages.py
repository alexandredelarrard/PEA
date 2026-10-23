"""Incremental public-book iteration for 13F feature histories."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd

from src.constants.constants import F13_SETTLE_TRADING_DAYS
from src.utils.institutional_vintages import filing_public_dates, public_books


def iter_public_books(
    manifests: pd.DataFrame,
    holdings: pd.DataFrame,
    trading_index: pd.DatetimeIndex,
    *,
    settle_days: int = F13_SETTLE_TRADING_DAYS,
    history_quarters: int | None = 1,
    expiry_quarters: int | None = None,
) -> Iterator[tuple[pd.Timestamp, pd.DataFrame, pd.DataFrame]]:
    """Resolve only changed manager/quarter histories; retain the active dependency quarters."""
    df_manifest = manifests.copy()
    df_manifest["period"] = pd.to_datetime(df_manifest["period"])
    df_manifest["_day"] = filing_public_dates(df_manifest, trading_index, settle_days)
    df_manifest = df_manifest.loc[df_manifest["_day"].notna()]
    groups = df_manifest.groupby(["cik", "period"], sort=False).indices
    positions = holdings.groupby("accession_number", sort=False).indices
    cache: dict[tuple[str, pd.Timestamp], tuple[pd.DataFrame, pd.DataFrame]] = {}
    active = pd.NaT
    publications = df_manifest.groupby("_day", sort=True).indices
    days = pd.DatetimeIndex(sorted(publications))
    if expiry_quarters is not None:
        expiry = days + pd.DateOffset(months=3 * expiry_quarters)
        grid = pd.DatetimeIndex(trading_index).normalize().drop_duplicates().sort_values()
        locations = grid.searchsorted(expiry, side="right")
        days = days.union(grid[locations[locations < len(grid)]]).unique().sort_values()
    for day in days:
        events = df_manifest.iloc[publications.get(day, np.array([], dtype=int))]
        floor = (
            (active.to_period("Q") - history_quarters).end_time.normalize() if pd.notna(active) and history_quarters is not None else pd.Timestamp.min
        )
        for key in events[["cik", "period"]].drop_duplicates().itertuples(index=False, name=None):
            if key[1] < floor:
                continue
            versions = df_manifest.iloc[groups[key]]
            indices = np.concatenate([positions.get(a, np.array([], dtype=int)) for a in versions.accession_number])
            cache[key] = public_books(versions.drop(columns="_day"), holdings.iloc[indices], day, trading_index, settle_days)
        usable = [key[1] for key, (_, state) in cache.items() if state.book_known.any()]
        if usable:
            active = max(usable)
            floor = (active.to_period("Q") - history_quarters).end_time.normalize() if history_quarters is not None else pd.Timestamp.min
        cache = {key: value for key, value in cache.items() if key[1] >= floor}
        if cache:
            books = pd.concat([value[0] for value in cache.values()], ignore_index=True)
            states = pd.concat([value[1] for value in cache.values()], ignore_index=True)
            yield pd.Timestamp(day), books, states
