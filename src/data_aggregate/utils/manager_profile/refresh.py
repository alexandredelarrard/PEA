"""Which report quarters the manager profile recomputes, and which stored profile rows a recomputation changes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data_aggregate.utils.manager_profile.quarter_stats import STAT_COLUMNS

#: Profile columns of `sec_13f_manager_details`, written by the profile step only.
PROFILE_COLUMNS = ("category", *STAT_COLUMNS, "n_quarters_used")
_KEYS = ["cik", "period"]


@dataclass(frozen=True)
class Refresh:
    """Quarters to pass over, oldest first, and the quarters whose rows are rewritten."""

    compute: list[pd.Timestamp]
    write: frozenset[pd.Timestamp]


def stored_details(details: pd.DataFrame) -> pd.DataFrame:
    """The stored rows with normalised period dates, `source_period` (null reads as `period`) and every profile column present (a column not added yet reads null)."""
    missing = [name for name in (*PROFILE_COLUMNS, "source_period") if name not in details.columns]
    out = details.reindex(columns=[*details.columns, *missing]).assign(period=pd.to_datetime(details["period"]).dt.normalize())
    return out.assign(source_period=pd.to_datetime(out["source_period"]).dt.normalize().fillna(out["period"]))


def _shift(period: pd.Timestamp, quarters: int) -> pd.Timestamp:
    """`period` moved by whole report quarters."""
    return period + pd.offsets.QuarterEnd(quarters)


def plan_refresh(details: pd.DataFrame, category: pd.Series, full: bool, window: int) -> Refresh | None:
    """The quarters to recompute, or None when every row is profiled.

    A row is stale without a stored category or with a category that differs from its current name's. A stale quarter
    rewrites the next `window` quarters; the pass starts `window` quarters earlier.
    """
    # Stale rows: all of them on a full rebuild.
    present = set(details["period"].unique())
    stale = pd.Series(True, index=details.index) if full else details["category"].isna() | details["category"].ne(category)
    quarters = sorted(details.loc[stale, "period"].unique())
    if not quarters:
        return None

    # Pass range and rewritten range, limited to the quarters the table holds.
    compute = {_shift(q, step) for q in quarters for step in range(-window, window + 1)} & present
    write = {_shift(q, step) for q in quarters for step in range(window + 1)} & present
    return Refresh(sorted(compute), frozenset(write))


def profile_frame(details: pd.DataFrame, category: pd.Series, profile: pd.DataFrame, write: frozenset[pd.Timestamp]) -> pd.DataFrame:
    """Profile columns for every stored row of the rewritten quarters; a row without a usable book keeps only its category."""
    rows = details.loc[details["period"].isin(write), _KEYS].assign(category=category)
    out = rows.merge(profile, on=_KEYS, how="left").reindex(columns=[*_KEYS, *PROFILE_COLUMNS])
    out["n_quarters_used"] = out["n_quarters_used"].astype("Int64")
    return out.astype({name: float for name in STAT_COLUMNS})


def changed_rows(profile: pd.DataFrame, details: pd.DataFrame) -> pd.DataFrame:
    """Rows of `profile` that differ from the stored profile columns of `details` (a missing column reads null)."""
    # Stored profile columns of the compared rows, aligned on the key.
    stored = details.reindex(columns=[*_KEYS, *PROFILE_COLUMNS]).merge(profile[_KEYS], on=_KEYS)
    stored = stored.set_index(_KEYS).sort_index()
    new = profile.set_index(_KEYS).sort_index()

    # Equal where both are null or both hold the same value.
    numeric = [name for name in PROFILE_COLUMNS if name != "category"]
    differs = ~(
        (new[numeric].astype(float).to_numpy() == stored[numeric].astype(float).to_numpy())
        | (new[numeric].isna().to_numpy() & stored[numeric].isna().to_numpy())
    )
    same_category = (new["category"] == stored["category"]).to_numpy()
    return new[np.any(differs, axis=1) | ~same_category].reset_index()
