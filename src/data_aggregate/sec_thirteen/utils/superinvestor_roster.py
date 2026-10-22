"""Frame-based point-in-time roster selection and whole-book manager identity."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from src.utils.string import pad_cik_series
from src.utils.superinvestor_roster import load_superinvestor_overrides, snapshot_as_of


def roster_as_of(history: pd.DataFrame | None, as_of: object = None, *, config_dir: str | Path | None = None) -> set[str]:
    """The manager IDs on the roster at `as_of` -- THE point-in-time accessor.

    A set, so the `brk` / `BRK` duplicate (two Dataroma codes, one manager, CIK
    0001067983) collapses to one member, and so does a manager stored under either filer of
    its chain. Unresolved managers carry a NULL `cik` and are dropped here; the writer is
    what must fail loudly on them (they must never silently shrink the eligible pool, which
    is the survivorship bug this table exists to remove)."""
    snap = snapshot_as_of(history, as_of)
    if snap is None:
        return set()
    overrides = load_superinvestor_overrides(config_dir)
    return {c for raw in snap["cik"].dropna() if (c := overrides.manager_id(raw))}


def first_snapshot_date(history: pd.DataFrame | None) -> pd.Timestamp | None:
    """The earliest `snapshot_date` in the table, or None when it has never been written.

    The roster history starts after the first 13F books, so `roster_as_of(q)` is EMPTY for
    the earliest quarters and a caller that used it unguarded would zero every manager there.
    Callers floor their lookup at this date: extrapolating the oldest roster backwards is a
    compromise, but the alternative is today's roster, which is the survivorship bias this
    table exists to remove."""
    df = history
    if df is None:
        return None
    snapshot_dates = df["snapshot_date"]
    return cast(pd.Timestamp, pd.Timestamp(snapshot_dates.min()))


def to_manager_books(df: pd.DataFrame, config_dir: str | Path | None = None) -> pd.DataFrame:
    """A book read by filer CIK, keyed by manager ID: each chain member's rows are kept only for
    periods inside one of its windows and relabelled to the manager ID; other rows are untouched. Padded
    and unpadded CIK strings both match; `period` may be date, Timestamp or ISO string."""
    overrides = load_superinvestor_overrides(config_dir)
    windows = [(mid, w) for mid, chain in overrides.manager_ciks.items() for w in chain]
    if df.empty or not windows:
        return df
    padded = pad_cik_series(df["cik"]).to_numpy()
    chained = np.isin(padded, [w.cik for _, w in windows])
    if not chained.any():
        return df
    periods = pd.to_datetime(df["period"].where(chained)).to_numpy()
    label = np.full(len(df), None, dtype=object)
    for mid, w in windows:  # a returning filer has several windows
        hit = padded == w.cik
        if w.window.start is not None:
            hit &= periods >= np.datetime64(w.window.start)
        if w.window.end is not None:
            hit &= periods <= np.datetime64(w.window.end)
        label[hit] = mid
    keep = ~chained | pd.notna(label)
    out = df[keep].copy()
    out["cik"] = np.where(chained[keep], label[keep], out["cik"].to_numpy(dtype=object))
    return out
