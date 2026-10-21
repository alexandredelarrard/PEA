"""
cv.py  (src/modelling/utils/cv.py)
----------------------------------
Temporal primitives: purged/embargoed walk-forward folds, observed label endpoints and
exponential time-decay sample weights. Random folds are never used:
cross-sectional forward-return labels overlap in time, so a shuffled split leaks the future.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd

CALENDAR_DAYS_PER_YEAR = 365.25


def time_decay_weights(dates: pd.Series, half_life_years: float, reference: pd.Timestamp | None = None) -> np.ndarray:
    """Exponential sample weights by age: w = 0.5 ** (age / half_life).

    ``reference`` defaults to the latest date in ``dates`` (weight = 1.0 there). Age is in
    calendar days; a 2-year half-life gives weight 0.5 for rows exactly two years earlier."""
    if half_life_years <= 0:
        raise ValueError("half_life_years must be positive")
    dt = pd.to_datetime(dates).dt.normalize()
    ref = pd.Timestamp(reference).normalize() if reference is not None else dt.max()
    half_life_days = half_life_years * CALENDAR_DAYS_PER_YEAR
    age = (ref - dt).dt.days.to_numpy(dtype=np.float64)
    age = np.clip(age, 0.0, None)
    return np.power(0.5, age / half_life_days).astype(np.float32)


def purged_wf_splits(
    dates: pd.Series,
    n_splits: int = 5,
    embargo: int = 20,
    *,
    test_windows: list[tuple[str, str]] | None = None,
    calendar: pd.DatetimeIndex | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window walk-forward folds over the unique dates: fold k trains on the first
    k blocks and tests on the next block after skipping `embargo` dates, so no training label
    window overlaps a test date (the embargo must be >= the label horizon)."""
    unique_days = np.sort(np.asarray(pd.unique(dates)))
    if n_splits < 1 or embargo < 0:
        raise ValueError("n_splits must be positive and embargo non-negative")
    if test_windows is not None:
        yield from _windowed_splits(unique_days, test_windows, n_splits, embargo, calendar)
        return
    n = len(unique_days)
    fold = n // (n_splits + 1)
    if fold <= embargo:
        raise ValueError("Not enough dates for the requested n_splits/embargo.")

    for k in range(1, n_splits + 1):
        train_end = fold * k
        test_start = train_end + embargo
        test_end = min(test_start + fold, n)
        if test_start >= n:
            break
        yield unique_days[:train_end], unique_days[test_start:test_end]


def _windowed_splits(
    unique_days: np.ndarray, windows: list[tuple[str, str]], n_splits: int, embargo: int, calendar: pd.DatetimeIndex | None
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    if len(windows) != n_splits:
        raise ValueError("Explicit test windows must match n_splits")
    bounds = [(pd.Timestamp(start), pd.Timestamp(end)) for start, end in windows]
    previous = None
    for start, end in bounds:
        if start > end or (previous is not None and start <= previous):
            raise ValueError("Explicit test windows must be ordered without overlap")
        previous = end
    sessions = pd.DatetimeIndex(unique_days if calendar is None else calendar).normalize().sort_values().drop_duplicates()
    observed = pd.DatetimeIndex(unique_days)
    if not observed.isin(sessions).all():
        raise ValueError("Fold dates must belong to the observed calendar")
    for start, end in bounds:
        test = unique_days[(observed >= start) & (observed <= end)]
        if not len(test):
            raise ValueError("Explicit test window has no usable observations")
        train_end = int(sessions.searchsorted(pd.Timestamp(test[0]))) - embargo
        if train_end <= 0:
            raise ValueError("Not enough dates before the explicit window/embargo")
        train = unique_days[observed < sessions[train_end]]
        if not len(train):
            raise ValueError("Explicit test window has no usable training observations")
        yield train, test


def label_end_dates(calendar: pd.DatetimeIndex, horizon: int) -> pd.Series:
    """Map feature sessions to their final forward-label session on the complete tape."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    sessions = pd.DatetimeIndex(calendar).normalize().drop_duplicates().sort_values()
    return pd.Series(sessions, index=sessions).shift(-horizon)
