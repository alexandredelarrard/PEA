"""
cv.py  (src/modelling/utils/cv.py)
----------------------------------
Temporal validation primitives: purged + embargoed walk-forward folds, the chronological
early-stopping tail, and exponential time-decay sample weights. Random folds are never used:
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


def purged_wf_splits(dates: pd.Series, n_splits: int = 5, embargo: int = 20) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window walk-forward folds over the unique dates: fold k trains on the first
    k blocks and tests on the next block after skipping `embargo` dates, so no training label
    window overlaps a test date (the embargo must be >= the label horizon)."""
    unique_days = np.sort(np.asarray(pd.unique(dates)))
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


def temporal_valid_split(panel: pd.DataFrame, train_frac: float = 0.9) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the last `1 - train_frac` of the dates (chronologically) for early stopping."""
    dates = np.sort(panel["date"].unique())
    cut = max(1, int(len(dates) * train_frac))
    if cut >= len(dates):
        cut = len(dates) - 1
    train = panel[panel["date"].isin(dates[:cut])]
    valid = panel[panel["date"].isin(dates[cut:])]
    return train, valid
