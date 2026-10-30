"""Calendar boundaries use observed sessions, including holiday-shortened weeks."""

from __future__ import annotations

import numpy as np
import pandas as pd


def due_flags(dates: pd.DatetimeIndex, frequency: str) -> np.ndarray:
    """Element i says whether session i opens a new period versus session i-1; element 0 is False."""
    if frequency == "daily":
        keys = np.arange(len(dates))
    elif frequency == "weekly":
        iso = dates.isocalendar()
        keys = iso.year.to_numpy(np.int64) * 100 + iso.week.to_numpy(np.int64)
    elif frequency == "monthly":
        keys = dates.year.to_numpy(np.int64) * 12 + dates.month.to_numpy(np.int64)
    else:
        raise ValueError(f"Unknown frequency: {frequency}")
    return np.r_[False, keys[1:] != keys[:-1]]


def cash_returns(dates: pd.DatetimeIndex, rate: float) -> np.ndarray:
    """Effective annual cash interest for each whole elapsed calendar interval; element 0 is zero."""
    days = np.r_[0, (dates[1:] - dates[:-1]).days]
    return (1 + rate) ** (days / 365) - 1
