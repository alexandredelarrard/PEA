"""Calendar boundaries use observed sessions, including holiday-shortened weeks."""

from __future__ import annotations

import pandas as pd


def is_due(previous: pd.Timestamp, current: pd.Timestamp, frequency: str) -> bool:
    """Decide the next session's schedule using its calendar date only."""
    if frequency == "daily":
        return True
    if frequency == "weekly":
        return previous.isocalendar()[:2] != current.isocalendar()[:2]
    if frequency == "monthly":
        return (previous.year, previous.month) != (current.year, current.month)
    raise ValueError(f"Unknown frequency: {frequency}")


def cash_return(previous: pd.Timestamp, current: pd.Timestamp, rate: float) -> float:
    """Effective annual cash interest for the whole elapsed calendar interval."""
    return (1 + rate) ** ((current - previous).days / 365) - 1
