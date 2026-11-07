"""Generic daily date coercion."""

from __future__ import annotations

import pandas as pd


def to_day(s) -> pd.Series:
    """Coerce to a midnight-normalized `datetime64` Series.

    ⚠ NOT `to_datetime(..., format="%Y-%m-%d")`. On the
    `DATE` / `TIMESTAMP` columns these sources actually have, an explicit format is IGNORED,
    so it buys nothing; it does NOT strip a time component, where `.dt.normalize()` does; and
    on a STRING column any value carrying a time silently becomes `NaT` under
    `errors="coerce"`. Every date in this package is joined against a normalized trading
    grid, so midnight is the requirement and the format string is not.

    Takes a Series, so it is for COLUMNS. An already-datetime INDEX is normalized with
    `pd.DatetimeIndex(idx).normalize()` instead -- `to_day` would return a Series and lose it.
    """
    return pd.to_datetime(s, errors="coerce").dt.normalize()
