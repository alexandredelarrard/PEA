"""
quarters.py  (src/utils/quarters.py)
------------------------------------------------------------------------------------
A calendar quarter as ONE monotonically increasing integer: `year * 4 + (quarter - 1)`.

Consecutive quarters differ by exactly 1 across a year boundary (2024Q4 -> 2025Q1), so gap
detection, window checks and "N quarters back" are integer arithmetic rather than date
arithmetic with a special case at December. Nothing here reasons about month lengths or
52/53-week retail calendars; callers pass a date already normalised to a quarter end.

Lives in `src/utils/` because three unrelated subfolders need it -- the Sharadar TTM window
check, the Sharadar completeness gate and the earnings-call gap scan -- and `src/` subfolders
must not import from one another.
"""
from __future__ import annotations

import pandas as pd

#: Quarters in a trailing year. The window width every consecutive-quarter check is measured
#: against.
QUARTERS_PER_YEAR = 4


def quarter_ordinal(dates: pd.Series) -> pd.Series:
    """A column of dates -> their quarter ordinals. Unparseable dates become NA."""
    stamps = pd.to_datetime(dates, errors="coerce")
    return stamps.dt.year * QUARTERS_PER_YEAR + stamps.dt.quarter - 1


def quarter_ordinal_of(year: int, quarter: int) -> int:
    """One `(year, quarter)` pair -> its ordinal."""
    return year * QUARTERS_PER_YEAR + (quarter - 1)


def quarter_label(ordinal: int) -> str:
    """An ordinal -> its `2025Q1` label. The inverse of `quarter_ordinal_of`."""
    return f"{ordinal // QUARTERS_PER_YEAR}Q{ordinal % QUARTERS_PER_YEAR + 1}"
