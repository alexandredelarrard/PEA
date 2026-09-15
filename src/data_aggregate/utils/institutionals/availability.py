"""
availability.py  (src/data_aggregate/utils/institutionals/availability.py)
--------------------------------------------------------------------------
THE ONE DECLARATION OF WHEN A 13F PERIOD BECOMES PUBLIC.

A 13F-HR reports positions as of the quarter END and is not public until it is filed, which
the statute puts within `period + 45 calendar days`. Turning that into a date the panel can
be stamped on needs two corrections the statute does not make, and both are measured:

  1. THE 45TH DAY IS OFTEN NOT A TRADING DAY -- 13 of 53 quarters (6 Saturdays, 7 Sundays,
     measured 2026-09-15 against `cube_part_prices`' own calendar). The deadline crowd then
     files on the next session, so the date is SNAPPED FORWARD onto the trading calendar.
     Snapping BACKWARDS would be a look-ahead; the next session is the first day the
     information can be acted on, which is the same rule `decay.snap_to_grid` applies to any
     event date that misses a session.
  2. THE DEADLINE ITSELF IS NOT WHEN THE DOLLARS ARRIVE. Filer COUNT completes early and
     SHARES do not -- small filers file early, the mega-filers file at the deadline and
     settle over the following sessions. So the snapped deadline is advanced
     `F13_SETTLE_TRADING_DAYS` trading days, a buffer sized by sweep; its constant carries
     the table.

⚠ SNAP ON THE TRADING CALENDAR, NOT ON `BDay`. A market holiday is a business day and is not
a session, so `BDay` lands the availability date on a day the panel has no row for and the
stamp is then silently carried by the following `reindex`.

⚠ THIS FUNCTION IS IMPORTED, NEVER RESTATED. `validate/institutionals.py` used to re-derive
the snap itself (`grid[np.clip(grid.searchsorted(deadlines))]`), which made the availability
rule two declarations that had to be kept in step by hand -- and the settle buffer existed in
only one of them. The builder stamps with this function and the leakage checks score against
it, so a change to the rule cannot move one without the other.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.constants.constants import F13_SETTLE_TRADING_DAYS, SEC_13F_FILING_LAG_DAYS
from src.data_aggregate.utils.common.data_utils import to_day


def availability_date(
    periods: pd.Series | pd.DatetimeIndex,
    trading_index: pd.DatetimeIndex,
    *,
    settle_trading_days: int = F13_SETTLE_TRADING_DAYS,
) -> pd.Series:
    """The date each 13F `period` becomes public: deadline, snapped forward, plus the settle.

    Returns a `datetime64` Series on `periods`' own index (a `DatetimeIndex` input gets itself
    as the index, so the result is a period -> date lookup).

    ⚠ `NaT` PAST THE END OF THE GRID, NEVER THE LAST SESSION. A period whose availability date
    the calendar has not reached yet is NOT available, and clamping it onto the last trading
    day would publish the newest quarter days-to-weeks early -- the exact look-ahead the snap
    direction is chosen to avoid. `fundamentals_to_daily` reindexes such a stamp away anyway,
    so the honest value costs nothing and the clamped one is a trap.

    ⚠ AND THE FIRST SESSION, WITH NO SETTLE, BEFORE THE START OF THE GRID -- which is the
    opposite case and takes the opposite treatment. `sec13f_hr` carries periods back to
    1987-03-31 while the trading grid starts in 1995, so those deadlines fall off the front.
    They are not "unavailable": they were public years before the panel begins, and the first
    session is the earliest date the panel can carry them. Adding the settle there would say
    the 1987 filing season was still settling on the panel's fourth day, which is how the
    L1/L9 floor came out four sessions later than the panel's own start.

    `settle_trading_days=0` gives the snapped deadline with no buffer. That is the setting the
    ISOLATION TEST uses, not a production option.
    """
    if settle_trading_days < 0:
        raise ValueError(f"settle_trading_days must be >= 0, got {settle_trading_days!r}")
    # A `DatetimeIndex` input gets ITSELF as the index, which is what makes the result a
    # `period -> availability date` lookup a caller can `.map()` with.
    values = to_day(periods if isinstance(periods, pd.Series) else pd.Series(pd.DatetimeIndex(periods), index=pd.DatetimeIndex(periods)))

    grid = pd.DatetimeIndex(trading_index).normalize().unique().sort_values()
    if grid.empty:
        return pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")

    deadline = values + pd.Timedelta(days=SEC_13F_FILING_LAG_DAYS)
    snap = grid.searchsorted(deadline.to_numpy(), side="left")
    # The settle applies only where the deadline is IN the grid's span. Before the start it
    # would push a long-settled 1987 season into the panel's first week (see the docstring).
    pos = np.where(deadline.to_numpy() < grid[0].to_datetime64(), snap, snap + settle_trading_days)
    return pd.Series(
        np.where(
            (pos < len(grid)) & deadline.notna().to_numpy(),
            grid.to_numpy()[np.clip(pos, 0, len(grid) - 1)],
            np.datetime64("NaT"),
        ),
        index=values.index,
        dtype="datetime64[ns]",
    )
