"""
holdings_clean.py  (src/data_aggregate/utils/institutionals/holdings_clean.py)
-----------------------------------------------------------------------------
The ONE 13F cleaner, for BOTH holdings tables.

`institutional_features` reads `sec13f_hr` (all filers, S&P 500 names only, ~21.7M rows) and
`superinvestor_features` reads `sec13f_manager_holdings` (the roster managers' whole books at
CUSIP grain). Two tables, two grains, and -- until this module -- two hand-maintained cleaners
that shared five of their six steps and drifted apart on the sixth.

What they genuinely differ on is exactly three things, and each is a parameter here:
the dedup KEY, the `position_type == "common"` filter, and whether `cik` needs padding.
Everything else -- the date coercion, the numeric coercion, the required-key dropna, and the
amendment-wins ordering -- is identical and is stated once.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence

import pandas as pd

from src.data_aggregate.utils.common.data_utils import to_day
from src.utils.string import pad_cik

logger = logging.getLogger(__name__)

#: The DEFAULT numeric set: `sec13f_hr`'s four legs. Coerced to numeric and zero-filled when
#: present, CREATED as 0.0 when not -- `_quarter_features` sums `call_value` / `put_value`
#: unconditionally, and on that table a missing option leg means "no options reported", never
#: "unknown".
#:
#: ⚠ THE ELITE TABLE PASSES ITS OWN TWO, and that is not symmetry-breaking for its own sake.
#: `sec13f_manager_holdings` classifies positions with `position_type`, so the cleaner already
#: filters options out by row; manufacturing zero-valued `call_value` / `put_value` columns
#: there would add two all-zero columns to the conviction frame that nothing reads. Measured:
#: passing the 13F default unchanged widened `prim.super_conviction` from 13 to 15 columns.
_NUMERIC_13F = ("shares", "value_usd", "call_value", "put_value")

#: Coerced to a midnight `datetime64` when present. `filing_date` is OPTIONAL on both tables
#: (it is in `sec13f_hr.optional_columns`), which is why this is a membership test and not a
#: bare list comprehension -- reading `h["filing_date"]` on a frame that lacks it raises
#: `KeyError` before the assignment can create it.
_DATES = ("period", "filing_date")


def clean_holdings(holdings: pd.DataFrame, *, key: Sequence[str],
                   numeric: Sequence[str] = _NUMERIC_13F,
                   common_only: bool = False,
                   pad_ciks: bool = False) -> pd.DataFrame:
    """The shared 13F cleaner for BOTH tables.

    `key` is the grain: `sec13f_hr` dedups on `(ticker, cik, period)` and
    `sec13f_manager_holdings` on `(cik, period, cusip)` -- the elite table has NO ticker
    column at all, which is the whole reason the two callers differ.

    `common_only` applies the `position_type == "common"` filter. Elite table only:
    `sec13f_hr` is already classified at extraction, so the column is not there to filter on.

    `pad_ciks` applies `pad_cik`, needed only where the source writes `cik` straight from the
    SEC submission rather than through the padding the roster join expects.

    `numeric` is the set to coerce and zero-fill -- see `_NUMERIC_13F`. The elite table passes
    only `("shares", "value_usd")`, because it resolves options by ROW (`position_type`) and
    two all-zero option columns on its conviction frame would be noise.

    ⚠ `sort_values("filing_date")` BEFORE `drop_duplicates(keep="last")` is what makes an
    AMENDMENT win over the original. Reordering these two lines silently keeps the superseded
    row -- no error, no log, just a stale position. The sort is skipped (and a note logged)
    when the column was projected away, because an unsorted `keep="last"` is arbitrary rather
    than wrong-but-deterministic.

    ⚠ DATES GO THROUGH `to_day`, NEVER `to_datetime(..., format="%Y-%m-%d")`. These columns
    arrive as `DATE`/`TIMESTAMP` where the format is ignored anyway, and on any string value
    carrying a time the format turns it into `NaT` under `errors="coerce"` -- so the row is
    dropped by the `dropna` below and the holding vanishes. `to_day` normalizes instead.
    """
    h = holdings.copy()
    if pad_ciks:
        h["cik"] = h["cik"].map(pad_cik)

    for col in _DATES:
        if col in h.columns:
            h[col] = to_day(h[col])
    for col in numeric:
        h[col] = (pd.to_numeric(h[col], errors="coerce").fillna(0.0) if col in h.columns
                  else pd.Series(0.0, index=h.index))

    h = h.dropna(subset=list(key))
    if common_only and "position_type" in h.columns:
        h = h[h["position_type"].astype(str).str.lower() == "common"]

    if "filing_date" in h.columns:
        h = h.sort_values("filing_date")
    else:
        logger.info("13F clean: no `filing_date` column -> amendments cannot be ordered, so "
                    "`keep='last'` falls back to source order for %s duplicate key(s)",
                    f"{int(h.duplicated(list(key)).sum()):,}")
    return h.drop_duplicates(list(key), keep="last")
