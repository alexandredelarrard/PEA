"""
build_ttm.py  (src/data_extract/utils/fundamentals_sharadar/build_ttm.py)
------------------------------------------------------------------------------------
The repo-named DISCRETE-QUARTER frame -> the repo's TTM/instant contract (D17).

Two rules, and they are the SEC path's, not new ones:

  * a DURATION field is the SUM OF FOUR DISCRETE QUARTERS -- never a carried-forward annual.
    Fewer than four and the value is NULL, which is `periods.INSUFFICIENT_QUARTERS`. That
    fallback is what froze 1,622 of 26,242 consecutive `totalRevenue` pairs in the legacy
    table (6.2%; APA 100%, XOM 36%) and made `revenueGrowth` exactly 0 for three quarters in
    four. Coverage drops here on purpose.
  * an INSTANT field is the PERIOD-END value, never an average of the window.

And a third the two share counts need: a WEIGHTED-AVERAGE count is not additive. Summing four
quarterly averages gives four times the year's average, not a trailing twelve, so `basicShares`
and `dilutedShares` take the four-quarter MEAN -- the SEC path's `TTM_FOUR_QUARTER_MEAN`. The
catalogue already declares them `not_additive`; `field_map._basis_for` reads that flag, so the
two layers cannot drift.

⚠ **Sharadar's `ART` is NOT used** (D17). Sharadar documents it as *not* equal to the sum of
four ARQ rows, so taking it would silently redefine every duration column in the table -- and
it would do so invisibly, because the numbers would still look like plausible trailing twelves.

## What "four quarters" is measured on

`calendardate`, which Sharadar has already normalised to the nearest calendar quarter-end. A
window is four rows whose normalised quarters are CONSECUTIVE -- not four rows that happen to
be adjacent in the sort, which would splice a gap shut and report a 15-month "TTM" as if it
were a year. Phase 2's completeness gate measured 0 missing and 0 duplicate normalised
quarters over all 30 tickers, so the normalisation is sound on this roster; the test is here
because the roster is going to widen.

The 45-day cap is `build_history.TTM_STALENESS_DAYS`, IMPORTED rather than restated. In the
SEC path it asks whether a trailing-twelve window belongs to the same fiscal quarter as the
period it is reported against; here it does the same job one level down, on the only place the
two dates can disagree -- `reportperiod` (the filer's real period end) against `calendardate`
(Sharadar's normalisation of it). Half a quarter can only ever admit the SAME quarter, so a
row whose normalisation moved it further than that is refused rather than counted into a
window it does not belong to.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data_extract.utils.fundamentals.build_history import TTM_STALENESS_DAYS
from src.data_extract.utils.fundamentals.periods import TTM_QUARTERS
from src.data_extract.utils.fundamentals_sharadar.field_map import (
    DURATION, INSTANT, MEAN, FieldMap, TranslationReport, apply_derived, deadjust_splits)
from src.utils.quarters import quarter_ordinal

log = logging.getLogger(__name__)

#: The only dimension this builds from. ARQ rows are AS-REPORTED and immutable; MRQ restates
#: in place, and ART is a vendor trailing twelve that is not the sum of four ARQ.
ARQ = "ARQ"

#: The frame's own key columns, carried through untouched.
TTM_KEYS: tuple[str, ...] = ("ticker", "date", "reportperiod", "calendardate", "fiscalperiod")


def _window_is_whole(frame: pd.DataFrame) -> pd.Series:
    """Is each row the end of FOUR CONSECUTIVE normalised quarters, within its own ticker?

    `ordinal - ordinal.shift(3) == 3` is true only when the three preceding rows are the three
    preceding quarters. A gap, a duplicate quarter or a short history all fail it, and all
    three must: a "TTM" spliced across a missing quarter is a 15-month number wearing a
    12-month label. The shift is per TICKER, so one issuer's first quarters can never borrow
    the previous issuer's last ones.
    """
    ordinals = quarter_ordinal(frame["calendardate"])
    prior = ordinals.groupby(frame["ticker"], sort=False).shift(TTM_QUARTERS - 1)
    return (ordinals - prior) == (TTM_QUARTERS - 1)


def _normalisation_is_sane(frame: pd.DataFrame) -> pd.Series:
    """Does Sharadar's `calendardate` name the same quarter as the filer's own period end?

    True where the two are within `TTM_STALENESS_DAYS`. Half a quarter, so the tolerance can
    admit the same quarter and never the previous one -- the SEC path's reasoning, applied to
    the one pair of dates that can disagree here. A 52/53-week filer's ends walk by weeks; a
    misnormalised row moves by a quarter.
    """
    reported = pd.to_datetime(frame["reportperiod"], errors="coerce")
    normalised = pd.to_datetime(frame["calendardate"], errors="coerce")
    drift = (normalised - reported).dt.days.abs()
    return drift.isna() | (drift <= TTM_STALENESS_DAYS)


def build_ttm(frame: pd.DataFrame, field_map: FieldMap, *,
              actions: pd.DataFrame | None = None,
              quarter_columns: dict[str, str] | None = None,
              report: TranslationReport | None = None) -> pd.DataFrame:
    """Discrete ARQ quarters -> one TTM/instant row per (ticker, filing date).

    The output carries every column the map declares, on the repo's contract: duration fields
    summed over four consecutive quarters, weighted-average counts averaged over the same
    four, instants read off the period end, and the derived formulas evaluated LAST on the
    result. `sec` and `null` columns pass through as NaN.

    `quarter_columns` are the two slices that expose the single quarter beside the TTM line it
    was cut from; they are read off the map's `op == "quarter"` entries by default.

    ⚠ `actions` is here, rather than in `translate`, because the SPLIT DE-ADJUSTMENT MUST RUN
    AFTER THE AGGREGATION. Sharadar keeps the whole share block on one basis (today's), so
    de-adjusting each QUARTER first put pre- and post-split numbers into the same four-quarter
    window: NVDA's 2024-08-28 `dilutedShares` came out 8.08bn against a true 24.9bn, and
    `epsDiluted` 6.56 against ~2.16. Aggregating first and de-adjusting the RESULT once, with
    the factor at the row's OWN filing date, is both coherent within the window and correct
    point-in-time. Passing `actions=None` leaves the share block on the vendor's retroactive
    basis, which is NOT point-in-time -- `deadjust_splits` warns when it happens.
    """
    if frame.empty:
        return frame.copy()
    if "dimension" in frame.columns:
        others = sorted(set(frame["dimension"].dropna().unique()) - {ARQ})
        if others:
            raise RuntimeError(f"build_ttm takes ARQ only; the frame also holds {others}. "
                               f"Sharadar's ART is NOT the sum of four ARQ (D17).")

    quarter_columns = quarter_columns or {
        name: spec.inputs[0] for name, spec in field_map.derived.items()
        if spec.op == "quarter"}

    out = frame.sort_values(["ticker", "calendardate"]).reset_index(drop=True)
    sane = _normalisation_is_sane(out)
    if not sane.all():
        log.warning("%d row(s) whose `calendardate` sits more than %d days from their own "
                    "`reportperiod` are excluded from every TTM window",
                    int((~sane).sum()), TTM_STALENESS_DAYS)
        out = out[sane].reset_index(drop=True)

    whole = _window_is_whole(out)

    # Classify every output ONCE, then aggregate per class. Rolling is the slowest path in
    # pandas and there are only two aggregations, so the ~88 mapped columns cost two grouped
    # passes rather than one apiece.
    basis_of: dict[str, str] = {}
    for name, spec in field_map.outputs.items():
        if spec.kind == "derived":
            continue
        if spec.kind in ("sec", "null"):
            basis_of[name] = "null"
            continue
        if spec.basis not in (DURATION, INSTANT, MEAN):
            raise RuntimeError(f"{name} has basis {spec.basis!r}; expected one of "
                               f"{(DURATION, INSTANT, MEAN)}")
        basis_of[name] = spec.basis

    rolling_names = [n for n, basis in basis_of.items() if basis in (DURATION, MEAN)]
    out[rolling_names] = out[rolling_names].astype("float64")
    grouped = out.groupby("ticker", sort=False)
    rolled: dict[str, pd.DataFrame] = {}
    for basis, how in ((DURATION, "sum"), (MEAN, "mean")):
        names = [n for n, b in basis_of.items() if b == basis]
        if not names:
            continue
        # `rolling(4)` without `min_periods` already NULLs a window holding a NaN, which IS
        # the contract: a quarter the zero rule or a correction removed must NOT contribute
        # silently to a sum, it must null the trailing twelve it belongs to.
        window = grouped[names].rolling(TTM_QUARTERS)
        frame_out = (window.sum() if how == "sum" else window.mean())
        rolled[basis] = frame_out.reset_index(level=0, drop=True).reindex(out.index)

    # Assembled in the map's own order, then concatenated ONCE -- inserting ~90 columns one at
    # a time refragments the block manager on every insert.
    columns: dict[str, pd.Series] = {}
    for name, basis in basis_of.items():
        if basis == "null":
            columns[name] = pd.Series(np.nan, index=out.index)
        elif basis == INSTANT:
            columns[name] = out[name].astype("float64")
        else:
            columns[name] = rolled[basis][name].where(whole, np.nan)
    for name, source in quarter_columns.items():
        columns[name] = out[source].astype("float64")

    keys = out[[c for c in TTM_KEYS if c in out.columns]]
    result = pd.concat([keys, pd.DataFrame(columns, index=out.index)], axis=1)

    # De-adjust the AGGREGATE, then derive. Both orderings matter: after the rolling window so
    # the window is on one basis, and before `apply_derived` so `epsDiluted` reads a corrected
    # `dilutedShares` rather than being computed from a hybrid one.
    result = deadjust_splits(result, field_map, actions, report=report)
    return apply_derived(result, field_map)
