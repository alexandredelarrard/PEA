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
were a year.

That normalisation is NOT clean on the live roster: 180 gap events remain, and the ARQ grain is
one row per FILING, so 543 duplicate `(ticker, calendardate)` groups over 316 tickers arrive as
original + amendment pairs (EDGAR-confirmed on IBM, KO, GOOGL). A repeated quarter fails a
contiguity check exactly as a missing one does, so `_one_row_per_quarter` collapses each
`(ticker, reportperiod)` to its EARLIEST filing before the window maths sees it.

## Why the window is validated on `reportperiod`, not on drift

`calendardate` is a PER-TICKER FISCAL OFFSET, not a bounded normalisation: Sharadar maps AVGO's
2024-02-04 period end FORWARD to 2024-03-31 (+56d) and WMT's 1995-07-31 BACKWARD to 1995-06-30
(-31d). So neither an absolute-drift cap nor a containment test measures anything real. A
45-day cap on `|calendardate - reportperiod|` used to sit here and deleted 239 correct rows
over 4 tickers -- every one of AVGO's, leaving it absent from `fundamentals_history` entirely,
and KR/AZO at 100% NULL revenue. A containment test scores worse still: 6,083 false rejects.

What holds regardless of the filer's calendar is that four consecutive quarters span about a
year of the filer's OWN period ends, so `TTM_SPAN_DAYS` guards the window on
`reportperiod - reportperiod.shift(3)`. It is a tripwire against a spliced window, not a filter:
it rejects nothing on today's data.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

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

#: A trailing twelve must span three quarter-steps of the FILER'S OWN calendar. Measured over
#: the 49,500 windows the contiguity check accepts: min 240 days, median 274 (~39 weeks, i.e.
#: 3 x 13), max 315. It is a TRIPWIRE against a spliced window rather than a filter on today's
#: data -- it rejects 0 of those 49,500.
#:
#: ⚠ The LOW edge sits exactly on the observed minimum, so the margin is one-sided: a window
#: shorter than any seen so far trips it. That is deliberate -- 240 days is ~34 weeks against a
#: normal 39, already an outlier -- but it means a filer changing to a shorter fiscal calendar
#: would null rather than publish. The tripwire LOGS, so that would be visible, not silent.
#:
#: It replaces a 45-day cap on `calendardate` vs `reportperiod`, which measured the wrong
#: thing: Sharadar's normalisation legitimately drifts 56-59 days for a filer whose quarters
#: end early in the calendar quarter (AVGO), and that cap silently deleted 239 correct rows
#: over 4 tickers -- every one of AVGO's, leaving it absent from `fundamentals_history`.
#:
#: Module-local, not in `constants.py`: one non-test consumer, the function below.
TTM_SPAN_DAYS: tuple[int, int] = (240, 320)


def _one_row_per_quarter(frame: pd.DataFrame) -> pd.DataFrame:
    """One ARQ row per `(ticker, reportperiod)` -- the EARLIEST filing.

    Sharadar's ARQ grain is one row per FILING, amendments included. IBM's quarter ended
    2004-09-30 arrives twice, as the 10-Q of 2004-10-28 and the 10-Q/A of 2004-11-01; EDGAR
    confirms the same for KO (10-K/10-K/A, 2002-03-11 and -13) and GOOGL (10-Q/10-Q/A,
    2007-05-09 and -10). A REPEATED quarter fails `_window_is_whole` exactly as a MISSING one
    does, so 543 duplicate groups over 316 tickers were each nulling three trailing twelves.

    The EARLIEST filing wins because it is what the market knew on the day: AR* is as-reported
    and immutable, and taking the amendment would file a later restatement under an earlier
    publication date. 439 of the 543 groups carry identical values, so the choice only bites on
    the 97 that were genuinely restated -- and those remain recoverable from
    `fundamentals_sharadar`, which this transform leaves lossless.

    Keyed on `reportperiod`, NEVER `calendardate`: 7 groups over 4 tickers (BBY, GPN, OKE, KR)
    are two REAL quarters whose fiscal ends normalise onto one calendar quarter, and keying on
    the normalisation would DELETE one of them.
    """
    return (frame.sort_values(["ticker", "reportperiod", "date"])
                 .drop_duplicates(["ticker", "reportperiod"], keep="first"))


def _window_is_whole(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Is each row the end of FOUR CONSECUTIVE quarters spanning a real twelve months?

    Two independent tests, and both are needed:

      * `ordinal - ordinal.shift(3) == 3` on `calendardate` -- the vendor's own quarter LABELS
        are contiguous. A gap, a repeated quarter the dedup could not resolve, or a short
        history all fail it, and all three must: a "TTM" spliced across a missing quarter is a
        15-month number wearing a 12-month label.
      * `reportperiod - reportperiod.shift(3)` inside `TTM_SPAN_DAYS` -- the ECONOMICS really
        do span a year, measured on the FILER'S OWN period ends. This trusts no vendor
        normalisation, so it survives both a 52/53-week calendar and a mid-history fiscal
        calendar change, which every drift-based test measured here does not.

    Both shifts are per TICKER, so one issuer's first quarters can never borrow the previous
    issuer's last ones.

    Returns `(whole, tripwire)`. `tripwire` marks the rows the SPAN alone refused -- contiguous
    labels over an impossible number of days -- so the caller can log it. The two are returned
    together rather than recomputed because the caller would otherwise have to restate this
    shift arithmetic to tell the two refusals apart.
    """
    ordinals = quarter_ordinal(frame["calendardate"])
    by_ticker = frame["ticker"]
    prior = ordinals.groupby(by_ticker, sort=False).shift(TTM_QUARTERS - 1)
    contiguous = (ordinals - prior) == (TTM_QUARTERS - 1)

    reported = pd.to_datetime(frame["reportperiod"], errors="coerce")
    span = (reported - reported.groupby(by_ticker, sort=False).shift(TTM_QUARTERS - 1)).dt.days
    low, high = TTM_SPAN_DAYS
    spans_a_year = span.between(low, high)
    return contiguous & spans_a_year, contiguous & ~spans_a_year


def build_ttm(frame: pd.DataFrame, field_map: FieldMap, *,
              actions: pd.DataFrame | None = None,
              yf_splits: pd.DataFrame | None = None,
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
    point-in-time.

    ⚠ Only ONE column is de-adjusted now: `sharesOutstandingPit`. The four feature columns
    (`sharesOutstanding`, `basicShares`, `dilutedShares`, `dividendsPerShare`) deliberately
    STAY on the vendor's split-adjusted basis, because that is the basis `close_split` is on,
    and on it the future-split factor cancels identically in every price x count product. See
    the `_SPLIT_ADJUSTMENT` block in `sharadar_field_map.json`.

    `actions` and `yf_splits` are the two split sources, unioned under the corroboration rule
    in `split_events` -- `sharadar_actions` alone has nine known holes. Passing both as None
    leaves `sharesOutstandingPit` on the vendor basis, which is NOT point-in-time;
    `deadjust_splits` warns when it happens.
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

    # De-duplicate FIRST, then sequence on `reportperiod` -- the filer's own period ends, which
    # are unique after the dedup and so give a deterministic order even where two rows share a
    # `calendardate` (the class-A collisions).
    out = _one_row_per_quarter(frame).sort_values(["ticker", "reportperiod"]) \
                                     .reset_index(drop=True)
    if len(out) < len(frame):
        log.info("%d amended/re-published ARQ row(s) collapsed to one per "
                 "(ticker, reportperiod), earliest filing kept", len(frame) - len(out))

    whole, tripwire = _window_is_whole(out)
    if tripwire.any():
        # Contiguous quarter LABELS over an impossible number of real days: the vendor's
        # normalisation has spliced something. NULL the window -- never raise, never publish --
        # but say so, because a silent filter is the failure mode this file already had once.
        offenders = out.loc[tripwire, ["ticker", "reportperiod", "calendardate"]]
        log.warning("%d window(s) hold four consecutive quarter labels but do not span "
                    "%d-%d days of the filer's own calendar; nulled:\n%s",
                    int(tripwire.sum()), *TTM_SPAN_DAYS, offenders.to_string())

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
    result = deadjust_splits(result, field_map, actions, yf_splits, report=report)
    return apply_derived(result, field_map)
