"""
institutional_features.py  (src/data_aggregate/utils/institutionals/institutional_features.py)
--------------------------------------------------------------------------------
Broad, point-in-time 13F institutional ownership features (`ic_inst_*`).

The module aggregates manager-level holdings into per-stock measures of
institutional breadth, accumulation, concentration, option positioning,
ownership, and institutional value/flow.

Expected input is one row per manager, security, and quarter:

    [cik, period, ticker, shares, value_usd, call_value, put_value]

`filing_date` is optional. `period` is the reported quarter-end, not the date
on which the information became public.

Point-in-time availability
--------------------------
A quarter is first emitted at `availability_date(period)`: the statutory
45-calendar-day deadline, snapped forward to the trading calendar, plus
`F13_SETTLE_TRADING_DAYS`.

Only filings public by that date enter the first publication. Later filings
are added cumulatively on their actual filing dates and emitted as revisions
when the aggregate move exceeds `F13_REVISION_MIN_MOVE`. They are never
back-dated or permanently discarded.

The module does not emit incomplete pre-deadline aggregates because 13F filing
volume is strongly back-loaded and filer count is not a reliable proxy for
share or dollar completeness.

Quarter-over-quarter features compare against the fully revised previous
quarter. The filing-date band ensures that its revision window closes before
the next quarter's first publication.

Data-quality rules
------------------
* `clean_holdings` applies the configured early/late filing band. Its late edge
  bounds the revision window and prevents stale back-filings from contaminating
  prior-quarter comparisons.
* Prior-quarter share counts are restated onto the current quarter's split
  basis before calculating share growth or manager buying.
* `ic_inst_holders` is a breadth share:
  ticker holders divided by all distinct filers in the quarter.
* Universe-wide coverage holes suppress both levels and deltas.
* Universe-wide coverage breaks suppress deltas only.
* Levels before `INST_LEVEL_FLOOR_PERIOD` and deltas before
  `INST_DELTA_FLOOR_PERIOD` are not emitted.
* Ticker-level deltas are suppressed when the previous quarter has fewer than
  `MIN_PRIOR_HOLDERS`.
* Implausible ownership ratios above `OWNERSHIP_CEILING` are nulled rather than
  clipped.

The resulting panel contains only information public at each `as_of` date,
while retaining legitimate late filings as dated revisions.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.constants.constants import (
    F13_MAX_EARLY_DAYS,
    F13_MAX_LATE_DAYS,
    F13_REVISION_MIN_MOVE,
    F13_SETTLE_TRADING_DAYS,
    SEC_13F_FILING_LAG_DAYS,
)
from src.data_aggregate.utils.common.panel import build_peer_relative_panel
from src.data_aggregate.utils.common.pit import daily_market_cap, fundamentals_to_daily
from src.data_aggregate.utils.common.price_frames import PriceFrames
from src.data_aggregate.utils.institutionals.availability import availability_date
from src.data_aggregate.utils.institutionals.holdings_clean import clean_holdings
from src.data_aggregate.utils.institutionals.split_basis import future_split_factor
from src.data_aggregate.utils.institutionals.value_basis import log_register, repair_value_basis

logger = logging.getLogger(__name__)

#: The first period whose all-filer coverage describes the market rather than the fetch
#: (measured above: 63 -> 2,938 in-universe filers, post-band).
INST_LEVEL_FLOOR_PERIOD = pd.Timestamp("2013-06-30")

#: The first period whose PREDECESSOR is also post-break, so a QoQ delta is meaningful.
INST_DELTA_FLOOR_PERIOD = pd.Timestamp("2013-09-30")

#: D17. A quarter-on-quarter move in the universe-wide filer count larger than this is a
#: coverage discontinuity, not a market event. A research parameter, so the step passes the
#: configured value in; this is the declared default.
COVERAGE_BREAK_DEFAULT = 0.50

#: Minimum number of holders a ticker must have in the PREVIOUS quarter for its
#: quarter-over-quarter features to be considered meaningful.
#:
#: This is the per-ticker counterpart to the universe-wide D16/D17 coverage
#: guards. A newly covered stock can move from a handful of holders to hundreds,
#: causing mechanically extreme share growth, buyer, exit, and flow measures
#: that describe coverage onset rather than institutional activity.
#:
#: When the prior quarter has fewer than this threshold, all `DELTA_FEATURES`
#: and `inst_value_flow` are nulled. Level features remain valid: a stock may
#: genuinely have few institutional holders in that quarter.
#:
#: The threshold of 100 is based on the measured break in delta stability. It is
#: treated as a fixed data-quality boundary, not a research parameter.
MIN_PRIOR_HOLDERS = 100

#: The quarter-over-quarter features. Nulled before `INST_DELTA_FLOOR_PERIOD` (D16) and on a
#: break quarter (D17). `new_buyer_ratio` / `exit_ratio` / `cluster_buying` are all statements
#: about the PREVIOUS quarter's filer set, so they belong here with the arithmetic differences.
DELTA_FEATURES: tuple[str, ...] = (
    "ic_inst_breadth_chg",
    "ic_inst_shares_chg",
    "ic_inst_new_buyer_ratio",
    "ic_inst_exit_ratio",
    "ic_inst_cluster_buying",
    "ic_inst_flow_to_mcap",
)

#: `ic_inst_ownership_pct` above this many times shares outstanding is a denominator defect,
#: not a holder, and the cell is NULLED (never clipped -- see `_capped_ownership`). 2.0 rather
#: than 1.0 because securities lending double-counts a position and 100-130% is a real, widely
#: published reading; 9.68x is not.
OWNERSHIP_CEILING = 2.0

#: The level features. Nulled before `INST_LEVEL_FLOOR_PERIOD` (D16).
LEVEL_FEATURES: tuple[str, ...] = (
    "ic_inst_holders",
    "ic_inst_concentration",
    "ic_inst_net_options_ratio",
    "ic_inst_ownership_pct",
    "ic_inst_value_to_mcap",
)

#: Emission (D27, registry section 0.10), with the two corrections section 0.10a MEASURED on this
#: family: `_xs` earns its column where the raw leg re-ranks pooled, and the discriminator is
#: CARDINALITY, not boundedness. `ic_inst_concentration` (rho 0.556) and `ic_inst_value_to_mcap`
#: (0.607) re-rank heavily and take the percentile leg even though section 0.10 reasoned them as
#: `raw`; `ic_inst_ownership_pct` is the most date-stable of the family (0.930) and its
#: `raw+peers` assignment is confirmed -- institutional ownership is the textbook sector-normed
#: rate, where an absolute 40% means nothing without the sector's norm.
EMISSION: dict[str, str] = {
    "ic_inst_holders": "raw",  # D28 share of the quarter's filers, in [0, 1]
    "ic_inst_breadth_chg": "raw",  # a difference of that share
    "ic_inst_shares_chg": "raw+xs",  # unbounded growth rate, distribution drifts
    "ic_inst_new_buyer_ratio": "raw",  # bounded [0, 1]
    "ic_inst_exit_ratio": "raw",  # bounded [0, 1]
    "ic_inst_cluster_buying": "raw",  # bounded [-1, 1]
    "ic_inst_concentration": "raw+xs",  # Herfindahl; measured rho 0.556
    "ic_inst_net_options_ratio": "raw",  # bounded [-1, 1]
    "ic_inst_ownership_pct": "raw+peers",  # sector-normed rate; measured rho 0.930
    "ic_inst_value_to_mcap": "raw+xs",  # measured rho 0.607
    "ic_inst_flow_to_mcap": "raw+xs",  # mcap-scaled dollar flow, skewed and drifting
}


def _coverage_periods(h: pd.DataFrame, break_pct: float) -> tuple[set, set, pd.DataFrame]:
    """`(hole periods, break periods, the per-period filer table)` -- see the module docstring.

    A HOLE is a quarter under `1 - break_pct` of BOTH neighbours (missing data). A BREAK is any
    quarter whose filer count moved more than `break_pct` against the one before it (a
    discontinuity, which includes both sides of a hole and the 2013 regime start).
    """
    filers = h.groupby("period")["cik"].nunique().sort_index()
    table = pd.DataFrame({"filers": filers})
    table["prev"] = filers.shift(1)
    table["next"] = filers.shift(-1)
    ratio = (filers - table["prev"]).abs() / table["prev"]
    breaks = set(table.index[ratio > break_pct])
    frac = 1.0 - break_pct
    holes = set(table.index[(filers < frac * table["prev"]) & (filers < frac * table["next"])])
    return holes, breaks, table


def _split_factors(h: pd.DataFrame, splits: pd.DataFrame | None) -> dict:
    """`(ticker, period) -> the factor restating the PREVIOUS quarter's share count onto this
    quarter's basis`: the product of the splits effective in `(prev_period, period]`.

    Computed once over the (ticker, period) grid rather than per manager -- the factor is a
    property of the two dates, not of who held the shares.
    """
    pairs = h[["ticker", "period"]].drop_duplicates().sort_values(["ticker", "period"])
    if pairs.empty:
        return {}

    pairs["prev_period"] = pairs.groupby("ticker")["period"].shift(1)
    pairs = pairs.dropna(subset=["prev_period"])
    if pairs.empty:
        return {}
    f_now = future_split_factor(splits, pairs["ticker"], pairs["period"])
    f_prev = future_split_factor(splits, pairs["ticker"], pairs["prev_period"])
    factor = np.divide(f_prev, f_now, out=np.ones_like(f_prev), where=f_now > 0)
    n = int((np.abs(factor - 1.0) > 1e-9).sum())
    if n:
        logger.info(
            "13F split restatement: %s of %s (ticker, quarter) pairs had the prior " "quarter's share count rebased (V5 trigger count)", n, len(pairs)
        )
    return {(t, p): f for t, p, f in zip(pairs["ticker"], pairs["period"], factor, strict=False)}


def _capped_ownership(ratio: pd.DataFrame) -> pd.DataFrame:
    """NULL `ic_inst_ownership_pct` above `OWNERSHIP_CEILING`, and report the count.

    ⚠ ABOVE 100% IS NOT BY ITSELF WRONG, which is why the ceiling is 2.0 and not 1.0.
    Securities lending double-counts a position -- the lender still reports it on its 13F and
    the borrower's buyer reports it too -- so a heavily-lent name legitimately reads 100-130%,
    and data vendors publish such figures. Measured on the 2026-09-12 build: 24,347 of
    1,516,801 cells (1.6%) above 1.05 and 3,623 above 1.30, which is that phenomenon.

    What is NOT credible is the extreme tail: **169 of 1,449,884 cells (0.0117%) above 2.0
    across 9 tickers, topping out at 9.676 (DUK)** -- re-measured 2026-09-14 after the refill,
    the band and the value-unit repair. ⚠ THE MAXIMUM DID NOT MOVE, AND THAT IS THE POINT: the
    1000x defect `value_basis` repairs is in `value_usd`, and this ratio is built from `shares`,
    which that repair never touches. A number that survives a repair of the wrong column
    unchanged is evidence the defect is somewhere else -- here, in the denominator. The tickers
    involved -- DUK, NVDA, SMCI, TSCO, AMCR, ANET and the rest -- all have splits in
    `prices_splits`. That is the recorded
    `sharesOutstandingPit` split defect (right on 40 of 49 sampled names, out by up to 4x on
    multi-split ones, and rescaling by the split factor makes it WORSE) arriving through the
    denominator. It is a fundamentals-layer defect, not one this module can fix.

    So the treatment is D16's own principle applied to a value rather than a date: a plausible
    wrong number is more dangerous than an absent one, and 9.68 is not even plausible. The cell
    is nulled, never clipped -- a clip at 2.0 would invent a 200%-owned company and hand the
    model a fabricated level. This is the same shape as the elite panel's `share-change guard`.
    """
    over = ratio.gt(OWNERSHIP_CEILING)
    n = int(over.to_numpy().sum())
    if n:
        logger.info(
            "ownership guard: %s of %s non-null `ic_inst_ownership_pct` cells (%.3f%%) "
            "above %.1fx shares outstanding -> nulled (the `sharesOutstandingPit` split "
            "defect, not a holder); max seen %.2f",
            f"{n:,}",
            f"{int(ratio.notna().to_numpy().sum()):,}",
            100.0 * n / max(1, int(ratio.notna().to_numpy().sum())),
            OWNERSHIP_CEILING,
            float(np.nanmax(ratio.to_numpy(dtype="float64"))),
        )
    return ratio.mask(over)


def _stamp_availability(
    holdings: pd.DataFrame, trading_index: pd.DatetimeIndex, *, settle_trading_days: int = F13_SETTLE_TRADING_DAYS
) -> pd.DataFrame:
    """Add `first_pub` (the period's availability date) and `as_of` (this ROW's) to `h`.

        first_pub = availability_date(period)
        as_of     = max(first_pub, filing_date)

    ⚠ REVISED, NOT EXCLUDED. A filing that misses its period's availability date is NOT
    dropped: it is real new public information about the quarter and it enters the panel on the
    day it arrives, with `as_of` set to its own filing date. Dropping it instead would make the
    panel permanently blind to 3.34% of 13F shares -- and, worse, blind to exactly the filings
    a daily refresh exists to pick up, including the four quarters where the missing filer is
    Vanguard (see `F13_SETTLE_TRADING_DAYS`).

    ⚠ `NaT` PROPAGATES FROM `first_pub`, WHICH IS WHY THIS IS NOT A BARE `.max(axis=1)`.
    `DataFrame.max` skips NaN, so a period whose availability date is past the end of the
    trading grid would fall back to its raw `filing_date` -- publishing the newest quarter
    weeks early, which is the one thing the whole rule exists to prevent. A row whose
    `filing_date` is absent keeps `first_pub`, matching the filing band, which KEEPS an undated
    row rather than assuming it late.

    ⚠ NO `filing_date` COLUMN MEANS NO CUTOFF IS POSSIBLE. That is a real read on this table
    (`sec13f_hr.optional_columns`) and on fixtures, so it degrades to `as_of = first_pub` for
    every row -- one emission per period, every filing inside it -- and says so.
    """
    h = holdings.copy()
    h["first_pub"] = availability_date(h["period"], trading_index, settle_trading_days=settle_trading_days)

    if "filing_date" not in h.columns:
        logger.info(
            "13F availability: no `filing_date` column -> `as_of` is the period's "
            "availability date for every row; the cutoff CANNOT be applied on this read, so "
            "a late filing is inside the first publication",
        )
        h["as_of"] = h["first_pub"]
    else:
        later = h["filing_date"].notna() & h["first_pub"].notna() & (h["filing_date"] > h["first_pub"])
        h["as_of"] = h["first_pub"].where(~later, h["filing_date"])

    unavailable = h["as_of"].isna()
    if unavailable.any():
        # The trading grid has not reached these periods' availability dates. They cannot be
        # published on any date the panel has a row for, so they are dropped HERE rather than
        # silently reindexed away later -- and `prev` must not carry a period the panel never
        # emitted, or the next quarter's deltas difference against an invisible predecessor.
        logger.info(
            "13F availability: %s row(s) across period(s) %s have no availability date " "inside the trading calendar (it ends %s) -> not emitted",
            f"{int(unavailable.sum()):,}",
            sorted({str(pd.Timestamp(p).date()) for p in h.loc[unavailable, "period"].unique()})[:6],
            str(pd.DatetimeIndex(trading_index).max().date()) if len(trading_index) else "(empty)",
        )
        h = h[~unavailable]
    return h


def _assert_emission_windows_ordered(qf: pd.DataFrame) -> None:
    """Every period's LAST emission must precede the next period's FIRST one.

    ⚠ ASSERTED RATHER THAN TRUSTED, BECAUSE IT DEPENDS ON THE BAND EDGE. `q-1`'s revisions
    stop at `q-1 + 105d` (`F13_MAX_LATE_DAYS`) and `q` first publishes at about `q-1 + 141d`,
    so the clearance is ~36 days and `as_of` is strictly increasing in `period`. That is what
    lets `fundamentals_to_daily`'s `pivot_table(aggfunc="last").ffill()` stay correct with
    several rows per period: no two periods ever compete for one `(ticker, as_of)` cell, so
    "last" is never a coin toss. Widen the band past ~`+130d` and this breaks SILENTLY -- one
    quarter's revision would overwrite the next quarter's first publication and the panel
    would go BACKWARDS in time. This log is the alarm.

    Raw FILING windows do overlap by ~10 days (period 2015-03-31's filings run to 2015-07-10
    and 2015-06-30's start 2015-07-01); that is harmless and is not what this checks. Only the
    EMISSION windows matter.
    """
    span = qf.groupby("period")["as_of"].agg(["min", "max"]).sort_index()
    # `bad` marks the LATER period of an offending pair -- the one whose first publication is
    # at or before its predecessor's last revision -- so the pair is `(index[i - 1], index[i])`.
    bad = np.flatnonzero((span["max"].shift(1) >= span["min"]).to_numpy())
    if len(bad):
        overlaps = [(str(span.index[i - 1].date()), str(span.index[i].date())) for i in bad]
        raise ValueError(
            "13F emission windows OVERLAP across consecutive periods -- a later quarter's "
            f"first publication is at or before an earlier one's last revision: {overlaps[:6]}. "
            "`fundamentals_to_daily` would resolve the collision with `aggfunc='last'` and the "
            "panel would step backwards. Narrow `F13_MAX_LATE_DAYS` or widen the settle."
        )
    duplicated = int(qf.duplicated(["ticker", "as_of"]).sum())
    if duplicated:
        raise ValueError(
            f"13F emission: {duplicated:,} (ticker, as_of) pair(s) are emitted twice, so "
            "`fundamentals_to_daily` would silently keep one of them. Two periods share an "
            "availability date, which the window ordering above should have made impossible."
        )


def _report_first_publication_shortfall(h: pd.DataFrame) -> None:
    """How much of each quarter arrives AFTER its own first publication -- logged per quarter.

    This is the number the availability cutoff makes meaningful. It is NOT an exclusion rate:
    a filing behind its period's first publication is stamped on its own filing date and
    emitted as a REVISION, so what this measures is how much of a quarter the panel learns
    late, not how much it never learns. A quarter whose shortfall jumps is a filing-season
    anomaly worth seeing, and it is the same axis `validate institutionals` V14 scores.

    ⚠ MEASURED ON SHARES, NOT ON `value_usd`, and the docstring of `F13_SETTLE_TRADING_DAYS`
    carries why: a share count is immune to the 1000x unit defect and to price moves, and on
    2020-12-31 the raw-value basis reads a 91%-complete quarter as 33% complete.

    ⚠ `as_of` / `filing_date` ARE OPTIONAL ON THIS TABLE (`sec13f_hr.optional_columns`), so
    the measure has to be skippable -- reading either unguarded raised `KeyError` on every
    fixture that omits them and killed the whole panel to print a diagnostic.
    """
    if not len(h) or not {"as_of", "first_pub"}.issubset(h.columns):
        logger.info("13F availability: no `as_of`/`first_pub` -> the first-publication shortfall cannot be measured on this read")
        return

    late = (h["as_of"] > h["first_pub"]).fillna(False)
    shares = pd.to_numeric(h["shares"], errors="coerce")
    total_sh = float(shares.sum())
    logger.info(
        "13F availability (settle %s trading day(s)): %s of %s filer-row(s) (%.2f%%) and "
        "%.2f%% of SHARES arrive after their period's first publication -> emitted as "
        "revisions on their own filing date, never back-dated",
        F13_SETTLE_TRADING_DAYS,
        f"{int(late.sum()):,}",
        f"{len(h):,}",
        100.0 * float(late.mean()),
        100.0 * float(shares[late].sum()) / total_sh if total_sh > 0 else 0.0,
    )

    per_q = pd.DataFrame({"period": h["period"], "shares": shares, "behind": shares.where(late, 0.0)}).groupby("period").sum()
    shortfall = (per_q["behind"] / per_q["shares"].replace(0.0, np.nan)).dropna().sort_values(ascending=False)
    worst = shortfall.head(6)
    logger.info(
        "13F availability: worst 6 quarters by share of SHARES behind the first publication: %s (universe-wide median %.2f%%)",
        {str(pd.Timestamp(p).date()): f"{100.0 * v:.2f}%" for p, v in worst.items()},
        100.0 * float(shortfall.median()),
    )


def _availability_coverage(h: pd.DataFrame) -> pd.Series:
    """Per period: the share of the PREVIOUS quarter's 13F SHARES held by filers that have
    reported for this quarter by its first publication. Universe-wide, one number per quarter.

    ⚠ WEIGHTED BY THE PRIOR QUARTER, NOT BY THIS ONE, and that is what makes it point-in-time.
    A filer's prior-quarter book is known at `as_of(q)`; this quarter's universe total is not,
    so weighting by it would score the quarter against a number nobody had yet. The prior
    quarter also supplies the only honest denominator for "who is missing": a filer absent
    from `q` contributes its `q-1` size, which is exactly the weight its silence costs.

    ⚠ SHARES, NEVER VALUE. See `F13_SETTLE_TRADING_DAYS`; and see the module docstring for why
    a FILER-COUNT version of this diagnostic is useless -- measured 2026-09-15, the filer-count
    basis never falls below 96.67% while this one reaches 82.58%.

    A DIAGNOSTIC, NOT A GATE, and the measurement is what settles that. With the unit defect
    repaired and the two fetch holes refilled, the fixed `snap + settle` rule puts every
    quarter that has data above 82%, so an adaptive gate has nothing left to fix; its only
    remaining effect would be to defer the same quarters the hole guard already suppresses,
    and two mechanisms nulling one quarter for overlapping reasons is how a guard stops being
    auditable. The number is logged and scored by `validate institutionals` V14 instead.

    Measured 2026-09-15 (52 quarters, in-universe, banded, settle 3): p50 96.12%, p05 85.52%,
    min 82.58% (2026-03-31), 10 quarters below 90% and none below 80%.
    """

    if not len(h) or "first_pub" not in h.columns:
        return pd.Series(dtype="float64")
    # ⚠ AN UNPROJECTED `shares` IS THE FAILURE MODE THIS GUARD EXISTS FOR, and it has already
    # happened once. `clean_holdings` CREATES a missing numeric leg as 0.0 (see `_NUMERIC_13F`),
    # so a read that projected `[ticker, period, cik, filing_date]` arrives here with an
    # all-zero share column, every weight sums to 0, and the coverage comes back EMPTY -- which
    # a caller reads as "no quarter has a prior quarter", a true-sounding sentence about the
    # wrong thing. Raising the distinction to the caller is the point: no data and no basis for
    # the measurement are different answers.
    if float(pd.to_numeric(h["shares"], errors="coerce").abs().sum()) == 0.0:
        logger.info(
            "13F availability coverage: the `shares` column is all zero -- it was almost "
            "certainly projected away and zero-filled by `clean_holdings`, so there is no "
            "basis to weight the diagnostic on. Not measured."
        )
        return pd.Series(dtype="float64")

    # One row per (cik, period): the filer's universe-wide share count, and the first date any
    # of its filings for that period was public.
    per_filer = h.groupby(["cik", "period"], sort=False).agg(shares=("shares", "sum"), pub=("as_of", "min")).reset_index()
    first_pub = h.groupby("period")["first_pub"].first()
    periods = pd.Index(sorted(first_pub.dropna().index))
    pos = {p: i for i, p in enumerate(periods)}
    per_filer["pi"] = per_filer["period"].map(pos)
    per_filer = per_filer.dropna(subset=["pi"])

    # `q-1`'s filers, aligned onto `q`, then matched against the filers public at `q`'s own
    # first publication. `pub <= first_pub(q)` IS "reported by the availability date".
    prior = per_filer[["cik", "pi", "shares"]].assign(pi=lambda x: x["pi"] + 1)
    reported = per_filer.loc[per_filer["pub"] <= per_filer["period"].map(first_pub), ["cik", "pi"]].assign(_seen=True)
    joined = prior.merge(reported, on=["cik", "pi"], how="left")
    # ⚠ `pi + 1` PUTS THE LAST QUARTER'S FILERS ONE PAST THE END, and that row has no quarter
    # to describe. Dropping it BEFORE the groupby rather than trimming the index afterwards is
    # what keeps the index assignment length-safe -- the earlier form filtered the list and not
    # the Series, which would raise on exactly the frame that triggered it.
    joined = joined[joined["pi"] < len(periods)]
    if joined.empty:
        return pd.Series(dtype="float64")
    seen = joined["_seen"].fillna(False).to_numpy(dtype=bool)
    weight = joined.groupby("pi")["shares"].sum().replace(0.0, np.nan)
    held = joined.loc[seen].groupby("pi")["shares"].sum()
    coverage = (held.reindex(weight.index).fillna(0.0) / weight).dropna()
    coverage.index = pd.DatetimeIndex([periods[int(i)] for i in coverage.index])
    return coverage.sort_index()


def _report_availability_coverage(h: pd.DataFrame) -> None:
    """Log `_availability_coverage` beside the filer-count coverage table."""
    coverage = _availability_coverage(h)
    if coverage.empty:
        return
    thin = coverage[coverage < 0.90]
    logger.info(
        "13F availability coverage (share of q-1's SHARES held by filers public by as_of(q)): "
        "median %.2f%%, p05 %.2f%%, min %.2f%% on %s; %s of %s quarter(s) below 90%%%s",
        100.0 * float(coverage.median()),
        100.0 * float(coverage.quantile(0.05)),
        100.0 * float(coverage.min()),
        str(pd.Timestamp(coverage.idxmin()).date()),
        len(thin),
        len(coverage),
        (" -> " + ", ".join(f"{pd.Timestamp(p).date()}={100.0 * v:.1f}%" for p, v in thin.sort_values().head(8).items())) if len(thin) else "",
    )


def _quarter_features(
    h: pd.DataFrame,
    splits: pd.DataFrame | None = None,
    break_pct: float = COVERAGE_BREAK_DEFAULT,
    min_prior_holders: int = MIN_PRIOR_HOLDERS,
    revision_min_move: float = F13_REVISION_MIN_MOVE,
) -> pd.DataFrame:
    """Manager-grain 13F -> one row per `(ticker, period, as_of)` with the eleven features.

    `h` must already carry `as_of` (and `first_pub`) from `availability.availability_date` --
    `build_institutional_feature_panel` stamps them. A frame WITHOUT `as_of` degrades to one
    emission per period on the bare `period + 45d` deadline, which is the pre-availability
    behaviour: it exists for the isolation test and for fixtures, not as a production path.

    ⚠ ONE PERIOD EMITS SEVERAL ROWS, AND EACH IS CUMULATIVE OVER `filing_date <= as_of`. The
    walk advances through a period's availability dates in order, accumulating; every emission
    therefore contains every filing public at its own stamp. That is what makes
    `revision_min_move` safe: skipping an immaterial date DELAYS those filings to the next
    emitted one and never drops them.

    ⚠ `prev` IS THE FULLY REVISED PREVIOUS PERIOD, NOT ITS FIRST PUBLICATION, and that is the
    NAIVE delta basis -- measured (see `F13_SETTLE_TRADING_DAYS`) to beat a matched sample at
    this settle buffer. It is also leak-free rather than merely convenient: `q-1`'s revision
    window closes at `q-1 + 105d` and `q` first publishes at about `q-1 + 141d`, so every
    filing in `prev` was public before `q`'s first emission.
    `test_emission_windows_do_not_overlap` asserts that clearance rather than trusting it.
    """

    holes, breaks, coverage = _coverage_periods(h, break_pct)
    if holes or breaks:
        logger.info(
            "13F coverage guard: %s hole quarter(s) %s (every feature nulled), " "%s break quarter(s) %s (deltas nulled). Filer counts: %s",
            len(holes),
            sorted(str(p.date()) for p in holes),
            len(breaks),
            sorted(str(p.date()) for p in breaks),
            {str(p.date()): int(coverage.loc[p, "filers"]) for p in sorted(holes | breaks)},
        )

    # D28: the denominator that turns the holder COUNT into a breadth share.
    n_filers = coverage["filers"]

    # ⚠ THE PRE-FLOOR CUT RUNS AFTER `_coverage_periods`, NEVER BEFORE IT. The guard is a
    # statement about the TABLE's coverage, and it is what identifies 2013-06-30 as the regime
    # start in the first place; cutting first would delete the evidence for the floor and then
    # report a clean axis. Cutting here instead is provably free: every surviving period keeps
    # its own `n_filers`, and `isin(holes | breaks)` below simply matches nothing for the
    # periods that are gone.
    h = h[h["period"] >= INST_LEVEL_FLOOR_PERIOD]
    factors = _split_factors(h, splits)

    if "as_of" in h.columns:
        walk = h
    else:
        # The degraded path, announced rather than silent -- this IS the leaking behaviour.
        logger.info(
            "13F emission: no `as_of` column -> one emission per period on the bare "
            "period+%sd deadline, with no availability cutoff (every filing for the "
            "period is inside it, including the late ones)",
            SEC_13F_FILING_LAG_DAYS,
        )
        walk = h.assign(as_of=h["period"] + pd.Timedelta(days=SEC_13F_FILING_LAG_DAYS))
    # `kind="stable"` keeps each stamp's rows in the order `clean_holdings` left them, which is
    # what makes the numeric aggregates reproducible across runs rather than merely close.
    walk = walk.sort_values(["ticker", "period", "as_of"], kind="stable")

    rows: list[dict] = []
    n_revisions = n_skipped = 0
    for ticker, tdf in walk.groupby("ticker", sort=False):
        prev: dict = {}
        prev_total = np.nan
        prev_value = np.nan
        prev_period = None
        for p, pdf in tdf.groupby("period", sort=True):
            # remove the nan values before doing any sum, otherwise stops
            # 60680 nans, due to the ratio work before, nulling strange values
            pdf = pdf.loc[pdf["value_usd"].notnull()]
            if pdf.empty:
                continue

            ciks = pdf["cik"].to_numpy()
            shares = pdf["shares"].to_numpy(dtype="float64")
            values = pdf["value_usd"].to_numpy(dtype="float64")
            calls = pdf["call_value"].to_numpy(dtype="float64")
            puts = pdf["put_value"].to_numpy(dtype="float64")
            stamps = pdf["as_of"].to_numpy()

            # END position of each availability date's block, so `[:k]` is "public by `k`".
            ends = np.flatnonzero(np.r_[stamps[1:] != stamps[:-1], True]) + 1
            # Sorting the cik axis ONCE per period keeps the Herfindahl identical to the
            # `groupby("cik")` it replaces: with one row per (ticker, cik, period) the groups
            # are singletons, so the group sum is just the value in cik-sorted order.
            cik_order = np.argsort(ciks, kind="stable")
            cik_list, share_list = ciks.tolist(), shares.tolist()

            n_prev = len(prev)
            has_prev = n_prev > 0
            # The prior quarter's counts, restated onto THIS quarter's split basis.
            factor = factors.get((ticker, p), 1.0) if has_prev else 1.0
            prev_shares = prev_total * factor if has_prev else np.nan
            pool = float(n_filers.get(p, np.nan))
            prev_pool = float(n_filers.get(prev_period, np.nan)) if prev_period is not None else np.nan
            prev_share = n_prev / prev_pool if (has_prev and prev_pool > 0) else np.nan

            # Running per-filer counters, advanced row by row. The SET arithmetic
            # (`cur & prev`, the increaser / decreaser tally) is incremental because each
            # filer joins the aggregate exactly once and never leaves it; only the numeric
            # sums are re-derived over the `[:k]` slice, which is what keeps them
            # bit-identical to a single-stamp aggregation of the same rows.
            n_both = inc = dec = 0
            at = 0
            last_emitted = np.nan
            for k in ends:
                for cik, held in zip(cik_list[at:k], share_list[at:k], strict=False):
                    was = prev.get(cik)
                    if was is not None:
                        n_both += 1
                        restated = was * factor
                        if held > restated:
                            inc += 1
                        elif held < restated:
                            dec += 1
                at = int(k)

                # ⚠ EVERY NUMERIC LEG IS RE-DERIVED OVER `[:k]`, NOT ACCUMULATED ACROSS STAMPS,
                # and that is what makes the isolation test exact rather than approximate: a
                # period with one availability date reduces over the same array, with the same
                # reduction, as a period with twelve. Accumulating `+=` across stamps instead
                # would make a quarter's total depend on how many times it was emitted, at
                # ~1e-16 -- small, and a bit-identity claim that is quietly false.
                inst_shares = float(shares[:k].sum())
                if np.isfinite(last_emitted):
                    move = abs(inst_shares - last_emitted) / last_emitted if last_emitted > 0 else np.inf
                    if move < revision_min_move:
                        n_skipped += 1
                        continue
                    n_revisions += 1

                holders = int(k)
                inst_value = float(values[:k].sum())
                call_v = float(calls[:k].sum())
                put_v = float(puts[:k].sum())
                total_invested = inst_value + call_v + put_v  # long equity + option exposure
                opt_ratio = ((call_v - put_v) / total_invested) if total_invested > 0 else np.nan

                # crowding: Herfindahl of managers' VALUE shares (high = few dominant holders)
                mv = values[cik_order[cik_order < k]]
                tot_mv = float(mv.sum())
                hhi = float(((mv / tot_mv) ** 2).sum()) if tot_mv > 0 else np.nan
                share = holders / pool if pool > 0 else np.nan

                rows.append(
                    {
                        "ticker": ticker,
                        "period": pd.Timestamp(p),
                        "as_of": pd.Timestamp(stamps[k - 1]),
                        # Carried only so the per-ticker coverage-onset guard can be applied
                        # vectorised below; dropped before the frame is returned.
                        "_prev_holders": float(n_prev) if has_prev else np.nan,
                        "ic_inst_holders": share,
                        "inst_shares": inst_shares,
                        "inst_value": inst_value,
                        # net QoQ dollar flow (long value); NaN on the first observed quarter
                        "inst_value_flow": (inst_value - prev_value) if (has_prev and np.isfinite(prev_value)) else np.nan,
                        "ic_inst_breadth_chg": (share - prev_share) if np.isfinite(prev_share) else np.nan,
                        "ic_inst_shares_chg": (inst_shares / prev_shares - 1.0) if (has_prev and prev_shares > 0) else np.nan,
                        "ic_inst_new_buyer_ratio": ((holders - n_both) / holders) if (has_prev and holders > 0) else np.nan,
                        "ic_inst_exit_ratio": ((n_prev - n_both) / n_prev) if has_prev else np.nan,
                        "ic_inst_cluster_buying": ((inc - dec) / holders) if (has_prev and holders > 0) else np.nan,
                        "ic_inst_net_options_ratio": opt_ratio,
                        "ic_inst_concentration": hhi,
                    }
                )
                last_emitted = inst_shares

            # The next quarter differences against this one FULLY REVISED -- see the docstring.
            prev = dict(zip(cik_list, share_list, strict=False))
            prev_total = float(sum(prev.values()))
            prev_value = float(values.sum())
            prev_period = pd.Timestamp(p)

    if n_revisions or n_skipped:
        logger.info(
            "13F emission: %s first publication(s) + %s revision(s) = %s row(s); %s "
            "candidate revision date(s) skipped as immaterial (< %.3f%% of the last "
            "emitted share count -- DEFERRED to the next emission, never dropped)",
            f"{len(rows) - n_revisions:,}",
            f"{n_revisions:,}",
            f"{len(rows):,}",
            f"{n_skipped:,}",
            100.0 * revision_min_move,
        )

    qf = pd.DataFrame(rows)
    if qf.empty:
        return qf
    # D17 / the hole guard, applied on the PERIOD (never on the availability date: two
    # quarters share no date, and the mask has to travel with the quarter it describes).
    is_hole = qf["period"].isin(holes)
    is_break = qf["period"].isin(breaks) | is_hole
    value_cols = ["inst_shares", "inst_value", "inst_value_flow"]
    for c in DELTA_FEATURES + ("inst_value_flow",):
        if c in qf.columns:
            qf.loc[is_break, c] = np.nan
    for c in LEVEL_FEATURES + tuple(value_cols):
        if c in qf.columns:
            qf.loc[is_hole, c] = np.nan

    # D16: a hard cutoff on the PERIOD, so nothing computed off the pre-break regime survives.
    qf.loc[qf["period"] < INST_LEVEL_FLOOR_PERIOD, [c for c in LEVEL_FEATURES if c in qf.columns] + value_cols] = np.nan
    qf.loc[qf["period"] < INST_DELTA_FLOOR_PERIOD, [c for c in DELTA_FEATURES if c in qf.columns] + ["inst_value_flow"]] = np.nan

    # The per-ticker coverage-onset guard (see `MIN_PRIOR_HOLDERS`). Same column set as D17's
    # break guard, applied on the TICKER's own prior filer count instead of the universe's.
    # `_prev_holders` is NaN on a ticker's first observed quarter, where every delta is already
    # NaN; `< floor` is False there, so the first quarter is not double-counted in the log.
    delta_cols = [c for c in DELTA_FEATURES if c in qf.columns] + ["inst_value_flow"]
    thin = qf["_prev_holders"] < min_prior_holders

    # ⚠ REPORT THE ROWS THIS GUARD ACTUALLY REMOVES, NOT THE ROWS IT MATCHES. Most thin
    # quarters are already NaN from D16 or D17, and counting tickers over the whole `thin`
    # mask says "488 tickers" -- essentially the universe, because nearly every name has some
    # early quarter with a handful of filers. Both figures below are taken over `removed`, so
    # the row count and the ticker count describe the same population.
    removed = thin & qf[delta_cols].notna().any(axis=1)
    if removed.any():
        logger.info(
            "13F per-ticker onset guard: %s of %s (ticker, quarter) deltas nulled "
            "-- the prior quarter carried fewer than %s filers for that name "
            "(%s ticker(s))",
            f"{int(removed.sum()):,}",
            f"{len(qf):,}",
            min_prior_holders,
            qf.loc[removed, "ticker"].nunique(),
        )
    qf.loc[thin, delta_cols] = np.nan
    return qf.drop(columns=["_prev_holders"])


def build_institutional_feature_panel(
    frames: PriceFrames,
    holdings: pd.DataFrame | None,
    *,
    shares_out_history: pd.DataFrame | None = None,
    splits: pd.DataFrame | None = None,
    break_pct: float = COVERAGE_BREAK_DEFAULT,
    min_prior_holders: int = MIN_PRIOR_HOLDERS,
    settle_trading_days: int = F13_SETTLE_TRADING_DAYS,
    revision_min_move: float = F13_REVISION_MIN_MOVE,
) -> pd.DataFrame:
    """Long-format 13F feature panel (`f_<name>` per `EMISSION`). Empty if no holdings.

    `shares_out_history` (fundamentals carrying `sharesOutstandingPit` / `sharesOutstanding`)
    enables `ic_inst_ownership_pct`; with `stock_close` it also enables the value/market-cap
    weight and the size-scaled net flow, through a point-in-time daily market cap. `splits`
    (`prices_splits`) restates the prior quarter's share counts -- without it a split reads as
    accumulation, so its absence is logged by the builder rather than silently tolerated.

    `min_prior_holders` is the per-ticker coverage-onset guard (see the constant). It is
    exposed as an argument for ONE reason: a unit fixture holds a handful of synthetic filers,
    so the production floor would null every delta in it and a test of the QoQ arithmetic would
    be asserting against NaN. A test that means to exercise the arithmetic passes 0 and says so;
    a test that means to exercise the guard passes the floor it is testing.

    `settle_trading_days` and `revision_min_move` are the two availability dials, and both are
    research parameters in the sense that they trade STALENESS against COMPLETENESS -- so the
    step passes the configured values in and the constants are the declared defaults, the same
    pattern as `break_pct`. `settle_trading_days=0` with no `filing_date` column reproduces the
    pre-availability behaviour exactly, which is what the isolation test uses.
    """

    # frames data
    peer_dict = frames.peers
    trading_index = frames.trading_index
    close_split = frames.close_split
    level_factor = frames.level_factor

    need = {"cik", "period", "ticker", "shares"}
    if holdings is None or holdings.empty or not need.issubset(holdings.columns):
        return pd.DataFrame(columns=["date", "ticker"])
    if splits is None or splits.empty:
        logger.warning("No `prices_splits` -> 13F share changes are NOT split-restated; a " "20-for-1 split will read as +1,900%% accumulation.")

    # clean and report holdings
    holdings = clean_holdings(holdings, key=("ticker", "cik", "period"), filing_band=(F13_MAX_EARLY_DAYS, F13_MAX_LATE_DAYS))

    # ⚠ THE UNIT REPAIR RUNS BEFORE ANYTHING READS A VALUE, and that ordering is the whole
    # point: `_quarter_features` sums value into `ic_inst_concentration`,
    # `ic_inst_net_options_ratio`, `ic_inst_value_to_mcap` and `ic_inst_flow_to_mcap`.
    # Measured 2026-09-14 on the repaired table, 1.074% of rows (2,458 filings in the
    # divide-by-1000 band) carry 84.08% of the table's filed dollars, against 15.22% for the
    # 95.57% of rows that are already correct. Every value-weighted statement made before this
    # call is a statement about those 2,458 filings.
    value_before = pd.to_numeric(holdings.get("value_usd"), errors="coerce").sum()
    holdings, register = repair_value_basis(holdings, close_split)
    log_register(register, float(value_before), logger)

    holdings = _stamp_availability(holdings, trading_index, settle_trading_days=settle_trading_days)
    _report_first_publication_shortfall(holdings)
    _report_availability_coverage(holdings)

    # build features
    qf = _quarter_features(holdings, splits=splits, break_pct=break_pct, min_prior_holders=min_prior_holders, revision_min_move=revision_min_move)
    if qf.empty:
        return pd.DataFrame(columns=["date", "ticker"])
    _assert_emission_windows_ordered(qf)

    feats = [c for c in EMISSION if c in qf.columns]
    fields = {f: fundamentals_to_daily(qf, f, trading_index) for f in feats}

    if shares_out_history is not None and not shares_out_history.empty:
        # ownership % by SHARES (aggregate 13F shares / shares outstanding)
        inst_sh = fundamentals_to_daily(qf, "inst_shares", trading_index)

        # ⚠ `sharesOutstandingPit`, NOT `sharesOutstanding`.
        shares = fundamentals_to_daily(shares_out_history, "sharesOutstandingPit", trading_index)
        if not shares.empty and shares.notna().any().any():
            fields["ic_inst_ownership_pct"] = _capped_ownership((inst_sh / shares.where(shares > 0)).replace([np.inf, -np.inf], np.nan))

    if shares_out_history is not None and close_split is not None and not close_split.empty:
        # institutional WEIGHT by VALUE and size-scaled net $ flow, via a point-in-time
        # daily market cap (ffilled sharesOutstanding x daily close x S(d)).
        mcap = daily_market_cap(shares_out_history, close_split, level_factor=level_factor)
        if mcap.empty:
            # ⚠ NOT SILENT. An empty return here means `shares_out_history` was projected
            # without `sharesOutstanding` (the VENDOR basis `daily_market_cap` requires, NOT
            # `sharesOutstandingPit`), and the two features below would simply be absent from
            # the cube with nothing in the log saying so.
            logger.warning(
                "daily_market_cap returned no columns (shares_out_history has %s; "
                "it needs `sharesOutstanding`, the VENDOR basis) -> "
                "ic_inst_value_to_mcap / ic_inst_flow_to_mcap are skipped.",
                sorted(shares_out_history.columns),
            )
        else:
            mpos = mcap.where(mcap > 0)
            inst_val = fundamentals_to_daily(qf, "inst_value", trading_index)
            iv = (inst_val / mpos).replace([np.inf, -np.inf], np.nan)
            if iv.notna().any().any():
                fields["ic_inst_value_to_mcap"] = iv
            flow = fundamentals_to_daily(qf, "inst_value_flow", trading_index)
            fm = (flow / mpos).replace([np.inf, -np.inf], np.nan)
            if fm.notna().any().any():
                fields["ic_inst_flow_to_mcap"] = fm

    # D16 again, now on the DAILY grid. The period-space mask above cannot reach the two
    # market-cap-scaled fields (their numerator is a quarterly value but their denominator is
    # a daily close, so `fundamentals_to_daily` is not the only thing that fills them), and a
    # date-space floor is what L1/L9 actually scores.
    #
    # ⚠ THE AVAILABILITY DATE, NOT `period + 45d`. The bare deadline leaves a 2-5 day window in
    # which the two market-cap-scaled fields could carry a value that no emission stands
    # behind -- and L1/L9 now takes the same availability floor, so the two would disagree by
    # exactly that window and the check would fail on a correct panel.
    floors = availability_date(
        pd.DatetimeIndex([INST_LEVEL_FLOOR_PERIOD, INST_DELTA_FLOOR_PERIOD]), trading_index, settle_trading_days=settle_trading_days
    )
    level_floor = floors.get(INST_LEVEL_FLOOR_PERIOD, INST_LEVEL_FLOOR_PERIOD + pd.Timedelta(days=SEC_13F_FILING_LAG_DAYS))
    delta_floor = floors.get(INST_DELTA_FLOOR_PERIOD, INST_DELTA_FLOOR_PERIOD + pd.Timedelta(days=SEC_13F_FILING_LAG_DAYS))
    if pd.isna(level_floor):
        level_floor = INST_LEVEL_FLOOR_PERIOD + pd.Timedelta(days=SEC_13F_FILING_LAG_DAYS)
    if pd.isna(delta_floor):
        delta_floor = INST_DELTA_FLOOR_PERIOD + pd.Timedelta(days=SEC_13F_FILING_LAG_DAYS)
    for name, frame in list(fields.items()):
        if frame is None or frame.empty:
            fields.pop(name)
            continue
        floor = delta_floor if name in DELTA_FEATURES else level_floor
        fields[name] = frame.where(pd.Series(frame.index >= floor, index=frame.index), axis=0)

    fields = {k: v for k, v in fields.items() if v.notna().any().any()}
    if not fields:
        return pd.DataFrame(columns=["date", "ticker"])
    emission = {k: v for k, v in EMISSION.items() if k in fields}
    return build_peer_relative_panel(fields, peer_dict, emission=emission)
