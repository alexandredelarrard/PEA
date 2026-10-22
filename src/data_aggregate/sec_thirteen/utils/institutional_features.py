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
* Publication-time universe coverage holes suppress both levels and deltas.
* Publication-time universe coverage breaks suppress deltas only.
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
from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import pandas as pd

from src.constants.constants import F13_SETTLE_TRADING_DAYS, SEC_13F_FILING_LAG_DAYS
from src.data_aggregate.sec_thirteen.utils.holdings_clean import clean_holdings
from src.data_aggregate.sec_thirteen.utils.institutional_vintages import iter_public_books
from src.data_aggregate.sec_thirteen.utils.value_basis import log_register, nullable_value_to_daily, repair_value_basis
from src.data_aggregate.utils.common.availability import availability_date, filing_public_dates, public_books
from src.data_aggregate.utils.common.panel import build_peer_relative_panel
from src.data_aggregate.utils.common.pit import daily_market_cap, fundamentals_to_daily
from src.data_aggregate.utils.common.price_frames import PriceFrames
from src.data_aggregate.utils.common.split_basis import future_split_factor, split_cum_product
from src.data_aggregate.utils.common.typing import frame_column
from src.data_store.schema import Resume, Tables
from src.utils.price_adjustments import ShareBasis

logger = logging.getLogger(__name__)
F13_REVISION_MIN_MOVE = 0.001

#: The first period whose all-filer coverage describes the market rather than the fetch
#: (measured above: 63 -> 2,938 in-universe filers, post-band).
INST_LEVEL_FLOOR_PERIOD = pd.Timestamp(cast(str, cast(Resume, Tables.sec13f_hr.resume).coverage_start))

#: The first period whose PREDECESSOR is also post-break, so a QoQ delta is meaningful.
INST_DELTA_FLOOR_PERIOD = INST_LEVEL_FLOOR_PERIOD + pd.offsets.QuarterEnd(1)

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
#: are nulled. Level features remain valid: a stock may
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

#: Preserve interpretable economic units. Peer variants are not emitted: the fixed candidate
#: has no target/OOS evidence with which to pass the approved retention gate.
EMISSION: dict[str, str] = {
    "ic_inst_holders": "raw",  # D28 share of the quarter's filers, in [0, 1]
    "ic_inst_breadth_chg": "raw",  # a difference of that share
    "ic_inst_shares_chg": "raw",  # an interpretable growth rate
    "ic_inst_new_buyer_ratio": "raw",  # bounded [0, 1]
    "ic_inst_exit_ratio": "raw",  # bounded [0, 1]
    "ic_inst_cluster_buying": "raw",  # bounded [-1, 1]
    "ic_inst_concentration": "raw",  # Herfindahl concentration
    "ic_inst_net_options_ratio": "raw",  # bounded [-1, 1]
    "ic_inst_ownership_pct": "raw",
    "ic_inst_value_to_mcap": "raw",
}


def _coverage_states(h: pd.DataFrame, break_pct: float) -> pd.DataFrame:
    """Public filer count and causal coverage flags at every `(period, as_of)` stamp.

    A downward break is a hole at that publication. A later filing may repair subsequent
    revisions, but cannot rewrite the denominator or quality state already public earlier.
    """
    first_public = cast(pd.DataFrame, h.groupby(["period", "cik"], as_index=False)["as_of"].min())
    new_filers = first_public.groupby(["period", "as_of"]).size().rename("new_filers")
    stamps = h[["period", "as_of"]].drop_duplicates().set_index(["period", "as_of"]).sort_index()
    table = stamps.join(new_filers, how="left")
    table["filers"] = table["new_filers"].fillna(0).groupby(level="period").cumsum()
    final_filers = table.groupby(level="period")["filers"].last()
    previous = final_filers.shift(1)
    table["prev"] = table.index.get_level_values("period").map(previous)
    ratio = (table["filers"] - table["prev"]).abs() / table["prev"]
    table["break"] = ratio > break_pct
    table["hole"] = table["filers"] < (1.0 - break_pct) * table["prev"]
    return table


def _split_factors(h: pd.DataFrame, splits: pd.DataFrame | None) -> dict:
    """`(ticker, period) -> the factor restating the PREVIOUS quarter's share count onto this
    quarter's basis`: the product of the splits effective in `(prev_period, period]`.

    Computed once over the (ticker, period) grid rather than per manager -- the factor is a
    property of the two dates, not of who held the shares.
    """
    pairs = cast(pd.DataFrame, h[["ticker", "period"]]).drop_duplicates().sort_values(["ticker", "period"])
    if pairs.empty:
        return {}

    pairs["prev_period"] = pairs.groupby("ticker")["period"].shift(1)
    pairs = pairs.dropna(subset=["prev_period"])
    if pairs.empty:
        return {}
    f_now = future_split_factor(splits, frame_column(pairs, "ticker"), frame_column(pairs, "period"))
    f_prev = future_split_factor(splits, frame_column(pairs, "ticker"), frame_column(pairs, "prev_period"))
    factor = np.divide(f_prev, f_now, out=np.ones_like(f_prev), where=f_now > 0)
    n = int((np.abs(factor - 1.0) > 1e-9).sum())
    if n:
        logger.info(
            "13F split restatement: %s of %s (ticker, quarter) pairs had the prior quarter's share count rebased (V5 trigger count)", n, len(pairs)
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
    h["first_pub"] = availability_date(frame_column(h, "period"), trading_index, settle_trading_days=settle_trading_days)

    if "filing_date" not in h.columns:
        logger.info(
            "13F availability: no `filing_date` column -> `as_of` is the period's "
            "availability date for every row; the cutoff CANNOT be applied on this read, so "
            "a late filing is inside the first publication",
        )
        h["as_of"] = h["first_pub"]
    else:
        filing_date = frame_column(h, "filing_date")
        first_pub = frame_column(h, "first_pub")
        later = filing_date.notna() & first_pub.notna() & (filing_date > first_pub)
        h["as_of"] = first_pub.where(~later, filing_date)

    unavailable = frame_column(h, "as_of").isna()
    if unavailable.any():
        # The trading grid has not reached these periods' availability dates. They cannot be
        # published on any date the panel has a row for, so they are dropped HERE rather than
        # silently reindexed away later -- and `prev` must not carry a period the panel never
        # emitted, or the next quarter's deltas difference against an invisible predecessor.
        logger.info(
            "13F availability: %s row(s) across period(s) %s have no availability date inside the trading calendar (it ends %s) -> not emitted",
            f"{int(unavailable.sum()):,}",
            sorted({str(pd.Timestamp(p).date()) for p in cast(pd.Series, h.loc[unavailable, "period"]).unique()})[:6],
            str(pd.Timestamp(cast(Any, pd.DatetimeIndex(trading_index).max())).date()) if len(trading_index) else "(empty)",
        )
        h = cast(pd.DataFrame, h.loc[~unavailable])
    return cast(pd.DataFrame, h)


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
    span = cast(pd.DataFrame, qf.groupby("period")["as_of"].agg(["min", "max"]).sort_index())
    # `bad` marks the LATER period of an offending pair -- the one whose first publication is
    # at or before its predecessor's last revision -- so the pair is `(index[i - 1], index[i])`.
    bad = np.flatnonzero((frame_column(span, "max").shift(1) >= frame_column(span, "min")).to_numpy())
    if len(bad):
        period_index: pd.DatetimeIndex = pd.DatetimeIndex(span.index)
        overlaps = [(str(period_index[int(i) - 1].date()), str(period_index[int(i)].date())) for i in bad]
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

    late = (frame_column(h, "as_of") > frame_column(h, "first_pub")).fillna(False)
    shares = cast(pd.Series, pd.to_numeric(frame_column(h, "shares"), errors="coerce"))
    total_sh = float(shares.sum())
    logger.info(
        "13F availability (settle %s trading day(s)): %s of %s filer-row(s) (%.2f%%) and "
        "%.2f%% of SHARES arrive after their period's first publication -> emitted as "
        "revisions on their own filing date, never back-dated",
        F13_SETTLE_TRADING_DAYS,
        f"{int(late.sum()):,}",
        f"{len(h):,}",
        100.0 * float(late.mean()),
        100.0 * float(cast(pd.Series, shares.loc[late]).sum()) / total_sh if total_sh > 0 else 0.0,
    )

    per_q = cast(
        pd.DataFrame,
        pd.DataFrame({"period": frame_column(h, "period"), "shares": shares, "behind": shares.where(late, 0.0)}).groupby("period").sum(),
    )
    shortfall = (frame_column(per_q, "behind") / frame_column(per_q, "shares").replace(0.0, np.nan)).dropna().sort_values(ascending=False)
    worst = shortfall.head(6)
    logger.info(
        "13F availability: worst 6 quarters by share of SHARES behind the first publication: %s (universe-wide median %.2f%%)",
        {str(pd.Timestamp(cast(Any, p)).date()): f"{100.0 * v:.2f}%" for p, v in worst.items()},
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
    shares = cast(pd.Series, pd.to_numeric(frame_column(h, "shares"), errors="coerce"))
    if float(shares.abs().sum()) == 0.0:
        logger.info(
            "13F availability coverage: the `shares` column is all zero -- it was almost "
            "certainly projected away and zero-filled by `clean_holdings`, so there is no "
            "basis to weight the diagnostic on. Not measured."
        )
        return pd.Series(dtype="float64")

    # One row per (cik, period): the filer's universe-wide share count, and the first date any
    # of its filings for that period was public.
    per_filer = h.groupby(["cik", "period"], sort=False).agg(shares=("shares", "sum"), pub=("as_of", "min")).reset_index()
    first_pub = cast(pd.Series, h.groupby("period")["first_pub"].first())
    periods = pd.Index(sorted(first_pub.dropna().index))
    pos = {p: i for i, p in enumerate(periods)}
    per_filer["pi"] = frame_column(per_filer, "period").map(pos)
    per_filer = per_filer.dropna(subset=["pi"])

    # `q-1`'s filers, aligned onto `q`, then matched against the filers public at `q`'s own
    # first publication. `pub <= first_pub(q)` IS "reported by the availability date".
    prior = cast(pd.DataFrame, per_filer[["cik", "pi", "shares"]].copy())
    prior["pi"] = frame_column(prior, "pi") + 1
    public_by_deadline = frame_column(per_filer, "pub") <= frame_column(per_filer, "period").map(first_pub)
    reported = cast(pd.DataFrame, per_filer.loc[public_by_deadline, ["cik", "pi"]]).assign(_seen=True)
    joined = prior.merge(reported, on=["cik", "pi"], how="left")
    # ⚠ `pi + 1` PUTS THE LAST QUARTER'S FILERS ONE PAST THE END, and that row has no quarter
    # to describe. Dropping it BEFORE the groupby rather than trimming the index afterwards is
    # what keeps the index assignment length-safe -- the earlier form filtered the list and not
    # the Series, which would raise on exactly the frame that triggered it.
    joined = cast(pd.DataFrame, joined.loc[frame_column(joined, "pi") < len(periods)])
    if joined.empty:
        return pd.Series(dtype="float64")
    seen = frame_column(joined, "_seen").fillna(False).to_numpy(dtype=bool)
    weight = cast(pd.Series, joined.groupby("pi")["shares"].sum()).replace(0.0, np.nan)
    held = cast(pd.Series, joined.loc[seen].groupby("pi")["shares"].sum())
    coverage = (held.reindex(weight.index).fillna(0.0) / weight).dropna()
    coverage.index = pd.DatetimeIndex([periods[int(i)] for i in coverage.index])
    return coverage.sort_index()


def _report_availability_coverage(h: pd.DataFrame) -> None:
    """Log `_availability_coverage` beside the filer-count coverage table."""
    coverage = _availability_coverage(h)
    if coverage.empty:
        return
    thin = cast(pd.Series, coverage.loc[coverage < 0.90])
    logger.info(
        "13F availability coverage (share of q-1's SHARES held by filers public by as_of(q)): "
        "median %.2f%%, p05 %.2f%%, min %.2f%% on %s; %s of %s quarter(s) below 90%%%s",
        100.0 * float(coverage.median()),
        100.0 * float(coverage.quantile(0.05)),
        100.0 * float(coverage.min()),
        str(pd.Timestamp(cast(Any, coverage.idxmin())).date()),
        len(thin),
        len(coverage),
        (" -> " + ", ".join(f"{pd.Timestamp(cast(Any, p)).date()}={100.0 * v:.1f}%" for p, v in thin.sort_values().head(8).items()))
        if len(thin)
        else "",
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

    coverage = _coverage_states(walk, break_pct)
    holes = coverage[coverage["hole"]]
    breaks = coverage[coverage["break"]]
    if not holes.empty or not breaks.empty:
        logger.info(
            "13F coverage guard: %s publication state(s) are holes (every feature nulled), %s are breaks (deltas nulled)",
            len(holes),
            len(breaks),
        )
    final_filers = coverage.groupby(level="period")["filers"].last()

    # The coverage table must see the complete source history, including the pre-floor regime
    # that establishes the predecessor count. Feature arithmetic starts at the declared floor.
    walk = cast(pd.DataFrame, walk.loc[frame_column(walk, "period") >= INST_LEVEL_FLOOR_PERIOD])
    factors = _split_factors(walk, splits)

    # `kind="stable"` keeps each stamp's rows in the order `clean_holdings` left them, which is
    # what makes the numeric aggregates reproducible across runs rather than merely close.
    walk = walk.sort_values(["ticker", "period", "as_of"], kind="stable")

    rows: list[dict] = []
    n_revisions = n_skipped = 0
    for ticker, tdf in walk.groupby("ticker", sort=False):
        prev: dict = {}
        prev_total = np.nan
        prev_period = None
        for p, pdf in tdf.groupby("period", sort=True):
            ciks = pdf["cik"].to_numpy()
            shares = pdf["shares"].to_numpy(dtype="float64")
            values = pdf["value_usd"].to_numpy(dtype="float64")
            calls = pdf["call_value"].to_numpy(dtype="float64")
            puts = pdf["put_value"].to_numpy(dtype="float64")
            stamps = pdf["as_of"].to_numpy()

            # Every universe-wide filer arrival can move this ticker's breadth denominator or
            # repair its coverage state even when that filer does not hold the ticker. Start at
            # the ticker's own first filing so a future holding cannot create a zero history.
            period_stamps = pd.DatetimeIndex(coverage.loc[p].index)
            emission_stamps = period_stamps[period_stamps >= pd.Timestamp(stamps[0])]
            ends = np.searchsorted(stamps, emission_stamps.to_numpy(), side="right")
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
            prev_pool = float(cast(Any, final_filers.get(prev_period, np.nan))) if prev_period is not None else np.nan
            prev_share = n_prev / prev_pool if (has_prev and prev_pool > 0) else np.nan

            # Running per-filer counters, advanced row by row. The SET arithmetic
            # (`cur & prev`, the increaser / decreaser tally) is incremental because each
            # filer joins the aggregate exactly once and never leaves it; only the numeric
            # sums are re-derived over the `[:k]` slice, which is what keeps them
            # bit-identical to a single-stamp aggregation of the same rows.
            n_both = inc = dec = 0
            at = 0
            last_emitted = last_share = np.nan
            last_hole = last_break = None
            for stamp, k in zip(emission_stamps, ends, strict=True):
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
                holders = int(k)
                coverage_state = coverage.loc[(pd.Timestamp(cast(Any, p)), stamp)]
                pool = float(cast(Any, coverage_state["filers"]))
                share = holders / pool if pool > 0 else np.nan
                is_hole = bool(coverage_state["hole"])
                is_break = bool(coverage_state["break"])
                if np.isfinite(last_emitted):
                    move = abs(inst_shares - last_emitted) / last_emitted if last_emitted > 0 else np.inf
                    state_changed = share != last_share or is_hole != last_hole or is_break != last_break
                    if move < revision_min_move and not state_changed:
                        n_skipped += 1
                        continue
                    n_revisions += 1

                inst_value = float(values[:k].sum())
                call_v = float(calls[:k].sum())
                put_v = float(puts[:k].sum())
                total_invested = inst_value + call_v + put_v  # long equity + option exposure
                opt_ratio = ((call_v - put_v) / total_invested) if total_invested > 0 else np.nan
                # crowding: Herfindahl of managers' VALUE shares (high = few dominant holders)
                mv = values[cik_order[cik_order < k]]
                tot_mv = float(mv.sum())
                hhi = float(((mv / tot_mv) ** 2).sum()) if tot_mv > 0 else np.nan
                rows.append(
                    {
                        "ticker": ticker,
                        "period": pd.Timestamp(cast(Any, p)),
                        "as_of": stamp,
                        # Carried only so the per-ticker coverage-onset guard can be applied
                        # vectorised below; dropped before the frame is returned.
                        "_prev_holders": float(n_prev) if has_prev else np.nan,
                        "_coverage_hole": is_hole,
                        "_coverage_break": is_break,
                        "ic_inst_holders": share,
                        "inst_shares": inst_shares,
                        "inst_value": inst_value,
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
                last_share = share
                last_hole = is_hole
                last_break = is_break

            # The next quarter differences against this one FULLY REVISED -- see the docstring.
            prev = dict(zip(cik_list, share_list, strict=False))
            prev_total = float(sum(prev.values()))
            prev_period = pd.Timestamp(cast(Any, p))

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
    # D17 / the hole guard travels with the publication state that was observable then. A later
    # revision can repair later rows but cannot unmask an earlier incomplete publication.
    is_hole = qf.pop("_coverage_hole").astype(bool)
    is_break = qf.pop("_coverage_break").astype(bool) | is_hole
    value_cols = ["inst_shares", "inst_value"]
    for c in DELTA_FEATURES:
        if c in qf.columns:
            qf.loc[is_break, c] = np.nan
    for c in LEVEL_FEATURES + tuple(value_cols):
        if c in qf.columns:
            qf.loc[is_hole, c] = np.nan

    # D16: a hard cutoff on the PERIOD, so nothing computed off the pre-break regime survives.
    qf.loc[qf["period"] < INST_LEVEL_FLOOR_PERIOD, [c for c in LEVEL_FEATURES if c in qf.columns] + value_cols] = np.nan
    qf.loc[qf["period"] < INST_DELTA_FLOOR_PERIOD, [c for c in DELTA_FEATURES if c in qf.columns]] = np.nan

    # The per-ticker coverage-onset guard (see `MIN_PRIOR_HOLDERS`). Same column set as D17's
    # break guard, applied on the TICKER's own prior filer count instead of the universe's.
    # `_prev_holders` is NaN on a ticker's first observed quarter, where every delta is already
    # NaN; `< floor` is False there, so the first quarter is not double-counted in the log.
    delta_cols = [c for c in DELTA_FEATURES if c in qf.columns]
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


def _daily_owned_shares(qf: pd.DataFrame, trading_index: pd.DatetimeIndex, share_basis: ShareBasis | None) -> pd.DataFrame:
    """Carry reported quantities in actual units without changing their publication clock."""
    if share_basis is None or not share_basis.policies:
        return nullable_value_to_daily(qf, "inst_shares", trading_index)
    df_owned = qf.copy()
    configured = df_owned["ticker"].isin(share_basis.policies)
    periods = pd.to_datetime(df_owned["period"])
    factor = share_basis.factor(df_owned["ticker"], periods, "actual")
    df_owned.loc[configured, "inst_shares"] *= factor.loc[configured]
    # Aggregated books without an explicit security title cannot establish ADS units.
    units = share_basis.factor(df_owned["ticker"], periods, "units")
    df_owned.loc[configured & units.ne(1.0), "inst_shares"] = np.nan
    daily = nullable_value_to_daily(df_owned, "inst_shares", trading_index)
    for ticker in share_basis.policies.keys() & set(daily.columns):
        names = pd.Series(ticker, index=trading_index)
        dates = pd.Series(trading_index, index=trading_index)
        daily[ticker] /= share_basis.factor(names, dates, "actual")
        daily[ticker] = daily[ticker].where(share_basis.factor(names, dates, "units").eq(1.0))
    return daily


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
    valuation_close: pd.DataFrame | None = None,
    value_basis: Mapping[str, float] | None = None,
    filings: pd.DataFrame | None = None,
    share_basis: ShareBasis | None = None,
) -> pd.DataFrame:
    """Long-format 13F feature panel (`f_<name>` per `EMISSION`). Empty if no holdings.

    `shares_out_history` (fundamentals carrying `sharesOutstandingPit` / `sharesOutstanding`)
    enables `ic_inst_ownership_pct`; with `stock_close` it also enables the value/market-cap
    weight through a point-in-time daily market cap. `splits`
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
        logger.warning("No `prices_splits` -> 13F share changes are NOT split-restated; a 20-for-1 split will read as +1,900%% accumulation.")

    if filings is not None:
        holdings, register = repair_value_basis(holdings, valuation_close, **dict(value_basis or {}))
        log_register(register, float(holdings.value_usd.sum()))
        if "value_basis_excluded" in holdings:
            holdings.loc[holdings.value_basis_excluded, ["shares", "value_usd"]] = np.nan
        qf = _versioned_quarter_features(
            filings, holdings, trading_index, splits=splits, settle_days=settle_trading_days, min_prior_holders=min_prior_holders
        )
    else:
        # clean and report holdings
        holdings = clean_holdings(holdings, key=("ticker", "cik", "period"), filing_band=(45, 60))

        # ⚠ THE UNIT REPAIR RUNS BEFORE ANYTHING READS A VALUE, and that ordering is the whole
        # point: `_quarter_features` sums value into `ic_inst_concentration`,
        # `ic_inst_net_options_ratio` and `ic_inst_value_to_mcap`.
        # Measured 2026-09-14 on the repaired table, 1.074% of rows (2,458 filings in the
        # divide-by-1000 band) carry 84.08% of the table's filed dollars, against 15.22% for the
        # 95.57% of rows that are already correct. Every value-weighted statement made before this
        # call is a statement about those 2,458 filings.
        value_before = cast(pd.Series, pd.to_numeric(frame_column(holdings, "value_usd"), errors="coerce")).sum()
        holdings, register = repair_value_basis(holdings, valuation_close, **dict(value_basis or {}))
        log_register(register, float(value_before))
        if "value_basis_excluded" in holdings:
            holdings = holdings.loc[~holdings["value_basis_excluded"]]
        if holdings.empty:
            return pd.DataFrame(columns=["date", "ticker"])

        holdings = _stamp_availability(holdings, trading_index, settle_trading_days=settle_trading_days)
        _report_first_publication_shortfall(holdings)
        _report_availability_coverage(holdings)

        # build features
        qf = _quarter_features(holdings, splits=splits, break_pct=break_pct, min_prior_holders=min_prior_holders, revision_min_move=revision_min_move)
    if qf.empty:
        return pd.DataFrame(columns=["date", "ticker"])
    if filings is None:
        _assert_emission_windows_ordered(qf)

    feats = [c for c in EMISSION if c in qf.columns]
    value_fields = {"ic_inst_concentration", "ic_inst_net_options_ratio"}
    fields = {
        f: (nullable_value_to_daily if filings is not None or f in value_fields else fundamentals_to_daily)(qf, f, trading_index) for f in feats
    }

    if shares_out_history is not None and not shares_out_history.empty:
        # ownership % by SHARES (aggregate 13F shares / shares outstanding)
        inst_sh = _daily_owned_shares(qf, trading_index, share_basis)

        # ⚠ `sharesOutstandingPit`, NOT `sharesOutstanding`.
        shares = fundamentals_to_daily(shares_out_history, "sharesOutstandingPit", trading_index, share_basis=share_basis)
        if not shares.empty and shares.notna().any().any():
            fields["ic_inst_ownership_pct"] = _capped_ownership((inst_sh / shares.where(shares > 0)).replace([np.inf, -np.inf], np.nan))

    if shares_out_history is not None and close_split is not None and not close_split.empty:
        # institutional WEIGHT by VALUE via a point-in-time daily market cap
        # (ffilled sharesOutstanding x daily close x S(d)).
        mcap = daily_market_cap(shares_out_history, close_split, level_factor=level_factor)
        if mcap.empty:
            # ⚠ NOT SILENT. An empty return here means `shares_out_history` was projected
            # without `sharesOutstanding` (the VENDOR basis `daily_market_cap` requires, NOT
            # `sharesOutstandingPit`), and the feature below would simply be absent from
            # the cube with nothing in the log saying so.
            logger.warning(
                "daily_market_cap returned no columns (shares_out_history has %s; "
                "it needs `sharesOutstanding`, the VENDOR basis) -> "
                "ic_inst_value_to_mcap is skipped.",
                sorted(shares_out_history.columns),
            )
        else:
            mpos = mcap.where(mcap > 0)
            inst_val = nullable_value_to_daily(qf, "inst_value", trading_index)
            iv = (inst_val / mpos).replace([np.inf, -np.inf], np.nan)
            if iv.notna().any().any():
                fields["ic_inst_value_to_mcap"] = iv

    # D16 again, now on the DAILY grid. The period-space mask above cannot reach the two
    # market-cap-scaled field (its numerator is a quarterly value but its denominator is
    # a daily close, so `fundamentals_to_daily` is not the only thing that fills them), and a
    # date-space floor is what L1/L9 actually scores.
    #
    # ⚠ THE AVAILABILITY DATE, NOT `period + 45d`. The bare deadline leaves a 2-5 day window in
    # which the market-cap-scaled field could carry a value that no emission stands
    # behind -- and L1/L9 now takes the same availability floor, so the checks would disagree by
    # exactly that window and the check would fail on a correct panel.
    floors = availability_date(
        pd.DatetimeIndex([INST_LEVEL_FLOOR_PERIOD, INST_DELTA_FLOOR_PERIOD]), trading_index, settle_trading_days=settle_trading_days
    )
    level_floor = pd.Timestamp(cast(Any, floors.get(INST_LEVEL_FLOOR_PERIOD, INST_LEVEL_FLOOR_PERIOD + pd.Timedelta(days=SEC_13F_FILING_LAG_DAYS))))
    delta_floor = pd.Timestamp(cast(Any, floors.get(INST_DELTA_FLOOR_PERIOD, INST_DELTA_FLOOR_PERIOD + pd.Timedelta(days=SEC_13F_FILING_LAG_DAYS))))
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
    emission = {name: EMISSION[name] for name in fields}
    return build_peer_relative_panel(fields, peer_dict, emission=emission, availability=frames.availability)


def vintage_changes(
    manifests: pd.DataFrame,
    holdings: pd.DataFrame,
    as_of: object,
    trading_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Scoped manager/CUSIP preview: current levels and causally resolved matched-quarter flows."""
    books, states = public_books(manifests, holdings, as_of, trading_index)
    output: list[pd.DataFrame] = []
    for cik, group in states.groupby("cik", sort=True):
        usable = group.loc[group["book_known"]].sort_values("period")
        if usable.empty:
            continue
        current = usable.iloc[-1]
        prior_period = (pd.Timestamp(current.period).to_period("Q") - 1).end_time.normalize()
        prior_state = group.loc[group["period"].eq(prior_period) & group["book_known"]]
        cur = books.loc[books.cik.eq(cik) & books.period.eq(current.period)].set_index("cusip")
        prev = books.loc[books.cik.eq(cik) & books.period.eq(prior_period)].set_index("cusip")
        names = cur.index.union(prev.index)
        result = pd.DataFrame(index=names)
        result["shares"] = cur["shares"].reindex(names)
        result["prior_shares"] = prev["shares"].reindex(names)
        paired = not prior_state.empty and bool(current.comparison_eligible) and bool(prior_state.iloc[0].comparison_eligible)
        if current.zero_eligible:
            result.loc[~result.index.isin(cur.index), "shares"] = 0.0
        if not prior_state.empty and prior_state.iloc[0].zero_eligible:
            result.loc[~result.index.isin(prev.index), "prior_shares"] = 0.0
        result["comparison_known"] = paired & result["shares"].notna() & result["prior_shares"].notna()
        result["shares_change"] = (result["shares"] - result["prior_shares"]).where(result["comparison_known"])
        result["reported_exit"] = result["comparison_known"] & result["shares"].eq(0) & result["prior_shares"].gt(0)
        result["value_usd"] = cur["value_usd"].reindex(names)
        result["prior_value_usd"] = prev["value_usd"].reindex(names)
        if current.zero_eligible:
            result.loc[~result.index.isin(cur.index), "value_usd"] = 0.0
        if not prior_state.empty and prior_state.iloc[0].zero_eligible:
            result.loc[~result.index.isin(prev.index), "prior_value_usd"] = 0.0
        result["value_change"] = (result["value_usd"] - result["prior_value_usd"]).where(result["comparison_known"])
        result["cik"] = cik
        result["period"] = current.period
        result["prior_period"] = prior_period
        result["as_of"] = pd.Timestamp(as_of)
        result["current_public_at"] = current.public_at
        result["resolution_status"] = current.resolution_status
        result["publication_age_days"] = (pd.Timestamp(as_of) - pd.Timestamp(current.public_at)).days
        result["prior_public_at"] = prior_state.iloc[0].public_at if not prior_state.empty else pd.NaT
        result["prior_resolution_status"] = prior_state.iloc[0].resolution_status if not prior_state.empty else "missing_original"
        output.append(result.reset_index())
    columns = [
        "cusip",
        "shares",
        "prior_shares",
        "shares_change",
        "reported_exit",
        "comparison_known",
        "value_usd",
        "prior_value_usd",
        "value_change",
        "cik",
        "period",
        "prior_period",
        "as_of",
        "current_public_at",
        "publication_age_days",
        "prior_public_at",
        "resolution_status",
        "prior_resolution_status",
    ]
    return pd.concat(output, ignore_index=True)[columns] if output else pd.DataFrame(columns=columns)


def vintage_publications(
    manifests: pd.DataFrame,
    holdings: pd.DataFrame,
    trading_index: pd.DatetimeIndex,
    *,
    until: object,
) -> pd.DataFrame:
    """Emit per-manager snapshots for relevant public vintages, including value-only corrections."""
    public_dates = filing_public_dates(manifests, trading_index)
    frames: list[pd.DataFrame] = []
    for day in public_dates.dropna().drop_duplicates().sort_values():
        if day > pd.Timestamp(until):
            continue
        preview = vintage_changes(manifests, holdings, day, trading_index)
        active = preview[["cik", "period", "prior_period"]].drop_duplicates().rename(columns={"period": "current_period"})
        events = manifests.loc[public_dates.eq(day), ["cik", "period"]].copy()
        events["period"] = pd.to_datetime(events["period"])
        relevant = events.merge(active, on="cik", how="inner")
        matches = relevant["period"].eq(relevant["current_period"]) | relevant["period"].eq(relevant["prior_period"])
        ciks = relevant.loc[matches, "cik"]
        if not ciks.empty:
            frames.append(preview.loc[preview["cik"].isin(ciks)])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def vintage_cohort_snapshot(
    manifests: pd.DataFrame,
    holdings: pd.DataFrame,
    as_of: object,
    trading_index: pd.DatetimeIndex,
    *,
    security_tickers: Mapping[str, str],
    period: object | None = None,
    splits: pd.DataFrame | None = None,
    settle_days: int = F13_SETTLE_TRADING_DAYS,
) -> pd.DataFrame:
    """Scoped daily preview over complete public books from one report quarter.

    Callers must supply CUSIPs independently verified to identify the same common
    security across both quarters, and already-classified genuine splits. Changed
    CUSIPs and manager reporting-perimeter bridges need separate source evidence;
    this preview never guesses aliases or activates the latest-only source.
    """
    if len(set(security_tickers.values())) != len(security_tickers):
        raise ValueError("Preview requires one verified stable CUSIP per ticker; changed identities need a verified bridge")
    books, states = public_books(manifests, holdings, as_of, trading_index, settle_days)
    known = states.loc[states.book_known]
    if known.empty or not security_tickers:
        return pd.DataFrame()
    current_period = pd.Timestamp(period) if period is not None else known.period.max()
    prior_period = (current_period.to_period("Q") - 1).end_time.normalize()
    current_states = known.loc[known.period.eq(current_period)].set_index("cik")
    all_prior_states = known.loc[known.period.eq(prior_period)].set_index("cik")
    if current_states.empty:
        return pd.DataFrame()
    ciks = current_states.index
    prior_states = all_prior_states.reindex(ciks)
    paired = current_states.comparison_eligible & prior_states.comparison_eligible.eq(True)

    def quantities(cusip: str, report: pd.Timestamp, column: str, zero: pd.Series, managers: pd.Index) -> pd.Series:
        selected = books.loc[books.period.eq(report) & books.cusip.eq(cusip)].set_index("cik")
        amounts = selected[column].reindex(managers).astype("float64")
        # Only absence from a complete, unrestricted public book establishes zero.
        amounts.loc[~managers.isin(selected.index) & zero.eq(True)] = 0.0
        return amounts

    def total(amounts: pd.Series) -> float:
        # Unknown contributions cannot become zero or an apparently complete total.
        return float(amounts.sum()) if len(amounts) and amounts.notna().all() else np.nan

    tickers = pd.Series(list(security_tickers.values()))
    factors = split_cum_product(splits, tickers, pd.Series(current_period, index=tickers.index)) / split_cum_product(
        splits, tickers, pd.Series(prior_period, index=tickers.index)
    )
    rows: list[dict[str, Any]] = []
    for (cusip, ticker), factor in zip(security_tickers.items(), factors, strict=True):
        current = quantities(cusip, current_period, "shares", current_states.zero_eligible, ciks)
        prior = quantities(cusip, prior_period, "shares", prior_states.zero_eligible, ciks) * factor
        value = quantities(cusip, current_period, "value_usd", current_states.zero_eligible, ciks)
        prior_value = quantities(cusip, prior_period, "value_usd", prior_states.zero_eligible, ciks)
        all_prior = quantities(cusip, prior_period, "shares", all_prior_states.zero_eligible, all_prior_states.index) * factor
        all_prior_total = total(all_prior)
        comparable = paired & current.notna() & prior.notna()
        cur, prev = current.loc[comparable], prior.loc[comparable]
        cur_total, prev_total = total(cur), total(prev)
        cur_value, prev_value = total(value.loc[comparable]), total(prior_value.loc[comparable])
        holders, old_holders = int(cur.gt(0).sum()), int(prev.gt(0).sum())
        both = cur.gt(0) & prev.gt(0)
        rows.append(
            {
                "as_of": pd.Timestamp(as_of),
                "period": current_period,
                "prior_period": prior_period,
                "cusip": cusip,
                "ticker": ticker,
                "public_books": len(ciks),
                "paired_books": int(paired.sum()),
                "matched_books": int(comparable.sum()),
                "share_support": int(current.notna().sum()),
                "value_support": int(value.notna().sum()),
                "shares": total(current),
                "value_usd": total(value),
                "public_prior_books": len(all_prior_states),
                "prior_share_support": int(all_prior.notna().sum()),
                "public_prior_shares": all_prior_total,
                "prior_share_coverage": float(prev.sum()) / all_prior_total if all_prior_total > 0 else np.nan,
                "publication_reference_at": current_states.public_at.median(),
                "publication_age_days": float((pd.Timestamp(as_of) - current_states.public_at).dt.days.median()),
                "matched_shares": cur_total,
                "matched_prior_shares": prev_total,
                "matched_value_usd": cur_value,
                "matched_prior_value_usd": prev_value,
                "shares_chg": cur_total / prev_total - 1 if prev_total > 0 else np.nan,
                "value_chg": cur_value / prev_value - 1 if prev_value > 0 else np.nan,
                "matched_holders": holders,
                "matched_prior_holders": old_holders,
                "new_ratio": float((cur.gt(0) & prev.eq(0)).sum()) / holders if holders else np.nan,
                "exit_ratio": float((cur.eq(0) & prev.gt(0)).sum()) / old_holders if old_holders else np.nan,
                "cluster_buying": float((both & cur.gt(prev)).sum() - (both & cur.lt(prev)).sum()) / holders if holders else np.nan,
                "breadth_chg": (holders - old_holders) / len(cur) if len(cur) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def vintage_cohort_publications(
    manifests: pd.DataFrame,
    holdings: pd.DataFrame,
    trading_index: pd.DatetimeIndex,
    *,
    until: object,
    security_tickers: Mapping[str, str],
    splits: pd.DataFrame | None = None,
    settle_days: int = F13_SETTLE_TRADING_DAYS,
) -> pd.DataFrame:
    """Publish changed cohort snapshots, including value/support and active-prior changes.

    Between events the consumer carries the last state, including its unknowns.
    No extra waiting buffer or hindsight-complete denominator is introduced.
    """
    frames: list[pd.DataFrame] = []
    previous: pd.DataFrame | None = None
    days = filing_public_dates(manifests, trading_index, settle_days).dropna().drop_duplicates().sort_values()
    for day in days.loc[days.le(pd.Timestamp(until))]:
        snapshot = vintage_cohort_snapshot(
            manifests, holdings, day, trading_index, security_tickers=security_tickers, splits=splits, settle_days=settle_days
        )
        if snapshot.empty:
            continue
        # Passage of time alone is not a filing revision; age advances between events.
        state = snapshot.drop(columns=["as_of", "publication_age_days"])
        if previous is None or not state.equals(previous):
            frames.append(snapshot)
            previous = state
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _versioned_quarter_features(
    manifests: pd.DataFrame,
    holdings: pd.DataFrame,
    trading_index: pd.DatetimeIndex,
    *,
    splits: pd.DataFrame | None,
    settle_days: int,
    min_prior_holders: int,
) -> pd.DataFrame:
    """Aggregate observed levels and identical eligible book cohorts at each public event."""
    rows: list[pd.DataFrame] = []
    seen: set[str] = set()
    numeric = ["shares", "value_usd", "call_value", "put_value"]
    for day, books, states in iter_public_books(manifests, holdings, trading_index, settle_days=settle_days):
        known = states.loc[states.book_known]
        if known.empty:
            continue
        period = known.period.max()
        prior_period = (period.to_period("Q") - 1).end_time.normalize()
        current_states = known.loc[known.period.eq(period)].set_index("cik")
        prior_states = known.loc[known.period.eq(prior_period)].set_index("cik")
        mapped = books.loc[books.ticker.notna()]
        grouped = mapped.groupby(["period", "cik", "ticker"])[numeric]
        totals = grouped.sum(min_count=1).where(grouped.count().eq(mapped.groupby(["period", "cik", "ticker"]).size(), axis=0))
        securities = mapped.loc[mapped.shares.gt(0)].groupby(["period", "cik", "ticker"]).cusip.agg(frozenset)
        current = totals.xs(period).copy() if period in totals.index.get_level_values("period") else totals.droplevel("period").iloc[:0]
        prior = totals.xs(prior_period).copy() if prior_period in totals.index.get_level_values("period") else current.iloc[:0]
        observed = current[["shares", "call_value", "put_value"]].gt(0).any(axis=1)
        seen.update(current.loc[observed].index.get_level_values("ticker"))
        names = sorted(seen)
        if not names:
            continue
        out = pd.DataFrame(index=pd.Index(names, name="ticker"))
        level = current.groupby("ticker")
        out["inst_shares"] = level.shares.agg(lambda amounts: amounts.sum(skipna=False))
        out["inst_value"] = level.value_usd.agg(lambda amounts: amounts.sum(skipna=False))
        holders = current.shares.gt(0).groupby("ticker").sum()
        out["ic_inst_holders"] = holders / len(current_states)
        unknown_quantity = current.shares.isna().groupby("ticker").any()
        out.loc[unknown_quantity.reindex(out.index).eq(True), "ic_inst_holders"] = np.nan
        if current_states.zero_eligible.all():
            empty_names = ~out.index.isin(current.index.get_level_values("ticker"))
            out.loc[empty_names, ["inst_shares", "inst_value", "ic_inst_holders"]] = 0.0
        value_total = current.value_usd.groupby("ticker").transform(lambda amounts: amounts.sum(skipna=False)).reindex(current.index)
        hhi = (current.value_usd / value_total.where(value_total.gt(0))).pow(2).groupby("ticker").sum(min_count=1)
        out["ic_inst_concentration"] = hhi.where(out.inst_value.notna())
        calls = level.call_value.agg(lambda amounts: amounts.sum(skipna=False))
        puts = level.put_value.agg(lambda amounts: amounts.sum(skipna=False))
        investment = out.inst_value + calls + puts
        out["ic_inst_net_options_ratio"] = (calls - puts) / investment.where(investment.gt(0))
        paired = current_states.index[current_states.comparison_eligible & prior_states.comparison_eligible.reindex(current_states.index).eq(True)]
        joined = current.add_suffix("_current").join(prior.add_suffix("_prior"), how="outer")
        joined = joined.loc[joined.index.get_level_values("cik").isin(paired)]
        for suffix, values, state in (("current", current, current_states), ("prior", prior, prior_states)):
            absent = ~joined.index.isin(values.index)
            zeros = state.zero_eligible.reindex(joined.index.get_level_values("cik")).eq(True).to_numpy()
            joined.loc[absent & zeros, [col + "_" + suffix for col in numeric]] = 0.0
        factors = split_cum_product(
            splits, pd.Series(joined.index.get_level_values("ticker")), pd.Series(period, index=range(len(joined)))
        ) / split_cum_product(splits, pd.Series(joined.index.get_level_values("ticker")), pd.Series(prior_period, index=range(len(joined))))
        joined["shares_prior"] *= factors
        if period in securities.index.get_level_values("period") and prior_period in securities.index.get_level_values("period"):
            now_ids = securities.xs(period).reindex(joined.index)
            old_ids = securities.xs(prior_period).reindex(joined.index)
            changed = now_ids.notna() & old_ids.notna() & now_ids.ne(old_ids)
            joined.loc[changed, ["shares_current", "shares_prior"]] = np.nan
        supported = joined.shares_current.notna() & joined.shares_prior.notna()
        matched = joined.loc[supported]
        groups = matched.groupby("ticker")
        current_shares = groups.shares_current.sum(min_count=1)
        previous_shares = groups.shares_prior.sum(min_count=1)
        now_held, was_held = matched.shares_current.gt(0), matched.shares_prior.gt(0)
        holder_count = now_held.groupby("ticker").sum()
        prior_count = was_held.groupby("ticker").sum()
        out["ic_inst_shares_chg"] = current_shares / previous_shares.where(previous_shares.gt(0)) - 1
        zero_books = current_states.zero_eligible.reindex(paired) & prior_states.zero_eligible.reindex(paired)
        safe_absence = zero_books.reindex(joined.index.get_level_values("cik")).to_numpy(dtype=bool)
        support_count = (
            int(zero_books.sum())
            + pd.Series(
                (supported.to_numpy() & ~safe_absence).astype(int) - (~supported.to_numpy() & safe_absence).astype(int),
                index=joined.index,
            )
            .groupby("ticker")
            .sum()
        )
        out["ic_inst_breadth_chg"] = (holder_count - prior_count) / support_count.where(support_count.gt(0))
        out["ic_inst_new_buyer_ratio"] = (now_held & matched.shares_prior.eq(0)).groupby("ticker").sum() / holder_count.where(holder_count.gt(0))
        out["ic_inst_exit_ratio"] = (matched.shares_current.eq(0) & was_held).groupby("ticker").sum() / prior_count.where(prior_count.gt(0))
        increases = (now_held & was_held & matched.shares_current.gt(matched.shares_prior)).groupby("ticker").sum()
        decreases = (now_held & was_held & matched.shares_current.lt(matched.shares_prior)).groupby("ticker").sum()
        out["ic_inst_cluster_buying"] = (increases - decreases) / holder_count.where(holder_count.gt(0))
        out.loc[prior_count.reindex(out.index).fillna(0).lt(min_prior_holders), list(DELTA_FEATURES)] = np.nan
        out["period"], out["as_of"] = period, day
        rows.append(out.reset_index())
    result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    logger.info("13F immutable publication: %s ticker states; unfiled/restricted books never imply exits.", len(result))
    return result
