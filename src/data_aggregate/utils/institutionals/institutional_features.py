"""
institutional_features.py  (src/data_aggregate/utils/institutionals/institutional_features.py)
--------------------------------------------------------------------------------
Broad 13F institutional ownership (`ic_inst_*`): per stock, aggregate ALL filers' reported
positions each quarter and measure breadth, accumulation, concentration, option positioning
and the dollar weight of institutions in the name. Registry section 1, features #1-#11.

Input `holdings` is manager-grain long (one row per manager x security x quarter):
    [cik, period, ticker, shares, value_usd, call_value, put_value]  (filing_date optional)
`period` is the quarter-end; `value_usd` is long-equity value (era-adjusted upstream),
`call_value` / `put_value` the option exposure the manager reported on the name.

POINT-IN-TIME. A 13F reports positions as of the quarter END and is public only once it is
filed, so each period is aggregated over ONLY the filings public at the date it is stamped --
see `availability.availability_date`, which is the one declaration of that date.

⚠ EACH `(ticker, period)` IS FIRST PUBLISHED ONCE AND THEN REVISED, and the revision is the
half that is easy to mistake for a leak. The first publication is at
`availability_date(period)` over `filing_date <= as_of`; the period is then RE-EMITTED on
each later date one of its filings arrives, stamped on that date, aggregated cumulatively,
and gated on materiality (`F13_REVISION_MIN_MOVE`). A filing that lands on day 79 is real new
public information about the quarter and it enters the panel on day 79 -- not on day 45 (a
leak) and not never (a blind spot). What the family does NOT do is re-aggregate on every date
INSIDE the pre-deadline window: measured below, the aggregate is under 15% complete for most
of those days, so those emissions would carry a sawtooth and no information.

MEASURED, 2026-09-15, on the in-universe banded frame at `F13_SETTLE_TRADING_DAYS = 3`:
**2.32% of filer-rows and 3.34% of SHARES arrive after their period's first publication**,
across 23,359 of 23,406 ticker-quarters, and the materiality gate turns 312,138 candidate
availability dates into ~99k emissions (23,406 first publications + ~75.5k revisions) while
still stamping 99.866% of shares on the date they became public. `_report_first_publication_shortfall`
re-measures the first two figures per quarter on every build.

⚠ THE FILING SEASON IS VIOLENTLY BACK-LOADED, AND FILER COUNT IS NOT A PROXY FOR IT. Mean
cumulative share of a `(ticker, period)`'s final total by days since period end:

    days since period end   shares   value   filers
    0-34                     0.079   0.074    0.375
    35-39                    0.147   0.142    0.493
    40-44                    0.471   0.476    0.680
    45-49                    0.871   0.875    0.887
    50-54                    0.989   0.989    0.983

Small filers file early and the mega-filers file at the deadline, so at day 40 the filer
COUNT is 68% complete while the dollars are 48%. Any gate or diagnostic built on filer count
fires too early -- which is why the coverage diagnostic below is weighted by SHARES
(universe-wide, the filer-count basis never falls below 96.7% while the shares basis reaches
82.6%).

⚠ EXPECT MEASURED IC ON THIS FAMILY TO FALL, AND EXPECT THAT. The stamp used to be the bare
`period + 45d` deadline for every row of the period, so a filing made on day 90 sat inside
the number published on day 45. That is look-ahead being removed, not signal being lost, and
a backtest that IMPROVES on this change has a bug.

⚠ THE `[-45, +60]` BAND (`F13_MAX_EARLY_DAYS` / `F13_MAX_LATE_DAYS`, applied in
`clean_holdings`) NOW EARNS ITS PLACE FOR TWO DIFFERENT REASONS, and they are not
interchangeable. Its LATE half no longer protects the current period -- anything filed after
`as_of` is stamped on its own filing date rather than back-dated, so nothing needs excluding.
What it still does is (a) bound the REVISION window at `period + 105d`, past which a period
is closed and a back-file would disturb a quarter nothing is reading any more, and (b) keep a
2015 period back-filed in 2025 out of the PRIOR-QUARTER lookup that `q`'s deltas difference
against. `sec13f_manager_holdings` keeps its own rule and inherits neither.

SPLIT RESTATEMENT (registry #3/#6, and value-sanity V4). 13F share counts are AS FILED, so
`Sigma shares(q) / Sigma shares(q-1)` reads GOOGL's 2022 20-for-1 split as +1,900% of
accumulation -- and the per-manager `shares(q) > shares(q-1)` test behind `ic_inst_cluster_buying`
makes EVERY holder an increaser, printing maximum unanimity on the one quarter nobody decided
anything. The prior quarter's counts are restated onto the current quarter's basis through
`split_basis.future_split_factor` before either feature is computed, and the number of
restated ticker-quarters is logged (V5).

D28 -- `ic_inst_holders` IS A SHARE, NOT A COUNT: `holders(q) / n_filers(q)`, the filer count
being every distinct `cik` in the table that quarter. The raw count jumps 41 -> 556 per ticker
at 2013-06-30 on a fetch artifact; the share does not, and the same normalization is what held
the two interrupted quarters below in place before they were refilled. `ic_inst_breadth_chg`
therefore differences the SHARE, not the count.

D16 / D17 -- THE COVERAGE GUARDS. Measured distinct filers per quarter on `sec13f_hr` **after
the step's universe cut and the `[-45, +60]` filing band** (2026-09-14) -- that scope is the
point, because this is the same population the D28 denominator counts, and a filer whose only
holdings are off-universe names, or whose filing arrived two years late, belongs in neither:

    period       filers    what it is
    2013-03-31       63    the pre-break regime
    2013-06-30    2,938    +4,563%: the coverage REGIME START (a fetch artifact, not a market
                           event) -> `INST_LEVEL_FLOOR_PERIOD`, and its deltas have no
                           comparable predecessor -> `INST_DELTA_FLOOR_PERIOD` is the next one
    2013-09-30    2,966    and from here the axis is monotone-ish and healthy ...
    2023-09-30    5,953
    2023-12-31    6,354    ... including the two quarters that used to be 13x and 29x short
    2024-03-31    6,361
    2025-03-31    7,099
    2025-06-30    7,180    the second one
    2026-06-30    7,898

⚠ THE TWO HOLES ARE GONE, AND THAT IS A FACT ABOUT THE TABLE, NOT ABOUT THIS GUARD. 2023-12-31
held 463 filers and 2025-06-30 held 254 until the 2026-09-14 window backfill refilled both
(`thirteen-f --filing-window`, which bypasses the watermark in both directions -- a gap BEHIND
`max(filing_date) - lookback_days` is unreachable by the incremental path, which is why they
survived). The guard is kept at full strength anyway: it costs nothing when it finds nothing,
and `validate institutionals` V13a now scores the same axis on the raw table so the next
interrupted fetch is caught rather than discovered.

The guard has two halves, because a hole and a regime start need different treatment:

  * a HOLE (a quarter whose filer count is under half of BOTH neighbours) has every feature
    it can corrupt SUPPRESSED, level and delta alike -- extending D17, which as written covers
    only deltas. A hole is missing data, and its `ic_inst_ownership_pct` is a thirteenth of the
    truth, which is D16's own principle: a plausible wrong number is more dangerous than an
    absent one. ⚠ Suppressing a LEVEL means the quarter contributes no observation, so
    `fundamentals_to_daily` holds the PREVIOUS quarter's value across it. That is not a
    workaround -- it is exactly what the panel does on the other 89 days of every quarter, and
    a stale level is a statement the source actually supports where a 13x-low one is not.
  * a BREAK (|change| > `COVERAGE_BREAK_DEFAULT` against the previous quarter) nulls only the
    DELTAS, which is D17 exactly. This catches the regime start at 2013-06-30 and the two
    RECOVERY quarters, whose levels are correct but whose predecessor is a hole.

⚠ THE BREAK GUARD NOW FIRES ON EXACTLY ONE QUARTER -- 2013-06-30 (2,938 filers), the regime
start -- AND NULLS NOTHING D16 HAD NOT. It used to fire on eleven: the two holes, their two
recovery quarters, and a seven-quarter pre-2013 sparse tail running back to 2002-03-31 (2
filers). The refill removed the first four and the band removed the tail, which is the more
interesting half of the finding:

⚠ EVERY PRE-2013 ROW IN THIS TABLE WAS FILED LATE, AND THAT IS WHY THE PRE-FLOOR CUT IS SAFE.
`min(filing_date)` on the whole table is **2013-05-20** -- nothing was filed before EDGAR's
13F-HR full-text era, so the 65,634 rows carrying a pre-2013 `period` are back-filed archive
entries, every one of them more than 60 days past its own deadline. The band removes them all
on its own, and after it the earliest surviving period is 2013-03-31. So `INST_LEVEL_FLOOR_PERIOD`
is not just a threshold somebody chose from a filer-count jump: it is where the table's
point-in-time history actually begins, arrived at from a second, independent direction.

Both lists are logged with their measured filer counts, never applied silently.

⚠ D16 AND D17 ARE BOTH UNIVERSE-WIDE, WHICH LEAVES A THIRD GUARD TO DECLARE. They ask when the
MARKET's 13F coverage began and when it jumped; neither asks when THIS TICKER's did. A name that
joins the index on a spin-off, an IPO, a redomicile or an emergence from bankruptcy has one or
two filers in its first quarter and several hundred in its second, and `shares / prev_shares - 1`
reads that as a flow of millions of percent -- `f_ic_inst_shares_chg` peaked at **34,264,348**
(CCI, 2014-12-31). `MIN_PRIOR_HOLDERS` is the per-ticker analogue, applied to the same column set
as D17's break guard; its constant carries the measured band table it was sized on.
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
from src.data_aggregate.utils.institutionals.holdings_clean import clean_holdings as _clean
from src.data_aggregate.utils.institutionals.split_basis import future_split_factor
from src.data_aggregate.utils.institutionals.value_basis import log_register, repair_value_basis

logger = logging.getLogger(__name__)

#: D16. The first period whose all-filer coverage describes the market rather than the fetch
#: (measured above: 63 -> 2,938 in-universe filers, post-band). Every LEVEL is NaN before its
#: availability date, and since step 2.4 those periods are not computed at all.
INST_LEVEL_FLOOR_PERIOD = pd.Timestamp("2013-06-30")

#: D16. The first period whose PREDECESSOR is also post-break, so a QoQ delta is meaningful.
INST_DELTA_FLOOR_PERIOD = pd.Timestamp("2013-09-30")

#: D17. A quarter-on-quarter move in the universe-wide filer count larger than this is a
#: coverage discontinuity, not a market event. A research parameter, so the step passes the
#: configured value in; this is the declared default.
COVERAGE_BREAK_DEFAULT = 0.50

#: THE PER-TICKER ANALOGUE OF D16/D17: the fewest filers a ticker's PREVIOUS quarter may carry
#: for this quarter's QoQ delta to describe a flow rather than the discovery of the name.
#:
#: ⚠ D16 AND D17 ARE BOTH UNIVERSE-WIDE, and that is the hole this closes. They ask when the
#: MARKET's 13F coverage began and when it jumped; neither asks when THIS TICKER's did. A name
#: that joins the index on a spin-off, an IPO, a redomicile or an emergence from bankruptcy has
#: one or two filers in its first quarter and several hundred in its second, and
#: `shares / prev_shares - 1` reads that as a flow of several million percent.
#: `f_ic_inst_shares_chg` peaked at **34,264,348** (CCI, 2014-12-31: 9 shares held by 1 filer,
#: then 308,379,140 held by 452).
#:
#: 100 is where the measured cliff is. Distribution of the surviving delta by the prior
#: quarter's filer count, over the 22,917 (ticker, quarter) pairs that already clear D16 + D17
#: (re-measured 2026-09-14 on the in-universe `sec13f_hr`, after the refill, the `[-45, +60]`
#: band and the value-unit repair). The values are `shares / prev_shares - 1` as a RATIO, so
#: 2.52 is +252% and 5,326,051 is the CCI-shaped artifact, not a percentage:
#:
#:     prior filers        n       p50         p95         p99          max
#:     1                  33     2.523 2,709,575.3 4,861,247.1  5,326,051.5
#:     2                   6   545.049    78,362.1    96,689.7    101,271.6
#:     3                   1 1,211.432     1,211.4     1,211.4      1,211.4
#:     4-5                 3    84.639       112.7       115.2        115.9
#:     6-10                7   225.522     5,979.3     7,262.9      7,583.8
#:     11-20               7    53.571       880.3     1,136.5      1,200.6
#:     21-50              17    35.242       259.1       283.4        289.5
#:     51-100             26     0.240        41.7        49.8         51.3
#:     101-200           303     0.012         0.5         4.2         83.9
#:     >200           22,514     0.002         0.1         0.2         17.8
#:
#: A MEDIAN of +54,504% (the 2-filer band) is not a flow. The >200 band is the healthy one and
#: 98.2% of the population sits in it.
#:
#: ⚠ THE CLIFF IS IN THE SAME PLACE IT WAS, AND THE GUARD IS NOW TWELVE TIMES CHEAPER. Raising
#: the floor from 10 to 100 still takes the surviving maximum from 2,235.6 to 83.9 (it was
#: 1,998.9 -> 89.9 on the 2026-09-12 population), and a floor of 2 still leaves a 101,271x
#: survivor standing. What changed is the COST: at 100 the guard now nulls **100 of 22,917
#: deltas (0.44%)**, against 1,133 of 22,070 (5.13%) before, because the band filter deletes
#: the late-filed rows that were most of what made a name look newly discovered. The constant
#: is NOT re-tuned on this measurement (out of scope, README) -- it is restated because the
#: population under it changed, and 100 is still where the cliff is.
#:
#: NOT IN CONFIG, deliberately, and for the same reason `INST_LEVEL_FLOOR_PERIOD` and
#: `OWNERSHIP_CEILING` are not: it is a measured fact about when 13F coverage of a name begins,
#: with a flat optimum and a cliff, not a research dial anyone would sweep.
#:
#: ⚠ APPLIES TO THE WHOLE `DELTA_FEATURES` SET, not just `shares_chg`. Every one of them is a
#: statement about the previous quarter's filer set, and they are all equally meaningless
#: against a set of size one -- `ic_inst_new_buyer_ratio` reads CCI's 2014-12-31 as 451/452 =
#: 99.8% new buyers, which is true arithmetic and a false fact. The LEVELS are left alone: a
#: thinly-held name really is thinly held that quarter, and `ic_inst_holders` measures it.
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


def _stamp_availability(h: pd.DataFrame, trading_index: pd.DatetimeIndex, *, settle_trading_days: int = F13_SETTLE_TRADING_DAYS) -> pd.DataFrame:
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
    h = h.copy()
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
            "13F availability: %s row(s) across period(s) %s have no availability date "
            "inside the trading calendar (it ends %s) -> not emitted",
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


def clean_holdings(holdings: pd.DataFrame) -> pd.DataFrame:
    """`sec13f_hr` at its own grain: one row per (ticker, manager, quarter).

    A thin call into the SHARED cleaner -- `holdings_clean.clean_holdings` -- which both 13F
    tables now go through. The only thing that is specific here is the KEY: this table has a
    `ticker` column and the elite one does not, and `position_type` is already resolved at
    extraction so there is nothing to filter, and `cik` arrives padded.

    ⚠ AND THE FILING BAND, which this table opts INTO and the elite one does not. A holding
    filed a year after the quarter it describes is a real position, but the quarter it
    describes is long closed: the band's late edge is what CLOSES a period's revision window
    (`period + 105d`) and what keeps a 2015 quarter back-filed in 2025 out of the
    prior-quarter lookup the next quarter's deltas difference against. It is no longer
    protecting the CURRENT period from back-dating -- the availability stamp does that, by
    putting every filing on its own filing date. Measured 2026-09-14: 1,050,431 of 23,801,899
    rows (4.413%) across 2,679 filers, carrying 1.052% of as-filed value; the constants carry
    the full lateness table. The elite table abstains pending its own measurement.
    """
    return _clean(holdings, key=("ticker", "cik", "period"), filing_band=(F13_MAX_EARLY_DAYS, F13_MAX_LATE_DAYS))


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
    #
    # The reason for the cut is NOT speed -- measured 2026-09-14, pre-floor is 65,634 of
    # 23,801,899 rows (0.28%) and the saving is negligible. It is that D16 nulls every feature
    # computed from those quarters a hundred lines below, so the loop was doing work whose
    # output is discarded, and a reader had no way to tell that from the code.
    #
    # ⚠ IT DOES NOT MOVE THE FIRST SURVIVING QUARTER. `prev` is carried across periods, so
    # 2013-06-30 loses the 2013-03-31 predecessor its deltas were differenced against -- but
    # `INST_DELTA_FLOOR_PERIOD` (2013-09-30) already nulls exactly those deltas, and
    # 2013-09-30's own predecessor (2013-06-30) survives the cut intact. Asserted in
    # `test_the_pre_floor_cut_leaves_the_delta_floor_onward_identical`, not assumed.
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

    ⚠ `frames` RATHER THAN FOUR UNPACKED FIELDS. `peer_dict`, `trading_index`, `stock_close` and
    `level_factor` were all read off one `PriceFrames` at the call site. Naming the object makes
    the basis un-mistakable: there is one `close_split` and one `close_total` on it, and neither
    can arrive under the other's parameter name.

    ⚠ NO `frames.require(...)`, AND THAT IS MEASURED RATHER THAN FORGOTTEN. Every wide frame
    this builder reads sits behind an explicit `is None` guard, or is handed to a callee that
    documents `None` as a MEANING rather than an error -- `daily_market_cap`'s
    `level_factor=None` IS "S is 1.0 everywhere". `require` would turn each of those graceful
    degrades into a raise, which is exactly what its own docstring warns against.

    The non-frame arguments are KEYWORD-ONLY. A positional slip between two same-typed
    `pd.DataFrame | None` neighbours is a silent wrong-frame bug that reads as a plausible
    call; the keyword form makes it unrepresentable.
    """
    peer_dict = frames.peers
    trading_index = frames.trading_index
    stock_close = frames.close_split
    level_factor = frames.level_factor
    need = {"cik", "period", "ticker", "shares"}
    if holdings is None or holdings.empty or not need.issubset(holdings.columns):
        return pd.DataFrame(columns=["date", "ticker"])
    if splits is None or splits.empty:
        logger.warning("No `prices_splits` -> 13F share changes are NOT split-restated; a " "20-for-1 split will read as +1,900%% accumulation.")

    # clean and report holdings
    holdings = clean_holdings(holdings)
    # ⚠ THE UNIT REPAIR RUNS BEFORE ANYTHING READS A VALUE, and that ordering is the whole
    # point: `_quarter_features` sums value into `ic_inst_concentration`,
    # `ic_inst_net_options_ratio`, `ic_inst_value_to_mcap` and `ic_inst_flow_to_mcap`.
    # Measured 2026-09-14 on the repaired table, 1.074% of rows (2,458 filings in the
    # divide-by-1000 band) carry 84.08% of the table's filed dollars, against 15.22% for the
    # 95.57% of rows that are already correct. Every value-weighted statement made before this
    # call is a statement about those 2,458 filings.
    value_before = pd.to_numeric(holdings.get("value_usd"), errors="coerce").sum()
    holdings, register = repair_value_basis(holdings, stock_close)
    log_register(register, float(value_before), logger)
    holdings = _stamp_availability(holdings, trading_index, settle_trading_days=settle_trading_days)
    _report_first_publication_shortfall(holdings)
    _report_availability_coverage(holdings)

    # build features
    qf = _quarter_features(
        holdings, splits=splits, break_pct=break_pct, min_prior_holders=min_prior_holders, revision_min_move=revision_min_move
    )
    if qf.empty:
        return pd.DataFrame(columns=["date", "ticker"])
    _assert_emission_windows_ordered(qf)

    feats = [c for c in EMISSION if c in qf.columns]
    fields = {f: fundamentals_to_daily(qf, f, trading_index) for f in feats}

    have_shares = shares_out_history is not None and not shares_out_history.empty
    if have_shares:
        # ownership % by SHARES (aggregate 13F shares / shares outstanding)
        inst_sh = fundamentals_to_daily(qf, "inst_shares", trading_index)

        # ⚠ `sharesOutstandingPit`, NOT `sharesOutstanding`. A 13F reports the shares a
        # manager ACTUALLY HELD on the filing date, so the denominator must be the count that
        # actually existed then. The vendor-basis column is back-filled to today's split
        # basis, which would read GOOGL's post-2022 ownership 20x too low. This ratio and the
        # insider one are the only two consumers of the PIT column in the whole repo.
        shares = fundamentals_to_daily(shares_out_history, "sharesOutstandingPit", trading_index)
        if not shares.empty and shares.notna().any().any():
            fields["ic_inst_ownership_pct"] = _capped_ownership((inst_sh / shares.where(shares > 0)).replace([np.inf, -np.inf], np.nan))

    if have_shares and stock_close is not None and not stock_close.empty:
        # institutional WEIGHT by VALUE and size-scaled net $ flow, via a point-in-time
        # daily market cap (ffilled sharesOutstanding x daily close x S(d)).
        mcap = daily_market_cap(shares_out_history, stock_close, level_factor=level_factor)
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
