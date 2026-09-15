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

POINT-IN-TIME. A 13F reports positions as of the quarter END and is public only at the SEC
deadline ~45 days later, so every quarter's aggregate is stamped
`as_of = period + SEC_13F_FILING_LAG_DAYS` and forward-filled from there.

⚠ THE AGGREGATE IS STAMPED ON THE DEADLINE, NOT ON `max(deadline, filing_date)`, and that is a
deliberate departure from registry section 0.5. The rule is right for the ELITE family, where one
manager IS the feature and a late filer must not become visible on the statutory date it
missed; it cannot be applied to a 3,000-filer aggregate without re-aggregating the whole
universe once per date, because the honest per-filer version makes a count visible in as many
instalments as it has contributors.

MEASURED RESIDUAL (2026-09-12 full build, and `_report_late_filings` re-measures it every run):
**2,969,232 of 22,335,985 (ticker, manager, quarter) rows -- 13.3% -- were filed AFTER
period+45d, and they carry 19.3% of 13F value.** So roughly a fifth of the dollars behind these
eleven features become visible on the statutory date rather than on the date they were actually
filed. That is the size of the compromise, stated rather than implied, and it is the same shape
of finding Phase 2.2 measured for the elite family (16.5% of filings / 36.7% of value) and there
chose to FIX, because 87 managers can be put on an availability grid and 3,000 filers cannot be
cheaply. Removing it here means emitting the aggregate per availability date over the filings
public at that date -- a real change, not a parameter -- and it is recorded as open work rather
than pretended away.

SPLIT RESTATEMENT (registry #3/#6, and value-sanity V4). 13F share counts are AS FILED, so
`Sigma shares(q) / Sigma shares(q-1)` reads GOOGL's 2022 20-for-1 split as +1,900% of
accumulation -- and the per-manager `shares(q) > shares(q-1)` test behind `ic_inst_cluster_buying`
makes EVERY holder an increaser, printing maximum unanimity on the one quarter nobody decided
anything. The prior quarter's counts are restated onto the current quarter's basis through
`split_basis.future_split_factor` before either feature is computed, and the number of
restated ticker-quarters is logged (V5).

D28 -- `ic_inst_holders` IS A SHARE, NOT A COUNT: `holders(q) / n_filers(q)`, the filer count
being every distinct `cik` in the table that quarter. The raw count jumps 41 -> 556 per ticker
at 2013-06-30 on a fetch artifact; the share does not, and the same normalization is what keeps
the two broken quarters below from moving it. `ic_inst_breadth_chg` therefore differences the
SHARE, not the count.

D16 / D17 -- THE COVERAGE GUARDS, AND BOTH FIRE ON LIVE DATA. Measured distinct filers per
quarter on `sec13f_hr` **after the step's universe cut** (2026-09-12 build) -- that scope is the
point, because this is the same population the D28 denominator counts, and a filer whose only
holdings are off-universe names belongs in neither:

    period       filers    what it is
    2013-03-31      192    the pre-break regime
    2013-06-30    3,045    +1,486%: the coverage REGIME START (a fetch artifact, not a market
                           event) -> `INST_LEVEL_FLOOR_PERIOD`, and its deltas have no
                           comparable predecessor -> `INST_DELTA_FLOOR_PERIOD` is the next one
    2023-09-30    6,199
    2023-12-31      463    -92.5%: a HOLE. Not a regime change -- 2024-03-31 returns to 6,648
    2025-03-31    7,230
    2025-06-30      254    -96.5%: the second hole; 2025-09-30 returns to 7,286

so the guard has two halves, because a hole and a regime start need different treatment:

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

⚠ THE BREAK GUARD FIRES ON ELEVEN QUARTERS, NOT THREE, and the count matters because a reader
who expects three will treat the other eight as a bug. Measured on the same build:

    2002-03-31 (2 filers)   2009-12-31 (17)    2013-06-30 (3,045)   2025-06-30 (254)
    2006-12-31 (5)          2012-12-31 (107)   2023-12-31 (463)     2025-09-30 (7,286)
    2008-12-31 (9)          2013-03-31 (192)   2024-03-31 (6,648)

SEVEN of them sit at or before `INST_DELTA_FLOOR_PERIOD` and are already NaN from D16, so the
guard removes something D16 had not on exactly FOUR: the two holes and the two recoveries. The
pre-2013 firings are the sparse tail of the same fetch artifact -- a quarter with 2 filers in
it -- and cost nothing, because a guard that only fires where it is load-bearing is a guard
tuned to the data it was measured on.

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

from src.data_aggregate.utils.common.pit import daily_market_cap, fundamentals_to_daily
from src.data_aggregate.utils.institutionals.holdings_clean import clean_holdings as _clean
from src.data_aggregate.utils.common.panel import build_peer_relative_panel
from src.data_aggregate.utils.institutionals.split_basis import future_split_factor
from src.data_aggregate.utils.institutionals.value_basis import log_register, repair_value_basis
from src.constants.constants import SEC_13F_FILING_LAG_DAYS
from src.data_aggregate.utils.common.price_frames import PriceFrames

logger = logging.getLogger(__name__)

#: D16. The first period whose all-filer coverage describes the market rather than the fetch
#: (measured above: 192 -> 3,046 filers). Every LEVEL is NaN before its availability date.
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
#: quarter's filer count, over the 22,070 (ticker, quarter) pairs that already clear D16 + D17
#: (measured 2026-09-12 on the in-universe `sec13f_hr`):
#:
#:     prior filers        n     p50      p95         p99          max
#:     1                 620   0.000  6,471.1   411,069.3   34,264,347.9
#:     2                 212   0.000  9,170.8   145,148.6      357,122.9
#:     3                 115   0.006  9,774.8    70,960.7      313,086.3
#:     4-5                85   0.052 23,140.2    81,427.8      168,347.9
#:     6-10               45   0.189 23,495.3    67,808.5      101,091.1
#:     11-20              12  50.189  1,569.2     1,913.0        1,998.9
#:     21-50              19  24.280    183.6       207.3          213.2
#:     51-100             27   0.276     74.8       153.2          176.5
#:     101-200           257   0.012      0.6        34.1           89.9
#:     >200           20,678  -0.001      0.1         0.2           46.2
#:
#: A MEDIAN of +2,400% (the 11-20 band) is not a flow. The >200 band is the healthy one and
#: 93.7% of the population sits in it. The marginal cost of the floor is tiny and its marginal
#: benefit is not: raising it from 10 to 100 nulls **58 more rows (0.26%)** and takes the
#: surviving maximum from 1,998.9 to 89.9. At 100 the guard nulls 1,133 of 22,070 deltas
#: (5.13%), against 2.81% for a floor of 2 that leaves a 357,122x survivor standing.
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
    "ic_inst_breadth_chg", "ic_inst_shares_chg", "ic_inst_new_buyer_ratio",
    "ic_inst_exit_ratio", "ic_inst_cluster_buying", "ic_inst_flow_to_mcap",
)

#: `ic_inst_ownership_pct` above this many times shares outstanding is a denominator defect,
#: not a holder, and the cell is NULLED (never clipped -- see `_capped_ownership`). 2.0 rather
#: than 1.0 because securities lending double-counts a position and 100-130% is a real, widely
#: published reading; 9.68x is not.
OWNERSHIP_CEILING = 2.0

#: The level features. Nulled before `INST_LEVEL_FLOOR_PERIOD` (D16).
LEVEL_FEATURES: tuple[str, ...] = (
    "ic_inst_holders", "ic_inst_concentration", "ic_inst_net_options_ratio",
    "ic_inst_ownership_pct", "ic_inst_value_to_mcap",
)

#: Emission (D27, registry section 0.10), with the two corrections section 0.10a MEASURED on this
#: family: `_xs` earns its column where the raw leg re-ranks pooled, and the discriminator is
#: CARDINALITY, not boundedness. `ic_inst_concentration` (rho 0.556) and `ic_inst_value_to_mcap`
#: (0.607) re-rank heavily and take the percentile leg even though section 0.10 reasoned them as
#: `raw`; `ic_inst_ownership_pct` is the most date-stable of the family (0.930) and its
#: `raw+peers` assignment is confirmed -- institutional ownership is the textbook sector-normed
#: rate, where an absolute 40% means nothing without the sector's norm.
EMISSION: dict[str, str] = {
    "ic_inst_holders":           "raw",       # D28 share of the quarter's filers, in [0, 1]
    "ic_inst_breadth_chg":       "raw",       # a difference of that share
    "ic_inst_shares_chg":        "raw+xs",    # unbounded growth rate, distribution drifts
    "ic_inst_new_buyer_ratio":   "raw",       # bounded [0, 1]
    "ic_inst_exit_ratio":        "raw",       # bounded [0, 1]
    "ic_inst_cluster_buying":    "raw",       # bounded [-1, 1]
    "ic_inst_concentration":     "raw+xs",    # Herfindahl; measured rho 0.556
    "ic_inst_net_options_ratio": "raw",       # bounded [-1, 1]
    "ic_inst_ownership_pct":     "raw+peers",  # sector-normed rate; measured rho 0.930
    "ic_inst_value_to_mcap":     "raw+xs",    # measured rho 0.607
    "ic_inst_flow_to_mcap":      "raw+xs",    # mcap-scaled dollar flow, skewed and drifting
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
        logger.info("13F split restatement: %s of %s (ticker, quarter) pairs had the prior "
                    "quarter's share count rebased (V5 trigger count)", n, len(pairs))
    return {(t, p): f for t, p, f in zip(pairs["ticker"], pairs["period"], factor)}


def _capped_ownership(ratio: pd.DataFrame) -> pd.DataFrame:
    """NULL `ic_inst_ownership_pct` above `OWNERSHIP_CEILING`, and report the count.

    ⚠ ABOVE 100% IS NOT BY ITSELF WRONG, which is why the ceiling is 2.0 and not 1.0.
    Securities lending double-counts a position -- the lender still reports it on its 13F and
    the borrower's buyer reports it too -- so a heavily-lent name legitimately reads 100-130%,
    and data vendors publish such figures. Measured on the 2026-09-12 build: 24,347 of
    1,516,801 cells (1.6%) above 1.05 and 3,623 above 1.30, which is that phenomenon.

    What is NOT credible is the extreme tail: 231 cells above 2.0, topping out at **9.68**
    (DUK), and **all twelve** of the tickers involved -- DUK, NVDA, SMCI, TSCO, AMCR, ANET,
    PANW, DD, TPL, APH, MNST, ETR -- have splits in `prices_splits`. That is the recorded
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
        logger.info("ownership guard: %s of %s non-null `ic_inst_ownership_pct` cells (%.3f%%) "
                    "above %.1fx shares outstanding -> nulled (the `sharesOutstandingPit` split "
                    "defect, not a holder); max seen %.2f",
                    f"{n:,}", f"{int(ratio.notna().to_numpy().sum()):,}",
                    100.0 * n / max(1, int(ratio.notna().to_numpy().sum())),
                    OWNERSHIP_CEILING, float(np.nanmax(ratio.to_numpy(dtype='float64'))))
    return ratio.mask(over)


def _report_late_filings(h: pd.DataFrame) -> None:
    """Measure the residual the deadline stamp accepts -- the module docstring's one caveat.

    Stamping the aggregate on `period + 45d` rather than `max(deadline, filing_date)` makes a
    LATE filer's holdings visible on the statutory date it missed. The size of that is a
    property of the table, not an assumption, so it is measured here every build: the share of
    (ticker, manager, quarter) rows whose `filing_date` falls after the deadline, and the share
    of 13F VALUE they carry -- which is the number that matters, since one late mega-filer
    outweighs a hundred late small ones.

    ⚠ `filing_date` IS OPTIONAL ON THIS TABLE (`sec13f_hr.optional_columns`), so the measure
    has to be skippable. Reading it unguarded raised `KeyError` on every fixture and on any
    live table predating the column -- it killed the whole panel to print a diagnostic.
    """
    if not len(h):
        return
    if "filing_date" not in h.columns:
        logger.info("13F deadline stamp: no `filing_date` column -> the late-filing residual "
                    "cannot be measured on this read; the period+%sd stamp is unaffected.",
                    SEC_13F_FILING_LAG_DAYS)
        return

    deadline = h["period"] + pd.Timedelta(days=SEC_13F_FILING_LAG_DAYS)
    late = h["filing_date"].gt(deadline).fillna(False)

    value = h.get("value_usd")
    total = float(value.sum()) if value is not None else 0.0
    logger.info("13F deadline stamp: %s of %s (ticker, manager, quarter) rows filed AFTER "
                "period+%sd (%.1f%%), carrying %.1f%% of 13F value; those holdings are visible "
                "from the deadline rather than from their filing date (documented departure "
                "from registry 0.5, deliberate for a multi-thousand-filer aggregate)",
                f"{int(late.sum()):,}", f"{len(h):,}", SEC_13F_FILING_LAG_DAYS,
                100.0 * float(late.mean()),
                100.0 * float(value[late].sum()) / total
                if (total > 0 and value is not None) else 0.0)

def clean_holdings(holdings: pd.DataFrame) -> pd.DataFrame:
    """`sec13f_hr` at its own grain: one row per (ticker, manager, quarter).

    A thin call into the SHARED cleaner -- `holdings_clean.clean_holdings` -- which both 13F
    tables now go through. The only thing that is specific here is the KEY: this table has a
    `ticker` column and the elite one does not, and `position_type` is already resolved at
    extraction so there is nothing to filter, and `cik` arrives padded.
    """
    return _clean(holdings, key=("ticker", "cik", "period"))

def _quarter_features(h: pd.DataFrame, splits: pd.DataFrame | None = None,
                      break_pct: float = COVERAGE_BREAK_DEFAULT,
                      min_prior_holders: int = MIN_PRIOR_HOLDERS) -> pd.DataFrame:
    """Manager-grain 13F -> one row per (ticker, quarter) with the eleven features, stamped
    `as_of = period + 45 days` (the leak-free availability date)."""

    # TODO: verify as_of = filing_date is better than the 45 days rule applied -> real filling date 
    # TODO: check the 2 holes : '2023-12-31', '2025-06-30'. 
    # TODO: clean trade values lower than 1000 ?? larger than 4.e+11 -> seems crazy -> verify who and why
    # TODO: keep (h['deadline']- h['filing_date']).dt.days between(-30,50) -> otherwise do not know when it was done ... To verify
    # TODO: Remove the sec 13 hr data before 2013 cut date, its noise and wrong, do it before looping, save compute and time

    holes, breaks, coverage = _coverage_periods(h, break_pct)
    if holes or breaks:
        logger.info("13F coverage guard: %s hole quarter(s) %s (every feature nulled), "
                    "%s break quarter(s) %s (deltas nulled). Filer counts: %s",
                    len(holes), sorted(str(p.date()) for p in holes),
                    len(breaks), sorted(str(p.date()) for p in breaks),
                    {str(p.date()): int(coverage.loc[p, "filers"])
                     for p in sorted(holes | breaks)})
        
    # D28: the denominator that turns the holder COUNT into a breadth share.
    n_filers = coverage["filers"]
    factors = _split_factors(h, splits)

    rows = []
    for ticker, tdf in h.groupby("ticker"):
        prev: dict = {}
        prev_value = np.nan
        prev_period = None
        for p in sorted(tdf["period"].unique()):
            cur_rows = tdf[tdf["period"] == p]
            cur = dict(zip(cur_rows["cik"], cur_rows["shares"]))
            cur_ciks, prev_ciks = set(cur), set(prev)
            holders = len(cur_ciks)
            n_prev = len(prev_ciks)
            has_prev = n_prev > 0

            # The prior quarter's counts, restated onto THIS quarter's split basis.
            factor = factors.get((ticker, p), 1.0) if has_prev else 1.0
            both = cur_ciks & prev_ciks
            inc = sum(1 for c in both if cur[c] > prev[c] * factor)
            dec = sum(1 for c in both if cur[c] < prev[c] * factor)

            inst_shares = float(sum(cur.values()))
            prev_shares = float(sum(prev.values())) * factor if has_prev else np.nan
            inst_value = float(cur_rows["value_usd"].sum())
            call_v = float(cur_rows["call_value"].sum())
            put_v = float(cur_rows["put_value"].sum())
            total_invested = inst_value + call_v + put_v      # long equity + option exposure
            opt_ratio = ((call_v - put_v) / total_invested) if total_invested > 0 else np.nan

            # crowding: Herfindahl of managers' VALUE shares (high = few dominant holders)
            mv = cur_rows.groupby("cik")["value_usd"].sum()
            tot_mv = float(mv.sum())
            hhi = float(((mv / tot_mv) ** 2).sum()) if tot_mv > 0 else np.nan
            pool = float(n_filers.get(p, np.nan))
            share = holders / pool if pool > 0 else np.nan
            prev_pool = float(n_filers.get(prev_period, np.nan)) if prev_period is not None \
                else np.nan
            prev_share = n_prev / prev_pool if (has_prev and prev_pool > 0) else np.nan

            rows.append({
                "ticker": ticker,
                "period": pd.Timestamp(p),
                "as_of": pd.Timestamp(p) + pd.Timedelta(days=SEC_13F_FILING_LAG_DAYS), # 

                # Carried only so the per-ticker coverage-onset guard can be applied
                # vectorised below; dropped before the frame is returned.
                "_prev_holders": float(n_prev) if has_prev else np.nan,
                "ic_inst_holders": share,
                "inst_shares": inst_shares,
                "inst_value": inst_value,

                # net QoQ dollar flow (long value); NaN on the first observed quarter
                "inst_value_flow": (inst_value - prev_value)
                                   if (has_prev and np.isfinite(prev_value)) else np.nan,
                "ic_inst_breadth_chg": (share - prev_share)
                                   if np.isfinite(prev_share) else np.nan,
                "ic_inst_shares_chg": (inst_shares / prev_shares - 1.0)
                                   if (has_prev and prev_shares > 0) else np.nan,
                "ic_inst_new_buyer_ratio": (len(cur_ciks - prev_ciks) / holders)
                                   if (has_prev and holders > 0) else np.nan,
                "ic_inst_exit_ratio": (len(prev_ciks - cur_ciks) / n_prev)
                                   if has_prev else np.nan,
                "ic_inst_cluster_buying": ((inc - dec) / holders)
                                   if (has_prev and holders > 0) else np.nan,
                "ic_inst_net_options_ratio": opt_ratio,
                "ic_inst_concentration": hhi,
            })
            prev, prev_value, prev_period = cur, inst_value, pd.Timestamp(p)

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
    qf.loc[qf["period"] < INST_LEVEL_FLOOR_PERIOD,
           [c for c in LEVEL_FEATURES if c in qf.columns] + value_cols] = np.nan
    qf.loc[qf["period"] < INST_DELTA_FLOOR_PERIOD,
           [c for c in DELTA_FEATURES if c in qf.columns] + ["inst_value_flow"]] = np.nan

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
        logger.info("13F per-ticker onset guard: %s of %s (ticker, quarter) deltas nulled "
                    "-- the prior quarter carried fewer than %s filers for that name "
                    "(%s ticker(s))", f"{int(removed.sum()):,}", f"{len(qf):,}",
                    min_prior_holders, qf.loc[removed, "ticker"].nunique())
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
        logger.warning("No `prices_splits` -> 13F share changes are NOT split-restated; a "
                       "20-for-1 split will read as +1,900%% accumulation.")

    # clean and report holdings
    holdings = clean_holdings(holdings)
    # ⚠ THE UNIT REPAIR RUNS BEFORE ANYTHING READS A VALUE, and that ordering is the whole
    # point: `_report_late_filings` reports a share OF VALUE, and `_quarter_features` sums
    # value into `ic_inst_concentration`, `ic_inst_net_options_ratio`, `ic_inst_value_to_mcap`
    # and `ic_inst_flow_to_mcap`. Measured 2026-09-14 on the repaired table, 1.074% of rows
    # (2,458 filings in the divide-by-1000 band) carry 84.08% of the table's filed dollars,
    # against 15.22% for the 95.57% of rows that are already correct. Every value-weighted
    # statement made before this call is a statement about those 2,458 filings.
    value_before = pd.to_numeric(holdings.get("value_usd"), errors="coerce").sum()
    holdings, register = repair_value_basis(holdings, stock_close)
    log_register(register, float(value_before), logger)
    _report_late_filings(holdings)

    # build features
    qf = _quarter_features(holdings, splits=splits, break_pct=break_pct,
                           min_prior_holders=min_prior_holders)
    if qf.empty:
        return pd.DataFrame(columns=["date", "ticker"])

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
        shares = fundamentals_to_daily(shares_out_history, "sharesOutstandingPit",
                                       trading_index)
        if not shares.empty and shares.notna().any().any():
            fields["ic_inst_ownership_pct"] = _capped_ownership(
                (inst_sh / shares.where(shares > 0)).replace([np.inf, -np.inf], np.nan))

    if have_shares and stock_close is not None and not stock_close.empty:
        # institutional WEIGHT by VALUE and size-scaled net $ flow, via a point-in-time
        # daily market cap (ffilled sharesOutstanding x daily close x S(d)).
        mcap = daily_market_cap(shares_out_history, stock_close,
                                level_factor=level_factor)
        if mcap.empty:
            # ⚠ NOT SILENT. An empty return here means `shares_out_history` was projected
            # without `sharesOutstanding` (the VENDOR basis `daily_market_cap` requires, NOT
            # `sharesOutstandingPit`), and the two features below would simply be absent from
            # the cube with nothing in the log saying so.
            logger.warning("daily_market_cap returned no columns (shares_out_history has %s; "
                           "it needs `sharesOutstanding`, the VENDOR basis) -> "
                           "ic_inst_value_to_mcap / ic_inst_flow_to_mcap are skipped.",
                           sorted(shares_out_history.columns))
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
    level_floor = INST_LEVEL_FLOOR_PERIOD + pd.Timedelta(days=SEC_13F_FILING_LAG_DAYS)
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
