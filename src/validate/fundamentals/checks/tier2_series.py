"""
tier2_series.py  (src/validate/fundamentals/checks/)
--------------------------------------------------------------------------------------------
TIER 2 -- does a (ticker, field) SERIES behave, on `fundamentals_facts`.

Seven checks, and they are DELIBERATELY OVERLAPPING. Each one misses what another catches, and
the overlap is the design rather than redundancy:

| check           | catches                                          | misses                       |
|-----------------|--------------------------------------------------|------------------------------|
| `trend_break`   | a flat 3x level jump, interpretable, any field   | a 2.3x step; seasonality     |
| `level_outlier` | an anomaly large vs the field's OWN volatility   | a jump in an already-lumpy   |
|                 |                                                  | field, where the MAD is wide |
| `basis_step`    | a step at a ROUTE boundary, at ANY magnitude     | a wrong value that never     |
|                 |                                                  | changes route                |
| `peer_ratio`    | a value in the wrong UNITS or CONCEPT entirely,  | a whole regime wrong the     |
|                 | even when its own series is perfectly smooth     | same way                     |
| `series_shape`  | a field that starts late, goes dark, or has      | a series that is complete    |
|                 | interior holes -- the SHAPE, which no per-cell   | and wrong                    |
|                 | check can see                                    |                              |

## WHY THE FACTS GRAIN AND NOT `fundamentals_history` (decision 41)

On the history grain `level_outlier` and `frozen_series` fire BY CONSTRUCTION over the ~20
forward-filled instant columns -- a balance-sheet line carried unchanged between filings is
"frozen" every time, correctly and uselessly. The facts table is strictly as-filed, so a
repeated value there is a repeated FILING, which is a real question.

Findings still map back to a `(ticker, as_of)` where one exists, so the report reads per
publication event.

## NEVER KEY ON `(fiscal_year, fiscal_period)`

`period_end` is the fact identity. The fiscal LABELS collide **18,604 times in 337,190 rows
(5.5%)** -- edgartools tags a 10-K's current year and its first comparative with the same
`fiscal_year` -- and every check in this module joins across filings, so keying on the label
would silently compare two different periods and call the difference a break.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_extract.utils.fundamentals import reason_codes as rc
from src.validate.fundamentals.checks import (
    FACTS, GRAIN_CELL, GRAIN_SERIES, check)
from src.validate.fundamentals.finding import (
    Finding, HIGH, INFO, MEDIUM, period_key_for_range)
from src.validate.fundamentals.substrate import Substrates
from src.validate.outliers import log_change, modified_zscore

# --------------------------------------------------------------------------- #
# thresholds -- code, with the measurement that set them                        #
# --------------------------------------------------------------------------- #

#: `trend_break`: a value more than this multiple of its own trailing MEDIAN, or less than its
#: reciprocal. A flat, interpretable rule -- "is this 3x its own trend?" -- which is precisely
#: the question a MAD z-score does NOT answer: for a lumpy field (capex, bank provisions) the
#: MAD is wide, so a real 3x jump can score under 3.5 and be missed entirely.
TREND_BREAK_RATIO = 3.0

#: `trend_break`'s reference window, in prior periods. MEDIAN, never mean: a mean lets the
#: outlier inflate its own reference and score itself normal.
TREND_WINDOW = 8
#: Below this many priors there is no trend to break, and the check ABSTAINS.
TREND_MIN_PRIORS = 4

#: `level_outlier`: Iglewicz & Hoaglin's conventional modified-Z cut. Scored on the QoQ LOG
#: CHANGE (decision 60), not the raw level -- see `src/validate/outliers.py` for the 10x-growth
#: measurement that retired the raw-level kernel.
LEVEL_OUTLIER_Z = 3.5
#: Minimum periods before the modified Z has a distribution worth scoring against.
LEVEL_OUTLIER_MIN_PERIODS = 8

#: `scale`: an order-of-magnitude jump. Deliberately far above `trend_break`'s 3x -- this is
#: the "someone tagged thousands as units" rule (DQC 0091/0095/0103/0139/0157), not a growth
#: rule, and at 10x a real business change is essentially never the explanation.
SCALE_RATIO = 10.0

#: `basis_step` / `tag_switch_break`: the level step that makes a provenance change
#: interesting. LOW on purpose -- 1.5x catches CSCO's 2.3x `depAmort` two-basis series, which
#: every statistical check misses because the filer tags each era perfectly consistently.
PROVENANCE_STEP_RATIO = 1.5

#: `peer_ratio`: below this many peers a regime distribution says nothing. GS is the only
#: `broker_dealer` on the roster and `real_estate` has 2 in-sample; both ABSTAIN, visibly.
PEER_RATIO_MIN_PEERS = 5

#: The modified-Z beyond which a filer's ratio to its denominator is out of line with peers.
#:
#: 3.5, Iglewicz & Hoaglin's conventional cut, and NOT the 5.0 this was first written with.
#: Measured while writing `tests/validate/fundamentals/test_planted_violations.py`: the
#: modified Z of a LONE outlier among k otherwise-identical peers is bounded above by
#: `MODIFIED_Z_SCALE * k` -- the mean-absolute-deviation fallback puts the outlier's own
#: deviation in the denominator, divided by k. So at a 5-peer group the maximum achievable
#: score is 0.6745 * 5 = 3.37, and a threshold of 5.0 made the check LITERALLY UNABLE TO FIRE
#: at its own declared minimum peer count. A threshold that cannot be reached is not a strict
#: check; it is a silent one.
#:
#: The bound only bites on a degenerate peer group. Real regimes have dispersion, MAD > 0, and
#: the score is unbounded -- BRK-B's totalDebt/totalAssets at 0.006 against an industrial
#: median of 0.281 scores -8.4. But the degenerate case is exactly what a small regime looks
#: like, which is why the interaction is written down here rather than rediscovered.
PEER_RATIO_Z = 3.5

#: `series_shape`: a series shorter than this has no shape to classify.
SERIES_SHAPE_MIN_EVENTS = 8

#: `peer_ratio`'s denominators. A BALANCE-SHEET field is scaled by total assets and a FLOW by
#: total revenue -- the two quantities every filer in every regime reports, so the ratio is
#: comparable across a peer group without importing a second field's coverage problems.
_BALANCE_DENOMINATOR = "totalAssets"
_FLOW_DENOMINATOR = "totalRevenue"

#: Reason codes that make a gap in a series BENIGN BY CONSTRUCTION, with what each means.
#: `series_shape` reads the modal code inside the gap, and this is what turns the shape from a
#: noisy observation into a diagnosis.
_BENIGN_GAP_CODES: dict[str, str] = {
    rc.INSUFFICIENT_QUARTERS: "the TTM window: fewer than four discrete quarters are visible, "
                              "which is benign by construction at the start of a history",
    rc.NOT_APPLICABLE_FOR_REGIME: "the regime register declares the field absent here",
    rc.NOT_APPLICABLE: "the catalogue declares the field structurally absent for this filer",
    rc.REGIME_BREAK: "a definitional discontinuity in the field itself (ASC 842 / 606 / "
                     "ASU 2016-18 / LDTI), real accounting rather than a data defect",
}


# --------------------------------------------------------------------------- #
# the shared series view                                                       #
# --------------------------------------------------------------------------- #

def _series_frame(sub: Substrates, *, duration_type: str = "quarterly") -> pd.DataFrame:
    """One row per (ticker, field, period_end): the LATEST-FILED value, in period order.

    THE view every check in this module reads, built once per run. Three things it does, each
    of which is load-bearing:

      * collapses amendments -- an amended fact coexisting as its own row must not look like a
        second, disagreeing observation of the same period;
      * keys on `period_end`, never on the fiscal labels (see the module docstring);
      * carries the provenance columns forward, so `basis_step` and `tag_switch_break` can ask
        "did the ROUTE change here?" on the same frame everything else uses.
    """
    facts = sub.facts
    if facts.empty:
        return pd.DataFrame()
    scoped = facts[(facts["duration_type"] == duration_type) & facts["value"].notna()]
    if scoped.empty:
        return pd.DataFrame()
    return (scoped.sort_values(["ticker", "field", "period_end", "filing_date"])
            .drop_duplicates(subset=["ticker", "field", "period_end"], keep="last")
            .reset_index(drop=True))


def _finding_from(row: pd.Series, sub: Substrates, *, check_name: str, severity: str,
                  observed=None, expected=None, deviation=None,
                  detail: dict | None = None, tier: int = 2) -> Finding:
    """A `Finding` carrying one fact row's full provenance -- decision 47's packet.

    Every Tier-2 finding goes through here so that no check can accidentally emit a thin one:
    the reviewing agent's first question is always "is the CHECK wrong?", and that is only
    answerable in one hop if the row says which concept, which route and which filing.
    """
    return Finding(
        check_name=check_name, ticker=str(row["ticker"]), severity=severity, tier=tier,
        substrate=FACTS, field=str(row["field"]),
        period_key=str(pd.Timestamp(row["period_end"]).date()),
        as_of=row.get("filing_date"),
        observed=_float(observed), expected=_float(expected), deviation=_float(deviation),
        source_concept=_text(row.get("source_concept")),
        resolution_method=_text(row.get("resolution_method")),
        roll_up_children=_text(row.get("roll_up_children")),
        root_anchor=_text(row.get("root_anchor")), role_uri=_text(row.get("role_uri")),
        accession_number=_text(row.get("accession_number")),
        cik=sub.cik_for(str(row["ticker"])), detail=detail or {})


# --------------------------------------------------------------------------- #
# the provenance-change checks -- in CORE because no statistic sees them        #
# --------------------------------------------------------------------------- #

@check(name="basis_step", tier=2, substrate=FACTS, severity=HIGH, grain=GRAIN_CELL,
       expected_fire_rate_ceiling=0.02)
def basis_step(sub: Substrates) -> list[Finding]:
    """A level step at the exact boundary where `resolution_method` changes.

    THE check no cross-vintage test can replace, and the reason it is in 5b-core rather than
    5b-stats. MCD's capex steps **35.6x** across the single 2017->2018 route boundary; CSCO's
    `depAmort` runs 2.3x on a two-basis series; VLO's capex goes dark from 2023. In every one
    of those the filer tags the SAME narrow concept consistently within each era, so
    `cross_vintage`, `q4_footing` and `annual_footing` all agree with themselves and report
    nothing.

    The threshold is deliberately LOW (1.5x). A route change is not itself suspicious -- routes
    exist because filers differ -- but a route change that coincides with a level step is a
    change in what the number MEANS, at any magnitude. That is why this catches CSCO at 2.3x
    where `trend_break`'s flat 3x rule does not.
    """
    return _provenance_break(sub, column="resolution_method", check_name="basis_step",
                             why="the RESOLUTION ROUTE changed at this boundary and the "
                                 "level stepped with it -- the number now means something "
                                 "different, which no cross-vintage test can see")


@check(name="tag_switch_break", tier=2, substrate=FACTS, severity=HIGH, grain=GRAIN_CELL,
       expected_fire_rate_ceiling=0.02)
def tag_switch_break(sub: Substrates) -> list[Finding]:
    """A `source_concept` change coinciding with a level step.

    The SEC's own filing-review category, and the mechanism behind two measured defects: MTB's
    110 rows on a post-provision revenue basis (~32% understated) and AXP's 91, both closed by
    a `never_use` entry. Base rate 0.67% / 0.71% across the two rosters.

    Distinct from `basis_step`, which watches the ROUTE: a filer can keep the same route and
    change the concept it resolves through (the ASC-606 cutover), or keep the concept and
    change route (a roll-up becoming a declared total). Both are basis changes; neither
    subsumes the other.
    """
    return _provenance_break(sub, column="source_concept", check_name="tag_switch_break",
                             why="the SOURCE CONCEPT changed at this boundary and the level "
                                 "stepped with it -- an era-aware basis change, which is why "
                                 "comparing each row's tag to a global mode does not work")


def _provenance_break(sub: Substrates, *, column: str, check_name: str,
                      why: str) -> list[Finding]:
    """Shared engine: a change in `column` between consecutive periods, WITH a level step.

    Both conditions, always. A provenance change alone is ordinary; a level step alone is
    `trend_break`'s. It is the coincidence that says the basis moved.
    """
    out: list[Finding] = []
    series = _series_frame(sub)
    if series.empty or column not in series.columns:
        return out
    sub.denominator(check_name, len(series))

    for (ticker, field), rows in series.groupby(["ticker", "field"], sort=False):
        if len(rows) < 2:
            continue
        values = rows["value"].astype(float).to_numpy()
        provenance = rows[column].astype(object).to_numpy()
        for i in range(1, len(rows)):
            if provenance[i] == provenance[i - 1] or values[i - 1] == 0:
                continue
            ratio = abs(values[i]) / abs(values[i - 1]) if values[i - 1] else np.inf
            if not np.isfinite(ratio) or ratio == 0:
                continue
            if PROVENANCE_STEP_RATIO > ratio > 1 / PROVENANCE_STEP_RATIO:
                continue
            out.append(_finding_from(
                rows.iloc[i], sub, check_name=check_name, severity=HIGH,
                observed=values[i], expected=values[i - 1], deviation=ratio,
                detail={"changed": column,
                        "from": _text(provenance[i - 1]), "to": _text(provenance[i]),
                        "previous_period_end":
                            str(pd.Timestamp(rows.iloc[i - 1]["period_end"]).date()),
                        "step_ratio": round(float(ratio), 3),
                        "threshold": PROVENANCE_STEP_RATIO, "why": why}))
    return out


# --------------------------------------------------------------------------- #
# series_shape -- the missing dimension                                        #
# --------------------------------------------------------------------------- #

COMPLETE, INTERIOR_GAP, LATE_START, EARLY_STOP, SPARSE = (
    "complete", "interior_gap", "late_start", "early_stop", "sparse")


@check(name="series_shape", tier=2, substrate=FACTS, severity=HIGH, grain=GRAIN_SERIES,
       expected_fire_rate_ceiling=0.15)
def series_shape(sub: Substrates) -> list[Finding]:
    """Classify each (ticker, field) series against the filer's own period grid.

    ## Why this check had to exist

    `coverage_field` fires on a CELL. It can never distinguish a random hole from a legitimate
    late start, because it never looks at the shape of the series. That was the missing
    dimension, and it is the only thing that catches three whole defect classes:

        interior_gap   present before AND after a hole   -> "random quarters missing"
        late_start     absent, then present from D on    -> "the concept started late"
        early_stop     present until D, absent after     -> "it went dark" (VLO capex, 2023-07)
        sparse         no contiguous run                 -> hand to `periodicity` (5b-stats)

    ## The gap's modal `dc_code` IS the diagnosis

    This is what makes the check precise instead of noisy. A gap whose code is
    `insufficient_quarters` is the TTM window and benign by construction; one whose code is
    `not_disclosed` is a MISSING TAG and worth a filing.

    ## The `late_start` oracle ladder, in order (decision 56)

      1. the catalogue's own `regime_break` block -- ASC 842 / ASC 606 / ASU 2016-18 / LDTI
         adoption dates. A match is `info` and never enters the queue: AAPL's
         `operatingLeaseLiability` starting 2020-01-29 IS Apple adopting ASC 842;
      2. a `cik_cutover` entry or a recent first-trade date -- also `info`;
      3. the modal `dc_code` in the absent stretch.

    None of the three explains it -> `high`. AAPL's `totalDebt` from 2013-07-24 is that case:
    no regime break, listed 1980, `not_disclosed` over 16 events -> `high` -> the agent opens
    the filing -> `accepted`, evidence "first bond issue 30 April 2013". Settled once, forever.

    Flagging every late start WITHOUT the ladder was rejected outright: it is a one-time queue
    of every ASC-842 and ASC-606 adopter across ~500 tickers x 60 fields.
    """
    out: list[Finding] = []
    examined = 0
    # BOTH duration shapes, each against its OWN grid. A balance-sheet field is an `instant`
    # and a flow is `quarterly`; running only the quarterly frame made every instant field --
    # `totalDebt`, `cash`, `goodwill`, the share counts, roughly a third of the table --
    # invisible to the one check that detects a shape. The grids must stay separate too: a
    # filer publishes an instant at every period end and a duration only for closed windows,
    # so a shared grid would report a permanent gap on whichever shape is sparser.
    for duration_type in ("quarterly", "instant"):
        frame = _series_frame(sub, duration_type=duration_type)
        if frame.empty:
            continue
        grids = {ticker: sorted(pd.to_datetime(rows["period_end"]).unique())
                 for ticker, rows in frame.groupby("ticker")}
        examined += sum(len(sub.catalogue.history_fields) for _ in grids)

        for (ticker, field), rows in frame.groupby(["ticker", "field"], sort=False):
            grid = grids.get(ticker, [])
            if len(grid) < SERIES_SHAPE_MIN_EVENTS:
                continue                   # too short to have a shape; abstains
            present = set(pd.to_datetime(rows["period_end"]))
            shape, span = _classify(grid, present)
            if shape == COMPLETE:
                continue
            gap_code = _modal_gap_code(sub, str(ticker), field)
            severity, reason = _shape_severity(sub, str(ticker), field, shape, gap_code, span)
            out.append(Finding(
                check_name="series_shape", ticker=str(ticker), severity=severity, tier=2,
                substrate=FACTS, field=str(field),
                period_key=period_key_for_range(*span),
                as_of=rows["filing_date"].max(),
                observed=float(len(present)), expected=float(len(grid)),
                source_concept=_text(rows.iloc[-1].get("source_concept")),
                resolution_method=_text(rows.iloc[-1].get("resolution_method")),
                cik=sub.cik_for(str(ticker)),
                detail={"shape": shape, "modal_gap_dc_code": gap_code,
                        "duration_type": duration_type,
                        "missing_periods": int(len(grid) - len(present)),
                        "grid_periods": int(len(grid)), "verdict": reason}))
    sub.denominator("series_shape", examined)
    return out


def _classify(grid: list, present: set) -> tuple[str, tuple]:
    """`(shape, (span_start, span_end))` for one series against its filer's period grid.

    The span is the STRETCH THE FINDING IS ABOUT -- the gap, the absent head, the absent tail
    -- not the whole series. That is what `period_key` records, so two different gaps in one
    (ticker, field) are two findings with two identities rather than one that keeps changing.
    """
    flags = [period in present for period in grid]
    if all(flags):
        return COMPLETE, (grid[0], grid[-1])
    if not any(flags):
        return SPARSE, (grid[0], grid[-1])

    first, last = flags.index(True), len(flags) - 1 - flags[::-1].index(True)
    interior_missing = [i for i in range(first, last + 1) if not flags[i]]
    # A run so broken it has no contiguous stretch is `sparse`, and belongs to `periodicity`
    # (5b-stats) rather than here -- an annual-only field is not a series with holes.
    if len(interior_missing) > (last - first) / 2:
        return SPARSE, (grid[first], grid[last])
    if interior_missing:
        return INTERIOR_GAP, (grid[interior_missing[0]], grid[interior_missing[-1]])
    if first > 0:
        return LATE_START, (grid[0], grid[first])
    return EARLY_STOP, (grid[last], grid[-1])


def _modal_gap_code(sub: Substrates, ticker: str, field: str) -> str | None:
    """The most common `dc_code` this (ticker, field) ever earned -- the gap's diagnosis."""
    if sub.codes.empty:
        return None
    rows = sub.codes[(sub.codes["ticker"] == ticker) & (sub.codes["field"] == field)]
    return str(rows["dc_code"].mode().iloc[0]) if len(rows) else None


def _shape_severity(sub: Substrates, ticker: str, field: str, shape: str,
                    gap_code: str | None, span: tuple) -> tuple[str, str]:
    """`(severity, the reason in words)` -- the oracle ladder of decision 56.

    The reason string goes into the finding, so an agent reads WHY it is `info` rather than
    having to re-derive the ladder. An `info` with no stated reason is indistinguishable from
    a check that gave up.
    """
    if shape == SPARSE:
        return INFO, ("no contiguous run at all -- this is a periodicity question "
                      "(an annual-only field), not a gap; 5b-stats' `periodicity` owns it")
    if gap_code in _BENIGN_GAP_CODES:
        return INFO, _BENIGN_GAP_CODES[gap_code]
    if gap_code == rc.PERIOD_INTERSECTION_PARTIAL:
        return MEDIUM, ("route 3b's strict period intersection refused these windows: the "
                        "filer declares the leaves and reported only SOME of them, so the sum "
                        "would be short by a whole leg. 128 known rows across EQIX capex 40, "
                        "EQIX depAmort 40, SCHW cash 34, NEE ppeNet 8, VRT depAmort 6")

    if shape == LATE_START:
        # Oracle 1: does a definitional break in the FIELD land on the start date?
        effective = sub.catalogue.regime_break_effective(field)
        if effective is not None and _within_a_year(effective, span[1]):
            return INFO, (f"the start matches this field's own regime_break date "
                          f"{pd.Timestamp(effective).date()} -- a standard's adoption "
                          "(ASC 842 / 606 / ASU 2016-18 / LDTI), which is real accounting")
        # Oracle 2 is the LISTING date, which is not in these substrates. Say so.
        return HIGH, ("no regime_break explains this start, and the listing / cutover date is "
                      "NOT readable from fundamentals_* -- check it before accepting. AAPL's "
                      "totalDebt from 2013-07-24 is this exact shape and the answer was its "
                      "first bond issue, 30 April 2013")
    if shape == EARLY_STOP:
        return HIGH, ("a field that goes dark mid-history is almost always a defect -- VLO's "
                      "capex from 2023-07 tags neither concept undimensioned in 21 of 63 "
                      "filings, and nothing else in the tier detects a shape")
    return HIGH, (f"an interior gap with modal code {gap_code!r} -- present before AND after, "
                  "so the filer kept reporting and we stopped resolving. A MISSING TAG")


def _within_a_year(effective, date) -> bool:
    """Is `date` within a year of `effective`? Adoption dates are per-filer, not per-day: a
    standard effective for fiscal years beginning after a date lands on different calendar
    quarters for a Sep, Jan and Dec year-end."""
    if effective is None or date is None or pd.isna(date):
        return False
    return abs((pd.Timestamp(date) - pd.Timestamp(effective)).days) <= 365


# --------------------------------------------------------------------------- #
# the level checks                                                             #
# --------------------------------------------------------------------------- #

@check(name="trend_break", tier=2, substrate=FACTS, severity=HIGH, grain=GRAIN_CELL,
       expected_fire_rate_ceiling=0.06)
def trend_break(sub: Substrates) -> list[Finding]:
    """A value more than 3x, or less than 1/3x, its own trailing MEDIAN of the last 4-8 periods.

    ## Why a MAD z-score is not this rule

    "Is this value 3x its own trend?" reads like a restatement of `level_outlier`, and it is
    not. For a LUMPY field -- capex, bank provisions -- the MAD is wide, so a genuine 3x jump
    scores UNDER 3.5 and is never reported. This is the flat, interpretable rule that catches
    it, and it is why both checks are in core.

    MEDIAN, never mean: a mean lets the outlier drag its own reference upward and score itself
    normal. Abstains below `TREND_MIN_PRIORS` priors, because four points are not a trend.

    ## THE SEASONALITY COST, STATED UP FRONT

    A plain trailing median WILL fire on genuinely seasonal filers -- retail Q4, KR's 16-week
    Q1, weather-driven utility quarters. That is an accepted cost (decision 59), not an
    oversight, and the calibration pass reports the fire rate BY REGIME so the false-positive
    population is a number rather than a surprise.

    NAMED REMEDY, recorded here so it does not have to be re-derived: `fiscal_quarter` is on
    every row since Phase 5, so switching the reference to the same fiscal quarter of the prior
    three years is a one-line change. A same-fiscal-quarter median was proposed and NOT chosen
    up front, because it triples the history a filer needs before the check says anything.

    ## THE BY-REGIME MEASUREMENT decision 59 asked for -- 2026-08-24, 54 tickers

    Overall 5.24% (1,553 of 29,661). Findings per ticker, worst first:

        energy 68.0 | insurer 41.8 | hybrid 33.0 | industrial 25.2
        utility 23.8 | real_estate 22.7 | bank 20.9 | broker_dealer 20.5

    ENERGY FIRES 2.7x INDUSTRIAL, and that is the predicted cost arriving on schedule: a
    commodity-price-driven quarter genuinely moves 3x, and so does an insurer's catastrophe
    quarter. The remedy above is NOT applied yet -- 5.24% is inside the 6% ceiling, so the
    queue is not being drowned, and the trigger for switching to a same-fiscal-quarter
    reference is `trend_break` dominating a real triage session rather than a number here.

    CEILING 6%, SET FROM THAT MEASUREMENT (was 3%, a guess made before any data existed).

    A TTM basis was also rejected: TTM smears one bad quarter across four windows, so the check
    fires 4x per defect and points at the wrong quarter.
    """
    out: list[Finding] = []
    series = _series_frame(sub)
    if series.empty:
        return out
    sub.denominator("trend_break", len(series))

    for (ticker, field), rows in series.groupby(["ticker", "field"], sort=False):
        values = rows["value"].astype(float).to_numpy()
        for i in range(TREND_MIN_PRIORS, len(values)):
            window = values[max(0, i - TREND_WINDOW):i]
            reference = float(np.median(window))
            if reference == 0 or not np.isfinite(reference):
                continue
            ratio = values[i] / reference
            if 1 / TREND_BREAK_RATIO <= ratio <= TREND_BREAK_RATIO:
                continue
            out.append(_finding_from(
                rows.iloc[i], sub, check_name="trend_break", severity=HIGH,
                observed=values[i], expected=reference, deviation=ratio,
                detail={"trailing_median": reference, "window": int(len(window)),
                        "ratio": round(float(ratio), 3), "threshold": TREND_BREAK_RATIO,
                        "known_false_positive_population":
                            "seasonal filers -- retail Q4, KR's 16-week Q1, weather-driven "
                            "utility quarters. Check the regime before believing this one",
                        "why": "a flat 3x rule, because for a lumpy field the MAD is wide "
                               "enough that a real 3x jump scores under 3.5"}))
    return out


@check(name="level_outlier", tier=2, substrate=FACTS, severity=MEDIUM, grain=GRAIN_CELL,
       expected_fire_rate_ceiling=0.06)
def level_outlier(sub: Substrates) -> list[Finding]:
    """Modified-Z > 3.5 (Iglewicz & Hoaglin) on the QoQ LOG CHANGE, per (ticker, field).

    ## The plan and the code disagreed, and the plan won (decision 60)

    `src/validate/outliers.py` scored `modified_zscore(raw levels)`. That rule flags the entire
    recent era of any growing company: a 10x compounding revenue over fifteen years puts every
    recent quarter many MADs above its own median, correctly and uselessly. The kernel now
    scores a log CHANGE, and `tests/validate/test_outliers.py` is the guard -- a smooth 10x
    series over 60 quarters must produce ZERO findings, and the same series with one planted 3x
    spike must produce exactly the spike and its reversion.

    `medium`, not `high`: this is a statistical CANDIDATE. It says the step is unlike this
    field's other steps, which is a reason to look and never a verdict.

    CEILING 6%, measured at 5.28% (1,566 of 29,661) on the 54-ticker table 2026-08-24. The
    previous 5% was a guess set before any data existed, and it sat just under the true rate.

    ## Where it goes blind

    A log ratio does not exist across a zero or a sign change, so a value crossing zero is
    INVISIBLE here -- that belongs to `impossible_value` and `sign_convention`. A series with
    no dispersion at all scores zero everywhere. Both abstentions are in the kernel's docstring.
    """
    out: list[Finding] = []
    series = _series_frame(sub)
    if series.empty:
        return out
    sub.denominator("level_outlier", len(series))

    for (ticker, field), rows in series.groupby(["ticker", "field"], sort=False):
        if len(rows) < LEVEL_OUTLIER_MIN_PERIODS:
            continue
        values = rows["value"].astype(float).to_numpy()
        changes = log_change(values, lag=1)
        defined = changes[np.isfinite(changes)]
        if defined.size < 3:
            continue
        scores = np.where(np.isfinite(changes),
                          modified_zscore(changes, reference=defined), np.nan)
        for i in np.flatnonzero(scores > LEVEL_OUTLIER_Z):
            out.append(_finding_from(
                rows.iloc[int(i)], sub, check_name="level_outlier", severity=MEDIUM,
                observed=values[int(i)], expected=values[int(i) - 1] if i else None,
                deviation=float(scores[int(i)]),
                detail={"modified_z_on_log_change": round(float(scores[int(i)]), 2),
                        "threshold": LEVEL_OUTLIER_Z,
                        "log_change": round(float(changes[int(i)]), 4),
                        "periods_scored": int(defined.size),
                        "why": "the STEP into this period is unlike this field's other "
                               "steps. A candidate, not a verdict"}))
    return out


@check(name="scale", tier=2, substrate=FACTS, severity=MEDIUM, grain=GRAIN_CELL,
       expected_fire_rate_ceiling=0.02)
def scale(sub: Substrates) -> list[Finding]:
    """An ORDER-OF-MAGNITUDE jump against the field's own history (DQC 0091/0095/0103/0139/0157).

    Distinct from `trend_break` by threshold and by intent. At 10x, a real business change is
    essentially never the explanation -- this is the "someone tagged thousands where the rest
    of the filing is units" rule, and the scale-factor defects XBRL-US's own DQC ruleset spends
    five separate rules on. ORCL's FY2020 Q4 revenue at $39,068M against a ~$10,439M quarter is
    the shape, and it reached the table because a full-year Revenues fact carried a Q4 window.

    Scored against the field's own MEDIAN over the whole history rather than a trailing window:
    a units error is a property of one filing, so the comparison should be to everything else
    the filer ever said, not just to last quarter.

    CEILING 2%, measured at 1.53% (455 of 29,661) on the 54-ticker table 2026-08-24.
    """
    out: list[Finding] = []
    series = _series_frame(sub)
    if series.empty:
        return out
    sub.denominator("scale", len(series))

    for (ticker, field), rows in series.groupby(["ticker", "field"], sort=False):
        if len(rows) < TREND_MIN_PRIORS:
            continue
        values = rows["value"].astype(float).to_numpy()
        reference = float(np.median(np.abs(values[values != 0]))) if (values != 0).any() else 0
        if reference == 0 or not np.isfinite(reference):
            continue
        for i, value in enumerate(values):
            if value == 0:
                continue
            ratio = abs(value) / reference
            if 1 / SCALE_RATIO <= ratio <= SCALE_RATIO:
                continue
            out.append(_finding_from(
                rows.iloc[i], sub, check_name="scale", severity=MEDIUM,
                observed=value, expected=reference, deviation=ratio,
                detail={"median_abs_level": reference, "ratio": round(float(ratio), 2),
                        "threshold": SCALE_RATIO, "periods": int(len(values)),
                        "why": "at 10x against the field's own median, a units error is a "
                               "likelier explanation than a business change"}))
    return out


# --------------------------------------------------------------------------- #
# peer_ratio -- the only generic wrong-concept detector                        #
# --------------------------------------------------------------------------- #

@check(name="peer_ratio", tier=2, substrate=FACTS, severity=HIGH, grain=GRAIN_CELL,
       expected_fire_rate_ceiling=0.03)
def peer_ratio(sub: Substrates) -> list[Finding]:
    """A field's ratio to a stable denominator, MAD-scored against SAME-REGIME peers AT THE
    SAME DATE.

    ## The only rule that catches a value resolved to an ENTIRELY WRONG CONCEPT

    Every other check in this tier compares a series to ITSELF. A field resolved to the wrong
    concept for its whole history is perfectly smooth, never changes route, never steps, and
    passes all of them. `totalDebt` resolving to an operating-lease liability is exactly that:
    BRK-B at $4.9-6.3bn against a real long-term debt in the tens of billions, META at
    $7.6-16.7bn, PGR at $179-211M. All three were found by a human noticing. This is the rule
    that finds the next one WITHOUT anyone knowing the mechanism first, and the same rule
    covers AMT's $1.9M note-level `longTermDebt`, PG's $28,400M annual revenue and MCD's
    12x-low capex.

        BRK-B totalDebt / totalAssets = 0.006
          industrial peers @2021-06-30, n=27
          median 0.281   MAD-z -8.4
          -> HIGH: 47x below the peer median

    ## IT ABSTAINS, VISIBLY, AND IT IS NEVER `critical`

    Below `PEER_RATIO_MIN_PEERS` there is no distribution: GS is the only `broker_dealer` on
    the roster and `real_estate` has two in-sample. Those regimes are simply NOT CHECKED, and
    the abstention is reported rather than read as a pass.

    The other limit is structural and no threshold fixes it: if a whole regime is wrong the
    same way, the peer median is wrong too and this check goes blind. It is corroboration for
    `basis_step`, never a standalone verdict -- which is why it is `high` and not `critical`.
    """
    out: list[Finding] = []
    series = _series_frame(sub, duration_type="instant")
    flows = _series_frame(sub)
    if series.empty and flows.empty:
        return out

    regimes = sub.regime_by_ticker
    denominators = _denominator_levels(sub)
    sub.denominator("peer_ratio", len(series) + len(flows))

    for frame, denominator_field in ((series, _BALANCE_DENOMINATOR),
                                     (flows, _FLOW_DENOMINATOR)):
        if frame.empty:
            continue
        scaled = _scaled_ratios(frame, denominators, denominator_field, regimes)
        for (regime, field, period_end), group in scaled.groupby(
                ["regime", "field", "period_end"], sort=False):
            if len(group) < PEER_RATIO_MIN_PEERS:
                continue                     # abstains; reported by `peer_ratio_abstentions`
            scores = modified_zscore(group["ratio"].to_numpy())
            median = float(np.median(group["ratio"].to_numpy()))
            for position, score in zip(group.index, scores):
                if not np.isfinite(score) or score <= PEER_RATIO_Z:
                    continue
                row = frame.loc[position]
                ratio = float(scaled.loc[position, "ratio"])
                out.append(_finding_from(
                    row, sub, check_name="peer_ratio", severity=HIGH,
                    observed=ratio, expected=median,
                    deviation=float(score),
                    detail={"denominator": denominator_field,
                            "regime": str(regime), "peers": int(len(group)),
                            "peer_median_ratio": median, "mad_z": round(float(score), 2),
                            "times_off_median": (round(median / ratio, 1) if ratio
                                                 else None),
                            "blind_spot": "if the whole regime is wrong the same way, the "
                                          "peer median is wrong too and this says nothing",
                            "why": "the only rule that catches a value resolved to an "
                                   "entirely wrong concept without a human noticing first"}))
    return out


@check(name="peer_ratio_abstentions", tier=2, substrate=FACTS, severity=INFO,
       grain=GRAIN_SERIES, expected_fire_rate_ceiling=1.0)
def peer_ratio_abstentions(sub: Substrates) -> list[Finding]:
    """Which regimes `peer_ratio` could NOT check, and how many peers each has.

    A separate check rather than a log line, because an abstention that is not in the finding
    ledger is invisible to anyone reading the ledger -- and "peer_ratio reported nothing for
    GS" would otherwise read as a pass. `info`.
    """
    regimes = sub.regime_by_ticker
    if not regimes:
        return []
    counts = pd.Series(list(regimes.values())).value_counts()
    sub.denominator("peer_ratio_abstentions", len(counts))
    return [Finding(
        check_name="peer_ratio_abstentions", ticker="", severity=INFO, tier=2,
        substrate=FACTS, field=str(regime), observed=float(count),
        expected=float(PEER_RATIO_MIN_PEERS),
        detail={"regime": str(regime), "filers_in_scope": int(count),
                "minimum": PEER_RATIO_MIN_PEERS,
                "why": "peer_ratio did NOT run for this regime -- too few filers to have a "
                       "distribution. Silence here is an abstention, not a pass"})
        for regime, count in counts.items() if count < PEER_RATIO_MIN_PEERS]


def _denominator_levels(sub: Substrates) -> dict[tuple[str, str, pd.Timestamp], float]:
    """`{(ticker, denominator_field, period_end): value}` for the two scaling denominators."""
    facts = sub.facts
    if facts.empty:
        return {}
    wanted = facts[facts["field"].isin({_BALANCE_DENOMINATOR, _FLOW_DENOMINATOR})
                   & facts["value"].notna()]
    latest = (wanted.sort_values("filing_date")
              .drop_duplicates(subset=["ticker", "field", "period_end"], keep="last"))
    return {(t, f, p): float(v) for t, f, p, v in
            zip(latest["ticker"], latest["field"], latest["period_end"], latest["value"])}


def _scaled_ratios(frame: pd.DataFrame, denominators: dict, denominator_field: str,
                   regimes: dict[str, str]) -> pd.DataFrame:
    """`frame` reduced to the rows that HAVE a denominator, with `ratio` and `regime` attached.

    Rows whose denominator is missing or zero are dropped rather than scored: a ratio to a
    missing total is not a small number, it is no number, and scoring it as one would put
    every filer with a coverage gap at the extreme of the peer distribution.
    """
    ratios, keep = [], []
    for position, row in frame.iterrows():
        if row["field"] in (_BALANCE_DENOMINATOR, _FLOW_DENOMINATOR):
            continue                        # a denominator is not compared against itself
        denominator = denominators.get((row["ticker"], denominator_field, row["period_end"]))
        if not denominator:
            continue
        ratios.append(float(row["value"]) / denominator)
        keep.append(position)
    out = frame.loc[keep, ["ticker", "field", "period_end"]].copy()
    out["ratio"] = ratios
    out["regime"] = out["ticker"].map(regimes)
    return out.dropna(subset=["regime"])


# --------------------------------------------------------------------------- #
# shared helpers                                                               #
# --------------------------------------------------------------------------- #

def _float(value) -> float | None:
    """A payload number as a plain float, or None. Never NaN -- see `finding._float`."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(out) else out


def _text(value) -> str | None:
    """A provenance cell as a string, or None. Keeps `pd.NA` / NaN out of the JSON payload."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value)
    return text if text and text.lower() not in ("nan", "none", "<na>") else None
