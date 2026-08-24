"""
periods.py (src/data_extract/utils/fundamentals/periods.py)
--------------------------------------------------------------------------------------
The period engine: turn the as-filed duration facts in `fundamentals_facts` into
**discrete quarters that are actually discrete**, and a TTM that is not a staircase.

`fundamentals_facts` is deliberately as-filed -- one row per period SHAPE the filer
actually tagged. Filers tag flows three different ways in the same year (a discrete
3-month column, a cumulative year-to-date column, an annual column), and ASC 230 only
requires cash-flow amounts CUMULATIVELY, so for many fields the discrete quarter is never
published at all. This module reconstructs it, in memory, and records how.

Two defects it exists to remove, both measured on the legacy table:

  * **The Q4 footing check was vacuous.** All 203,798 legacy Q4 rows were derived as
    `FY - (Q1+Q2+Q3)`, so the validator's `Q1+Q2+Q3+Q4 == FY` test passed 99.73% **by
    construction**. An identity cannot take its own output as an input. The primary ladder
    here is `Q4 = FY - YTD9` -- one as-reported nine-month number instead of three derived
    quarters -- which leaves `Q1+Q2+Q3` genuinely free to disagree, and so makes the
    footing check a real test for the first time.
  * **The TTM staircase.** The legacy annual fallback (`ttm_a -> <field>_ann.ffill(4)`)
    froze **1,622 of 26,242 consecutive `totalRevenue` pairs (6.2%; APA 100%, XOM 36%)**,
    which made `revenueGrowth` exactly 0 for three quarters in four. A TTM is emitted here
    ONLY from four discrete quarters; otherwise it is NULL with `insufficient_quarters`.
    This REDUCES coverage on purpose.

**Nothing in this module reads `fiscal_period`.** SEC's `fp` labels a fact by the
*filing's* period focus, so one calendar quarter can appear as Q1, Q2 AND Q3 across
successive 10-Qs (edgartools GH #848), and a 10-K's discrete fourth-quarter column is
labelled `FY` like the annual one. Every input is selected by its calendar WINDOW. That
makes three expensively-learned repairs in the deleted 1,140-line engine structurally
unnecessary rather than merely dropped: the `YTD3/6/9/12` -> quarter label remap, the
"never trust a native Q4 label" rule, and `_reassign_misordered_native_q4` (MAA tagged all
three of FY2017's discrete quarters `Q4`). The shape bands below replace all of them.

**The subtraction is CROSS-FILING, and that is the hard part.** The FY fact comes from the
10-K and the YTD9 fact from the Q3 10-Q -- two separate `resolve_field` calls, which can
land on two different concepts. Requiring them to match outright is what made 107 real
quarters underivable in the legacy engine (ATO tagged D&A `DepreciationAndAmortization` in
its 10-Qs and `DepreciationDepletionAndAmortization` in its 10-K for NINE consecutive
years; AFL/DTE/ATO/C alternate between the two pretax-income variants; several filers
switch revenue concept at the ASC-606 cutover). So a switch is allowed and recorded, and
only then does the scale test decide -- which the legacy engine had to apply
unconditionally, because it had no `source_concept` column to compare.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

from src.data_extract.utils.fundamentals.kpi_catalogue import DEFAULT_CONFIG_DIR, FieldSpec
from src.utils.config import read_config


@dataclass(frozen=True)
class PeriodGuards:
    """The two calibrated thresholds the derivation guards need, injected rather than
    imported so a known-truth test can state its own and so the numbers stay in
    `configs/configs.yml` where they are readable without opening a module."""

    max_opposite_sign_ratio: float
    concept_switch_scale_max: float
    share_basis_max_ratio: float


@lru_cache(maxsize=4)
def load_guards(config_dir: str = DEFAULT_CONFIG_DIR) -> PeriodGuards:
    """Read the guards from `configs.yml`. Cached per config dir -- the period engine runs
    once per (ticker, field) and must not re-read a YAML tree ~50 times a filing."""
    block = read_config(config_dir).data_extract.fundamentals_periods
    return PeriodGuards(
        max_opposite_sign_ratio=float(block.max_opposite_sign_q4_ratio),
        concept_switch_scale_max=float(block.q4_tag_mismatch_fy_max),
        share_basis_max_ratio=float(block.share_basis_max_ratio))

# --------------------------------------------------------------------- period shapes ---

#: Day-count bands classifying a duration fact by SHAPE rather than by the filer's own
#: `fiscal_period` label. Bands are wide because a 52/53-week issuer's fiscal quarter can
#: run ~112 days while a calendar one is 90/91/92, and they are deliberately DISJOINT with
#: gaps: a 130-day duration is neither a quarter nor a half, and calling it either would
#: corrupt a TTM far more quietly than carrying it as unusable. Measured: exactly 10 of
#: 109,267 valued in-sample rows land outside every band.
_DURATION_BANDS: tuple[tuple[int, int, str], ...] = (
    (60, 120, "quarterly"),
    (150, 210, "ytd6"),
    (240, 300, "ytd9"),
    (330, 400, "annual"),
)
QUARTERLY, YTD6, YTD9, ANNUAL = (name for _, _, name in _DURATION_BANDS)
INSTANT = "instant"
OTHER_SHAPE = "other"

def period_shape(period_type: str, days: float | None) -> str:
    """The period's SHAPE -- what the ladder below selects on.

    `other` is a first-class outcome, not a failure: a 47-day stub period is real, and the
    honest thing is to carry it unusable rather than to round it into a band.
    """
    if period_type == INSTANT:
        return INSTANT
    if days is None or pd.isna(days):
        return OTHER_SHAPE
    for low, high, name in _DURATION_BANDS:
        if low <= days <= high:
            return name
    return OTHER_SHAPE


# --------------------------------------------------------------------------- vocabulary ---

#: How a discrete quarter was obtained. Carried on the value's own row, never re-joined --
#: `q4_footing` is only a genuine test on the rows whose basis is NOT `FY_MINUS_QUARTERS`,
#: and the validator cannot know which those are unless the row says so.
AS_REPORTED = "as_reported"
Q2_FROM_YTD6 = "ytd6_minus_q1"
Q3_FROM_YTD9 = "ytd9_minus_ytd6"
FY_MINUS_YTD9 = "fy_minus_ytd9"
FY_MINUS_QUARTERS = "fy_minus_q1q2q3"

#: TTM bases.
TTM_FOUR_QUARTERS = "sum_4q"
TTM_FOUR_QUARTER_MEAN = "mean_4q"
TTM_AS_REPORTED_ANNUAL = "as_reported_annual"

#: `dc_code`s this module attaches to a TTM it refuses to emit.
#:
#: A refused *quarter* carries no code -- `_derived` returns None and the window simply has
#: no row. That is a genuine hole against the plan's "zero unexplained nulls", and it is
#: Phase 5's to close: `fundamentals_reason_codes` is the table that records why a value is
#: absent, and inventing a second mechanism here would give it two sources of truth.
INSUFFICIENT_QUARTERS = "insufficient_quarters"
SPLIT_BASIS_MISMATCH = "split_basis_mismatch"

#: `dc_code` for the D1b refusal (4c.8): a duration fact tagged into a QUARTERLY context
#: whose value is the whole fiscal year, where the filer publishes NO annual-window fact for
#: that year at all -- so there is nothing to compare it against and D1's value test cannot
#: run. Declining to guess is not inference; reclassifying it would be, because the only
#: available "is this really the year?" test would use the very quarters being derived.
#:
#: Measured on ORCL `totalRevenue`: **9 rows across fiscal 2018-2022**, in three consecutive
#: 10-Ks. The plan recorded this as one row (FY2020, $39,068M against a true ~$10,439M); it
#: is a five-year habit of one filer, and every one of those years would otherwise propagate
#: a ~4x Q4 into four TTM windows, `revenueGrowth`, and every peer z-score built on them.
AMBIGUOUS_DURATION = "ambiguous_duration"

#: A trailing-twelve window must be four quarters covering roughly a year. The bounds are
#: the annual band's, so a 52/53-week issuer's 371-day year and a calendar 365 both pass
#: while a year with a stub or a missing quarter does not.
TTM_MIN_DAYS, TTM_MAX_DAYS = 330, 400
TTM_QUARTERS = 4

#: Two period ends this close are the SAME period tagged twice -- see `_latest_per_window`.
_SAME_PERIOD_DAYS = 7

_QUARTER_COLUMNS: tuple[str, ...] = (
    "ticker", "field", "period_start", "period_end", "period_days", "value", "basis",
    "known_from", "source_concept", "concept_switch", "fiscal_year", "fiscal_quarter")


# ------------------------------------------------------------------ selection helpers ---

def _inclusive_days(days):
    """Day counts for the share-day arithmetic, counting BOTH endpoints.

    `period_days` is `(end - start).days`, which is one short of the days a period actually
    covers and, worse, is not additive across abutting periods: a calendar year reads 365
    while its nine-month and fourth-quarter legs read 273 + 91 = 364. Multiplying and
    dividing by that loses a day at every junction and put the share-day derivation ~1%
    out. Adding the endpoint back makes the legs foot exactly (274 + 92 = 366).
    """
    return days + 1


def _latest_per_window(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per calendar window, the LATEST filing winning.

    A window is re-tagged every time it appears as a comparative in a later filing, and the
    values can differ: `us-gaap:Revenues` for BAC FY2023 is $98,581M as filed and
    **$102,769M** as re-presented in the FY2025 10-K. Taking the latest is correct *inside a
    point-in-time replay*, because Phase 5 hands this function only the facts with
    `filing_date <= as_of`; the restatement therefore becomes visible on the day it was
    published and not a day earlier.

    The window's identity is its **END, within a few days** -- not its exact (start, end)
    pair. Filers nudge the boundary day between filings and the two contexts are the same
    quarter: BRK-B tags Q3-2013 as both `06-29 ->` and `07-01 -> 09-30`, KR ships three
    variants of the quarter ending 2011-11-05, and GS tags Q1-2013 as both `-> 03-30` and
    `-> 03-31`. Keyed on the exact pair all of them survive and the fiscal-quarter label
    then lands on two rows at once: measured, 61 such collisions across 6 tickers, and an
    exact-end key still left GS's 8. `_SAME_PERIOD_DAYS` is an order of magnitude below the
    smallest real gap between two quarter ends on this roster (KR's 4-4-5 calendar, 82
    days), so it cannot merge two genuine quarters. Shapes are deduped separately, so a YTD
    and a quarter that share an end are never confused.
    """
    if frame.empty:
        return frame
    ordered = frame.sort_values(["period_end", "filing_date", "period_days"])
    ends = pd.to_datetime(ordered["period_end"])
    bucket = (ends.diff().dt.days.fillna(0) > _SAME_PERIOD_DAYS).cumsum()
    return ordered[~bucket.duplicated(keep="last")]


def _shape(frame: pd.DataFrame, shape: str) -> pd.DataFrame:
    return _latest_per_window(frame[frame["duration_type"] == shape])


#: How close a nine-month cumulative's end must sit to a fourth quarter's start for the two
#: to be the same fiscal year's contiguous pieces. Days rather than an exact match because a
#: 52/53-week filer's Q4 can begin a day or two off the cumulative's last day.
_CONTIGUOUS_DAYS = 4


def _is_ambiguous_duration(q, ytd9: pd.DataFrame) -> bool:
    """D1b: is this `quarterly`-shaped fact really the whole fiscal year?

    Called only where D1 found NO annual fact to compare against, so this is the last
    non-circular evidence available: the filer's own nine-month cumulative, which is an
    as-filed fact and not a derived quarter.

    True when a contiguous `ytd9` exists, is materially non-zero, runs in the SAME DIRECTION,
    and is smaller in magnitude than the "quarter" -- a fourth quarter exceeding the nine
    months before it would have to be more than three quarters of the year, which is not a
    shape a real fourth quarter takes.

    **The same-sign condition is load-bearing and was learned the expensive way.** Without it
    the rule fires on a genuine loss quarter: LLY's Q4 2017 is **-$1,656.9M** (the Tax Cuts
    and Jobs Act charge) against a nine-month **+$1,452.8M**, so a magnitude-only test deleted
    a correct quarter -- and the four quarters foot to LLY's real FY2017 net loss of $204.1M,
    which is how the annual-footing report caught it. Note the catalogue cannot help here:
    `netIncome` AND `totalRevenue` both declare `sign: any`, so gating on the field's sign
    would have disabled the ORCL case this guard exists for.
    """
    if ytd9.empty or pd.isna(q.period_start) or pd.isna(q.value):
        return False
    gap = (q.period_start - ytd9["period_end"]).dt.days
    contiguous = ytd9[(gap >= 0) & (gap <= _CONTIGUOUS_DAYS)]
    if contiguous.empty:
        return False
    quarter = float(q.value)
    # Compare against the SAME-DIRECTION cumulative. A cumulative of the opposite sign is
    # evidence the year turned, not evidence the window is mislabelled.
    same_sign = contiguous[contiguous["value"] * quarter > 0]
    if same_sign.empty:
        return False
    nine = same_sign["value"].abs().max()
    return bool(nine > 0.01 * abs(quarter) and abs(quarter) > nine)


def _drop_annual_masquerading_as_quarter(frame: pd.DataFrame,
                                         refusals: list[dict] | None = None,
                                         ) -> pd.DataFrame:
    """Drop duration facts whose window says *quarter* but whose value is the FULL YEAR.

    Some filers tag the annual figure against a fourth-quarter context, so the row arrives
    with a ~92-day window and the twelve-month number in it. The day-count bands cannot see
    this -- the window really is 92 days -- and because an as-reported quarter outranks a
    derived one, the mislabelled row then WINS over `FY - YTD9` and inflates Q4 roughly
    fourfold. Measured out-of-sample: ORCL `totalRevenue`, 5 rows across fiscal 2019, 2021
    and 2022, e.g. fiscal 2022 Q4 read $42,440M against a true $11,840M. In-sample: 0 rows,
    so this is not a blanket rescale of anything that already worked.

    The test has to be tight, because a fourth quarter legitimately EQUALS its fiscal year
    whenever the first nine months were zero -- a capex programme that only spends in Q4 is
    unusual but not wrong. So all three conditions must hold:

      1. an ANNUAL-shaped fact ends within `_SAME_PERIOD_DAYS` of the quarter's end (a real
         Q4 shares its year's end date, which is exactly why the two get confused);
      2. the two values agree to within 0.1% -- a *coincidence* at that precision on a
         nine-figure number is not a thing that happens; and
      3. an interim cumulative fact inside the same year is materially non-zero (>1% of the
         annual), which PROVES the year accumulated before its final quarter and therefore
         that the fourth quarter cannot be the whole of it.

    Condition 3 is what makes this safe rather than merely plausible: without it the rule
    would silently delete the one shape it is allowed to keep.

    **D1b (4c.8): the same defect where no annual fact exists at all.** Conditions 1 and 2
    both need the filer's own annual figure, so they cannot run when the mislabelled fact is
    the ONLY place the year appears. That is not hypothetical: ORCL tags its full-year
    `us-gaap:Revenues` into a 91-day Q4 context and publishes no annual-window `Revenues`
    for those years, in **9 rows across fiscal 2018-2022**. The plan proposed gating D1b on
    the WINDOW LENGTH -- "a ~365-day fact in a quarterly slot" -- but measurement refutes
    the premise: the window really is 91 days, so `duration_type` is `quarterly` and there
    is no length anomaly to see.

    What IS available, and is not circular, is the filer's own NINE-MONTH CUMULATIVE. Both
    facts are as-filed and neither is a derived quarter, so the two conditions are:

      1b. no annual fact ends within `_SAME_PERIOD_DAYS` of the quarter's end -- i.e. D1
          declined to judge this row; and
      2b. the contiguous `ytd9` fact for the same year exists, is materially non-zero
          (>1% of the quarter, the same guard as condition 3), and is SMALLER than the
          "quarter". A fourth quarter that exceeds the whole nine months preceding it is a
          mislabelled year: it would need to be more than three quarters of the annual.

    Refused rather than reclassified, per decision 24. Reclassifying would be inference, and
    the only test for "is this really the year?" uses the quarters being derived. Refusals
    are appended to `refusals` with `AMBIGUOUS_DURATION` so Phase 5 can reason-code them --
    `fundamentals_reason_codes` stays the single source of truth for why a value is absent.
    """
    quarters = frame[frame["duration_type"] == QUARTERLY]
    annual = frame[frame["duration_type"] == ANNUAL]
    interim = frame[frame["duration_type"].isin((YTD6, YTD9))]
    ytd9 = frame[frame["duration_type"] == YTD9]
    # `annual` may legitimately be EMPTY and the function must still run: D1b exists exactly
    # for the filer that never tags an annual window, so short-circuiting on it -- as this
    # did until 4c.8 -- disables the new branch on the only frames it is meant to judge.
    if quarters.empty or (annual.empty and ytd9.empty):
        return frame
    drop: list = []
    for q in quarters.itertuples():
        near = annual[(annual["period_end"] - q.period_end).abs()
                      <= pd.Timedelta(days=_SAME_PERIOD_DAYS)]
        if near.empty:
            if _is_ambiguous_duration(q, ytd9):
                drop.append(q.Index)
                if refusals is not None:
                    refusals.append({
                        "period_start": q.period_start, "period_end": q.period_end,
                        "period_days": q.period_days, "value": float(q.value),
                        "known_from": q.filing_date, "dc_code": AMBIGUOUS_DURATION,
                        "source_concept": getattr(q, "source_concept", None)})
            continue
        scale = near["value"].abs().max()
        if scale < 1 or (near["value"] - q.value).abs().min() > 0.001 * scale:
            continue
        # Scoped to the ANNUAL window, not to the quarter's: a nine-month cumulative ends
        # exactly where the fourth quarter begins, so anchoring on `q.period_start` would
        # exclude the one fact that proves the year accumulated.
        year_start = near["period_start"].min()
        accumulated = interim[(interim["period_end"] > year_start)
                              & (interim["period_end"] <= q.period_end)]
        if (accumulated["value"].abs() > 0.01 * scale).any():
            drop.append(q.Index)
    return frame.drop(index=drop) if drop else frame


def _same_start_before(candidates: pd.DataFrame, start, end) -> pd.Series | None:
    """The candidate sharing this window's START and ending strictly earlier -- the only
    fact a cumulative total may be differenced against.

    Sharing the start is what makes the subtraction arithmetically valid: `YTD9 - YTD6` is
    a quarter only if both run from the same first day of the fiscal year. The legacy
    engine matched on "the nearest earlier period end", which silently differenced across a
    fiscal-year boundary whenever a quarter was missing.
    """
    if candidates.empty or pd.isna(start):
        return None
    hits = candidates[(candidates["period_start"] == start)
                      & (candidates["period_end"] < end)]
    if hits.empty:
        return None
    return hits.sort_values("period_end").iloc[-1]


# ---------------------------------------------------------------------------- guards ---

def _scale_agrees(total: float, total_days: float, part: float, part_days: float,
                  guards: PeriodGuards, two_sided: bool) -> bool:
    """Are the two legs of a subtraction plausibly the SAME line, judged purely on scale?

    Compared as **per-day rates**, which is the only dimensionally honest comparison: the
    legs are a twelve-month figure and a nine-month one, so their raw magnitudes differ by
    construction and any count-based annualisation (`x 4 / len(parts)`) is wrong the moment
    a leg is itself cumulative.

    Using the parts' summed MAGNITUDES rather than the magnitude of their sum is deliberate
    and was a real bug in the legacy engine: `abs(q1 + q2 + q3)` collapses toward zero
    whenever a year contains offsetting quarters, so every annual figure then looks wildly
    out of scale. That rejected four confirmed, perfectly derivable quarters whose only
    problem was one loss quarter in the year (Cboe FY2022 read as 2.34x, Dow FY2020 as 88x,
    PG&E FY2021 at 0.12x, EA FY2012 at 0.17x).

    **`two_sided` is the difference between a flow and a stock.** For an additive flow only
    an upper bound can work -- a year of offsetting quarters legitimately foots to a small
    annual figure, and the case a lower bound would catch (an annual fact on a different,
    smaller concept, e.g. JPM's `Revenues` against `RevenuesNetOfInterestExpense` quarters)
    already yields a negative quarter that `_is_coherent` rejects on sign. For a
    **non-additive share count the bound is two-sided and exact**, because a share count
    cannot halve or double in a quarter without a corporate action, and a corporate action
    is precisely what this has to catch: see `quarterize`'s stock-split note.
    """
    if total_days <= 0 or part_days <= 0 or part == 0:
        return False
    ratio = abs(float(total) / total_days) / abs(float(part) / part_days)
    # A share count gets its own, much tighter bound: `concept_switch_scale_max` is 2.0 and
    # a 2-for-1 split lands at 1.996-2.003, so the commonest split ratio in existence sat
    # exactly on the threshold and half of them passed.
    bound = (guards.share_basis_max_ratio if two_sided
             else guards.concept_switch_scale_max)
    if ratio > bound:
        return False
    return not two_sided or ratio >= 1 / bound


def _is_coherent(derived: float, siblings: list[float], spec: FieldSpec,
                 guards: PeriodGuards) -> bool:
    """Is a derived quarter broadly consistent with the quarters already observed?

    Deliberately permissive about magnitude alone: a real business can have a legitimately
    much larger or smaller quarter, so a value sharing its sign with ANY sibling is
    accepted regardless of size. The legacy engine required the sign to match EVERY
    sibling, and that single `all(...)` nulled **745 of the 950 missing Q4s** measured in a
    10-ticker audit -- one loss-making quarter anywhere in the year destroyed that year's
    Q4 for every income-statement field at once (GLW FY2016: -368 / +2,207 / +284 against
    an FY of 3,695 has a perfectly correct Q4 of +1,572, thrown away).

    The sharpest test runs first and needs no threshold at all: for a field the catalogue
    declares `non_negative`, a negative derived value is arithmetically impossible, so it
    is proof the two inputs measured different things. That one rule subsumes both
    confirmed mismatched-concept failures (JPM's -$63B revenue quarter, CBRE FY2016's
    -$6.4B cost of revenue) and catches cases magnitude missed entirely (KeyCorp's D&A,
    -$152M in each of eight consecutive years).
    """
    if spec.sign == "non_negative" and derived < 0:
        return False
    if not siblings:
        return True
    if any((derived >= 0) == (s >= 0) for s in siblings):
        return True
    # The one case where a large opposite-sign quarter is arithmetically FORCED: the year's
    # own total came out the opposite sign from the periods before it, which can only
    # happen if this one outweighed all of them. Confirmed real: Citigroup FY2017 (nine
    # months +$12.1B, FY -$6.8B -> a -$18.9B Q4 at 4.6x the largest quarter) and Corning
    # FY2017 (3.2x), both the December-2017 Tax Cuts and Jobs Act deferred-tax writedown --
    # a systematic fiscal-2017 hole across the index, not two outliers.
    if ((derived + sum(siblings)) >= 0) != (sum(siblings) >= 0):
        return True
    largest = max(abs(s) for s in siblings)
    return largest == 0 or abs(derived) <= largest * guards.max_opposite_sign_ratio


# ----------------------------------------------------------------------- the ladder ---

def _derived(total, subtrahend, basis: str, spec: FieldSpec, siblings: list[float],
             guards: PeriodGuards) -> dict | None:
    """One subtraction, guarded. Returns None where the guards refuse it, because a NULL
    the validator can explain is worth more than a plausible wrong number."""
    value = float(total["value"]) - float(subtrahend["value"])
    switched = str(total["source_concept"]) != str(subtrahend["source_concept"])
    # The scale test runs on a concept switch (the legs may be two different lines) and
    # ALWAYS for a non-additive share count (the legs may be two different SPLIT BASES --
    # same concept, same line, incompatible units).
    if switched or not spec.is_additive:
        if not _scale_agrees(total["value"], total["period_days"], subtrahend["value"],
                             subtrahend["period_days"], guards,
                             two_sided=not spec.is_additive):
            return None
    if not _is_coherent(value, siblings, spec, guards):
        return None
    start = pd.Timestamp(subtrahend["period_end"]) + pd.Timedelta(days=1)
    end = pd.Timestamp(total["period_end"])
    return {
        "period_start": start, "period_end": end, "period_days": (end - start).days,
        "value": value, "basis": basis,
        "known_from": max(pd.Timestamp(total["filing_date"]),
                          pd.Timestamp(subtrahend["filing_date"])),
        "source_concept": total["source_concept"],
        "concept_switch": switched,
    }


def quarterize(facts: pd.DataFrame, spec: FieldSpec,
               guards: PeriodGuards | None = None,
               year_ends: list[pd.Timestamp] | None = None,
               refusals: list[dict] | None = None) -> pd.DataFrame:
    """One (ticker, field)'s duration facts -> discrete quarters, with provenance.

    Reported discrete quarters are kept as they are. Everything else climbs the ladder:
    `Q2 = YTD6 - Q1`, `Q3 = YTD9 - YTD6`, then `Q4 = FY - YTD9` and only failing that
    `Q4 = FY - (Q1+Q2+Q3)`.

    **A non-additive field is differenced in SHARE-DAYS, not refused.** A weighted-average
    share count is not additive across quarters -- four summed quarterly averages are four
    times the share count -- but `average x days` IS, because that product is the number of
    share-days outstanding and share-days simply accumulate. So the ladder runs in that
    space and converts back, which makes `Q4 = (FY.avg*FY.days - YTD9.avg*YTD9.days) /
    Q4.days` exact rather than approximate.

    This is a deliberate departure from the plan's *"refuse share counts"*, and from
    edgartools' `_is_additive_concept`, which refuses them too. Measured, refusing them
    leaves `dilutedShares_ttm` computable at **129 of 1,532 points (8%)** -- because filers
    never publish a discrete Q4 average, so a four-quarter run essentially never closes --
    and decision #9 defines `epsDiluted` as `netIncome_ttm / dilutedShares_ttm`. Refusing
    the derivation therefore does not protect Tier-2 EPS, it deletes it. What decision #9
    actually forbids is summing four quarterly **EPS** figures: EPS is a ratio of two flows
    and its denominator moves, which is a different thing from a time-average of a stock.
    Ratios and per-share amounts never reach here at all -- `build_periods` only walks
    fields whose `kind` is `duration`, and both are `ratio`/`derived`.

    `refusals`, when a list is passed, collects the D1b `ambiguous_duration` rows this call
    declined -- an out-parameter rather than a fourth return value because it is empty on
    essentially every (ticker, field) and Phase 5 is the only consumer.
    """
    guards = guards or load_guards()
    if facts.empty:
        return pd.DataFrame(columns=list(_QUARTER_COLUMNS))
    frame = facts[facts["value"].notna() & facts["period_start"].notna()
                  & facts["period_end"].notna()].copy()
    frame["period_start"] = pd.to_datetime(frame["period_start"])
    frame["period_end"] = pd.to_datetime(frame["period_end"])
    frame["filing_date"] = pd.to_datetime(frame["filing_date"])
    # Before the share-day transform, so the value comparison is on as-filed numbers: a
    # 92-day window and a 365-day one are multiplied by different factors and a mislabelled
    # annual would stop matching its own annual fact.
    frame = _drop_annual_masquerading_as_quarter(frame, refusals)
    if not spec.is_additive:
        frame["value"] = frame["value"] * _inclusive_days(frame["period_days"])

    quarters = _shape(frame, QUARTERLY)
    rows: list[dict] = [{
        "period_start": r.period_start, "period_end": r.period_end,
        "period_days": r.period_days, "value": float(r.value), "basis": AS_REPORTED,
        "known_from": r.filing_date, "source_concept": r.source_concept,
        "concept_switch": False,
    } for r in quarters.itertuples()]                # still in share-days if weighted

    y6, y9, annual = (_shape(frame, s) for s in (YTD6, YTD9, ANNUAL))
    rows.extend(_ladder(quarters, y6, y9, annual, spec, guards))

    out = pd.DataFrame(rows, columns=[c for c in _QUARTER_COLUMNS
                                      if c not in ("ticker", "field", "fiscal_year",
                                                   "fiscal_quarter")])
    if out.empty:
        return pd.DataFrame(columns=list(_QUARTER_COLUMNS))
    if not spec.is_additive:
        out["value"] = out["value"] / _inclusive_days(out["period_days"])
    # An as-reported quarter always beats a derived one for the same window: it is the
    # filer's own number rather than our arithmetic on two of them.
    out["_rank"] = (out["basis"] != AS_REPORTED).astype(int)
    out = (out.sort_values(["period_end", "_rank", "known_from"])
              .drop_duplicates(subset=["period_end"], keep="first")
              .drop(columns="_rank"))
    out.insert(0, "field", spec.name)
    out.insert(0, "ticker", facts["ticker"].iloc[0])
    # The calendar is the TICKER's, never this field's own annual facts. A field with one
    # annual fact in fifteen years has a one-bucket calendar, and every quarter it ever
    # reported then lands in that single fiscal year: measured, AMT's `interestExpense` put
    # 2015, 2016 and 2017 all into FY2017 and produced four Q1s. 69 such collisions across
    # 11 tickers, and every one of them disappears once the calendar is shared.
    return label_fiscal_periods(
        out, fiscal_year_ends(frame) if year_ends is None else year_ends)


def _ladder(quarters: pd.DataFrame, y6: pd.DataFrame, y9: pd.DataFrame,
            annual: pd.DataFrame, spec: FieldSpec,
            guards: PeriodGuards) -> list[dict]:
    """The three decumulation rungs plus the two Q4 routes, in that order."""
    out: list[dict] = []
    for cumulative, earlier, basis in ((y6, quarters, Q2_FROM_YTD6),
                                       (y9, y6, Q3_FROM_YTD9)):
        for row in cumulative.itertuples():
            prior = _same_start_before(earlier, row.period_start, row.period_end)
            if prior is None:
                continue
            derived = _derived(row._asdict(), prior, basis, spec,
                               [float(prior["value"])], guards)
            if derived:
                out.append(derived)

    for fy in annual.itertuples():
        fy_row = fy._asdict()
        inside = quarters[(quarters["period_start"] >= fy.period_start)
                          & (quarters["period_end"] <= fy.period_end)]
        # A discrete quarter already ending on the fiscal year-end IS Q4 as reported --
        # nothing to derive, and deriving anyway would duplicate the window.
        if (inside["period_end"] == fy.period_end).any():
            continue
        siblings = [float(v) for v in inside["value"]]

        ytd9 = _same_start_before(y9, fy.period_start, fy.period_end)
        if ytd9 is not None:
            derived = _derived(fy_row, ytd9, FY_MINUS_YTD9, spec, siblings, guards)
            if derived:
                out.append(derived)
                continue

        # Fallback. Exactly the three quarters that precede Q4 -- anything else (a gap, an
        # overlap, a stub) is ambiguous, and emitting a value from an ambiguous input set
        # is how the legacy engine made its Q4 footing check tautological.
        if len(inside) != TTM_QUARTERS - 1:
            continue
        total = float(inside["value"].sum())
        last = inside.sort_values("period_end").iloc[-1]
        derived = _derived(fy_row, {"value": total, "period_end": last["period_end"],
                                    "period_days": float(inside["period_days"].sum()),
                                    "filing_date": inside["filing_date"].max(),
                                    "source_concept": last["source_concept"]},
                           FY_MINUS_QUARTERS, spec, siblings, guards)
        if derived:
            out.append(derived)
    return out


# ------------------------------------------------------------------ fiscal calendar ---

def fiscal_year_ends(facts: pd.DataFrame) -> list[pd.Timestamp]:
    """The issuer's own fiscal year-end dates, taken from the ANNUAL-shaped facts' window
    ends rather than from a month-of-year rule.

    Keying off the filer's own period ends is what makes a 52/53-week issuer work: its year
    end walks by a day or six and lands in a different calendar month every few years, so
    any fixed-month rule mislabels it. Annual-SHAPED, not `FY`-labelled: Skyworks FY2020
    tags both a 370-day and a **97-day** fact as `fp='FY'`, and only the day count tells
    them apart.
    """
    annual = facts[(facts["duration_type"] == ANNUAL) & facts["value"].notna()]
    ends = sorted(pd.Timestamp(e) for e in
                  pd.to_datetime(annual["period_end"]).dropna().unique())
    if not ends:
        return []
    # Extrapolate ONE year past the last 10-K. The quarters a model actually trades on are
    # the ones filed since the most recent annual report, and without this they fall off
    # the end of the calendar unlabelled -- measured, that silently dropped AAPL's three
    # most recent quarters. One year only: two would be inventing a fiscal calendar rather
    # than extending the filer's own.
    # Fill a MISSING year before extrapolating. A filer files every year, so a gap in the
    # annual facts is a gap in what survived entity scoping, not in the calendar -- and an
    # unfilled gap dumps two or three years of quarters into one bucket: measured, MAA's
    # 2013 quarters were all labelled FY2017 Q1.
    filled = [ends[0]]
    for end in ends[1:]:
        previous = filled[-1]
        years = max(round((end - previous).days / 364), 1)
        step = (end - previous) / years
        filled.extend(previous + step * n for n in range(1, years))
        filled.append(end)
    last = filled[-1]
    span = (last - filled[-2]).days if len(filled) > 1 else 364
    return [*filled, last + pd.Timedelta(days=span)]


def label_fiscal_periods(quarters: pd.DataFrame,
                         year_ends: list[pd.Timestamp]) -> pd.DataFrame:
    """Attach `fiscal_year` and `fiscal_quarter`, positioned against the fiscal year's own
    START and its own LENGTH.

    Not a fixed day-count divisor -- calendar quarters run 90/91/92 days and a 52/53-week
    issuer's year is 364 or 371, so `days // 91` misclassifies at the boundary. Dividing
    the quarter's offset into the year by *that year's own* quarter length and rounding is
    exact for any regular calendar, 53-week years included.

    Not a chronological rank either, which was the first attempt and is wrong for the year
    the filer is still inside: ranking AAPL's three post-10-K quarters from the end labelled
    them Q2/Q3/Q4 instead of Q1/Q2/Q3, and those are precisely the quarters a model trades
    on. Ranking from the start instead breaks the mirror case, an early-history year seen
    only from Q3 onward. Anchoring on the calendar handles both without a special case.

    `fiscal_year` is the calendar year of the year-end the quarter falls in, matching SEC's
    own `fy` convention.
    """
    out = quarters.copy()
    out["fiscal_year"] = pd.NA
    out["fiscal_quarter"] = pd.NA
    if not year_ends or out.empty:
        return out[list(_QUARTER_COLUMNS)]
    ends = sorted(pd.Timestamp(e) for e in year_ends)
    starts = [ends[0] - pd.Timedelta(days=364), *[e + pd.Timedelta(days=1) for e in ends[:-1]]]
    bounds = pd.Series(ends)
    # side='left' puts a quarter ending exactly ON a year end into that year, where Q4 lives.
    slot = bounds.searchsorted(out["period_end"].values, side="left")
    for position, index in zip(slot, out.index):
        if position >= len(ends):
            continue
        year_start, year_end = starts[position], ends[position]
        quarter_length = max((year_end - year_start).days + 1, 1) / TTM_QUARTERS
        offset = (pd.Timestamp(out.at[index, "period_start"]) - year_start).days
        out.at[index, "fiscal_year"] = year_end.year
        out.at[index, "fiscal_quarter"] = int(
            min(max(round(offset / quarter_length) + 1, 1), TTM_QUARTERS))
    return out[list(_QUARTER_COLUMNS)]


# ----------------------------------------------------------------- trailing twelve ---

def trailing_twelve(quarters: pd.DataFrame, spec: FieldSpec,
                    annual: pd.DataFrame | None = None,
                    guards: PeriodGuards | None = None) -> pd.DataFrame:
    """A trailing-twelve-month value at every quarter end it can be built from FOUR
    DISCRETE QUARTERS -- and nowhere else.

    This is the staircase fix. The legacy fallback carried the last annual figure forward
    up to four quarters, which froze **6.2% of consecutive `totalRevenue` pairs** and made
    `revenueGrowth` exactly 0 for three quarters in four (APA 100% frozen, XOM 36%). A
    carried-forward annual is not a trailing twelve months; it is last year's number
    wearing this quarter's date. Where the four quarters do not exist, the value is NULL
    with `insufficient_quarters`, and coverage drops on purpose.

    A **non-additive** field (a weighted-average share count) is averaged, not summed --
    four summed quarterly averages are four times the share count. Where the window ends on
    a fiscal year end and the filer published the annual figure, that as-reported
    twelve-month average is used instead, because it is the filer's own exact number rather
    than our mean of four means.
    """
    empty = pd.DataFrame(columns=["ticker", "field", "period_end", "value", "basis",
                                  "known_from", "n_quarters", "dc_code"])
    if quarters.empty:
        return empty
    guards = guards or load_guards()
    ordered = quarters.sort_values("period_end").reset_index(drop=True)
    reported_annual = _annual_by_end(annual)
    rows = []
    for i in range(len(ordered)):
        window = ordered.iloc[max(0, i - TTM_QUARTERS + 1): i + 1]
        end = window["period_end"].iloc[-1]
        base = {"ticker": ordered["ticker"].iloc[0], "field": spec.name, "period_end": end}
        if not spec.is_additive and end in reported_annual:
            fact = reported_annual[end]
            rows.append({**base, "value": float(fact["value"]),
                         "basis": TTM_AS_REPORTED_ANNUAL,
                         "known_from": fact["filing_date"], "n_quarters": 0,
                         "dc_code": None})
            continue
        if len(window) < TTM_QUARTERS or not _window_is_contiguous(window):
            rows.append({**base, "value": None, "basis": None, "known_from": None,
                         "n_quarters": len(window), "dc_code": INSUFFICIENT_QUARTERS})
            continue
        if not spec.is_additive and not _one_share_basis(window, guards):
            rows.append({**base, "value": None, "basis": None, "known_from": None,
                         "n_quarters": len(window), "dc_code": SPLIT_BASIS_MISMATCH})
            continue
        # Share-days again for a non-additive field: a twelve-month weighted average is the
        # share-day total over the days, not the mean of four means (which is only equal
        # when all four quarters are the same length -- a 53-week year's Q4 is not).
        days = _inclusive_days(window["period_days"])
        value = (window["value"].sum() if spec.is_additive
                 else (window["value"] * days).sum() / days.sum())
        rows.append({**base, "value": float(value),
                     "basis": TTM_FOUR_QUARTERS if spec.is_additive
                     else TTM_FOUR_QUARTER_MEAN,
                     "known_from": window["known_from"].max(),
                     "n_quarters": TTM_QUARTERS, "dc_code": None})
    out = pd.DataFrame(rows, columns=list(empty.columns))
    # One null representation, not two: a python `None` and a numpy NaN in the same object
    # column survive a parquet round-trip as two distinct values and make every downstream
    # `== None` test quietly wrong.
    return out.astype({"basis": "string", "dc_code": "string"})


def _annual_by_end(annual: pd.DataFrame | None) -> dict:
    if annual is None or annual.empty:
        return {}
    latest = _latest_per_window(annual[annual["duration_type"] == ANNUAL])
    return {pd.Timestamp(r.period_end): {"value": r.value, "filing_date": r.filing_date}
            for r in latest.itertuples()}


def _one_share_basis(window: pd.DataFrame, guards: PeriodGuards) -> bool:
    """Are the four quarterly share counts on ONE split basis?

    A stock split retroactively rescales every prior share count, and each quarter in a
    trailing window comes from a different filing -- so a window straddling a split mixes
    a pre- and a post-split basis and averages two incompatible units. Measured on the
    26-ticker roster: **45 `dilutedShares` windows across 8 tickers** (AAPL 7:1 2014 and
    4:1 2020, NEE 4:1 2020, AFL 2:1 2018, EOG 2:1 2014, KR 2:1 2015, plus VRT's SPAC merger
    and BRK-B's A/B classes, whose spread is 246,000x).

    A share count cannot move more than ~15% in a year organically, so the bound is not
    close to anything real -- and the value it protects is `epsDiluted`, where a
    split-straddling denominator is wrong by an exact integer factor and looks entirely
    plausible. Refused with a reason code rather than repaired: the repair would be to pick
    one basis, and there is no way to know from the facts alone which one the consumer
    wants.
    """
    values = window["value"].abs()
    smallest = values.min()
    return smallest > 0 and (values.max() / smallest) <= guards.share_basis_max_ratio


def _window_is_contiguous(window: pd.DataFrame) -> bool:
    """Four quarters make a year only if they abut and span one. Checked rather than
    assumed, because a ticker with a filing gap has four quarters spanning two years and
    summing them produces a number that looks entirely reasonable."""
    starts = list(window["period_start"])
    ends = list(window["period_end"])
    for previous_end, next_start in zip(ends, starts[1:]):
        if abs((pd.Timestamp(next_start) - pd.Timestamp(previous_end)).days) > 1:
            return False
    span = (pd.Timestamp(ends[-1]) - pd.Timestamp(starts[0])).days
    return TTM_MIN_DAYS <= span <= TTM_MAX_DAYS


# --------------------------------------------------------------------------- instants ---

#: An instant tagged with the fiscal YEAR is the year-END snapshot, which occupies the same
#: grid slot as the fourth quarter. Every other label already names its own quarter.
_YEAR_END_LABEL = {"FY": "Q4", "YTD12": "Q4"}


def instant_stock(facts: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time facts on the quarterly grid: balance-sheet levels, the cover-page
    share count, and the 10-K's headcount.

    There is no ladder here -- an instant is already discrete -- so the work is two rules:

      * **A fiscal-year label on an instant means the year END.** A balance sheet tagged
        `FY` is the 31 December snapshot, not a twelve-month measure, so it belongs in the
        Q4 slot. A DURATION field legitimately has both an `FY` and a `Q4` flavour and must
        never be relabelled this way, which is why the discriminator is the absence of a
        `period_start` and not the label itself.
      * **One row per (field, date), latest filing wins** -- the same rule the duration path
        uses and for the same reason: a level is re-tagged as a comparative in every later
        filing, and inside a point-in-time replay the latest visible one is the right one.
    """
    if facts.empty:
        return facts
    if "period_start" in facts.columns:
        out = facts[facts["period_start"].isna()].copy()
    else:
        out = facts[facts["duration_type"] == INSTANT].copy()
    if out.empty:
        return out
    if "fiscal_period" in out.columns:
        out["fiscal_period"] = out["fiscal_period"].replace(_YEAR_END_LABEL)
    keys = [c for c in ("ticker", "field", "period_end") if c in out.columns]
    if keys and "filing_date" in out.columns:
        out = (out.sort_values([*keys, "filing_date"])
                  .drop_duplicates(subset=keys, keep="last"))
    return out


# ----------------------------------------------------------------------- entry point ---

def build_periods(facts: pd.DataFrame, catalogue,
                  guards: PeriodGuards | None = None,
                  refusals: list[dict] | None = None,
                  ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Every field's discrete quarters, TTM values and instants for one ticker's facts.

    Returns `(quarters, ttm, instants)` -- all three, rather than leaving instants to a
    second call, because a history built from the first two alone is silently missing every
    balance-sheet level and would still look complete.

    All three are long frames; the history build pivots them onto the publication-event
    grain. Nothing here is written to a table: a derived quarter is not a fact the filer
    published, and `fundamentals_facts` stays a faithful record of what was.

    `refusals`, when a list is passed, collects every D1b `ambiguous_duration` row, tagged
    with its field, for Phase 5 to write to `fundamentals_reason_codes`. Left None it is
    simply not collected -- a caller that does not reason-code must not be forced to.
    """
    if facts.empty:
        return (pd.DataFrame(columns=list(_QUARTER_COLUMNS)),
                trailing_twelve(pd.DataFrame(), catalogue.field(
                    catalogue.extracted_fields[0])), facts)
    durations = facts[~facts["duration_type"].isin([INSTANT, OTHER_SHAPE])]
    # One calendar for the whole ticker, built from every annual-shaped fact any field
    # reported -- see `quarterize`.
    year_ends = fiscal_year_ends(durations)
    all_quarters, all_ttm = [], []
    for name, group in durations.groupby("field", sort=True):
        spec = catalogue.field(name)
        if spec.kind != "duration":
            continue
        field_refusals: list[dict] = []
        quarters = quarterize(group, spec, guards, year_ends, field_refusals)
        if refusals is not None:
            refusals.extend({**r, "field": name} for r in field_refusals)
        if quarters.empty:
            continue
        all_quarters.append(quarters)
        all_ttm.append(trailing_twelve(quarters, spec, annual=group, guards=guards))
    quarters = (pd.concat(all_quarters, ignore_index=True) if all_quarters
                else pd.DataFrame(columns=list(_QUARTER_COLUMNS)))
    ttm = pd.concat(all_ttm, ignore_index=True) if all_ttm else pd.DataFrame()
    return quarters, ttm, instant_stock(facts)
