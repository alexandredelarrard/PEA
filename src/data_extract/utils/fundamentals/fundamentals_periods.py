"""
fundamentals_periods.py
------------------------
Fiscal-period resolution + Q1-Q4 duration decumulation for the edgartools-based
fundamentals pipeline (`fetch_fundamentals_edgar.py` / `fundamentals_derive.py`).

Generalizes `fetch_fundamentals.py`'s proven `_quarterly_flow`/`_annual_flow`/
`_instant_stock` logic (group facts sharing a period, diff consecutive
cumulative values into discrete quarters, derive Q4 = FY - Q1 - Q2 - Q3) but
re-keys it by the issuer's ACTUAL fiscal year/period (edgartools' native
`fiscal_year`/`fiscal_period`, sourced from SEC's own `fy`/`fp` XBRL fields)
instead of raw calendar-date grouping. This fixes the confirmed WIP-file bug
(`fiscal_year = int(pd.to_datetime(period_of_report).year)` — calendar year of
the period END, broadcast filing-wide — breaks any non-calendar-fiscal-year
issuer, since Q1/Q2 of a June-FYE company fall in one calendar year and Q3/FY
in the next).
"""
from __future__ import annotations

import math

import pandas as pd

from src.data_extract.utils.fundamentals.fundamentals_tags import (
    ANNUAL_MAX_DAYS, ANNUAL_MIN_DAYS, FISCAL_YEAR_EXTRAPOLATION_GRACE_DAYS,
    FISCAL_YEAR_MEAN_DAYS, IMPLIED_QUARTER_MAX_DAYS, IMPLIED_QUARTER_MIN_DAYS,
    MAX_OPPOSITE_SIGN_Q4_RATIO, MIN_FISCAL_YEAR_LABEL_VOTES, MIN_PARTIAL_FY_FIELDS,
    NON_NEGATIVE_FLOW_FIELDS, PARTIAL_FY_TOLERANCE, Q4_TAG_MISMATCH_FY_MAX,
)

# Ordinal position of each fiscal-period label within its fiscal year. SEC's own
# `fp` labels a period by its END regardless of whether the underlying duration
# fact is discrete (~90d) or YTD-cumulative (~181d/~273d) -- Q2's YTD 6-month
# fact and a hypothetical discrete-Q2 fact are BOTH natively labeled fp='Q2'.
FISCAL_PERIOD_ORDER: dict[str, int] = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4, "FY": 4}
_QUARTER_LABELS = ("Q1", "Q2", "Q3")

# Concepts filers NEVER tag as a discrete 3-month figure (cash-flow-statement items
# are the confirmed case -- e.g. NetCashProvidedByUsedInOperatingActivities,
# PaymentsForCapitalImprovements) come back from edgartools natively labeled
# fiscal_period='YTD3'/'YTD6'/'YTD9', not 'Q1'/'Q2'/'Q3' -- confirmed empirically
# against real MAA 10-Qs. Without this remap, decumulate_quarterly_flow's
# `isin(_QUARTER_LABELS)` filter silently dropped every such fact: Q2/Q3 (and thus
# the FY-Q1-Q2-Q3 Q4 derivation, which requires all three) never resolved for any
# cash-flow-only field. The discrete-vs-cumulative decision still comes from
# `classify_duration_shape` (duration days), unaffected by this label remap.
_YTD_FISCAL_PERIOD_MAP: dict[str, str] = {"YTD3": "Q1", "YTD6": "Q2", "YTD9": "Q3", "YTD12": "FY"}


def normalize_fiscal_period_label(fiscal_period):
    """Map a native 'YTDn' fiscal_period label to its quarter/FY bucket; pass
    through 'Q1'..'Q4'/'FY' and anything else (e.g. NaN) unchanged."""
    return _YTD_FISCAL_PERIOD_MAP.get(fiscal_period, fiscal_period)


def duration_days(period_start, period_end) -> float:
    """(period_end - period_start) in days, NaN if either is missing."""
    if period_start is None or period_end is None or pd.isna(period_start) or pd.isna(period_end):
        return float("nan")
    return (pd.Timestamp(period_end) - pd.Timestamp(period_start)).days


def classify_duration_shape(days: float) -> str:
    """Classify a duration fact's day-span into 'quarterly' / 'semiannual' /
    'nine_month' / 'annual' / 'other', using this repo's own, already-tested
    day windows (`fundamentals_tags.py`) -- NOT edgartools' narrower
    `classify_duration` (85-95d quarterly), which would incorrectly reject a
    52/53-week issuer's ~112d fiscal Q1 (e.g. KR/COST/TGT). Used only as a
    shape classifier; `resolve_fiscal_period` below decides WHICH (fiscal_year,
    fiscal_period) bucket a fact belongs to."""
    if pd.isna(days):
        return "other"
    if ANNUAL_MIN_DAYS <= days <= ANNUAL_MAX_DAYS:
        return "annual"
    if IMPLIED_QUARTER_MIN_DAYS <= days <= IMPLIED_QUARTER_MAX_DAYS:
        return "quarterly"
    if 170 <= days <= 190:
        return "semiannual"
    if 260 <= days <= 290:
        return "nine_month"
    return "other"


def resolve_fiscal_period(
    fact_fiscal_year: int | None,
    fact_fiscal_period: str | None,
    filing_fiscal_year: int | None = None,
    filing_fiscal_period: str | None = None,
    period_start=None,
    period_end=None,
    anchor_fy: int | None = None,
) -> tuple[int | None, str | None, str]:
    """Resolve one fact's (fiscal_year, fiscal_period), tiered:

      1. native   -- the fact's own edgartools fiscal_year/fiscal_period (SEC's fy/fp).
      2. native   -- the filing's cover-page dei:DocumentFiscalYearFocus/PeriodFocus,
                     used when the specific fact's own context lacks resolvable fy/fp
                     but the filing itself still carries it.
      3. date_arithmetic_fallback -- this repo's day-window shape classifier, stepping
                     from `anchor_fy` (the nearest already-resolved ANNUAL fact's fiscal
                     year) by quarter-offset. Never derives fiscal_year from
                     `period_end.year` directly -- that is the exact WIP-file bug this
                     module fixes.

    Returns (fiscal_year, fiscal_period, fiscal_period_source) where
    fiscal_period_source in {'native', 'date_arithmetic_fallback'}.
    """
    if fact_fiscal_year is not None and fact_fiscal_period:
        return int(fact_fiscal_year), str(fact_fiscal_period), "native"
    if filing_fiscal_year is not None and filing_fiscal_period:
        return int(filing_fiscal_year), str(filing_fiscal_period), "native"

    days = duration_days(period_start, period_end)
    shape = classify_duration_shape(days)
    if shape == "annual":
        fy = anchor_fy if anchor_fy is not None else (
            pd.Timestamp(period_end).year if period_end is not None and not pd.isna(period_end) else None)
        return fy, "FY", "date_arithmetic_fallback"
    if anchor_fy is None:
        # No annual anchor to step from and no native metadata -- cannot resolve the
        # fiscal year reliably; caller should treat this fact as unresolved.
        return None, None, "date_arithmetic_fallback"
    # Quarter-offset from the anchor's fiscal-year start, by calendar month distance --
    # a coarse but date-arithmetic-only (never period_end.year-direct) placement.
    if period_end is None or pd.isna(period_end):
        return anchor_fy, None, "date_arithmetic_fallback"
    return anchor_fy, None, "date_arithmetic_fallback"


# A genuine Q4 never STARTS before ~60% of the way through its fiscal year --
# even a lopsided 52/53-week calendar's last quarter starts around 70-75% in.
# Anything starting earlier than this ratio cannot be a real Q4, regardless of
# how many days a "quarter" spans for this filer.
_MIN_Q4_START_RATIO = 0.6
_REASSIGN_LABELS = ("Q1", "Q2", "Q3")


def _is_mislabeled_q4(start: pd.Timestamp, fy_start: pd.Timestamp, fy_length_days: int) -> bool:
    """True if a fact natively labeled fiscal_period='Q4' cannot genuinely be
    the fiscal year's last quarter, because its OWN start falls in the first
    `_MIN_Q4_START_RATIO` of the fiscal year."""
    if fy_length_days <= 0 or pd.isna(start):
        return False
    return (start - fy_start).days / fy_length_days < _MIN_Q4_START_RATIO


def _relabel_by_chronological_rank(candidates: pd.DataFrame, start_col: str) -> dict:
    """Among facts ALREADY established as mislabeled 'Q4' (see `_is_mislabeled_q4`),
    the correct label is their RELATIVE chronological order (earliest -> Q1,
    next -> Q2, next -> Q3) -- not a fixed day-count divisor, which misclassifies
    at quarter boundaries (calendar quarters are 90/91/92 days, never a uniform
    91). Returns {index: new_fiscal_period}."""
    ordered = candidates.sort_values(start_col)
    return {idx: _REASSIGN_LABELS[rank] for rank, idx in enumerate(ordered.index)
           if rank < len(_REASSIGN_LABELS)}


def _q4_is_coherent(q4_val: float, q1: float, q2: float, q3: float,
                    *, field: str | None = None) -> bool:
    """A derived Q4 must be broadly consistent with Q1/Q2/Q3 -- a signal that
    catches an upstream data problem (mismatched XBRL concepts across periods,
    a corrupted quarter or FY value) rather than a genuine, if unusual, Q4.
    Deliberately permissive about MAGNITUDE alone: a real business can have a
    legitimately much-larger-or-smaller quarter (retail holiday seasonality, a
    one-off gain), so a Q4 sharing its sign with ANY of Q1/Q2/Q3 is accepted
    regardless of size. Only when the sign matches NONE of them does magnitude
    decide, and then the bar is the largest quarter already observed that year
    (`MAX_OPPOSITE_SIGN_Q4_RATIO`).

    Both of those rules were wrong before, and between them silently nulled
    745 of the 950 missing Q4s measured across a 10-ticker audit:

      * the sign test was `all(...)`, i.e. "reject unless Q4 matches EVERY
        quarter's sign" -- not what this docstring described, and it meant ONE
        loss-making quarter anywhere in the year destroyed that year's Q4 for
        every income-statement field at once. Confirmed: GLW fiscal 2016
        (Q1 -$368M, Q2 +$2,207M, Q3 +$284M, FY $3,695M) has a perfectly
        correct derived Q4 of +$1,572M that was thrown away, taking
        netIncome/epsDiluted/pretaxIncome/operatingIncome with it -- the same
        pattern hit CB 2017/2020, AFL 2024, MET 2016/2017, REG 2012/2016/2017/
        2020 and GLW 2016/2017/2018/2020/2024.
      * the opposite-sign magnitude bar was `FUNDAMENTALS_DISCONTINUITY_MIN`
        (0.2), which rejects genuine loss quarters: Citigroup's fiscal 2023 Q4
        was a REAL -$1.84B against +$2.9-4.6B quarters (0.4x the largest), and
        Atmos Energy's fiscal 2012 Q4 a real -$36.9M summer-quarter pretax loss
        (0.21x) -- both as-filed, both nulled.

    The SHARPEST test runs first and needs no threshold at all: for a field that
    cannot be negative (`NON_NEGATIVE_FLOW_FIELDS` -- a top line, a cost line, a
    cash amount paid), a negative derived Q4 is arithmetically impossible, so it
    is proof the FY and the quarters measured different things. That single rule
    subsumes both confirmed mismatched-concept failures this guard was built for
    -- JPM (Q1/Q2/Q3 all +$22-27B of `RevenuesNetOfInterestExpense`, a
    mismatched-tag `Revenues` FY produced a derived Q4 of -$63B) and CBRE fiscal
    2016 (cost of revenue ~$2.2B/quarter against a $78.5M FY, -$6.4B) -- and
    catches cases the magnitude bar missed entirely (KeyCorp's D&A, -$152M in
    each of eight consecutive years).

    Only for a genuinely SIGNED field does magnitude decide, and then the bar is
    `MAX_OPPOSITE_SIGN_Q4_RATIO` times the largest quarter already observed --
    deliberately loose, because a charge-driven loss quarter routinely dwarfs the
    run-rate and nulling it is the more expensive error (see that constant's note
    for the seven confirmed real quarters the old 1.0 bar threw away)."""
    if field in NON_NEGATIVE_FLOW_FIELDS and q4_val < 0:
        return False

    quarters = (q1, q2, q3)
    if any((q4_val >= 0) == (q >= 0) for q in quarters):
        return True

    # The one case where a LARGE opposite-sign Q4 is not just plausible but
    # arithmetically forced: the fiscal year's own total came out the opposite
    # sign from its first nine months, which can only happen if the fourth
    # quarter outweighed all three. Confirmed real quarters this rescues, both
    # otherwise indistinguishable from a data error on magnitude alone:
    # Citigroup fiscal 2017 (nine months +$12.1B, FY -$6.8B -> a -$18.9B Q4,
    # 4.6x the largest quarter) and Corning fiscal 2017 (+$0.9B, FY -$0.5B ->
    # -$1.4B, 3.2x) -- both the December-2017 Tax Cuts and Jobs Act deferred-tax
    # writedown, an event that hit a large share of the index in the SAME
    # quarter, so this is a systematic fiscal-2017 hole, not two outliers.
    # Neither confirmed failure mode shows the flip: JPM's mismatched-concept FY
    # (+$16.0B) and MAA's dimensioned-slice FY (+$14.4M) both keep their nine-
    # month sign and are simply too SMALL, which is what makes them detectable.
    q123_sum = q1 + q2 + q3
    if ((q4_val + q123_sum) >= 0) != (q123_sum >= 0):
        return True

    max_abs = max(abs(q) for q in quarters)
    return max_abs == 0 or abs(q4_val) <= max_abs * MAX_OPPOSITE_SIGN_Q4_RATIO


def _reassign_misordered_native_q4(grp: pd.DataFrame) -> pd.DataFrame:
    """A fact natively labeled fiscal_period='Q4' cannot genuinely be the fiscal
    year's LAST quarter if its OWN period_start falls in the first 60% of the
    fiscal year. Confirmed empirically (MAA, FY2013/FY2017): a genuinely Q1
    cash-flow fact (period_start/end = Jan1-Mar31, SAME period_start as that
    year's FY fact) was natively mislabeled 'Q4' instead of 'Q1'/'YTD3' -- the
    mirror image of the already-handled "native Q4 secretly carries the FY
    value" bug (ORCL): there the LABEL was right and the VALUE was wrong; here
    the CONTEXT DATES are right and the LABEL is wrong. Also confirmed: this is
    not always confined to Q1 -- MAA's `rentalIncome` mislabeled ALL THREE of
    its FY2017 Q1/Q2/Q3 discrete facts as native 'Q4' (three different
    accessions, three genuinely different, correctly-shaped quarterly values,
    all claiming 'Q4'). Every mislabeled candidate's TRUE label is its relative
    chronological rank among the other mislabeled candidates (not a fixed-day
    bucket, which misclassifies at real quarter boundaries -- calendar quarters
    run 90/91/92 days, never a uniform 91). Requires a fiscal-year anchor (the
    'FY' fact's own period_start/period_end); otherwise native labels are left
    untouched (no anchor = no safe fix)."""
    quarterly_shaped = grp["duration_days"].apply(classify_duration_shape) == "quarterly"
    q4_candidates = grp[quarterly_shaped & (grp["fiscal_period"] == "Q4")]
    if q4_candidates.empty:
        return grp
    fy_anchor = grp[grp["fiscal_period"] == "FY"]
    if fy_anchor.empty or pd.isna(fy_anchor.iloc[0]["period_start"]) or pd.isna(fy_anchor.iloc[0]["period_end"]):
        return grp
    fy_start = pd.Timestamp(fy_anchor.iloc[0]["period_start"])
    fy_length = (pd.Timestamp(fy_anchor.iloc[0]["period_end"]) - fy_start).days
    mislabeled = q4_candidates[q4_candidates["period_start"].apply(
        lambda s: _is_mislabeled_q4(pd.Timestamp(s), fy_start, fy_length))]
    if mislabeled.empty:
        return grp
    grp = grp.copy()
    for idx, new_fp in _relabel_by_chronological_rank(mislabeled, "period_start").items():
        grp.loc[idx, "fiscal_period"] = new_fp
    return grp


def _fy_matches_quarterly_run_rate(fy_value: float, quarters: tuple[float, float, float]) -> bool:
    """Whether an FY fact is plausibly the SAME economic line as the Q1-Q3
    facts it will be differenced against, judged purely on scale.

    Filers rename the concept behind a line item mid-history far more often
    than they change what the line MEANS -- confirmed across the audit:
    ATO tagged D&A as `DepreciationAndAmortization` in its 10-Qs and
    `DepreciationDepletionAndAmortization` in its 10-K for NINE consecutive
    years; AFL/DTE/ATO/C alternate between the two `IncomeLossFromContinuing-
    OperationsBeforeIncomeTaxes...` variants; DTE/ATO/GLW/MET/REG switch
    revenue concept at the ASC-606 cutover. Requiring Q1/Q2/Q3/FY to share one
    source_tag outright (the original rule) made every such year's Q4
    permanently underivable -- 107 cases across 10 tickers -- even though the
    subtraction was perfectly valid.

    The scale is the quarters' SUMMED MAGNITUDES, annualized -- NOT the
    magnitude of their sum, which was a real bug: `abs(q1 + q2 + q3)` collapses
    toward zero whenever a year contains offsetting quarters, and every FY value
    then looks wildly out of scale against it. That rejected four confirmed,
    perfectly derivable quarters where the only problem was one loss quarter in
    the year -- Cboe fiscal 2022 (+110/-185/+150, FY 235 read as 2.34x its
    "run-rate" of 100), Dow fiscal 2020 (+239/-225/-25, FY 1,294 read as 88x),
    PG&E fiscal 2021 (0.12x) and EA fiscal 2012 (0.17x) -- none of which
    involved a concept mismatch at all.

    Only an UPPER bound is applied. A lower bound cannot be made to work: a year
    of offsetting quarters legitimately foots to a small annual figure (PG&E
    above), so "FY is much smaller than the quarters" is not evidence of
    anything. The case it was meant to catch -- an FY on a different, smaller
    concept, e.g. JPM's `Revenues` against `RevenuesNetOfInterestExpense`
    quarters -- yields a NEGATIVE derived Q4, which `_q4_is_coherent` now
    rejects outright on sign for any non-negative field."""
    run_rate = sum(abs(float(q)) for q in quarters) * 4 / 3
    if run_rate == 0:
        return False
    return abs(float(fy_value)) / run_rate <= Q4_TAG_MISMATCH_FY_MAX


def _annual_shaped(grp: pd.DataFrame) -> pd.Series:
    """Mask of rows whose OWN duration is a full fiscal year (~340-380d), i.e. the
    same shape filter `annual_flow` applies. A native fiscal_period label of 'FY'
    does NOT imply it: SEC's `fp` labels a fact by the FILING's period focus, so a
    10-K's discrete fourth-quarter column is natively labeled 'FY' too."""
    return grp["duration_days"].apply(classify_duration_shape) == "annual"


def _annual_anchor_tag(grp: pd.DataFrame) -> str | None:
    """The XBRL concept this fiscal year's figures should all be read from: the one
    the filer used for its ANNUAL total, which is the line it reports as the headline
    for this field (and the value Q4 is derived against, so agreeing with it is what
    makes the subtraction valid).

    Falls back to the most frequently occurring concept in the year when no
    annual-shaped fact exists (a fiscal year seen only through its 10-Qs), and to
    None when the year carries no `source_tag` at all -- in which case the tie-break
    it feeds is a no-op (`source_tag` is optional on this frame, exactly as the
    per-row `r.get("source_tag")` reads below assume)."""
    if "source_tag" not in grp.columns:
        return None
    annual = grp[_annual_shaped(grp) & grp["source_tag"].notna()]
    if not annual.empty:
        return str(annual.sort_values("filing_date").iloc[0]["source_tag"])
    tags = grp["source_tag"].dropna()
    return str(tags.mode().iloc[0]) if not tags.empty else None


def decumulate_quarterly_flow(facts: pd.DataFrame) -> pd.DataFrame:
    """Turn one (ticker, field)'s raw duration facts into discrete-quarter rows,
    generalizing `fetch_fundamentals.py::_quarterly_flow`'s YTD-decumulation + Q4
    derivation, re-keyed by native (fiscal_year, fiscal_period) instead of raw
    `start`-date grouping.

    `facts` columns required: fiscal_year (int), fiscal_period (one of
    'Q1'/'Q2'/'Q3'/'Q4'/'FY'), period_start, period_end, value (float),
    filing_date, accession_number, form, source_tag, is_amendment,
    fiscal_period_source.

    Returns rows with columns: fiscal_year, fiscal_period (only 'Q1'..'Q4'),
    value, filing_date, accession_number, form, derived (bool),
    derived_from_accessions (list[str]), period_start, period_end, source_tag,
    is_amendment, fiscal_period_source.

    Per fiscal_year:
      * Q1 is used AS-IS (a fiscal year's first quarter is never cumulative).
      * Q2/Q3: when BOTH a genuinely quarter-shaped (~75-120d) fact AND a
        longer (YTD-cumulative) fact exist for the SAME accession -- confirmed
        empirically: GAAP income-statement concepts are typically tagged with
        both a "3 months ended" AND a "6/9 months ended" context in the SAME
        10-Q -- the quarter-shaped one is used AS-IS (it is the as-filed
        ground truth; no arithmetic, no risk of compounding an earlier
        quarter's error). Subtraction (this value - the prior accepted
        discrete quarters) is the FALLBACK, only when no quarter-shaped fact
        was ever tagged (the norm for cash-flow-statement concepts, which
        ASC 230 only requires cumulatively).
      * Q4 is ALWAYS DERIVED as FY - (Q1 + Q2 + Q3), NEVER read from a native
        Q4 fact directly -- native Q4 labels have proven unreliable in
        multiple, unrelated ways this session (a Q4-shaped context carrying
        the FY-sized value; a genuinely-earlier-quarter fact mislabeled 'Q4').
        Derivation additionally requires: (a) when Q1/Q2/Q3/FY did NOT all
        resolve from the same underlying XBRL concept (`source_tag`), the FY
        value must still sit on the quarters' own scale
        (`_fy_matches_quarterly_run_rate`) -- filers rename a line's concept
        far more often than they change what it measures, so a mismatch alone
        is not evidence of a bad subtraction, but a scale gap is (confirmed
        real bug: JPM's FY sometimes resolves via a DIFFERENT candidate tag
        than its quarters -- `Revenues` vs `RevenuesNetOfInterestExpense`,
        JPM's actual bank-revenue concept -- so FY-(Q1+Q2+Q3) mixed two
        unrelated numbers and produced a wildly negative "Q4"). (b) the result
        passes `_q4_is_coherent` against Q1/Q2/Q3 -- catches whatever a scale
        check wouldn't (e.g. a corrupted individual quarter value). Either
        failure means NO Q4 row for that fiscal year --
        Q1/Q2/Q3 remain available, matching this pipeline's "null, never guess
        wrong" convention (`apply_plausibility_guards`). `filing_date` on the
        derived row = max(filing_date) across its four inputs (never stamped
        before all inputs are public).

    Every accession beyond the CANONICAL (earliest-filed, quarter-shaped-
    preferred) one for a given (fiscal_year, fiscal_period) -- almost always a
    10-Q/A or 10-K/A restating that quarter -- COEXISTS as its own extra row
    (its own accession_number, is_amendment=1.0) rather than being discarded:
    point-in-time reconstruction (`fundamentals_derive._resolve_latest_per_period`)
    needs the correction to appear exactly at its own filing_date. A per-
    (fiscal_period, accession_number) `seen` guard prevents any accidental PK
    collision (the same accession emitted twice for one fiscal_period).
    """
    out_cols = ["fiscal_year", "fiscal_period", "value", "filing_date",
                "accession_number", "form", "derived", "derived_from_accessions",
                "period_start", "period_end", "source_tag", "is_amendment",
                "fiscal_period_source"]
    if facts is None or facts.empty:
        return pd.DataFrame(columns=out_cols)

    d = facts.dropna(subset=["fiscal_year", "period_end", "value"]).copy()
    if d.empty:
        return pd.DataFrame(columns=out_cols)
    d["fiscal_period"] = d["fiscal_period"].map(normalize_fiscal_period_label)
    d["duration_days"] = [duration_days(s, e) for s, e in zip(d["period_start"], d["period_end"])]
    # This frame is always ONE logical field (the caller slices `raw` by field before
    # calling); `_q4_is_coherent` needs its name to know whether the field can go
    # negative. Absent when a caller passes a bare frame -- then the sign test simply
    # does not apply.
    field_name = str(d["field"].iloc[0]) if "field" in d.columns and not d.empty else None

    rows: list[dict] = []
    for fy, grp in d.groupby("fiscal_year", sort=True):
        # Canonical-selection order within a shared fiscal_period: earliest
        # filing_date first (original before amendment); then, among facts
        # sharing that filing_date (i.e. the SAME accession tagged this field
        # more than once), the fiscal year's ANCHOR CONCEPT (see
        # `_annual_anchor_tag`); then the quarter-shaped context over the
        # YTD-shaped one, so the as-filed discrete quarter is preferred over
        # arithmetic instead of an arbitrary tie-break.
        #
        # The anchor-concept key sits BETWEEN those two, and its absence was a
        # confirmed bug worth ~200x on real data. The quarter-shaped preference
        # is a rule about a fact's SHAPE, and it was being applied across facts
        # that resolved DIFFERENT source concepts -- so within one filing it
        # could pick a discrete-quarter fact belonging to a small, unrelated
        # line over the YTD fact of the line the rest of the year actually uses.
        # Confirmed on Valero for TEN consecutive years: its true D&A is
        # `DepreciationAmortizationAndAccretionNet` (2021: $2,405M/yr, tagged
        # YTD-only), but it ALSO tags a $47M/yr `DepreciationAndAmortization`
        # line WITH a discrete-quarter context -- so every fiscal Q2/Q3 picked
        # the ~$11M discrete fact and the stored series collapsed from ~$500M to
        # ~$12M and back, every single year. Keying on filing_date FIRST leaves
        # the amendment-coexistence ordering (and therefore point-in-time
        # reconstruction) exactly as it was.
        grp = grp.assign(_quarterly_shaped=grp["duration_days"].apply(classify_duration_shape) == "quarterly")
        anchor_tag = _annual_anchor_tag(grp)
        grp = grp.assign(_is_anchor_tag=grp["source_tag"].eq(anchor_tag) if anchor_tag else False)
        grp = grp.sort_values(["filing_date", "_is_anchor_tag", "_quarterly_shaped"],
                              ascending=[True, False, False])
        grp = _reassign_misordered_native_q4(grp)
        seen_keys: set[tuple] = set()

        def _append(fp: str, r: pd.Series, value: float, *, derived: bool = False,
                    derived_from_accessions=None, period_start=pd.NaT, period_end=None) -> None:
            acc = r["accession_number"]
            key = (fp, acc)
            if key in seen_keys:
                return
            seen_keys.add(key)
            rows.append({
                "fiscal_year": fy, "fiscal_period": fp, "value": value,
                "filing_date": r["filing_date"], "accession_number": acc,
                "form": r["form"], "derived": derived,
                "derived_from_accessions": derived_from_accessions,
                "period_start": period_start,
                "period_end": period_end if period_end is not None else r["period_end"],
                "source_tag": r.get("source_tag"), "is_amendment": r.get("is_amendment", 0.0),
                "fiscal_period_source": r.get("fiscal_period_source"),
            })

        quarters = grp[grp["fiscal_period"].isin(_QUARTER_LABELS)]
        by_fp_all = {fp: g for fp, g in quarters.groupby("fiscal_period", sort=False)}

        discrete: dict[str, float] = {}
        cumulative_so_far = 0.0
        for fp in _QUARTER_LABELS:
            g = by_fp_all.get(fp)
            if g is None or g.empty:
                continue
            prior_cumulative = cumulative_so_far

            def _discrete_value(row: pd.Series, _fp: str = fp, _prior: float = prior_cumulative) -> float:
                if _fp == "Q1" or classify_duration_shape(row["duration_days"]) == "quarterly":
                    return float(row["value"])
                # YTD-cumulative (semiannual/nine_month/other-but-longer-than-a-quarter):
                # discrete = this value - the sum of prior accepted discrete quarters.
                return float(row["value"]) - _prior

            canonical = g.iloc[0]
            val = _discrete_value(canonical)
            discrete[fp] = val
            cumulative_so_far += val
            p_start = (canonical["period_end"] - pd.Timedelta(days=round(canonical["duration_days"]))
                      if pd.notna(canonical["duration_days"]) else pd.NaT)
            _append(fp, canonical, val, period_start=p_start)
            for _, dup in g.iloc[1:].iterrows():
                dup_start = (dup["period_end"] - pd.Timedelta(days=round(dup["duration_days"]))
                            if pd.notna(dup["duration_days"]) else pd.NaT)
                _append(fp, dup, _discrete_value(dup), period_start=dup_start)

        # Q4: ALWAYS derived as FY - (Q1+Q2+Q3) -- see the module/function
        # docstrings for why a native Q4 fact is never trusted directly.
        #
        # The FY anchor must be ANNUAL-SHAPED, not merely labeled 'FY'. SEC's
        # native `fp` labels a fact by the FILING's period focus, so a 10-K that
        # publishes a fourth-quarter column tags THAT ~91-day fact 'FY' as well
        # -- and the quarter-shaped tie-break above then sorted it AHEAD of the
        # real ~365-day annual figure, so `iloc[0]` used one quarter as if it
        # were the whole year. Confirmed on Skyworks fiscal 2020, whose 10-K
        # carries both 3,355.7M (370d) and 956.8M (97d) under fp='FY': the
        # derivation computed 956.8 - 2,399.0 = -1,442.2M, which
        # `_q4_is_coherent` then (correctly) threw away -- so the quarter was
        # LOST even though the filer had published it outright. Where the guard
        # happened to pass instead, the bad value was STORED: 77 (ticker, field,
        # year) cells fail the Q1+Q2+Q3+Q4 == FY footing, concentrated in
        # 53-week fiscal years (Cisco 2017 alone accounts for 41 fields, Skyworks
        # 2014 for 3), because a 53rd week is what pushes a filer to publish the
        # extra quarterly column. Applying the same shape filter `annual_flow`
        # already uses removes both failure modes -- and Skyworks 2020 then
        # derives 3,355.7 - 2,399.0 = 956.7M, matching the filer's own published
        # 956.8M to rounding.
        fy_rows = grp[(grp["fiscal_period"] == "FY") & _annual_shaped(grp)]
        can_derive = not fy_rows.empty and all(k in discrete for k in _QUARTER_LABELS)
        if not can_derive:
            continue

        fy_r = fy_rows.iloc[0]
        q1, q2, q3 = by_fp_all["Q1"].iloc[0], by_fp_all["Q2"].iloc[0], by_fp_all["Q3"].iloc[0]
        inputs = [fy_r, q1, q2, q3]

        q123_sum = sum(discrete[k] for k in _QUARTER_LABELS)
        q123 = (discrete["Q1"], discrete["Q2"], discrete["Q3"])
        source_tags = {r.get("source_tag") for r in inputs if r.get("source_tag")}
        if len(source_tags) > 1 and not _fy_matches_quarterly_run_rate(fy_r["value"], q123):
            continue   # Q1/Q2/Q3/FY on genuinely different concepts -- unresolvable

        derived_val = float(fy_r["value"]) - q123_sum
        if not _q4_is_coherent(derived_val, *q123, field=field_name):
            continue   # wildly incoherent vs Q1-Q3 -- null rather than store a bad number

        _append("Q4", fy_r, derived_val, derived=True,
                derived_from_accessions=[r["accession_number"] for r in inputs],
                period_start=pd.NaT, period_end=fy_r["period_end"])
        rows[-1]["filing_date"] = max(r["filing_date"] for r in inputs)
        rows[-1]["source_tag"] = None
        rows[-1]["is_amendment"] = 0.0

    return pd.DataFrame(rows, columns=out_cols)


def annual_flow(facts: pd.DataFrame) -> pd.DataFrame:
    """Keep annual (~365d duration) observations, native-fiscal-year-keyed. One
    row PER ACCESSION -- a 10-K/A restating the annual figure must coexist with
    the original (own accession_number, is_amendment=1.0) rather than being
    discarded; `fundamentals_derive._resolve_latest_per_period` picks whichever
    qualifies as of a given as_of_cutoff for point-in-time reconstruction."""
    out_cols = ["fiscal_year", "fiscal_period", "value", "filing_date", "accession_number", "form",
               "period_start", "period_end", "source_tag", "is_amendment", "fiscal_period_source"]
    if facts is None or facts.empty:
        return pd.DataFrame(columns=out_cols)
    d = facts.dropna(subset=["fiscal_year", "period_start", "period_end", "value"]).copy()
    d["fiscal_period"] = d["fiscal_period"].map(normalize_fiscal_period_label)
    d["duration_days"] = [duration_days(s, e) for s, e in zip(d["period_start"], d["period_end"])]
    d = d[(d["duration_days"] >= ANNUAL_MIN_DAYS) & (d["duration_days"] <= ANNUAL_MAX_DAYS)]
    d = d[d["fiscal_period"] == "FY"]
    if d.empty:
        return pd.DataFrame(columns=out_cols)
    d = d.sort_values(["fiscal_year", "filing_date"]).drop_duplicates(
        subset=["fiscal_year", "accession_number"], keep="first")
    for c in ("source_tag", "is_amendment", "fiscal_period_source"):
        if c not in d.columns:
            d[c] = None
    return d[out_cols].sort_values(["fiscal_year", "filing_date"]).reset_index(drop=True)


def drop_derived_q4_for_partial_fiscal_years(facts: pd.DataFrame) -> pd.DataFrame:
    """Discard every DERIVED Q4 for a fiscal year whose FY anchor is demonstrably
    NOT the whole year -- detected CROSS-FIELD, which is the only way to see it.

    `_q4_is_coherent`'s sign test catches a partial FY anchor one field at a time,
    but only for a field that cannot go negative. When the FY row is partial for the
    WHOLE YEAR, the signed fields (netIncome, operatingIncome, pretaxIncome, EPS)
    have no such tell and silently take a garbage Q4. Confirmed on Johnson Controls
    fiscal 2012, where the 10-K's annual row is roughly one quarter's worth of every
    line: revenue FY $10,403M against $13,022M already booked in Q1-Q3 (true FY
    ~$42B), cost of revenue $6,626M vs $7,916M, SG&A $2,903M vs $3,525M. Revenue
    cannot shrink, so those three were correctly nulled on sign -- but the same bad
    anchor also produced a -$430M netIncome and -$638M operatingIncome Q4 that
    looked like an ordinary restructuring quarter and would have been stored.
    S&P Global fiscal 2012 fails the same way across 5 fields.

    The test is deliberately CROSS-FIELD and needs `MIN_PARTIAL_FY_FIELDS`
    independent non-negative fields to agree, because a SINGLE field failing it
    means something different and much more common: that one field's FY resolved a
    different concept than its quarters (KeyCorp's D&A does this in eight separate
    years, Valero's and United Rentals' in one each). Those are already handled per
    field by the sign test, and vetoing a whole year on that evidence would throw
    away every other field's perfectly good Q4.

    As-reported rows are never touched -- only rows this pipeline itself computed."""
    required = {"field", "duration_type", "fiscal_period", "fiscal_year", "value", "derived"}
    if facts is None or facts.empty or not required.issubset(facts.columns):
        return facts

    non_negative = facts["field"].isin(NON_NEGATIVE_FLOW_FIELDS)
    q123 = facts[non_negative & (facts["duration_type"] == "quarterly")
                 & facts["fiscal_period"].isin(_QUARTER_LABELS)]
    annual = facts[non_negative & (facts["duration_type"] == "annual")]
    if q123.empty or annual.empty:
        return facts

    # only a COMPLETE Q1-Q3 set proves anything -- a missing quarter makes the
    # cumulative smaller for an innocent reason
    cum = q123.groupby(["field", "fiscal_year"])["value"].agg(["size", "sum"])
    cum = cum[cum["size"] == len(_QUARTER_LABELS)]
    fy = annual.groupby(["field", "fiscal_year"])["value"].first()
    joined = cum.join(fy.rename("fy_value"), how="inner")
    if joined.empty:
        return facts

    partial = joined[joined["fy_value"] < joined["sum"] * PARTIAL_FY_TOLERANCE]
    if partial.empty:
        return facts
    suspect_years = {y for _f, y in partial.index
                     if sum(1 for _f2, y2 in partial.index if y2 == y) >= MIN_PARTIAL_FY_FIELDS}
    if not suspect_years:
        return facts

    drop = (facts["fiscal_period"] == "Q4") & (facts["duration_type"] == "quarterly") \
        & (facts["derived"].astype(float) == 1.0) & facts["fiscal_year"].isin(suspect_years)
    return facts[~drop].reset_index(drop=True)


def reassign_misordered_instant_facts(facts: pd.DataFrame) -> pd.DataFrame:
    """Instant (balance-sheet) facts inherit their fiscal_year/fiscal_period via
    backfill from a DURATION fact in the SAME filing
    (`fetch_fundamentals_edgar.backfill_fiscal_period_from_filing`) -- so a
    filing-wide native fiscal_period mislabeling (confirmed empirically: MAA's
    Q1-2013/Q1-2017 10-Qs) propagates onto EVERY instant concept in that filing,
    not just the duration concept it originated on. Confirmed against real data:
    MAA's sharesOutstanding/cash/notesPayable/rentalIncome/realEstateGross all
    showed THREE different accessions (the Apr/Jul/Oct 2017 10-Qs) claiming the
    SAME (fiscal_year=2017, fiscal_period='Q4') even though their period_end
    dates are genuinely three different quarters -- goodwill happened to be
    unchanged across them, masking the bug for that one field.

    Operates on the FULL per-ticker `fundamentals_facts`-shaped frame (all
    fields, all duration_types) rather than one field at a time, since it needs
    a fiscal-year ANCHOR that an instant field never has on its own (it has no
    'annual' counterpart) -- borrowed from ANY 'annual'-duration_type row for
    that fiscal_year (any field). Ranking (see `_relabel_by_chronological_rank`)
    happens PER (fiscal_year, field), never across different fields, which are
    unrelated business concepts. Call BEFORE `populate_amends_accession` (which
    groups by the resolved fiscal_period -- relabeling after would leave stale
    amends_accession links for the reassigned rows)."""
    if facts is None or facts.empty or "duration_type" not in facts.columns:
        return facts
    annual = facts[facts["duration_type"] == "annual"]
    fy_bounds = (annual.dropna(subset=["period_start", "period_end"])
                .groupby("fiscal_year")[["period_start", "period_end"]].first())
    if fy_bounds.empty:
        return facts
    facts = facts.copy()
    instant_q4 = facts[(facts["duration_type"] == "instant") & (facts["fiscal_period"] == "Q4")]
    for (fy, _field), candidates in instant_q4.groupby(["fiscal_year", "field"]):
        if fy not in fy_bounds.index:
            continue
        fy_start = pd.Timestamp(fy_bounds.loc[fy, "period_start"])
        fy_length = (pd.Timestamp(fy_bounds.loc[fy, "period_end"]) - fy_start).days
        mislabeled = candidates[candidates["period_end"].apply(
            lambda s: _is_mislabeled_q4(pd.Timestamp(s), fy_start, fy_length))]
        if mislabeled.empty:
            continue
        for idx, new_fp in _relabel_by_chronological_rank(mislabeled, "period_end").items():
            facts.loc[idx, "fiscal_period"] = new_fp
    return facts


def instant_stock(facts: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time balance-sheet items, native-(fiscal_year, fiscal_period)-
    keyed. One row PER ACCESSION -- see `annual_flow`'s docstring for the
    amendment-coexistence rationale (identical here).

    An instant fact's fiscal_period is BORROWED from a duration fact in the
    same filing (`fetch_fundamentals_edgar.backfill_fiscal_period_from_filing`
    -- instant facts carry no native fy/fp of their own), so a 10-K's
    balance-sheet snapshot inherits that filing's 'FY' label. But a fiscal
    year has no separate "FY balance sheet": the year-end snapshot IS the Q4
    one, and labelling it 'FY' left every instant field with a hole at Q4 in
    all four quarters of coverage -- 14 of 59 quarters per field on all 10
    tickers audited (9,387 rows table-wide against just 130 genuine 'Q4'
    instants). Normalized to 'Q4' here so `fundamentals_facts` carries one
    consistent quarter grid for flow and instant fields alike. Safe as a
    blanket rule: 'FY' is only ever assigned from a 10-K, and
    `_filing_current_period_rows` has already restricted every fact to
    period_end == that filing's period_of_report, so an instant labeled 'FY'
    is by construction the fiscal-year-end snapshot. `fundamentals_derive`
    keys instant series on period_end, not fiscal_period, so nothing
    downstream changes.

    Restricted to facts with NO period_start, which is what distinguishes a
    genuine instant from the `LATEST_DURATION_TAGS` fields routed through this
    same function (dilutedShares, basicShares, effectiveTaxRate,
    reportableSegments -- duration facts merely TAKEN point-in-time). For those
    'FY' and 'Q4' are two DIFFERENT measures that a 10-K legitimately tags side
    by side: confirmed on CBRE fiscal 2011, whose 10-K reports a 318,454,191
    full-year weighted-average basic share count AND a 320,638,316 Q4-only one,
    both dated 2011-12-31. Renaming those would collapse the pair onto one
    primary key and keep an arbitrary one of the two."""
    out_cols = ["fiscal_year", "fiscal_period", "value", "filing_date", "accession_number", "form",
               "period_start", "period_end", "source_tag", "is_amendment", "fiscal_period_source"]
    if facts is None or facts.empty:
        return pd.DataFrame(columns=out_cols)
    d = facts.dropna(subset=["fiscal_year", "fiscal_period", "value", "filing_date"]).copy()
    if "period_start" not in d.columns:
        d["period_start"] = pd.NaT
    year_end_snapshot = (d["fiscal_period"] == "FY") & d["period_start"].isna()
    d.loc[year_end_snapshot, "fiscal_period"] = "Q4"
    if "period_end" not in d.columns:
        d["period_end"] = pd.NaT
    d = d.sort_values(["fiscal_year", "fiscal_period", "filing_date"]).drop_duplicates(
        subset=["fiscal_year", "fiscal_period", "accession_number"], keep="first")
    for c in ("source_tag", "is_amendment", "fiscal_period_source"):
        if c not in d.columns:
            d[c] = None
    return d[out_cols].sort_values(["fiscal_year", "fiscal_period"]).reset_index(drop=True)


_MAX_10Q_FISCAL_QUARTERS = 3


def backfill_fiscal_period_by_filing_order(raw: pd.DataFrame) -> pd.DataFrame:
    """Some filers' 10-Q filings carry NO native fiscal_period on ANY fact in the
    ENTIRE filing (confirmed empirically: KR's Q1 10-Qs -- e.g. accession
    0001104659-15-048764, period_end 2015-05-23 -- have fiscal_year natively
    populated but fiscal_period blank on all ~60 tagged facts that filing
    produces, so `fetch_fundamentals_edgar.backfill_fiscal_period_from_filing`'s
    same-filing borrow has no native row anywhere to draw from). Left
    unresolved, such a fact never matches `decumulate_quarterly_flow`'s
    `fiscal_period.isin(_QUARTER_LABELS)` filter and silently vanishes -- this
    is why KR's Q1 revenue was entirely absent while Q2/Q3/FY (whose filings DO
    carry native fp) were present, and why KR's Q2 then looked like a mislabeled
    ~2.5x jump: nothing was wrong with Q2 itself, Q1 was just missing.

    Operates on the FULL per-ticker raw-facts frame (all filings, all fields),
    since resolving a 10-Q's quarter needs the ticker's OTHER 10-Qs in the SAME
    fiscal year for context:

      * 10-K (or 10-K/A): an annual-shaped duration (~340-380d) is unambiguous
        on its own, no cross-filing context needed -- always 'FY'.
      * 10-Q (or 10-Q/A): a fiscal year has AT MOST three 10-Qs (Q1-Q3; Q4 is
        only ever reported via the 10-K), so the earliest period_end among them
        is Q1, the next Q2, the next Q3 -- regardless of whether any of them
        carries a native label. A label already claimed by a NATIVELY-resolved
        sibling filing in the same fiscal year is never reassigned to a
        different period (that slot is skipped); if there are more distinct
        missing-fp period_ends than remaining unclaimed labels, the whole
        fiscal year is left untouched rather than guess wrong -- matching this
        pipeline's "null, never guess wrong" convention.
    """
    if raw is None or raw.empty or "form" not in raw.columns:
        return raw
    raw = raw.copy()
    missing_fp = raw["fiscal_period"].isna() & raw["fiscal_year"].notna()

    is_10k = raw["form"].astype(str).str.upper().str.startswith("10-K")
    days = pd.Series(
        [duration_days(s, e) for s, e in zip(raw.get("period_start"), raw.get("period_end"))],
        index=raw.index)
    is_annual_shaped = days.between(ANNUAL_MIN_DAYS, ANNUAL_MAX_DAYS)
    raw.loc[missing_fp & is_10k & is_annual_shaped, "fiscal_period"] = "FY"

    missing_fp = raw["fiscal_period"].isna() & raw["fiscal_year"].notna()
    is_10q = raw["form"].astype(str).str.upper().str.startswith("10-Q")
    candidates = raw[is_10q & missing_fp]
    if candidates.empty:
        return raw

    for fy, grp in candidates.groupby("fiscal_year"):
        period_ends = sorted(pd.to_datetime(grp["period_end"], errors="coerce").dropna().unique())
        if not period_ends or len(period_ends) > _MAX_10Q_FISCAL_QUARTERS:
            continue
        fy_rows = raw[raw["fiscal_year"] == fy]
        taken = set(fy_rows.loc[fy_rows["fiscal_period"].notna(), "fiscal_period"].unique())
        available_labels = [lbl for lbl in _QUARTER_LABELS if lbl not in taken]
        if len(period_ends) > len(available_labels):
            continue
        for period_end, label in zip(period_ends, available_labels):
            match = (is_10q & missing_fp & (raw["fiscal_year"] == fy)
                    & (pd.to_datetime(raw["period_end"], errors="coerce") == period_end))
            raw.loc[match, "fiscal_period"] = label
    return raw


def _fiscal_year_end_dates(raw: pd.DataFrame) -> list[pd.Timestamp]:
    """The ticker's ACTUAL fiscal-year-end dates, read off its 10-K filings: the
    period_end of every annual-shaped (~340-380d) duration fact a 10-K reported
    for its own current period. Sorted, de-duplicated. These are exact -- no
    month/day rule is assumed -- which is what makes the calendar below immune to
    a 52/53-week filer's drift (Cisco's fiscal year ends 07-30, 07-28, 07-27,
    07-26, ... so any fixed anniversary rule misplaces a quarter at the boundary)."""
    is_10k = raw["form"].astype(str).str.upper().str.startswith("10-K")
    days = pd.Series([duration_days(s, e) for s, e in zip(raw["period_start"], raw["period_end"])],
                     index=raw.index)
    ends = raw.loc[is_10k & days.between(ANNUAL_MIN_DAYS, ANNUAL_MAX_DAYS), "period_end"]
    return sorted(pd.to_datetime(ends, errors="coerce").dropna().unique())


def _fiscal_year_index(period_end: pd.Timestamp, fye_dates: list[pd.Timestamp]) -> int:
    """Ordinal of the fiscal year a `period_end` belongs to, counting from
    `fye_dates[0]`'s fiscal year as 0: the index of the FIRST fiscal-year end at
    or after it. A quarter end always sits ~1-9 months BEFORE its own fiscal-year
    end, and a fiscal-year end matches its own entry exactly, so no tolerance is
    needed inside the observed range.

    Outside that range the ordinal is extrapolated off `fye_dates[0]` at
    `FISCAL_YEAR_MEAN_DAYS` per year -- forward for the in-progress fiscal year
    (whose 10-K does not exist yet) and backward for the quarters preceding the
    earliest 10-K in the history window. Normally a single step either way, so the
    ~6d a 53-week year adds cannot accumulate; `FISCAL_YEAR_EXTRAPOLATION_GRACE_
    DAYS` absorbs it on that step."""
    for i, fye in enumerate(fye_dates):
        if period_end <= fye:
            if i > 0 or (fye - period_end).days <= FISCAL_YEAR_MEAN_DAYS:
                return i
            break                       # >1 year before the earliest -- extrapolate
    offset_days = (period_end - fye_dates[0]).days - FISCAL_YEAR_EXTRAPOLATION_GRACE_DAYS
    return math.ceil(offset_days / FISCAL_YEAR_MEAN_DAYS)


def resolve_fiscal_year_by_filing_calendar(raw: pd.DataFrame) -> pd.DataFrame:
    """Re-key every raw fact's `fiscal_year` to the ticker's OWN fiscal calendar,
    reconstructed from its 10-K period ends, instead of trusting a per-fact or
    per-filing label that carries filer/parser typos.

    Both available labels are individually unreliable, in DIFFERENT filings, and
    both failures are silent because the wrong year is a perfectly well-formed
    one -- it simply collides with a real fiscal year and destroys BOTH:

      * edgartools' PER-FACT `fiscal_year`: Cisco's fiscal-2016 10-K (accession
        0000858877-16-000117, period_end 2016-07-30) tags every current-period
        fact `fiscal_year=2017`, so its annual + Q4 rows landed on fiscal 2017
        beside the real fiscal-2017 10-K. Cisco fiscal 2016 ended up with NO
        annual row and no Q4 at all, and fiscal 2017 with two filings' worth
        (98 FY rows against the ~49 every other year has). Johnson Controls'
        fiscal-2016 Q1 10-Q (period_end 2015-12-31) fails the same way, labeled
        `fiscal_year=2015`.
      * the FILING's cover page (`dei:DocumentFiscalYearFocus`): J.M. Smucker's
        fiscal-2015 Q1 10-Q (period_end 2014-07-31) says 2014, one year early --
        its own neighbours (Q1 2013-07-31 -> 2014, Q2 2014-10-31 -> 2015) prove
        2015. So "prefer the cover page" is not a fix either.

    What IS reliable is the SHAPE of a fiscal calendar: consecutive 10-K period
    ends are ~1 year apart, and every quarter belongs to the fiscal year whose
    end comes next. So the DATES come from the 10-Ks (exact, `_fiscal_year_end_
    dates`) and only the starting LABEL is voted on -- across every filing and
    both label sources at once (`_fiscal_year_label_offset`), which is what makes
    a single typo on either side get outvoted rather than propagate.

    Requires a `cover_fiscal_year` column (the filing-level cover-page focus,
    attached by `fetch_fundamentals_edgar._filing_current_period_rows`); absent
    it, only the native labels vote. Leaves `fiscal_year` untouched when there is
    no 10-K to anchor on or too few votes to be meaningful
    (`MIN_FISCAL_YEAR_LABEL_VOTES`) -- this pipeline's "null, never guess wrong"
    default, which also keeps a thin incremental run from re-labelling a ticker
    off one filing."""
    required = {"form", "period_start", "period_end", "fiscal_year"}
    if raw is None or raw.empty or not required.issubset(raw.columns):
        return raw
    fye_dates = _fiscal_year_end_dates(raw)
    if not fye_dates:
        return raw

    period_end = pd.to_datetime(raw["period_end"], errors="coerce")
    resolvable = period_end.notna()
    if not resolvable.any():
        return raw
    index = period_end[resolvable].map(lambda p: _fiscal_year_index(p, fye_dates))

    offset = _fiscal_year_label_offset(raw.loc[resolvable], index)
    if offset is None:
        return raw
    raw = raw.copy()
    raw.loc[resolvable, "fiscal_year"] = (index + offset).astype("int64")
    return raw


def _fiscal_year_label_offset(raw: pd.DataFrame, index: pd.Series) -> int | None:
    """The single integer that turns a fiscal-year ORDINAL (`_fiscal_year_index`)
    into the issuer's own fiscal-year NUMBER, chosen as the most common value of
    `label - ordinal` across every labelled row and both label sources
    (edgartools' per-fact `fiscal_year` and the filing's cover-page
    `cover_fiscal_year`).

    Voting rather than reading one filing is the whole point: a correct label
    source agrees with the calendar on every row, so it contributes one identical
    vote per row, while a typo contributes a different one on the handful of rows
    it affects. Returns None when there are fewer than
    `MIN_FISCAL_YEAR_LABEL_VOTES` votes -- too thin to overrule anything."""
    votes: list[int] = []
    for col in ("cover_fiscal_year", "fiscal_year"):
        if col not in raw.columns:
            continue
        labels = pd.to_numeric(raw[col], errors="coerce")
        votes.extend((labels - index).dropna().astype(int).tolist())
    if len(votes) < MIN_FISCAL_YEAR_LABEL_VOTES:
        return None
    return int(pd.Series(votes).mode().iloc[0])


_SNAPSHOT_KEY = ["fiscal_year", "fiscal_period", "accession_number"]
_SNAPSHOT_VALUE_COLS = ["value", "filing_date", "period_start", "period_end", "form",
                        "is_amendment", "fiscal_period_source"]
# the `stockholdersEquity` candidate that ALREADY includes noncontrolling
# interests -- when it is the resolved tag, `minorityInterest` must not be
# added on top or NCI is counted twice.
_EQUITY_INCL_NCI_TAG = "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"


def _snapshot_values(facts: pd.DataFrame, field: str) -> pd.DataFrame:
    """One instant field's value per balance-sheet snapshot (fiscal_year,
    fiscal_period, accession_number). Only the KEY + value-bearing columns are
    returned, never the whole row -- `ticker`/`cik` are constant across this
    single-ticker frame and would otherwise collide under merge suffixes."""
    cols = _SNAPSHOT_KEY + _SNAPSHOT_VALUE_COLS
    sub = facts.loc[facts["field"] == field, [c for c in cols if c in facts.columns]]
    return sub.drop_duplicates(subset=_SNAPSHOT_KEY)


def _as_derived_liabilities(facts: pd.DataFrame, merged: pd.DataFrame,
                            value: pd.Series, suffix: str) -> pd.DataFrame:
    """Shape a computed total-liabilities series into `fundamentals_facts` rows,
    taking provenance from the `suffix`-suffixed leg of `merged` (the balance-
    sheet line the derivation is anchored on) and the LATEST filing_date across
    every leg that fed it."""
    constant_cols = {c: facts[c].iloc[0] for c in ("ticker", "cik") if c in facts.columns}
    filed_cols = [c for c in merged.columns if c.startswith("filing_date")]
    rows = merged.assign(
        field="totalLiabilities", duration_type="instant", value=value,
        filing_date=merged[filed_cols].max(axis=1),
        period_start=merged[f"period_start{suffix}"], period_end=merged[f"period_end{suffix}"],
        form=merged[f"form{suffix}"], source_tag=None,
        is_amendment=merged[f"is_amendment{suffix}"],
        fiscal_period_source=merged[f"fiscal_period_source{suffix}"],
        derived=1.0, derived_from_accessions=merged["accession_number"],
        **constant_cols,
    )
    # `reindex`, not `rows[facts.columns]`: these derivations build only the
    # columns they know about, so any OTHER column present on the incoming frame
    # (`unit`/`amends_accession` are added later in the pipeline, but a caller
    # passing already-persisted rows has them from the start) must come through
    # as null rather than raising KeyError.
    return rows.reindex(columns=facts.columns)


def _liabilities_from_current_split(facts: pd.DataFrame) -> pd.DataFrame | None:
    """`totalLiabilities = currentLiabilities + totalLiabilitiesNoncurrent`, for
    filers (confirmed: ACN, ADI, ADM) that tag only the split, never the
    combined `us-gaap:Liabilities`."""
    current = _snapshot_values(facts, "currentLiabilities")
    noncurrent = _snapshot_values(facts, "totalLiabilitiesNoncurrent")
    if current.empty or noncurrent.empty:
        return None
    merged = current.merge(noncurrent, on=_SNAPSHOT_KEY, suffixes=("_cur", "_ncur"))
    if merged.empty:
        return None
    return _as_derived_liabilities(facts, merged,
                                   merged["value_cur"] + merged["value_ncur"], "_cur")


def _liabilities_from_footing(facts: pd.DataFrame) -> pd.DataFrame | None:
    """`totalLiabilities = LiabilitiesAndStockholdersEquity - equity(incl NCI)
    - redeemable NCI`, rearranging the balance-sheet identity the filer itself
    published.

    Needed because the current/noncurrent split above only helps a filer that
    tags BOTH halves. Confirmed on live data: McDonald's and Atmos Energy never
    tag `Liabilities` OR `LiabilitiesNoncurrent` in ANY filing, so
    `totalLiabilities` was absent for their entire history; DTE Energy tagged
    `LiabilitiesNoncurrent` only through fiscal 2019 and nothing after, cutting
    its series off mid-history. All three tag `LiabilitiesAndStockholdersEquity`
    in every filing (it is the balance sheet's footing, universal across the
    S&P 500), so the identity closes exactly.

    Mezzanine/temporary equity (`redeemableNCI`) sits BETWEEN liabilities and
    equity in that footing and so must come out too; it is absent for most
    filers, in which case it contributes nothing."""
    footing = _snapshot_values(facts, "balanceSheetFooting")
    equity = _snapshot_values(facts, "stockholdersEquity")
    if footing.empty or equity.empty:
        return None
    merged = footing.merge(equity, on=_SNAPSHOT_KEY, suffixes=("_foot", "_eq"))
    if merged.empty:
        return None

    # `stockholdersEquity` coalesces a parent-only and an incl-NCI candidate;
    # only the parent-only one needs `minorityInterest` added back.
    equity_tags = (facts.loc[facts["field"] == "stockholdersEquity", _SNAPSHOT_KEY + ["source_tag"]]
                  .drop_duplicates(subset=_SNAPSHOT_KEY))
    merged = merged.merge(equity_tags, on=_SNAPSHOT_KEY, how="left")
    equity_is_parent_only = ~merged["source_tag"].astype(str).str.endswith(_EQUITY_INCL_NCI_TAG)

    def _addend(field: str, mask: pd.Series | None = None) -> pd.Series:
        """One optional balance-sheet leg aligned to `merged`, 0 where the filer
        does not report it (most do not) or where `mask` excludes it."""
        zeros = pd.Series(0.0, index=merged.index)
        part = _snapshot_values(facts, field)
        if part.empty:
            return zeros
        vals = merged.merge(part[_SNAPSHOT_KEY + ["value"]].rename(columns={"value": "_v"}),
                            on=_SNAPSHOT_KEY, how="left")["_v"].fillna(0.0)
        vals.index = merged.index
        return vals if mask is None else vals.where(mask, 0.0)

    total_equity = merged["value_eq"] + _addend("minorityInterest", equity_is_parent_only)
    value = merged["value_foot"] - total_equity - _addend("redeemableNCI")
    return _as_derived_liabilities(facts, merged.drop(columns="source_tag"), value, "_foot")


# `CashAndDueFromBanks` is a bank balance sheet's FIRST cash line, not its cash
# TOTAL -- the rest sits in `InterestBearingDepositsInBanks` (reserves at the
# central bank, balances at correspondents).
_PARTIAL_BANK_CASH_TAG = "CashAndDueFromBanks"


def derive_bank_cash(facts: pd.DataFrame) -> pd.DataFrame:
    """Complete `cash` wherever it resolved to the PARTIAL bank line
    `CashAndDueFromBanks`, by adding the interest-bearing balances the filer
    reports beside it.

    Banks tag `CashAndCashEquivalentsAtCarryingValue` inconsistently, and the
    `cash` coalesce falls through to `CashAndDueFromBanks` whenever they don't
    -- which silently swaps in a number several times smaller. Confirmed on
    live data: Citigroup's series ALTERNATES between the two concepts every
    couple of years, so stored cash jumps $22.6B -> $202.7B -> $24.4B with no
    business event behind it; Regions Financial stopped tagging the total after
    2021-Q3 and the series stepped down from $27.5B to $2.2B.

    The reconstruction is exact, not an estimate -- verified against the years
    where the filer tags all three: Citigroup 2018-Q3 $25.727B + $173.559B ==
    $199.286B, Regions 2018-Q3 $1.911B + $1.584B == $3.495B, both equal to the
    filer's own `CashAndCashEquivalentsAtCarryingValue` to the dollar. The
    partial value is REPLACED rather than kept alongside, for the same reason
    the ASC-606 revenue slice is excluded in `fetch_fundamentals_edgar.
    build_tag_frames`: a correctly-tagged component is still the wrong answer
    for a field that means "total cash", and the PK
    (ticker, accession, field, fiscal_year, fiscal_period, duration_type)
    admits only one `cash` row per snapshot anyway."""
    if facts is None or facts.empty or "field" not in facts.columns:
        return facts
    partial = ((facts["field"] == "cash")
              & facts["source_tag"].astype(str).str.endswith(_PARTIAL_BANK_CASH_TAG))
    if not partial.any():
        return facts
    deposits = _snapshot_values(facts, "interestBearingDepositsInBanks")[_SNAPSHOT_KEY + ["value"]]
    if deposits.empty:
        return facts

    matched = (facts.loc[partial, _SNAPSHOT_KEY]
              .merge(deposits.rename(columns={"value": "_deposits"}), on=_SNAPSHOT_KEY, how="left")
              .set_index(facts.index[partial])["_deposits"])
    idx = matched.index[matched.notna()]
    if idx.empty:
        return facts
    facts = facts.copy()
    facts.loc[idx, "value"] = facts.loc[idx, "value"] + matched[idx]
    facts.loc[idx, "source_tag"] = None
    facts.loc[idx, "derived"] = 1.0
    facts.loc[idx, "derived_from_accessions"] = facts.loc[idx, "accession_number"]
    return facts


def derive_missing_pretax_income(facts: pd.DataFrame) -> pd.DataFrame:
    """`pretaxIncome = netIncome + incomeTaxExpense (+ nciIncome)`, for filers
    that tag no pre-tax income concept at all.

    Both `IncomeLossFromContinuingOperationsBeforeIncomeTaxes...` variants are
    genuinely absent from some filers' entire XBRL history -- confirmed:
    McDonald's tags neither in any filing (only the Domestic/Foreign tax-
    footnote split, which is annual-only), and Chubb tags neither before fiscal
    2015. The identity closes exactly: `nciIncome` is added back only when the
    resolved `netIncome` is the parent-only `NetIncomeLoss`, since the
    alternative candidate (`ProfitLoss`) already includes noncontrolling
    interests and would otherwise double-count them. Filers with no NCI at all
    (McDonald's, Atmos) contribute nothing from that leg.

    Fill-only: a snapshot that already carries an as-reported `pretaxIncome`
    is never touched. Runs per duration_type, so quarterly and annual rows are
    each derived from their own basis and never mixed."""
    if facts is None or facts.empty or "field" not in facts.columns:
        return facts
    key = _SNAPSHOT_KEY + ["duration_type"]
    net = facts.loc[facts["field"] == "netIncome",
                   key + _SNAPSHOT_VALUE_COLS + ["source_tag"]].drop_duplicates(subset=key)
    tax = facts.loc[facts["field"] == "incomeTaxExpense", key + ["value"]].drop_duplicates(subset=key)
    if net.empty or tax.empty:
        return facts
    merged = net.merge(tax.rename(columns={"value": "_tax"}), on=key)
    if merged.empty:
        return facts

    nci = facts.loc[facts["field"] == "nciIncome", key + ["value"]].drop_duplicates(subset=key)
    if nci.empty:
        merged["_nci"] = 0.0
    else:
        merged = merged.merge(nci.rename(columns={"value": "_nci"}), on=key, how="left")
        parent_only = merged["source_tag"].astype(str).str.endswith("NetIncomeLoss")
        merged["_nci"] = merged["_nci"].fillna(0.0).where(parent_only, 0.0)

    covered = set(map(tuple, facts.loc[facts["field"] == "pretaxIncome", key].to_numpy()))
    merged = merged[~merged[key].apply(tuple, axis=1).isin(covered)]
    if merged.empty:
        return facts

    constant_cols = {c: facts[c].iloc[0] for c in ("ticker", "cik") if c in facts.columns}
    derived_rows = merged.assign(
        field="pretaxIncome", source_tag=None, derived=1.0,
        value=merged["value"] + merged["_tax"] + merged["_nci"],
        derived_from_accessions=merged["accession_number"], **constant_cols)
    # see `_as_derived_liabilities` for why this reindexes rather than subscripts
    return pd.concat([facts, derived_rows.reindex(columns=facts.columns)], ignore_index=True)


def derive_missing_total_liabilities(facts: pd.DataFrame) -> pd.DataFrame:
    """Fill `totalLiabilities` for the snapshots where the filer tagged no
    combined `us-gaap:Liabilities` total, trying each derivation in turn and
    only ever for a (fiscal_year, fiscal_period, accession_number) that is
    still uncovered -- an as-reported total, when present, is always preferred
    and never overridden, matching this pipeline's "as-filed beats derived"
    convention already used for Q4 (`decumulate_quarterly_flow`). The same
    precedence applies BETWEEN derivations: the current/noncurrent split is
    the filer's own subtotalling, so it outranks rearranging the footing.

    Operates on the FULL per-ticker frame (all fields already resolved,
    `field`/`duration_type` columns present) since this is a CROSS-field
    derivation, unlike the single-field `instant_stock`/`decumulate_quarterly_flow`.
    """
    if facts is None or facts.empty or "field" not in facts.columns:
        return facts
    for build in (_liabilities_from_current_split, _liabilities_from_footing):
        derived = build(facts)
        if derived is None or derived.empty:
            continue
        covered = set(map(tuple, facts.loc[facts["field"] == "totalLiabilities",
                                           _SNAPSHOT_KEY].to_numpy()))
        derived = derived[~derived[_SNAPSHOT_KEY].apply(tuple, axis=1).isin(covered)]
        if not derived.empty:
            facts = pd.concat([facts, derived], ignore_index=True)
    return facts
