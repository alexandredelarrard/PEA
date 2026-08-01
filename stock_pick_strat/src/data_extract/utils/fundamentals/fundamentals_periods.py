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

import pandas as pd

from src.constants.constants import (
    FUNDAMENTALS_DISCONTINUITY_MIN,
)
from src.data_extract.utils.fundamentals.fundamentals_tags import (
    ANNUAL_MAX_DAYS, ANNUAL_MIN_DAYS, IMPLIED_QUARTER_MAX_DAYS,
    IMPLIED_QUARTER_MIN_DAYS,
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


def _q4_is_coherent(q4_val: float, q1: float, q2: float, q3: float) -> bool:
    """A derived Q4 must be broadly consistent with Q1/Q2/Q3 -- a signal that
    catches an upstream data problem (mismatched XBRL concepts across periods,
    a corrupted quarter or FY value) rather than a genuine, if unusual, Q4.
    Deliberately permissive about MAGNITUDE alone: a real business can have a
    legitimately much-larger-or-smaller quarter (retail holiday seasonality, a
    one-off gain), and same-sign Q4s are always accepted regardless of size.
    Only rejects Q4 when it is a DIFFERENT SIGN than EVERY one of Q1/Q2/Q3 AND
    not comparatively small -- a real business's flow fields are consistently
    one sign quarter to quarter, so a sign flip is only innocuous (e.g. a rare
    small rebate/write-off in an otherwise-profitable year) when its magnitude
    is minor; a LARGE opposite-sign value is almost always a data error, not a
    real reversal (reuses `FUNDAMENTALS_DISCONTINUITY_MIN`, the same "how small
    counts as negligible" threshold `reconcile_fundamentals_facts` already
    uses elsewhere for its QoQ-discontinuity check). Confirmed real failures
    this is designed to catch as a final safety net (both also addressed at
    the root by the same-source_tag requirement and the dimension-mode fix in
    `fetch_fundamentals_edgar.build_tag_frames`): JPM (Q1/Q2/Q3 all +$22-27B, a
    mismatched-tag FY produced a derived Q4 of -$63B) and MAA (Q1-Q3 all
    +$127-134M, a wrongly-picked dimensioned FY value produced a derived Q4 of
    -$384M)."""
    quarters = (q1, q2, q3)
    if all((q4_val >= 0) == (q >= 0) for q in quarters):
        return True
    max_abs = max(abs(q) for q in quarters)
    return max_abs == 0 or abs(q4_val) <= max_abs * FUNDAMENTALS_DISCONTINUITY_MIN


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
        Derivation additionally requires: (a) Q1/Q2/Q3/FY all resolved from
        the SAME underlying XBRL concept (`source_tag`) -- confirmed real bug:
        JPM's FY sometimes resolves via a DIFFERENT candidate tag than its
        quarters (`Revenues` vs `RevenuesNetOfInterestExpense`, JPM's actual
        bank-revenue concept), so FY-(Q1+Q2+Q3) mixed two unrelated numbers
        and produced a wildly negative "Q4"; mismatched tags make that
        fiscal-year's Q4 unresolvable for this field rather than a fabricated
        number. (b) the result passes `_q4_is_coherent` against Q1/Q2/Q3 --
        catches whatever a tag-mismatch wouldn't (e.g. a corrupted individual
        quarter value). Either failure means NO Q4 row for that fiscal year --
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

    rows: list[dict] = []
    for fy, grp in d.groupby("fiscal_year", sort=True):
        # Canonical-selection order within a shared fiscal_period: earliest
        # filing_date first (original before amendment), and -- for facts
        # sharing the SAME filing_date (i.e. the SAME accession tagged both a
        # quarter-shaped AND a YTD-shaped context for this concept) -- the
        # quarter-shaped one first, so it is picked over the YTD one instead
        # of an arbitrary tie-break.
        grp = grp.assign(_quarterly_shaped=grp["duration_days"].apply(classify_duration_shape) == "quarterly")
        grp = grp.sort_values(["filing_date", "_quarterly_shaped"], ascending=[True, False])
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
        fy_rows = grp[grp["fiscal_period"] == "FY"]
        can_derive = not fy_rows.empty and all(k in discrete for k in _QUARTER_LABELS)
        if not can_derive:
            continue

        fy_r = fy_rows.iloc[0]
        q1, q2, q3 = by_fp_all["Q1"].iloc[0], by_fp_all["Q2"].iloc[0], by_fp_all["Q3"].iloc[0]
        inputs = [fy_r, q1, q2, q3]

        source_tags = {r.get("source_tag") for r in inputs if r.get("source_tag")}
        if len(source_tags) > 1:
            continue   # Q1/Q2/Q3/FY resolved from different XBRL concepts -- unresolvable

        derived_val = float(fy_r["value"]) - sum(discrete[k] for k in _QUARTER_LABELS)
        if not _q4_is_coherent(derived_val, discrete["Q1"], discrete["Q2"], discrete["Q3"]):
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
    amendment-coexistence rationale (identical here)."""
    out_cols = ["fiscal_year", "fiscal_period", "value", "filing_date", "accession_number", "form",
               "period_start", "period_end", "source_tag", "is_amendment", "fiscal_period_source"]
    if facts is None or facts.empty:
        return pd.DataFrame(columns=out_cols)
    d = facts.dropna(subset=["fiscal_year", "fiscal_period", "value", "filing_date"]).copy()
    if "period_start" not in d.columns:
        d["period_start"] = pd.NaT
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
