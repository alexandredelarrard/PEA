"""
fundamentals_validation.py
---------------------------
Plausibility guards (relocated verbatim from `fetch_fundamentals.py`, single
source of truth for both extraction paths) + the new reconciliation /
missing-concept diagnostics for the edgartools-based `fundamentals_facts`
pipeline (`fetch_fundamentals_edgar.py` / `fundamentals_derive.py`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.constants.constants import (
    BALANCE_SHEET_IDENTITY_TOLERANCE, BALANCE_SHEET_MIN_ASSETS_TO_REVENUE,
    DEBT_TO_EQUITY_ABS_MAX, DILUTED_SHARES_MIN_SHARE_OF_BASIC,
    DIVIDEND_PER_SHARE_ABS_MAX, EFFECTIVE_TAX_RATE_MAX, EFFECTIVE_TAX_RATE_MIN,
    EPS_ABS_MAX, FUNDAMENTALS_DISCONTINUITY_MAX, FUNDAMENTALS_DISCONTINUITY_MIN,
    OPERATING_MARGIN_ABS_MAX, PROFIT_MARGIN_ABS_MAX, Q4_RECONCILIATION_TOLERANCE,
    RATIO_DENOMINATOR_MIN_FRACTION, RETURN_ON_EQUITY_ABS_MAX,
    SHARES_OUTSTANDING_MAX, SHARES_OUTSTANDING_MIN,
)


def _null_where(frame: pd.DataFrame, column: str, bad: pd.Series) -> int:
    """NULL `column` on the `bad` rows; returns how many were cleared. NULL, never
    clipped: a clipped value still asserts a magnitude we know to be wrong, and the
    downstream winsorizers would then treat the boundary as a real observation."""
    if column not in frame.columns:
        return 0
    mask = bad.reindex(frame.index, fill_value=False).fillna(False) & frame[column].notna()
    n = int(mask.sum())
    if n:
        frame.loc[mask, column] = np.nan
    return n


def apply_plausibility_guards(
    out: pd.DataFrame,
    return_audit: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Null values that are accounting-impossible or arithmetic artifacts.

    `grossMargins` already had such a guard (GROSS_MARGIN_MIN/MAX) and was consequently
    the only clean ratio in the table; this gives its siblings the same treatment and
    adds the scale checks the 2026-07 audit surfaced. Measured on the live table
    (30,133 rows / 498 tickers), the defects this clears are:

      * SCALE, both directions -- `sharesOutstanding` 370 rows outside 1e6..2e10
        (ORCL 2012 stored 4.819e15 against a true 4.819e9, exactly 1e6x; 166 zeros),
        and the mirror image on the balance sheet where a stub / registration-era
        filing reports an internally consistent but 1e6x-too-small statement
        (LUV 2011 totalAssets 1.788e4 for a real $17.88bn, KMB 1.9e4, SW 108).
      * WRONG TAG in a per-share field -- `epsDiluted` 21 rows |eps| > 500 (ICE 2016
        captured the diluted SHARE COUNT, 1.2e8, as EPS), `dividendsPerShare` 19 rows
        > 100 (ROK 3.88e6, STX 2.8e6 = the dollar dividend TOTAL).
      * NEAR-ZERO DENOMINATOR -- returnOnEquity reached 5.52e7, debtToEquity 9.69e7,
        operatingMargins -209, profitMargins -148.7. The inputs are fine; the quotient
        is not, so the RATIO is nulled while its numerator/denominator are kept.

    Pure and idempotent, so it is unit-testable and safe to re-apply on a rebuild.

    `return_audit=False` (default): behavior is IDENTICAL to before this parameter
    existed -- same content/dtype returned, in the same order, for every existing
    call site/test. `return_audit=True` additionally returns a `(row_index, column,
    rule, original_value)` audit frame of every value actually nulled, feeding
    `_diagnose_missing_field`'s "filtered_by_quality_rule" category without
    re-deriving the guard logic.
    """
    audit_cols = ["row_index", "column", "rule", "original_value"]
    if out is None or out.empty:
        empty_audit = pd.DataFrame(columns=audit_cols)
        return (out, empty_audit) if return_audit else out
    out = out.copy()
    audit_rows: list[dict] = []

    def num(name: str) -> pd.Series:
        if name not in out.columns:
            return pd.Series(np.nan, index=out.index, dtype="float64")
        return pd.to_numeric(out[name], errors="coerce")

    def guard(column: str, bad: pd.Series, rule: str) -> int:
        if return_audit and column in out.columns:
            mask = bad.reindex(out.index, fill_value=False).fillna(False) & out[column].notna()
            audit_rows.extend(
                {"row_index": i, "column": column, "rule": rule, "original_value": v}
                for i, v in out.loc[mask, column].items())
        return _null_where(out, column, bad)

    revenue, assets = num("totalRevenue"), num("totalAssets")
    liabilities, equity = num("totalLiabilities"), num("stockholdersEquity")
    scale = revenue.abs().where(revenue.notna(), assets.abs())

    # ---- 1. share count / per-share fields ------------------------------------
    shares = num("sharesOutstanding")
    guard("sharesOutstanding",
          (shares < SHARES_OUTSTANDING_MIN) | (shares > SHARES_OUTSTANDING_MAX),
          "shares_outstanding_scale")
    guard("commonSharesIssued",
          (num("commonSharesIssued") < SHARES_OUTSTANDING_MIN)
          | (num("commonSharesIssued") > SHARES_OUTSTANDING_MAX),
          "shares_issued_scale")
    for col in ("commonSharesAuthorized", "preferredSharesAuthorized", "antidilutiveShares"):
        guard(col, num(col) > SHARES_OUTSTANDING_MAX * 10, "shares_authorized_scale")
    authorized = num("commonSharesAuthorized")
    guard("commonSharesAuthorized",
          (authorized <= 0) | (authorized < num("sharesOutstanding")),
          "shares_authorized_below_outstanding")
    basic, diluted = num("basicShares"), num("dilutedShares")
    guard("dilutedShares",
          (diluted <= 0)
          | (basic.notna() & (diluted < basic * DILUTED_SHARES_MIN_SHARE_OF_BASIC)),
          "diluted_below_basic")
    guard("effectiveTaxRate",
          (num("effectiveTaxRate") < EFFECTIVE_TAX_RATE_MIN)
          | (num("effectiveTaxRate") > EFFECTIVE_TAX_RATE_MAX),
          "effective_tax_rate_band")
    guard("epsDiluted", num("epsDiluted").abs() > EPS_ABS_MAX, "eps_abs_max")
    guard("epsBasic", num("epsBasic").abs() > EPS_ABS_MAX, "eps_abs_max")
    guard("dividendsPerShare",
          num("dividendsPerShare").abs() > DIVIDEND_PER_SHARE_ABS_MAX,
          "dividends_per_share_abs_max")

    # ---- 2. wrongly-scaled balance sheet --------------------------------------
    bad_scale = (revenue.notna() & assets.notna() & (revenue.abs() > 0)
                 & (assets.abs() < revenue.abs() * BALANCE_SHEET_MIN_ASSETS_TO_REVENUE))
    nci = num("minorityInterest").fillna(0.0) + num("redeemableNCI").fillna(0.0)
    denom = assets.abs().replace(0, np.nan)
    gap_ex_nci = (assets - (liabilities + equity)).abs() / denom
    gap_inc_nci = (assets - (liabilities + equity + nci)).abs() / denom
    identity_gap = pd.concat([gap_ex_nci, gap_inc_nci], axis=1).min(axis=1)
    bad_identity = (assets.notna() & liabilities.notna() & equity.notna()
                    & (identity_gap > BALANCE_SHEET_IDENTITY_TOLERANCE))
    bad_bs = bad_scale | bad_identity | (assets <= 0)
    for c in ("totalAssets", "totalLiabilities", "stockholdersEquity"):
        guard(c, bad_bs, "balance_sheet_scale_or_identity")

    # ---- 3. ratios whose denominator is too small to divide by ----------------
    thin_rev = revenue.abs() < scale * RATIO_DENOMINATOR_MIN_FRACTION
    thin_eq = equity.abs() < scale * RATIO_DENOMINATOR_MIN_FRACTION
    for column, denom_thin, cap in (
        ("returnOnEquity", thin_eq, RETURN_ON_EQUITY_ABS_MAX),
        ("debtToEquity", thin_eq, DEBT_TO_EQUITY_ABS_MAX),
        ("operatingMargins", thin_rev, OPERATING_MARGIN_ABS_MAX),
        ("profitMargins", thin_rev, PROFIT_MARGIN_ABS_MAX),
    ):
        guard(column, denom_thin | (num(column).abs() > cap), "thin_denominator_or_band")

    # ---- 4. quantities that cannot be negative --------------------------------
    for column in ("totalRevenue", "cash", "inventory", "goodwill", "ppeNet",
                   "currentAssets", "totalLiabilities", "totalDebt"):
        guard(column, num(column) < 0, "cannot_be_negative")

    if return_audit:
        return out, pd.DataFrame(audit_rows, columns=audit_cols)
    return out


# --------------------------------------------------------------------------- #
# Reconciliation checks over fundamentals_facts (flag-only, never auto-fix)   #
# --------------------------------------------------------------------------- #
def reconcile_fundamentals_facts(facts: pd.DataFrame) -> pd.DataFrame:
    """Flag-only diagnostics over a `fundamentals_facts`-shaped frame (columns:
    ticker, field, fiscal_year, fiscal_period, duration_type, value, is_amendment,
    derived, accession_number, filing_date). Never nulls or rescales a value --
    mirrors `apply_plausibility_guards`' philosophy of surfacing, not silently
    correcting. Returns one row per (ticker, field, fiscal_year, fiscal_period,
    check, detail, severity)."""
    cols = ["ticker", "field", "fiscal_year", "fiscal_period", "check", "detail", "severity"]
    if facts is None or facts.empty:
        return pd.DataFrame(columns=cols)

    rows: list[dict] = []
    key = ["ticker", "field", "fiscal_year", "fiscal_period", "duration_type"]

    # 1. duplicate_fiscal_period: two non-amendment rows share the full key.
    originals = facts[facts["is_amendment"] != 1.0] if "is_amendment" in facts.columns else facts
    dupe_mask = originals.duplicated(subset=key, keep=False)
    for _, r in originals[dupe_mask].iterrows():
        rows.append({"ticker": r["ticker"], "field": r["field"], "fiscal_year": r["fiscal_year"],
                     "fiscal_period": r["fiscal_period"], "check": "duplicate_fiscal_period",
                     "detail": f"accession={r.get('accession_number')}", "severity": "warning"})

    # 2. mismatched_fiscal_year_inputs: Q1/Q2/Q3 feeding a derived Q4 don't share fiscal_year
    #    (already enforced structurally by fundamentals_periods.decumulate_quarterly_flow's
    #    per-fiscal_year grouping -- this check catches any UPSTREAM label disagreement that
    #    slipped through, e.g. two facts resolved to different fiscal_year for the same
    #    accession+field+period_end).
    if {"fiscal_year", "period_end"}.issubset(facts.columns):
        by_end = facts.dropna(subset=["period_end"]).groupby(
            ["ticker", "field", "period_end"])["fiscal_year"].nunique()
        for (ticker, field, period_end), n in by_end[by_end > 1].items():
            rows.append({"ticker": ticker, "field": field, "fiscal_year": None,
                         "fiscal_period": None, "check": "mismatched_fiscal_year_inputs",
                         "detail": f"period_end={period_end} resolved to {n} different fiscal years",
                         "severity": "error"})

    # 3. inconsistent_duration_selection: a STOCK-family concept resolved as anything but
    #    'instant', or a 'annual'-labeled row whose actual span isn't ~annual.
    if {"duration_type", "period_start", "period_end"}.issubset(facts.columns):
        span_days = (pd.to_datetime(facts["period_end"]) - pd.to_datetime(facts["period_start"])).dt.days
        bad_annual = (facts["duration_type"] == "annual") & span_days.notna() & ~span_days.between(340, 380)
        for _, r in facts[bad_annual].iterrows():
            rows.append({"ticker": r["ticker"], "field": r["field"], "fiscal_year": r["fiscal_year"],
                         "fiscal_period": r["fiscal_period"], "check": "inconsistent_duration_selection",
                         "detail": f"labeled annual but span={(pd.to_datetime(r['period_end']) - pd.to_datetime(r['period_start'])).days}d",
                         "severity": "warning"})

    # 4. q4_reconciliation_gap: Q1+Q2+Q3+Q4 vs FY, within Q4_RECONCILIATION_TOLERANCE.
    # FLOW fields only (duration_type in {'quarterly','annual'}) -- this is a flow-
    # additivity identity (four quarters of activity sum to the year's activity) that is
    # meaningless for an INSTANT/level field (a balance-sheet snapshot, or a roughly-constant
    # share count) repeated once per quarter: summing four near-identical snapshots and
    # comparing to one of them produces a ~3x "gap" by construction, not a real defect.
    # Confirmed against real data: `sharesOutstanding` (an INSTANT_FIELD_TAGS field, routed
    # through instant_stock() not decumulate_quarterly_flow()) was flagged across nearly
    # every ticker in the 15-ticker integration run with a suspiciously uniform ~3.0 ratio --
    # exactly |4x - x| / x, i.e. this check running on a field it was never meant to cover.
    if {"fiscal_period", "value", "duration_type"}.issubset(facts.columns):
        flow_only = originals[originals["duration_type"].isin(["quarterly", "annual"])]
        piv = (flow_only[flow_only["fiscal_period"].isin(["Q1", "Q2", "Q3", "Q4", "FY"])]
               .pivot_table(index=["ticker", "field", "fiscal_year"],
                             columns="fiscal_period", values="value", aggfunc="first"))
        for col in ("Q1", "Q2", "Q3", "Q4", "FY"):
            if col not in piv.columns:
                piv[col] = np.nan
        have_all = piv[["Q1", "Q2", "Q3", "Q4", "FY"]].notna().all(axis=1)
        qsum = piv[["Q1", "Q2", "Q3", "Q4"]].sum(axis=1)
        rel_gap = (qsum - piv["FY"]).abs() / piv["FY"].abs().replace(0, np.nan)
        bad = have_all & (rel_gap > Q4_RECONCILIATION_TOLERANCE)
        for (ticker, field, fy), is_bad in bad.items():
            if is_bad:
                rows.append({"ticker": ticker, "field": field, "fiscal_year": fy,
                             "fiscal_period": "FY", "check": "q4_reconciliation_gap",
                             "detail": f"|sum(Q1..Q4) - FY| / |FY| = {rel_gap.loc[(ticker, field, fy)]:.4f}"
                                       f" > {Q4_RECONCILIATION_TOLERANCE}",
                             "severity": "error"})

    # 5. large_discontinuity: flag, never fix, a QoQ move outside a wide band.
    if {"fiscal_period", "value"}.issubset(facts.columns):
        q = originals[originals["fiscal_period"].isin(["Q1", "Q2", "Q3", "Q4"])].copy()
        if not q.empty:
            q = q.sort_values(["ticker", "field", "fiscal_year", "fiscal_period"])
            q["_ratio"] = q.groupby(["ticker", "field"])["value"].transform(
                lambda s: s / s.shift(1))
            disc = q["_ratio"].notna() & (
                (q["_ratio"] > FUNDAMENTALS_DISCONTINUITY_MAX)
                | (q["_ratio"] < FUNDAMENTALS_DISCONTINUITY_MIN))
            for _, r in q[disc].iterrows():
                rows.append({"ticker": r["ticker"], "field": r["field"], "fiscal_year": r["fiscal_year"],
                             "fiscal_period": r["fiscal_period"], "check": "large_discontinuity",
                             "detail": f"QoQ ratio={r['_ratio']:.3f} outside "
                                       f"[{FUNDAMENTALS_DISCONTINUITY_MIN}, {FUNDAMENTALS_DISCONTINUITY_MAX}]",
                             "severity": "info"})

    return pd.DataFrame(rows, columns=cols)


# --------------------------------------------------------------------------- #
# Missing-concept diagnostics: 4-way taxonomy                                 #
# --------------------------------------------------------------------------- #
# Categories `_diagnose_missing_field` classifies a gap into:
NO_FACT_IN_SOURCE = "no_fact_in_source"
MAPPED_BUT_ABSENT = "mapped_but_absent"
FILTERED_BY_QUALITY_RULE = "filtered_by_quality_rule"
AMBIGUOUS_MULTIPLE_MATCHES = "ambiguous_multiple_matches"


def diagnose_missing_field(
    raw_facts: pd.DataFrame,
    field: str,
    candidates: list[str],
    period_type: str,
    guard_audit: pd.DataFrame | None = None,
) -> dict:
    """Classify why `field` has no resolved value for the filing(s) `raw_facts`
    represents, using the 4-way taxonomy:

      1. no_fact_in_source        -- none of `candidates`, AND no OTHER undimensioned
                                      fact of the matching `period_type` ('instant' for
                                      a STOCK field, 'duration' for a FLOW field) TEXTUALLY
                                      RESEMBLES one of the candidate tag names, exists
                                      anywhere in `raw_facts`. Genuinely absent; no
                                      tag-list fix is possible.
      2. mapped_but_absent        -- none of the TRIED candidates match, but some OTHER
                                      undimensioned fact of the right period_type textually
                                      CONTAINS a candidate tag name as a substring (e.g. MAA
                                      `capex` -- the thin WIP list omitted
                                      `PaymentsForCapitalImprovements`, which IS present and
                                      undimensioned). Deliberately a targeted substring
                                      search, not "any other fact exists" (near-always true
                                      on a real 10-K) or fuzzy NLP similarity -- the surfaced
                                      concept is a REVIEW SUGGESTION, not an auto-applied
                                      fix: a component line item can textually resemble a
                                      candidate (e.g. `AccountsPayableAndAccruedLiabilities-
                                      Current` resembles `LiabilitiesCurrent`) while being
                                      economically wrong to substitute (a component of
                                      current liabilities, not the total) -- a human must
                                      still judge whether it is a genuine like-for-like fix.
      3. filtered_by_quality_rule -- a candidate DOES match, but only dimensioned (no
                                      undimensioned version), wrong duration bucket, or
                                      nulled by `apply_plausibility_guards` (checked via
                                      `guard_audit`, the second return value of
                                      `apply_plausibility_guards(..., return_audit=True)`).
      4. ambiguous_multiple_matches -- >1 candidate reports a DIFFERENT value for the
                                      exact same period with no priority tie-break.

    `raw_facts` columns expected: concept (str, e.g. 'us-gaap:Revenues'), value,
    is_dimensioned (bool), period_type ('instant'|'duration').

    Returns {"category": ..., "surfaced_candidates": [...], "detail": ...}.
    """
    if guard_audit is not None and not guard_audit.empty and field in set(guard_audit["column"]):
        return {"category": FILTERED_BY_QUALITY_RULE, "surfaced_candidates": [],
               "detail": "nulled by apply_plausibility_guards"}

    if raw_facts is None or raw_facts.empty:
        return {"category": NO_FACT_IN_SOURCE, "surfaced_candidates": [],
               "detail": "no raw facts available for this filing"}

    candidate_set = {c.lower() for c in candidates}
    concept_names = raw_facts["concept"].astype(str)
    bare = concept_names.str.split(":").str[-1].str.lower()
    matched = raw_facts[bare.isin(candidate_set)]

    undimensioned_matched = matched[matched["is_dimensioned"] == False]  # noqa: E712
    if len(undimensioned_matched) > 1:
        vals = undimensioned_matched["value"].astype(float).round(2).unique()
        if len(vals) > 1:
            return {"category": AMBIGUOUS_MULTIPLE_MATCHES,
                   "surfaced_candidates": sorted(undimensioned_matched["concept"].unique().tolist()),
                   "detail": f"{len(vals)} distinct values across matched candidates: {list(vals)}"}

    if not matched.empty and undimensioned_matched.empty:
        return {"category": FILTERED_BY_QUALITY_RULE, "surfaced_candidates": [],
               "detail": "candidate concept(s) present only as dimensioned (segment/member) facts"}

    if not undimensioned_matched.empty:
        # A candidate DID resolve cleanly -- this function is only called for a
        # confirmed gap, so an unexpectedly-successful match here means the caller's
        # field really did resolve; nothing to diagnose.
        return {"category": FILTERED_BY_QUALITY_RULE, "surfaced_candidates": [],
               "detail": "a candidate matched cleanly; check period/duration filtering upstream"}

    # No candidate matched at all -- look for another undimensioned fact of the right
    # period_type whose bare name TEXTUALLY CONTAINS one of our candidate tag names
    # as a contiguous substring (e.g. "LiabilitiesCurrent" inside
    # "AccountsPayableAndAccruedLiabilitiesCurrent") -- deliberately NOT "any other
    # undimensioned fact of the same period_type exists", which is almost always true
    # on a real 10-K (hundreds of instant facts) and would make this category
    # meaningless. This is exactly how the MAA capex/currentLiabilities cases were
    # actually diagnosed (a targeted substring search), not fuzzy NLP similarity --
    # the surfaced candidate is a REVIEW SUGGESTION, not an auto-applied fix: a
    # component line item (e.g. `AccountsPayableAndAccruedLiabilitiesCurrent`) can
    # textually resemble a candidate (`LiabilitiesCurrent`) while being economically
    # wrong to substitute (a component, not the total) -- a human must still judge
    # whether the surfaced concept is a genuine like-for-like fix.
    other = raw_facts[(raw_facts["is_dimensioned"] == False) & (raw_facts["period_type"] == period_type)]  # noqa: E712
    other = other[~bare.reindex(other.index).isin(candidate_set)]
    other_bare = other["concept"].astype(str).str.split(":").str[-1]
    resembles = other_bare.apply(lambda n: any(c in n for c in candidates))
    surfaced = sorted(other.loc[resembles, "concept"].unique().tolist())
    if surfaced:
        return {"category": MAPPED_BUT_ABSENT, "surfaced_candidates": surfaced,
               "detail": f"{len(surfaced)} other undimensioned {period_type} fact(s) textually "
                         f"resembling a candidate tag are present -- review before adding "
                         f"(may be a component, not a valid total substitute)"}
    return {"category": NO_FACT_IN_SOURCE, "surfaced_candidates": [],
           "detail": f"no undimensioned {period_type} fact resembling {field} anywhere in the filing"}
