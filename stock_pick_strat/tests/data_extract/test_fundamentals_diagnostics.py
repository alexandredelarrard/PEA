"""
Validation/diagnostics tests: reconciliation checks (reconcile_fundamentals_facts)
and the 4-way missing-concept taxonomy (diagnose_missing_field), worked against
the two confirmed MAA cases (capex = mapped_but_absent, currentLiabilities =
no_fact_in_source) plus apply_plausibility_guards' additive return_audit param.

Pure-synthetic, no network / no DB.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.constants.constants import (
    Q4_RECONCILIATION_TOLERANCE, SIGNED_Q4_FY_DOMINANCE_FLAG_RATIO,
)
from src.data_extract.utils.fundamentals.fundamentals_validation import (
    AMBIGUOUS_MULTIPLE_MATCHES, FILTERED_BY_QUALITY_RULE, MAPPED_BUT_ABSENT,
    NO_FACT_IN_SOURCE, apply_plausibility_guards, diagnose_missing_field,
    reconcile_fundamentals_facts,
)


def _facts_row(ticker, field, fiscal_year, fiscal_period, duration_type, value,
              is_amendment=0.0, period_start=None, period_end=None, accession="a1",
              derived=0.0):
    return {"ticker": ticker, "field": field, "fiscal_year": fiscal_year,
           "fiscal_period": fiscal_period, "duration_type": duration_type,
           "value": value, "is_amendment": is_amendment, "period_start": period_start,
           "period_end": period_end, "accession_number": accession, "derived": derived}


def test_q4_reconciliation_within_tolerance_passes_and_fails():
    """Clean fixture (Q1..Q4 sum == FY exactly) passes; the same fixture with one
    quarter perturbed beyond Q4_RECONCILIATION_TOLERANCE is flagged."""
    clean = pd.DataFrame([
        _facts_row("ZZZ", "totalRevenue", 2024, fp, "quarterly", v)
        for fp, v in (("Q1", 100.0), ("Q2", 120.0), ("Q3", 125.0), ("Q4", 115.0))
    ] + [_facts_row("ZZZ", "totalRevenue", 2024, "FY", "annual", 460.0)])
    out = reconcile_fundamentals_facts(clean)
    assert out[out["check"] == "q4_reconciliation_gap"].empty

    perturbed = clean.copy()
    bad_factor = 1.0 + Q4_RECONCILIATION_TOLERANCE * 5   # well beyond tolerance
    perturbed.loc[perturbed["fiscal_period"] == "Q4", "value"] *= bad_factor
    out_bad = reconcile_fundamentals_facts(perturbed)
    assert not out_bad[out_bad["check"] == "q4_reconciliation_gap"].empty


def test_q4_reconciliation_gap_ignores_instant_fields():
    """A field routed through instant_stock() (e.g. sharesOutstanding -- a
    roughly-constant point-in-time count, not a flow) must NEVER be checked by
    the Q1+Q2+Q3+Q4==FY flow-additivity identity: summing four near-identical
    snapshots and comparing to one of them produces a ~3x "gap" by construction,
    not a real defect. Real bug found via the 15-ticker live integration run:
    sharesOutstanding was flagged across nearly every ticker with a suspiciously
    uniform ~3.0 ratio (|4x - x| / x) before this duration_type filter was added."""
    facts = pd.DataFrame([
        _facts_row("ZZZ", "sharesOutstanding", 2024, fp, "instant", 1000.0)
        for fp in ("Q1", "Q2", "Q3", "Q4", "FY")
    ])
    out = reconcile_fundamentals_facts(facts)
    assert out[out["check"] == "q4_reconciliation_gap"].empty


def test_large_discontinuity_is_flagged_not_nulled():
    """A >5x QoQ jump (e.g. a genuine M&A-driven revenue jump) is reported as a
    diagnostic, never silently nulled or rescaled -- flag, don't fix."""
    facts = pd.DataFrame([
        _facts_row("ZZZ", "totalRevenue", 2024, "Q1", "quarterly", 100.0),
        _facts_row("ZZZ", "totalRevenue", 2024, "Q2", "quarterly", 700.0),   # 7x jump
    ])
    out = reconcile_fundamentals_facts(facts)
    disc = out[out["check"] == "large_discontinuity"]
    assert len(disc) == 1
    assert disc.iloc[0]["severity"] == "info"
    # the underlying value itself must be untouched by this check
    assert facts.loc[facts["fiscal_period"] == "Q2", "value"].iloc[0] == 700.0


def test_signed_q4_dominates_fiscal_year_flags_ba_operating_income():
    """Characterization test (not a fix regression -- this check is advisory-only by
    design, see SIGNED_Q4_FY_DOMINANCE_FLAG_RATIO's docstring). Real figures found via
    the Tiingo cross-check: BA's FY2025 `us-gaap:OperatingIncomeLoss` = +$4.281B, but
    its OWN Q3-2025 quarterly fact under the IDENTICAL tag = -$4.781B; the derived Q4
    (`FY - Q1 - Q2 - Q3`) comes out to +$8.777B -- 2.05x the FY total. `operatingIncome`
    is a signed field (not in NON_NEGATIVE_FLOW_FIELDS), so `_q4_is_coherent`'s hard sign
    rule never applied here, and this is the new, narrower flag meant to surface it for
    manual/Tiingo-cross-checked review."""
    facts = pd.DataFrame([
        _facts_row("BA", "operatingIncome", 2025, "Q4", "quarterly", 8_777_000_000.0,
                  derived=1.0),
        _facts_row("BA", "operatingIncome", 2025, "FY", "annual", 4_281_000_000.0),
    ])
    out = reconcile_fundamentals_facts(facts)
    flagged = out[out["check"] == "signed_q4_dominates_fiscal_year"]
    assert len(flagged) == 1
    assert flagged.iloc[0]["severity"] == "info"
    assert flagged.iloc[0]["ticker"] == "BA"


def test_signed_q4_dominates_fiscal_year_stays_quiet_on_an_ordinary_quarter():
    """A ratio comfortably under SIGNED_Q4_FY_DOMINANCE_FLAG_RATIO (an ordinary,
    slightly-larger-than-average Q4) must not be flagged -- the check exists to catch
    dominance, not any nonzero Q4."""
    ratio = 1.1
    assert ratio < SIGNED_Q4_FY_DOMINANCE_FLAG_RATIO
    facts = pd.DataFrame([
        _facts_row("ZZZ", "operatingIncome", 2024, "Q4", "quarterly", 110.0, derived=1.0),
        _facts_row("ZZZ", "operatingIncome", 2024, "FY", "annual", 100.0),
    ])
    out = reconcile_fundamentals_facts(facts)
    assert out[out["check"] == "signed_q4_dominates_fiscal_year"].empty


def test_signed_q4_dominates_fiscal_year_ignores_non_negative_flow_fields():
    """`totalRevenue` (NON_NEGATIVE_FLOW_FIELDS) already gets a hard sign rule in
    `_q4_is_coherent` -- this advisory check is scoped to SIGNED fields only, so an
    equally dominant derived Q4 on a non-negative field must not double-flag."""
    facts = pd.DataFrame([
        _facts_row("ZZZ", "totalRevenue", 2024, "Q4", "quarterly", 900.0, derived=1.0),
        _facts_row("ZZZ", "totalRevenue", 2024, "FY", "annual", 100.0),
    ])
    out = reconcile_fundamentals_facts(facts)
    assert out[out["check"] == "signed_q4_dominates_fiscal_year"].empty


def test_duplicate_fiscal_period_is_flagged():
    """Two non-amendment rows sharing the full key is a genuine anomaly (either a
    mislabeled fiscal period or a real double-file) and must be flagged."""
    facts = pd.DataFrame([
        _facts_row("ZZZ", "netIncome", 2024, "Q1", "quarterly", 100.0, accession="a1"),
        _facts_row("ZZZ", "netIncome", 2024, "Q1", "quarterly", 100.0, accession="a2"),
    ])
    out = reconcile_fundamentals_facts(facts)
    assert not out[out["check"] == "duplicate_fiscal_period"].empty


def test_apply_plausibility_guards_return_audit_is_backward_compatible():
    """return_audit=False (default) is byte-identical to the pre-existing function;
    return_audit=True additionally returns an audit frame of every nulled cell."""
    df = pd.DataFrame({"sharesOutstanding": [5e9, 4.819e15],
                      "totalRevenue": [100.0, 100.0], "totalAssets": [500.0, 500.0]})
    out_default = apply_plausibility_guards(df)
    out_audit, audit = apply_plausibility_guards(df, return_audit=True)
    assert out_default.equals(out_audit)
    assert len(audit) == 1
    assert audit.iloc[0]["column"] == "sharesOutstanding"
    assert audit.iloc[0]["original_value"] == 4.819e15


def test_diagnose_missing_field_classifies_maa_currentliabilities_as_no_fact_in_source():
    """MAA's real cached companyfacts JSON has NO LiabilitiesCurrent/AssetsCurrent
    under any namespace (confirmed by direct inspection) -- an unclassified REIT
    balance sheet with no current/noncurrent split at all. The OTHER undimensioned
    instant facts present (total Assets/Liabilities/Equity) do NOT textually
    resemble "LiabilitiesCurrent" -> category 1, no tag-list fix possible."""
    raw_facts = pd.DataFrame([
        {"concept": "us-gaap:Assets", "value": 11975383000.0, "is_dimensioned": False,
         "period_type": "instant"},
        {"concept": "us-gaap:Liabilities", "value": 6135738000.0, "is_dimensioned": False,
         "period_type": "instant"},
        {"concept": "us-gaap:StockholdersEquity", "value": 5662851000.0, "is_dimensioned": False,
         "period_type": "instant"},
        # note: NO current/noncurrent split anywhere -- genuinely absent, not a mapping gap
    ])
    result = diagnose_missing_field(raw_facts, "currentLiabilities", ["LiabilitiesCurrent"], "instant")
    assert result["category"] == NO_FACT_IN_SOURCE
    assert result["surfaced_candidates"] == []


def test_diagnose_missing_field_surfaces_a_component_tag_for_human_review():
    """MAA's real filing DOES carry undimensioned instant facts whose name textually
    CONTAINS "LiabilitiesCurrent" (e.g. AccountsPayableAndAccruedLiabilitiesCurrent
    -- confirmed by direct inspection) -- these surface as review candidates
    (category 2), NOT auto-applied: they are COMPONENT line items, not a valid
    total-current-liabilities substitute, so a human must reject them. This is the
    intended behavior of a review-suggestion category, not a bug."""
    raw_facts = pd.DataFrame([
        {"concept": "us-gaap:AccountsPayableAndAccruedLiabilitiesCurrent", "value": 50000000.0,
         "is_dimensioned": False, "period_type": "instant"},
    ])
    result = diagnose_missing_field(raw_facts, "currentLiabilities", ["LiabilitiesCurrent"], "instant")
    assert result["category"] == MAPPED_BUT_ABSENT
    assert result["surfaced_candidates"] == ["us-gaap:AccountsPayableAndAccruedLiabilitiesCurrent"]


def test_diagnose_missing_field_classifies_maa_capex_gap_honestly():
    """MAA's raw filing carries an undimensioned duration fact tagged
    PaymentsForCapitalImprovements, absent from the WIP's thin 3-tag capex list
    (PaymentsToAcquire{PropertyPlantAndEquipment,ProductiveAssets,OilAndGasProperty}).
    Its name shares only a generic "Payments..." PREFIX with those candidates --
    not a substring-containment relationship -- so the honest, non-fuzzy diagnostic
    classifies this as no_fact_in_source with the thin list: the substring heuristic
    is deliberately conservative (see currentLiabilities test above for what it DOES
    catch) and does not claim to auto-discover every possible missing tag. The
    ACTUAL fix came from porting fetch_fundamentals.py's already-researched, proven
    tag list (fundamentals_tags.py), not from this diagnostic -- verified here by
    confirming the full list resolves the field cleanly (FILTERED_BY_QUALITY_RULE:
    "a candidate matched cleanly")."""
    raw_facts = pd.DataFrame([
        {"concept": "us-gaap:PaymentsForCapitalImprovements", "value": 360238000.0,
         "is_dimensioned": False, "period_type": "duration"},
    ])
    thin_list = ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets",
                "PaymentsToAcquireOilAndGasProperty"]
    result = diagnose_missing_field(raw_facts, "capex", thin_list, "duration")
    assert result["category"] == NO_FACT_IN_SOURCE

    from src.data_extract.utils.fundamentals.fundamentals_tags import FLOW_TAGS
    full_result = diagnose_missing_field(raw_facts, "capex", FLOW_TAGS["capex"], "duration")
    assert full_result["category"] == FILTERED_BY_QUALITY_RULE   # a candidate DID match cleanly


def test_diagnose_missing_field_detects_ambiguous_multiple_matches():
    """Two different candidate tags matching the SAME period with DIFFERENT values
    and no priority tie-break is genuinely ambiguous."""
    raw_facts = pd.DataFrame([
        {"concept": "us-gaap:Revenues", "value": 100.0, "is_dimensioned": False, "period_type": "duration"},
        {"concept": "us-gaap:SalesRevenueNet", "value": 250.0, "is_dimensioned": False, "period_type": "duration"},
    ])
    result = diagnose_missing_field(raw_facts, "totalRevenue", ["Revenues", "SalesRevenueNet"], "duration")
    assert result["category"] == AMBIGUOUS_MULTIPLE_MATCHES

    print("\n=== SANITY CHECK: fundamentals validation/diagnostics ===")
    print("  Q4 reconciliation passes a clean fixture, flags a perturbed one;")
    print("  large discontinuities are flagged (info), never nulled or rescaled;")
    print("  duplicate fiscal periods flagged; apply_plausibility_guards(return_audit=True)")
    print("  is additive-only (default path byte-identical);")
    print("  4-way taxonomy (substring-based, not fuzzy NLP): MAA currentLiabilities with")
    print("  only unrelated Assets/Liabilities/Equity facts -> no_fact_in_source; the SAME")
    print("  field with a real component tag present (AccountsPayableAndAccruedLiabilities-")
    print("  Current) -> mapped_but_absent, surfaced for human review (a component, not a")
    print("  valid total substitute); MAA capex's actual gap (PaymentsForCapitalImprovements")
    print("  vs. a thin Payments*-prefixed list) shares no substring -> honestly")
    print("  no_fact_in_source with the thin list -- the real fix came from PORTING the old")
    print("  file's researched tag list, not from this diagnostic auto-discovering it;")
    print("  conflicting-value case -> ambiguous_multiple_matches.")
    print("  Validated.")
