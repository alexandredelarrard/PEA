"""
Unit tests for the edgartools-based fundamentals retrieval adapter
(fetch_fundamentals_edgar.py) and its dimension-aware tag coalescing.

Pure-synthetic, no network / no DB (except the idempotency test, which uses a
throwaway SQLite DataStore, mirroring
tests/data_extract/test_sec_bulk_datasets.py::test_insider_incremental_state_converges).
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

import pandas as pd
import pytest
from sqlalchemy import create_engine

from src.data_extract.utils.fundamentals.fetch_fundamentals_edgar import (
    _cover_page_shares_fallback, backfill_fiscal_period_from_filing, build_tag_frames,
)
from src.data_extract.utils.fundamentals.fundamentals_periods import decumulate_quarterly_flow
from src.data_extract.utils.fundamentals.fundamentals_tags import SHARES_TAGS
from src.data_store.store import DataStore

_CLASS_AXIS = "us-gaap:StatementClassOfStockAxis"
_CLASS_AXIS_COL = "dim_us-gaap_StatementClassOfStockAxis"


def _fact(concept, value, period_start, period_end, period_type, fiscal_year, fiscal_period,
         is_dimensioned=False, unit_ref="U_USD", numeric_value=None, period_instant=None):
    return {
        "concept": concept, "value": str(value), "numeric_value": numeric_value if numeric_value is not None else float(value),
        "unit_ref": unit_ref, "period_type": period_type, "period_start": period_start,
        "period_end": period_end, "period_instant": period_instant,
        "fiscal_year": fiscal_year, "fiscal_period": fiscal_period,
        "is_dimensioned": is_dimensioned,
    }


def _class_fact(concept, value, member, date, *, second_axis=None):
    """An INSTANT share-count fact broken out by share class, shaped like edgartools'
    `XBRL.facts.to_dataframe()` output: one `dim_<prefix>_<Axis>` column per axis plus
    the flat `dimension`/`member` pair (which holds only the FIRST axis)."""
    row = _fact(concept, value, None, None, "instant", None, None,
               is_dimensioned=True, period_instant=date)
    row |= {"dimension": _CLASS_AXIS, "member": member, _CLASS_AXIS_COL: member}
    if second_axis is not None:
        row[f"dim_{second_axis}"] = "member-x"
    return row


def test_build_tag_frames_excludes_dimensioned_duplicate():
    """A concept reported BOTH consolidated (is_dimensioned=False) and per-segment
    (is_dimensioned=True) for the same period must resolve to only the consolidated
    value -- replaces the WIP's `iloc[-1]` (non-deterministic, could pick the segment)."""
    facts_df = pd.DataFrame([
        _fact("us-gaap:NetIncomeLoss", 100.0, "2024-01-01", "2024-03-31", "duration", 2024, "Q1",
             is_dimensioned=False),
        _fact("us-gaap:NetIncomeLoss", 40.0, "2024-01-01", "2024-03-31", "duration", 2024, "Q1",
             is_dimensioned=True),   # segment slice -- must be excluded
    ])
    out = build_tag_frames(facts_df, {"netIncome": ["NetIncomeLoss"]})
    assert len(out) == 1
    assert out.iloc[0]["value"] == 100.0


def test_build_tag_frames_admits_dimensioned_fact_when_all_members_agree():
    """A concept tagged ONLY under a dimension, where every dimensioned instance for
    that (concept, period) reports the IDENTICAL value, must still be admitted as
    "the total" -- confirmed empirically against real MAA filings: a dual-registrant
    10-Q/10-K (REIT + its operating partnership) tags consolidated `Revenues` once
    per dei:LegalEntityAxis member (ParentCompanyMember / LimitedPartnerMember),
    BOTH reporting the same figure, with NO undimensioned duplicate at all for
    2014-2019 -- the WIP's blanket is_dimensioned==False filter dropped 5 fiscal
    years of revenue entirely."""
    facts_df = pd.DataFrame([
        _fact("us-gaap:Revenues", 272236000.0, "2016-04-01", "2016-06-30", "duration", 2016, "Q2",
             is_dimensioned=True),   # Parent Company member
        _fact("us-gaap:Revenues", 272236000.0, "2016-04-01", "2016-06-30", "duration", 2016, "Q2",
             is_dimensioned=True),   # Limited Partner member -- same figure
    ])
    out = build_tag_frames(facts_df, {"totalRevenue": ["Revenues"]})
    assert len(out) == 1
    assert out.iloc[0]["value"] == 272236000.0


def test_build_tag_frames_prefers_parent_company_member_on_dual_registrant_disagreement():
    """A dual-registrant filer (REIT + its operating partnership) can tag a concept
    ONLY under dei:LegalEntityAxis with GENUINELY DIFFERENT values per registrant --
    confirmed empirically: MAA's FY2013 10-K tags `PaymentsForCapitalImprovements`
    exclusively dimensioned that year, Parent Company $53.439M vs Limited Partner
    $53.357M, no undimensioned duplicate. The single-distinct-value admission rule
    (previous test) correctly excludes genuine per-member disagreement, but that
    silently dropped the WHOLE period here. Since the ticker universe is always the
    parent entity (the publicly-traded security, never its LP), the member whose
    label identifies it as the parent must be preferred over dropping both."""
    facts_df = pd.DataFrame([
        _fact("us-gaap:PaymentsForCapitalImprovements", 53439000.0, "2013-01-01", "2013-12-31",
             "duration", 2013, "FY", is_dimensioned=True),
        _fact("us-gaap:PaymentsForCapitalImprovements", 53357000.0, "2013-01-01", "2013-12-31",
             "duration", 2013, "FY", is_dimensioned=True),
    ])
    facts_df["dimension"] = ["dei:LegalEntityAxis", "dei:LegalEntityAxis"]
    facts_df["dimension_member_label"] = ["Parent Company", "Limited Partner"]
    out = build_tag_frames(facts_df, {"capex": ["PaymentsForCapitalImprovements"]})
    assert len(out) == 1
    assert out.iloc[0]["value"] == 53439000.0


def test_build_tag_frames_recognizes_consolidated_entities_axis_as_a_parent_identifying_axis():
    """Real bug found via live data: MAA's `dividendsPaid` (PaymentsOfDividends-
    CommonStock) is tagged dimensioned under `us-gaap:ConsolidatedEntitiesAxis`
    (member label "Parent Company"), NOT `dei:LegalEntityAxis` -- a different axis
    name for the same parent-vs-combining-entity concept. With only
    `dei:LegalEntityAxis` recognized, this fact had no undimensioned duplicate and
    no repeats (sole candidate), so it was excluded entirely: FY2015-2017
    dividendsPaid vanished from MAA's FY2018 10-K filing. The parent-identifying
    axis check must recognize `us-gaap:ConsolidatedEntitiesAxis` (and
    `srt:ConsolidatedEntitiesAxis`) exactly like `dei:LegalEntityAxis`."""
    facts_df = pd.DataFrame([
        _fact("us-gaap:PaymentsOfDividendsCommonStock", 395294000.0, "2017-01-01", "2017-12-31",
             "duration", 2017, "FY", is_dimensioned=True),
    ])
    facts_df["dimension"] = ["us-gaap:ConsolidatedEntitiesAxis"]
    facts_df["dimension_member_label"] = ["Parent Company"]
    out = build_tag_frames(facts_df, {"dividendsPaid": ["PaymentsOfDividendsCommonStock"]})
    assert len(out) == 1
    assert out.iloc[0]["value"] == 395294000.0


def test_build_tag_frames_prefers_the_repeated_value_over_other_parent_labeled_slices():
    """Real bug found via live data: MAA's FY2014 10-K tags `Assets` with SIX
    dimensioned rows for one (concept, period) -- $6.83B appears TWICE (Limited
    Partner + Parent Company, the true consolidated total) while FOUR OTHER
    rows, from a hidden SECOND axis our generic dimension columns don't
    capture, are ALSO labeled "Parent Company" ($1.25B, $4.51B, $1.00B,
    $66.6M -- components of a consolidating breakdown). The prior version's
    "prefer the Parent-labeled member" rule admitted all five Parent-labeled
    rows and a later `keep='last'` reduction picked one ARBITRARILY -- which
    is how $66.6M got stored as if it were total assets. The value with the
    HIGHEST REPEAT COUNT ($6.83B, appearing twice) must be preferred over any
    single-occurrence "Parent Company"-labeled value, even though there are
    MULTIPLE rows sharing that label."""
    facts_df = pd.DataFrame([
        _fact("us-gaap:Assets", 6831028000.0, None, None, "instant", 2014, "FY",
             is_dimensioned=True, period_instant="2014-12-31"),   # Limited Partner
        _fact("us-gaap:Assets", 6831028000.0, None, None, "instant", 2014, "FY",
             is_dimensioned=True, period_instant="2014-12-31"),   # Parent Company -- agrees
        _fact("us-gaap:Assets", 1253995000.0, None, None, "instant", 2014, "FY",
             is_dimensioned=True, period_instant="2014-12-31"),   # Parent Company -- component
        _fact("us-gaap:Assets", 4506951000.0, None, None, "instant", 2014, "FY",
             is_dimensioned=True, period_instant="2014-12-31"),   # Parent Company -- component
        _fact("us-gaap:Assets", 1003426000.0, None, None, "instant", 2014, "FY",
             is_dimensioned=True, period_instant="2014-12-31"),   # Parent Company -- component
        _fact("us-gaap:Assets", 66656000.0, None, None, "instant", 2014, "FY",
             is_dimensioned=True, period_instant="2014-12-31"),   # Parent Company -- component
    ])
    facts_df["dimension"] = ["dei:LegalEntityAxis"] * 6
    facts_df["dimension_member_label"] = (
        ["Limited Partner", "Parent Company"] + ["Parent Company"] * 4)
    out = build_tag_frames(facts_df, {"totalAssets": ["Assets"]})
    assert len(out) == 1
    assert out.iloc[0]["value"] == 6831028000.0


def test_build_tag_frames_undimensioned_total_wins_even_if_a_dimensioned_zero_repeats_more():
    """Real bug found via live data: ADP's FY2019 Q1 10-Q tags
    `RevenueFromContractWithCustomerExcludingAssessedTax` with ONE undimensioned
    fact ($3.3232B, the true quarterly revenue) PLUS a `ProductOrServiceAxis`/
    `StatementBusinessSegmentsAxis` product/segment breakdown where SEVERAL
    lines are legitimately $0 that quarter (a discontinued/immaterial product
    line) -- $0 ends up the most-repeated DIMENSIONED value (more repeats than
    any real, non-zero product figure). The repeat-count rule alone would
    prefer $0 over the true total -- it must never even be CONSULTED when an
    undimensioned fact already answers the question; the undimensioned total
    wins outright, full stop, regardless of what any dimensioned value repeats."""
    facts_df = pd.DataFrame([
        _fact("us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax", 3323200000.0,
             "2018-07-01", "2018-09-30", "duration", 2019, "Q1", is_dimensioned=False),
        _fact("us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax", 118500000.0,
             "2018-07-01", "2018-09-30", "duration", 2019, "Q1", is_dimensioned=True),
        _fact("us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax", 0.0,
             "2018-07-01", "2018-09-30", "duration", 2019, "Q1", is_dimensioned=True),
        _fact("us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax", 473500000.0,
             "2018-07-01", "2018-09-30", "duration", 2019, "Q1", is_dimensioned=True),
        _fact("us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax", 0.0,
             "2018-07-01", "2018-09-30", "duration", 2019, "Q1", is_dimensioned=True),
        _fact("us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax", 0.0,
             "2018-07-01", "2018-09-30", "duration", 2019, "Q1", is_dimensioned=True),
        _fact("us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax", 1520300000.0,
             "2018-07-01", "2018-09-30", "duration", 2019, "Q1", is_dimensioned=True),
        _fact("us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax", 0.0,
             "2018-07-01", "2018-09-30", "duration", 2019, "Q1", is_dimensioned=True),
        _fact("us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax", 653400000.0,
             "2018-07-01", "2018-09-30", "duration", 2019, "Q1", is_dimensioned=True),
        _fact("us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax", 0.0,
             "2018-07-01", "2018-09-30", "duration", 2019, "Q1", is_dimensioned=True),
    ])
    out = build_tag_frames(facts_df, {"totalRevenue": ["RevenueFromContractWithCustomerExcludingAssessedTax"]})
    assert len(out) == 1
    assert out.iloc[0]["value"] == 3323200000.0


def test_build_tag_frames_prefers_a_lower_priority_undimensioned_fact_over_a_shaky_repeat():
    """Real bug found via live data: ABBV's Q1-2019 10-Q tags the priority-0
    candidate `RevenueFromContractWithCustomerExcludingAssessedTax` with ~30
    purely-DIMENSIONED product/geography facts (no undimensioned duplicate for
    THIS tag at all) where TWO of them coincidentally share the same value
    ($25M out of dozens of otherwise-distinct slices) -- a repeat count of 2 is
    enough to pass the tag-LOCAL modal-repeat guard (`test_build_tag_frames_
    admits_dimensioned_fact_when_all_members_agree`'s rule), which has no
    visibility into the fact that a DIFFERENT, lower-priority candidate
    (`Revenues`) has a clean UNDIMENSIONED fact for the exact same period --
    the true total ($7.828B). The field-level override must prefer that
    undimensioned value over the higher-priority tag's shaky dimensioned
    admission, regardless of raw tag priority."""
    facts_df = pd.DataFrame([
        _fact("us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax", 848000000.0,
             "2019-01-01", "2019-03-31", "duration", 2019, "Q1", is_dimensioned=True),
        _fact("us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax", 340000000.0,
             "2019-01-01", "2019-03-31", "duration", 2019, "Q1", is_dimensioned=True),
        _fact("us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax", 25000000.0,
             "2019-01-01", "2019-03-31", "duration", 2019, "Q1", is_dimensioned=True),
        _fact("us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax", 25000000.0,
             "2019-01-01", "2019-03-31", "duration", 2019, "Q1", is_dimensioned=True),
        _fact("us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax", 1022000000.0,
             "2019-01-01", "2019-03-31", "duration", 2019, "Q1", is_dimensioned=True),
        _fact("us-gaap:Revenues", 7828000000.0,
             "2019-01-01", "2019-03-31", "duration", 2019, "Q1", is_dimensioned=False),
    ])
    tag_map = {"totalRevenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues"]}
    out = build_tag_frames(facts_df, tag_map)
    assert len(out) == 1
    assert out.iloc[0]["value"] == 7828000000.0
    assert out.iloc[0]["source_tag"] == "us-gaap:Revenues"


def test_build_tag_frames_rejects_a_negative_balance_sheet_magnitude():
    """Real bug found via live data (the case `NON_NEGATIVE_STOCK_FIELDS` was added
    for): DTE's FY2012 10-K tags the priority-0 `shortTermDebt` candidate
    `us-gaap:DebtCurrent` with -$634M, because it uses that concept for the "Less
    amount due within one year" DEDUCTION row of its long-term-debt footnote and
    baked the presentation sign into the instance document. Both that fact and the
    real balance-sheet `ShortTermBorrowings` ($240M) are UNDIMENSIONED, so no
    dimension rule could reject it and raw tag priority handed a NEGATIVE short-term
    debt to the panel. A debt balance cannot be negative, so the fact is inadmissible
    and the coalesce must fall through to the next candidate."""
    facts_df = pd.DataFrame([
        _fact("us-gaap:DebtCurrent", -634000000.0, None, None, "instant", 2012, "FY",
             period_instant="2012-12-31"),
        _fact("us-gaap:ShortTermBorrowings", 240000000.0, None, None, "instant", 2012, "FY",
             period_instant="2012-12-31"),
    ])
    tag_map = {"shortTermDebt": ["DebtCurrent", "LongTermDebtCurrent", "ShortTermBorrowings"]}
    out = build_tag_frames(facts_df, tag_map)
    assert len(out) == 1
    assert out.iloc[0]["value"] == 240000000.0
    assert out.iloc[0]["source_tag"] == "us-gaap:ShortTermBorrowings"


def test_build_tag_frames_leaves_a_negative_stock_field_null_with_no_other_candidate():
    """The rejection must NULL the field rather than fall back to `abs()`: flipping the
    sign would rewrite the filer's own number on a guess. DTE's FY2011 10-K is the live
    case for the fall-through (a `ShortTermBorrowings` sibling exists); this pins the
    no-sibling case, where NULL is the only honest answer."""
    facts_df = pd.DataFrame([
        _fact("us-gaap:DebtCurrent", -355000000.0, None, None, "instant", 2011, "FY",
             period_instant="2011-12-31"),
    ])
    out = build_tag_frames(facts_df, {"shortTermDebt": ["DebtCurrent"]})
    assert out.empty


def test_build_tag_frames_rejects_an_implausibly_small_share_count():
    """Real bug found via the Tiingo cross-check: MCD's FY2024 10-Qs tag
    `us-gaap:WeightedAverageNumberOfSharesOutstandingBasic`/`...Diluted` as `721.8`/
    `725.9` where the true counts are `721,800,000`/`725,900,000` -- a 1,000,000x scale
    defect baked into the raw XBRL instance (edgartools' `numeric_value` is never
    rescaled by decimals/scale, confirmed against the installed package). No S&P/Dow
    constituent has a weighted-average share count under a million, so with no other
    candidate reporting the period, NULL is the only honest answer -- never `abs()` or a
    blind x1e6 rescale."""
    facts_df = pd.DataFrame([
        _fact("us-gaap:WeightedAverageNumberOfSharesOutstandingBasic", 721.8,
             "2024-01-01", "2024-03-31", "duration", 2024, "Q1"),
    ])
    out = build_tag_frames(facts_df, {"basicShares": ["WeightedAverageNumberOfSharesOutstandingBasic"]})
    assert out.empty


def test_build_tag_frames_falls_through_to_a_correctly_scaled_share_count_candidate():
    """The rejection must fall through the coalesce like any other inadmissible fact --
    a genuine, correctly-scaled sibling candidate for the same period must still win."""
    facts_df = pd.DataFrame([
        _fact("us-gaap:WeightedAverageNumberOfSharesOutstandingBasic", 721.8,
             "2024-01-01", "2024-03-31", "duration", 2024, "Q1"),
        _fact("us-gaap:WeightedAverageNumberOfShareOutstandingBasicAndDiluted", 721_800_000.0,
             "2024-01-01", "2024-03-31", "duration", 2024, "Q1"),
    ])
    tag_map = {"basicShares": ["WeightedAverageNumberOfSharesOutstandingBasic",
                               "WeightedAverageNumberOfShareOutstandingBasicAndDiluted"]}
    out = build_tag_frames(facts_df, tag_map)
    assert len(out) == 1
    assert out.iloc[0]["value"] == 721_800_000.0


def test_build_tag_frames_share_count_magnitude_guard_is_scoped_to_its_own_fields():
    """A small value is only implausible for a SHARE-COUNT-shaped field. `epsDiluted`
    legitimately sits well under the 1,000,000 floor and must survive untouched, or the
    guard would delete good data far outside its intended scope."""
    facts_df = pd.DataFrame([
        _fact("us-gaap:EarningsPerShareDiluted", 2.02, "2024-01-01", "2024-03-31",
             "duration", 2024, "Q1"),
    ])
    out = build_tag_frames(facts_df, {"epsDiluted": ["EarningsPerShareDiluted"]})
    assert len(out) == 1
    assert out.iloc[0]["value"] == 2.02


def test_build_tag_frames_denies_a_concept_the_listed_filer_misuses():
    """`FIELD_TAG_DENYLIST` is the per-issuer escape hatch for a defect no global rule
    can express. DTE tags `us-gaap:DebtCurrent` on the "Less amount due within one year"
    row of its long-term-debt FOOTNOTE, and from FY2013 that value is POSITIVE ($694M) --
    so the sign guard cannot see it, even though DTE's own balance sheet reports $131M
    short-term borrowings and $898M current-portion for the same date. Denying the concept
    for this ticker makes the coalesce continue to the balance-sheet line."""
    facts_df = pd.DataFrame([
        _fact("us-gaap:DebtCurrent", 694000000.0, None, None, "instant", 2013, "FY",
             period_instant="2013-12-31"),
        _fact("us-gaap:ShortTermBorrowings", 131000000.0, None, None, "instant", 2013, "FY",
             period_instant="2013-12-31"),
    ])
    tag_map = {"shortTermDebt": ["DebtCurrent", "LongTermDebtCurrent", "ShortTermBorrowings"]}
    out = build_tag_frames(facts_df, tag_map, ticker="DTE")
    assert len(out) == 1
    assert out.iloc[0]["value"] == 131000000.0
    assert out.iloc[0]["source_tag"] == "us-gaap:ShortTermBorrowings"


def test_build_tag_frames_denylist_is_scoped_to_its_own_ticker():
    """Deny, never pin, and never global: an unlisted ticker -- and the `ticker=None`
    default every other caller and test uses -- must resolve exactly as before, so the
    escape hatch can never silently change resolution for the other 499 names."""
    facts_df = pd.DataFrame([
        _fact("us-gaap:DebtCurrent", 694000000.0, None, None, "instant", 2013, "FY",
             period_instant="2013-12-31"),
        _fact("us-gaap:ShortTermBorrowings", 131000000.0, None, None, "instant", 2013, "FY",
             period_instant="2013-12-31"),
    ])
    tag_map = {"shortTermDebt": ["DebtCurrent", "LongTermDebtCurrent", "ShortTermBorrowings"]}
    for ticker in (None, "AEP"):
        out = build_tag_frames(facts_df, tag_map, ticker=ticker)
        assert len(out) == 1
        assert out.iloc[0]["source_tag"] == "us-gaap:DebtCurrent", ticker


def test_build_tag_frames_denies_aep_costofrevenue_misuse():
    """Real bug found via `fundamentals_tag_ledger`: AEP tags the priority-0
    `costOfRevenue` candidate `us-gaap:CostOfGoodsAndServicesSold` with $0 to
    -$223M every quarter FY2018-FY2023 -- impossible for a utility with
    ~$17-19B/year of revenue and a fuel/purchased-power cost line that
    dominates it. The ledger scores AEP's FY2024 cutover to
    `CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization` (its
    real, billions-scale cost line) at a 38,331x pooled-level jump unique to
    AEP -- not a taxonomy migration, AEP's own mis-tagging. Denying the bad
    tag must fall through to the correct excl-D&A candidate."""
    facts_df = pd.DataFrame([
        _fact("us-gaap:CostOfGoodsAndServicesSold", -172000000.0,
             "2021-01-01", "2021-03-31", "duration", 2021, "Q1"),
        _fact("us-gaap:CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization",
             1852900000.0, "2025-01-01", "2025-03-31", "duration", 2025, "Q1"),
    ])
    tag_map = {"costOfRevenue": ["CostOfGoodsAndServicesSold",
                                 "CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization"]}
    out = build_tag_frames(facts_df, tag_map, ticker="AEP")
    assert len(out) == 1
    assert out.iloc[0]["value"] == 1852900000.0
    assert out.iloc[0]["source_tag"] == "us-gaap:CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization"


def test_build_tag_frames_sums_sga_companion_when_ga_only_wins():
    """Real gap found via the Tiingo cross-check: CRM (Salesforce) tags ONLY
    `GeneralAndAdministrativeExpense` (a component) and `SellingAndMarketingExpense`
    (a genuinely additive companion) for the same period, never the combined
    `SellingGeneralAndAdministrativeExpense` concept. The priority coalesce picks G&A
    alone and would otherwise understate sellingGeneralAdmin by ~5x -- Tiingo's own
    normalized figure matched G&A + S&M summed to the dollar."""
    facts_df = pd.DataFrame([
        _fact("us-gaap:GeneralAndAdministrativeExpense", 711000000.0,
             "2024-08-01", "2024-10-31", "duration", 2025, "Q3"),
        _fact("us-gaap:SellingAndMarketingExpense", 3224000000.0,
             "2024-08-01", "2024-10-31", "duration", 2025, "Q3"),
    ])
    tag_map = {"sellingGeneralAdmin": ["SellingGeneralAndAdministrativeExpense",
                                       "GeneralAndAdministrativeExpense",
                                       "SellingAndMarketingExpense"]}
    out = build_tag_frames(facts_df, tag_map)
    assert len(out) == 1
    assert out.iloc[0]["value"] == 711000000.0 + 3224000000.0
    assert out.iloc[0]["source_tag"] == "us-gaap:GeneralAndAdministrativeExpense"


def test_build_tag_frames_sga_ga_only_untouched_with_no_companion():
    """No `SellingAndMarketingExpense` fact for the period -> G&A stands alone,
    unmodified (nothing to add)."""
    facts_df = pd.DataFrame([
        _fact("us-gaap:GeneralAndAdministrativeExpense", 711000000.0,
             "2024-08-01", "2024-10-31", "duration", 2025, "Q3"),
    ])
    tag_map = {"sellingGeneralAdmin": ["SellingGeneralAndAdministrativeExpense",
                                       "GeneralAndAdministrativeExpense",
                                       "SellingAndMarketingExpense"]}
    out = build_tag_frames(facts_df, tag_map)
    assert len(out) == 1
    assert out.iloc[0]["value"] == 711000000.0


def test_build_tag_frames_sga_combined_tag_never_gets_companion_added():
    """When the filer tags the COMBINED `SellingGeneralAndAdministrativeExpense`
    concept, it wins outright by priority and must NEVER be summed with a
    companion -- that would double-count a filer who also separately discloses a
    Selling-and-marketing sub-line within its combined SG&A."""
    facts_df = pd.DataFrame([
        _fact("us-gaap:SellingGeneralAndAdministrativeExpense", 4000000000.0,
             "2024-08-01", "2024-10-31", "duration", 2025, "Q3"),
        _fact("us-gaap:SellingAndMarketingExpense", 3224000000.0,
             "2024-08-01", "2024-10-31", "duration", 2025, "Q3"),
    ])
    tag_map = {"sellingGeneralAdmin": ["SellingGeneralAndAdministrativeExpense",
                                       "GeneralAndAdministrativeExpense",
                                       "SellingAndMarketingExpense"]}
    out = build_tag_frames(facts_df, tag_map)
    assert len(out) == 1
    assert out.iloc[0]["value"] == 4000000000.0
    assert out.iloc[0]["source_tag"] == "us-gaap:SellingGeneralAndAdministrativeExpense"


def test_build_tag_frames_sga_companion_summing_is_scoped_to_its_own_field():
    """The G&A-only-plus-companion rule must never touch any OTHER field that
    happens to resolve `GeneralAndAdministrativeExpense` for some other purpose --
    scoped strictly to `field == 'sellingGeneralAdmin'`."""
    facts_df = pd.DataFrame([
        _fact("us-gaap:GeneralAndAdministrativeExpense", 711000000.0,
             "2024-08-01", "2024-10-31", "duration", 2025, "Q3"),
        _fact("us-gaap:SellingAndMarketingExpense", 3224000000.0,
             "2024-08-01", "2024-10-31", "duration", 2025, "Q3"),
    ])
    tag_map = {"someOtherField": ["GeneralAndAdministrativeExpense"]}
    out = build_tag_frames(facts_df, tag_map)
    assert len(out) == 1
    assert out.iloc[0]["value"] == 711000000.0


def test_build_tag_frames_denies_cat_costofrevenue_misuse():
    """Real bug found via the Tiingo cross-check: CAT tags `us-gaap:CostOfRevenue`
    correctly and consistently every quarter from 2018 on (~$9-11B/quarter, matching
    its own ~$40B/year revenue), but FY2024/FY2025 it ALSO tags the priority-0
    `costOfRevenue` candidate `us-gaap:CostOfGoodsAndServicesSold` at $33M -- a ~300x
    understatement, both undimensioned so no dimension rule catches it. Denying the
    bad tag for CAT must fall through to the correct, and far larger, CostOfRevenue."""
    facts_df = pd.DataFrame([
        _fact("us-gaap:CostOfGoodsAndServicesSold", 33000000.0,
             "2024-01-01", "2024-12-31", "duration", 2024, "FY"),
        _fact("us-gaap:CostOfRevenue", 41485000000.0,
             "2024-01-01", "2024-12-31", "duration", 2024, "FY"),
    ])
    tag_map = {"costOfRevenue": ["CostOfGoodsAndServicesSold", "CostOfRevenue"]}
    out = build_tag_frames(facts_df, tag_map, ticker="CAT")
    assert len(out) == 1
    assert out.iloc[0]["value"] == 41485000000.0
    assert out.iloc[0]["source_tag"] == "us-gaap:CostOfRevenue"


def test_build_tag_frames_denies_mcd_depamort_misuse():
    """Real bug found via the Tiingo cross-check, same shape as the CAT entry: MCD
    tags BOTH `us-gaap:DepreciationDepletionAndAmortization` (priority-0, $99M -- a
    small, wrong figure repeating quarter to quarter) AND
    `us-gaap:DepreciationAndAmortization` (priority-2, $510M) undimensioned, for the
    same period. Confirmed against Tiingo's `depamor`: $510M, matching the
    lower-priority tag exactly. Denying the small tag for MCD must fall through to
    the correct, much larger D&A figure."""
    facts_df = pd.DataFrame([
        _fact("us-gaap:DepreciationDepletionAndAmortization", 99000000.0,
             "2024-01-01", "2024-03-31", "duration", 2024, "Q1"),
        _fact("us-gaap:DepreciationAndAmortization", 510000000.0,
             "2024-01-01", "2024-03-31", "duration", 2024, "Q1"),
    ])
    tag_map = {"depAmort": ["DepreciationDepletionAndAmortization",
                            "DepreciationAmortizationAndAccretionNet",
                            "DepreciationAndAmortization"]}
    out = build_tag_frames(facts_df, tag_map, ticker="MCD")
    assert len(out) == 1
    assert out.iloc[0]["value"] == 510000000.0
    assert out.iloc[0]["source_tag"] == "us-gaap:DepreciationAndAmortization"


def test_build_tag_frames_keeps_a_negative_value_on_a_genuinely_signed_field():
    """The guard is scoped to `NON_NEGATIVE_STOCK_FIELDS` only. A negative value is a
    REAL business fact for a signed balance-sheet field -- a buyback-driven equity
    deficit, a contra-account -- and must survive untouched, or the guard would delete
    good data (the over-strictness that the `NON_NEGATIVE_FLOW_FIELDS` pass fixed)."""
    facts_df = pd.DataFrame([
        _fact("us-gaap:StockholdersEquity", -1500000000.0, None, None, "instant", 2024, "FY",
             period_instant="2024-12-31"),
    ])
    out = build_tag_frames(facts_df, {"stockholdersEquity": ["StockholdersEquity"]})
    assert len(out) == 1
    assert out.iloc[0]["value"] == -1500000000.0


def test_build_tag_frames_excludes_partial_revenue_concept_when_companion_present():
    """Real bug found via live data: ADM splits total revenue between an
    ASC-606 in-scope concept (`RevenueFromContractWithCustomerExcludingAssessed
    Tax`, $24.956B) and an ASC-606 out-of-scope companion
    (`RevenueNotFromContractWithCustomer`, e.g. commodity trading/merchandising
    revenue, $55.313B) -- BOTH tagged undimensioned, BOTH genuinely correct,
    but each only PART of total revenue; they sum exactly to the filer's own
    `Revenues` total ($80.269B). Since both the priority-0 candidate and
    `Revenues` are undimensioned, the plain field-level-undimensioned override
    alone cannot tell them apart -- the companion's presence must specifically
    exclude the partial concept so the true whole-company total wins."""
    facts_df = pd.DataFrame([
        _fact("us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax", 24956000000.0,
             "2025-01-01", "2025-12-31", "duration", 2025, "FY", is_dimensioned=False),
        _fact("us-gaap:RevenueNotFromContractWithCustomer", 55313000000.0,
             "2025-01-01", "2025-12-31", "duration", 2025, "FY", is_dimensioned=False),
        _fact("us-gaap:Revenues", 80269000000.0,
             "2025-01-01", "2025-12-31", "duration", 2025, "FY", is_dimensioned=False),
    ])
    tag_map = {"totalRevenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues"]}
    out = build_tag_frames(facts_df, tag_map)
    assert len(out) == 1
    assert out.iloc[0]["value"] == 80269000000.0
    assert out.iloc[0]["source_tag"] == "us-gaap:Revenues"


def test_build_tag_frames_excludes_dimensioned_facts_when_members_disagree():
    """A true multi-value business-segment slice (each member reporting a DIFFERENT
    number) must stay excluded -- no single member there represents the whole
    company, and admitting one at random would silently understate the total. This
    is the key safety property of the single-distinct-value admission rule: it only
    ever admits a number that is UNAMBIGUOUSLY the same as every other tagged
    variant, never a guess at which member is "the total"."""
    facts_df = pd.DataFrame([
        _fact("us-gaap:Revenues", 100.0, "2024-01-01", "2024-03-31", "duration", 2024, "Q1",
             is_dimensioned=True),   # segment A
        _fact("us-gaap:Revenues", 250.0, "2024-01-01", "2024-03-31", "duration", 2024, "Q1",
             is_dimensioned=True),   # segment B -- genuinely different value
    ])
    out = build_tag_frames(facts_df, {"totalRevenue": ["Revenues"]})
    assert out.empty


def test_build_tag_frames_sums_cover_page_share_classes_into_the_company_total():
    """A MULTI-CLASS filer tags NO undimensioned share count anywhere: every fact for
    both candidate tags carries a `StatementClassOfStockAxis` member and the classes
    report different numbers, so all of them are (correctly) refused as "the total" and
    `sharesOutstanding` came out NULL -- 36 of 498 tickers were >=60% NULL on exactly
    this. Real META Q2-2026 figures: cover page Class A 2,205,128,509 + Class B
    342,377,716, balance sheet 2,206,000,000 + 342,000,000 (rounded to millions).
    The cover page's per-class enumeration is exhaustive by SEC requirement, so its sum
    is the company total."""
    facts_df = pd.DataFrame([
        _class_fact("us-gaap:CommonStockSharesOutstanding", 2_206_000_000.0,
                   "us-gaap:CommonClassAMember", "2026-06-30"),
        _class_fact("us-gaap:CommonStockSharesOutstanding", 342_000_000.0,
                   "us-gaap:CommonClassBMember", "2026-06-30"),
        _class_fact("dei:EntityCommonStockSharesOutstanding", 2_205_128_509.0,
                   "us-gaap:CommonClassAMember", "2026-07-24"),
        _class_fact("dei:EntityCommonStockSharesOutstanding", 342_377_716.0,
                   "us-gaap:CommonClassBMember", "2026-07-24"),
    ])
    out = build_tag_frames(facts_df, SHARES_TAGS)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["value"] == 2_205_128_509.0 + 342_377_716.0
    assert row["source_tag"] == "dei:EntityCommonStockSharesOutstanding"
    assert pd.Timestamp(row["period_end"]) == pd.Timestamp("2026-07-24")


def test_cover_page_class_total_reaches_the_filings_own_period():
    """End-to-end contract with `_cover_page_shares_fallback`: the class sum is dated at
    the COVER date (a few weeks after the period it reports on), so the current-period
    filter drops it and only the fallback makes it reachable -- the same route the
    single-class filers already take, re-stamped onto the period of report."""
    facts_df = pd.DataFrame([
        _class_fact("dei:EntityCommonStockSharesOutstanding", 101_432_177.0,
                   "us-gaap:CommonClassAMember", "2026-07-17"),
        _class_fact("dei:EntityCommonStockSharesOutstanding", 749_349_405.0,
                   "us-gaap:CommonClassBMember", "2026-07-17"),
    ])
    tagged = build_tag_frames(facts_df, SHARES_TAGS)
    por = pd.Timestamp("2026-06-30")
    current = tagged[pd.to_datetime(tagged["period_end"]).dt.normalize() == por]
    assert current.empty                       # cover date is AFTER the period of report
    out = _cover_page_shares_fallback(tagged, current, por)
    assert len(out) == 1
    assert out.iloc[0]["value"] == 101_432_177.0 + 749_349_405.0   # real UPS Q2-2026
    assert pd.Timestamp(out.iloc[0]["period_end"]) == por


def test_build_tag_frames_never_admits_one_share_class_as_the_company_total():
    """Confirmed on CME: each per-class balance-sheet fact is tagged TWICE (class axis
    alone, then class + `StatementEquityComponentsAxis`), so BOTH Class A (359,275,000)
    and Class B (3,000) reach a repeat count of 2 and pass the modal-repeat rule -- which
    reads redundancy, not meaning. `drop_duplicates(keep="last")` then decided between
    them on frame order alone. A class-dimensioned share count is a COMPONENT and must
    never be admitted as the total, so the balance-sheet tag yields nothing here and the
    cover-page sum (359,576,125 + 3,138) answers instead."""
    facts_df = pd.DataFrame([
        _class_fact("us-gaap:CommonStockSharesOutstanding", 359_275_000.0,
                   "us-gaap:CommonClassAMember", "2026-06-30"),
        _class_fact("us-gaap:CommonStockSharesOutstanding", 359_275_000.0,
                   "us-gaap:CommonClassAMember", "2026-06-30",
                   second_axis="us-gaap_StatementEquityComponentsAxis"),
        _class_fact("us-gaap:CommonStockSharesOutstanding", 3_000.0,
                   "us-gaap:CommonClassBMember", "2026-06-30"),
        _class_fact("us-gaap:CommonStockSharesOutstanding", 3_000.0,
                   "us-gaap:CommonClassBMember", "2026-06-30",
                   second_axis="us-gaap_StatementEquityComponentsAxis"),
        _class_fact("dei:EntityCommonStockSharesOutstanding", 359_576_125.0,
                   "us-gaap:CommonClassAMember", "2026-07-08"),
        _class_fact("dei:EntityCommonStockSharesOutstanding", 3_138.0,
                   "cme:ClassBCommonStockClassB1Member", "2026-07-08"),
    ])
    out = build_tag_frames(facts_df, SHARES_TAGS)
    assert len(out) == 1
    assert out.iloc[0]["value"] == 359_576_125.0 + 3_138.0
    assert pd.Timestamp(out.iloc[0]["period_end"]) == pd.Timestamp("2026-07-08")


def test_build_tag_frames_class_sum_counts_a_twice_tagged_class_once():
    """The same class re-tagged under a SECOND axis must not be added twice. Only facts
    whose ONLY dimension is the class axis are summed, and the flat `dimension` column
    cannot see the second axis (edgartools stores only the FIRST) -- so the `dim_*`
    columns are what makes this safe."""
    facts_df = pd.DataFrame([
        _class_fact("dei:EntityCommonStockSharesOutstanding", 100_000_000.0,
                   "us-gaap:CommonClassAMember", "2026-07-24"),
        _class_fact("dei:EntityCommonStockSharesOutstanding", 100_000_000.0,
                   "us-gaap:CommonClassAMember", "2026-07-24",
                   second_axis="us-gaap_StatementEquityComponentsAxis"),
        _class_fact("dei:EntityCommonStockSharesOutstanding", 40_000_000.0,
                   "us-gaap:CommonClassBMember", "2026-07-24"),
    ])
    out = build_tag_frames(facts_df, SHARES_TAGS)
    assert len(out) == 1
    assert out.iloc[0]["value"] == 140_000_000.0


def test_build_tag_frames_class_sum_takes_a_roll_up_member_alone():
    """A member whose value equals the sum of all the others is a ROLL-UP of them, not a
    sibling to add -- confirmed on V (`CommonClassB1B2AndB3` beside its own B-1/B-2/B-3)
    and BRK-B (`EquivalentClassA`). Adding it would overstate the count by ~2x."""
    facts_df = pd.DataFrame([
        _class_fact("dei:EntityCommonStockSharesOutstanding", 100_000_000.0,
                   "us-gaap:CommonClassAMember", "2026-07-21"),
        _class_fact("dei:EntityCommonStockSharesOutstanding", 50_000_000.0,
                   "us-gaap:CommonClassBMember", "2026-07-21"),
        _class_fact("dei:EntityCommonStockSharesOutstanding", 150_000_000.0,
                   "v:CommonClassAAndBMember", "2026-07-21"),
    ])
    out = build_tag_frames(facts_df, SHARES_TAGS)
    assert len(out) == 1
    assert out.iloc[0]["value"] == 150_000_000.0


def test_build_tag_frames_class_sum_keeps_two_equally_sized_classes():
    """Two classes of EQUAL size each satisfy "my value is the sum of the others", so the
    roll-up rule must only fire when EXACTLY ONE member does -- otherwise a legitimate
    50/50 split would be halved."""
    facts_df = pd.DataFrame([
        _class_fact("dei:EntityCommonStockSharesOutstanding", 60_000_000.0,
                   "us-gaap:CommonClassAMember", "2026-07-21"),
        _class_fact("dei:EntityCommonStockSharesOutstanding", 60_000_000.0,
                   "us-gaap:CommonClassBMember", "2026-07-21"),
    ])
    out = build_tag_frames(facts_df, SHARES_TAGS)
    assert len(out) == 1
    assert out.iloc[0]["value"] == 120_000_000.0


def test_build_tag_frames_class_sum_includes_a_class_below_the_magnitude_floor():
    """`SHARE_COUNT_MIN_ABS` screens the TOTAL, never the individual classes: a real
    class routinely sits under a million shares on its own (BRK-B Class A 505,697, ERIE
    Class B 2,542, SPG Class B 8,000), and screening components would drop exactly the
    small classes the sum exists to add in."""
    facts_df = pd.DataFrame([
        _class_fact("dei:EntityCommonStockSharesOutstanding", 505_697.0,
                   "us-gaap:CommonClassAMember", "2026-04-14"),
        _class_fact("dei:EntityCommonStockSharesOutstanding", 1_398_309_000.0,
                   "us-gaap:CommonClassBMember", "2026-04-14"),
    ])
    out = build_tag_frames(facts_df, SHARES_TAGS)
    assert len(out) == 1
    assert out.iloc[0]["value"] == 505_697.0 + 1_398_309_000.0   # real BRK-B Q1-2026


def test_build_tag_frames_class_sum_still_obeys_the_magnitude_floor_on_the_total():
    """The floor is applied to the total, so a filer whose whole class enumeration is
    1,000,000x too small is still rejected rather than shipped."""
    facts_df = pd.DataFrame([
        _class_fact("dei:EntityCommonStockSharesOutstanding", 500.0,
                   "us-gaap:CommonClassAMember", "2026-04-14"),
        _class_fact("dei:EntityCommonStockSharesOutstanding", 1_390.0,
                   "us-gaap:CommonClassBMember", "2026-04-14"),
    ])
    assert build_tag_frames(facts_df, SHARES_TAGS).empty


def test_build_tag_frames_never_sums_the_balance_sheet_share_classes():
    """Only the COVER-PAGE enumeration is summable. The balance-sheet parenthetical is a
    presentation footnote whose per-class facts are routinely INCOMPLETE -- confirmed on
    EL, which tags `CommonStockSharesOutstanding` for Class B ONLY (114,507,344) against
    a true 361,794,915, and on ACN, which tags only Class X (302,358 of 667,810,900).
    Summing it would ship a silently wrong total, so the cover page answers instead."""
    facts_df = pd.DataFrame([
        _class_fact("us-gaap:CommonStockSharesOutstanding", 114_507_344.0,
                   "us-gaap:CommonClassBMember", "2026-03-31"),
        _class_fact("dei:EntityCommonStockSharesOutstanding", 247_287_571.0,
                   "us-gaap:CommonClassAMember", "2026-04-24"),
        _class_fact("dei:EntityCommonStockSharesOutstanding", 114_507_344.0,
                   "us-gaap:CommonClassBMember", "2026-04-24"),
    ])
    out = build_tag_frames(facts_df, SHARES_TAGS)
    assert len(out) == 1
    assert out.iloc[0]["value"] == 247_287_571.0 + 114_507_344.0   # real EL Q3-FY2026
    assert pd.Timestamp(out.iloc[0]["period_end"]) == pd.Timestamp("2026-04-24")


def _scalar_fact(concept, value, date, member=None):
    """An undimensioned (or singly class-dimensioned) ratio/percentage fact."""
    row = _fact(concept, value, None, None, "instant", None, None,
               is_dimensioned=member is not None, period_instant=date, unit_ref="U_pure")
    if member is not None:
        row |= {"dimension": _CLASS_AXIS, "member": member, _CLASS_AXIS_COL: member}
    return row


def _ibkr_equity_evidence():
    """IBKR's two equity bases: parent $5.363bn of a consolidated $20.472bn = 26.2%,
    which is what corroborates its tagged 26.6% parent ownership as GROUP-level."""
    return [
        _fact("us-gaap:StockholdersEquity", 5_363_000_000.0, None, None,
             "instant", None, None, period_instant="2026-06-30"),
        _fact("us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
             20_472_000_000.0, None, None, "instant", None, None, period_instant="2026-06-30"),
    ]


def test_build_tag_frames_converts_a_class_by_its_own_tagged_conversion_ratio():
    """Where classes do NOT convert 1:1 the raw sum is wrong, not merely imprecise -- and
    the filer publishes the factor. Real ERIE Q2-2026: Class A 46,189,068 + Class B 2,542,
    with `erie:CommonStockConversionRatio` = 2,400 on the Class B member. 46,189,068 +
    2,542 x 2,400 = 52,289,868, which is Yahoo's all-classes figure to the share (the raw
    sum, 46,191,610, is 12% light)."""
    facts_df = pd.DataFrame([
        _class_fact("dei:EntityCommonStockSharesOutstanding", 46_189_068.0,
                   "us-gaap:CommonClassAMember", "2026-07-24"),
        _class_fact("dei:EntityCommonStockSharesOutstanding", 2_542.0,
                   "us-gaap:CommonClassBMember", "2026-07-24"),
        _scalar_fact("erie:CommonStockConversionRatio", 2_400.0, "2026-06-30",
                    member="us-gaap:CommonClassBMember"),
    ])
    out = build_tag_frames(facts_df, SHARES_TAGS)
    assert len(out) == 1
    assert out.iloc[0]["value"] == 52_289_868.0


def test_build_tag_frames_ignores_a_conversion_ratio_at_or_below_one():
    """A ratio <= 1 states the INVERSE direction (CVNA tags 0.8 = Class A shares per LLC
    unit) or the identity. Applying it would SHRINK a class, so it is ignored and the
    plain sum stands -- fill-only in both directions."""
    facts_df = pd.DataFrame([
        _class_fact("dei:EntityCommonStockSharesOutstanding", 719_916_415.0,
                   "us-gaap:CommonClassAMember", "2026-07-27"),
        _class_fact("dei:EntityCommonStockSharesOutstanding", 380_547_355.0,
                   "us-gaap:CommonClassBMember", "2026-07-27"),
        _scalar_fact("cvna:CommonStockConversionRatio", 0.8, "2026-06-30",
                    member="us-gaap:CommonClassAMember"),
    ])
    out = build_tag_frames(facts_df, SHARES_TAGS)
    assert len(out) == 1
    assert out.iloc[0]["value"] == 719_916_415.0 + 380_547_355.0   # real CVNA Q2-2026


def test_build_tag_frames_scales_the_senior_class_by_an_economic_equivalent_percentage():
    """BRK states the relationship as a PERCENTAGE instead: `brka:EconomicEquivalent-
    PercentageOfClassBCommonShareToClassACommonShare` = 6.67e-4, i.e. one Class A share is
    worth ~1,500 Class B. Applied to the SMALLEST class (a share worth 1,500 of the other
    is necessarily the rarer one), 505,697 x 1,499.25 + 1,398,309,000 = 2.156e9 -- Yahoo's
    figure is 2,156,853,797, and the raw sum (1.399e9) is 35% light."""
    facts_df = pd.DataFrame([
        _class_fact("dei:EntityCommonStockSharesOutstanding", 505_697.0,
                   "us-gaap:CommonClassAMember", "2026-04-14"),
        _class_fact("dei:EntityCommonStockSharesOutstanding", 1_398_309_000.0,
                   "us-gaap:CommonClassBMember", "2026-04-14"),
        _scalar_fact("brka:EconomicEquivalentPercentageOfClassBCommonShareToClassACommonShare",
                    6.67e-4, "2026-03-31"),
    ])
    out = build_tag_frames(facts_df, SHARES_TAGS)
    assert len(out) == 1
    assert out.iloc[0]["value"] == pytest.approx(2_156_853_797.0, rel=5e-4)


def test_build_tag_frames_grosses_an_up_c_count_up_to_the_consolidated_group():
    """IBKR's Class B is 400 shares, so the IBG LLC members hold NO paired common stock --
    the registered classes cover only the corporate slice. `MinorityInterestOwnership-
    PercentageByParent` = 0.266 says so (Class A is 99.99998% of the sum against a tagged
    26.6%), and grossing up puts the count on the same consolidated basis as netIncome and
    equity: 453,078,171 / 0.266 = 1.703e9, against Yahoo's 1,701,309,599."""
    facts_df = pd.DataFrame([
        _class_fact("dei:EntityCommonStockSharesOutstanding", 453_077_771.0,
                   "us-gaap:CommonClassAMember", "2026-08-05"),
        _class_fact("dei:EntityCommonStockSharesOutstanding", 400.0,
                   "us-gaap:CommonClassBMember", "2026-08-05"),
        _scalar_fact("us-gaap:MinorityInterestOwnershipPercentageByParent", 0.266,
                    "2026-07-31"),
        *_ibkr_equity_evidence(),
    ])
    out = build_tag_frames(facts_df, SHARES_TAGS)
    assert len(out) == 1
    assert out.iloc[0]["value"] == pytest.approx(1_701_309_599.0, rel=3e-3)


def test_build_tag_frames_leaves_an_up_c_count_alone_when_the_classes_already_cover_it():
    """The mirror case, and the reason the gross-up is conditional rather than automatic.
    Each CVNA Class B share is paired 1:1 with an exchangeable LLC unit, so Class A is
    65.4% of the class sum against a tagged 65% -- the filing's own ownership disclosure
    CONFIRMING that the sum is already the whole group in Class A units. Dividing by 0.65
    would count the Garcia interest twice (1.100e9 -> 1.693e9)."""
    facts_df = pd.DataFrame([
        _class_fact("dei:EntityCommonStockSharesOutstanding", 719_916_415.0,
                   "us-gaap:CommonClassAMember", "2026-07-27"),
        _class_fact("dei:EntityCommonStockSharesOutstanding", 380_547_355.0,
                   "us-gaap:CommonClassBMember", "2026-07-27"),
        _scalar_fact("us-gaap:MinorityInterestOwnershipPercentageByParent", 0.65,
                    "2026-06-30"),
    ])
    out = build_tag_frames(facts_df, SHARES_TAGS)
    assert len(out) == 1
    assert out.iloc[0]["value"] == 1_100_463_770.0


def test_build_tag_frames_ignores_a_stale_ownership_percentage():
    """IBKR's ownership footnote also restates the ORIGINAL 2007 IPO split (10%/90%)
    alongside today's 26.6%. Taking the stale one would gross the count up ~10x, so the
    LATEST-dated qualifying fact is the one used."""
    facts_df = pd.DataFrame([
        _class_fact("dei:EntityCommonStockSharesOutstanding", 453_077_771.0,
                   "us-gaap:CommonClassAMember", "2026-08-05"),
        _class_fact("dei:EntityCommonStockSharesOutstanding", 400.0,
                   "us-gaap:CommonClassBMember", "2026-08-05"),
        _scalar_fact("us-gaap:MinorityInterestOwnershipPercentageByParent", 0.10,
                    "2007-05-03"),
        _scalar_fact("us-gaap:MinorityInterestOwnershipPercentageByParent", 0.266,
                    "2026-07-31"),
        *_ibkr_equity_evidence(),
    ])
    out = build_tag_frames(facts_df, SHARES_TAGS)
    assert len(out) == 1
    assert out.iloc[0]["value"] == pytest.approx(1_701_309_599.0, rel=3e-3)


def test_build_tag_frames_ignores_an_ownership_percentage_that_is_not_group_level():
    """`MinorityInterestOwnershipPercentageByParent` is NOT self-identifying -- filers use
    it just as readily for one SUBSIDIARY or joint venture, and believing it there
    multiplies the share count. Both cases below are real and were caught only by the live
    cross-check: CMCSA tags 0.30 and UHS 0.20 for holdings of theirs, which grossed their
    counts up by 3.33x and 5.00x. The percentage must match the parent's share of
    consolidated EQUITY (here ~0.98) before it is believed."""
    facts_df = pd.DataFrame([
        _class_fact("dei:EntityCommonStockSharesOutstanding", 3_546_404_000.0,
                   "us-gaap:CommonClassAMember", "2026-07-23"),
        _class_fact("dei:EntityCommonStockSharesOutstanding", 2_232_573.0,
                   "us-gaap:CommonClassBMember", "2026-07-23"),
        _scalar_fact("us-gaap:MinorityInterestOwnershipPercentageByParent", 0.30,
                    "2026-06-30"),
        _fact("us-gaap:StockholdersEquity", 98_000_000_000.0, None, None,
             "instant", None, None, period_instant="2026-06-30"),
        _fact("us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
             100_000_000_000.0, None, None, "instant", None, None, period_instant="2026-06-30"),
    ])
    out = build_tag_frames(facts_df, SHARES_TAGS)
    assert len(out) == 1
    assert out.iloc[0]["value"] == 3_546_404_000.0 + 2_232_573.0


def test_build_tag_frames_no_gross_up_without_equity_evidence():
    """Fails CLOSED: with no equity to corroborate it, a tagged ownership percentage is
    not acted on at all, so the plain class sum stands."""
    facts_df = pd.DataFrame([
        _class_fact("dei:EntityCommonStockSharesOutstanding", 453_077_771.0,
                   "us-gaap:CommonClassAMember", "2026-08-05"),
        _class_fact("dei:EntityCommonStockSharesOutstanding", 400.0,
                   "us-gaap:CommonClassBMember", "2026-08-05"),
        _scalar_fact("us-gaap:MinorityInterestOwnershipPercentageByParent", 0.266,
                    "2026-07-31"),
    ])
    out = build_tag_frames(facts_df, SHARES_TAGS)
    assert len(out) == 1
    assert out.iloc[0]["value"] == 453_077_771.0 + 400.0


def test_build_tag_frames_prefers_a_tagged_as_converted_total_over_everything():
    """V publishes the whole company already converted into Class A units, undimensioned
    and dated at PERIOD END (1.880e9 at 2026-06-30, vs Yahoo's 1.867e9) -- so it needs
    neither the class summing nor any conversion arithmetic, and being priority-0 it also
    keeps the measurement date aligned with every other instant field."""
    facts_df = pd.DataFrame([
        _fact("v:SharesOutstandingAsConvertedBasis", 1_880_000_000.0, None, None,
             "instant", None, None, period_instant="2026-06-30"),
        _class_fact("us-gaap:CommonStockSharesOutstanding", 1_702_000_000.0,
                   "us-gaap:CommonClassAMember", "2026-06-30"),
        _class_fact("dei:EntityCommonStockSharesOutstanding", 1_704_113_000.0,
                   "us-gaap:CommonClassAMember", "2026-07-21"),
    ])
    out = build_tag_frames(facts_df, SHARES_TAGS)
    at_period_end = out[pd.to_datetime(out["period_end"]) == pd.Timestamp("2026-06-30")]
    assert len(at_period_end) == 1
    assert at_period_end.iloc[0]["value"] == 1_880_000_000.0
    assert at_period_end.iloc[0]["source_tag"] == "v:SharesOutstandingAsConvertedBasis"


def test_build_tag_frames_conversion_rules_do_not_touch_an_ordinary_filer():
    """None of the three hooks exists for a filer with one basis, so the plain class sum
    is returned unchanged -- the fill-only property that makes them safe to add."""
    facts_df = pd.DataFrame([
        _class_fact("dei:EntityCommonStockSharesOutstanding", 2_205_128_509.0,
                   "us-gaap:CommonClassAMember", "2026-07-24"),
        _class_fact("dei:EntityCommonStockSharesOutstanding", 342_377_716.0,
                   "us-gaap:CommonClassBMember", "2026-07-24"),
    ])
    out = build_tag_frames(facts_df, SHARES_TAGS)
    assert len(out) == 1
    assert out.iloc[0]["value"] == 2_205_128_509.0 + 342_377_716.0


def test_build_tag_frames_reported_total_wins_over_the_class_sum():
    """FILL-ONLY, like every other recovery in this module: a filer that DID report an
    undimensioned cover-page total (confirmed: CRWD, VEEV, WBD) keeps it -- nothing is
    summed for that date, so the two can never both produce a row and be picked between
    arbitrarily."""
    facts_df = pd.DataFrame([
        _fact("dei:EntityCommonStockSharesOutstanding", 253_614_100.0, None, None,
             "instant", None, None, period_instant="2026-02-27"),
        _class_fact("dei:EntityCommonStockSharesOutstanding", 250_000_000.0,
                   "us-gaap:CommonClassAMember", "2026-02-27"),
        _class_fact("dei:EntityCommonStockSharesOutstanding", 3_614_100.0,
                   "us-gaap:CommonClassBMember", "2026-02-27"),
    ])
    out = build_tag_frames(facts_df, SHARES_TAGS)
    assert len(out) == 1
    assert out.iloc[0]["value"] == 253_614_100.0


def test_build_tag_frames_single_class_filer_is_untouched_by_the_class_rules():
    """The overwhelming majority of filers tag an UNDIMENSIONED balance-sheet share
    count. Nothing above may change them: the balance-sheet tag still wins on priority
    and still carries the period-end date, so no measurement date shifts."""
    facts_df = pd.DataFrame([
        _fact("us-gaap:CommonStockSharesOutstanding", 137_622_108.0, None, None,
             "instant", None, None, period_instant="2025-12-31"),
        _fact("dei:EntityCommonStockSharesOutstanding", 137_500_000.0, None, None,
             "instant", None, None, period_instant="2026-02-17"),
    ])
    out = build_tag_frames(facts_df, SHARES_TAGS)
    at_period_end = out[pd.to_datetime(out["period_end"]) == pd.Timestamp("2025-12-31")]
    assert len(at_period_end) == 1
    assert at_period_end.iloc[0]["value"] == 137_622_108.0
    assert at_period_end.iloc[0]["source_tag"] == "us-gaap:CommonStockSharesOutstanding"


def test_build_tag_frames_class_rules_are_scoped_to_the_share_count_field():
    """The class-of-stock rules are scoped to `SHARE_CLASS_COMPONENT_FIELDS`. Any other
    field dimensioned on that axis keeps the pre-existing behavior -- two disagreeing
    members with no undimensioned duplicate stay EXCLUDED, never summed."""
    facts_df = pd.DataFrame([
        _class_fact("us-gaap:CommonStockValue", 100.0, "us-gaap:CommonClassAMember",
                   "2026-06-30"),
        _class_fact("us-gaap:CommonStockValue", 250.0, "us-gaap:CommonClassBMember",
                   "2026-06-30"),
    ])
    assert build_tag_frames(facts_df, {"commonStockValue": ["CommonStockValue"]}).empty


def test_build_tag_frames_reads_instant_facts_from_period_instant_column():
    """INSTANT (balance-sheet) facts carry their date in a SEPARATE
    `period_instant` column, not `period_end` (both NaN for them in edgartools'
    own to_dataframe() output) -- confirmed empirically against a real MAA 10-K.
    Without this normalization, every STOCK field (totalAssets, ...) silently
    vanished: grouped into one NaN/NaN bucket and later excluded entirely by the
    current-period filter (period_end == filing.period_of_report)."""
    facts_df = pd.DataFrame([
        _fact("us-gaap:Assets", 11975383000.0, None, None, "instant", None, None,
             period_instant="2025-12-31"),
    ])
    out = build_tag_frames(facts_df, {"totalAssets": ["Assets"]})
    assert len(out) == 1
    assert pd.Timestamp(out.iloc[0]["period_end"]) == pd.Timestamp("2025-12-31")
    assert out.iloc[0]["value"] == 11975383000.0


def test_build_tag_frames_normalizes_ytd_labels_before_backfill_can_propagate_them():
    """fiscal_period is normalized ('YTD6' -> 'Q2') at the EARLIEST capture point in
    build_tag_frames, not just inside decumulate_quarterly_flow -- otherwise
    backfill_fiscal_period_from_filing (which borrows whichever OTHER row in the
    same filing has native fy/fp to fill an instant fact's blank one) could borrow
    a raw 'YTD6' from a cash-flow duration fact and stamp it onto a balance-sheet
    field, which then flows through instant_stock() unmodified. Real bug: MAA's
    totalAssets showed literal 'YTD6'/'YTD9' fiscal_period values in
    fundamentals_facts before this fix."""
    facts_df = pd.DataFrame([
        _fact("us-gaap:NetCashProvidedByUsedInOperatingActivities", 463907000.0,
             "2022-01-01", "2022-06-30", "duration", 2022, "YTD6"),
    ])
    out = build_tag_frames(facts_df, {"operatingCashFlow": ["NetCashProvidedByUsedInOperatingActivities"]})
    assert len(out) == 1
    assert out.iloc[0]["fiscal_period"] == "Q2"


def test_backfill_fiscal_period_from_filing_fills_instant_facts_from_duration_facts():
    """A filing's instant facts (no native fiscal_year/fiscal_period) must inherit
    the (fiscal_year, fiscal_period) carried by any duration fact from the SAME
    filing (e.g. the dei cover-page tags, which always have it)."""
    tagged = pd.DataFrame([
        {"field": "totalAssets", "fiscal_year": None, "fiscal_period": None, "value": 100.0},
        {"field": "netIncome", "fiscal_year": 2025, "fiscal_period": "FY", "value": 50.0},
    ])
    out = backfill_fiscal_period_from_filing(tagged)
    row = out[out["field"] == "totalAssets"].iloc[0]
    assert row["fiscal_year"] == 2025 and row["fiscal_period"] == "FY"


def test_backfill_fiscal_period_from_filing_noop_when_nothing_native():
    """If NOTHING in the frame carries a native fiscal_year/fiscal_period (should
    not happen in practice -- every filing has cover-page dei facts -- but must not
    crash), the frame is returned unchanged rather than raising."""
    tagged = pd.DataFrame([{"field": "totalAssets", "fiscal_year": None, "fiscal_period": None, "value": 100.0}])
    out = backfill_fiscal_period_from_filing(tagged)
    assert out.iloc[0]["fiscal_year"] is None


def test_build_tag_frames_coalesces_by_candidate_priority():
    """When two DIFFERENT candidate tags both report the same period, the
    higher-priority (earlier-listed) candidate wins -- the same coalescing rule as
    fetch_fundamentals.py::_extract_concept, generalized to run per-filing."""
    facts_df = pd.DataFrame([
        _fact("us-gaap:SalesRevenueNet", 50.0, "2024-01-01", "2024-03-31", "duration", 2024, "Q1"),
        _fact("us-gaap:Revenues", 100.0, "2024-01-01", "2024-03-31", "duration", 2024, "Q1"),
    ])
    tag_map = {"totalRevenue": ["Revenues", "SalesRevenueNet"]}   # Revenues = higher priority
    out = build_tag_frames(facts_df, tag_map)
    assert len(out) == 1
    assert out.iloc[0]["value"] == 100.0
    assert out.iloc[0]["source_tag"] == "us-gaap:Revenues"


def test_build_tag_frames_drops_non_numeric_values():
    """A mis-matched or text-valued fact (empty string / non-numeric) must be
    dropped rather than crash downstream period-decumulation math -- the exact
    real-data bug this test pins (ValueError: could not convert string to float)."""
    facts_df = pd.DataFrame([
        _fact("us-gaap:NetIncomeLoss", "", "2024-01-01", "2024-03-31", "duration", 2024, "Q1",
             numeric_value=float("nan")),
        _fact("us-gaap:NetIncomeLoss", 100.0, "2024-04-01", "2024-06-30", "duration", 2024, "Q2"),
    ])
    out = build_tag_frames(facts_df, {"netIncome": ["NetIncomeLoss"]})
    assert len(out) == 1
    assert out.iloc[0]["fiscal_period"] == "Q2"


def test_lease_maturity_1y_maps_rolling_twelve_month_tag():
    """MAA's operating-lease maturity ladder tags the FIRST rung under
    `LesseeOperatingLeaseLiabilityPaymentsDueNextRollingTwelveMonths`, a distinct
    2019+ us-gaap concept for filers disclosing the schedule "rolling" forward from
    the balance-sheet date -- NOT an alternate name for `...NextTwelveMonths`. This
    tag was entirely absent from the candidate list, silently zeroing out
    leaseMaturity1y (and, combined with the debt-maturity variant, most of the
    refinancing-wall feature) for every filer using this convention."""
    from src.data_extract.utils.fundamentals.fetch_fundamentals_edgar import INSTANT_FIELD_TAGS
    facts_df = pd.DataFrame([
        _fact("us-gaap:LesseeOperatingLeaseLiabilityPaymentsDueNextRollingTwelveMonths", 5000000.0,
             None, None, "instant", 2025, "FY", period_instant="2025-12-31"),
    ])
    out = build_tag_frames(facts_df, {"leaseMaturity1y": INSTANT_FIELD_TAGS["leaseMaturity1y"]})
    assert len(out) == 1
    assert out.iloc[0]["value"] == 5000000.0


def test_capex_maps_reit_maintenance_tag_maa_style():
    """MAA (REIT) reports capex under PaymentsForCapitalImprovements, not the
    generic PP&E tag -- the confirmed root cause of the WIP file's capex gap.
    Ported tag list (fundamentals_tags.py) must resolve it."""
    from src.data_extract.utils.fundamentals.fetch_fundamentals_edgar import FLOW_FIELD_TAGS
    facts_df = pd.DataFrame([
        _fact("us-gaap:PaymentsForCapitalImprovements", 360238000.0, "2025-01-01", "2025-12-31",
             "duration", 2025, "FY"),
    ])
    out = build_tag_frames(facts_df, {"capex": FLOW_FIELD_TAGS["capex"]})
    assert len(out) == 1
    assert out.iloc[0]["value"] == 360238000.0
    assert out.iloc[0]["source_tag"] == "us-gaap:PaymentsForCapitalImprovements"


def test_quarterly_derivation_backs_out_q4():
    """End-to-end: raw quarterly facts -> decumulate_quarterly_flow -> Q4 derived
    and reconciles exactly to FY."""
    facts = pd.DataFrame([
        {"fiscal_year": 2024, "fiscal_period": "Q1", "period_start": "2024-01-01", "period_end": "2024-03-31",
         "value": 100.0, "filing_date": "2024-04-20", "accession_number": "a1", "form": "10-Q"},
        {"fiscal_year": 2024, "fiscal_period": "Q2", "period_start": "2024-01-01", "period_end": "2024-06-30",
         "value": 220.0, "filing_date": "2024-07-20", "accession_number": "a2", "form": "10-Q"},
        {"fiscal_year": 2024, "fiscal_period": "Q3", "period_start": "2024-01-01", "period_end": "2024-09-30",
         "value": 345.0, "filing_date": "2024-10-20", "accession_number": "a3", "form": "10-Q"},
        {"fiscal_year": 2024, "fiscal_period": "FY", "period_start": "2024-01-01", "period_end": "2024-12-31",
         "value": 460.0, "filing_date": "2025-02-15", "accession_number": "a4", "form": "10-K"},
    ])
    for c in ("period_start", "period_end", "filing_date"):
        facts[c] = pd.to_datetime(facts[c])
    out = decumulate_quarterly_flow(facts)
    q = {r.fiscal_period: r.value for r in out.itertuples()}
    assert q["Q1"] == 100.0 and q["Q2"] == 120.0 and q["Q3"] == 125.0 and q["Q4"] == 115.0
    assert abs(sum(q.values()) - 460.0) < 1e-9
    q4_row = out[out.fiscal_period == "Q4"].iloc[0]
    assert bool(q4_row["derived"]) is True
    assert set(q4_row["derived_from_accessions"]) == {"a1", "a2", "a3", "a4"}


def test_fetch_fundamentals_edgartools_saves_incrementally_per_ticker(tmp_path, monkeypatch):
    """Each ticker's rows are persisted immediately after its OWN extraction
    finishes -- not accumulated in memory and saved once at the end. A large
    filer can take minutes; a later ticker's failure must never lose an
    earlier ticker's already-extracted work."""
    import src.data_extract.utils.fundamentals.fetch_fundamentals_edgar as mod

    # **_ absorbs the per-ticker extras the real signature takes (employee_history,
    # log) -- this test is about the SAVE cadence, not about what is extracted
    def _fake_build(ticker, *, done_accessions=frozenset(), since=None, **_):
        if ticker == "BAD":
            raise RuntimeError("simulated failure for BAD ticker")
        return pd.DataFrame([{
            "ticker": ticker, "cik": None, "accession_number": f"acc-{ticker}", "field": "netIncome",
            "fiscal_year": 2024, "fiscal_period": "Q1", "duration_type": "quarterly", "form": "10-Q",
            "filing_date": pd.Timestamp("2024-04-20"), "period_start": pd.Timestamp("2024-01-01"),
            "period_end": pd.Timestamp("2024-03-31"), "value": 100.0, "unit": "USD",
            "source_tag": "us-gaap:NetIncomeLoss", "is_amendment": 0.0, "amends_accession": None,
            "derived": 0.0, "derived_from_accessions": None, "fiscal_period_source": "native",
        }])

    monkeypatch.setattr(mod, "_configure_identity", lambda: None)
    monkeypatch.setattr(mod, "build_ticker_facts_edgar", _fake_build)

    engine = create_engine(f"sqlite:///{tmp_path / 't.db'}")
    store = DataStore(engine)
    context = SimpleNamespace(
        config=SimpleNamespace(data_extract=SimpleNamespace(years_history=15)),
        store=store, log=logging.getLogger("test"), paths={"DATA_STORE": tmp_path})

    out = mod.fetch_fundamentals_edgartools(context, ["GOOD", "BAD"])

    # GOOD's row is already persisted even though BAD (processed afterward) failed.
    persisted = store.load("fundamentals_facts")
    assert len(persisted) == 1 and persisted.iloc[0]["ticker"] == "GOOD"
    assert len(out) == 1 and out.iloc[0]["ticker"] == "GOOD"
    engine.dispose()


def test_fundamentals_facts_upsert_is_idempotent(tmp_path):
    """Saving the same fundamentals_facts frame twice must not duplicate rows or
    change values (registry PK: ticker, accession_number, field, fiscal_year,
    fiscal_period, duration_type). Mirrors test_sec_bulk_datasets.py's throwaway-
    SQLite pattern (pytest's `tmp_path` fixture, not `tempfile.TemporaryDirectory`
    -- the latter's cleanup races SQLAlchemy's file handle release on Windows)."""
    df = pd.DataFrame([
        {"ticker": "ZZZ", "cik": "0000000001", "accession_number": "acc-1", "field": "netIncome",
         "fiscal_year": 2024, "fiscal_period": "Q1", "duration_type": "quarterly", "form": "10-Q",
         "filing_date": pd.Timestamp("2024-04-20"), "period_start": pd.Timestamp("2024-01-01"),
         "period_end": pd.Timestamp("2024-03-31"), "value": 100.0, "unit": "USD",
         "source_tag": "us-gaap:NetIncomeLoss", "is_amendment": 0.0, "amends_accession": None,
         "derived": 0.0, "derived_from_accessions": None, "fiscal_period_source": "native"},
    ])
    engine = create_engine(f"sqlite:///{tmp_path / 't.db'}")
    store = DataStore(engine)
    n1 = store.save("fundamentals_facts", df)
    n2 = store.save("fundamentals_facts", df)
    loaded = store.load("fundamentals_facts")
    assert n1 == 1 and n2 == 1
    assert len(loaded) == 1
    assert loaded.iloc[0]["value"] == 100.0
    engine.dispose()

    print("\n=== SANITY CHECK: fundamentals_facts retrieval + persistence ===")
    print("  dimensioned duplicates excluded; candidate-priority coalescing correct;")
    print("  non-numeric facts dropped (not crashed); MAA-style REIT capex tag resolves;")
    print("  instant (balance-sheet) facts correctly read from period_instant and")
    print("  backfilled with their filing's fiscal_year/fiscal_period (real bug found and")
    print("  fixed against live MAA data -- totalAssets/stockholdersEquity/totalLiabilities")
    print("  were silently empty before this fix);")
    print("  Q1-Q4 decumulation reconciles exactly to FY; repeated save is idempotent (no dupes).")
    print("  MULTI-CLASS share counts: a class-dimensioned fact is never admitted as the")
    print("  company total (CME/CVNA used to store ONE class, chosen by frame order), and")
    print("  the total is rebuilt by summing the COVER-PAGE classes only -- the balance-sheet")
    print("  parenthetical is never summed (EL tags Class B only, ACN Class X only, V/BRK-B")
    print("  add a roll-up member). Roll-ups taken alone, twice-tagged classes counted once,")
    print("  sub-million classes kept in the sum while the floor screens the total, a reported")
    print("  total always wins, and single-class filers resolve exactly as before.")
    print("  NON-1:1 CLASSES and UP-C structures, all from factors the filers tag themselves:")
    print("  ERIE x2400 on Class B -> 52,289,868 (exact vs Yahoo, raw sum was 12% light);")
    print("  BRK-B economic-equivalent 6.67e-4 on the senior class -> 2.156e9 (was 35% light);")
    print("  V's own as-converted total wins outright at period end; IBKR grossed up by its")
    print("  tagged 26.6% parent ownership -> 1.703e9, while CVNA is LEFT ALONE because its")
    print("  65.4% class split already agrees with the tagged 65% (paired LLC units).")
    print("  An inverse ratio (<=1) and a stale 2007 ownership fact are both ignored.")
    print("  Validated.")
