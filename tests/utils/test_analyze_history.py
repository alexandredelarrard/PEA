"""
Tests for src/utils/analyze_history.py: level-outlier detection (Modified
Z-score) and cross-duration_type source_tag misalignment over
`fundamentals_facts`-shaped data.

Pure-synthetic, no network / no DB.
"""
from __future__ import annotations

import pandas as pd

from src.validate.analyze_history import detect_level_outliers, detect_source_tag_misalignment


def _row(fiscal_year, fiscal_period, value, source_tag="us-gaap:X", duration_type="quarterly",
        filed=None, derived=0.0, is_amendment=0.0, ticker="ZZZ", field="totalRevenue"):
    filed = filed or f"{fiscal_year}-06-15"
    return {"ticker": ticker, "field": field, "fiscal_year": fiscal_year, "fiscal_period": fiscal_period,
           "duration_type": duration_type, "value": value, "source_tag": source_tag,
           "filing_date": pd.Timestamp(filed), "derived": derived, "is_amendment": is_amendment,
           "accession_number": f"acc-{fiscal_year}-{fiscal_period}"}


def test_derived_rows_pass_through_without_crashing_the_outlier_stats():
    """A derived Q4 row (source_tag=None by design) must still appear in the
    output, participating normally in the level/YoY outlier stats (its own
    coherence is already enforced at build time by
    `fundamentals_periods._q4_is_coherent`) -- no source_tag comparison here
    at all (see `detect_source_tag_misalignment` for that, which IS era-aware)."""
    rows = [
        _row(2022, "Q1", 100.0), _row(2022, "Q2", 110.0), _row(2022, "Q3", 120.0),
        _row(2022, "Q4", 118.0, source_tag=None, derived=1.0),
        _row(2023, "Q1", 105.0), _row(2023, "Q2", 115.0), _row(2023, "Q3", 125.0),
        _row(2023, "Q4", 122.0, source_tag=None, derived=1.0),
    ]
    df = pd.DataFrame(rows)
    out = detect_level_outliers(df, "ZZZ", "totalRevenue")
    derived_rows = out[out["derived"] == 1.0]
    assert len(derived_rows) == 2
    assert not derived_rows["is_level_outlier"].any()


def test_yoy_check_does_not_flag_the_first_four_periods():
    """The first (up to) 4 periods have no defined 4-quarter-lag YoY diff --
    they must never be flagged as a "YoY Shift Anomaly" just because their
    filled-in placeholder diff looks large relative to later periods' real
    YoY diffs."""
    rows = [
        _row(2021, "Q1", 100.0), _row(2021, "Q2", 105.0), _row(2021, "Q3", 110.0), _row(2021, "Q4", 115.0),
        _row(2022, "Q1", 102.0), _row(2022, "Q2", 107.0), _row(2022, "Q3", 112.0), _row(2022, "Q4", 117.0),
    ]
    df = pd.DataFrame(rows)
    out = detect_level_outliers(df, "ZZZ", "totalRevenue")
    first_four = out.iloc[:4]
    assert not first_four["is_yoy_outlier"].any()


def test_level_outlier_flags_a_genuine_spike():
    """A genuinely wrong single-quarter value (an order of magnitude off from
    every other quarter) must still be caught."""
    rows = [
        _row(2021, "Q1", 100.0), _row(2021, "Q2", 105.0), _row(2021, "Q3", 110.0), _row(2021, "Q4", 108.0),
        _row(2022, "Q1", 9.0),   # a real bug: 10x lower than every other quarter
        _row(2022, "Q2", 107.0), _row(2022, "Q3", 112.0), _row(2022, "Q4", 111.0),
    ]
    df = pd.DataFrame(rows)
    out = detect_level_outliers(df, "ZZZ", "totalRevenue")
    bad = out[(out["fiscal_year"] == 2022) & (out["fiscal_period"] == "Q1")]
    assert bad.iloc[0]["is_level_outlier"]


def test_tag_misalignment_flags_annual_vs_quarterly_mismatch():
    """Real bug pattern (JPM): the ANNUAL row resolves a DIFFERENT XBRL
    concept than its own quarters for the SAME fiscal year -- flagged."""
    rows = [
        _row(2016, "Q1", 23.2, source_tag="us-gaap:RevenuesNetOfInterestExpense"),
        _row(2016, "Q2", 24.4, source_tag="us-gaap:RevenuesNetOfInterestExpense"),
        _row(2016, "Q3", 24.7, source_tag="us-gaap:RevenuesNetOfInterestExpense"),
        _row(2016, "FY", 16.0, source_tag="us-gaap:Revenues", duration_type="annual"),
    ]
    df = pd.DataFrame(rows)
    out = detect_source_tag_misalignment(df, "ZZZ", "totalRevenue")
    assert len(out) == 1
    assert out.iloc[0]["mismatch_period_end_vs_interim"]
    assert out.iloc[0]["period_end_source_tag"] == "us-gaap:Revenues"
    assert out.iloc[0]["interim_source_tags"] == ["us-gaap:RevenuesNetOfInterestExpense"]


def test_tag_misalignment_not_flagged_for_a_clean_taxonomy_transition():
    """A permanent tag switch where annual and quarterly AGREE within each
    year (e.g. a post-ASC-606 taxonomy cutover) is NOT a bug -- must not be
    flagged, even though the tag differs across fiscal years."""
    rows = [
        _row(2017, "Q1", 100.0, source_tag="us-gaap:SalesRevenueNet"),
        _row(2017, "Q2", 105.0, source_tag="us-gaap:SalesRevenueNet"),
        _row(2017, "Q3", 110.0, source_tag="us-gaap:SalesRevenueNet"),
        _row(2017, "FY", 430.0, source_tag="us-gaap:SalesRevenueNet", duration_type="annual"),
        _row(2018, "Q1", 108.0, source_tag="us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"),
        _row(2018, "Q2", 112.0, source_tag="us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"),
        _row(2018, "Q3", 118.0, source_tag="us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"),
        _row(2018, "FY", 450.0, source_tag="us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
            duration_type="annual"),
    ]
    df = pd.DataFrame(rows)
    out = detect_source_tag_misalignment(df, "ZZZ", "totalRevenue")
    assert out.empty


def test_tag_misalignment_and_outliers_cover_instant_balance_sheet_fields():
    """Balance-sheet fields (totalAssets/totalLiabilities/cash) are always
    duration_type='instant' -- never 'annual' or 'quarterly' -- and have no
    separate annual bucket at all (the FY-end snapshot is just another
    'instant' row, fiscal_period='FY' or 'Q4'). Both checks must still work
    for them via the fiscal_period-based (not duration_type-based) split."""
    rows = [
        _row(2020, "Q1", 100.0, source_tag="us-gaap:Assets", duration_type="instant", field="totalAssets"),
        _row(2020, "Q2", 102.0, source_tag="us-gaap:Assets", duration_type="instant", field="totalAssets"),
        _row(2020, "Q3", 103.0, source_tag="us-gaap:Assets", duration_type="instant", field="totalAssets"),
        _row(2020, "Q4", 5.0, source_tag="us-gaap:AssetsHeldForSaleNotPartOfDisposalGroup",
            duration_type="instant", field="totalAssets"),   # wrong dimensioned slice, real bug pattern
    ]
    df = pd.DataFrame(rows)
    out = detect_level_outliers(df, "ZZZ", "totalAssets", duration_type="instant", check_yoy=False)
    assert not out.empty
    bad = out[(out["fiscal_year"] == 2020) & (out["fiscal_period"] == "Q4")]
    assert bad.iloc[0]["is_level_outlier"]

    mismatch = detect_source_tag_misalignment(df, "ZZZ", "totalAssets")
    assert len(mismatch) == 1
    assert mismatch.iloc[0]["mismatch_period_end_vs_interim"]
    assert mismatch.iloc[0]["period_end_source_tag"] == "us-gaap:AssetsHeldForSaleNotPartOfDisposalGroup"

    print("\n=== SANITY CHECK: fundamentals_facts audit tool ===")
    print("  derived Q4 rows pass through the level/YoY outlier stats without a bogus tag check;")
    print("  the first 4 periods (no defined YoY lag) never flagged as a YoY anomaly; a genuine")
    print("  10x single-quarter spike is still caught; an annual-vs-quarterly XBRL concept")
    print("  mismatch within the SAME fiscal year (real JPM bug) is flagged, while a clean,")
    print("  permanent taxonomy transition where annual and quarterly still agree each year")
    print("  (e.g. post-ASC-606) is correctly left unflagged.")
    print("  Validated.")
