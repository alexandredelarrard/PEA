"""Tests for the FMP historical-endpoint normalizers
(src/data_extract/fetch_fmp_history.py). No network -- synthetic payloads shaped
exactly like the live FMP responses (verified via probe).
"""
from __future__ import annotations

import pandas as pd

from src.data_extract.fetch_fmp_history import (
    normalize_grades, normalize_actions, normalize_exec_comp, normalize_estimates,
)


def test_normalize_grades_monthly_distribution():
    recs = [
        {"symbol": "AAPL", "date": "2026-07-01", "analystRatingsStrongBuy": 6,
         "analystRatingsBuy": 20, "analystRatingsHold": 17, "analystRatingsSell": 2,
         "analystRatingsStrongSell": 1},
        {"symbol": "AAPL", "date": "2026-06-01", "analystRatingsStrongBuy": 7,
         "analystRatingsBuy": 19, "analystRatingsHold": 16, "analystRatingsSell": 2,
         "analystRatingsStrongSell": 1},
        {"symbol": "AAPL", "date": None, "analystRatingsBuy": 5},   # no date -> drop
    ]
    df = normalize_grades(recs, "AAPL")
    assert len(df) == 2 and df["as_of"].is_monotonic_increasing
    assert df.iloc[-1]["strong_buy"] == 6 and df.iloc[-1]["hold"] == 17
    assert df.iloc[-1]["as_of"] == pd.Timestamp("2026-07-01")
    assert normalize_grades([], "AAPL").empty

    print("\n=== SANITY CHECK: analyst grades (monthly) ===")
    print(f"  2 monthly rows, latest 2026-07-01 SB=6/Hold=17; undated dropped. Validated.")


def test_normalize_actions_events_dedup():
    recs = [
        {"date": "2026-06-25", "gradingCompany": "Evercore ISI", "previousGrade": "Buy",
         "newGrade": "Buy", "action": "maintain"},
        {"date": "2026-06-22", "gradingCompany": "KGI Securities", "previousGrade": "Buy",
         "newGrade": "Hold", "action": "downgrade"},
        # exact duplicate event -> collapsed
        {"date": "2026-06-25", "gradingCompany": "Evercore ISI", "previousGrade": "Buy",
         "newGrade": "Buy", "action": "maintain"},
    ]
    df = normalize_actions(recs, "AAPL")
    assert len(df) == 2
    dg = df[df["action"] == "downgrade"].iloc[0]
    assert dg["grading_company"] == "KGI Securities" and dg["new_grade"] == "Hold"

    print("\n=== SANITY CHECK: analyst actions (events) ===")
    print("  2 distinct dated events (dup collapsed); downgrade parsed. Validated.")


def test_normalize_exec_comp_annual_pit():
    recs = [
        {"symbol": "AAPL", "acceptedDate": "2025-01-08", "filingDate": "2025-01-07",
         "year": 2024, "nameAndPosition": "Tim Cook Chief Executive Officer",
         "salary": 3000000, "bonus": 0, "stockAward": 58000000, "total": 74609802},
        {"symbol": "AAPL", "acceptedDate": "2024-01-10", "filingDate": "2024-01-09",
         "year": 2023, "nameAndPosition": "Tim Cook Chief Executive Officer",
         "salary": 3000000, "bonus": 0, "stockAward": 46000000, "total": 63209845},
    ]
    df = normalize_exec_comp(recs, "AAPL")
    assert len(df) == 2
    ceo24 = df[df["fiscal_year"] == 2024].iloc[0]
    # as_of is the public acceptance date (not the fiscal year)
    assert ceo24["as_of"] == pd.Timestamp("2025-01-08")
    assert ceo24["total"] == 74609802
    assert "Chief Executive Officer" in ceo24["name_and_position"]

    print("\n=== SANITY CHECK: executive compensation (annual) ===")
    print("  FY2024 Cook total=74.6M, public as_of 2025-01-08 (proxy accept date). Validated.")


def test_normalize_estimates_snapshot_pull_date():
    recs = [
        {"symbol": "AAPL", "date": "2027-09-27", "epsAvg": 9.5, "epsHigh": 10.1,
         "epsLow": 8.9, "revenueAvg": 4.7e11, "ebitdaAvg": 1.6e11,
         "netIncomeAvg": 1.2e11, "numAnalystsEps": 20, "numAnalystsRevenue": 22},
        {"symbol": "AAPL", "date": "2026-09-27", "epsAvg": 7.9, "epsHigh": 8.3,
         "epsLow": 7.4, "revenueAvg": 4.2e11, "ebitdaAvg": 1.4e11,
         "netIncomeAvg": 1.0e11, "numAnalystsEps": 28, "numAnalystsRevenue": 30},
    ]
    df = normalize_estimates(recs, "AAPL")
    today = pd.Timestamp.today().normalize()
    assert len(df) == 2
    # no per-row estimate date -> as_of stamped with the pull date (accrues PIT)
    assert (df["as_of"] == today).all()
    fwd = df[df["fiscal_date"] == pd.Timestamp("2027-09-27")].iloc[0]
    assert fwd["eps_avg"] == 9.5 and fwd["num_analysts_eps"] == 20

    print("\n=== SANITY CHECK: analyst estimates (annual snapshot) ===")
    print(f"  fiscal FY2027 epsAvg=9.5 (#analysts=20); as_of=pull date {today.date()} "
          f"-> accrues point-in-time. Validated.")
