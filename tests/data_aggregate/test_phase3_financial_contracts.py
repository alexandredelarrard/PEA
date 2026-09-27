"""Known-truth contracts for the Phase-3 fundamentals corrections."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_aggregate.utils.fundamentals.fundamental_features import _derived_fields
from src.data_aggregate.utils.fundamentals.sector_features import compute_sector_kpis


def test_bank_roa_uses_average_comparable_fiscal_assets_without_fallback() -> None:
    history = pd.DataFrame(
        [
            {
                "ticker": "BANK",
                "as_of": "2023-05-01",
                "fiscal_end": "2023-03-31",
                "sector": "Financials",
                "industry_group": "Banks",
                "totalAssets": 800.0,
                "netIncome": 12.0,
            },
            {
                "ticker": "BANK",
                "as_of": "2024-05-15",
                "fiscal_end": "2024-04-15",
                "sector": "Financials",
                "industry_group": "Banks",
                "totalAssets": 1_000.0,
                "netIncome": 20.0,
            },
        ]
    )

    result = compute_sector_kpis(history)

    assert pd.isna(result.loc[0, "bank_roa"])
    assert result.loc[1, "bank_roa"] == pytest.approx(20.0 / 900.0)

    print("\n=== SANITY CHECK: bank ROA definition ===")
    print("  first fiscal observation is null; second is 20 / mean(800, 1000) = " f"{result.loc[1, 'bank_roa']:.6f}. Validated.")


def test_capitalized_rd_roic_tax_adjusts_operating_profit() -> None:
    history = pd.DataFrame(
        [
            {
                "ticker": "RD",
                "as_of": "2024-05-01",
                "fiscal_end": "2024-03-31",
                "sector": "Health Care",
                "industry_group": "Pharmaceuticals, Biotechnology & Life Sciences",
                "researchAndDevelopment": 250.0,
                "operatingIncome": 300.0,
                "incomeTaxExpense": 60.0,
                "pretaxIncome": 300.0,
                "stockholdersEquity": 1_000.0,
                "longTermDebt": 400.0,
                "shortTermDebt": 100.0,
                "cash": 150.0,
            }
        ]
    )

    result = compute_sector_kpis(history).iloc[0]
    expected = ((300.0 + 250.0) * (1.0 - 0.20)) / (1_000.0 + 500.0 + 250.0 - 150.0)

    assert result["rd_capitalized_roic"] == pytest.approx(expected)

    print("\n=== SANITY CHECK: capitalized-R&D ROIC definition ===")
    print(f"  adjusted operating profit is tax-adjusted at 20%; ROIC={expected:.6f}. " "Validated.")


def test_earnings_yield_uses_positive_common_income_without_fallback() -> None:
    idx = pd.bdate_range("2024-05-01", "2024-05-10")
    base = {
        "ticker": "AAA",
        "as_of": "2024-05-01",
        "fiscal_end": "2024-03-31",
        "netIncome": 100.0,
        "netIncomeCommon": 60.0,
        "sharesOutstanding": 10.0,
    }
    close = pd.DataFrame({"AAA": 100.0, "MISS": 100.0}, index=idx)
    history = pd.DataFrame(
        [
            base,
            {
                **base,
                "ticker": "MISS",
                "netIncome": 100.0,
                "netIncomeCommon": None,
            },
        ]
    )

    fields = _derived_fields(history, idx, close)

    assert fields["earnings_yield"].loc[idx[-1], "AAA"] == pytest.approx(60.0 / 1_000.0)
    assert "MISS" not in fields["earnings_yield"].columns or pd.isna(fields["earnings_yield"].loc[idx[-1], "MISS"])

    print("\n=== SANITY CHECK: common-equity earnings yield ===")
    print("  AAA uses common income 60 / market cap 1000; MISS stays null rather than " "falling back to consolidated income 100. Validated.")


def test_invalid_gross_profit_sentinel_is_null_before_feature_transforms() -> None:
    idx = pd.bdate_range("2024-05-01", "2024-05-10")
    history = pd.DataFrame(
        [
            {
                "ticker": "BAD",
                "as_of": "2024-05-01",
                "fiscal_end": "2024-03-31",
                "totalRevenue": 100.0,
                "costOfRevenue": 0.0,
                "grossProfit": 100.0,
                "grossMargins": 1.0,
                "totalAssets": 200.0,
            }
        ]
    )

    fields = _derived_fields(history, idx, close=None)

    assert "grossMargins" not in fields or fields["grossMargins"]["BAD"].isna().all()
    assert "gross_profitability" not in fields or fields["gross_profitability"]["BAD"].isna().all()
    assert history.loc[0, "grossMargins"] == 1.0, "aggregate sanitization mutated raw input"

    print("\n=== SANITY CHECK: impossible gross-margin sentinel ===")
    print(
        "  positive revenue + zero cost + unit gross margin is null before derived/rank "
        "features, while the caller's raw frame remains unchanged. Validated."
    )


def test_piotroski_uses_prior_fiscal_year_not_a_daily_grid_offset() -> None:
    idx = pd.bdate_range("2023-05-01", "2024-08-09")
    rows = [
        {
            "ticker": "AAA",
            "as_of": "2023-05-01",
            "fiscal_end": "2023-03-31",
            "totalAssets": 100.0,
            "netIncome": 10.0,
            "operatingCashFlow": 15.0,
            "longTermDebt": 30.0,
            "currentAssets": 50.0,
            "currentLiabilities": 25.0,
            "sharesOutstanding": 100.0,
            "grossMargins": 0.40,
            "totalRevenue": 100.0,
        },
        # An intervening quarter is deliberately stronger than the current row. A daily
        # 252-session offset from the late current filing lands here and gets the signs wrong.
        {
            "ticker": "AAA",
            "as_of": "2023-08-01",
            "fiscal_end": "2023-06-30",
            "totalAssets": 100.0,
            "netIncome": 30.0,
            "operatingCashFlow": 35.0,
            "longTermDebt": 10.0,
            "currentAssets": 80.0,
            "currentLiabilities": 20.0,
            "sharesOutstanding": 80.0,
            "grossMargins": 0.60,
            "totalRevenue": 200.0,
        },
        {
            "ticker": "AAA",
            "as_of": "2024-08-01",
            "fiscal_end": "2024-03-30",
            "totalAssets": 100.0,
            "netIncome": 20.0,
            "operatingCashFlow": 25.0,
            "longTermDebt": 20.0,
            "currentAssets": 60.0,
            "currentLiabilities": 20.0,
            "sharesOutstanding": 100.0,
            "grossMargins": 0.50,
            "totalRevenue": 120.0,
        },
    ]

    fields = _derived_fields(pd.DataFrame(rows), idx, close=None)
    score = fields["piotroski_f_score"].loc[pd.Timestamp("2024-08-02"), "AAA"]

    assert score == pytest.approx(9.0)

    print("\n=== SANITY CHECK: Piotroski fiscal predecessor ===")
    print(
        "  late 2024 filing compares with fiscal 2023-03-31, not the intervening strong "
        f"quarter selected by a 252-session offset; F-score={score:.0f}/9. Validated."
    )


def test_quarterly_projection_expires_on_day_186() -> None:
    filed = pd.Timestamp("2024-01-01")
    idx = pd.date_range(filed, filed + pd.Timedelta(days=186))
    history = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "as_of": filed,
                "fiscal_end": "2023-12-31",
                "grossMargins": 0.40,
            }
        ]
    )

    fields = _derived_fields(history, idx, close=None)
    gross_margin = fields["grossMargins"]["AAA"]

    assert gross_margin.loc[filed + pd.Timedelta(days=185)] == pytest.approx(0.40)
    assert pd.isna(gross_margin.loc[filed + pd.Timedelta(days=186)])

    print("\n=== SANITY CHECK: quarterly feature freshness ===")
    print("  source value is alive on day 185 and null on day 186. Validated.")
