"""Tier-A quick-win factors + the sector KPIs that survived the Sharadar substrate.

Universal (fundamental_features): A2 asset growth, A3 Piotroski F-score, A5 Rule of 40.
Sector (sector_features): A6 book-value growth, A8 REIT AFFO dividend coverage, B1 AOCI.

WHAT THIS FILE USED TO ASSERT AND NO LONGER CAN. The bank NII/loan growth pair (A7), the
insurance premium/float growth pair (A6), the HTM unrealized-loss ratio (B1) and the
NPL/net-charge-off pair (B3) all read `netInterestIncome`, `loans`, `premiumsWritten`,
`insuranceReserves`, `htmSecurities`, `nonaccrualLoans` or `netChargeOffs`. Sharadar SF1
carries none of them, so every one measured ZERO values against the live table. `rpo_growth`
went the same way (`remainingPerformanceObligation`). The tests passed because the fixture
hand-built the missing columns.
"""
from __future__ import annotations

import pandas as pd

from src.data_aggregate.utils.fundamentals.sector_features import compute_sector_kpis
from src.data_aggregate.utils.fundamentals.fundamental_features import _derived_fields


def test_sector_quick_wins_that_sharadar_can_feed():
    rows = []
    for yr, m in [(2022, 1.0), (2023, 1.2)]:                    # +20% YoY on growth items
        rows += [
            {"ticker": "BANK", "sector": "Financials", "industry_group": "Banks",
             "as_of": f"{yr}-12-31", "stockholdersEquity": 300.0,
             "totalAssets": 2000.0, "netIncome": 50.0,
             "accumulatedOtherComprehensiveIncome": -30.0},
            {"ticker": "INS", "sector": "Financials", "industry_group": "Insurance",
             "as_of": f"{yr}-12-31", "stockholdersEquity": 400 * m,
             "sellingGeneralAdmin": 40.0, "totalAssets": 1500.0, "netIncome": 60.0},
            {"ticker": "REIT", "sector": "Real Estate",
             "industry_group": "Equity Real Estate Investment Trusts (REITs)",
             "as_of": f"{yr}-12-31",
             "netIncome": 100.0, "depAmort": 150.0,
             "capex": 40.0, "dividendsPaid": 180.0,
             "totalRevenue": 420.0, "operatingIncome": 200.0},
        ]
    k = compute_sector_kpis(pd.DataFrame(rows))
    last = k[k["as_of"] == "2023-12-31"].set_index("ticker")

    # A6 book-value growth, financials-scoped so BOTH the bank and the insurer are in scope
    assert abs(last.loc["INS", "book_value_growth"] - 0.20) < 1e-6
    # A8: FFO = 100 + 150 = 250 (no disposal-gain leg in SF1); AFFO = 250 - 40 = 210
    assert abs(last.loc["REIT", "affo_dividend_coverage"] - 210.0 / 180.0) < 1e-6
    # B1: AOCI / equity = -30/300 = -0.10 -- the surviving half of the SVB signal
    assert abs(last.loc["BANK", "aoci_to_equity"] - (-0.10)) < 1e-6
    # the bank ROA that replaced the NIM/efficiency/provision family
    assert abs(last.loc["BANK", "bank_roa"] - 50.0 / 2000.0) < 1e-6

    for gone in ("nii_growth", "loan_growth", "premium_growth", "float_growth",
                 "htm_unrealized_loss_ratio", "npl_ratio", "net_charge_off_rate"):
        assert gone not in k.columns, f"{gone} needs a tag SF1 does not deliver"

    print("\n=== SANITY: the sector quick-wins Sharadar CAN feed ===")
    print(f"  INS book-value growth=+{last.loc['INS','book_value_growth']:.0%}; "
          f"BANK AOCI/eq={last.loc['BANK','aoci_to_equity']:.2f} (unrealized securities "
          f"losses eroding capital), ROA={last.loc['BANK','bank_roa']:.3f}; "
          f"REIT AFFO coverage={last.loc['REIT','affo_dividend_coverage']:.2f}x.")
    print("  The NII/loan/premium/float/HTM/NPL/NCO KPIs are absent, not NaN -- SF1 carries "
          "none of their inputs. Validated.")


def test_universal_quick_wins_A2_A3_A5():
    idx = pd.bdate_range("2020-06-01", "2024-06-30")             # extends past the last filing
    base = dict(totalRevenue=1000, totalAssets=5000, netIncome=100, operatingCashFlow=130,
                currentAssets=800, currentLiabilities=400, longTermDebt=1000,
                sharesOutstanding=100, grossMargins=0.40, freeCashflow=120)
    impr = dict(totalRevenue=1200, totalAssets=5250, netIncome=130, operatingCashFlow=170,
                currentAssets=900, currentLiabilities=400, longTermDebt=900,
                sharesOutstanding=100, grossMargins=0.44, freeCashflow=180)
    rows = ([{"ticker": "AAA", "as_of": f"{yr}-12-31", **base} for yr in (2021, 2022)]
            + [{"ticker": "AAA", "as_of": "2023-12-31", **impr}])
    F = _derived_fields(pd.DataFrame(rows), idx, pd.DataFrame({"AAA": 100.0}, index=idx))
    d = idx[-1]

    assert abs(F["asset_growth"].loc[d, "AAA"] - 0.05) < 1e-6              # A2: 5250/5000-1
    assert abs(F["rule_of_40"].loc[d, "AAA"] - (20.0 + 15.0)) < 1e-3      # A5: 20% growth + 15% FCF margin
    assert F["piotroski_f_score"].loc[d, "AAA"] >= 8                       # A3: everything improving
    assert "rpo_growth" not in F      # remainingPerformanceObligation is not an SF1 column

    print("\n=== SANITY: universal A2/A3/A5 ===")
    print(f"  asset_growth={F['asset_growth'].loc[d,'AAA']:.2f} (CMA), "
          f"rule_of_40={F['rule_of_40'].loc[d,'AAA']:.0f}, "
          f"Piotroski F={F['piotroski_f_score'].loc[d,'AAA']:.0f}/9. "
          f"rpo_growth absent (no RPO tag in SF1). Validated.")
