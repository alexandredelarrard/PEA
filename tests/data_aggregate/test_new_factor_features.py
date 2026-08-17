"""Tier-A quick-win factors + B1/B3 sector extraction KPIs.

Universal (fundamental_features): A2 asset growth, A3 Piotroski F-score, A5 Rule
of 40 + RPO growth. Sector (sector_features): A6 insurance book-value/premium/
float growth, A7 bank NII/loan growth, A8 REIT AFFO dividend coverage, B1 AOCI /
HTM unrealized-loss ratios, B3 NPL / net-charge-off rates.
"""
from __future__ import annotations

import pandas as pd

from src.data_aggregate.utils.fundamentals.sector_features import compute_sector_kpis
from src.data_aggregate.utils.fundamentals.fundamental_features import _derived_fields


def test_sector_quick_wins_and_B_kpis():
    rows = []
    for yr, m in [(2022, 1.0), (2023, 1.2)]:                    # +20% YoY on growth items
        rows += [
            {"ticker": "BANK", "sector": "Financials", "industry_group": "Banks",
             "as_of": f"{yr}-12-31", "netInterestIncome": 100 * m,
             "loans": 1000 * m, "deposits": 1200.0, "stockholdersEquity": 300.0,
             "totalAssets": 2000.0, "netIncome": 50.0, "accumulatedOCI": -30.0,
             "htmSecurities": 500.0, "htmSecuritiesFairValue": 450.0,
             "nonaccrualLoans": 20.0, "netChargeOffs": 10.0},
            {"ticker": "INS", "sector": "Financials", "industry_group": "Insurance",
             "as_of": f"{yr}-12-31", "premiumsEarned": 200.0,
             "premiumsWritten": 210 * m, "insuranceReserves": 800 * m,
             "stockholdersEquity": 400 * m, "claimsIncurred": 120.0,
             "sellingGeneralAdmin": 40.0, "totalAssets": 1500.0, "netIncome": 60.0},
            {"ticker": "REIT", "sector": "Real Estate",
             "industry_group": "Equity Real Estate Investment Trusts (REITs)",
             "as_of": f"{yr}-12-31", "realEstateNet": 5000.0,
             "rentalIncome": 400.0, "netIncome": 100.0, "depAmort": 150.0,
             "gainOnDispositions": 10.0, "capex": 40.0, "dividendsPaid": 180.0,
             "totalRevenue": 420.0, "operatingIncome": 200.0},
        ]
    k = compute_sector_kpis(pd.DataFrame(rows))
    last = k[k["as_of"] == "2023-12-31"].set_index("ticker")

    assert abs(last.loc["BANK", "nii_growth"] - 0.20) < 1e-6        # A7
    assert abs(last.loc["BANK", "loan_growth"] - 0.20) < 1e-6
    assert abs(last.loc["INS", "book_value_growth"] - 0.20) < 1e-6  # A6
    assert abs(last.loc["INS", "premium_growth"] - 0.20) < 1e-6
    assert abs(last.loc["INS", "float_growth"] - 0.20) < 1e-6
    # A8: FFO=100+150-10=240; AFFO=240-40=200; /180 dividends = 1.111
    assert abs(last.loc["REIT", "affo_dividend_coverage"] - 200.0 / 180.0) < 1e-6
    # B1: AOCI/equity=-30/300=-0.10; HTM loss=(500-450)/300=0.1667
    assert abs(last.loc["BANK", "aoci_to_equity"] - (-0.10)) < 1e-6
    assert abs(last.loc["BANK", "htm_unrealized_loss_ratio"] - 50.0 / 300.0) < 1e-6
    # B3: loans(2023)=1200 -> NPL=20/1200, NCO=10/1200
    assert abs(last.loc["BANK", "npl_ratio"] - 20.0 / 1200.0) < 1e-6
    assert abs(last.loc["BANK", "net_charge_off_rate"] - 10.0 / 1200.0) < 1e-6

    print("\n=== SANITY: sector A6/A7/A8 + B1/B3 KPIs ===")
    print(f"  BANK nii/loan growth=+20%; AOCI/eq={last.loc['BANK','aoci_to_equity']:.2f}, "
          f"HTM loss/eq={last.loc['BANK','htm_unrealized_loss_ratio']:.3f} (SVB signal), "
          f"NPL={last.loc['BANK','npl_ratio']:.4f}; INS book-value+20%; "
          f"REIT AFFO coverage={last.loc['REIT','affo_dividend_coverage']:.2f}x. Validated.")


def test_universal_quick_wins_A2_A3_A5():
    idx = pd.bdate_range("2020-06-01", "2024-06-30")             # extends past the last filing
    base = dict(totalRevenue=1000, totalAssets=5000, netIncome=100, operatingCashFlow=130,
                currentAssets=800, currentLiabilities=400, longTermDebt=1000,
                sharesOutstanding=100, grossMargins=0.40, freeCashflow=120,
                remainingPerformanceObligation=300)
    impr = dict(totalRevenue=1200, totalAssets=5250, netIncome=130, operatingCashFlow=170,
                currentAssets=900, currentLiabilities=400, longTermDebt=900,
                sharesOutstanding=100, grossMargins=0.44, freeCashflow=180,
                remainingPerformanceObligation=450)
    rows = ([{"ticker": "AAA", "as_of": f"{yr}-12-31", **base} for yr in (2021, 2022)]
            + [{"ticker": "AAA", "as_of": "2023-12-31", **impr}])
    F = _derived_fields(pd.DataFrame(rows), idx, pd.DataFrame({"AAA": 100.0}, index=idx))
    d = idx[-1]

    assert abs(F["asset_growth"].loc[d, "AAA"] - 0.05) < 1e-6              # A2: 5250/5000-1
    assert abs(F["rule_of_40"].loc[d, "AAA"] - (20.0 + 15.0)) < 1e-3      # A5: 20% growth + 15% FCF margin
    assert abs(F["rpo_growth"].loc[d, "AAA"] - 0.5) < 1e-6                # A5: 450/300-1
    assert F["piotroski_f_score"].loc[d, "AAA"] >= 8                       # A3: everything improving

    print("\n=== SANITY: universal A2/A3/A5 ===")
    print(f"  asset_growth={F['asset_growth'].loc[d,'AAA']:.2f} (CMA), "
          f"rule_of_40={F['rule_of_40'].loc[d,'AAA']:.0f}, rpo_growth={F['rpo_growth'].loc[d,'AAA']:.2f}, "
          f"Piotroski F={F['piotroski_f_score'].loc[d,'AAA']:.0f}/9. Validated.")
