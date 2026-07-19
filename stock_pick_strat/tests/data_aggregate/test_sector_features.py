"""
Sector-specific KPI math (src/data_aggregate/utils/sector_features.py).

Synthetic, known-truth fundamentals rows (one per sector) with hand-computed
expected ratios — the right tool to prove each KPI formula and, crucially, that
KPIs are AVAILABILITY-GATED (a sector's KPI is NaN when its inputs weren't
reported, so a bank never gets a loss_ratio, an industrial never gets NIM).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.sector_features import compute_sector_kpis


def _fundamentals() -> pd.DataFrame:
    rows = [
        # ---- Bank ---- (+ new: allowance / tier1 / domestic deposits / total liabs)
        {"ticker": "BANK", "sector": "Financials", "totalAssets": 1000.0, "netIncome": 20.0,
         "netInterestIncome": 40.0, "noninterestIncome": 10.0, "noninterestExpense": 30.0,
         "loans": 800.0, "deposits": 900.0, "provisionForCreditLosses": 8.0,
         "allowanceCreditLosses": 16.0, "tier1CapitalRatio": 0.12, "depositsDomestic": 700.0,
         "totalLiabilities": 920.0},
        # ---- Insurer ---- (+ net investment income)
        {"ticker": "INSR", "sector": "Financials", "premiumsEarned": 500.0,
         "claimsIncurred": 350.0, "sellingGeneralAdmin": 100.0, "dacAmortization": 20.0,
         "netInvestmentIncome": 60.0},
        # ---- REIT ---- (+ operatingIncome / debt / cash / capex for AFFO & EBITDAre)
        {"ticker": "REIT", "sector": "Real Estate", "netIncome": 50.0, "depAmort": 100.0,
         "gainOnDispositions": 10.0, "totalRevenue": 300.0, "rentalIncome": 280.0,
         "dividendsPaid": 120.0, "realEstateNet": 2000.0, "operatingIncome": 90.0,
         "longTermDebt": 800.0, "shortTermDebt": 100.0, "cash": 50.0, "capex": 40.0},
        # ---- Industrial (universal KPIs) ----
        {"ticker": "INDU", "sector": "Industrials", "totalRevenue": 1000.0, "costOfRevenue": 600.0,
         "grossProfit": 400.0, "accountsReceivable": 200.0, "inventory": 150.0,
         "accountsPayable": 100.0, "totalAssets": 2000.0, "ebitda": 250.0, "interestExpense": 25.0,
         "netIncome": 120.0, "operatingCashFlow": 180.0, "incomeTaxExpense": 30.0,
         "pretaxIncome": 150.0, "capex": 80.0, "depAmort": 60.0, "dividendsPaid": 40.0,
         "buybacks": 60.0, "longTermDebt": 500.0, "shortTermDebt": 50.0, "cash": 100.0,
         "shortTermInvestments": 50.0, "operatingIncome": 180.0, "stockholdersEquity": 900.0,
         "returnOnEquity": 0.13},
        # ---- Utility ---- (rate-base growth proxy)
        {"ticker": "UTIL", "sector": "Utilities", "totalAssets": 5000.0, "capex": 400.0,
         "regulatoryAssets": 600.0, "goodwill": 200.0},
        # ---- Pharma (single period; capitalized-R&D has no history here) ----
        {"ticker": "PHRM", "sector": "Health Care", "researchAndDevelopment": 250.0,
         "amortizationIntangibles": 80.0, "operatingCashFlow": 200.0, "operatingIncome": 300.0,
         "stockholdersEquity": 1000.0, "longTermDebt": 400.0, "shortTermDebt": 100.0, "cash": 150.0},
        # ---- Oil & gas ----
        {"ticker": "OILX", "sector": "Energy", "oilGasPropertyNet": 5000.0, "explorationExpense": 70.0,
         "operatingIncome": 180.0, "depAmort": 110.0, "operatingCashFlow": 250.0, "totalRevenue": 1000.0},
    ]
    return pd.DataFrame(rows)


def test_universal_kpis():
    k = compute_sector_kpis(_fundamentals()).set_index("ticker")
    r = k.loc["INDU"]
    assert r["effective_tax_rate"] == pytest.approx(0.20)
    assert r["interest_coverage"] == pytest.approx(10.0)
    assert r["net_debt_to_ebitda"] == pytest.approx((500 + 50 - 100 - 50) / 250)   # 1.6
    assert r["accruals_ratio"] == pytest.approx((120 - 180) / 2000)                 # -0.03
    assert r["gross_profitability"] == pytest.approx(400 / 2000)                    # 0.20
    assert r["asset_turnover"] == pytest.approx(0.50)
    assert r["capex_intensity"] == pytest.approx(0.08)
    assert r["capex_to_dep"] == pytest.approx(80 / 60)
    assert r["payout_ratio"] == pytest.approx((40 + 60) / 120)                      # 0.8333
    assert r["buyback_intensity"] == pytest.approx(0.06)
    assert r["days_sales_outstanding"] == pytest.approx(200 * 365 / 1000)           # 73
    assert r["days_inventory_outstanding"] == pytest.approx(150 * 365 / 600)        # 91.25
    assert r["days_payable_outstanding"] == pytest.approx(100 * 365 / 600)          # 60.833
    assert r["cash_conversion_cycle"] == pytest.approx(73 + 91.25 - (100 * 365 / 600))

    print("\n=== SANITY CHECK: universal KPIs (industrial) ===")
    print(f"  tax={r['effective_tax_rate']:.2f} int_cov={r['interest_coverage']:.1f} "
          f"netdebt/EBITDA={r['net_debt_to_ebitda']:.2f} accruals={r['accruals_ratio']:.3f}")
    print(f"  CCC={r['cash_conversion_cycle']:.1f}d (DSO {r['days_sales_outstanding']:.0f} + "
          f"DIO {r['days_inventory_outstanding']:.1f} - DPO {r['days_payable_outstanding']:.1f}). Validated.")


def test_bank_kpis():
    k = compute_sector_kpis(_fundamentals()).set_index("ticker")
    r = k.loc["BANK"]
    assert r["net_interest_margin"] == pytest.approx(40 / 1000)          # 0.04
    assert r["efficiency_ratio"] == pytest.approx(30 / (40 + 10))        # 0.60
    assert r["provision_rate"] == pytest.approx(8 / 800)                 # 0.01
    assert r["loan_to_deposit"] == pytest.approx(800 / 900)
    assert r["bank_roa"] == pytest.approx(20 / 1000)                     # 0.02
    print("\n=== SANITY CHECK: bank KPIs ===")
    print(f"  NIM={r['net_interest_margin']:.3f} efficiency={r['efficiency_ratio']:.2f} "
          f"provision={r['provision_rate']:.3f} L/D={r['loan_to_deposit']:.3f} ROA={r['bank_roa']:.3f}. Validated.")


def test_insurance_kpis():
    k = compute_sector_kpis(_fundamentals()).set_index("ticker")
    r = k.loc["INSR"]
    assert r["loss_ratio"] == pytest.approx(350 / 500)                   # 0.70
    assert r["expense_ratio"] == pytest.approx((100 + 20) / 500)         # 0.24
    assert r["combined_ratio"] == pytest.approx(0.70 + 0.24)             # 0.94 (<1 => profit)
    print("\n=== SANITY CHECK: insurance KPIs ===")
    print(f"  loss={r['loss_ratio']:.2f} expense={r['expense_ratio']:.2f} "
          f"combined={r['combined_ratio']:.2f} (<1.0 = underwriting profit). Validated.")


def test_reit_kpis():
    k = compute_sector_kpis(_fundamentals()).set_index("ticker")
    r = k.loc["REIT"]
    ffo = 50 + 100 - 10                                                  # 140
    assert r["ffo_margin"] == pytest.approx(ffo / 300)
    assert r["ffo_payout"] == pytest.approx(120 / ffo)
    assert r["rental_margin"] == pytest.approx(280 / 300)
    print("\n=== SANITY CHECK: REIT KPIs ===")
    print(f"  FFO={ffo} margin={r['ffo_margin']:.3f} payout={r['ffo_payout']:.3f} "
          f"rental_margin={r['rental_margin']:.3f}. Validated.")


def test_capital_efficiency_kpis():
    r = compute_sector_kpis(_fundamentals()).set_index("ticker").loc["INDU"]
    # NOPAT = EBIT 180 * (1 - tax 0.20) = 144 ; invested capital = 550 debt + 900 eq - 100 cash
    assert r["roic"] == pytest.approx(144.0 / (550 + 900 - 100))          # 0.1067
    assert r["earnings_quality"] == pytest.approx(180.0 / 120.0)         # 1.5 (OCF/NI)
    assert r["fixed_cost_coverage_margin"] == pytest.approx((400 - 250) / 1000)  # 0.15
    assert r["sustainable_growth_rate"] == pytest.approx(0.13 * (1 - 40 / 120))  # ROE*(1-payout)
    assert r["gmroi"] == pytest.approx(400.0 / 150.0)                    # 2.667 (single period)
    assert r["asset_turnover"] == pytest.approx(0.50)                    # single row -> period-end
    print("\n=== SANITY CHECK: capital-efficiency KPIs ===")
    print(f"  ROIC={r['roic']:.4f} earnings_quality={r['earnings_quality']:.2f} "
          f"fixed_cost_margin={r['fixed_cost_coverage_margin']:.2f} SGR={r['sustainable_growth_rate']:.4f} "
          f"GMROI={r['gmroi']:.2f}. Validated.")


def test_reinvestment_rate_multiperiod():
    """Reinvestment rate needs the ΔNWC over a year -> a ticker's own history."""
    yrs = pd.date_range("2023-12-31", periods=2, freq="YE")
    df = pd.DataFrame([
        {"ticker": "RE1", "as_of": yrs[0], "operatingIncome": 200.0, "incomeTaxExpense": 40.0,
         "pretaxIncome": 200.0, "capex": 100.0, "depAmort": 60.0,
         "currentAssets": 500.0, "currentLiabilities": 300.0},   # NWC = 200
        {"ticker": "RE1", "as_of": yrs[1], "operatingIncome": 200.0, "incomeTaxExpense": 40.0,
         "pretaxIncome": 200.0, "capex": 100.0, "depAmort": 60.0,
         "currentAssets": 560.0, "currentLiabilities": 300.0},   # NWC = 260 -> ΔNWC = +60
    ])
    k = compute_sector_kpis(df).set_index("as_of")
    last = k.iloc[-1]
    nopat = 200 * (1 - 40 / 200)                                  # 160
    # reinvestment = (capex 100 - D&A 60 + ΔNWC 60) / NOPAT 160 = 100/160
    assert last["reinvestment_rate"] == pytest.approx((100 - 60 + 60) / nopat)
    assert pd.isna(k.iloc[0]["reinvestment_rate"])               # first year: no prior NWC
    print("\n=== SANITY CHECK: reinvestment rate (multi-period) ===")
    print(f"  (capex-D&A+dNWC)/NOPAT = (100-60+60)/160 = {last['reinvestment_rate']:.3f}; "
          f"first year NaN (no prior). Validated.")


def test_new_bank_kpis():
    r = compute_sector_kpis(_fundamentals()).set_index("ticker").loc["BANK"]
    assert r["bank_operating_margin"] == pytest.approx((40 + 10 - 8 - 30) / (40 + 10))  # 0.24
    assert r["tier1_capital_ratio"] == pytest.approx(0.12)
    assert r["deposit_stickiness"] == pytest.approx(700 / 920)                          # 0.7609
    print("\n=== SANITY CHECK: new bank KPIs ===")
    print(f"  op_margin={r['bank_operating_margin']:.3f} tier1={r['tier1_capital_ratio']:.2f} "
          f"deposit_stickiness={r['deposit_stickiness']:.3f}. Validated.")


def test_utility_kpi():
    r = compute_sector_kpis(_fundamentals()).set_index("ticker").loc["UTIL"]
    assert r["capex_to_rate_base"] == pytest.approx(400 / (5000 - 600 - 200))           # 0.09524
    assert r["regulatory_asset_ratio"] == pytest.approx(600 / 5000)
    print("\n=== SANITY CHECK: utility rate-base KPI ===")
    print(f"  capex_to_rate_base={r['capex_to_rate_base']:.4f} "
          f"(capex / clean assets ex reg-assets & goodwill). Validated.")


def test_pharma_kpis_single_period():
    r = compute_sector_kpis(_fundamentals()).set_index("ticker").loc["PHRM"]
    assert r["patent_cliff"] == pytest.approx(80 / 200)                                 # 0.40
    # single period: R&D asset = current R&D only (no prior layers), amortization = 0
    adj_oi = 300 + 250 - 0
    adj_cap = 1000 + (400 + 100) + 250 - 150
    assert r["rd_capitalized_roic"] == pytest.approx(adj_oi / adj_cap)                  # 0.34375
    print("\n=== SANITY CHECK: pharma KPIs ===")
    print(f"  patent_cliff={r['patent_cliff']:.2f} rd_capitalized_roic={r['rd_capitalized_roic']:.4f}. Validated.")


def test_insurance_investment_income():
    r = compute_sector_kpis(_fundamentals()).set_index("ticker").loc["INSR"]
    assert r["investment_income_ratio"] == pytest.approx(60 / 500)                      # 0.12
    print("\n=== SANITY CHECK: insurance investment-income ratio ===")
    print(f"  investment_income_ratio={r['investment_income_ratio']:.3f}. Validated.")


def test_reit_affo_and_leverage():
    r = compute_sector_kpis(_fundamentals()).set_index("ticker").loc["REIT"]
    ffo = 50 + 100 - 10                                                                 # 140
    assert r["affo_margin"] == pytest.approx((ffo - 40) / 300)                          # 0.3333
    assert r["net_debt_to_ebitdare"] == pytest.approx((800 + 100 - 50) / (90 + 100))    # 4.4737
    print("\n=== SANITY CHECK: REIT AFFO & EBITDAre leverage ===")
    print(f"  affo_margin={r['affo_margin']:.3f} net_debt/EBITDAre={r['net_debt_to_ebitdare']:.3f}. Validated.")


def test_energy_kpis():
    r = compute_sector_kpis(_fundamentals()).set_index("ticker").loc["OILX"]
    assert r["ebitdax_margin"] == pytest.approx((180 + 110 + 70) / 1000)                # 0.36
    assert r["property_overvaluation_cushion"] == pytest.approx(5000 / (250 * 4))       # 5.0
    print("\n=== SANITY CHECK: oil & gas KPIs ===")
    print(f"  ebitdax_margin={r['ebitdax_margin']:.2f} property_cushion={r['property_overvaluation_cushion']:.2f}x. Validated.")


def test_multiperiod_reserve_velocity_and_capitalized_rd():
    """The two KPIs that need a ticker's OWN history: bank reserve-build velocity
    (QoQ provision change / allowance) and the 5-year capitalized-R&D pool."""
    # bank: 4 quarterly filings, provisions stepping up
    bank = pd.DataFrame([
        {"ticker": "BK2", "as_of": q, "netInterestIncome": 40.0, "loans": 800.0,
         "provisionForCreditLosses": p, "allowanceCreditLosses": 20.0}
        for q, p in zip(pd.date_range("2024-03-31", periods=4, freq="QE"), [8.0, 9.0, 12.0, 13.0])
    ])
    kb = compute_sector_kpis(bank).set_index("as_of")
    # last QoQ jump = 13 - 12 = 1.0 over allowance 20 -> 0.05
    assert kb["reserve_coverage_velocity"].iloc[-1] == pytest.approx(1.0 / 20.0)
    assert pd.isna(kb["reserve_coverage_velocity"].iloc[0])                             # no prior quarter

    # pharma: 5 yearly filings of R&D 100..100 (flat) -> asset = 100*(1+.8+.6+.4+.2)=300,
    # amort = 100*0.2*5 = 100 (5 prior layers) at the last filing.
    yrs = pd.date_range("2021-12-31", periods=6, freq="YE")
    ph = pd.DataFrame([
        {"ticker": "PH2", "as_of": y, "researchAndDevelopment": 100.0, "operatingIncome": 300.0,
         "stockholdersEquity": 1000.0, "longTermDebt": 0.0, "shortTermDebt": 0.0, "cash": 0.0}
        for y in yrs
    ])
    kp = compute_sector_kpis(ph).set_index("as_of")
    last = kp.iloc[-1]
    # asset pool at last row = 100*(1.0+0.8+0.6+0.4+0.2) = 300 ; amort = 100*0.2*5 = 100
    adj_oi = 300 + 100 - 100
    adj_cap = 1000 + 0 + 300 - 0
    assert last["rd_capitalized_roic"] == pytest.approx(adj_oi / adj_cap)               # 300/1300
    print("\n=== SANITY CHECK: multi-period reserve velocity & capitalized R&D ===")
    print(f"  reserve_velocity(last)={kb['reserve_coverage_velocity'].iloc[-1]:.3f} "
          f"| rd_capitalized_roic(5y flat R&D)={last['rd_capitalized_roic']:.4f} (asset 300, amort 100). Validated.")


def test_kpis_are_availability_gated():
    """A sector KPI must be NaN when its inputs weren't reported (the sector gate)."""
    k = compute_sector_kpis(_fundamentals()).set_index("ticker")
    # bank/industrial never report premiums -> no loss ratio
    assert np.isnan(k.loc["BANK", "loss_ratio"])
    assert np.isnan(k.loc["INDU", "loss_ratio"])
    # industrial/insurer never report net interest income -> no NIM / bank_roa
    assert np.isnan(k.loc["INDU", "net_interest_margin"])
    assert np.isnan(k.loc["INDU", "bank_roa"])
    # non-REIT rows without real estate -> no FFO margin
    assert np.isnan(k.loc["BANK", "ffo_margin"])
    # insurer gets its loss ratio
    assert not np.isnan(k.loc["INSR", "loss_ratio"])
    # NEW KPIs are gated too: rate-base only for the utility (reports regulatory assets)
    assert not np.isnan(k.loc["UTIL", "capex_to_rate_base"])
    assert np.isnan(k.loc["INDU", "capex_to_rate_base"])
    # EBITDAX only for the oil & gas name (reports oil-gas property)
    assert not np.isnan(k.loc["OILX", "ebitdax_margin"])
    assert np.isnan(k.loc["INDU", "ebitdax_margin"])
    # AFFO / EBITDAre leverage only for the REIT
    assert not np.isnan(k.loc["REIT", "net_debt_to_ebitdare"])
    assert np.isnan(k.loc["BANK", "net_debt_to_ebitdare"])
    # investment-income ratio only for the insurer (reports premiums)
    assert not np.isnan(k.loc["INSR", "investment_income_ratio"])
    assert np.isnan(k.loc["INDU", "investment_income_ratio"])
    print("\n=== SANITY CHECK: availability gating ===")
    print("  loss_ratio only for insurer; NIM/bank_roa only for bank; FFO/AFFO only for REIT; "
          "rate-base only for utility; EBITDAX only for oil&gas. Validated.")
