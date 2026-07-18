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
        # ---- Bank ----
        {"ticker": "BANK", "sector": "Financials", "totalAssets": 1000.0, "netIncome": 20.0,
         "netInterestIncome": 40.0, "noninterestIncome": 10.0, "noninterestExpense": 30.0,
         "loans": 800.0, "deposits": 900.0, "provisionForCreditLosses": 8.0},
        # ---- Insurer ----
        {"ticker": "INSR", "sector": "Financials", "premiumsEarned": 500.0,
         "claimsIncurred": 350.0, "sellingGeneralAdmin": 100.0, "dacAmortization": 20.0},
        # ---- REIT ----
        {"ticker": "REIT", "sector": "Real Estate", "netIncome": 50.0, "depAmort": 100.0,
         "gainOnDispositions": 10.0, "totalRevenue": 300.0, "rentalIncome": 280.0,
         "dividendsPaid": 120.0, "realEstateNet": 2000.0},
        # ---- Industrial (universal KPIs) ----
        {"ticker": "INDU", "sector": "Industrials", "totalRevenue": 1000.0, "costOfRevenue": 600.0,
         "grossProfit": 400.0, "accountsReceivable": 200.0, "inventory": 150.0,
         "accountsPayable": 100.0, "totalAssets": 2000.0, "ebitda": 250.0, "interestExpense": 25.0,
         "netIncome": 120.0, "operatingCashFlow": 180.0, "incomeTaxExpense": 30.0,
         "pretaxIncome": 150.0, "capex": 80.0, "depAmort": 60.0, "dividendsPaid": 40.0,
         "buybacks": 60.0, "longTermDebt": 500.0, "shortTermDebt": 50.0, "cash": 100.0,
         "shortTermInvestments": 50.0},
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
    print("\n=== SANITY CHECK: availability gating ===")
    print("  loss_ratio only for insurer; NIM/bank_roa only for bank; FFO only for REIT. Validated.")
