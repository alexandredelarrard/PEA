"""
Clean reconstruction of balance-sheet pools that filers report under split/alternate
elements — each target is DERIVED from its own components so signals are never mixed:

  * ppeGross      = gross tag, else net + accumulated depreciation  (+ new ppeNet column)
  * oilGasPropertyNet = net tag, else gross - accumulated DD&A       (E&P filers)
  * deferredRevenue   = combined tag, else current + noncurrent      (no double-count)
  * provisionForCreditLosses coalesces the CECL-era element

All values are synthetic known-truth so the arithmetic (and the no-double-count
guarantee) is asserted exactly.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.data_extract.utils.fundamentals.fetch_fundamentals import (
    build_ticker_history, EXTRA_FLOW_TAGS, FLOW_TAGS)


def _q(end, start, val):
    return {"end": end, "start": start,
            "filed": (pd.Timestamp(end) + pd.Timedelta(days=40)).date().isoformat(),
            "form": "10-Q", "fp": "Q1", "val": val}


def _year(val, years):
    return [_q(f"{y}-{e}", f"{y}-{s}", val) for y in years
            for s, e in [("01-01","03-31"),("04-01","06-30"),("07-01","09-30"),("10-01","12-31")]]


def _inst(val, years):
    return [{"end": f"{y}-12-31", "start": None, "filed": f"{y+1}-02-10",
             "form": "10-K", "fp": "FY", "val": val} for y in years]


def _build(gaap):
    return build_ticker_history("T", {"facts": {"us-gaap": gaap, "dei": {}}})


def _ye(fe, year="2020-12-31"):
    return fe[fe["fiscal_end"] == year].iloc[0]


def test_ppe_gross_reconstructed_from_net_plus_accumulated():
    gaap = {
        "Revenues": {"units": {"USD": _year(1000.0, [2019, 2020])}},
        "PropertyPlantAndEquipmentNet": {"units": {"USD": _inst(800.0, [2019, 2020])}},
        "AccumulatedDepreciationDepletionAndAmortizationPropertyPlantAndEquipment":
            {"units": {"USD": _inst(200.0, [2019, 2020])}},
        # NO PropertyPlantAndEquipmentGross
    }
    ye = _ye(_build(gaap))
    assert ye["ppeNet"] == pytest.approx(800.0)                 # new column populated
    assert ye["ppeGross"] == pytest.approx(1000.0)             # 800 + 200 (reconstructed)
    print("\n=== SANITY CHECK: ppeGross reconstruction ===")
    print(f"  ppeNet=800 (new column), ppeGross={ye['ppeGross']:.0f} = net 800 + accumulated 200. Validated.")


def test_oilgas_net_reconstructed_from_gross_minus_accumulated():
    gaap = {
        "Revenues": {"units": {"USD": _year(1000.0, [2019, 2020])}},
        "OilAndGasPropertySuccessfulEffortMethodGross": {"units": {"USD": _inst(5000.0, [2019, 2020])}},
        "AccumulatedDepreciationDepletionAndAmortizationPropertyPlantAndEquipment":
            {"units": {"USD": _inst(2000.0, [2019, 2020])}},
        # NO ...Net
    }
    ye = _ye(_build(gaap))
    assert ye["oilGasPropertyNet"] == pytest.approx(3000.0)    # 5000 - 2000
    print("\n=== SANITY CHECK: oilGasPropertyNet reconstruction ===")
    print(f"  oilGasPropertyNet={ye['oilGasPropertyNet']:.0f} = gross 5000 - accumulated 2000 (E&P). Validated.")


def test_deferred_revenue_total_no_double_count():
    # filer reports the COMBINED total AND the split parts -> total must be the
    # combined value, NOT combined + current + noncurrent.
    gaap = {
        "Revenues": {"units": {"USD": _year(1000.0, [2019, 2020])}},
        "ContractWithCustomerLiability": {"units": {"USD": _inst(300.0, [2019, 2020])}},
        "ContractWithCustomerLiabilityCurrent": {"units": {"USD": _inst(250.0, [2019, 2020])}},
        "ContractWithCustomerLiabilityNoncurrent": {"units": {"USD": _inst(50.0, [2019, 2020])}},
    }
    ye = _ye(_build(gaap))
    assert ye["deferredRevenue"] == pytest.approx(300.0), "combined tag must win (no double-count)"

    # filer reports ONLY the split parts -> total = current + noncurrent
    gaap2 = {
        "Revenues": {"units": {"USD": _year(1000.0, [2019, 2020])}},
        "ContractWithCustomerLiabilityCurrent": {"units": {"USD": _inst(250.0, [2019, 2020])}},
        "ContractWithCustomerLiabilityNoncurrent": {"units": {"USD": _inst(50.0, [2019, 2020])}},
    }
    ye2 = _ye(_build(gaap2))
    assert ye2["deferredRevenue"] == pytest.approx(300.0), "split parts must sum to the total"

    print("\n=== SANITY CHECK: deferredRevenue total (no double-count) ===")
    print(f"  combined-reporter total={ye['deferredRevenue']:.0f} (=300, not 600); "
          f"split-only reporter total={ye2['deferredRevenue']:.0f} (=250+50). Validated.")


def test_provision_for_credit_losses_cecl_tag():
    assert "ProvisionForLoanAndLeaseLosses" in EXTRA_FLOW_TAGS["provisionForCreditLosses"]
    gaap = {
        "Revenues": {"units": {"USD": _year(1000.0, [2018, 2019, 2020])}},
        "ProvisionForLoanAndLeaseLosses": {"units": {"USD": _year(50.0, [2018, 2019, 2020])}},
    }
    ye = _ye(_build(gaap))
    assert ye["provisionForCreditLosses"] == pytest.approx(200.0)   # TTM = 4 x 50
    print("\n=== SANITY CHECK: provisionForCreditLosses CECL coalesce ===")
    print(f"  TTM provision={ye['provisionForCreditLosses']:.0f} (4x50) from CECL tag. Validated.")


def test_total_debt_pool():
    """totalDebt = single combined ST+LT tag when present, else long+short term,
    else notes payable (REITs) — a clean universal debt pool, never mixed."""
    # 1) combined tag present -> use it (not LT+ST)
    g1 = {"Revenues": {"units": {"USD": _year(1000.0, [2019, 2020])}},
          "LongTermDebt": {"units": {"USD": _inst(1000.0, [2019, 2020])}},
          "DebtCurrent": {"units": {"USD": _inst(200.0, [2019, 2020])}},
          "DebtLongtermAndShorttermCombinedAmount": {"units": {"USD": _inst(1150.0, [2019, 2020])}}}
    assert _ye(_build(g1))["totalDebt"] == pytest.approx(1150.0)

    # 2) no combined -> long-term + short-term
    g2 = {"Revenues": {"units": {"USD": _year(1000.0, [2019, 2020])}},
          "LongTermDebt": {"units": {"USD": _inst(1000.0, [2019, 2020])}},
          "DebtCurrent": {"units": {"USD": _inst(200.0, [2019, 2020])}}}
    assert _ye(_build(g2))["totalDebt"] == pytest.approx(1200.0)

    # 3) REIT-style: only notes payable
    g3 = {"Revenues": {"units": {"USD": _year(1000.0, [2019, 2020])}},
          "NotesPayable": {"units": {"USD": _inst(800.0, [2019, 2020])}}}
    assert _ye(_build(g3))["totalDebt"] == pytest.approx(800.0)

    print("\n=== SANITY CHECK: totalDebt universal pool ===")
    print("  combined tag=1150 (not 1200); LT+ST=1200; notes-payable REIT=800. Validated.")


def test_total_debt_zero_for_debt_free_filer():
    """A filer with a balance sheet but NO debt tag in any quarter is debt-free ->
    totalDebt = 0 (not missing)."""
    g = {"Revenues": {"units": {"USD": _year(1000.0, [2019, 2020])}},
         "Assets": {"units": {"USD": _inst(5000.0, [2019, 2020])}},
         "StockholdersEquity": {"units": {"USD": _inst(3000.0, [2019, 2020])}}}
    ye = _ye(_build(g))
    assert ye["totalDebt"] == pytest.approx(0.0)
    print("\n=== SANITY CHECK: debt-free totalDebt=0 ===")
    print(f"  balance sheet present, no debt tag -> totalDebt={ye['totalDebt']:.0f}. Validated.")


def test_cash_plain_tag_fallback():
    """A filer (e.g. AIG/ALL) that tags plain `Cash` but not the carrying-value
    element still populates cash."""
    g = {"Revenues": {"units": {"USD": _year(1000.0, [2019, 2020])}},
         "Cash": {"units": {"USD": _inst(500.0, [2019, 2020])}}}
    ye = _ye(_build(g))
    assert ye["cash"] == pytest.approx(500.0)
    print("\n=== SANITY CHECK: cash plain-tag fallback ===")
    print(f"  cash={ye['cash']:.0f} from the plain `Cash` element. Validated.")


def test_revenue_pre_asc606_goods_services_tags():
    """Filers that only tagged the pre-ASC-606 goods/services split (e.g. WDC=Goods,
    VRSK=Services) still get total revenue -> no pre-2017 truncation."""
    assert "SalesRevenueGoodsNet" in FLOW_TAGS["totalRevenue"]
    gaap = {"SalesRevenueGoodsNet": {"units": {"USD": _year(1000.0, [2018, 2019, 2020])}}}
    ye = _ye(_build(gaap))
    assert ye["totalRevenue"] == pytest.approx(4000.0)          # TTM = 4 x 1000
    print("\n=== SANITY CHECK: pre-ASC-606 revenue tag ===")
    print(f"  totalRevenue={ye['totalRevenue']:.0f} from SalesRevenueGoodsNet. Validated.")


def test_ebitda_bottom_up_fallback():
    """A filer with no operating-income / gross-profit line (e.g. integrated oil)
    still gets EBITDA bottom-up: net income + taxes + interest + D&A."""
    gaap = {
        "Revenues": {"units": {"USD": _year(1000.0, [2018, 2019, 2020])}},
        "NetIncomeLoss": {"units": {"USD": _year(100.0, [2018, 2019, 2020])}},
        "IncomeTaxExpenseBenefit": {"units": {"USD": _year(30.0, [2018, 2019, 2020])}},
        "InterestExpense": {"units": {"USD": _year(20.0, [2018, 2019, 2020])}},
        "DepreciationDepletionAndAmortization": {"units": {"USD": _year(50.0, [2018, 2019, 2020])}},
        # NO OperatingIncomeLoss, NO GrossProfit/CostOfRevenue
    }
    ye = _ye(_build(gaap))
    # TTM: NI 400 + tax 120 + interest 80 + D&A 200 = 800
    assert ye["ebitda"] == pytest.approx(800.0)
    print("\n=== SANITY CHECK: bottom-up EBITDA (no operating-income line) ===")
    print(f"  ebitda={ye['ebitda']:.0f} = NI 400 + tax 120 + interest 80 + D&A 200. Validated.")


def test_roe_defined_for_negative_equity():
    """Negative-equity firms (heavy buybacks, e.g. VRSN) get a (negative) ROE rather
    than being dropped by an equity>0 guard."""
    gaap = {
        "Revenues": {"units": {"USD": _year(1000.0, [2019, 2020])}},
        "NetIncomeLoss": {"units": {"USD": _year(100.0, [2019, 2020])}},
        "StockholdersEquity": {"units": {"USD": _inst(-500.0, [2019, 2020])}},
    }
    ye = _ye(_build(gaap))
    assert ye["returnOnEquity"] == pytest.approx(-0.8)          # NI 400 / equity -500
    print("\n=== SANITY CHECK: ROE for negative book equity ===")
    print(f"  returnOnEquity={ye['returnOnEquity']:.2f} (NI 400 / equity -500). Validated.")


def test_netincome_incl_nci_fallback_fills_gap():
    """A filer that stopped tagging ProfitLoss/NetIncomeLoss for an early stretch but
    kept the continuing-ops-INCL-NCI tag (same basis as ProfitLoss, e.g. WAT ~2011-2014)
    still gets netIncome there — the tag is coalesced fill-only, primaries win elsewhere."""
    incl_nci = "IncomeLossFromContinuingOperationsIncludingPortionAttributableToNoncontrollingInterest"
    assert incl_nci in FLOW_TAGS["netIncome"]
    gaap = {
        "Revenues": {"units": {"USD": _year(1000.0, [2018, 2019, 2020])}},
        "ProfitLoss": {"units": {"USD": _year(100.0, [2019, 2020])}},   # gap in 2018
        incl_nci: {"units": {"USD": _year(90.0, [2018])}},              # fills the 2018 gap
    }
    fe = _build(gaap)
    assert _ye(fe, "2018-12-31")["netIncome"] == pytest.approx(360.0)   # 4 x 90 (incl-NCI)
    assert _ye(fe, "2020-12-31")["netIncome"] == pytest.approx(400.0)   # 4 x 100 (ProfitLoss wins)
    print("\n=== SANITY CHECK: netIncome incl-NCI fill-only fallback ===")
    print("  2018 gap filled from incl-NCI tag (360=4x90); 2020 uses ProfitLoss (400=4x100). Validated.")


def test_netincome_to_common_fill_when_no_preferred():
    """A no-preferred filer (to-common == net income on overlapping periods) recovers
    an early netIncome gap from NetIncomeLossAvailableToCommonStockholdersBasic."""
    to_common = "NetIncomeLossAvailableToCommonStockholdersBasic"
    gaap = {
        "Revenues": {"units": {"USD": _year(1000.0, [2017, 2018, 2019, 2020])}},
        "NetIncomeLoss": {"units": {"USD": _year(100.0, [2018, 2019, 2020])}},          # gap in 2017
        # to-common EQUALS the primary on 2018-2020 (no preferred) -> guard trusts it,
        # so the 2017-only to-common value fills the gap.
        to_common: {"units": {"USD": _year(90.0, [2017]) + _year(100.0, [2018, 2019, 2020])}},
    }
    fe = _build(gaap)
    assert _ye(fe, "2017-12-31")["netIncome"] == pytest.approx(360.0)   # 4 x 90 (to-common fill)
    assert _ye(fe, "2020-12-31")["netIncome"] == pytest.approx(400.0)   # NetIncomeLoss wins
    print("\n=== SANITY CHECK: netIncome to-common fill (no preferred) ===")
    print("  guard confirmed to-common == primary on overlap; 2017 gap filled (360=4x90). Validated.")


def test_netincome_to_common_guarded_off_for_preferred():
    """A preferred-paying filer (to-common < net income by the preferred dividend) must
    NOT have its netIncome contaminated: the to-common tag is rejected by the guard, so
    a period with only a to-common value stays empty rather than mixing bases."""
    to_common = "NetIncomeLossAvailableToCommonStockholdersBasic"
    gaap = {
        "Revenues": {"units": {"USD": _year(1000.0, [2017, 2018, 2019, 2020])}},
        "NetIncomeLoss": {"units": {"USD": _year(100.0, [2018, 2019, 2020])}},          # gap in 2017
        # to-common is 20% below the primary on the overlap (preferred dividends) ->
        # guard rejects it; the 2017 gap is NOT filled with the contaminated figure.
        to_common: {"units": {"USD": _year(70.0, [2017]) + _year(80.0, [2018, 2019, 2020])}},
    }
    fe = _build(gaap)
    assert pd.isna(_ye(fe, "2017-12-31")["netIncome"]), "contaminated to-common must be rejected"
    assert _ye(fe, "2020-12-31")["netIncome"] == pytest.approx(400.0)   # primary untouched
    print("\n=== SANITY CHECK: netIncome to-common guarded off (preferred payer) ===")
    print("  to-common differs 20% on overlap -> rejected; 2017 stays empty; 2020 primary intact. Validated.")


def test_real_estate_cost_of_revenue_gross_margin():
    """Homebuilders / residential REITs tag COGS under the real-estate cost elements,
    not the goods/services ones — coalescing them recovers gross margin (e.g. PHM)."""
    assert "CostOfRealEstateRevenue" in FLOW_TAGS["costOfRevenue"]
    gaap = {
        "Revenues": {"units": {"USD": _year(1000.0, [2018, 2019, 2020])}},
        "CostOfRealEstateRevenue": {"units": {"USD": _year(750.0, [2018, 2019, 2020])}},
        # NO CostOfGoodsAndServicesSold / GrossProfit
    }
    ye = _ye(_build(gaap))
    # TTM gross profit = 4000 - 3000 = 1000 -> margin 0.25
    assert ye["grossMargins"] == pytest.approx(0.25)
    print("\n=== SANITY CHECK: real-estate cost-of-revenue gross margin ===")
    print(f"  grossMargins={ye['grossMargins']:.2f} from CostOfRealEstateRevenue (homebuilder COGS). Validated.")


def test_gross_margin_artifact_guard():
    """A revenue/cost period-or-scope mismatch (truncated revenue vs full cost, e.g. a
    REIT whose rental income moved to the ASC-842 lease tag) yields an implausible gross
    margin (< -200% or > 1); the guard nulls it rather than shipping a garbage feature."""
    # cost >> revenue -> gross margin ~ -3 -> must be nulled
    bad = {
        "Revenues": {"units": {"USD": _year(100.0, [2018, 2019, 2020])}},
        "CostOfGoodsAndServicesSold": {"units": {"USD": _year(400.0, [2018, 2019, 2020])}},
    }
    ye_bad = _ye(_build(bad))
    assert pd.isna(ye_bad["grossMargins"]), "gross margin of -300% must be nulled as an artifact"

    # a normal filer is untouched
    good = {
        "Revenues": {"units": {"USD": _year(1000.0, [2018, 2019, 2020])}},
        "CostOfGoodsAndServicesSold": {"units": {"USD": _year(300.0, [2018, 2019, 2020])}},
    }
    ye_good = _ye(_build(good))
    assert ye_good["grossMargins"] == pytest.approx(0.70)
    print("\n=== SANITY CHECK: gross-margin artifact guard ===")
    print(f"  cost>>revenue -> grossMargins nulled ({ye_bad['grossMargins']}); "
          f"normal filer kept ({ye_good['grossMargins']:.2f}). Validated.")


def test_bank_net_revenue_override_fee_slice():
    """Banks tag only a small fee slice under the ASC-606 contract-revenue element; their
    true top line is net interest income + noninterest income. That total must OVERRIDE
    the fee slice (else revenue is understated ~10x -> absurd margins, e.g. FITB 400%)."""
    gaap = {
        "NetIncomeLoss": {"units": {"USD": _year(300.0, [2019, 2020])}},
        # small ASC-606 fee slice the coalesce would otherwise treat as the top line
        "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": _year(100.0, [2019, 2020])}},
        "InterestIncomeExpenseNet": {"units": {"USD": _year(1000.0, [2019, 2020])}},
        "NoninterestIncome": {"units": {"USD": _year(500.0, [2019, 2020])}},
    }
    # the rebuild is gated to the Financials sector
    fe = build_ticker_history("BANK", {"facts": {"us-gaap": gaap, "dei": {}}}, "Financials")
    ye = fe[fe["fiscal_end"] == "2020-12-31"].iloc[0]
    assert ye["totalRevenue"] == pytest.approx(6000.0)   # 4x(1000+500), not 4x100
    assert ye["profitMargins"] == pytest.approx(0.20)    # 1200/6000, not 3.0
    print("\n=== SANITY CHECK: bank net-revenue override ===")
    print(f"  totalRevenue={ye['totalRevenue']:.0f} (NII+noninterest, not the 400 fee slice); "
          f"profitMargins={ye['profitMargins']:.2f}. Validated.")


def test_reit_revenue_rental_override():
    """REITs tag rent under the operating-lease elements (leases are outside ASC-606),
    so the contract-revenue element the coalesce grabs is only a small fee slice. The
    Real-Estate override recovers the rental total (max -> no double-count); a REIT that
    tags rent under contract-revenue keeps its higher total."""
    # fee-slice reporter (CPT/EXR): tiny RCC fee + big rent under the lease element
    g = {"RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": _year(10.0, [2019, 2020])}},
         "OperatingLeaseLeaseIncome": {"units": {"USD": _year(250.0, [2019, 2020])}}}
    ye = build_ticker_history("R", {"facts": {"us-gaap": g, "dei": {}}}, "Real Estate")
    assert ye[ye["fiscal_end"] == "2020-12-31"].iloc[0]["totalRevenue"] == pytest.approx(1000.0)  # 4x250, not 40

    # rent-in-contract-revenue reporter (ARE/PLD): RCC is the total -> no double-count with the component
    g2 = {"RevenueFromContractWithCustomerIncludingAssessedTax": {"units": {"USD": _year(500.0, [2019, 2020])}},
          "RealEstateRevenueNet": {"units": {"USD": _year(200.0, [2019, 2020])}}}
    ye2 = build_ticker_history("R2", {"facts": {"us-gaap": g2, "dei": {}}}, "Real Estate")
    assert ye2[ye2["fiscal_end"] == "2020-12-31"].iloc[0]["totalRevenue"] == pytest.approx(2000.0)  # 4x500, not 2800

    print("\n=== SANITY CHECK: REIT rental revenue override ===")
    print("  fee-slice REIT -> 1000 (rent, not the 40 fee); rent-in-RCC REIT -> 2000 (no double-count). Validated.")


def test_energy_and_healthcare_revenue_recovery():
    """Pure E&P tag the top line as oil&gas revenue (Energy-gated override; integrated
    majors' fuller Revenues wins). Health-care filers tag patient-service revenue, now
    coalesced into totalRevenue. A non-energy filer with a small oil&gas line is untouched."""
    # E&P: only oil&gas revenue (+ a net-income anchor for the quarter grid) -> recovered
    g = {"OilAndGasRevenue": {"units": {"USD": _year(500.0, [2019, 2020])}},
         "NetIncomeLoss": {"units": {"USD": _year(50.0, [2019, 2020])}}}
    ye = build_ticker_history("EP", {"facts": {"us-gaap": g, "dei": {}}}, "Energy")
    assert ye[ye["fiscal_end"] == "2020-12-31"].iloc[0]["totalRevenue"] == pytest.approx(2000.0)
    # integrated major: fuller Revenues wins (no double-count)
    g2 = {"Revenues": {"units": {"USD": _year(800.0, [2019, 2020])}},
          "OilAndGasRevenue": {"units": {"USD": _year(500.0, [2019, 2020])}}}
    ye2 = build_ticker_history("BIGOIL", {"facts": {"us-gaap": g2, "dei": {}}}, "Energy")
    assert ye2[ye2["fiscal_end"] == "2020-12-31"].iloc[0]["totalRevenue"] == pytest.approx(3200.0)
    # non-energy (asset mgr consolidating an oil&gas portfolio co): oil&gas ignored, not overridden
    g3 = {"Revenues": {"units": {"USD": _year(800.0, [2019, 2020])}},
          "OilAndGasRevenue": {"units": {"USD": _year(5.0, [2019, 2020])}}}
    yf = build_ticker_history("AM", {"facts": {"us-gaap": g3, "dei": {}}}, "Financials")
    assert yf[yf["fiscal_end"] == "2020-12-31"].iloc[0]["totalRevenue"] == pytest.approx(3200.0)  # Revenues, not 20

    # health care: patient-service revenue coalesced into totalRevenue
    gh = {"HealthCareOrganizationRevenue": {"units": {"USD": _year(1000.0, [2018, 2019, 2020])}}}
    yh = _ye(_build(gh))
    assert yh["totalRevenue"] == pytest.approx(4000.0)

    print("\n=== SANITY CHECK: energy + health-care revenue recovery ===")
    print("  E&P oil&gas -> 2000; integrated major keeps Revenues 3200 (no double-count); "
          "non-energy oil&gas line ignored (3200); health-care patient revenue -> 4000. Validated.")


def test_regulatory_assets_total_current_plus_noncurrent():
    """Utilities split regulatory assets into current + noncurrent; the total pool the
    utility KPIs need must SUM them (not coalesce one), and use the combined tag when the
    filer reports it (e.g. SO) without double-counting."""
    # split-only reporter (NEE/AEP/XEL) -> total = current + noncurrent
    g1 = {"Revenues": {"units": {"USD": _year(1000.0, [2019, 2020])}},
          "RegulatoryAssetsCurrent": {"units": {"USD": _inst(200.0, [2019, 2020])}},
          "RegulatoryAssetsNoncurrent": {"units": {"USD": _inst(800.0, [2019, 2020])}}}
    assert _ye(_build(g1))["regulatoryAssets"] == pytest.approx(1000.0)
    # combined-tag reporter (SO) -> use combined, NOT combined + split
    g2 = {"Revenues": {"units": {"USD": _year(1000.0, [2019, 2020])}},
          "RegulatoryAssets": {"units": {"USD": _inst(900.0, [2019, 2020])}},
          "RegulatoryAssetsCurrent": {"units": {"USD": _inst(200.0, [2019, 2020])}},
          "RegulatoryAssetsNoncurrent": {"units": {"USD": _inst(800.0, [2019, 2020])}}}
    assert _ye(_build(g2))["regulatoryAssets"] == pytest.approx(900.0)
    print("\n=== SANITY CHECK: regulatory assets total (current + noncurrent) ===")
    print("  split-only reporter total=1000 (200+800); combined-tag reporter=900 (no double-count). Validated.")


def test_operating_income_bottom_up_ebit_nonfinancial_only():
    """Non-financials with no operating-income tag and no gross profit (REIT / integrated
    oil, e.g. O, DVN, XOM) get operatingIncome = pre-tax income + interest (EBIT) so
    EBITDAre / EBITDAX compute; financials are excluded (they use their own proxy)."""
    pretax = ("IncomeLossFromContinuingOperationsBeforeIncomeTaxes"
              "MinorityInterestAndIncomeLossFromEquityMethodInvestments")
    gaap = {
        "Revenues": {"units": {"USD": _year(1000.0, [2019, 2020])}},
        "NetIncomeLoss": {"units": {"USD": _year(100.0, [2019, 2020])}},
        pretax: {"units": {"USD": _year(130.0, [2019, 2020])}},
        "InterestExpense": {"units": {"USD": _year(20.0, [2019, 2020])}},
        "DepreciationDepletionAndAmortization": {"units": {"USD": _year(50.0, [2019, 2020])}},
        # NO OperatingIncomeLoss, NO GrossProfit / CostOfRevenue
    }
    reit = build_ticker_history("R", {"facts": {"us-gaap": gaap, "dei": {}}}, "Real Estate")
    ye = reit[reit["fiscal_end"] == "2020-12-31"].iloc[0]
    assert ye["operatingIncome"] == pytest.approx(600.0)   # EBIT = pretax 520 + interest 80
    assert ye["depAmort"] == pytest.approx(200.0)          # now emitted as a column
    assert ye["ebitda"] == pytest.approx(800.0)            # EBITDAre = opInc 600 + D&A 200

    fin = build_ticker_history("B", {"facts": {"us-gaap": gaap, "dei": {}}}, "Financials")
    assert pd.isna(fin[fin["fiscal_end"] == "2020-12-31"].iloc[0]["operatingIncome"])
    print("\n=== SANITY CHECK: bottom-up EBIT operatingIncome (non-financials) ===")
    print(f"  REIT operatingIncome={ye['operatingIncome']:.0f} (EBIT), EBITDAre={ye['ebitda']:.0f}; "
          f"financial excluded. Validated.")


def test_operating_income_from_costs_and_expenses_nonfinancial():
    """A non-financial reporting 'Total costs and expenses' but no operating-income
    line (integrated oil pre-restructuring, e.g. COP 2012-2016) gets
    operatingIncome = revenue - total costs. A >60% implied margin (incompletely
    tagged costs, e.g. COP's pre-2012 integrated years) is rejected as an artifact;
    financials are excluded (gated)."""
    assert "CostsAndExpenses" in EXTRA_FLOW_TAGS["costsAndExpenses"]
    rev = {"Revenues": {"units": {"USD": _year(1000.0, [2019, 2020])}}}     # TTM 4000

    # complete costs -> OI = revenue - costs (20% margin, kept); preferred over EBIT
    g = {**rev, "CostsAndExpenses": {"units": {"USD": _year(800.0, [2019, 2020])}}}
    ye = _ye(build_ticker_history("OIL", {"facts": {"us-gaap": g, "dei": {}}}, "Energy"))
    assert ye["operatingIncome"] == pytest.approx(800.0)        # 4000 - 3200

    # incompletely-tagged costs -> 80% implied margin -> rejected -> NaN (no other fallback)
    g2 = {**rev, "CostsAndExpenses": {"units": {"USD": _year(200.0, [2019, 2020])}}}
    ye2 = _ye(build_ticker_history("OIL2", {"facts": {"us-gaap": g2, "dei": {}}}, "Energy"))
    assert pd.isna(ye2["operatingIncome"])                      # 80% margin = artifact

    # financials excluded even with complete costs
    fin = _ye(build_ticker_history("FIN", {"facts": {"us-gaap": g, "dei": {}}}, "Financials"))
    assert pd.isna(fin["operatingIncome"])

    print("\n=== SANITY CHECK: operatingIncome = revenue - CostsAndExpenses ===")
    print(f"  non-fin complete costs -> OI={ye['operatingIncome']:.0f} (4000-3200); "
          f"80%-margin (incomplete costs) rejected -> NaN; financial gated -> NaN. Validated.")


def test_financials_profit_margin_guard():
    """A Financials-sector net margin above ~1.5x (consolidated-fund NCI / one-time
    attribution, e.g. ARES pre-IPO) is nulled; the same margin in another sector (a
    one-time gain / biotech) is kept."""
    gaap = {"InterestIncomeExpenseNet": {"units": {"USD": _year(100.0, [2019, 2020])}},  # bank rev
            "NetIncomeLoss": {"units": {"USD": _year(200.0, [2019, 2020])}}}             # ni > rev
    fin = build_ticker_history("F", {"facts": {"us-gaap": gaap, "dei": {}}}, "Financials")
    assert pd.isna(fin[fin["fiscal_end"] == "2020-12-31"].iloc[0]["profitMargins"])

    gaap2 = {"Revenues": {"units": {"USD": _year(100.0, [2019, 2020])}},
             "NetIncomeLoss": {"units": {"USD": _year(200.0, [2019, 2020])}}}
    other = build_ticker_history("X", {"facts": {"us-gaap": gaap2, "dei": {}}}, "Health Care")
    ye_o = other[other["fiscal_end"] == "2020-12-31"].iloc[0]
    assert ye_o["profitMargins"] == pytest.approx(2.0)   # non-financial extreme kept
    print("\n=== SANITY CHECK: financials profit-margin guard ===")
    print(f"  financial ni>1.5x rev -> nulled; non-financial kept ({ye_o['profitMargins']:.1f}). Validated.")


def test_restaurant_food_beverage_revenue():
    """Company-operated restaurants tag pre-2016 revenue under FoodAndBeverageRevenue,
    not Revenues -> coalescing recovers the otherwise-missing early top line (e.g. CMG)."""
    assert "FoodAndBeverageRevenue" in FLOW_TAGS["totalRevenue"]
    gaap = {"FoodAndBeverageRevenue": {"units": {"USD": _year(1000.0, [2018, 2019, 2020])}},
            "NetIncomeLoss": {"units": {"USD": _year(100.0, [2018, 2019, 2020])}}}
    ye = _ye(_build(gaap))
    assert ye["totalRevenue"] == pytest.approx(4000.0)
    print("\n=== SANITY CHECK: restaurant FoodAndBeverageRevenue ===")
    print(f"  totalRevenue={ye['totalRevenue']:.0f} from FoodAndBeverageRevenue. Validated.")


def test_annual_only_flow_ttm_fallback():
    """A flow reported ONLY annually (no de-cumulable interim quarters, e.g. ADI/BALL
    working-capital changes) still populates: the full-year value IS the TTM."""
    def _ann(val, years):
        return [{"end": f"{y}-12-31", "start": f"{y}-01-01", "filed": f"{y+1}-02-10",
                 "form": "10-K", "fp": "FY", "val": val} for y in years]
    gaap = {
        "Revenues": {"units": {"USD": _year(1000.0, [2018, 2019, 2020])}},   # quarterly grid
        "IncreaseDecreaseInAccountsReceivable": {"units": {"USD": _ann(80.0, [2018, 2019, 2020])}},
    }
    fe = _build(gaap)
    ye = _ye(fe)
    assert ye["changeInReceivables"] == pytest.approx(80.0)      # annual value used as TTM
    assert ye["totalRevenue"] == pytest.approx(4000.0)          # quarterly reporter unaffected
    # interim quarter (no annual report yet that quarter) carries the last full-year value
    q1_2020 = fe[fe["fiscal_end"] == "2020-03-31"].iloc[0]
    assert q1_2020["changeInReceivables"] == pytest.approx(80.0)
    print("\n=== SANITY CHECK: annual-only flow -> TTM fallback ===")
    print(f"  changeInReceivables (annual-only) = {ye['changeInReceivables']:.0f} at FY-end and "
          f"{q1_2020['changeInReceivables']:.0f} carried into interim; revenue TTM {ye['totalRevenue']:.0f} intact. Validated.")
