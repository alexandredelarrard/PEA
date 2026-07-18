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
    build_ticker_history, EXTRA_FLOW_TAGS)


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
