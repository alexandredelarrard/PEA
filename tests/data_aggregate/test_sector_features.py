"""
Sector-specific KPI math (src/data_aggregate/utils/fundamentals/sector_features.py).

Synthetic, known-truth fundamentals rows (one per sector) with hand-computed
expected ratios — the right tool to prove each KPI formula and, crucially, that
KPIs are SCOPED (a sector's KPI is NaN outside its GICS scope or when its inputs
weren't reported, so an industrial never gets a bank ROA).

Every row carries `sector` + `industry_group`: the GICS labels are what
`sector_gates.py` scopes on. In production they are attached by
`step_cube_fundamentals` from `sp500_tickers` (`gics.attach_gics_columns`) —
`fundamentals_history` itself has never carried them. A row without them is
deliberately unclassifiable and gets NO sector KPI at all — see
`test_unclassified_rows_get_no_sector_kpi`.

⚠ THE FIXTURE ONLY USES COLUMNS THE LIVE TABLE ACTUALLY HAS. That is the point of the
rewrite: the previous version hand-built `netInterestIncome`, `loans`, `deposits`,
`claimsIncurred`, `premiumsEarned`, `tier1CapitalRatio`, `regulatoryAssets`,
`oilGasPropertyNet`, `explorationExpense` and `amortizationIntangibles` — none of which
Sharadar delivers — so 14 tests passed green over KPIs that emitted nothing in production.
A synthetic fixture may invent VALUES; it must not invent COLUMNS.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.fundamentals.sector_features import (
    SECTOR_KPI_COLS, compute_sector_kpis,
)

#: KPI families deleted with this audit because SF1 carries none of their inputs. Asserted
#: absent so a future re-add has to come with the data, not just the formula.
_DELETED_ON_MISSING_INPUTS = (
    "net_interest_margin", "efficiency_ratio", "provision_rate", "loan_to_deposit",
    "bank_operating_margin", "reserve_coverage_velocity", "tier1_capital_ratio",
    "deposit_stickiness", "nii_growth", "loan_growth", "npl_ratio", "net_charge_off_rate",
    "htm_unrealized_loss_ratio", "loss_ratio", "expense_ratio", "combined_ratio",
    "investment_income_ratio", "premium_growth", "float_growth", "rental_margin",
    "exploration_intensity", "property_overvaluation_cushion", "regulatory_asset_ratio",
    "patent_cliff", "rpo_coverage", "bad_debt_intensity",
)


def _fundamentals() -> pd.DataFrame:
    rows = [
        # ---- Bank ---- ROA is the only bank KPI SF1 can feed
        {"ticker": "BANK", "sector": "Financials", "industry_group": "Banks",
         "totalAssets": 1000.0, "netIncome": 20.0, "stockholdersEquity": 90.0,
         "accumulatedOtherComprehensiveIncome": -9.0, "totalLiabilities": 920.0},
        # ---- Insurer ---- only the universal + financials KPIs remain
        {"ticker": "INSR", "sector": "Financials", "industry_group": "Insurance",
         "totalRevenue": 500.0, "netIncome": 40.0, "stockholdersEquity": 200.0,
         "accumulatedOtherComprehensiveIncome": 10.0, "sellingGeneralAdmin": 100.0},
        # ---- REIT ---- FFO / AFFO from net income + D&A (no disposal-gain leg in SF1)
        {"ticker": "REIT", "sector": "Real Estate",
         "industry_group": "Equity Real Estate Investment Trusts (REITs)",
         "netIncome": 50.0, "depAmort": 100.0, "totalRevenue": 300.0,
         "dividendsPaid": 120.0, "operatingIncome": 90.0,
         "longTermDebt": 800.0, "shortTermDebt": 100.0, "cash": 50.0, "capex": 40.0},
        # ---- Industrial (universal KPIs) ----
        {"ticker": "INDU", "sector": "Industrials", "industry_group": "Capital Goods",
         "totalRevenue": 1000.0, "costOfRevenue": 600.0,
         "grossProfit": 400.0, "accountsReceivable": 200.0, "inventory": 150.0,
         "accountsPayable": 100.0, "totalAssets": 2000.0, "ebitda": 250.0,
         "interestExpense": 25.0,
         "netIncome": 120.0, "operatingCashFlow": 180.0, "incomeTaxExpense": 30.0,
         "pretaxIncome": 150.0, "capex": 80.0, "depAmort": 60.0, "dividendsPaid": 40.0,
         # NET ISSUANCE: negative = a 60 buyback (see capital.share_repurchases)
         "equityIssuanceNet": -60.0,
         "longTermDebt": 500.0, "shortTermDebt": 50.0, "cash": 100.0,
         "shortTermInvestments": 50.0, "operatingIncome": 180.0,
         "stockholdersEquity": 900.0, "returnOnEquity": 0.13,
         "deferredRevenue": 250.0},
        # ---- Utility ---- rate-base proxy on the plain asset base
        {"ticker": "UTIL", "sector": "Utilities", "industry_group": "Utilities",
         "totalAssets": 5000.0, "capex": 400.0},
        # ---- Pharma ---- capitalized-R&D ROIC (single period here)
        {"ticker": "PHRM", "sector": "Health Care",
         "industry_group": "Pharmaceuticals, Biotechnology & Life Sciences",
         "researchAndDevelopment": 250.0, "operatingCashFlow": 200.0,
         "operatingIncome": 300.0, "stockholdersEquity": 1000.0,
         "longTermDebt": 400.0, "shortTermDebt": 100.0, "cash": 150.0},
        # ---- Oil & gas ---- EBITDA margin + D&A intensity on the energy gate
        {"ticker": "OILX", "sector": "Energy", "industry_group": "Energy",
         "operatingIncome": 180.0, "depAmort": 110.0, "operatingCashFlow": 250.0,
         "totalRevenue": 1000.0},
    ]
    return pd.DataFrame(rows)


def test_deleted_kpis_are_gone_from_the_contract():
    """The 26 KPIs whose inputs SF1 does not carry must be absent from BOTH the declared
    column list and the computed frame — not silently all-NaN, which is how they survived
    a whole vintage of the table looking implemented."""
    k = compute_sector_kpis(_fundamentals())
    for name in _DELETED_ON_MISSING_INPUTS:
        assert name not in SECTOR_KPI_COLS, f"{name} is still declared"
        assert name not in k.columns, f"{name} is still computed"
    # the ones that moved to fundamental_features keep their single owner
    for moved in ("interest_coverage", "net_debt_to_ebitda", "gross_profitability",
                  "cash_conversion_cycle", "sbc_intensity", "roic",
                  "days_sales_outstanding", "days_inventory_outstanding",
                  "days_payable_outstanding", "net_debt_to_ebitdare", "implied_cap_rate"):
        assert moved not in SECTOR_KPI_COLS, f"{moved} must not be owned by this panel"

    print("\n=== SANITY CHECK: deleted / re-homed KPIs ===")
    print(f"  {len(_DELETED_ON_MISSING_INPUTS)} KPIs deleted for missing SF1 inputs and 11 "
          f"re-homed to the fundamental panel are all absent; {len(SECTOR_KPI_COLS)} remain. "
          f"Validated.")


def test_universal_kpis():
    k = compute_sector_kpis(_fundamentals()).set_index("ticker")
    r = k.loc["INDU"]
    assert r["effective_tax_rate"] == pytest.approx(0.20)
    assert r["accruals_ratio"] == pytest.approx((120 - 180) / 2000)                 # -0.03
    assert r["asset_turnover"] == pytest.approx(0.50)
    assert r["capex_intensity"] == pytest.approx(0.08)
    assert r["capex_to_dep"] == pytest.approx(80 / 60)
    # dividends 40 (stored outflow-POSITIVE) + buybacks 60 (from equityIssuanceNet -60)
    assert r["payout_ratio"] == pytest.approx((40 + 60) / 120)                      # 0.8333
    assert r["buyback_intensity"] == pytest.approx(0.06)
    assert r["deferred_rev_intensity"] == pytest.approx(0.25)

    print("\n=== SANITY CHECK: universal KPIs (industrial) ===")
    print(f"  tax={r['effective_tax_rate']:.2f} accruals={r['accruals_ratio']:.3f} "
          f"asset_turnover={r['asset_turnover']:.2f} capex_int={r['capex_intensity']:.2f}")
    print(f"  payout={r['payout_ratio']:.4f} from +40 dividends and a 60 buyback read off "
          f"equityIssuanceNet=-60 (NET ISSUANCE, negative = repurchase). Validated.")


def test_bank_roa_is_the_surviving_bank_kpi():
    k = compute_sector_kpis(_fundamentals()).set_index("ticker")
    assert k.loc["BANK", "bank_roa"] == pytest.approx(20 / 1000)                     # 0.02
    assert k.loc["BANK", "aoci_to_equity"] == pytest.approx(-9 / 90)                 # -0.10
    assert np.isnan(k.loc["INDU", "bank_roa"]), "an industrial must not get bank_roa"
    print("\n=== SANITY CHECK: bank KPIs (post-audit) ===")
    print(f"  bank_roa={k.loc['BANK','bank_roa']:.3f}, aoci_to_equity="
          f"{k.loc['BANK','aoci_to_equity']:+.3f} (negative AOCI = unrealized securities "
          f"losses eroding capital). NIM/efficiency/provision need tags SF1 lacks. Validated.")


def test_reit_kpis():
    k = compute_sector_kpis(_fundamentals()).set_index("ticker")
    r = k.loc["REIT"]
    # FFO = net income + D&A only: SF1 has no gainOnDispositions / realEstateImpairment leg
    ffo = 50 + 100                                                       # 150
    assert r["ffo_margin"] == pytest.approx(ffo / 300)
    assert r["ffo_payout"] == pytest.approx(120 / ffo)                   # dividends are POSITIVE
    assert r["affo_margin"] == pytest.approx((ffo - 40) / 300)           # less recurring capex
    assert r["affo_dividend_coverage"] == pytest.approx((ffo - 40) / 120)
    print("\n=== SANITY CHECK: REIT KPIs ===")
    print(f"  FFO={ffo} (net income + D&A; no disposal-gain leg in SF1) margin="
          f"{r['ffo_margin']:.3f} payout={r['ffo_payout']:.3f} affo_margin="
          f"{r['affo_margin']:.3f} affo_coverage={r['affo_dividend_coverage']:.3f} "
          f"(>1 = the distribution is covered). Validated.")


def test_capital_efficiency_kpis():
    r = compute_sector_kpis(_fundamentals()).set_index("ticker").loc["INDU"]
    assert r["earnings_quality"] == pytest.approx(180.0 / 120.0)         # 1.5 (OCF/NI)
    assert r["fixed_cost_coverage_margin"] == pytest.approx((400 - 250) / 1000)  # 0.15
    # SGR = ROE x (1 - payout). The payout leg is only a payout because `dividendsPaid` is
    # stored outflow-POSITIVE: while it was negative the clip(0,1) floored every payer at 0
    # and SGR collapsed onto returnOnEquity exactly (r = 1.0000).
    assert r["sustainable_growth_rate"] == pytest.approx(0.13 * (1 - 40 / 120))
    assert r["sustainable_growth_rate"] != pytest.approx(r["returnOnEquity"])
    assert r["gmroi"] == pytest.approx(400.0 / 150.0)                    # 2.667 (single period)
    assert r["asset_turnover"] == pytest.approx(0.50)                    # single row -> period-end
    print("\n=== SANITY CHECK: capital-efficiency KPIs ===")
    print(f"  earnings_quality={r['earnings_quality']:.2f} "
          f"fixed_cost_margin={r['fixed_cost_coverage_margin']:.2f} "
          f"SGR={r['sustainable_growth_rate']:.4f} vs ROE={r['returnOnEquity']:.4f} "
          f"(no longer identical) GMROI={r['gmroi']:.2f}. Validated.")


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


def test_utility_kpi():
    r = compute_sector_kpis(_fundamentals()).set_index("ticker").loc["UTIL"]
    # capex over the PLAIN asset base: SF1 has neither `regulatoryAssets` nor a standalone
    # `goodwill`, so the base is not cleaned of either.
    assert r["capex_to_rate_base"] == pytest.approx(400 / 5000)                        # 0.08
    print("\n=== SANITY CHECK: utility rate-base KPI ===")
    print(f"  capex_to_rate_base={r['capex_to_rate_base']:.4f} (capex / total assets; the "
          f"reg-asset and goodwill deductions need tags SF1 lacks). Validated.")


def test_pharma_kpi_single_period():
    r = compute_sector_kpis(_fundamentals()).set_index("ticker").loc["PHRM"]
    # single period: R&D asset = current R&D only (no prior layers), amortization = 0
    adj_oi = 300 + 250 - 0
    adj_cap = 1000 + (400 + 100) + 250 - 150
    assert r["rd_capitalized_roic"] == pytest.approx(adj_oi / adj_cap)                  # 0.34375
    print("\n=== SANITY CHECK: pharma KPI ===")
    print(f"  rd_capitalized_roic={r['rd_capitalized_roic']:.4f} (R&D treated as a 5y "
          f"intangible). `patent_cliff` is gone: it needed amortizationIntangibles. Validated.")


def test_energy_kpis():
    r = compute_sector_kpis(_fundamentals()).set_index("ticker").loc["OILX"]
    # EBITDA, not EBITDAX: `explorationExpense` is not in SF1 so nothing is added back
    assert r["ebitda_margin"] == pytest.approx((180 + 110) / 1000)                     # 0.29
    assert r["ddna_intensity"] == pytest.approx(110 / 1000)                            # 0.11
    print("\n=== SANITY CHECK: oil & gas KPIs ===")
    print(f"  ebitda_margin={r['ebitda_margin']:.2f} (NOT ebitdax -- no exploration add-back "
          f"available) ddna_intensity={r['ddna_intensity']:.2f}. Validated.")


def test_capitalized_rd_multiperiod():
    """The KPI that needs a ticker's OWN history: the 5-year capitalized-R&D pool."""
    yrs = pd.date_range("2021-12-31", periods=6, freq="YE")
    ph = pd.DataFrame([
        {"ticker": "PH2", "sector": "Health Care",
         "industry_group": "Pharmaceuticals, Biotechnology & Life Sciences",
         "as_of": y, "researchAndDevelopment": 100.0, "operatingIncome": 300.0,
         "stockholdersEquity": 1000.0, "longTermDebt": 0.0, "shortTermDebt": 0.0, "cash": 0.0}
        for y in yrs
    ])
    kp = compute_sector_kpis(ph).set_index("as_of")
    last = kp.iloc[-1]
    # asset pool at last row = 100*(1.0+0.8+0.6+0.4+0.2) = 300 ; amort = 100*0.2*5 = 100
    adj_oi = 300 + 100 - 100
    adj_cap = 1000 + 0 + 300 - 0
    assert last["rd_capitalized_roic"] == pytest.approx(adj_oi / adj_cap)               # 300/1300
    print("\n=== SANITY CHECK: capitalized R&D (multi-period) ===")
    print(f"  rd_capitalized_roic (5y flat R&D) = {last['rd_capitalized_roic']:.4f} "
          f"(asset 300, amort 100). Validated.")


def test_kpis_are_gics_scoped():
    """A sector KPI must be NaN outside its GICS scope — including for a name that DOES
    report the inputs. `INSR` is Financials/Insurance, so it never gets a BANK KPI even
    though it is in the same sector as `BANK`; the split needs the industry group."""
    k = compute_sector_kpis(_fundamentals()).set_index("ticker")
    # bank KPIs only for the bank -- INSR is Financials too, so `sector` alone is not enough
    assert not np.isnan(k.loc["BANK", "bank_roa"])
    for t in ("INSR", "INDU", "REIT", "UTIL", "OILX", "PHRM"):
        assert np.isnan(k.loc[t, "bank_roa"]), f"{t} got bank_roa"
    # book_value_growth / aoci_to_equity are FINANCIALS-scoped, so BOTH financials get them
    for t in ("BANK", "INSR"):
        assert not np.isnan(k.loc[t, "aoci_to_equity"]), f"{t} should get aoci_to_equity"
    for t in ("INDU", "REIT", "UTIL", "OILX", "PHRM"):
        assert np.isnan(k.loc[t, "aoci_to_equity"]), f"{t} got aoci_to_equity"
    # REIT KPIs only for the REIT
    for t in ("BANK", "INSR", "INDU", "UTIL", "OILX", "PHRM"):
        assert np.isnan(k.loc[t, "ffo_margin"]), f"{t} got an FFO margin"
    # utility rate base only for the utility, energy EBITDA margin only for energy
    assert not np.isnan(k.loc["UTIL", "capex_to_rate_base"])
    assert np.isnan(k.loc["INDU", "capex_to_rate_base"])
    assert not np.isnan(k.loc["OILX", "ebitda_margin"])
    assert np.isnan(k.loc["INDU", "ebitda_margin"])
    print("\n=== SANITY CHECK: GICS scoping ===")
    print("  bank_roa only for the BANK (not the same-sector insurer); aoci_to_equity for "
          "BOTH financials (sector-scoped); FFO/AFFO only for the REIT; rate-base only for "
          "the utility; energy EBITDA margin only for energy. Validated.")


def test_unclassified_rows_get_no_sector_kpi():
    """No GICS labels -> fail CLOSED: a name we cannot classify gets no sector KPI, even
    with every input present. Prevents a mis-mapped ticker being scored on the wrong
    business model.

    This is also the failure mode the whole audit turned on: `fundamentals_history` has
    never carried `sector`/`industry_group`, so before `gics.attach_gics_columns` was wired
    in, EVERY production row looked like this one and every sector KPI was gated off."""
    df = _fundamentals().drop(columns=["sector", "industry_group"])
    k = compute_sector_kpis(df).set_index("ticker")
    for col in ("bank_roa", "ffo_margin", "ebitda_margin", "capex_to_rate_base",
                "aoci_to_equity", "book_value_growth"):
        assert k[col].isna().all(), f"{col} emitted for unclassified rows"
    # the UNIVERSAL and AVAILABILITY-gated KPIs are unaffected -- they need no sector
    assert not np.isnan(k.loc["INDU", "asset_turnover"])
    assert not np.isnan(k.loc["INDU", "effective_tax_rate"])
    assert not np.isnan(k.loc["PHRM", "rd_capitalized_roic"])   # gated on R&D, not on GICS
    print("\n=== SANITY CHECK: unclassified rows ===")
    print("  sector/industry_group dropped -> every GICS-scoped KPI NaN; the universal ones "
          "(asset_turnover, effective_tax_rate) and the availability-gated "
          "rd_capitalized_roic still computed. This was PRODUCTION's state until the GICS "
          "join was wired in. Validated.")
