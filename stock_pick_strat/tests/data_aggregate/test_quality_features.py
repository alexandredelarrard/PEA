"""Correctness tests for the business-quality factors added to
fundamental_features.py (#2 D&A realism, #5 forensic, #3 M&A digestion,
#1 core/adjusted earnings).

Each helper takes the memoized `daily` accessor (field -> date x ticker frame);
here we feed it hand-built frames with KNOWN values and assert the exact ratio
math, plus the two things that matter most for the adjusted factors:
  * SIGN of the normalization (a charge quarter -> core EARNINGS > reported;
    a one-off gain quarter -> core < reported);
  * the Beneish M-score RANKS a manipulation profile above a clean one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_aggregate.utils.fundamental_features import (
    _da_realism_fields, _forensic_fields, _digestion_fields,
    _core_earnings_fields, _beneish_m_score, _ai_leverage_fields, _derived_fields,
)

IDX = pd.bdate_range("2022-01-03", periods=300)     # >252 so shift(_YEAR) has a year-ago
_SPLIT = 252                                          # rows [0:252) = "a year ago", [252:] = "now"


def _const(vals: dict) -> pd.DataFrame:
    return pd.DataFrame({t: float(v) for t, v in vals.items()}, index=IDX)


def _step(prev: dict, now: dict) -> pd.DataFrame:
    df = pd.DataFrame(index=IDX, columns=list(prev), dtype=float)
    for t in prev:
        df[t] = [prev[t]] * _SPLIT + [now[t]] * (len(IDX) - _SPLIT)
    return df


def _mock(frames: dict):
    empty = pd.DataFrame()
    return lambda field: frames.get(field, empty)


def test_da_realism_math_and_useful_life_extension():
    frames = {
        "ppeGross": _const({"AAA": 1000, "BBB": 1000}),
        # BBB's depreciation FALLS 105->80 (lives extended); AAA steady at 80
        "depAmort": _step({"AAA": 100, "BBB": 125}, {"AAA": 100, "BBB": 100}),
        "amortizationIntangibles": _const({"AAA": 20, "BBB": 20}),
        "accumulatedDepreciation": _const({"AAA": 400, "BBB": 600}),
        "stockBasedComp": _const({"AAA": 50, "BBB": 50}),
        "buybacks": _const({"AAA": 25, "BBB": 100}),
    }
    F = _da_realism_fields(_mock(frames))
    last = IDX[-1]
    # depreciation = D&A - intangible amort = 100-20 = 80 ; life = 1000/80 = 12.5
    assert abs(F["implied_useful_life"].loc[last, "AAA"] - 12.5) < 1e-6
    assert abs(F["asset_age"].loc[last, "AAA"] - 0.4) < 1e-6
    assert abs(F["intangible_amortization_share"].loc[last, "AAA"] - 0.2) < 1e-6
    assert abs(F["sbc_to_buyback"].loc[last, "AAA"] - 2.0) < 1e-6
    # BBB: depreciation fell 105->80 => implied life jumped 9.52 -> 12.5 => change > 0
    assert F["useful_life_change"].loc[last, "BBB"] > 0
    assert abs(F["useful_life_change"].loc[last, "AAA"]) < 1e-6   # AAA steady

    print("\n=== SANITY CHECK: #2 D&A realism ===")
    print(f"  AAA useful_life=1000/80={F['implied_useful_life'].loc[last,'AAA']:.1f}y, "
          f"asset_age=0.40, intang_amort_share=0.20, sbc_to_buyback=2.0 (buybacks < SBC).")
    print(f"  BBB useful_life jumped +{F['useful_life_change'].loc[last,'BBB']:.2f}y "
          f"(lives extended = lower depreciation = red flag). Validated.")


def test_core_earnings_normalization_signs():
    frames = {
        "totalRevenue": _const({"CHG": 1000, "GAIN": 1000}),
        "pretaxIncome": _const({"CHG": 100, "GAIN": 100}),
        "netIncome": _const({"CHG": 80, "GAIN": 80}),
        "operatingIncome": _const({"CHG": 120, "GAIN": 120}),
        "ebitda": _const({"CHG": 150, "GAIN": 150}),
        "incomeTaxExpense": _const({"CHG": 20, "GAIN": 20}),      # effective tax 20/100 = 20%
        "impairment": _const({"CHG": 30, "GAIN": 0}),
        "restructuring": _const({"CHG": 10, "GAIN": 0}),
        "gainOnDispositions": _const({"CHG": 0, "GAIN": 50}),
    }
    mcap = _const({"CHG": 1000, "GAIN": 1000})
    F = _core_earnings_fields(_mock(frames), mcap)
    last = IDX[-1]
    reported_margin = 80 / 1000     # netIncome / revenue = 0.08

    # CHG: net charges of 40 depressed reported earnings -> CORE is HIGHER
    assert abs(F["nonrecurring_pretax_share"].loc[last, "CHG"] - 0.40) < 1e-6
    assert abs(F["special_items_intensity"].loc[last, "CHG"] - 0.04) < 1e-6
    assert abs(F["core_profit_margin"].loc[last, "CHG"] - (80 + 40 * 0.8) / 1000) < 1e-6
    assert F["core_profit_margin"].loc[last, "CHG"] > reported_margin
    assert abs(F["core_operating_margin"].loc[last, "CHG"] - (120 + 40) / 1000) < 1e-6
    assert abs(F["adjusted_ebitda_margin"].loc[last, "CHG"] - (150 + 40) / 1000) < 1e-6
    assert abs(F["core_earnings_yield"].loc[last, "CHG"] - 112 / 1000) < 1e-6

    # GAIN: a one-off gain of 50 inflated reported earnings -> CORE is LOWER
    assert abs(F["core_profit_margin"].loc[last, "GAIN"] - (80 - 50 * 0.8) / 1000) < 1e-6
    assert F["core_profit_margin"].loc[last, "GAIN"] < reported_margin
    assert abs(F["adjusted_ebitda_margin"].loc[last, "GAIN"] - (150 - 50) / 1000) < 1e-6

    print("\n=== SANITY CHECK: #1 core/adjusted earnings ===")
    print(f"  reported net margin = {reported_margin:.3f} for both.")
    print(f"  CHG (impair+restr 40): core margin = {F['core_profit_margin'].loc[last,'CHG']:.3f} "
          f"> reported (charges added back).")
    print(f"  GAIN (one-off gain 50): core margin = {F['core_profit_margin'].loc[last,'GAIN']:.3f} "
          f"< reported (gain stripped). Both adjusted & reported kept. Validated.")


def test_beneish_ranks_manipulator_above_clean():
    frames = {
        "totalRevenue": _step({"CLN": 1000, "MAN": 1000}, {"CLN": 1050, "MAN": 1400}),
        "accountsReceivable": _step({"CLN": 200, "MAN": 200}, {"CLN": 210, "MAN": 500}),  # AR balloons
        "grossProfit": _step({"CLN": 400, "MAN": 400}, {"CLN": 420, "MAN": 420}),         # GM deteriorates
        "currentAssets": _const({"CLN": 500, "MAN": 500}),
        "ppeNet": _const({"CLN": 300, "MAN": 300}),
        "depAmort": _step({"CLN": 50, "MAN": 50}, {"CLN": 52, "MAN": 30}),                # slows depreciation
        "sellingGeneralAdmin": _step({"CLN": 150, "MAN": 150}, {"CLN": 158, "MAN": 150}),
        "longTermDebt": _const({"CLN": 200, "MAN": 200}),
        "currentLiabilities": _const({"CLN": 150, "MAN": 150}),
        "totalAssets": _const({"CLN": 1200, "MAN": 1200}),
        "netIncome": _step({"CLN": 80, "MAN": 80}, {"CLN": 84, "MAN": 140}),
        "operatingCashFlow": _step({"CLN": 78, "MAN": 78}, {"CLN": 82, "MAN": 60}),       # high accruals
    }
    m = _beneish_m_score(_mock(frames), IDX)
    last = IDX[-1]
    assert np.isfinite(m.loc[last, "CLN"]) and np.isfinite(m.loc[last, "MAN"])
    assert m.loc[last, "MAN"] > m.loc[last, "CLN"]

    print("\n=== SANITY CHECK: #5 Beneish M-score ===")
    print(f"  clean M = {m.loc[last,'CLN']:+.2f}  vs  manipulation-profile M = {m.loc[last,'MAN']:+.2f} "
          f"(AR/accruals/margin/depreciation flags -> higher M). Validated.")


def test_forensic_days_and_offbs_leverage():
    frames = {
        "totalRevenue": _const({"X": 1000}),
        "costOfRevenue": _const({"X": 600}),
        "accountsReceivable": _const({"X": 200}),
        "accountsPayable": _const({"X": 150}),
        "inventory": _const({"X": 100}),
        "longTermDebt": _const({"X": 300}),
        "shortTermDebt": _const({"X": 100}),
        "operatingLeaseLiability": _const({"X": 50}),
        "financeLeaseLiability": _const({"X": 20}),
        "pensionDeficit": _const({"X": 30}),         # recognized net deficit -> debt-like
        "cash": _const({"X": 80}),
        "ebitda": _const({"X": 150}),
    }
    F = _forensic_fields(_mock(frames), IDX)
    last = IDX[-1]
    dso, dpo, dio = 200 / 1000 * 365, 150 / 600 * 365, 100 / 600 * 365
    assert abs(F["dso"].loc[last, "X"] - dso) < 1e-6
    assert abs(F["dpo"].loc[last, "X"] - dpo) < 1e-6
    assert abs(F["dio"].loc[last, "X"] - dio) < 1e-6
    assert abs(F["cash_conversion_cycle"].loc[last, "X"] - (dso + dio - dpo)) < 1e-6
    # (debt 400 + leases 70 + pension deficit 30 - cash 80) / EBITDA 150
    assert abs(F["net_debt_incl_offbs_to_ebitda"].loc[last, "X"] - (400 + 70 + 30 - 80) / 150) < 1e-6

    print("\n=== SANITY CHECK: #5 forensic working-capital + off-BS leverage ===")
    print(f"  DSO={dso:.1f}d, DPO={dpo:.1f}d, DIO={dio:.1f}d, CCC={dso+dio-dpo:.1f}d; "
          f"net-debt-incl-offbs/EBITDA={(400+70+30-80)/150:.2f}x (leases + pension deficit "
          f"lifted leverage). Validated.")


def test_digestion_roic_wedge_and_goodwill_weight():
    frames = {
        "operatingIncome": _const({"X": 200}),
        "incomeTaxExpense": _const({"X": 40}),
        "pretaxIncome": _const({"X": 160}),          # effective tax 40/160 = 25%
        "stockholdersEquity": _const({"X": 500}),
        "cash": _const({"X": 100}),
        "longTermDebt": _const({"X": 200}),
        "shortTermDebt": _const({"X": 0}),
        "goodwill": _const({"X": 200}),
        "intangiblesExGoodwill": _const({"X": 50}),
        "totalAssets": _const({"X": 1000}),
    }
    F = _digestion_fields(_mock(frames), pd.DataFrame(), IDX, 4)
    last = IDX[-1]
    # NOPAT = 200*(1-0.25)=150 ; IC = 500+200-100 = 600 ; roic_incl = 0.25
    assert abs(F["roic_incl_goodwill"].loc[last, "X"] - 0.25) < 1e-6
    # IC ex goodwill+intangibles = 600-250 = 350 ; roic_ex = 150/350
    assert abs(F["roic_ex_goodwill"].loc[last, "X"] - 150 / 350) < 1e-6
    assert F["goodwill_roic_drag"].loc[last, "X"] < 0     # acquisitions dilute returns
    assert abs(F["goodwill_intangibles_to_assets"].loc[last, "X"] - 0.25) < 1e-6
    assert abs(F["goodwill_to_equity"].loc[last, "X"] - 0.40) < 1e-6

    print("\n=== SANITY CHECK: #3 M&A digestion ===")
    print(f"  ROIC incl goodwill = {F['roic_incl_goodwill'].loc[last,'X']:.3f} "
          f"vs ex goodwill = {F['roic_ex_goodwill'].loc[last,'X']:.3f} "
          f"-> drag {F['goodwill_roic_drag'].loc[last,'X']:+.3f} (goodwill dilutes returns); "
          f"goodwill+intangibles = 25% of assets. Validated.")


def test_core_earnings_widened_pool_and_discontinued_ops():
    """Once the extra special-items tags are extracted, the pool must widen:
    charges (impairment+restructuring+LITIGATION) added back, gains (disposals +
    GENERIC sale + bargain purchase + net UNUSUAL) removed, and DISCONTINUED ops
    (net of tax) removed from core net income directly."""
    frames = {
        "totalRevenue": _const({"W": 1000}),
        "pretaxIncome": _const({"W": 100}),
        "netIncome": _const({"W": 80}),
        "operatingIncome": _const({"W": 120}),
        "ebitda": _const({"W": 150}),
        "incomeTaxExpense": _const({"W": 20}),        # effective tax 20%
        "impairment": _const({"W": 20}),
        "restructuring": _const({"W": 10}),
        "litigationExpense": _const({"W": 10}),        # charge -> add back  (charges=40)
        "gainOnDispositions": _const({"W": 5}),
        "gainOnSaleGeneric": _const({"W": 5}),         # gains -> remove     (gains=10)
        "discontinuedOps": _const({"W": 10}),          # net-of-tax income -> remove
    }
    F = _core_earnings_fields(_mock(frames), _const({"W": 1000}))
    last = IDX[-1]
    # special = charges(40) - gains(10) = 30
    assert abs(F["nonrecurring_pretax_share"].loc[last, "W"] - 0.30) < 1e-6
    assert abs(F["special_items_intensity"].loc[last, "W"] - 0.03) < 1e-6
    # core_ni = 80 + 30*(1-0.2) - 10 (discontinued) = 80 + 24 - 10 = 94
    assert abs(F["core_profit_margin"].loc[last, "W"] - 0.094) < 1e-6

    print("\n=== SANITY CHECK: #1 widened special-items pool ===")
    print(f"  charges 40 (incl litigation) - gains 10 (incl generic sale) = 30 special; "
          f"core_ni = 80 + 24 - 10 disc-ops = 94 -> core margin "
          f"{F['core_profit_margin'].loc[last,'W']:.3f}. Widened pool + disc-ops. Validated.")


def test_ai_leverage_it_maturity():
    """capitalized_software / assets is the IT-investment (capability) proxy; it
    populates once CapitalizedComputerSoftware* is extracted."""
    frames = {
        "capitalizedSoftware": _const({"A": 200, "B": 20}),
        "totalAssets": _const({"A": 1000, "B": 1000}),
        "totalRevenue": _const({"A": 800, "B": 800}),
    }
    F = _ai_leverage_fields(_mock(frames))
    last = IDX[-1]
    assert abs(F["capitalized_software_intensity"].loc[last, "A"] - 0.20) < 1e-6
    assert F["capitalized_software_intensity"].loc[last, "A"] > F["capitalized_software_intensity"].loc[last, "B"]
    assert abs(F["software_to_revenue"].loc[last, "A"] - 200 / 800) < 1e-6

    print("\n=== SANITY CHECK: #4 AI-leverage IT maturity ===")
    print(f"  A capitalized-software/assets = 0.20 (IT-mature) >> B = 0.02; "
          f"software/revenue = 0.25. Capability proxy ready (score assembled as composite). Validated.")


def test_pension_adjusted_ev_and_overhang_leverage():
    """Pension/OPEB deficit is added to the True EV (debt-like), a pension_overhang_leverage
    ratio (deficit / market cap) is emitted, and the deficit is surfaced as
    pension_retirement_liability. The EV inclusion lowers the EV yields vs no-pension."""
    fh = pd.DataFrame([{"ticker": "P", "as_of": "2019-12-31",
                        "sharesOutstanding": 100.0, "ebitda": 50.0, "cash": 10.0,
                        "longTermDebt": 200.0, "pensionDeficit": 80.0}])
    idx = pd.bdate_range("2020-01-02", periods=30)
    close = pd.DataFrame({"P": 5.0}, index=idx)        # market cap = 100 * 5 = 500
    F = _derived_fields(fh, idx, close)
    d = idx[-1]
    assert abs(F["pension_retirement_liability"].loc[d, "P"] - 80.0) < 1e-6
    assert abs(F["pension_overhang_leverage"].loc[d, "P"] - 80.0 / 500.0) < 1e-9   # 0.16
    # True EV = 500 mcap + 200 debt + 80 pension - 10 cash = 770 ; ebitda_to_ev = 50/770
    assert abs(F["ebitda_to_ev"].loc[d, "P"] - 50.0 / 770.0) < 1e-6
    assert F["ebitda_to_ev"].loc[d, "P"] < 50.0 / 690.0    # lower than EV without the pension add

    print("\n=== SANITY CHECK: pension-adjusted EV + overhang leverage ===")
    print(f"  pension_retirement_liability=80; pension_overhang_leverage=80/500={80/500:.2f}; "
          f"EV=770 (incl. pension) -> ebitda_to_ev={50/770:.4f} < 50/690 (no pension). Validated.")


def test_operating_margin_5y_trend_and_refinancing_risk():
    """5y operating-margin trend (structural expansion) + refinancing risk
    (short-term debt vs cash + FCF liquidity)."""
    idx = pd.bdate_range("2018-01-02", periods=6 * 252 + 20)   # >5y so shift(_FIVE_YEARS) exists
    rows = []
    for i, yr in enumerate(range(2018, 2025)):                  # margin 0.10 -> 0.22 over 6y
        rows.append({"ticker": "M", "as_of": f"{yr}-06-30", "totalRevenue": 1000.0,
                     "operatingIncome": 100.0 + 20.0 * i, "shortTermDebt": 200.0,
                     "cash": 50.0, "freeCashflow": 50.0, "sharesOutstanding": 100.0})
    F = _derived_fields(pd.DataFrame(rows), idx, pd.DataFrame({"M": 10.0}, index=idx))
    d = idx[-1]
    # refinancing risk = ST debt / (cash + positive FCF) = 200 / (50+50) = 2.0
    assert abs(F["refinancing_risk"].loc[d, "M"] - 2.0) < 1e-9
    # operating margin expanded materially over ~5y (now ~0.22 vs ~0.12 five years back)
    assert "operating_margin_5y_chg" in F
    assert F["operating_margin_5y_chg"].loc[d, "M"] > 0.05

    print("\n=== SANITY CHECK: 5y margin trend + refinancing risk ===")
    print(f"  operating_margin_5y_chg={F['operating_margin_5y_chg'].loc[d,'M']:+.3f} (>0.05, "
          f"structural expansion); refinancing_risk={F['refinancing_risk'].loc[d,'M']:.2f} "
          f"(200 ST debt / 100 liquidity = 2.0x). Validated.")


def test_pension_footnote_features_from_notes_num():
    """Financial Statement & NOTES sets (`notes_num`) supply the footnote PBO and
    plan assets the primary statements never expose:
      * pension_funded_ratio = plan assets / PBO,
      * pbo_to_mcap, pension_underfunding_to_mcap (scale vs equity value),
      * and the footnote funded status (PBO - assets) GAP-FILLS the recognized
        pension deficit -> pension_retirement_liability + True EV for a name that
        has NO balance-sheet net-liability / companyfacts pension tag."""
    # No pensionDeficit / pension_facts here -> the ONLY pension source is the footnote.
    fh = pd.DataFrame([{"ticker": "U", "as_of": "2019-12-31",
                        "sharesOutstanding": 100.0, "ebitda": 50.0, "cash": 10.0,
                        "longTermDebt": 200.0}])
    idx = pd.bdate_range("2020-01-02", periods=30)
    close = pd.DataFrame({"U": 5.0}, index=idx)          # market cap = 100 * 5 = 500
    notes_num = pd.DataFrame([
        {"ticker": "U", "tag": "DefinedBenefitPlanBenefitObligation",
         "ddate": "2019-09-30", "qtrs": 0, "value": 1000.0, "filed": "2019-11-15"},
        {"ticker": "U", "tag": "DefinedBenefitPlanFairValueOfPlanAssets",
         "ddate": "2019-09-30", "qtrs": 0, "value": 600.0, "filed": "2019-11-15"},
        # a DURATION fact (qtrs>0) must be ignored by the instant PBO/asset reshape:
        {"ticker": "U", "tag": "DefinedBenefitPlanBenefitObligation",
         "ddate": "2019-09-30", "qtrs": 4, "value": 99.0, "filed": "2019-11-15"},
    ])
    F = _derived_fields(fh, idx, close, notes_num=notes_num)
    d = idx[-1]
    assert abs(F["pension_funded_ratio"].loc[d, "U"] - 0.6) < 1e-9          # 600 / 1000
    assert abs(F["pbo_to_mcap"].loc[d, "U"] - 1000.0 / 500.0) < 1e-9        # 2.0
    assert abs(F["pension_underfunding_to_mcap"].loc[d, "U"] - 400.0 / 500.0) < 1e-9  # 0.8
    # footnote deficit (1000-600=400) fills the recognized pension deficit + EV:
    assert abs(F["pension_retirement_liability"].loc[d, "U"] - 400.0) < 1e-6
    assert abs(F["pension_overhang_leverage"].loc[d, "U"] - 400.0 / 500.0) < 1e-9
    # True EV = 500 mcap + 200 debt + 400 pension - 10 cash = 1090
    assert abs(F["ebitda_to_ev"].loc[d, "U"] - 50.0 / 1090.0) < 1e-6

    # absent notes_num -> footnote features simply don't appear (no crash)
    F0 = _derived_fields(fh, idx, close)
    assert "pension_funded_ratio" not in F0 and "pbo_to_mcap" not in F0

    print("\n=== SANITY CHECK: pension FOOTNOTE features (notes_num) ===")
    print(f"  PBO=1000, plan assets=600 -> funded_ratio=0.60, pbo_to_mcap=2.0, "
          f"underfunding/mcap=0.80; footnote deficit 400 (no other source) fills "
          f"pension_retirement_liability=400 & EV=1090 -> ebitda_to_ev={50/1090:.4f}. "
          f"Duration (qtrs>0) PBO ignored. Absent notes_num -> features skipped. Validated.")


def test_absent_new_tags_do_not_break_existing_factors():
    """Before the DB is re-fetched the new tags are absent; the helpers must still
    produce the existing factors (empty daily frame -> component contributes 0)."""
    empty = pd.DataFrame()
    daily = lambda f: {  # only the pre-existing columns present
        "totalRevenue": _const({"X": 1000}), "pretaxIncome": _const({"X": 100}),
        "netIncome": _const({"X": 80}), "operatingIncome": _const({"X": 120}),
        "ebitda": _const({"X": 150}), "incomeTaxExpense": _const({"X": 20}),
        "impairment": _const({"X": 30}), "restructuring": _const({"X": 10}),
        "gainOnDispositions": _const({"X": 0}),
    }.get(f, empty)
    F = _core_earnings_fields(daily, _const({"X": 1000}))
    last = IDX[-1]
    # litigation/generic-gain/discontinued absent -> special = 30+10 = 40, core_ni = 80+32 = 112
    assert abs(F["core_profit_margin"].loc[last, "X"] - 0.112) < 1e-6
    print("\n=== SANITY CHECK: graceful degradation before re-fetch ===")
    print("  new tags absent -> core margin = 0.112 (impair+restr only), no crash. Validated.")
