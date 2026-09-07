"""Correctness tests for the business-quality factors in fundamental_features.py
(SBC-vs-buyback, #5 forensic, #3 M&A digestion, pension).

Each helper takes the memoized `daily` accessor (field -> date x ticker frame);
here we feed it hand-built frames with KNOWN values and assert the exact ratio
math, plus the two things that matter most:
  * the SIGN CONVENTIONS the live table actually stores -- `equityIssuanceNet` is NET
    ISSUANCE (negative = buyback), `dividendsPaid` is outflow-POSITIVE;
  * the Beneish M-score RANKS a manipulation profile above a clean one.

GONE, with the features they covered: the core/adjusted-earnings tests (the family needs
ten special-items tags Sharadar does not deliver), the AI-leverage test
(`capitalizedSoftware`), the D&A-realism test (`ppeGross` / `accumulatedDepreciation` /
`amortizationIntangibles`), and the graceful-degradation test that asserted a core margin.
Each was PASSING against a synthetic frame that hand-built the missing source column, which
is how 24 dead features stayed green for a whole vintage of the table.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_aggregate.utils.common import capital
from src.data_aggregate.utils.fundamentals.fundamental_features import (
    _beneish_m_score, _da_realism_fields, _derived_fields, _digestion_fields, _forensic_fields,
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


def test_sbc_to_buyback_reads_net_issuance_as_a_signed_line():
    """`sbc_to_buyback` > 1 means the repurchase does not even cover the stock given away.

    THE SOURCE COLUMN IS NET ISSUANCE, NOT BUYBACKS. `equityIssuanceNet` is negative for a
    repurchaser and positive for a firm raising equity, so the magnitude has to come from
    `capital.share_repurchases` (which floors the issuing side at 0) and NOT from `.abs()` --
    `.abs()` would read a $100 equity RAISE as $100 of buybacks, the opposite signal."""
    frames = {
        "stockBasedComp": _const({"BUYER": 50, "ISSUER": 50}),
        # BUYER repurchased 100 (negative = outflow); ISSUER raised 100 (positive)
        "equityIssuanceNet": _const({"BUYER": -100, "ISSUER": 100}),
    }
    F = _da_realism_fields(_mock(frames))
    last = IDX[-1]
    assert abs(F["sbc_to_buyback"].loc[last, "BUYER"] - 0.5) < 1e-9
    # the issuer bought back NOTHING -> the ratio is undefined, never 0.5
    assert np.isnan(F["sbc_to_buyback"].loc[last, "ISSUER"])

    print("\n=== SANITY CHECK: sbc_to_buyback on NET ISSUANCE ===")
    print(f"  BUYER equityIssuanceNet=-100 -> 100 repurchased, SBC 50 -> ratio "
          f"{F['sbc_to_buyback'].loc[last,'BUYER']:.2f} (buyback covers 2x the SBC).")
    print(f"  ISSUER equityIssuanceNet=+100 (a RAISE) -> repurchases floored to 0 -> NaN, "
          f"not the 0.50 an .abs() would have fabricated. Validated.")


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


def test_beneish_is_null_without_revenue_and_assets():
    """The M-score must be NaN for a ticker with no revenue/assets, not the neutral constant.

    The score is seeded as a DENSE frame at -4.84 and every absent index is filled with its
    neutral value, so a company with no data at all scored exactly
    -4.84 + 0.920 + 0.528 + 0.404 + 0.892 + 0.115 - 0.172 - 0.327 = -2.48 — below the
    classic M > -1.78 manipulation threshold, i.e. a clean bill of health for a company that
    did not exist yet. Measured on the live cube: 565,720 pre-listing rows all carrying that
    identical constant, contaminating a per-day cross-sectional rank.

    The support is keyed on DATA, not on listing: a spin-off that files before it starts
    trading keeps the score its filings support."""
    frames = {
        "totalRevenue": _const({"REAL": 1000, "GHOST": np.nan}),
        "totalAssets": _const({"REAL": 1200, "GHOST": np.nan}),
        "accountsReceivable": _const({"REAL": 200, "GHOST": np.nan}),
        "netIncome": _const({"REAL": 80, "GHOST": np.nan}),
    }
    m = _beneish_m_score(_mock(frames), IDX)
    last = IDX[-1]
    assert np.isfinite(m.loc[last, "REAL"])
    assert np.isnan(m.loc[last, "GHOST"]), "M fabricated for a ticker with no data"
    # the specific value it used to fabricate, so the regression is named not just implied
    assert not np.isclose(m.loc[last, "GHOST"], -2.48, equal_nan=False)
    assert m["GHOST"].notna().sum() == 0

    print("\n=== SANITY CHECK: Beneish support is enforced per cell ===")
    print(f"  REAL (revenue+assets present) M = {m.loc[last,'REAL']:+.2f}; GHOST (neither) "
          f"is NaN on all {len(IDX)} rows — not the -2.48 the all-neutral seed used to "
          f"fabricate, which would have read as a clean bill of health. Validated.")


def test_forensic_days_and_offbs_leverage():
    """The pension leg reaches `capital.py` ONLY as the caller's coalesced frame.

    It used to be supplied here as a `pensionDeficit` field on the `daily` accessor, which
    is a column `fundamentals_history` has never had on the Sharadar-first schema — so the
    test was green on a leg that contributed zero in production. `capital.py` no longer has
    that field-getter fallback; the deficit is the third positional argument, exactly as
    `_derived_fields` passes it from `_pension_pool`.

    ⚠ `net_debt_incl_offbs_to_ebitda` USED TO BE ASSERTED HERE AND IS NOW DELETED. Three of
    its four distinguishing legs — operating leases, finance leases and asset-retirement
    obligations — do not exist as columns on the Sharadar substrate, and this fixture
    hand-built two of them, which is exactly how a feature stays green while collapsing in
    production. Once the distress block was fixed to read the reconciled `totalDebt`, the
    feature became byte-identical to `net_debt_to_ebitda` on the fingerprint slice. What is
    asserted instead is the arithmetic it relied on, still live in `capital.py` and still
    reachable through `net_debt(off_balance_sheet=True)`."""
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
        "cash": _const({"X": 80}),
        "ebitda": _const({"X": 150}),
    }
    get = _mock(frames)
    F = _forensic_fields(get, IDX)
    last = IDX[-1]
    dso, dpo, dio = 200 / 1000 * 365, 150 / 600 * 365, 100 / 600 * 365
    assert abs(F["dso"].loc[last, "X"] - dso) < 1e-6
    assert abs(F["dpo"].loc[last, "X"] - dpo) < 1e-6
    assert abs(F["dio"].loc[last, "X"] - dio) < 1e-6
    assert abs(F["cash_conversion_cycle"].loc[last, "X"] - (dso + dio - dpo)) < 1e-6
    assert "net_debt_incl_offbs_to_ebitda" not in F        # deleted duplicate

    # the off-BS arithmetic itself, asserted where it now lives: recognized net deficit 30
    # is debt-like and is passed in, the way `_derived_fields` passes it from `_pension_pool`
    net_od = capital.net_debt(get, off_balance_sheet=True, pension=_const({"X": 30}))
    assert abs(net_od.loc[last, "X"] - (400 + 70 + 30 - 80)) < 1e-6

    print("\n=== SANITY CHECK: #5 forensic working-capital + off-BS leverage ===")
    print(f"  DSO={dso:.1f}d, DPO={dpo:.1f}d, DIO={dio:.1f}d, CCC={dso+dio-dpo:.1f}d.")
    print(f"  off-BS net debt = 400 debt + 70 leases + 30 pension - 80 cash = "
          f"{net_od.loc[last,'X']:.0f} (capital.py's arithmetic, still live).")
    print("  NOTE: the lease legs are hand-built here and do not exist on the Sharadar table; "
          "the feature that divided this by EBITDA was deleted for collapsing onto "
          "net_debt_to_ebitda once both read the reconciled totalDebt.")


def test_digestion_roic_wedge_on_combined_intangibles():
    """ROIC with vs without ACQUIRED INTANGIBLES, and the wedge between them.

    The deduction is Sharadar's COMBINED `intangibles` (goodwill + other), the only
    intangibles basis SF1 delivers. Reading the bare `goodwill` -- which no producer writes --
    made the ex-goodwill ROIC subtract NOTHING, so it correlated 1.0000 with its incl twin and
    the drag was identically zero. This test exists to keep the deduction non-empty."""
    frames = {
        "operatingIncome": _const({"X": 200}),
        "incomeTaxExpense": _const({"X": 40}),
        "pretaxIncome": _const({"X": 160}),          # effective tax 40/160 = 25%
        "stockholdersEquity": _const({"X": 500}),
        "cash": _const({"X": 100}),
        "longTermDebt": _const({"X": 200}),
        "shortTermDebt": _const({"X": 0}),
        "intangibles": _const({"X": 250}),           # goodwill + other, COMBINED
        "totalAssets": _const({"X": 1000}),
    }
    F = _digestion_fields(_mock(frames), pd.DataFrame(), IDX, 4)
    last = IDX[-1]
    # NOPAT = 200*(1-0.25)=150 ; IC = 500+200-100 = 600 ; roic_incl = 0.25
    assert abs(F["roic_incl_intangibles"].loc[last, "X"] - 0.25) < 1e-6
    # IC ex intangibles = 600-250 = 350 ; roic_ex = 150/350
    assert abs(F["roic_ex_intangibles"].loc[last, "X"] - 150 / 350) < 1e-6
    # the wedge must be NON-ZERO and negative: acquisitions dilute returns
    assert F["intangibles_roic_drag"].loc[last, "X"] < -1e-6
    assert abs(F["intangibles_to_assets"].loc[last, "X"] - 0.25) < 1e-6
    assert abs(F["intangibles_to_equity"].loc[last, "X"] - 0.50) < 1e-6

    print("\n=== SANITY CHECK: #3 M&A digestion on combined intangibles ===")
    print(f"  ROIC incl intangibles = {F['roic_incl_intangibles'].loc[last,'X']:.3f} "
          f"vs ex = {F['roic_ex_intangibles'].loc[last,'X']:.3f} "
          f"-> drag {F['intangibles_roic_drag'].loc[last,'X']:+.3f} (non-zero, so the "
          f"deduction is real); intangibles = 25% of assets, 50% of equity. Validated.")


def test_pension_adjusted_ev_and_overhang_leverage():
    """Pension/OPEB deficit is added to the True EV (debt-like), a pension_overhang_leverage
    ratio (deficit / market cap) is emitted, and the deficit is surfaced as
    pension_retirement_liability. The EV inclusion lowers the EV yields vs no-pension.

    The deficit comes from `pension_facts` — the bulk Financial-Statement-Data-Sets
    recognized net liability, the pool's PRIMARY leg. It used to be hand-built here as a
    `pensionDeficit` column on `fund_hist`, so this test asserted a source that cannot fire
    in production: the Sharadar-first `fundamentals_history` carries no pension column at
    all. Now it exercises the code path the live build actually takes."""
    fh = pd.DataFrame([{"ticker": "P", "as_of": "2019-12-31",
                        "sharesOutstanding": 100.0, "ebitda": 50.0, "cash": 10.0,
                        "longTermDebt": 200.0}])
    idx = pd.bdate_range("2020-01-02", periods=30)
    close = pd.DataFrame({"P": 5.0}, index=idx)        # market cap = 100 * 5 = 500
    pension_facts = pd.DataFrame([
        {"ticker": "P",
         "tag": "PensionAndOtherPostretirementDefinedBenefitPlansLiabilitiesNoncurrent",
         "ddate": "2019-09-30", "qtrs": 0, "value": 80.0, "filed": "2019-11-15"},
        # a DURATION fact (qtrs>0) is periodic pension COST, not the balance -> must be ignored
        {"ticker": "P",
         "tag": "PensionAndOtherPostretirementDefinedBenefitPlansLiabilitiesNoncurrent",
         "ddate": "2019-09-30", "qtrs": 4, "value": 999.0, "filed": "2019-11-15"},
    ])
    F = _derived_fields(fh, idx, close, pension_facts=pension_facts)
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
      * pbo_to_mcap (gross obligation vs equity value),
      * and the footnote funded status (PBO - assets) GAP-FILLS the recognized
        pension deficit -> pension_retirement_liability + True EV for a name that
        has NO balance-sheet net-liability / companyfacts pension tag.

    `pension_underfunding_to_mcap` is deliberately NOT asserted: it was deleted because it
    scaled the same footnote deficit that is the last fallback of the coalesced pool
    `pension_overhang_leverage` uses, measuring r = 1.000000 against it on its own support."""
    # No `pension_facts` here -> the ONLY pension source is the footnote.
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
    assert "pension_underfunding_to_mcap" not in F                          # deleted duplicate
    # footnote deficit (1000-600=400) fills the recognized pension deficit + EV:
    assert abs(F["pension_retirement_liability"].loc[d, "U"] - 400.0) < 1e-6
    assert abs(F["pension_overhang_leverage"].loc[d, "U"] - 400.0 / 500.0) < 1e-9
    # True EV = 500 mcap + 200 debt + 400 pension - 10 cash = 1090
    assert abs(F["ebitda_to_ev"].loc[d, "U"] - 50.0 / 1090.0) < 1e-6

    # absent notes_num -> footnote features simply don't appear (no crash)
    F0 = _derived_fields(fh, idx, close)
    assert "pension_funded_ratio" not in F0 and "pbo_to_mcap" not in F0

    print("\n=== SANITY CHECK: pension FOOTNOTE features (notes_num) ===")
    print(f"  PBO=1000, plan assets=600 -> funded_ratio=0.60, pbo_to_mcap=2.0; "
          f"footnote deficit 400 (no other source) fills pension_retirement_liability=400 "
          f"& EV=1090 -> ebitda_to_ev={50/1090:.4f}. Duration (qtrs>0) PBO ignored. "
          f"Absent notes_num -> features skipped. Validated.")


def test_liquid_assets_does_not_double_count_short_term_investments():
    """D1b: `cash` ALREADY CONTAINS short-term investments, so it must not be added again.

    The field map defines `cash = cashAndEquivalents + coalesce(shortTermInvestments, 0)`
    and maps `shortTermInvestments` straight from `investmentsc`. `liquid_assets` used to
    return `cash + shortTermInvestments + marketableSecuritiesCurrent`, subtracting the same
    money twice from EV, net debt and invested capital. Its docstring's justification —
    `cash` is "investment-free (the extractor nets the broader totals down)" — was true of
    the SEC-era extractor and false on the Sharadar substrate.

    Magnitude, `investmentsc`/market cap at filing grain: ~0 median in nine sectors but
    1.30% in Information Technology (p90 13.36%), so it understated EV most for exactly the
    cash-rich tech names where EV yields matter."""
    frames = {
        "cash": _const({"X": 125}),                  # = 100 cashneq + 25 ST investments
        "shortTermInvestments": _const({"X": 25}),   # the SAME 25, already inside `cash`
        "longTermDebt": _const({"X": 400}),
        "shortTermDebt": _const({"X": 0}),
    }
    get = _mock(frames)
    last = IDX[-1]
    liquid = capital.liquid_assets(get)
    assert liquid.loc[last, "X"] == 125.0, "short-term investments counted twice"
    # and net debt therefore nets 125, not 150
    assert capital.net_debt(get).loc[last, "X"] == 400.0 - 125.0

    print("\n=== SANITY CHECK: liquid_assets counts ST investments ONCE ===")
    print(f"  cash=125 (100 cashneq + 25 STI) with shortTermInvestments=25 also present -> "
          f"liquid_assets={liquid.loc[last,'X']:.0f}, not 150; net debt "
          f"{capital.net_debt(get).loc[last,'X']:.0f} = 400 - 125. Validated.")


def test_bank_cash_is_not_netted_from_ev_but_still_ratios_into_cash_to_debt():
    """D1c: a bank's cash is a CORE OPERATING ASSET, so it is excluded from every NETTING
    site — and from no ratio.

    "Cash and due from banks" is required reserves and interbank float; insurance cash is
    claims float. `liquid_assets` already excludes the AFS/HTM investment book
    (`investmentSecurities`) on exactly this reasoning. The gate became load-bearing when
    `cash` was widened to `cashneq + coalesce(investmentsc, 0)`: before that a bank's cash
    was simply NULL (an unclassified balance sheet reports no current-investments line), so
    the netting sites netted nothing anyway. Restoring it ungated would have re-rated every
    bank — median `cashneq` is 10.97% of Financials market cap, 101.95% at p90, so the top
    decile's cash exceeds its whole equity value and EV would go NEGATIVE.

    `cash_to_debt` is the control: a liquidity cushion is a real fact about a bank, and it
    is the feature the widening exists to restore (`cash` was NULL on 28.1% of Financials
    filings). Only the netting is wrong."""
    base = {"as_of": "2019-12-31", "sharesOutstanding": 100.0, "ebitda": 50.0,
            "cash": 300.0, "totalDebt": 200.0, "longTermDebt": 200.0, "shortTermDebt": 0.0}
    fh = pd.DataFrame([{"ticker": "BANKX", "industry_group": "Banks", **base},
                       {"ticker": "TECHY", "industry_group": "Software & Services", **base}])
    idx = pd.bdate_range("2020-01-02", periods=30)
    close = pd.DataFrame({"BANKX": 5.0, "TECHY": 5.0}, index=idx)   # mcap = 500 each
    F = _derived_fields(fh, idx, close)
    d = idx[-1]

    # TECHY nets its cash: EV = 500 mcap + 200 debt - 300 cash = 400
    assert abs(F["ebitda_to_ev"].loc[d, "TECHY"] - 50.0 / 400.0) < 1e-9
    # BANKX does NOT: EV = 500 + 200 = 700, identical to holding no cash at all
    assert abs(F["ebitda_to_ev"].loc[d, "BANKX"] - 50.0 / 700.0) < 1e-9
    assert F["ebitda_to_ev"].loc[d, "BANKX"] < F["ebitda_to_ev"].loc[d, "TECHY"]
    # ...but the RATIO is identical for both — the bank keeps its liquidity cushion
    assert abs(F["cash_to_debt"].loc[d, "BANKX"] - 1.5) < 1e-9
    assert abs(F["cash_to_debt"].loc[d, "TECHY"] - 1.5) < 1e-9

    print("\n=== SANITY CHECK: bank cash is gated out of NETTING, not out of RATIOS ===")
    print(f"  same balance sheet, different GICS group: EV(TECHY)=400 -> ebitda_to_ev "
          f"{F['ebitda_to_ev'].loc[d,'TECHY']:.4f}; EV(BANKX)=700 (cash NOT netted) -> "
          f"{F['ebitda_to_ev'].loc[d,'BANKX']:.4f}. cash_to_debt is 1.50 for BOTH — the "
          f"widening fix still reaches the bank. Validated.")


def test_distress_debt_prefers_the_reconciled_total_debt_column():
    """The distress block's DENOMINATOR fell into the same trap as `cash`.

    It built debt as `longTermDebt + shortTermDebt`, and both legs are absent for every
    filer that does not classify its balance sheet — measured, each is 0.277 populated for
    Financials and 0.274 for Real Estate, while the reconciled `totalDebt` column is 1.000
    in EVERY sector. So restoring `cash` (D1a) fixed the numerator of `cash_to_debt` and the
    row was still nulled by the denominator: Financials' cube fill sat at 0.256 with a
    balance-sheet anchor of 0.986, and BRK-B, BX, AXP, BAC, C and every other bank scored a
    flat 0.0.

    Substitution is safe because it is the same quantity: where both legs exist `totalDebt`
    equals their sum to within 1% on 100.00% of rows across all eleven sectors."""
    idx = pd.bdate_range("2020-01-02", periods=30)
    close = pd.DataFrame({"BANKX": 5.0, "MFG": 5.0}, index=idx)
    fh = pd.DataFrame([
        # a bank: NO current/non-current split, so neither leg column exists
        dict(ticker="BANKX", as_of="2019-12-31", sharesOutstanding=100.0, ebitda=50.0,
             cash=300.0, totalDebt=200.0, totalRevenue=400.0, totalAssets=2000.0,
             industry_group="Banks"),
        # a manufacturer: both legs reported, and totalDebt agrees with their sum
        dict(ticker="MFG", as_of="2019-12-31", sharesOutstanding=100.0, ebitda=50.0,
             cash=300.0, totalDebt=200.0, longTermDebt=150.0, shortTermDebt=50.0,
             totalRevenue=400.0, totalAssets=2000.0, industry_group="Capital Goods"),
    ])
    F = _derived_fields(fh, idx, close)
    d = idx[-1]

    # BOTH get the feature, and both get the same value: 300 / 200
    assert abs(F["cash_to_debt"].loc[d, "BANKX"] - 1.5) < 1e-9, \
        "a leg-less filer still has no debt denominator"
    assert abs(F["cash_to_debt"].loc[d, "MFG"] - 1.5) < 1e-9
    # the leg-reporting filer is UNCHANGED -- this is a coverage fix, not a re-basing
    assert abs(F["net_debt_to_ebitda"].loc[d, "MFG"] - (200.0 - 300.0) / 50.0) < 1e-9
    assert abs(F["net_debt_to_ebitda"].loc[d, "BANKX"] - 200.0 / 50.0) < 1e-9, \
        "BANKX is a bank: its cash must NOT be netted (D1c), so net debt is the gross 200"

    print("\n=== SANITY CHECK: distress debt uses the reconciled totalDebt ===")
    print(f"  BANKX reports NO longTermDebt/shortTermDebt (unclassified balance sheet) yet "
          f"cash_to_debt={F['cash_to_debt'].loc[d,'BANKX']:.2f} — it used to be NaN.")
    print(f"  MFG reports both legs summing to the same 200 -> identical "
          f"{F['cash_to_debt'].loc[d,'MFG']:.2f}: coverage gained, no value re-based.")
    print(f"  And D1c still holds: net_debt_to_ebitda is "
          f"{F['net_debt_to_ebitda'].loc[d,'BANKX']:.1f}x for the bank (cash NOT netted) "
          f"vs {F['net_debt_to_ebitda'].loc[d,'MFG']:.1f}x for the manufacturer. Validated.")


def test_pension_has_exactly_two_sources_no_fundamentals_history_column():
    """REGRESSION GUARD: the pension overhang has TWO sources, and a `fundamentals_history`
    column is not one of them.

    A third leg used to sit in the coalesce, reading `pensionDeficit` off the fundamentals
    frame. `fundamentals_history` has no pension column on the Sharadar-first schema — an
    `information_schema` match on `%pension%`/`%opeb%`/`%benefit%` returns nothing — so the
    leg contributed exactly zero on every live row while two tests kept it green by
    hand-building the column. Feeding it here must now change NOTHING.

    This is the D2 no-op assertion, expressed at the unit level: if this test starts
    failing, a field-getter fallback has been reintroduced and the cube would silently
    diverge from the two bulk SEC sources that are its real substrate."""
    base = {"ticker": "Z", "as_of": "2019-12-31", "sharesOutstanding": 100.0,
            "ebitda": 50.0, "cash": 10.0, "longTermDebt": 200.0}
    idx = pd.bdate_range("2020-01-02", periods=30)
    close = pd.DataFrame({"Z": 5.0}, index=idx)

    without = _derived_fields(pd.DataFrame([base]), idx, close)
    with_col = _derived_fields(pd.DataFrame([{**base, "pensionDeficit": 80.0}]), idx, close)

    # no pension source at all -> no pension features, with or without the phantom column
    for F in (without, with_col):
        assert "pension_retirement_liability" not in F
        assert "pension_overhang_leverage" not in F
    # and EV is pension-free either way: 500 mcap + 200 debt - 10 cash = 690
    for F in (without, with_col):
        assert abs(F["ebitda_to_ev"].loc[idx[-1], "Z"] - 50.0 / 690.0) < 1e-9

    print("\n=== SANITY CHECK: pension pool has exactly TWO sources ===")
    print(f"  A `pensionDeficit` column on fundamentals_history is IGNORED: no pension "
          f"feature is emitted and EV stays 690 (= 500 mcap + 200 debt - 10 cash), "
          f"identical to the run without it. The dead third leg cannot come back. "
          f"Live sources: pension_facts (125 tickers) + notes_num footnote (214 CIKs) "
          f"-> 199 of 489 tickers. Validated.")
