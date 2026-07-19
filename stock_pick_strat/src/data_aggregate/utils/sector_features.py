"""
sector_features.py  (src/data_aggregate/utils/sector_features.py)
------------------------------------------------------------------
Derived, sector-specific fundamental KPIs computed from the expanded SEC
fundamentals history (the raw line items added in fetch_fundamentals). Two
layers:

  * compute_sector_kpis(df)          -> pure, row-level ratios (one value per
                                        filing). Gated by INPUT AVAILABILITY, so a
                                        KPI is NaN unless its inputs were reported
                                        (e.g. loss_ratio only exists for insurers
                                        that report premiums) — inherently
                                        sector-correct, no hard-coded sector list.
  * build_sector_feature_panel(...)  -> turns the chosen KPIs into a daily,
                                        peer-relative feature panel (same
                                        machinery as the fundamental panel), so
                                        they drop straight into the cube.

Every KPI is point-in-time (computed from TTM flows / period-end levels keyed on
the filing date `as_of`) and unit-free (a ratio), so it is comparable across
names and safe to neutralize within an industry group for a L/S book.

KPIs (grouped):
  universal      effective_tax_rate, interest_coverage, net_debt_to_ebitda,
                 accruals_ratio, gross_profitability, asset_turnover (avg assets),
                 capex_intensity, capex_to_dep, payout_ratio, buyback_intensity,
                 cash_conversion_cycle (dso/dio/dpo), roic, earnings_quality,
                 reinvestment_rate, sustainable_growth_rate,
                 fixed_cost_coverage_margin, gmroi
  banks          net_interest_margin, efficiency_ratio, provision_rate,
                 loan_to_deposit, bank_roa, bank_operating_margin,
                 reserve_coverage_velocity, tier1_capital_ratio, deposit_stickiness
  insurance      loss_ratio, expense_ratio, combined_ratio, investment_income_ratio
  reits          ffo_margin, ffo_payout, rental_margin, affo_margin,
                 net_debt_to_ebitdare
  energy         exploration_intensity, ddna_intensity, ebitdax_margin,
                 property_overvaluation_cushion
  software/tech  deferred_rev_intensity, rpo_coverage, sbc_intensity
  utilities      regulatory_asset_ratio, capex_to_rate_base
  pharma/biotech patent_cliff, rd_capitalized_roic (5y-capitalized R&D)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_aggregate.utils.factors import fundamentals_to_daily
from src.data_aggregate.utils.fundamental_features import (
    _infer_yoy_periods, build_peer_relative_panel,
)

# KPI columns produced by compute_sector_kpis (the panel builder iterates these)
SECTOR_KPI_COLS: list[str] = [
    # universal
    "effective_tax_rate", "interest_coverage", "net_debt_to_ebitda", "accruals_ratio",
    "gross_profitability", "asset_turnover", "capex_intensity", "capex_to_dep",
    "payout_ratio", "buyback_intensity", "days_sales_outstanding",
    "days_inventory_outstanding", "days_payable_outstanding", "cash_conversion_cycle",
    "roic", "earnings_quality", "reinvestment_rate", "sustainable_growth_rate",
    "fixed_cost_coverage_margin", "gmroi",
    # banks
    "net_interest_margin", "efficiency_ratio", "provision_rate", "loan_to_deposit", "bank_roa",
    "bank_operating_margin", "reserve_coverage_velocity", "tier1_capital_ratio", "deposit_stickiness",
    # insurance
    "loss_ratio", "expense_ratio", "combined_ratio", "investment_income_ratio",
    # reits
    "ffo_margin", "ffo_payout", "rental_margin", "affo_margin", "net_debt_to_ebitdare",
    # energy
    "exploration_intensity", "ddna_intensity", "ebitdax_margin", "property_overvaluation_cushion",
    # software / tech
    "deferred_rev_intensity", "rpo_coverage", "sbc_intensity",
    # utilities
    "regulatory_asset_ratio", "capex_to_rate_base",
    # pharma / biotech
    "patent_cliff", "rd_capitalized_roic",
]


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """Numeric column or an all-NaN series when the tag was never reported."""
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def _safe_div(num: pd.Series, den: pd.Series, den_positive: bool = False) -> pd.Series:
    """Elementwise num/den, NaN where den is 0/NaN (or <=0 if den_positive)."""
    den = den.where(den > 0) if den_positive else den.replace(0, np.nan)
    return num / den


def _yearly_lag(df: pd.DataFrame, s: pd.Series, years_back: int, yoy: int) -> pd.Series:
    """`s` shifted back `years_back` fiscal YEARS within each ticker's own series
    (ordered by `as_of`). `yoy` is filings-per-year (4 quarterly, 1 annual). Used
    for multi-year constructions (capitalized R&D); NaN when ticker/as_of absent."""
    if "ticker" not in df.columns or "as_of" not in df.columns or years_back == 0:
        return s if years_back == 0 else pd.Series(np.nan, index=df.index)
    order = pd.to_datetime(df["as_of"], errors="coerce")
    tmp = pd.DataFrame({"ticker": df["ticker"], "o": order, "v": s})
    tmp = tmp.sort_values(["ticker", "o"])
    tmp["lag"] = tmp.groupby("ticker")["v"].shift(years_back * yoy)
    return tmp["lag"].reindex(df.index)


def _capitalized_rd(df: pd.DataFrame, rd: pd.Series, yoy: int) -> tuple[pd.Series, pd.Series]:
    """Damodaran-style capitalized-R&D asset pool and current-year amortization.

    Treats R&D as a 5-year-life intangible: the unamortized asset carries each of
    the last 5 years' R&D at declining weights (1.0, .8, .6, .4, .2), and this
    year's amortization is 1/5 of each of the prior 5 years' R&D. Both are keyed
    on the current filing; only defined when current R&D is reported."""
    asset = pd.Series(0.0, index=df.index)
    for t in range(5):                                   # layers t=0..4 -> weight (1 - 0.2t)
        asset = asset.add(_yearly_lag(df, rd, t, yoy).fillna(0.0) * (1.0 - 0.2 * t),
                          fill_value=0.0)
    amort = pd.Series(0.0, index=df.index)
    for t in range(1, 6):                                # last 5 years each amortize 1/5 this year
        amort = amort.add(_yearly_lag(df, rd, t, yoy).fillna(0.0) * 0.2, fill_value=0.0)
    valid = rd.notna()
    return asset.where(valid), amort.where(valid)


def compute_sector_kpis(fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Return `fundamentals` with the SECTOR_KPI_COLS appended (row-level).

    Pure and side-effect-free. Availability-gated: a KPI is NaN unless every
    input it needs was reported on that row, which is exactly the sector gate
    (insurers report premiums, banks report net interest income, REITs report
    real-estate/depreciation, ...)."""
    if fundamentals is None or fundamentals.empty:
        return fundamentals if fundamentals is not None else pd.DataFrame()

    df = fundamentals.copy()
    g = lambda n: _col(df, n)  # noqa: E731
    yoy = _infer_yoy_periods(df)                     # filings per year (4 quarterly, 1 annual)

    revenue = g("totalRevenue")
    assets = g("totalAssets")
    ebitda = g("ebitda")
    ni = g("netIncome")
    ocf = g("operatingCashFlow")
    cogs = g("costOfRevenue")
    oper_income = g("operatingIncome")
    depamort = g("depAmort")
    capex = g("capex")
    cash = g("cash")
    total_debt = g("longTermDebt").fillna(0) + g("shortTermDebt").fillna(0)

    # ---- universal ------------------------------------------------------- #
    df["effective_tax_rate"] = _safe_div(g("incomeTaxExpense"), g("pretaxIncome"), True)
    df["interest_coverage"] = _safe_div(ebitda, g("interestExpense"), True)

    net_debt = (g("longTermDebt").fillna(0) + g("shortTermDebt").fillna(0)
                + g("operatingLeaseLiability").fillna(0) + g("commercialPaper").fillna(0)
                - g("cash").fillna(0) - g("shortTermInvestments").fillna(0))
    df["net_debt_to_ebitda"] = _safe_div(net_debt, ebitda, True)

    # cash-flow accruals (Sloan): (net income - operating cash flow) / assets
    df["accruals_ratio"] = _safe_div(ni - ocf, assets, True)
    df["gross_profitability"] = _safe_div(g("grossProfit"), assets, True)  # Novy-Marx
    # asset turnover on AVERAGE total assets (mean of current & 1y-prior; falls back
    # to period-end when no prior year is available).
    prior_assets = _yearly_lag(df, assets, 1, yoy)
    avg_assets = ((assets + prior_assets) / 2.0).where(prior_assets.notna(), assets)
    df["asset_turnover"] = _safe_div(revenue, avg_assets, True)
    df["capex_intensity"] = _safe_div(g("capex"), revenue, True)
    df["capex_to_dep"] = _safe_div(g("capex"), g("depAmort"), True)
    df["payout_ratio"] = _safe_div(g("dividendsPaid").fillna(0) + g("buybacks").fillna(0), ni, True)
    df["buyback_intensity"] = _safe_div(g("buybacks"), revenue, True)

    df["days_sales_outstanding"] = _safe_div(g("accountsReceivable") * 365.0, revenue, True)
    df["days_inventory_outstanding"] = _safe_div(g("inventory") * 365.0, cogs, True)
    df["days_payable_outstanding"] = _safe_div(g("accountsPayable") * 365.0, cogs, True)
    df["cash_conversion_cycle"] = (df["days_sales_outstanding"]
                                   + df["days_inventory_outstanding"]
                                   - df["days_payable_outstanding"])

    # ---- capital efficiency & quality (value-creation core) -------------- #
    # NOPAT = operatingIncome x (1 - effective tax); tax clipped to [0,50%] and
    # defaulted to 21% when unreported so ROIC is defined for the whole universe.
    tax = df["effective_tax_rate"].clip(lower=0.0, upper=0.5).fillna(0.21)
    nopat = oper_income * (1.0 - tax)
    invested_capital = total_debt + g("stockholdersEquity").fillna(0.0) - cash.fillna(0.0)
    df["roic"] = _safe_div(nopat, invested_capital, True)          # value created if > WACC
    # earnings quality: operating cash flow backing reported profit (<0.8 = accrual risk)
    df["earnings_quality"] = _safe_div(ocf, ni, True)
    # reinvestment rate: net cash ploughed back (capex - D&A + ΔNWC) per $ of NOPAT
    nwc_now = g("currentAssets") - g("currentLiabilities")
    d_nwc = nwc_now - _yearly_lag(df, nwc_now, 1, yoy)
    df["reinvestment_rate"] = _safe_div(capex.fillna(0.0) - depamort.fillna(0.0) + d_nwc, nopat, True)
    # sustainable growth = ROE x retention (max organic growth w/o new equity/leverage)
    roe = g("returnOnEquity")
    div_payout = _safe_div(g("dividendsPaid"), ni, True).clip(lower=0.0, upper=1.0)
    df["sustainable_growth_rate"] = (roe * (1.0 - div_payout.fillna(0.0))).where(roe.notna())
    # fixed-cost coverage margin = (gross profit - EBITDA) / revenue = overhead intensity
    df["fixed_cost_coverage_margin"] = _safe_div(g("grossProfit") - ebitda, revenue, True)
    # GMROI (retail): gross profit per $ of average inventory investment
    inv = g("inventory")
    prior_inv = _yearly_lag(df, inv, 1, yoy)
    avg_inv = ((inv + prior_inv) / 2.0).where(prior_inv.notna(), inv)
    df["gmroi"] = _safe_div(g("grossProfit"), avg_inv, True).where(inv.notna())

    # ---- banks ----------------------------------------------------------- #
    nii = g("netInterestIncome")
    noninterest_income = g("noninterestIncome")
    bank_gate = nii.notna()
    df["net_interest_margin"] = _safe_div(nii, assets, True)              # NII / total assets (proxy)
    bank_revenue = nii.fillna(0) + noninterest_income.fillna(0)           # NII + noninterest income
    df["efficiency_ratio"] = _safe_div(g("noninterestExpense"), bank_revenue, True)
    df["provision_rate"] = _safe_div(g("provisionForCreditLosses"), g("loans"), True)
    df["loan_to_deposit"] = _safe_div(g("loans"), g("deposits"), True)
    df["bank_roa"] = _safe_div(ni, assets, True).where(bank_gate)         # only for banks
    # operating profitability of the banking model: (revenue - provisions - opex) / revenue
    bank_oi = (bank_revenue - g("provisionForCreditLosses").fillna(0)
               - g("noninterestExpense").fillna(0))
    df["bank_operating_margin"] = _safe_div(bank_oi, bank_revenue, True).where(bank_gate)
    # reserve-build velocity: QoQ change in provisioning relative to the loss allowance.
    # A sharp positive jump = management sees deteriorating credit -> forward-looking short.
    prov = g("provisionForCreditLosses")
    if "ticker" in df.columns and "as_of" in df.columns:
        _o = pd.to_datetime(df["as_of"], errors="coerce")
        _t = pd.DataFrame({"ticker": df["ticker"], "o": _o, "p": prov}).sort_values(["ticker", "o"])
        d_prov = _t.groupby("ticker")["p"].diff().reindex(df.index)
    else:
        d_prov = pd.Series(np.nan, index=df.index)
    df["reserve_coverage_velocity"] = _safe_div(d_prov, g("allowanceCreditLosses"), True)
    # capital adequacy (already a ratio) and deposit-franchise stickiness
    t1 = g("tier1CapitalRatio")
    df["tier1_capital_ratio"] = t1.where(t1 > 0)
    df["deposit_stickiness"] = _safe_div(g("depositsDomestic"), g("totalLiabilities"), True)

    # ---- insurance ------------------------------------------------------- #
    premiums = g("premiumsEarned")
    df["loss_ratio"] = _safe_div(g("claimsIncurred"), premiums, True)
    # expense ratio proxy: underwriting/opex (SG&A + DAC amortization) over premiums
    underwriting_exp = g("sellingGeneralAdmin").fillna(0) + g("dacAmortization").fillna(0)
    df["expense_ratio"] = _safe_div(underwriting_exp, premiums, True)
    df["combined_ratio"] = df["loss_ratio"] + df["expense_ratio"]        # <1 = underwriting profit
    # reliance on investment "float" vs underwriting: net investment income / premiums earned
    df["investment_income_ratio"] = _safe_div(g("netInvestmentIncome"), premiums, True)

    # ---- reits ----------------------------------------------------------- #
    # FFO ~= net income + real-estate depreciation - gains on property sales (NAREIT)
    ffo = ni + depamort.fillna(0) - g("gainOnDispositions").fillna(0)
    re_gate = g("realEstateNet").notna() | g("rentalIncome").notna()
    df["ffo_margin"] = _safe_div(ffo, revenue, True).where(re_gate)
    df["ffo_payout"] = _safe_div(g("dividendsPaid"), ffo, True).where(re_gate)
    df["rental_margin"] = _safe_div(g("rentalIncome"), revenue, True)
    # AFFO = FFO - maintenance capex (cash actually available to distribute)
    df["affo_margin"] = _safe_div(ffo - capex.fillna(0), revenue, True).where(re_gate)
    # leverage on a REIT-appropriate cash-earnings base: EBITDAre = operatingIncome + D&A
    ebitdare = oper_income.fillna(0) + depamort.fillna(0)
    re_net_debt = total_debt - cash.fillna(0)
    df["net_debt_to_ebitdare"] = _safe_div(re_net_debt, ebitdare, True).where(re_gate)

    # ---- energy ---------------------------------------------------------- #
    energy_gate = g("oilGasPropertyNet").notna()
    df["exploration_intensity"] = _safe_div(g("explorationExpense"), capex, True)
    ddna = g("depletionDDA").where(g("depletionDDA").notna(), depamort)
    df["ddna_intensity"] = _safe_div(ddna, revenue, True).where(energy_gate)
    # EBITDAX adds back exploration expense so Successful-Efforts and Full-Cost filers
    # are comparable; expressed as a margin so it is peer-rankable.
    ebitdax = oper_income.fillna(0) + depamort.fillna(0) + g("explorationExpense").fillna(0)
    df["ebitdax_margin"] = _safe_div(ebitdax, revenue, True).where(energy_gate)
    # capitalized-property vs cash generation: high = reserves carried at a value the
    # current cash flow cannot support -> impairment / overvaluation risk (short).
    df["property_overvaluation_cushion"] = _safe_div(
        g("oilGasPropertyNet"), ocf * 4.0, True).where(energy_gate)

    # ---- software / tech ------------------------------------------------- #
    df["deferred_rev_intensity"] = _safe_div(g("deferredRevenue"), revenue, True)
    df["rpo_coverage"] = _safe_div(g("remainingPerformanceObligation"), revenue, True)
    df["sbc_intensity"] = _safe_div(g("stockBasedComp"), revenue, True)

    # ---- utilities ------------------------------------------------------- #
    reg_assets = g("regulatoryAssets")
    df["regulatory_asset_ratio"] = _safe_div(reg_assets, assets, True)
    # rate-base growth proxy: capex over the CLEAN asset base (ex regulatory assets &
    # goodwill). A regulated utility only grows guaranteed earnings by expanding real
    # infrastructure, so a high ratio is a structural long. Gated to utilities (they
    # are the filers that report regulatory assets).
    clean_assets = assets - reg_assets.fillna(0) - g("goodwill").fillna(0)
    df["capex_to_rate_base"] = _safe_div(capex, clean_assets, True).where(reg_assets.notna())

    # ---- pharma / biotech ------------------------------------------------ #
    rd = g("researchAndDevelopment")
    # Patent-cliff vulnerability: acquired-drug amortization vs operating cash flow.
    # Rising = the current (bought) patents are expiring faster than cash is generated.
    df["patent_cliff"] = _safe_div(g("amortizationIntangibles"), ocf, True)
    # Capitalized-R&D adjusted ROIC: undo GAAP's immediate R&D expensing (treat R&D as
    # a 5-year intangible) so organic innovators are comparable to serial acquirers.
    rd_asset, rd_amort = _capitalized_rd(df, rd, yoy)
    adj_oper_income = oper_income.fillna(0) + rd.fillna(0) - rd_amort.fillna(0)
    adj_capital = (g("stockholdersEquity").fillna(0) + total_debt
                   + rd_asset.fillna(0) - cash.fillna(0))
    df["rd_capitalized_roic"] = _safe_div(adj_oper_income, adj_capital, True).where(rd.notna())

    return df


def build_sector_feature_panel(
    fundamentals: pd.DataFrame | None,
    peer_dict: dict,
    trading_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Long-format sector-KPI feature panel (`f_<kpi>_vs_peers`, `f_<kpi>_xs`).

    Computes the row-level KPIs, forward-fills each point-in-time from its
    `as_of`, and peer-relativizes — identical treatment to the fundamental /
    management panels. Empty if fundamentals are unavailable."""
    if (fundamentals is None or fundamentals.empty
            or "as_of" not in fundamentals.columns):
        return pd.DataFrame(columns=["date", "ticker"])

    kdf = compute_sector_kpis(fundamentals)
    fields: dict[str, pd.DataFrame] = {}
    for name in SECTOR_KPI_COLS:
        if name not in kdf.columns:
            continue
        daily = fundamentals_to_daily(kdf, name, trading_index)
        if not daily.empty and daily.notna().any().any():
            fields[name] = daily
    if not fields:
        return pd.DataFrame(columns=["date", "ticker"])
    return build_peer_relative_panel(fields, peer_dict)
