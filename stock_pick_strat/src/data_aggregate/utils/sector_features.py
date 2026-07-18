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
                                        machinery as management_features), so they
                                        drop straight into the cube.

Every KPI is point-in-time (computed from TTM flows / period-end levels keyed on
the filing date `as_of`) and unit-free (a ratio), so it is comparable across
names and safe to neutralize within an industry group for a L/S book.

KPIs (grouped):
  universal      effective_tax_rate, interest_coverage, net_debt_to_ebitda,
                 accruals_ratio, gross_profitability, asset_turnover,
                 capex_intensity, capex_to_dep, payout_ratio, buyback_intensity,
                 cash_conversion_cycle (dso/dio/dpo)
  banks          net_interest_margin, efficiency_ratio, provision_rate,
                 loan_to_deposit, bank_roa
  insurance      loss_ratio, expense_ratio, combined_ratio
  reits          ffo, ffo_margin, ffo_payout, rental_margin
  energy         exploration_intensity, ddna_intensity
  software/tech  deferred_rev_intensity, rpo_coverage, sbc_intensity
  utilities      regulatory_asset_ratio
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_aggregate.utils.factors import fundamentals_to_daily
from src.data_aggregate.utils.fundamental_features import build_peer_relative_panel

# KPI columns produced by compute_sector_kpis (the panel builder iterates these)
SECTOR_KPI_COLS: list[str] = [
    # universal
    "effective_tax_rate", "interest_coverage", "net_debt_to_ebitda", "accruals_ratio",
    "gross_profitability", "asset_turnover", "capex_intensity", "capex_to_dep",
    "payout_ratio", "buyback_intensity", "days_sales_outstanding",
    "days_inventory_outstanding", "days_payable_outstanding", "cash_conversion_cycle",
    # banks
    "net_interest_margin", "efficiency_ratio", "provision_rate", "loan_to_deposit", "bank_roa",
    # insurance
    "loss_ratio", "expense_ratio", "combined_ratio",
    # reits
    "ffo_margin", "ffo_payout", "rental_margin",
    # energy
    "exploration_intensity", "ddna_intensity",
    # software / tech
    "deferred_rev_intensity", "rpo_coverage", "sbc_intensity",
    # utilities
    "regulatory_asset_ratio",
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

    revenue = g("totalRevenue")
    assets = g("totalAssets")
    ebitda = g("ebitda")
    ni = g("netIncome")
    ocf = g("operatingCashFlow")
    cogs = g("costOfRevenue")

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
    df["asset_turnover"] = _safe_div(revenue, assets, True)
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

    # ---- banks ----------------------------------------------------------- #
    nii = g("netInterestIncome")
    df["net_interest_margin"] = _safe_div(nii, assets, True)              # NII / total assets (proxy)
    df["efficiency_ratio"] = _safe_div(g("noninterestExpense"),
                                       nii.fillna(0) + g("noninterestIncome").fillna(0), True)
    df["provision_rate"] = _safe_div(g("provisionForCreditLosses"), g("loans"), True)
    df["loan_to_deposit"] = _safe_div(g("loans"), g("deposits"), True)
    df["bank_roa"] = _safe_div(ni, assets, True).where(nii.notna())       # only for banks

    # ---- insurance ------------------------------------------------------- #
    premiums = g("premiumsEarned")
    df["loss_ratio"] = _safe_div(g("claimsIncurred"), premiums, True)
    # expense ratio proxy: underwriting/opex (SG&A + DAC amortization) over premiums
    underwriting_exp = g("sellingGeneralAdmin").fillna(0) + g("dacAmortization").fillna(0)
    df["expense_ratio"] = _safe_div(underwriting_exp, premiums, True)
    df["combined_ratio"] = df["loss_ratio"] + df["expense_ratio"]        # <1 = underwriting profit

    # ---- reits ----------------------------------------------------------- #
    # FFO ~= net income + real-estate depreciation - gains on property sales (NAREIT)
    ffo = ni + g("depAmort").fillna(0) - g("gainOnDispositions").fillna(0)
    re_gate = g("realEstateNet").notna() | g("rentalIncome").notna()
    df["ffo_margin"] = _safe_div(ffo, revenue, True).where(re_gate)
    df["ffo_payout"] = _safe_div(g("dividendsPaid"), ffo, True).where(re_gate)
    df["rental_margin"] = _safe_div(g("rentalIncome"), revenue, True)

    # ---- energy ---------------------------------------------------------- #
    df["exploration_intensity"] = _safe_div(g("explorationExpense"), g("capex"), True)
    ddna = g("depletionDDA").where(g("depletionDDA").notna(), g("depAmort"))
    df["ddna_intensity"] = _safe_div(ddna, revenue, True).where(g("oilGasPropertyNet").notna())

    # ---- software / tech ------------------------------------------------- #
    df["deferred_rev_intensity"] = _safe_div(g("deferredRevenue"), revenue, True)
    df["rpo_coverage"] = _safe_div(g("remainingPerformanceObligation"), revenue, True)
    df["sbc_intensity"] = _safe_div(g("stockBasedComp"), revenue, True)

    # ---- utilities ------------------------------------------------------- #
    df["regulatory_asset_ratio"] = _safe_div(g("regulatoryAssets"), assets, True)

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
