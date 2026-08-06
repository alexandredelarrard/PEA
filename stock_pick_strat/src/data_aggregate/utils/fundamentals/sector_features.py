"""
sector_features.py  (src/data_aggregate/utils/sector_features.py)
------------------------------------------------------------------
Derived, sector-specific fundamental KPIs computed from the expanded SEC
fundamentals history (the raw line items added in fetch_fundamentals). Two
layers:

  * compute_sector_kpis(df)          -> pure, row-level ratios (one value per
                                        filing). Scoped by GICS (`sector_gates.py`,
                                        driven by SECTOR_KPI_SCOPE) AND by input
                                        availability: a KPI is NaN unless the name is
                                        in the sector the metric is defined for AND
                                        its inputs were reported. Availability alone
                                        was not a sector gate — `InterestIncomeExpense-
                                        Net` is tagged by 59 non-Financials (bank KPIs
                                        on industrials) while only 3 of 21 Energy names
                                        tag `OilAndGasProperty*` (EBITDAX empty for
                                        86% of the sector).
  * build_sector_feature_panel(...)  -> turns the chosen KPIs into a daily,
                                        peer-relative feature panel (same
                                        machinery as the fundamental panel), so
                                        they drop straight into the cube.

Every KPI is point-in-time (computed from TTM flows / period-end levels keyed on
the filing date `as_of`) and unit-free (a ratio), so it is comparable across
names and safe to neutralize within an industry group for a L/S book.

NOT here (owned by fundamental_features.py, which emits the same feature NAMES):
  interest_coverage, net_debt_to_ebitda, gross_profitability, cash_conversion_cycle,
  sbc_intensity. Both panels used to build them with DIFFERENT formulas and the cube
  merges on ['date','ticker'] only, so pandas silently produced 20 `_x`/`_y` columns
  (`f_interest_coverage_vs_peers_x` / `_y`, ...) whose meaning depended on merge order.
  One owner each: the general-purpose ratios live in the fundamental panel (which also
  carries the richer `net_debt_incl_offbs_to_ebitda`), this module keeps only KPIs that
  are genuinely sector-specific.

KPIs (grouped):
  universal      effective_tax_rate, accruals_ratio, asset_turnover (avg assets),
                 capex_intensity, capex_to_dep, payout_ratio, buyback_intensity,
                 days_sales/inventory/payable_outstanding, roic, earnings_quality,
                 reinvestment_rate, sustainable_growth_rate,
                 fixed_cost_coverage_margin, gmroi, bad_debt_intensity
  banks          net_interest_margin, efficiency_ratio, provision_rate,
                 loan_to_deposit, bank_roa, bank_operating_margin,
                 reserve_coverage_velocity, tier1_capital_ratio, deposit_stickiness
  insurance      loss_ratio, expense_ratio, combined_ratio, investment_income_ratio
  reits          ffo_margin, ffo_payout, rental_margin, affo_margin,
                 net_debt_to_ebitdare
  energy         exploration_intensity, ddna_intensity, ebitdax_margin,
                 property_overvaluation_cushion
  software/tech  deferred_rev_intensity, rpo_coverage
  utilities      regulatory_asset_ratio, capex_to_rate_base
  pharma/biotech patent_cliff, rd_capitalized_roic (5y-capitalized R&D)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_aggregate.utils.common.pit import fundamentals_to_daily, infer_yoy_periods
from src.data_aggregate.utils.common.panel import build_peer_relative_panel
from src.data_aggregate.utils.common.sector_gates import row_gate
from src.data_aggregate.utils.common import capital

# KPI columns produced by compute_sector_kpis (the panel builder iterates these).
# interest_coverage / net_debt_to_ebitda / gross_profitability / cash_conversion_cycle /
# sbc_intensity are deliberately ABSENT -- fundamental_features.py owns those names (see
# the module docstring); listing them here collided in the cube merge.
SECTOR_KPI_COLS: list[str] = [
    # universal
    "effective_tax_rate", "accruals_ratio",
    "asset_turnover", "capex_intensity", "capex_to_dep",
    "payout_ratio", "buyback_intensity", "days_sales_outstanding",
    "days_inventory_outstanding", "days_payable_outstanding",
    "roic", "earnings_quality", "reinvestment_rate", "sustainable_growth_rate",
    "fixed_cost_coverage_margin", "gmroi", "bad_debt_intensity",
    # banks
    "net_interest_margin", "efficiency_ratio", "provision_rate", "loan_to_deposit", "bank_roa",
    "bank_operating_margin", "reserve_coverage_velocity", "tier1_capital_ratio", "deposit_stickiness",
    "nii_growth", "loan_growth",                                                         # A7
    "aoci_to_equity", "htm_unrealized_loss_ratio", "npl_ratio", "net_charge_off_rate",  # B1 + B3
    # insurance
    "loss_ratio", "expense_ratio", "combined_ratio", "investment_income_ratio",
    "book_value_growth", "premium_growth", "float_growth",                              # A6
    # reits
    "ffo_margin", "ffo_payout", "rental_margin", "affo_margin", "net_debt_to_ebitdare",
    "affo_dividend_coverage",                                                            # A8
    # energy
    "exploration_intensity", "ddna_intensity", "ebitdax_margin", "property_overvaluation_cushion",
    # software / tech
    "deferred_rev_intensity", "rpo_coverage",
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


def _safe_div(num: pd.Series, den: pd.Series | None, den_positive: bool = False) -> pd.Series:
    """Elementwise num/den, NaN where den is 0/NaN (or <=0 if den_positive). A `None`
    denominator (a `capital.*` helper with none of its inputs present) yields all-NaN
    rather than raising."""
    if den is None:
        return pd.Series(np.nan, index=num.index)
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

    Pure and side-effect-free. Doubly gated: by GICS (`sector_gates.row_gate`, so a
    metric only exists for the sector whose business model it describes) and by input
    availability (NaN unless the filer reported every input). GICS is the authority for
    "is this meaningful here"; tagging habits decide only "do we have the numbers"."""
    if fundamentals is None or fundamentals.empty:
        return fundamentals if fundamentals is not None else pd.DataFrame()

    df = fundamentals.copy()
    g = lambda n: _col(df, n)  # noqa: E731
    yoy = infer_yoy_periods(df)                     # filings per year (4 quarterly, 1 annual)

    revenue = g("totalRevenue")
    # ASC-842-adoption-free asset base (shared resolver: precomputed column, else derived,
    # else plain total assets -- so an older history vintage still works).
    assets = capital.assets_ex_lease(g)
    ebitda = g("ebitda")
    ni = g("netIncome")
    ocf = g("operatingCashFlow")
    cogs = g("costOfRevenue")
    oper_income = g("operatingIncome")
    depamort = g("depAmort")
    capex = g("capex")
    cash = g("cash")
    # ONE definition of debt / invested capital (src/data_aggregate/utils/capital.py):
    # borrowings + capitalized leases, with no commercial-paper double count.
    total_debt = capital.total_debt(g)

    # GICS scopes (business-model, not tagging-based) for the sector KPI families
    bank_gate = row_gate(df, "bank")
    ins_gate = row_gate(df, "insurance")
    fin_gate = row_gate(df, "financials")
    re_gate = row_gate(df, "reit")
    energy_gate = row_gate(df, "energy")
    util_gate = row_gate(df, "utilities")
    pharma_gate = row_gate(df, "pharma")

    # ---- universal ------------------------------------------------------- #
    df["effective_tax_rate"] = _safe_div(g("incomeTaxExpense"), g("pretaxIncome"), True)

    # cash-flow accruals (Sloan): (net income - operating cash flow) / assets
    df["accruals_ratio"] = _safe_div(ni - ocf, assets, True)
    # TRADE bad-debt expense / revenue (no sector gate: any seller can over-book).
    # Rising = sales are being recognised that the firm cannot collect. Split out of
    # the bank `provisionForCreditLosses` pool it used to contaminate.
    df["bad_debt_intensity"] = _safe_div(g("provisionDoubtfulAccounts"), revenue, True)
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
    # (their sum, cash_conversion_cycle, is emitted by fundamental_features.py)

    # ---- capital efficiency & quality (value-creation core) -------------- #
    # NOPAT = operatingIncome x (1 - effective tax); tax clipped to [0,50%] and
    # defaulted to 21% when unreported so ROIC is defined for the whole universe.
    tax = df["effective_tax_rate"].clip(lower=0.0, upper=0.5).fillna(0.21)
    nopat = oper_income * (1.0 - tax)
    df["roic"] = _safe_div(nopat, capital.invested_capital(g), True)   # value created if > WACC
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
    df["net_interest_margin"] = _safe_div(nii, assets, True).where(bank_gate)  # NII / total assets (proxy)
    bank_revenue = nii.fillna(0) + noninterest_income.fillna(0)           # NII + noninterest income
    df["efficiency_ratio"] = _safe_div(g("noninterestExpense"), bank_revenue, True).where(bank_gate)
    df["provision_rate"] = _safe_div(g("provisionForCreditLosses"), g("loans"), True).where(bank_gate)
    df["loan_to_deposit"] = _safe_div(g("loans"), g("deposits"), True).where(bank_gate)
    df["bank_roa"] = _safe_div(ni, assets, True).where(bank_gate)
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
    df["reserve_coverage_velocity"] = _safe_div(d_prov, g("allowanceCreditLosses"), True).where(bank_gate)
    # capital adequacy (already a ratio) and deposit-franchise stickiness
    t1 = g("tier1CapitalRatio")
    df["tier1_capital_ratio"] = t1.where((t1 > 0) & bank_gate)
    df["deposit_stickiness"] = _safe_div(g("depositsDomestic"), g("totalLiabilities"), True).where(bank_gate)

    # ---- insurance ------------------------------------------------------- #
    premiums = g("premiumsEarned")
    df["loss_ratio"] = _safe_div(g("claimsIncurred"), premiums, True).where(ins_gate)
    # expense ratio proxy: underwriting/opex (SG&A + DAC amortization) over premiums
    underwriting_exp = g("sellingGeneralAdmin").fillna(0) + g("dacAmortization").fillna(0)
    df["expense_ratio"] = _safe_div(underwriting_exp, premiums, True).where(ins_gate)
    df["combined_ratio"] = df["loss_ratio"] + df["expense_ratio"]        # <1 = underwriting profit
    # reliance on investment "float" vs underwriting: net investment income / premiums earned
    df["investment_income_ratio"] = _safe_div(g("netInvestmentIncome"), premiums, True).where(ins_gate)

    # ---- reits ----------------------------------------------------------- #
    # NAREIT FFO = net income + real-estate D&A - gains/losses on sales of real estate
    # + IMPAIRMENT write-downs of real estate (the 2018 white paper excludes all three
    # non-cash / non-recurring property items, not just the first two).
    re_impair = g("realEstateImpairment")
    ffo = (ni + depamort.fillna(0) - g("gainOnDispositions").fillna(0)
           + re_impair.fillna(0))
    df["ffo_margin"] = _safe_div(ffo, revenue, True).where(re_gate)
    df["ffo_payout"] = _safe_div(g("dividendsPaid"), ffo, True).where(re_gate)
    df["rental_margin"] = _safe_div(g("rentalIncome"), revenue, True).where(re_gate)
    # AFFO = FFO - recurring capex - NON-CASH straight-line rent - above/below-market
    # lease amortization (NAREIT declines to standardize AFFO, but these are the two
    # adjustments every REIT supplemental makes; both are sparsely tagged, so this is a
    # no-op where undisclosed rather than a guess).
    affo = (ffo - capex.fillna(0) - g("straightLineRent").fillna(0)
            - g("aboveBelowMarketLeaseAmort").fillna(0))
    df["affo_margin"] = _safe_div(affo, revenue, True).where(re_gate)
    # leverage on a REIT-appropriate cash-earnings base: EBITDAre = operatingIncome + D&A
    # + real-estate impairment (NAREIT EBITDAre adds back the same property write-downs).
    ebitdare = oper_income.fillna(0) + depamort.fillna(0) + re_impair.fillna(0)
    # NAREIT EBITDAre also excludes gains on property sales, but that is NOT subtracted here:
    # this starts from OPERATING income, and for the many REITs with no tagged
    # `OperatingIncomeLoss` the extractor derives it bottom-up as pre-tax income + interest,
    # which already contains the gain -- while for REITs that do tag it the gain usually sits
    # below the operating line. Subtracting unconditionally would double-remove it for the
    # first group. FFO (built from net income) removes the gain correctly.
    df["net_debt_to_ebitdare"] = _safe_div(capital.net_debt(g), ebitdare, True).where(re_gate)

    # ---- energy ---------------------------------------------------------- #
    df["exploration_intensity"] = _safe_div(g("explorationExpense"), capex, True).where(energy_gate)
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
    # NOT sector-gated on purpose: deferred revenue / RPO are meaningful for ANY
    # subscription or contract-backed model (industrials services, health-care IT),
    # and they are only reported by filers that have them.
    df["deferred_rev_intensity"] = _safe_div(g("deferredRevenue"), revenue, True)
    df["rpo_coverage"] = _safe_div(g("remainingPerformanceObligation"), revenue, True)

    # ---- utilities ------------------------------------------------------- #
    reg_assets = g("regulatoryAssets")
    df["regulatory_asset_ratio"] = _safe_div(reg_assets, assets, True).where(util_gate)
    # rate-base growth proxy: capex over the CLEAN asset base (ex regulatory assets &
    # goodwill). A regulated utility only grows guaranteed earnings by expanding real
    # infrastructure, so a high ratio is a structural long. Gated to utilities (they
    # are the filers that report regulatory assets).
    clean_assets = assets - reg_assets.fillna(0) - g("goodwill").fillna(0)
    df["capex_to_rate_base"] = _safe_div(capex, clean_assets, True).where(util_gate)

    # ---- pharma / biotech ------------------------------------------------ #
    rd = g("researchAndDevelopment")
    # Patent-cliff vulnerability: acquired-drug amortization vs operating cash flow.
    # Rising = the current (bought) patents are expiring faster than cash is generated.
    df["patent_cliff"] = _safe_div(g("amortizationIntangibles"), ocf, True).where(pharma_gate)
    # Capitalized-R&D adjusted ROIC: undo GAAP's immediate R&D expensing (treat R&D as
    # a 5-year intangible) so organic innovators are comparable to serial acquirers.
    rd_asset, rd_amort = _capitalized_rd(df, rd, yoy)
    adj_oper_income = oper_income.fillna(0) + rd.fillna(0) - rd_amort.fillna(0)
    adj_capital = (g("stockholdersEquity").fillna(0) + total_debt
                   + rd_asset.fillna(0) - cash.fillna(0))
    df["rd_capitalized_roic"] = _safe_div(adj_oper_income, adj_capital, True).where(rd.notna())

    # ---- financial-sector growth & capital (A6 insurance, A7 banks) ------- #
    def _yoy_growth(s: pd.Series) -> pd.Series:
        prior = _yearly_lag(df, s, 1, yoy)
        return _safe_div(s - prior, prior, True)

    df["nii_growth"] = _yoy_growth(nii).where(bank_gate)                       # A7
    df["loan_growth"] = _yoy_growth(g("loans")).where(bank_gate)               # A7
    equity = g("stockholdersEquity")
    df["book_value_growth"] = _yoy_growth(equity).where(fin_gate)              # A6 (compounding)
    df["premium_growth"] = _yoy_growth(g("premiumsWritten")).where(ins_gate)   # A6
    df["float_growth"] = _yoy_growth(g("insuranceReserves")).where(ins_gate)   # A6 (investable float)

    # ---- A8 REIT: AFFO dividend coverage (dividend safety) --------------- #
    df["affo_dividend_coverage"] = _safe_div(affo, g("dividendsPaid"), True).where(re_gate)

    # ---- B1 bank/insurer securities-mark drag (the 2023 SVB signal) ------ #
    # AOCI is mostly the AFS mark-to-market; a large NEGATIVE AOCI = unrealized
    # securities losses eroding tangible capital (signed: negative = losses).
    df["aoci_to_equity"] = _safe_div(g("accumulatedOCI"), equity, True).where(fin_gate)
    # HELD-TO-MATURITY unrealized loss = amortized cost - footnote fair value, the loss
    # hidden OFF the balance sheet (what sank SVB). Positive = unrecognized loss. The
    # DISCLOSED unrecognized holding loss is preferred where tagged: it needs only one
    # element, so it covers banks that tag just one of the two legs.
    htm_loss = g("htmUnrealizedLoss")
    htm_loss = htm_loss.where(htm_loss.notna(), g("htmSecurities") - g("htmSecuritiesFairValue"))
    df["htm_unrealized_loss_ratio"] = _safe_div(htm_loss, equity, True).where(fin_gate)

    # ---- B3 bank credit quality: non-performing loans + net charge-offs -- #
    df["npl_ratio"] = _safe_div(g("nonaccrualLoans"), g("loans"), True).where(bank_gate)
    df["net_charge_off_rate"] = _safe_div(g("netChargeOffs"), g("loans"), True).where(bank_gate)

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
