"""
sector_features.py  (src/data_aggregate/utils/sector_features.py)
------------------------------------------------------------------
Derived, sector-specific fundamental KPIs computed from `fundamentals_history`. Two layers:

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

⚠ THE GICS COLUMNS MUST BE ON THE FRAME. `row_gate` fails CLOSED, so a caller that hands
this a bare `fundamentals_history` gets every sector KPI silently gated off — which is what
happened for the whole life of this module until `step_cube_fundamentals` started attaching
them (`gics.attach_gics_columns`). The reference data was in `sp500_tickers` the entire time.

Every KPI is point-in-time (computed from TTM flows / period-end levels keyed on
the filing date `as_of`) and unit-free (a ratio), so it is comparable across
names and safe to neutralize within an industry group for a L/S book.

NOT here (owned by fundamental_features.py, which emits the same feature NAMES):
  interest_coverage, net_debt_to_ebitda, gross_profitability, cash_conversion_cycle,
  sbc_intensity, and the working-capital day counts `dso` / `dio` / `dpo`. Both panels
  used to build them with DIFFERENT formulas and the cube merges on ['date','ticker']
  only, so pandas silently produced 20 `_x`/`_y` columns whose meaning depended on merge
  order. One owner each: the general-purpose ratios live in the fundamental panel, this
  module keeps only KPIs that are genuinely sector-specific. `roic` moved the same way —
  it duplicated `fundamental_features.roic_incl_intangibles` at r = 0.973, the same
  NOPAT/invested-capital expression written twice. So did `net_debt_to_ebitdare` and
  `implied_cap_rate`: on the Sharadar substrate both reduce to a sector-masked copy of a
  general-purpose ratio (r = 0.9998 and 1.0000). A GATE IS NOT A SECOND FEATURE — the model
  already gets sector membership from the GICS categorical.

WHAT SHARADAR CANNOT FEED. The bank, insurance and several single-tag KPIs this module
used to define are GONE rather than dormant: `fundamentals_history` is Sharadar-first and
SF1 carries no `loans`, `deposits`Domestic, `noninterestExpense`, `claimsIncurred`,
`premiumsWritten`, `provisionForCreditLosses`, `tier1CapitalRatio`, `nonaccrualLoans`,
`netChargeOffs`, `regulatoryAssets`, `oilGasPropertyNet`, `explorationExpense` or
`amortizationIntangibles`. Each was measured emitting ZERO values against the live table.
They come back with the SEC extraction path, not before; keeping them as definitions made
the module look like it covered banks and insurers when it covered neither.

KPIs (grouped):
  universal      effective_tax_rate, accruals_ratio, asset_turnover (avg assets),
                 capex_intensity, capex_to_dep, payout_ratio, buyback_intensity,
                 earnings_quality, reinvestment_rate, sustainable_growth_rate,
                 fixed_cost_coverage_margin, gmroi
  financials     aoci_to_equity, book_value_growth
  banks          bank_roa
  reits          ffo_margin, ffo_payout, affo_margin, affo_dividend_coverage
  energy         ddna_intensity, ebitda_margin
  software/tech  deferred_rev_intensity
  utilities      capex_to_rate_base
  R&D-intensive  rd_capitalized_roic (5y-capitalized R&D) -- gated on R&D being REPORTED,
                 not on GICS: capitalizing R&D is meaningful for any research-intensive
                 filer, not only biotech
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_aggregate.utils.common.pit import fundamentals_to_daily, infer_yoy_periods
from src.data_aggregate.utils.common.frames import safe_div
from src.data_aggregate.utils.common.panel import build_peer_relative_panel
from src.data_aggregate.utils.common.sector_gates import row_gate
from src.data_aggregate.utils.common import capital

# KPI columns produced by compute_sector_kpis (the panel builder iterates these).
SECTOR_KPI_COLS: list[str] = [
    # universal
    "effective_tax_rate", "accruals_ratio", "asset_turnover",
    "capex_intensity", "capex_to_dep", "payout_ratio", "buyback_intensity",
    "earnings_quality", "reinvestment_rate", "sustainable_growth_rate",
    "fixed_cost_coverage_margin", "gmroi",
    # financials
    "aoci_to_equity", "book_value_growth",
    # banks
    "bank_roa",
    # reits
    "ffo_margin", "ffo_payout", "affo_margin", "affo_dividend_coverage",
    # energy
    "ddna_intensity", "ebitda_margin",
    # software / tech
    "deferred_rev_intensity",
    # utilities
    "capex_to_rate_base",
    # pharma / biotech
    "rd_capitalized_roic",
]


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """Numeric column or an all-NaN series when the tag was never reported.

    ⚠ SILENT BY DESIGN AND THAT IS A HAZARD. A missing column becomes all-NaN, and an
    all-NaN leg that is then `.fillna(0)`'d keeps its parent formula ALIVE while quietly
    changing its meaning — `payout_ratio` was a dividend payout ratio wearing a
    total-shareholder-payout name for exactly this reason. Every `.fillna(0)` below is on a
    leg confirmed to exist in the live table; adding a new one means checking the same."""
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(np.nan, index=df.index)


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
    # ONE definition of debt / invested capital (src/data_aggregate/utils/common/capital.py):
    # borrowings + capitalized leases, with no commercial-paper double count.
    total_debt = capital.total_debt(g)

    # GICS scopes (business-model, not tagging-based) for the sector KPI families
    bank_gate = row_gate(df, "bank")
    fin_gate = row_gate(df, "financials")
    re_gate = row_gate(df, "reit")
    energy_gate = row_gate(df, "energy")
    util_gate = row_gate(df, "utilities")
    # NO pharma gate: `patent_cliff` was the only GICS-scoped pharma KPI and it needed
    # `amortizationIntangibles`, which SF1 does not carry. `rd_capitalized_roic` is gated on
    # R&D being REPORTED, not on GICS -- deliberately, since capitalizing R&D is meaningful
    # for any research-intensive filer (tech and industrials as much as biotech).

    # ---- universal ------------------------------------------------------- #
    df["effective_tax_rate"] = safe_div(g("incomeTaxExpense"), g("pretaxIncome"), True)

    # cash-flow accruals (Sloan): (net income - operating cash flow) / assets
    df["accruals_ratio"] = safe_div(ni - ocf, assets, True)
    # asset turnover on AVERAGE total assets (mean of current & 1y-prior; falls back
    # to period-end when no prior year is available).
    prior_assets = _yearly_lag(df, assets, 1, yoy)
    avg_assets = ((assets + prior_assets) / 2.0).where(prior_assets.notna(), assets)
    df["asset_turnover"] = safe_div(revenue, avg_assets, True)
    df["capex_intensity"] = safe_div(capex, revenue, True)
    df["capex_to_dep"] = safe_div(capex, depamort, True)

    # TOTAL shareholder payout / net income. Both legs are positive magnitudes: the
    # `dividendsPaid` outflow is stored positive (the field map flips Sharadar's negative,
    # matching `capex`), and `share_repurchases` floors net issuance at 0 -- see capital.py,
    # the source column is NET ISSUANCE and reads negative for a repurchaser.
    buybacks = capital.share_repurchases(g)
    if buybacks is None:
        buybacks = pd.Series(np.nan, index=df.index)
    df["payout_ratio"] = safe_div(g("dividendsPaid").fillna(0) + buybacks.fillna(0), ni, True)
    df["buyback_intensity"] = safe_div(buybacks, revenue, True)

    # ---- capital efficiency & quality (value-creation core) -------------- #
    # NOPAT = operatingIncome x (1 - effective tax); tax clipped to [0,50%] and
    # defaulted to 21% when unreported so it is defined for the whole universe.
    tax = df["effective_tax_rate"].clip(lower=0.0, upper=0.5).fillna(0.21)
    nopat = oper_income * (1.0 - tax)
    # earnings quality: operating cash flow backing reported profit (<0.8 = accrual risk)
    df["earnings_quality"] = safe_div(ocf, ni, True)
    # reinvestment rate: net cash ploughed back (capex - D&A + ΔNWC) per $ of NOPAT
    nwc_now = g("currentAssets") - g("currentLiabilities")
    d_nwc = nwc_now - _yearly_lag(df, nwc_now, 1, yoy)
    df["reinvestment_rate"] = safe_div(capex.fillna(0.0) - depamort.fillna(0.0) + d_nwc, nopat, True)
    # sustainable growth = ROE x retention (max organic growth w/o new equity/leverage).
    # ⚠ THE `clip(0, 1)` MAKES THE `dividendsPaid` SIGN LOAD-BEARING. While the column was
    # stored outflow-NEGATIVE, every payer's payout clipped to exactly 0, retention to 1, and
    # SGR collapsed to `returnOnEquity` -- measured at Pearson r = 1.0000 against it. The
    # clip is still right (a payout above 100% of earnings is not a negative retention), but
    # it only reads as a payout ratio because the outflow is positive.
    roe = g("returnOnEquity")
    div_payout = safe_div(g("dividendsPaid"), ni, True).clip(lower=0.0, upper=1.0)
    df["sustainable_growth_rate"] = (roe * (1.0 - div_payout.fillna(0.0))).where(roe.notna())
    # fixed-cost coverage margin = (gross profit - EBITDA) / revenue = overhead intensity
    df["fixed_cost_coverage_margin"] = safe_div(g("grossProfit") - ebitda, revenue, True)
    # GMROI (retail): gross profit per $ of average inventory investment
    inv = g("inventory")
    prior_inv = _yearly_lag(df, inv, 1, yoy)
    avg_inv = ((inv + prior_inv) / 2.0).where(prior_inv.notna(), inv)
    df["gmroi"] = safe_div(g("grossProfit"), avg_inv, True).where(inv.notna())

    # ---- banks ----------------------------------------------------------- #
    # The only bank KPI Sharadar can feed: everything else needed `loans`, `deposits`Domestic,
    # `noninterestExpense`, `provisionForCreditLosses` or `netInterestIncome`, none of which
    # SF1 delivers. ROA is the right survivor -- for a bank, assets ARE the earning base.
    df["bank_roa"] = safe_div(ni, assets, True).where(bank_gate)

    # ---- reits ----------------------------------------------------------- #
    # NAREIT FFO = net income + real-estate D&A - gains/losses on sales of real estate
    # + impairment write-downs.
    # ⚠ ONLY THE D&A LEG IS AVAILABLE. `gainOnDispositions` and `realEstateImpairment` are
    # not in SF1, so this is net income + total D&A: a REIT that sold a property at a gain
    # reads high here, which is exactly what those two adjustments exist to remove. Directionally
    # right and comparable across REITs, but not a strict NAREIT FFO.
    ffo = ni + depamort.fillna(0)
    df["ffo_margin"] = safe_div(ffo, revenue, True).where(re_gate)
    df["ffo_payout"] = safe_div(g("dividendsPaid"), ffo, True).where(re_gate)
    # AFFO = FFO - recurring capex. The straight-line-rent and above/below-market lease
    # amortization legs the supplementals adjust for are likewise not in SF1.
    affo = ffo - capex.fillna(0)
    df["affo_margin"] = safe_div(affo, revenue, True).where(re_gate)
    # `net_debt_to_ebitdare` USED TO BE HERE AND IS DELETED. EBITDAre was
    # `operatingIncome + D&A` once the real-estate impairment leg proved absent from SF1 --
    # which is the field map's own DEFINITION of `ebitda`, so the KPI was
    # `fundamental_features.net_debt_to_ebitda` masked to REITs, measured at r = 0.9998
    # against it. One owner per name, and that owner is the general-purpose ratio.
    # dividend safety: does the cash left after maintenance capex cover the distribution?
    df["affo_dividend_coverage"] = safe_div(affo, g("dividendsPaid"), True).where(re_gate)

    # ---- energy ---------------------------------------------------------- #
    # Depletion/DD&A intensity. `depletionDDA` is not in SF1, so total D&A stands in: for an
    # E&P that IS overwhelmingly depletion, which is why the fallback was already the design.
    df["ddna_intensity"] = safe_div(depamort, revenue, True).where(energy_gate)
    # EBITDA margin on the energy gate. NOT named `ebitdax_margin` any more: EBITDAX adds
    # back exploration expense so Successful-Efforts and Full-Cost filers are comparable, and
    # `explorationExpense` is not in SF1, so nothing is added back. The name promised a
    # normalisation the data cannot perform -- correct for services and refiners, understated
    # for a Successful-Efforts E&P by exactly the exploration it expensed.
    ebitda_energy = oper_income.fillna(0) + depamort.fillna(0)
    df["ebitda_margin"] = safe_div(ebitda_energy, revenue, True).where(energy_gate)

    # ---- software / tech ------------------------------------------------- #
    # NOT sector-gated on purpose: deferred revenue is meaningful for ANY subscription or
    # contract-backed model (industrials services, health-care IT), and it is only reported
    # by filers that have it.
    df["deferred_rev_intensity"] = safe_div(g("deferredRevenue"), revenue, True)

    # ---- utilities ------------------------------------------------------- #
    # Rate-base growth proxy: capex over the asset base. A regulated utility only grows
    # guaranteed earnings by expanding real infrastructure, so a high ratio is a structural
    # long. ⚠ `regulatoryAssets` and a standalone `goodwill` are not in SF1, so the base is
    # NOT cleaned of either -- this is capex/assets on the utility gate, and it differs from
    # the universal `capex_intensity` only in its denominator (assets, not revenue).
    df["capex_to_rate_base"] = safe_div(capex, assets, True).where(util_gate)

    # ---- R&D-intensive (availability-gated, NOT GICS-gated) -------------- #
    rd = g("researchAndDevelopment")
    # Capitalized-R&D adjusted ROIC: undo GAAP's immediate R&D expensing (treat R&D as
    # a 5-year intangible) so organic innovators are comparable to serial acquirers.
    rd_asset, rd_amort = _capitalized_rd(df, rd, yoy)
    adj_oper_income = oper_income.fillna(0) + rd.fillna(0) - rd_amort.fillna(0)
    adj_capital = (g("stockholdersEquity").fillna(0) + total_debt
                   + rd_asset.fillna(0) - cash.fillna(0))
    df["rd_capitalized_roic"] = safe_div(adj_oper_income, adj_capital, True).where(rd.notna())

    # ---- financial-sector growth & capital ------------------------------- #
    equity = g("stockholdersEquity")
    prior_equity = _yearly_lag(df, equity, 1, yoy)
    df["book_value_growth"] = safe_div(equity - prior_equity, prior_equity, True).where(fin_gate)

    # AOCI is mostly the AFS mark-to-market; a large NEGATIVE AOCI = unrealized securities
    # losses eroding tangible capital (signed: negative = losses). The 2023 SVB signal, minus
    # the held-to-maturity leg, which needs footnote fair values SF1 does not carry.
    df["aoci_to_equity"] = safe_div(
        g("accumulatedOtherComprehensiveIncome"), equity, True).where(fin_gate)

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
