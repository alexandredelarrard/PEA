"""
fundamental_features.py  (src/data_aggregate/utils/fundamental_features.py)
---------------------------------------------------------------------------
The "firm vs its direct competitors" fundamental signals -- the differentiator
of this strategy. For every fundamental characteristic we express the stock
RELATIVE TO ITS PEER BASKET (from the peer dict), not in absolute terms:

    rel_i(t) = (X_i(t) - peer_weighted_mean_i(t)) / peer_weighted_std_i(t)

so "cheaper than its competitors", "growing faster than its competitors",
"diluting shareholders more than its competitors" become features, while a
market-wide value/growth LEVEL does not (that crowded style factor is already
stripped from the label). These peer-relative fundamentals are largely
orthogonal to the broad style factors, so they survive residualization and are
where real firm-specific edge lives.

CHARACTERISTICS
===============
Valuation (as YIELDS = fundamental / price, so cheap => high, and the metric is
bounded/rankable instead of exploding when the denominator -> 0). Market cap is
rebuilt DAILY as point-in-time sharesOutstanding * close, so valuation moves
every day with the price, not only when a new filing lands:
    earnings_yield  = netIncome        / marketCap      (E/P, inverse trailing PE)
    sales_yield     = totalRevenue     / marketCap      (S/P, inverse P/S)
    book_yield      = stockholdersEquity/ marketCap      (B/P, inverse P/B)
    fcf_yield       = freeCashflow     / marketCap      (FCF/P)
    ebitda_to_ev    = ebitda / TRUE EV (inverse EV/EBITDA), where the True
                      (fully-diluted) EV = diluted-shares*price + total debt +
                      lease liabilities + minority interest - cash - short-term
                      investments (only non-operating liquid assets are netted).
                      Real debt columns are used, with debtToEquity*equity as a
                      fallback for histories that lack them.
    fcf_to_ev       = freeCashflow / EV (Fully-Diluted FCF Yield) -- cash the whole
                      capital structure throws off vs its total price; the cross-
                      sector cash-valuation yield (and the energy "FCF EV yield").
    altman_z        = market-value Altman Z bankruptcy screen (WC/RE/EBIT/mcap/sales
                      over assets); pegy = P/E / (EPS growth% + dividend yield%).
    ffo_yield / implied_cap_rate (REITs), ebitdax_to_ev (energy) -- sector EV/price
                      multiples gated on the sector's balance-sheet signature.
    operating_leverage_elasticity = %ΔOperatingIncome / %ΔRevenue; nwc_elasticity =
                      %ΔNWC / %ΔRevenue; margin_expansion_delta = Δgross - ΔEBITDA
                      margin; diluted_shares_growth; ebit_interest_coverage.

Profitability / moat:
    grossMargins, operatingMargins, profitMargins, returnOnEquity  (raw ratios)
    fcf_margin      = freeCashflow / totalRevenue        (cash profitability)
    accruals        = (netIncome - freeCashflow)/totalRevenue  (earnings quality;
                      LOW/negative = earnings backed by cash = higher quality)

Growth (TTM, year-over-year):
    revenueGrowth, earningsGrowth  (YoY, from the fiscal series)
    fcf_growth                     (YoY free-cash-flow growth)
    gross_margin_chg               (YoY change in gross margin = margin expansion)

Latest-quarter momentum (discrete single quarter, what TTM smooths away):
    q_rev_growth      latest-quarter revenue YoY
    rev_growth_accel  change in that YoY vs the prior quarter (acceleration)
    q_earnings_growth latest-quarter net-income YoY
    q_margin_vs_ttm   latest-quarter profit margin minus the TTM margin (inflection)

Yearly-TTM momentum (current TTM vs TTM one year ago, less noisy than single quarter):
    y_rev_growth      TTM revenue YoY growth (computed fresh from totalRevenue)
    y_rev_growth_accel change in TTM revenue YoY (trend acceleration over a year)
    y_earnings_growth TTM net-income YoY growth
    y_margin_vs_ttm   YoY change in TTM profit margin (margin expansion/contraction)

Intrinsic value (two-stage DCF on TTM free cash flow, see intrinsic.py):
    intrinsic_yield   DCF equity value / market cap  ( >1 => below intrinsic )

Distress / solvency (debt-SERVICING ability, which debtToEquity ignores):
    net_debt_to_ebitda (total debt - cash) / EBITDA   (leverage; HIGH = worse)
    interest_coverage  EBITDA / interest expense       (HIGH = safer)
    current_ratio      current assets / current liabs  (near-term liquidity)
    cash_to_debt       cash / total debt               (liquidity cushion)

Marketing & sales efficiency:
    sga_intensity      SG&A / revenue                  (selling-cost discipline)
    sga_growth         YoY SG&A growth
    operating_leverage revenue growth - SG&A growth    (>0 = scaling profitably)

M&A footprint (organic vs inorganic growth; goodwill-impairment risk):
    acquisition_intensity acquisition spend / total assets
    goodwill_growth       YoY goodwill growth

Stock-based compensation ("employee shares given"; gross, unlike net dilution):
    sbc_intensity      stock-based comp / revenue
    sbc_to_ocf         stock-based comp / operating cash flow (cash-flow quality)

Valuation mean-reversion (self-history, emitted as `f_<yield>_vs_hist`):
    every valuation yield above ALSO gets a z-score versus the firm's OWN
    trailing history -> "cheap vs its own past" (e.g. PE below its 5y average),
    an axis orthogonal to the cross-sectional "cheap vs peers" signals.

Capital allocation / dilution ("stock given to employees" proxy):
    shares_growth   = YoY change in sharesOutstanding. Positive => issuing /
                      diluting (heavy SBC), negative => buying back.

R&D / intangibles moat (only when researchAndDevelopment was collected):
    rd_intensity    = researchAndDevelopment / totalRevenue

AVAILABILITY
============
Valuation needs `close`; growth/dilution/trend need the fiscal history. All are
point-in-time (values keyed on the SEC filing date via fundamentals_to_daily),
so there is no look-ahead. researchAndDevelopment must be present in the
history parquet (added to the SEC extractor); if absent, R&D features are
skipped automatically.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from src.context import Context
from src.data_aggregate.utils.fundamentals.earnings_features import ntm_ttm_eps
from src.data_aggregate.utils.common.pit import (
    daily_market_cap,
    fiscal_apply_to_daily,
    fiscal_change_to_daily,
    fundamentals_to_daily,
    infer_yoy_periods,
)
from src.data_aggregate.utils.fundamentals.intrinsic import intrinsic_value_daily
from src.data_aggregate.utils.common.frames import ratio, sanitize
from src.data_aggregate.utils.common.panel import build_peer_relative_panel
from src.data_aggregate.utils.common.xs import (
    HIST_MIN_PERIODS, HIST_WINDOW, self_history_z, winsorize_xs,
)
from src.data_aggregate.utils.common.sector_gates import family_tickers, mask_columns
from src.data_aggregate.utils.common import capital

_PENSION_FACTS_TABLE = "pension_facts"     # bulk Financial-Statement-Data-Sets pension facts (literal)
_NOTES_NUM_TABLE = "notes_num"             # footnote NUMERIC facts (10 tags; the panel uses 2)
_FACT_COLS = ["ticker", "tag", "ddate", "qtrs", "value", "filed"]   # the only cols the pension builders read


# Valuation yields that also get a self-history (mean-reversion) z-score, i.e.
# "cheap vs its OWN past" in addition to "cheap vs peers". High = cheaper than
# the firm's own norm -> classic valuation mean-reversion signal.
_MEAN_REVERSION_FIELDS = (
    "earnings_yield", "sales_yield", "book_yield",
    "fcf_yield", "ebitda_to_ev", "fcf_to_ev", "ffo_yield", "intrinsic_yield",
)
#: Re-exported from `common/xs.py`, which now owns the self-history z (it gained a second
#: consumer in the governance panel). Same values; the names stay so no call site here moves.
_HIST_WINDOW = HIST_WINDOW          # ~5 trading years of daily observations
_HIST_MIN_PERIODS = HIST_MIN_PERIODS  # require >= 1y of history before emitting a z-score

# Cross-sectional winsorization for the Z-SCORE features (peer-z + self-history z):
# clip each day's distribution to its [1%, 99%] percentiles so a few extreme names
# can't dominate the standardized value -> better generalization. The percentile-
# RANK features (`_xs`) are already outlier-proof and are left untouched.


# Company-regime STATE flags. These are absolute 0/1 indicators (is the firm
# profitable? cash-generative? in negative equity? growing fast?) emitted RAW
# into the panel -- NOT peer-standardized -- so the model can CONDITION on the
# regime instead of averaging a feature whose meaning flips between profitable
# and loss-making / hyper-growth names.
_STATE_FIELDS = ("profitable", "fcf_positive", "negative_equity", "hyper_growth")
_HYPER_GROWTH = 0.25     # YoY revenue growth above which a name is "hyper-growth"


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _combine_debt(daily, long_debt: pd.DataFrame, short_debt: pd.DataFrame,
                  fallback: pd.DataFrame) -> pd.DataFrame:
    """Total interest-bearing debt for the distress block.

    ⚠ PREFERS THE RECONCILED `totalDebt` COLUMN, via the shared `capital.borrowings`.
    This used to be `longTermDebt + shortTermDebt` only, and that is the SAME
    unclassified-balance-sheet trap `cash` fell into: a bank or REIT reports no
    current/non-current split (ASC 210-10-05-4), so BOTH leg columns are absent on
    exactly those filings. Measured: `longTermDebt` and `shortTermDebt` are each 0.277
    populated for Financials and 0.274 for Real Estate, while `totalDebt` -- which the
    extractor reconciles from the combined ST+LT tag, the two-leg sum and the REIT
    notes-payable fallback -- is **1.000 in every sector**. So restoring `cash` fixed the
    NUMERATOR of `cash_to_debt` and the denominator still nulled the row.

    Substituting it is safe because it is the same quantity, not a wider one: on every
    row where both legs exist, `totalDebt` equals their sum to within 1% on **100.00%**
    of rows in all eleven sectors (median relative gap 0.0). And it is genuinely debt
    rather than liabilities for the leg-less filers -- median `totalDebt`/assets is 0.095
    for Financials against `totalLiabilities`/assets of 0.877 (BAC: 732bn of debt inside
    3,198bn of liabilities on 3,499bn of assets; deposits are the difference).

    Falls back to the two legs, then to total liabilities, so a history vintage without
    the reconciled column still produces leverage."""
    borrowed = capital.borrowings(daily)
    if borrowed is not None and not borrowed.empty and borrowed.notna().any().any():
        return borrowed
    have_long = long_debt is not None and not long_debt.empty
    have_short = short_debt is not None and not short_debt.empty
    if have_long and have_short:
        return long_debt.add(short_debt, fill_value=0.0)
    if have_long:
        return long_debt
    if have_short:
        return short_debt
    return fallback if fallback is not None else pd.DataFrame()


def _enterprise_value(mcap: pd.DataFrame, additions: list, subtractions: list) -> pd.DataFrame:
    """True (fully-diluted) enterprise value, restricted to tickers with a market cap:

        EV = fully-diluted market cap
             + Σ additions      (total debt, lease liabilities, minority interest)
             − Σ subtractions   (cash, short-term investments)

    This is the production textbook EV: only NON-OPERATING liquid financial assets
    (cash + short-term investments) are netted, and the claims senior to / alongside
    common equity (debt, capitalized leases, minority interest) are added. NaN-
    tolerant: a missing component contributes 0."""
    ev = mcap.copy()
    for part in additions:
        if part is not None and not part.empty:
            ev = ev + part.reindex(columns=mcap.columns).fillna(0.0)
    for part in subtractions:
        if part is not None and not part.empty:
            ev = ev - part.reindex(columns=mcap.columns).fillna(0.0)
    return ev


#: The self-history z now lives in `common/xs.py` beside the other standardizers; this alias
#: keeps every call site in this module (and its test) working unchanged.
_self_history_z = self_history_z

# --------------------------------------------------------------------------- #
# Business-quality helpers (all from tags ALREADY extracted -- no new SEC pull)
#   #2 D&A/SBC realism, #5 forensic, #3 M&A digestion, #1 core/adjusted earnings.
# Each returns a {name: daily wide frame} dict that _derived_fields merges into F,
# so every field auto-expands to f_<name>_vs_peers + f_<name>_xs downstream.
# `daily` is the memoized accessor from _derived_fields (field -> date x ticker).
# --------------------------------------------------------------------------- #
_YEAR = 252   # trailing trading days ~= one calendar year (year-ago comparison)
_FIVE_YEARS = 5 * _YEAR   # ~5 trading years (multi-year trend comparison)


def _nopat_tax_rate(daily, default: float = 0.21) -> pd.DataFrame:
    """Point-in-time effective tax rate = tax / pretax, clipped to [0, 0.5] and
    defaulted to the US statutory ~21% where missing/nonsensical. INTERNAL ONLY: used to
    tax-adjust special items (core earnings) and to build NOPAT (ROIC).

    Named `_nopat_tax_rate`, not `_effective_tax_rate`, because `sector_features` emits a
    cube FEATURE called `effective_tax_rate` which is the RAW, unclipped, un-defaulted
    ratio. Two different numbers for two different jobs -- the shared name was a trap."""
    tax, pre = daily("incomeTaxExpense"), daily("pretaxIncome")
    if tax.empty or pre.empty:
        return pd.DataFrame()
    return ratio(tax, pre.where(pre > 0)).clip(0.0, 0.5).fillna(default)


def _da_realism_fields(daily) -> dict:
    """Does the buyback even cover the stock the firm gave away? `sbc_to_buyback` > 1 means
    the repurchase is not returning capital, it is mopping up option dilution.

    WAS ALSO HERE, now deleted: `implied_useful_life`, `useful_life_change`, `asset_age` and
    `intangible_amortization_share` -- the D&A-realism family, which asked whether reported
    depreciation is believable given the gross asset base. All four need `ppeGross`,
    `accumulatedDepreciation` or `amortizationIntangibles`. Sharadar delivers none of the
    three, and the SEC-owned `_sec` twins sit at 4.7% and 6.4% coverage, far too thin to
    rank a universe on. They return with the SEC extraction path."""
    F: dict[str, pd.DataFrame] = {}
    sbc = daily("stockBasedComp")
    buyback = capital.share_repurchases(daily)     # positive magnitude; see capital.py
    if not sbc.empty and buyback is not None and not buyback.empty:
        s2b = ratio(sbc, buyback, positive_den=True)
        if s2b.notna().any().any():
            F["sbc_to_buyback"] = s2b             # >1 = buybacks don't even cover SBC
    return F


def _beneish_m_score(daily, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Beneish (1999) 8-variable earnings-manipulation model, each index computed
    as this year vs one year ago (trailing 252 trading days). Higher M = more
    likely a manipulator (the classic screen is M > -1.78). Missing index ->
    neutral (1.0; TATA -> 0), so M is defined wherever revenue+assets exist.

    ⚠ THAT LAST CLAUSE IS ENFORCED, not just asserted. The score is seeded as a DENSE
    constant frame over the whole date x ticker grid and every absent index is filled with
    its neutral value, so a ticker with NO DATA AT ALL scored
    -4.84 + 0.920 + 0.528 + 0.404 + 0.892 + 0.115 - 0.172 - 0.327 = -2.48: a clean bill of
    health, below the -1.78 manipulation threshold, for a company that did not exist yet.
    Measured, that was 565,720 pre-listing rows carrying the identical constant. The guard
    above only checks that SOME ticker has revenue and assets, which is why the final
    `.where` is needed -- it is what makes the docstring's promise true per cell."""
    rev, assets = daily("totalRevenue"), capital.assets_ex_lease(daily)
    if rev.empty or assets.empty:
        return pd.DataFrame()
    ar, gp = daily("accountsReceivable"), daily("grossProfit")
    ca, ppe = daily("currentAssets"), daily("ppeNet")
    dep, sga = daily("depAmort"), daily("sellingGeneralAdmin")
    ltd, cl = daily("longTermDebt"), daily("currentLiabilities")
    ni, ocf = daily("netIncome"), daily("operatingCashFlow")

    def ix(cur: pd.DataFrame, prev: pd.DataFrame) -> pd.DataFrame:
        if cur is None or cur.empty or prev is None or prev.empty:
            return pd.DataFrame()
        return ratio(cur, prev.where(prev != 0))

    ar_sales = ratio(ar, rev)
    dsri = ix(ar_sales, ar_sales.shift(_YEAR))                 # days sales in receivables
    gm = ratio(gp, rev, positive_den=True)
    gmi = ix(gm.shift(_YEAR), gm)                              # gross-margin deterioration
    noncore = 1.0 - ratio(ca.add(ppe, fill_value=0.0), assets, positive_den=True)
    aqi = ix(noncore, noncore.shift(_YEAR))                    # asset-quality (soft assets)
    sgi = ix(rev, rev.shift(_YEAR))                            # sales growth
    deprate = ratio(dep, dep.add(ppe, fill_value=0.0), positive_den=True)
    depi = ix(deprate.shift(_YEAR), deprate)                   # slowing depreciation
    sgar = ratio(sga, rev, positive_den=True)
    sgai = ix(sgar, sgar.shift(_YEAR))                         # SG&A efficiency
    lev = ratio(ltd.add(cl, fill_value=0.0), assets, positive_den=True)
    lvgi = ix(lev, lev.shift(_YEAR))                           # leverage change
    tata = ratio(ni.sub(ocf, fill_value=np.nan), assets)     # total accruals / assets

    terms = [(0.920, dsri, 1.0), (0.528, gmi, 1.0), (0.404, aqi, 1.0),
             (0.892, sgi, 1.0), (0.115, depi, 1.0), (-0.172, sgai, 1.0),
             (-0.327, lvgi, 1.0), (4.679, tata, 0.0)]
    cols = sorted(set().union(*[set(t[1].columns) for t in terms
                                if isinstance(t[1], pd.DataFrame) and not t[1].empty]))
    if not cols:
        return pd.DataFrame()
    m = pd.DataFrame(-4.84, index=idx, columns=cols)
    for coef, df, neutral in terms:
        if isinstance(df, pd.DataFrame) and not df.empty:
            m = m + coef * df.reindex(index=idx, columns=cols).fillna(neutral).clip(-10.0, 10.0)
        else:
            m = m + coef * neutral
    # the docstring's support, enforced per cell: no revenue or no assets -> no score.
    # Deliberately keyed on DATA, not on listing, so a spin-off that files before it starts
    # trading (OTIS, CARR) keeps the score its filings support.
    support = (rev.reindex(index=idx, columns=cols).notna()
               & assets.reindex(index=idx, columns=cols).notna())
    return m.where(support).replace([np.inf, -np.inf], np.nan)


_NET_PENSION_TAGS = (
    "PensionAndOtherPostretirementDefinedBenefitPlansLiabilitiesNoncurrent",  # primary
    "DefinedBenefitPensionPlanLiabilitiesNoncurrent",                          # variant
)

# Footnote pension tags from the Financial Statement & NOTES sets (`notes_num`):
# the projected benefit obligation and fair value of plan assets, both INSTANT
# (balance-type) facts the primary statements never expose.
_FN_PBO_TAG = "DefinedBenefitPlanBenefitObligation"          # projected benefit obligation
_FN_PLAN_ASSETS_TAG = "DefinedBenefitPlanFairValueOfPlanAssets"


def _scope_to_universe(facts: pd.DataFrame | None,
                       universe: set[str]) -> pd.DataFrame | None:
    """A bulk facts frame restricted to the tickers the fundamentals panel is being built
    for. A no-op when the universe is unknown, so a caller that hands over a frame without
    a `ticker` column still gets the old behaviour rather than an empty one."""
    if facts is None or facts.empty or not universe or "ticker" not in facts.columns:
        return facts
    return facts[facts["ticker"].astype(str).isin(universe)]


def _pension_deficit_daily(pension_facts: pd.DataFrame | None,
                           idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Universe-wide recognized net DB-pension deficit from the Financial Statement
    Data Sets (`pension_facts` table), point-in-time on the FILING date, taking the
    latest period-end per filing. Primary net-liability tag preferred, the pension-
    only variant fills gaps. Empty frame if the table is unavailable."""
    if (pension_facts is None or pension_facts.empty
            or "tag" not in pension_facts.columns or "ticker" not in pension_facts.columns):
        return pd.DataFrame(index=idx)
    pf = pension_facts.copy()
    pf["as_of"] = pd.to_datetime(pf.get("filed"), errors="coerce")
    pf["value"] = pd.to_numeric(pf.get("value"), errors="coerce")
    if "qtrs" in pf.columns:                          # instant (balance-sheet) facts only
        pf = pf[pd.to_numeric(pf["qtrs"], errors="coerce").fillna(0) == 0]
    pf = pf.dropna(subset=["as_of", "value", "ticker"])

    def _one(tag: str) -> pd.DataFrame:
        d = pf[pf["tag"] == tag]
        if d.empty:
            return pd.DataFrame(index=idx)
        # sort so the latest period-end (ddate) wins within a filing (aggfunc='last')
        d = d.sort_values(["ticker", "as_of", "ddate"]).rename(columns={"value": "pension_deficit"})
        return fundamentals_to_daily(d, "pension_deficit", idx)

    prim, var = _one(_NET_PENSION_TAGS[0]), _one(_NET_PENSION_TAGS[1])
    if prim.empty:
        return var
    if var.empty:
        return prim
    return prim.combine_first(var)


def _notes_num_daily(notes_num: pd.DataFrame | None, tag: str,
                     idx: pd.DatetimeIndex, instant: bool = True) -> pd.DataFrame:
    """One footnote-numeric tag from the NOTES sets (`notes_num`) -> daily wide
    frame, point-in-time on the FILING date, latest period-end (`ddate`) per filing.
    `instant` keeps the balance-type facts (qtrs==0, e.g. PBO / plan assets); else
    the duration facts (qtrs>0, e.g. service cost). Empty frame if unavailable."""
    if (notes_num is None or notes_num.empty
            or "tag" not in notes_num.columns or "ticker" not in notes_num.columns):
        return pd.DataFrame(index=idx)
    d = notes_num[notes_num["tag"] == tag].copy()
    if d.empty:
        return pd.DataFrame(index=idx)
    d["as_of"] = pd.to_datetime(d.get("filed"), errors="coerce")
    d["value"] = pd.to_numeric(d.get("value"), errors="coerce")
    q = pd.to_numeric(d.get("qtrs"), errors="coerce").fillna(0)
    d = d[(q == 0) if instant else (q > 0)]
    d = d.dropna(subset=["as_of", "value", "ticker"])
    if d.empty:
        return pd.DataFrame(index=idx)
    d = d.sort_values(["ticker", "as_of", "ddate"]).rename(columns={"value": tag})
    return fundamentals_to_daily(d, tag, idx)


def load_tagged_facts(context: Context, table: str, tags: tuple[str, ...],
                      columns: list[str] | None = None) -> pd.DataFrame | None:
    """Read ONLY the rows whose `tag` the pension/footnote builders actually use — they touch just 2
    tags of each facts table (`notes_num` has 10 tags; only ~16% of its rows are these two). Pulling
    the whole table then filtering in-memory is the same waste pattern as the 13F/embedding tables.
    The tag filter is pushed down server-side. None if the table is absent/empty or no row matches."""
    df = context.store.load(table, columns=columns or _FACT_COLS,
                            where={"tag": list(tags)}, optional=True)
    return df.reset_index(drop=True) if df is not None else None


def load_pension_facts_scoped(context: Context) -> pd.DataFrame | None:
    """`pension_facts` restricted to the recognized net-liability tags the panel reads."""
    return load_tagged_facts(context, _PENSION_FACTS_TABLE, _NET_PENSION_TAGS)


def load_notes_num_scoped(context: Context) -> pd.DataFrame | None:
    """`notes_num` restricted to the footnote PBO + plan-asset tags the panel reads."""
    return load_tagged_facts(context, _NOTES_NUM_TABLE, (_FN_PBO_TAG, _FN_PLAN_ASSETS_TAG))


def _forensic_fields(daily, idx: pd.DatetimeIndex) -> dict:
    """#5 -- accounting-quality / hidden-leverage red flags: working-capital days
    and their year-over-year drift (supplier-funded growth = rising DPO; channel
    stuffing = rising DSO), off-balance-sheet-INCLUSIVE net leverage (adds lease
    liabilities + pension deficit), and the Beneish M-score. All inputs already
    extracted; `pension_deficit` is the bulk Financial-Statement-Data-Sets frame."""
    F: dict[str, pd.DataFrame] = {}
    rev, cogs = daily("totalRevenue"), daily("costOfRevenue")
    ar, ap, inv = daily("accountsReceivable"), daily("accountsPayable"), daily("inventory")

    dso = ratio(ar, rev, positive_den=True) * 365.0
    if dso.notna().any().any():
        F["dso"] = dso
        F["dso_change"] = (dso - dso.shift(_YEAR)).replace([np.inf, -np.inf], np.nan)
    dpo = ratio(ap, cogs, positive_den=True) * 365.0
    if dpo.notna().any().any():
        F["dpo"] = dpo
        F["dpo_change"] = (dpo - dpo.shift(_YEAR)).replace([np.inf, -np.inf], np.nan)
    dio = ratio(inv, cogs, positive_den=True) * 365.0
    if dio.notna().any().any():
        F["dio"] = dio
    if "dso" in F and "dpo" in F and "dio" in F:
        ccc = F["dso"].add(F["dio"], fill_value=np.nan).sub(F["dpo"], fill_value=np.nan)
        if ccc.notna().any().any():
            F["cash_conversion_cycle"] = ccc

    # DELETED 2026-09-05: `net_debt_incl_offbs_to_ebitda`.
    #
    # It was the off-balance-sheet-inclusive twin of `net_debt_to_ebitda`: borrowings +
    # capitalized leases + pension/OPEB deficit + asset-retirement obligations - liquid
    # assets, over EBITDA. THREE of those four distinguishing legs do not exist on the
    # Sharadar substrate -- `fundamentals_history` has no `operatingLeaseLiability`,
    # `financeLeaseLiability` or `assetRetirementObligation` column at all -- and the
    # fourth, the pension pool, reaches 199 of 491 tickers.
    #
    # It survived only because the distress block computed its own debt from
    # `longTermDebt + shortTermDebt` while this one went through `capital.borrowings`. Once
    # the distress block was fixed to read the reconciled `totalDebt` too (both were wrong
    # for banks and REITs, where neither leg column exists), the last artificial difference
    # went and the two became THE SAME NUMBER: identical fingerprint hashes on the frozen
    # slice (`bc782e6854f8f83d` / `57e0d7b74e5e2e23`), r = 0.9997 on the 18-ticker
    # redundancy fixture and 0.9955 across the live cube. The audit deleted
    # `net_debt_to_ebitdare` at r = 0.9998 for exactly this reason.
    #
    # Nothing is lost: the pension overhang it folded in is already carried, undiluted, by
    # `pension_overhang_leverage` and `pension_retirement_liability`. `net_debt_to_ebitda`
    # is kept as the survivor because it is the standard credit metric and the one
    # `lgbm_modelling.yml` already declares a monotone constraint on.
    #
    # With it went this block's only reads of `ebitda`, the coalesced pension pool and the
    # operating-cash gate, so all three left the signature too.
    m = _beneish_m_score(daily, idx)
    if not m.empty and m.notna().any().any():
        F["beneish_m_score"] = m
    return F


def _digestion_fields(daily, fund_hist: pd.DataFrame, idx: pd.DatetimeIndex,
                      yoy_periods: int,
                      operating_cash: frozenset[str] | None = None) -> dict:
    """#3 -- is bought growth being digested? ROIC WITH vs WITHOUT acquired intangibles (the
    wedge = how much acquisitions drag returns), the intangibles balance-sheet weight
    (writedown exposure), and SG&A elasticity to revenue (<1 = synergies captured, ~1 =
    bolt-on with no integration).

    ⚠ THE DEDUCTION IS GOODWILL **AND** OTHER INTANGIBLES, COMBINED, and the feature names
    say so. Sharadar's `intangibles` is the only intangibles basis SF1 delivers and it does
    not split the two; the SEC-owned split legs (`goodwill_sec` 10.2%,
    `intangiblesExGoodwill_sec` 7.1%) are far too thin to subtract from a universe-wide
    denominator. Reading the bare `goodwill` -- which no producer writes -- is what made the
    "ex-goodwill" ROIC subtract NOTHING and correlate 1.0000 with its incl-goodwill twin,
    with `goodwill_roic_drag` identically zero across 3.04 M rows.

    So `roic_ex_intangibles` is a WIDER deduction than a textbook ex-goodwill ROIC: it also
    removes purchased patents, customer lists and brands. That is the honest name for what
    the data supports, and it is the same deduction for every ticker, which is what makes
    the cross-sectional rank meaningful."""
    F: dict[str, pd.DataFrame] = {}
    # asset base EX the ASC-842 ROU asset, so the FY2019 adoption jump does not read as
    # balance-sheet growth (see `totalAssetsExLease` in the extractor).
    oi, assets = daily("operatingIncome"), capital.assets_ex_lease(daily)
    equity = daily("stockholdersEquity")
    intangibles = daily("intangibles")          # goodwill + other intangibles, COMBINED
    tax = _nopat_tax_rate(daily)

    if not oi.empty and not equity.empty:
        nopat = oi * (1.0 - tax) if not tax.empty else oi
        # invested capital now INCLUDES capitalized leases (shared definition), matching
        # how leases are already treated as debt in EV and in the leverage ratios.
        ic = capital.invested_capital(daily, operating_cash=operating_cash)
        roic_incl = ratio(nopat, ic, positive_den=True) if ic is not None else pd.DataFrame()
        if roic_incl.notna().any().any():
            F["roic_incl_intangibles"] = roic_incl
            if not intangibles.empty:
                roic_ex = ratio(nopat, ic.sub(intangibles, fill_value=0.0), positive_den=True)
                if roic_ex.notna().any().any():
                    F["roic_ex_intangibles"] = roic_ex
                    # incl - ex < 0 => acquired intangibles dilute returns (overpaid)
                    F["intangibles_roic_drag"] = roic_incl.sub(roic_ex, fill_value=np.nan)

    if not intangibles.empty and not assets.empty:
        gta = ratio(intangibles, assets, positive_den=True)
        if gta.notna().any().any():
            F["intangibles_to_assets"] = gta
        gte = ratio(intangibles, equity.where(equity > 0))
        if gte.notna().any().any():
            F["intangibles_to_equity"] = gte   # >1 => a writedown can wipe out book equity

    sga_g = fiscal_change_to_daily(fund_hist, "sellingGeneralAdmin", idx, kind="pct", periods=yoy_periods)
    rev_g = fiscal_change_to_daily(fund_hist, "totalRevenue", idx, kind="pct", periods=yoy_periods)
    if sga_g.notna().any().any() and rev_g.notna().any().any():
        el = ratio(sga_g, rev_g.where(rev_g.abs() >= 0.02))   # guard ~flat-revenue blow-ups
        if el.notna().any().any():
            F["sga_elasticity"] = el
    return F


def _per_share_and_profit_slice_fields(daily, fund_hist: pd.DataFrame, idx: pd.DatetimeIndex,
                                       yoy_periods: int,
                                       close: pd.DataFrame | None) -> dict:
    """Per-share economics and the non-controlled slice of reported profit.

    Per share  reported diluted EPS (98.8%) and the OPTION OVERHANG (diluted - basic) / basic,
               which the net share count hides once buybacks offset the SBC issuance.
    NCI        how much of the bottom line belongs to somebody else. `netIncome` maps from
               Sharadar's `consolinc`, which INCLUDES non-controlling interests, so this is
               the share of consolidated profit the parent's shareholders do not own.

    WAS ALSO HERE, now deleted: the debt-maturity ladder (`debtMaturity1y` / `5yTotal`), the
    cash-tax pair (`incomeTaxesPaid`), the valuation-allowance and unrecognized-tax-benefit
    ratios, the OCI drag (`comprehensiveIncome`), the receivable allowance and the segment
    count. SF1 carries none of those tags and each measured ZERO values against the live
    table. They return with the SEC extraction path.
    """
    F: dict[str, pd.DataFrame] = {}
    # (diluted - basic) / basic, already computed by the extractor on the periods where BOTH
    # counts are reported -- dividing the two independently forward-filled columns here would
    # compare a stale diluted count against a fresh basic one.
    overhang = daily("optionOverhang")
    if not overhang.empty and overhang.notna().any().any():
        F["option_overhang"] = overhang
    eps = daily("epsDiluted")
    if not eps.empty and close is not None:
        cols = eps.columns.intersection(close.columns)
        ey = ratio(eps[cols].where(eps[cols] > 0), close[cols], positive_den=True)
        if ey.notna().any().any():
            F["eps_yield"] = ey        # reported diluted EPS / price: E/P net of preferred
    dps_growth = fiscal_change_to_daily(fund_hist, "dividendsPerShare", idx,
                                         kind="pct", periods=yoy_periods)
    if dps_growth.notna().any().any():
        F["dps_growth"] = dps_growth

    # The NCI leg reads `netIncomeToNci`, the live column name. It was written `nciIncome`,
    # which nothing has ever produced, so the feature emitted nothing -- invisible to the
    # column-existence diff because the argument is a loop variable, not a literal.
    ni, nci = daily("netIncome"), daily("netIncomeToNci")
    if not nci.empty and not ni.empty:
        share = ratio(nci, ni.abs().where(ni.abs() > 0))
        if share.notna().any().any():
            F["nci_income_share"] = share
    return F


# --------------------------------------------------------------------------- #
# Per-block field builders                                                     #
#                                                                              #
# `_derived_fields` was one 541-line function producing ~110 features. It is now a
# composition of the blocks below, each following the convention the business-quality
# helpers above already established: take the memoized `daily` accessor (plus whatever
# frames its block genuinely shares), return {feature_name: date x ticker frame}, and
# emit a feature only where it has values. Granularity is one function per COHESIVE
# BLOCK, not per feature: a function per feature would have to thread `daily`,
# `revenue`, `mcap`, `ev` and `pension_ret` through ~110 signatures, which reads worse
# than the original, not better.
# --------------------------------------------------------------------------- #
def _pension_pool(notes_num: pd.DataFrame | None,
                  pension_facts: pd.DataFrame | None,
                  idx: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The three SHARED pension frames: (pbo, footnote deficit, coalesced overhang).

    Not features -- these feed the EV calculation, the forensic block and the scale
    features below, which is why they are computed once here rather than per block.

    The overhang is gap-filled (`combine_first`, NOT a sum -> no OPEB double-count)
    across TWO sources in order of directness:
      1) bulk Financial-Statement-Data-Sets recognized net liability (`pension_facts`) --
         6,244 rows over 125 tickers, median 17 years each,
      2) footnote funded status = PBO - plan assets from the NOTES sets (`notes_num`) --
         the widest single source at 4,209 plan-asset rows over 214 CIKs, but the PBO leg
         covers only 122, which is what caps `pension_funded_ratio` at 98 tickers.

    A third leg used to sit between them, reading a `pensionDeficit` column off
    `fundamentals_history`. That column does not exist on the Sharadar-first schema and
    never has, so the leg was dead code; it was removed rather than extracted, because
    these two sources are the substrate limit. Measured ceiling: 199 of 489 tickers ever
    receive a value, at 20.9% mean fill of their live span."""
    pbo = _notes_num_daily(notes_num, _FN_PBO_TAG, idx, instant=True)
    plan_assets = _notes_num_daily(notes_num, _FN_PLAN_ASSETS_TAG, idx, instant=True)
    fn_deficit = pd.DataFrame()
    if not pbo.empty and not plan_assets.empty:
        # funded status = plan assets - PBO; the deficit (underfunding) is the debt-like part.
        fn_deficit = pbo.sub(plan_assets).clip(lower=0.0)

    pension_ret = _pension_deficit_daily(pension_facts, idx)
    if not fn_deficit.empty:
        pension_ret = (pension_ret.combine_first(fn_deficit)
                       if not pension_ret.empty else fn_deficit)
    if not pension_ret.empty:
        pension_ret = pension_ret.clip(lower=0.0)          # underfunding only (>= 0)
    return pbo, fn_deficit, pension_ret


def _pension_health_fields(daily, pbo: pd.DataFrame, notes_num: pd.DataFrame | None,
                           idx: pd.DatetimeIndex, pension_ret: pd.DataFrame) -> dict:
    """Funded health straight off the footnote, plus the recognized overhang level.
    The NOTES footnote (PBO / plan assets) yields a standalone funded ratio that covers
    names the balance-sheet tag misses."""
    F: dict[str, pd.DataFrame] = {}
    plan_assets = _notes_num_daily(notes_num, _FN_PLAN_ASSETS_TAG, idx, instant=True)
    if not pbo.empty and not plan_assets.empty:
        funded_ratio = ratio(plan_assets, pbo, positive_den=True)   # 1.0 = fully funded
        if funded_ratio.notna().any().any():
            F["pension_funded_ratio"] = funded_ratio
    if not pension_ret.empty and pension_ret.notna().any().any():
        F["pension_retirement_liability"] = pension_ret
    return F


def _pension_scale_fields(pension_ret: pd.DataFrame, pbo: pd.DataFrame,
                          mcap: pd.DataFrame) -> dict:
    """Pension obligations SCALED by equity value -- the burden overhanging the stock.

    Two genuinely different quantities: `pbo_to_mcap` is the GROSS obligation, which flags
    rate/return sensitivity even for a fully FUNDED plan, while `pension_overhang_leverage`
    is the net DEFICIT, the debt-like part.

    `pension_underfunding_to_mcap` USED TO BE HERE AND IS DELETED. It scaled `fn_deficit`,
    the footnote funded status -- but `fn_deficit` is the LAST FALLBACK inside the coalesced
    pool `pension_overhang_leverage` already uses, so wherever the footnote was the winning
    source the two features were the same number. Measured r = 1.000000 on their overlap,
    against 8.3% coverage versus the pool's 28.3%: a strict subset, not a second view."""
    F: dict[str, pd.DataFrame] = {}
    for name, src in (("pension_overhang_leverage", pension_ret),
                      ("pbo_to_mcap", pbo)):
        if src is None or src.empty:
            continue
        r = ratio(src, mcap, positive_den=True)
        if r.notna().any().any():
            F[name] = r
    return F


def _valuation_yield_fields(mcap: pd.DataFrame, net_income: pd.DataFrame,
                            revenue: pd.DataFrame, equity: pd.DataFrame,
                            fcf: pd.DataFrame) -> dict:
    """Earnings / sales / book / FCF yields against market cap, plus the loss magnitude
    the yields deliberately exclude.

    Earnings/FCF/EV yields are only monotone as "cheapness" when the NUMERATOR is
    positive: a negative E/P is not "cheap", it is a loss. Those are masked to NaN
    (LightGBM handles NaN; the peer-z/rank then only ranks names where the metric is
    defined), and the `profitable` / `fcf_positive` flags carry the regime instead.
    Sales/price stays valid for everyone (revenue is always positive).

    ⚠ THE MASK IS MEASURED, NOT ASSUMED -- and the old justification for it was wrong.
    The claim used to be "ranking loss-makers by E/P is noise". It is not noise: it is
    REVERSED. Spearman IC of the signed yield against `target_rank`, month-end 1995-2026,
    computed inside each sign subset:

        numerator            NEGATIVE subset          positive subset
        netIncome            -0.0297  (t = -3.17)     +0.0116  (t = +3.31)
        freeCashflow         -0.0159  (t = -2.44)     +0.0162  (t = +4.98)
        stockholdersEquity   -0.0414  (t = -2.56)     -0.0062  (t = -2.18)

    A DEEPER loss predicts a BETTER forward return -- the opposite direction to the one
    E/P carries among profitable names, and negative in 4/4 sub-periods. So folding the
    two into one monotone column averages them away: the merged column's Q5-Q1 spread in
    mean `target_rank` is 0.0040 against the masked column's 0.0053. Un-masking is a
    measured REGRESSION. That is why the mask stays.

    `loss_intensity` is how the discarded information is kept instead: the annual loss as a
    POSITIVE share of market cap (house convention for `*_intensity`), defined ONLY where
    earnings are negative. NaN for profitable names, never 0 -- a zero would pile every
    profitable name into one tie at the bottom of the cross-sectional rank, which is the
    same defect the dividend zero-fill had. Given its own column the model can learn the
    reversed sign directly instead of having it cancel inside `earnings_yield`."""
    return {
        "earnings_yield": ratio(net_income.where(net_income > 0), mcap, positive_den=True),
        "sales_yield": ratio(revenue, mcap, positive_den=True),
        "book_yield": ratio(equity.where(equity > 0), mcap, positive_den=True),
        "fcf_yield": ratio(fcf.where(fcf > 0), mcap, positive_den=True),
        "loss_intensity": ratio(-net_income.where(net_income < 0), mcap, positive_den=True),
    }


def _enterprise_value_frame(daily, close: pd.DataFrame | None, mcap: pd.DataFrame,
                            equity: pd.DataFrame, d2e: pd.DataFrame,
                            cash: pd.DataFrame, pension_ret: pd.DataFrame,
                            operating_cash: frozenset[str] | None = None) -> pd.DataFrame:
    """True (fully-diluted) enterprise value; feeds every EV yield.

      EV = fully-diluted mcap
           + total debt (borrowings + capitalized leases, shared `capital` definition)
           + minority interest + redeemable NCI / temporary equity + PREFERRED equity
           + pension/OPEB net deficit
           - non-operating liquid assets (cash + ST investments + current marketables)

    Prefers diluted shares x price for the equity claim (falls back to basic mcap when
    diluted shares are absent); real debt columns preferred, with debtToEquity*equity as
    a last-resort fallback. PREFERRED stock and redeemable NCI rank ahead of / alongside
    common and were simply missing: `preferredEquity` is the balance-sheet carrying
    amount, which for a par-value-only filer understates the claim -- but never
    overstates it, so including it is strictly closer to the true EV than omitting it."""
    diluted = daily("dilutedShares")
    fd_mcap = mcap
    if close is not None and not diluted.empty and diluted.notna().any().any():
        cols = diluted.columns.intersection(close.columns)
        fd = (close[cols] * diluted[cols]).where(lambda x: x > 0)
        fd_mcap = fd.combine_first(mcap)          # diluted where available, else basic
    debt = capital.total_debt(daily)
    if debt is None or debt.empty:
        debt = (d2e.clip(lower=0.0) * equity.where(equity > 0)
                if not d2e.empty and not equity.empty else pd.DataFrame())
    liquid = capital.liquid_assets(daily, operating_cash=operating_cash)
    return _enterprise_value(
        fd_mcap,
        [debt, daily("minorityInterest"), daily("redeemableNCI"),
         daily("preferredEquity"), pension_ret],
        [liquid if liquid is not None
         else capital.drop_operating_cash(cash, operating_cash)])


def _ev_yield_fields(ebitda: pd.DataFrame, fcf: pd.DataFrame, ev: pd.DataFrame) -> dict:
    """EBITDA and FCF against the whole capital structure's price.

    FCF/EV is the cleanest cross-sector cash-valuation yield, and it is exactly the
    "Fully-Diluted FCF Yield" / energy FCF-EV yield (freeCashflow = OCF - capex)."""
    F: dict[str, pd.DataFrame] = {}
    if not ebitda.empty:
        F["ebitda_to_ev"] = ratio(ebitda.where(ebitda > 0), ev, positive_den=True)
    fcf_to_ev = ratio(fcf.where(fcf > 0), ev, positive_den=True)
    if not fcf_to_ev.empty and fcf_to_ev.notna().any().any():
        F["fcf_to_ev"] = fcf_to_ev
    return F


def _altman_z_fields(daily, mcap: pd.DataFrame, revenue: pd.DataFrame) -> dict:
    """Altman Z (market-value variant): the standard bankruptcy-risk screen.

      Z = 1.2*WC/TA + 1.4*RE/TA + 3.3*EBIT/TA + 0.6*mcap/TL + 1.0*Sales/TA

    ASC-842-adoption-free asset base (see `totalAssetsExLease`): the FY2019 ROU jump
    otherwise pushes every lease-heavy name toward "distressed" on Z."""
    assets_z = capital.assets_ex_lease(daily)
    if assets_z.empty:
        return {}
    ta = assets_z.where(assets_z > 0)
    wc = daily("currentAssets").sub(daily("currentLiabilities"), fill_value=np.nan)
    z = (1.2 * ratio(wc, ta) + 1.4 * ratio(daily("retainedEarnings"), ta)
         + 3.3 * ratio(daily("operatingIncome"), ta)
         + 0.6 * ratio(mcap, daily("totalLiabilities"), positive_den=True)
         + 1.0 * ratio(revenue, ta))
    if z.empty or not z.notna().any().any():
        return {}
    return {"altman_z": sanitize(z)}


def _pegy_fields(daily, fund_hist: pd.DataFrame, idx: pd.DatetimeIndex,
                 mcap: pd.DataFrame, net_income: pd.DataFrame, yoy_periods: int,
                 earnings_history: pd.DataFrame | None) -> dict:
    """PEGY = P/E / (EPS growth% + dividend yield%).

    Trailing P/E; the growth term PREFERS PROJECTED EPS growth (NTM/TTM-1 from the
    analyst-estimate archive) and falls back to TTM realized net-income growth.

    The dividend yield fills to 0 for non-payers, but GROWTH must be known or PEGY is
    undefined (don't silently treat unknown growth as 0). This is the SEC cash-flow
    (`dividendsPaid`) leg of the reconciled dividend yield -- the precise
    per-share/ex-date version is the standalone `dividend_yield` feature in
    dividend_features.py (both agree; see its reconciliation note)."""
    pe = ratio(mcap, net_income.where(net_income > 0), positive_den=True)
    growth_pct = None
    if earnings_history is not None and not earnings_history.empty:
        ntm_e, ttm_e = ntm_ttm_eps(earnings_history, idx)
        if not ntm_e.empty and not ttm_e.empty:
            proj = ratio(ntm_e, ttm_e.where(ttm_e > 0)) - 1.0        # projected EPS growth
            if proj.notna().any().any():
                growth_pct = proj * 100.0
    if growth_pct is None:
        growth_pct = fiscal_change_to_daily(fund_hist, "netIncome", idx,
                                            kind="pct", periods=yoy_periods) * 100.0
    div_yield_pct = ratio(daily("dividendsPaid"), mcap, positive_den=True) * 100.0
    denom = (growth_pct + div_yield_pct.fillna(0.0)).where(lambda x: x > 0)
    pegy = ratio(pe, denom)
    if pegy.empty or not pegy.notna().any().any():
        return {}
    return {"pegy": sanitize(pegy)}


def _reit_multiple_fields(daily, fund_hist: pd.DataFrame, net_income: pd.DataFrame,
                          mcap: pd.DataFrame, ev: pd.DataFrame) -> dict:
    """FFO yield (FFO / price = 1 / P-FFO), scoped to the GICS equity-REIT group.

    Previously gated on `realEstateNet | rentalIncome` being tagged, which handed FFO to ~56
    non-real-estate names (utilities, IT and industrial lessors tag
    `OperatingLeaseLeaseIncome`) while missing 9 of 31 REITs. GICS decides instead.

    ⚠ ONLY THE D&A LEG OF NAREIT FFO IS AVAILABLE. `gainOnDispositions` and
    `realEstateImpairment` are not in SF1, so a REIT that sold a property at a gain reads
    high here -- which is exactly what those two adjustments exist to remove.

    `implied_cap_rate` USED TO BE HERE AND IS DELETED. It was
    `(operatingIncome + depAmort + realEstateImpairment) / EV`, and with the impairment leg
    absent that is `(operatingIncome + depAmort) / EV` -- which is the field map's own
    DEFINITION of `ebitda`, so the feature was `ebitda_to_ev` masked to REITs and correlated
    with it at exactly 1.000000. A gate is not a second feature: the model already gets REIT
    membership from the sector categorical."""
    if not family_tickers(fund_hist, "reit"):
        return {}
    ffo = net_income.add(daily("depAmort"), fill_value=0.0)
    fy = mask_columns(ratio(ffo, mcap, positive_den=True), fund_hist, "reit")
    return {"ffo_yield": fy} if fy.notna().any().any() else {}


def _profitability_level_fields(daily, revenue: pd.DataFrame, net_income: pd.DataFrame,
                               fcf: pd.DataFrame) -> dict:
    """Profitability / moat: raw ratios straight from the history, plus FCF margin and
    accruals (the net-income-vs-cash gap)."""
    F: dict[str, pd.DataFrame] = {}
    for field in ["grossMargins", "operatingMargins", "profitMargins",
                  "returnOnEquity", "debtToEquity", "revenueGrowth", "earningsGrowth"]:
        f = daily(field)
        if not f.empty:
            F[field] = f

    fcf_margin = ratio(fcf, revenue, positive_den=True)
    if not fcf_margin.empty:
        F["fcf_margin"] = fcf_margin
    if not net_income.empty and not fcf.empty and not revenue.empty:
        cols = net_income.columns.intersection(fcf.columns)
        accr_num = net_income[cols] - fcf[cols]
        F["accruals"] = ratio(accr_num, revenue, positive_den=True)
    return F


def _growth_trend_fields(daily, fund_hist: pd.DataFrame, idx: pd.DatetimeIndex,
                         revenue: pd.DataFrame, rnd: pd.DataFrame,
                         yoy_periods: int) -> dict:
    """Growth / trend / dilution from the fiscal series (year-over-year), the 5-year
    margin trend, and R&D intensity.

    The 5-YEAR MARGIN TREND asks whether the operating margin is STRUCTURALLY expanding,
    not just having one good year: TTM operating income / revenue now vs ~5 trading years
    ago (point-in-time, percentage-point change). Positive = durable expansion."""
    F: dict[str, pd.DataFrame] = {}
    for name, field, kind in (("fcf_growth", "freeCashflow", "pct"),
                              ("shares_growth", "sharesOutstanding", "pct"),
                              ("gross_margin_chg", "grossMargins", "diff")):
        chg = fiscal_change_to_daily(fund_hist, field, idx, kind=kind, periods=yoy_periods)
        if chg.notna().any().any():
            F[name] = chg

    op_margin = ratio(daily("operatingIncome"), revenue, positive_den=True)
    if not op_margin.empty and op_margin.notna().any().any():
        om_5y = sanitize(op_margin - op_margin.shift(_FIVE_YEARS))
        if om_5y.notna().any().any():
            F["operating_margin_5y_chg"] = om_5y

    rd_intensity = ratio(rnd, revenue, positive_den=True)
    if not rd_intensity.empty and rd_intensity.notna().any().any():
        F["rd_intensity"] = rd_intensity
    return F


def _reinvestment_fields(daily, fund_hist: pd.DataFrame, idx: pd.DatetimeIndex,
                         yoy_periods: int) -> dict:
    """Reinvestment quality: depreciation & amortization vs capex.

    If D&A runs ABOVE capex (da_to_capex > 1) and, worse, is GROWING faster than capex
    (da_minus_capex_growth > 0), the firm is consuming its asset base faster than it
    reinvests -> aging PP&E / under-investment / a likely future capex cliff, and
    reported earnings flattered by low capex. The model learns the sign."""
    F: dict[str, pd.DataFrame] = {}
    depamort, capex = daily("depAmort"), daily("capex")
    if not depamort.empty and not capex.empty:
        da_to_capex = ratio(depamort.abs(), capex.abs(), positive_den=True)
        if da_to_capex.notna().any().any():
            F["da_to_capex"] = da_to_capex
    da_growth = fiscal_change_to_daily(fund_hist, "depAmort", idx,
                                       kind="pct", periods=yoy_periods)
    capex_growth = fiscal_change_to_daily(fund_hist, "capex", idx,
                                          kind="pct", periods=yoy_periods)
    if da_growth.notna().any().any() and capex_growth.notna().any().any():
        F["da_minus_capex_growth"] = da_growth - capex_growth
    return F


def _quality_regime_fields(daily, fund_hist: pd.DataFrame, idx: pd.DatetimeIndex,
                           revenue: pd.DataFrame, net_income: pd.DataFrame,
                           fcf: pd.DataFrame, long_debt: pd.DataFrame,
                           yoy_periods: int) -> dict:
    """REGIME-ROBUST quality measures that stay defined for loss-makers and growth names,
    where the earnings-based yields are masked out:

      gross_profitability  (Novy-Marx) gross profit / total assets -- monotone even for a
                           net-loss-making firm, so it scores the growth / unprofitable
                           cohort the yields cannot.
      asset_growth         (Fama-French CMA) firms that expand the asset base
                           aggressively subsequently UNDERperform (empire-building /
                           over-investment); sign is -1 in the model. On the ex-lease
                           base, so the FY2019 ASC-842 adoption jump is not read as
                           investment.
      rule_of_40           TTM revenue-growth% + FCF-margin%; >40 is elite (a fast grower
                           OR a cash cow).
      piotroski_f_score    0-9 fundamental-health binaries: profitability (ROA>0, CFO>0,
                           dROA>0, CFO/assets>ROA), lower leverage / better liquidity / no
                           dilution, and rising gross margin / asset turnover. Scored only
                           where the core inputs (assets, NI, CFO) exist, so a data-less
                           name is not a false 0.
    """
    F: dict[str, pd.DataFrame] = {}
    assets = capital.assets_ex_lease(daily)
    gm_lvl = daily("grossMargins")
    if not gm_lvl.empty and not revenue.empty and not assets.empty:
        cols = gm_lvl.columns.intersection(revenue.columns)
        gross_profit = gm_lvl[cols] * revenue[cols]           # grossMargins * revenue
        gp = ratio(gross_profit, assets, positive_den=True)
        if not gp.empty and gp.notna().any().any():
            F["gross_profitability"] = gp

    _assets_field = ("totalAssetsExLease" if "totalAssetsExLease" in fund_hist.columns
                     else "totalAssets")
    asset_growth = fiscal_change_to_daily(fund_hist, _assets_field, idx,
                                          kind="pct", periods=yoy_periods)
    if asset_growth.notna().any().any():
        F["asset_growth"] = asset_growth

    rev_growth_pct = fiscal_change_to_daily(fund_hist, "totalRevenue", idx,
                                            kind="pct", periods=yoy_periods) * 100.0
    fcf_margin_pct = ratio(fcf, revenue, positive_den=True) * 100.0
    rule40 = sanitize(rev_growth_pct + fcf_margin_pct)
    if rule40.notna().any().any():
        F["rule_of_40"] = rule40
    _oa = capital.assets_ex_lease(daily)
    _ocf = daily("operatingCashFlow")
    _sh = daily("sharesOutstanding")
    _roa = ratio(net_income, _oa, positive_den=True)
    _cr = ratio(daily("currentAssets"), daily("currentLiabilities"), positive_den=True)
    _lev = ratio(long_debt, _oa, positive_den=True)
    _gm = daily("grossMargins")
    _turn = ratio(revenue, _oa, positive_den=True)
    if not _oa.empty and not _ocf.empty and not net_income.empty:
        y = _YEAR
        parts = [
            (_roa > 0), (_ocf > 0), (_roa > _roa.shift(y)),
            (ratio(_ocf, _oa, positive_den=True) > _roa),                  # accruals: cash > profit
            (_lev < _lev.shift(y)), (_cr > _cr.shift(y)),
            (_sh <= _sh.shift(y) * 1.001),                                 # no net dilution
            (_gm > _gm.shift(y)), (_turn > _turn.shift(y)),
        ]
        fscore = sum(p.astype("float64") for p in parts)
        gate = _oa.notna() & net_income.notna() & _ocf.notna()
        fscore = fscore.where(gate)
        if fscore.notna().any().any():
            F["piotroski_f_score"] = fscore
    return F


def _state_flag_fields(daily, net_income: pd.DataFrame, fcf: pd.DataFrame,
                       equity: pd.DataFrame) -> dict:
    """Absolute 0/1 regime flags (emitted RAW, see _STATE_FIELDS).

    A NaN base -> NaN flag (never a false 0), so "no data" is not read as
    "unprofitable"."""
    def _flag(base: pd.DataFrame, cond: pd.DataFrame) -> pd.DataFrame:
        return cond.astype(float).where(base.notna())

    F: dict[str, pd.DataFrame] = {}
    if not net_income.empty:
        F["profitable"] = _flag(net_income, net_income > 0)
    if not fcf.empty:
        F["fcf_positive"] = _flag(fcf, fcf > 0)
    if not equity.empty:
        F["negative_equity"] = _flag(equity, equity <= 0)
    rev_growth_lvl = daily("revenueGrowth")
    if not rev_growth_lvl.empty:
        F["hyper_growth"] = _flag(rev_growth_lvl, rev_growth_lvl > _HYPER_GROWTH)
    return F


def _quarter_momentum_fields(daily, fund_hist: pd.DataFrame, idx: pd.DatetimeIndex,
                             yoy_periods: int) -> dict:
    """LATEST-QUARTER momentum (discrete single quarter, not TTM): captures the
    acceleration / inflection that TTM smooths away. Needs the discrete single-quarter
    columns emitted by the extractor."""
    F: dict[str, pd.DataFrame] = {}
    q_rev_yoy = fiscal_apply_to_daily(fund_hist, "revenue_q", idx,
                                      lambda s: s.pct_change(yoy_periods))
    if q_rev_yoy.notna().any().any():
        F["q_rev_growth"] = q_rev_yoy
        # acceleration = this quarter's YoY minus the previous quarter's YoY
        F["rev_growth_accel"] = fiscal_apply_to_daily(
            fund_hist, "revenue_q", idx,
            lambda s: s.pct_change(yoy_periods).diff(1))

    q_ni_yoy = fiscal_apply_to_daily(fund_hist, "netIncome_q", idx,
                                     lambda s: s.pct_change(yoy_periods))
    if q_ni_yoy.notna().any().any():
        F["q_earnings_growth"] = q_ni_yoy

    # latest-quarter margin vs TTM margin = margin inflection
    q_margin = ratio(daily("netIncome_q"), daily("revenue_q"), positive_den=True)
    profit_margins = daily("profitMargins")
    if not q_margin.empty and not profit_margins.empty:
        cols = q_margin.columns.intersection(profit_margins.columns)
        F["q_margin_vs_ttm"] = q_margin[cols] - profit_margins[cols]
    return F


def _yearly_ttm_momentum_fields(fund_hist: pd.DataFrame, idx: pd.DatetimeIndex,
                                yoy_periods: int) -> dict:
    """YEARLY-TTM momentum (current TTM vs TTM one year ago).

    Complements the single-quarter features: captures a multi-quarter trend rather than
    the most-recent quarter jolt, which is less noisy and works at longer horizons. Uses
    the fiscal history of level columns so seasonality is removed by construction (same
    quarter each year -> yoy_periods filings back)."""
    F: dict[str, pd.DataFrame] = {}
    y_rev_growth = fiscal_change_to_daily(fund_hist, "totalRevenue", idx,
                                          kind="pct", periods=yoy_periods)
    if y_rev_growth.notna().any().any():
        F["y_rev_growth"] = y_rev_growth
        F["y_rev_growth_accel"] = fiscal_apply_to_daily(
            fund_hist, "totalRevenue", idx,
            lambda s, n=yoy_periods: s.pct_change(n).diff(1))

    y_earnings_growth = fiscal_change_to_daily(fund_hist, "netIncome", idx,
                                               kind="pct", periods=yoy_periods)
    if y_earnings_growth.notna().any().any():
        F["y_earnings_growth"] = y_earnings_growth

    # YoY change in TTM profit margin (margin expansion / contraction trend)
    y_margin_chg = fiscal_change_to_daily(fund_hist, "profitMargins", idx,
                                          kind="diff", periods=yoy_periods)
    if y_margin_chg.notna().any().any():
        F["y_margin_vs_ttm"] = y_margin_chg
    return F


def _distress_fields(daily, ebitda: pd.DataFrame, cash: pd.DataFrame,
                     fcf: pd.DataFrame, long_debt: pd.DataFrame,
                     short_debt: pd.DataFrame,
                     operating_cash: frozenset[str] | None = None) -> dict:
    """DISTRESS / SOLVENCY: can the firm service and roll its debt?

    debtToEquity is a book ratio that says nothing about debt-SERVICING ability; these
    are the leverage / coverage / liquidity ratios credit desks watch. REFINANCING RISK
    is short-term debt / (cash + trailing free cash flow): HIGH (>>1) means the firm must
    roll a big slug of debt it cannot self-fund -> exposed to rate spikes / frozen credit
    markets.

    `operating_cash` reaches only the NET-DEBT line. `cash_to_debt` and `refinancing_risk`
    are cash RATIOS, not nettings: a liquidity cushion is a real fact about a bank, and
    those two are precisely the features the `cash` widening fix exists to restore (`cash`
    was NULL on 28.1% of Financials filings). Netting is the only operation where a bank's
    required reserves and interbank float would misstate the capital structure."""
    F: dict[str, pd.DataFrame] = {}
    total_debt = _combine_debt(daily, long_debt, short_debt, daily("totalLiabilities"))
    if not total_debt.empty and not ebitda.empty:
        net_cash = capital.drop_operating_cash(cash, operating_cash)
        cols = (total_debt.columns.intersection(net_cash.columns)
                if not net_cash.empty else total_debt.columns)
        net_debt = (total_debt[cols].sub(net_cash[cols], fill_value=0.0)
                    if not net_cash.empty else total_debt)
        # HIGH net-debt/EBITDA = more leveraged = worse (only meaningful for EBITDA>0)
        nd_ebitda = ratio(net_debt, ebitda, positive_den=True)
        if not nd_ebitda.empty and nd_ebitda.notna().any().any():
            F["net_debt_to_ebitda"] = nd_ebitda
    interest = daily("interestExpense")
    if not ebitda.empty and not interest.empty:
        # HIGH coverage = safer. Interest is an expense (take abs to be sign-safe).
        cov = ratio(ebitda, interest.abs(), positive_den=True)
        if not cov.empty and cov.notna().any().any():
            F["interest_coverage"] = cov
    current_ratio = ratio(daily("currentAssets"), daily("currentLiabilities"),
                          positive_den=True)
    if not current_ratio.empty and current_ratio.notna().any().any():
        F["current_ratio"] = current_ratio
    if not cash.empty and not total_debt.empty:
        cash_to_debt = ratio(cash, total_debt, positive_den=True)
        if not cash_to_debt.empty and cash_to_debt.notna().any().any():
            F["cash_to_debt"] = cash_to_debt

    liquidity = cash.add(fcf.where(fcf > 0), fill_value=0.0)
    refi = ratio(short_debt, liquidity, positive_den=True)
    if not refi.empty and refi.notna().any().any():
        F["refinancing_risk"] = refi
    return F


def _sga_efficiency_fields(daily, fund_hist: pd.DataFrame, idx: pd.DatetimeIndex,
                           revenue: pd.DataFrame, yoy_periods: int) -> dict:
    """MARKETING & SALES efficiency (operating leverage).

    `operating_leverage` = sales growing FASTER than selling cost (scalable); negative
    means growth is being "bought" with rising SG&A (margin risk ahead)."""
    F: dict[str, pd.DataFrame] = {}
    sga_intensity = ratio(daily("sellingGeneralAdmin"), revenue, positive_den=True)
    if not sga_intensity.empty and sga_intensity.notna().any().any():
        F["sga_intensity"] = sga_intensity
    sga_growth = fiscal_change_to_daily(fund_hist, "sellingGeneralAdmin", idx,
                                        kind="pct", periods=yoy_periods)
    rev_growth = fiscal_change_to_daily(fund_hist, "totalRevenue", idx,
                                        kind="pct", periods=yoy_periods)
    if sga_growth.notna().any().any():
        F["sga_growth"] = sga_growth
        if rev_growth.notna().any().any():
            cols = rev_growth.columns.intersection(sga_growth.columns)
            F["operating_leverage"] = rev_growth[cols] - sga_growth[cols]
    return F


def _ma_footprint_fields(daily, fund_hist: pd.DataFrame, idx: pd.DatetimeIndex,
                         revenue: pd.DataFrame, yoy_periods: int) -> dict:
    """M&A footprint: organic vs inorganic growth, and impairment risk."""
    F: dict[str, pd.DataFrame] = {}
    # Sharadar's `ncfbus`: net cash paid for acquisitions, stored outflow-negative, so the
    # magnitude is what "how acquisitive is this firm" means. 27,731 negative rows to 9,039
    # positive (a positive row is a net DISPOSAL year), hence `.abs()` rather than a floor:
    # both directions are M&A activity, and the intensity is unsigned by construction.
    acq = daily("businessAcquisitionsNet")
    assets = capital.assets_ex_lease(daily)
    acq_den = assets if not assets.empty else revenue
    acq_intensity = ratio(acq.abs() if not acq.empty else acq, acq_den, positive_den=True)
    if not acq_intensity.empty and acq_intensity.notna().any().any():
        F["acquisition_intensity"] = acq_intensity
    # YoY growth in the acquired-intangibles balance: the balance-sheet trace of M&A, and
    # the exposure a future writedown lands on. On Sharadar's COMBINED `intangibles` -- the
    # bare `goodwill` this read before is written by no producer, so it grew nothing.
    intangibles_growth = fiscal_change_to_daily(fund_hist, "intangibles", idx,
                                                kind="pct", periods=yoy_periods)
    if intangibles_growth.notna().any().any():
        F["intangibles_growth"] = intangibles_growth
    return F


def _sbc_fields(daily, revenue: pd.DataFrame, sbc: pd.DataFrame) -> dict:
    """STOCK-BASED COMPENSATION ("employee shares given").

    `shares_growth` is NET of buybacks and can be masked; SBC is the GROSS give-away, and
    `sbc_to_ocf` shows how much of reported operating cash flow is really non-cash comp."""
    F: dict[str, pd.DataFrame] = {}
    sbc_intensity = ratio(sbc, revenue, positive_den=True)
    if not sbc_intensity.empty and sbc_intensity.notna().any().any():
        F["sbc_intensity"] = sbc_intensity
    ocf = daily("operatingCashFlow")
    if not sbc.empty and not ocf.empty:
        sbc_to_ocf = ratio(sbc, ocf, positive_den=True)
        if not sbc_to_ocf.empty and sbc_to_ocf.notna().any().any():
            F["sbc_to_ocf"] = sbc_to_ocf
    return F


def _valuation_engine_fields(daily, fund_hist: pd.DataFrame, idx: pd.DatetimeIndex,
                             revenue: pd.DataFrame, ebitda: pd.DataFrame,
                             yoy_periods: int) -> dict:
    """REFINED VALUATION-ENGINE RATIOS (elasticity / divergence / dilution).

      operating_leverage_elasticity  %dOperatingIncome / %dRevenue (>1 = scalable model,
                                     exponential profit vs linear sales; <1 =
                                     diseconomies of scale). Distinct from
                                     `operating_leverage` (revenue - SG&A growth).
      margin_expansion_delta         gross margin expanding while EBITDA margin lags =
                                     losing SG&A/overhead control; both expanding = true
                                     pricing power.
      nwc_elasticity                 %dNWC / %dRevenue (>1 = cash-hungry growth).
      diluted_shares_growth          fully-diluted shareholder dilution rate.
      ebit_interest_coverage         EBIT / interest; LOW (< ~2x) = structural distress
                                     risk. Complements the EBITDA-based
                                     `interest_coverage`.

    The elasticities guard against a ~flat revenue denominator (require a >=2% move),
    which would otherwise make them meaningless and explode."""
    F: dict[str, pd.DataFrame] = {}
    oi_growth = fiscal_change_to_daily(fund_hist, "operatingIncome", idx,
                                       kind="pct", periods=yoy_periods)
    rev_growth_f = fiscal_change_to_daily(fund_hist, "totalRevenue", idx,
                                          kind="pct", periods=yoy_periods)
    ol_el = ratio(oi_growth, rev_growth_f.where(rev_growth_f.abs() >= 0.02))
    if not ol_el.empty and ol_el.notna().any().any():
        F["operating_leverage_elasticity"] = ol_el

    gm = ratio(daily("grossProfit"), revenue, positive_den=True)
    if gm.empty:
        gm = daily("grossMargins")
    em = ratio(ebitda, revenue, positive_den=True)
    if not gm.empty and not em.empty:
        med = (gm - gm.shift(252)) - (em - em.shift(252))     # ~1y change divergence
        if med.notna().any().any():
            F["margin_expansion_delta"] = sanitize(med)

    cur_a2, cur_l2 = daily("currentAssets"), daily("currentLiabilities")
    if not cur_a2.empty and not cur_l2.empty and not revenue.empty:
        nwc = cur_a2.sub(cur_l2, fill_value=np.nan)
        prev_nwc = nwc.shift(252)
        nwc_g = ratio(nwc - prev_nwc, prev_nwc, positive_den=True)
        prev_rev = revenue.shift(252)
        rev_g = ratio(revenue - prev_rev, prev_rev, positive_den=True)
        nwc_el = ratio(nwc_g, rev_g.where(rev_g.abs() >= 0.02))
        if not nwc_el.empty and nwc_el.notna().any().any():
            F["nwc_elasticity"] = nwc_el

    dil_growth = fiscal_change_to_daily(fund_hist, "dilutedShares", idx,
                                        kind="pct", periods=yoy_periods)
    if dil_growth.notna().any().any():
        F["diluted_shares_growth"] = dil_growth

    eic = ratio(daily("operatingIncome"), daily("interestExpense").abs(), positive_den=True)
    if not eic.empty and eic.notna().any().any():
        F["ebit_interest_coverage"] = eic
    return F


def _intrinsic_fields(fund_hist: pd.DataFrame, close: pd.DataFrame | None,
                      idx: pd.DatetimeIndex, intrinsic_cfg: dict | None,
                      level_factor: pd.DataFrame | None = None) -> dict:
    """INTRINSIC VALUE (two-stage DCF on TTM FCF) vs price. `intrinsic_cfg` is optional --
    every caller up the chain defaults it to None, so fall back to `intrinsic_value_daily`'s
    own documented DCF defaults rather than splatting None."""
    if close is None:
        return {}
    iy = intrinsic_value_daily(fund_hist, close, idx, level_factor=level_factor,
                               **(intrinsic_cfg or {})).get("yield")
    if iy is None or iy.empty or not iy.notna().any().any():
        return {}
    return {"intrinsic_yield": iy}


# --------------------------------------------------------------------------- #
# Characteristics                                                              #
# --------------------------------------------------------------------------- #
def _derived_fields(
    fund_hist: pd.DataFrame,
    idx: pd.DatetimeIndex,
    close: pd.DataFrame | None,
    yoy_periods: int = 1,
    intrinsic_cfg: dict | None = None,
    earnings_history: pd.DataFrame | None = None,
    pension_facts: pd.DataFrame | None = None,
    notes_num: pd.DataFrame | None = None,
    level_factor: pd.DataFrame | None = None,
) -> dict:
    """Build daily wide frames (date x ticker) for every characteristic.

    A thin composition of the per-block builders above: this function owns only the
    frames that are genuinely SHARED between blocks (the memoized `daily` accessor, the
    handful of hoisted balance-sheet levels, the market cap, the pension pool and the
    enterprise value) and the order the blocks run in.

    `yoy_periods` is how many filings span a year (4 for quarterly history, 1 for
    annual); growth/trend features use it for a true YoY comparison. `intrinsic_cfg`
    overrides the DCF parameters for the intrinsic-value yield.
    """

    F: dict[str, pd.DataFrame] = {}
    _daily_cache: dict[str, pd.DataFrame] = {}

    def daily(field):
        # memoized: several fields are reused across blocks (and the business-quality
        # helpers), so cache the pivot+ffill instead of recomputing it.
        if field not in _daily_cache:
            _daily_cache[field] = fundamentals_to_daily(fund_hist, field, idx)
        return _daily_cache[field]

    # ---- frames shared by MORE THAN ONE block (hoisted once) ---- #
    revenue = daily("totalRevenue")
    net_income = daily("netIncome")
    fcf = daily("freeCashflow")
    equity = daily("stockholdersEquity")
    ebitda = daily("ebitda")
    d2e = daily("debtToEquity")
    rnd = daily("researchAndDevelopment")
    cash = daily("cash")
    long_debt = daily("longTermDebt")
    short_debt = daily("shortTermDebt")
    sbc = daily("stockBasedComp")

    # pension/OPEB overhang: three frames feeding the EV, forensic and scale blocks.
    #
    # ⚠ SCOPED TO THE FUNDAMENTALS UNIVERSE FIRST. `pension_facts` and `notes_num` are bulk
    # SEC tables keyed on their own filer set, and the helpers below pivot them WIDE: every
    # ticker they carry becomes a column, and the first arithmetic against a fundamentals
    # frame aligns on the UNION. Unscoped, that grafts filers with no balance sheet into the
    # panel -- rows carrying a pension ratio and nothing else, since every other feature
    # needs a fundamentals row to divide by. Measured against the live tables: 4 tickers in
    # `pension_facts` and 7 in `notes_num` are absent from `fundamentals_history`, and on a
    # 10-ticker slice the two tables alone widened the panel to 279.
    universe = set(fund_hist["ticker"].astype(str)) if "ticker" in fund_hist.columns else set()
    pension_facts = _scope_to_universe(pension_facts, universe)
    notes_num = _scope_to_universe(notes_num, universe)

    pbo, fn_deficit, pension_ret = _pension_pool(notes_num, pension_facts, idx)
    F.update(_pension_health_fields(daily, pbo, notes_num, idx, pension_ret))

    # Tickers whose `cash` is a CORE OPERATING ASSET, not spare cash: bank required
    # reserves / interbank float and insurance claims float. They are excluded from every
    # site that NETS cash (EV, net debt, invested capital) and from none that ratios it.
    #
    # This became load-bearing when `cash` was widened to `cashneq + coalesce(investmentsc,
    # 0)`. Before that, banks and insurers had a NULL cash -- an unclassified balance sheet
    # (ASC 210-10-05-4) reports no current-investments line -- so the netting sites already
    # netted nothing for them. Restoring it without this gate would have silently re-rated
    # every bank: measured, median `cashneq` is 10.97% of market cap for Financials and
    # 101.95% at p90, so the top decile's "cash" exceeds its entire equity value and EV
    # would turn NEGATIVE, nulling a feature that has a value today.
    operating_cash = frozenset(family_tickers(fund_hist, "bank")
                               | family_tickers(fund_hist, "insurance"))

    # ---- everything that needs a daily market cap ---- #
    # `close` is Optional all the way down this module (`_intrinsic_fields` returns early on
    # None, `_enterprise_value_frame` and the earnings-yield block both test it), but
    # `daily_market_cap` declares a non-Optional frame and reads `close_split.index` on its
    # first line. An empty frame on `idx` is the same answer `PitCache.market_cap` gives when
    # either input is absent, and every downstream block already handles an empty mcap.
    mcap = (pd.DataFrame(index=idx) if close is None or close.empty
            else daily_market_cap(fund_hist, close, level_factor=level_factor))

    # Every field below is a yield or a multiple, so all of them are undefined without a
    # price. They are SKIPPED rather than emitted empty, because a key carrying an empty
    # frame becomes an all-NaN column downstream and reads as "this company has no earnings
    # yield" instead of "there was no price to compute one" -- the same distinction
    # `_intrinsic_fields` already makes by returning `{}` when `close is None`.
    if not mcap.empty:
        ev = _enterprise_value_frame(daily, close, mcap, equity, d2e, cash, pension_ret,
                                     operating_cash)
        F.update(_valuation_yield_fields(mcap, net_income, revenue, equity, fcf))
        F.update(_pension_scale_fields(pension_ret, pbo, mcap))
        F.update(_ev_yield_fields(ebitda, fcf, ev))
        F.update(_altman_z_fields(daily, mcap, revenue))
        F.update(_pegy_fields(daily, fund_hist, idx, mcap, net_income, yoy_periods,
                              earnings_history))
        F.update(_reit_multiple_fields(daily, fund_hist, net_income, mcap, ev))

    # ---- price-independent blocks ---- #
    F.update(_profitability_level_fields(daily, revenue, net_income, fcf))
    F.update(_growth_trend_fields(daily, fund_hist, idx, revenue, rnd, yoy_periods))
    F.update(_reinvestment_fields(daily, fund_hist, idx, yoy_periods))
    F.update(_quality_regime_fields(daily, fund_hist, idx, revenue, net_income, fcf,
                                    long_debt, yoy_periods))
    F.update(_state_flag_fields(daily, net_income, fcf, equity))
    F.update(_quarter_momentum_fields(daily, fund_hist, idx, yoy_periods))
    F.update(_yearly_ttm_momentum_fields(fund_hist, idx, yoy_periods))
    F.update(_distress_fields(daily, ebitda, cash, fcf, long_debt, short_debt,
                              operating_cash))
    F.update(_sga_efficiency_fields(daily, fund_hist, idx, revenue, yoy_periods))
    F.update(_ma_footprint_fields(daily, fund_hist, idx, revenue, yoy_periods))
    F.update(_sbc_fields(daily, revenue, sbc))
    F.update(_valuation_engine_fields(daily, fund_hist, idx, revenue, ebitda, yoy_periods))
    F.update(_intrinsic_fields(fund_hist, close, idx, intrinsic_cfg, level_factor))

    # ---- BUSINESS-QUALITY blocks ---- #
    #   SBC-vs-buyback, #5 forensic red flags, #3 M&A digestion, per-share economics.
    #
    # The CORE/ADJUSTED-EARNINGS family that used to run here is gone. It normalized profit
    # for non-recurring items over ten tags (`impairment`, `restructuring`,
    # `gainOnDispositions`, `litigationExpense`, `unusualItems`, ...) and Sharadar delivers
    # NONE of them -- the only non-recurring item SF1 isolates is `netIncomeDiscontinued`.
    # It emitted nothing while looking fully implemented, which is worse than absent. A true
    # core-earnings series requires the SEC path; the 72 rows where `netIncome > totalRevenue`
    # (KKR to 14.1x, APO, ARES, BX -- all real, not extraction bugs) are exactly the
    # observations it would neutralise, and nothing does.
    F.update(_da_realism_fields(daily))
    F.update(_forensic_fields(daily, idx))
    F.update(_digestion_fields(daily, fund_hist, idx, yoy_periods, operating_cash))
    F.update(_per_share_and_profit_slice_fields(daily, fund_hist, idx, yoy_periods, close))

    return F


def _merge_feature_panels(panels: list[pd.DataFrame]) -> pd.DataFrame:
    """Outer-merge the non-empty long panels on ['date','ticker']."""
    out = None
    for p in panels:
        if p is None or p.empty or list(p.columns) == ["date", "ticker"]:
            continue
        out = p if out is None else out.merge(p, on=["date", "ticker"], how="outer")
    return out if out is not None else pd.DataFrame(columns=["date", "ticker"])


def build_state_panel(fields: dict) -> pd.DataFrame:
    """Stack raw 0/1 regime flags into long `f_<name>` columns -- absolute state
    indicators the model conditions on, NOT peer-standardized."""
    if not fields:
        return pd.DataFrame(columns=["date", "ticker"])
    long_frames = []
    for name, fdf in fields.items():
        if fdf is None or fdf.empty:
            continue
        s = fdf.stack().astype("float32")
        s.index.set_names(["date", "ticker"], inplace=True)
        long_frames.append(s.rename(f"f_{name}"))
    if not long_frames:
        return pd.DataFrame(columns=["date", "ticker"])
    # .copy() consolidates the many single-column blocks that concat(axis=1) doesn't 
        # trip the "highly fragmented DataFrame" PerformanceWarning
    return pd.concat(long_frames, axis=1).copy().reset_index()


def build_self_history_panel(fields: dict) -> pd.DataFrame:
    """Stack already-z-scored self-history frames into long `f_<name>_vs_hist`
    columns. The input frames are the OUTPUT of `_self_history_z` (final signal),
    so they are NOT re-standardized cross-sectionally the way peer features are."""
    if not fields:
        return pd.DataFrame(columns=["date", "ticker"])
    long_frames = []
    for name, zdf in fields.items():
        if zdf is None or zdf.empty:
            continue
        s = zdf.stack().astype("float32")
        s.index.set_names(["date", "ticker"], inplace=True)
        long_frames.append(s.rename(f"f_{name}_vs_hist"))
    if not long_frames:
        return pd.DataFrame(columns=["date", "ticker"])
    
    # .copy() consolidates the many single-column blocks that concat(axis=1) doesn't 
    # trip the "highly fragmented DataFrame" PerformanceWarning
    return pd.concat(long_frames, axis=1).copy().reset_index()


def build_fundamental_feature_panel(
    fundamentals_history: pd.DataFrame | None,
    peer_dict: dict,
    trading_index: pd.DatetimeIndex,
    stock_close: pd.DataFrame | None = None,
    intrinsic_cfg: dict | None = None,
    hist_window: int = _HIST_WINDOW,
    hist_min_periods: int = _HIST_MIN_PERIODS,
    earnings_history: pd.DataFrame | None = None,
    pension_facts: pd.DataFrame | None = None,
    notes_num: pd.DataFrame | None = None,
    level_factor: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Long-format panel: ['date','ticker', f_<char>_vs_peers, f_<char>_xs,
    f_<yield>_vs_hist, ...].

    Three complementary views:
      * f_<char>_vs_peers : firm minus its direct competitors (peer basket),
                            standardized -> the firm-specific edge.
      * f_<char>_xs       : cross-sectional percentile across the whole universe.
      * f_<yield>_vs_hist : each valuation yield versus the firm's OWN trailing
                            history (a z-score over `hist_window` days) -> the
                            time-series valuation mean-reversion signal ("cheap
                            vs its own past", e.g. PE below its 5y average).

    `stock_close` is required for the valuation features (daily market cap);
    without it valuation is skipped but every other feature is still built.
    Empty frame if no fundamentals available.
    """
    
    if fundamentals_history is None or fundamentals_history.empty:
        raise ValueError("Need to build fundamentals_history table first")

    yoy_periods = infer_yoy_periods(fundamentals_history)
    fields = _derived_fields(fund_hist=fundamentals_history, 
                             idx=trading_index, 
                             close=stock_close,
                             yoy_periods=yoy_periods, 
                             intrinsic_cfg=intrinsic_cfg,
                             earnings_history=earnings_history, 
                             pension_facts=pension_facts,
                             notes_num=notes_num,
                             level_factor=level_factor)

    # float32 to reduce space vs float 64, no need of too much detail since z scored or ranked
    fields = {k: (v.astype("float32") if isinstance(v, pd.DataFrame) and not v.empty else v)
              for k, v in fields.items()}

    # regime state flags -> RAW `f_<name>`; everything else -> peer-relative.
    state_fields = {k: v for k, v in fields.items() if k in _STATE_FIELDS}
    peer_fields = {k: v for k, v in fields.items() if k not in _STATE_FIELDS}

    peer_panel = build_peer_relative_panel(peer_fields, peer_dict)

    # Self-history (mean-reversion) z-scores on the valuation yields only.
    hist_fields = {
        name: _self_history_z(fields[name], window=hist_window, min_periods=hist_min_periods)
        for name in _MEAN_REVERSION_FIELDS
        if name in fields and fields[name] is not None and not fields[name].empty
    }
    hist_panel = build_self_history_panel(hist_fields)
    state_panel = build_state_panel(state_fields)

    return _merge_feature_panels([peer_panel, hist_panel, state_panel])
