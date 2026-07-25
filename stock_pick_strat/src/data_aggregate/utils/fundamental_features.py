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
from sqlalchemy import bindparam, text

from src.context import Context
from src.data_aggregate.utils.factors import fundamentals_to_daily, daily_market_cap
from src.data_aggregate.utils.intrinsic import intrinsic_value_daily

_PENSION_FACTS_TABLE = "pension_facts"     # bulk Financial-Statement-Data-Sets pension facts (literal)
_NOTES_NUM_TABLE = "notes_num"             # footnote NUMERIC facts (10 tags; the panel uses 2)
_FACT_COLS = ["ticker", "tag", "ddate", "qtrs", "value", "filed"]   # the only cols the pension builders read


# Valuation yields that also get a self-history (mean-reversion) z-score, i.e.
# "cheap vs its OWN past" in addition to "cheap vs peers". High = cheaper than
# the firm's own norm -> classic valuation mean-reversion signal.
_MEAN_REVERSION_FIELDS = (
    "earnings_yield", "sales_yield", "book_yield",
    "fcf_yield", "ebitda_to_ev", "fcf_to_ev", "ffo_yield", "intrinsic_yield",
    "core_earnings_yield",   # adjusted E/P also gets a "cheap vs its own past" z
)
_HIST_WINDOW = 1260      # ~5 trading years of daily observations
_HIST_MIN_PERIODS = 252  # require >= 1y of history before emitting a z-score

# Cross-sectional winsorization for the Z-SCORE features (peer-z + self-history z):
# clip each day's distribution to its [1%, 99%] percentiles so a few extreme names
# can't dominate the standardized value -> better generalization. The percentile-
# RANK features (`_xs`) are already outlier-proof and are left untouched.
_WINSOR_LO, _WINSOR_HI = 0.01, 0.99


def _winsorize_xs(df: pd.DataFrame, lo: float = _WINSOR_LO,
                  hi: float = _WINSOR_HI) -> pd.DataFrame:
    """Clip each ROW (date) to its cross-sectional [lo, hi] quantiles across tickers.
    NaN-safe: an all-NaN row yields NaN bounds -> clip is a no-op there."""
    if df is None or df.empty:
        return df
    lo_q = df.quantile(lo, axis=1)
    hi_q = df.quantile(hi, axis=1)
    return df.clip(lower=lo_q, upper=hi_q, axis=0)

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
def _ratio(num: pd.DataFrame, den: pd.DataFrame, positive_den: bool = False) -> pd.DataFrame:
    """Column-aligned num/den on the common tickers, inf -> NaN. When
    `positive_den` the denominator is masked to strictly-positive values."""
    if num.empty or den.empty:
        return pd.DataFrame()
    cols = num.columns.intersection(den.columns)
    if len(cols) == 0:
        return pd.DataFrame()
    d = den[cols]
    d = d.where(d > 0) if positive_den else d.where(d != 0)
    out = num[cols] / d
    return out.replace([np.inf, -np.inf], np.nan)


def _combine_debt(long_debt: pd.DataFrame, short_debt: pd.DataFrame,
                  fallback: pd.DataFrame) -> pd.DataFrame:
    """Total interest-bearing debt = long-term + short-term, NaN-tolerant
    (a firm with only one of the two keeps that one). Falls back to total
    liabilities when neither debt tag is present, so leverage is still defined."""
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


def _self_history_z(field_df: pd.DataFrame, window: int = _HIST_WINDOW,
                    min_periods: int = _HIST_MIN_PERIODS, clip: float = 8.0) -> pd.DataFrame:
    """Time-series z-score of each ticker versus its OWN trailing `window`:

        z(t) = (x(t) - trailing_mean(t)) / trailing_std(t)

    The rolling window is trailing (right-edge = today), so it uses only
    current-and-past values -- strictly point-in-time, no look-ahead. On a
    valuation YIELD, a high z means the firm is currently cheaper than its own
    historical norm (mean-reversion long side). Winsorized to +-`clip`."""
    if field_df is None or field_df.empty:
        return pd.DataFrame()
    mean = field_df.rolling(window, min_periods=min_periods).mean()
    std = field_df.rolling(window, min_periods=min_periods).std()
    z = (field_df - mean) / std.where(std > 0)
    z = z.clip(-clip, clip).replace([np.inf, -np.inf], np.nan)
    return _winsorize_xs(z)            # trim per-day cross-sectional 1%/99% outliers


def _infer_yoy_periods(fund_hist: pd.DataFrame) -> int:
    """Number of filing periods that make up one year, from the median gap
    between consecutive `as_of` dates. Quarterly history -> 4, annual -> 1.
    Used so growth is always a true year-over-year comparison (no seasonality)
    regardless of the reporting cadence."""
    if "as_of" not in fund_hist.columns or fund_hist.empty:
        return 1
    d = fund_hist[["ticker", "as_of"]].copy()
    d["as_of"] = pd.to_datetime(d["as_of"], errors="coerce")
    gaps = d.sort_values(["ticker", "as_of"]).groupby("ticker")["as_of"].diff().dt.days
    med = gaps.median()
    if not np.isfinite(med) or med <= 0:
        return 1
    return int(min(4, max(1, round(365.0 / med))))


def _fiscal_change_to_daily(
    fund_hist: pd.DataFrame,
    field: str,
    idx: pd.DatetimeIndex,
    kind: str = "pct",
    periods: int = 1,
) -> pd.DataFrame:
    """Change of a fiscal field over `periods` filings, forward-filled onto
    trading days. With `periods` = one year of filings this is a seasonality-free
    year-over-year change.

    Computed per ticker on ITS OWN fiscal series (ordered by filing date), then
    ffilled point-in-time so the change lands on the day the new filing is
    public. `kind='pct'` -> relative growth; `kind='diff'` -> absolute change
    (use for ratios like margins).
    """

    if field not in fund_hist.columns:
        return pd.DataFrame(index=idx)
    df = fund_hist[["ticker", "as_of", field]].copy()
    df["as_of"] = pd.to_datetime(df["as_of"])
    df[field] = pd.to_numeric(df[field], errors="coerce")
    df = df.dropna(subset=[field]).sort_values(["ticker", "as_of"])
    if df.empty:
        return pd.DataFrame(index=idx)

    grp = df.groupby("ticker")[field]
    if kind == "pct":
        df["chg"] = grp.pct_change(periods=periods)
    elif kind == "diff":
        df["chg"] = grp.diff(periods=periods)
    else:
        raise ValueError("kind must be 'pct' or 'diff'")

    wide = df.pivot_table(index="as_of", columns="ticker", values="chg", aggfunc="last")
    wide = wide.replace([np.inf, -np.inf], np.nan).sort_index()
    return wide.reindex(wide.index.union(idx)).ffill().reindex(idx)


def _fiscal_apply_to_daily(fund_hist, field, idx, func) -> pd.DataFrame:
    """Apply a per-ticker series transform (e.g. YoY growth, or acceleration =
    change in YoY) to a fiscal field, forward-filled point-in-time onto trading
    days. `func` receives one ticker's chronological series and returns a series
    of the same length."""
    if field not in fund_hist.columns:
        return pd.DataFrame(index=idx)
    df = fund_hist[["ticker", "as_of", field]].copy()
    df["as_of"] = pd.to_datetime(df["as_of"])
    df[field] = pd.to_numeric(df[field], errors="coerce")
    df = df.dropna(subset=[field]).sort_values(["ticker", "as_of"])
    if df.empty:
        return pd.DataFrame(index=idx)
    df["v"] = df.groupby("ticker")[field].transform(func)
    wide = df.pivot_table(index="as_of", columns="ticker", values="v", aggfunc="last")
    wide = wide.replace([np.inf, -np.inf], np.nan).sort_index()
    return wide.reindex(wide.index.union(idx)).ffill().reindex(idx)


# --------------------------------------------------------------------------- #
# Business-quality helpers (all from tags ALREADY extracted -- no new SEC pull)
#   #2 D&A/SBC realism, #5 forensic, #3 M&A digestion, #1 core/adjusted earnings.
# Each returns a {name: daily wide frame} dict that _derived_fields merges into F,
# so every field auto-expands to f_<name>_vs_peers + f_<name>_xs downstream.
# `daily` is the memoized accessor from _derived_fields (field -> date x ticker).
# --------------------------------------------------------------------------- #
_YEAR = 252   # trailing trading days ~= one calendar year (year-ago comparison)
_FIVE_YEARS = 5 * _YEAR   # ~5 trading years (multi-year trend comparison)


def _effective_tax_rate(daily, default: float = 0.21) -> pd.DataFrame:
    """Point-in-time effective tax rate = tax / pretax, clipped to [0, 0.5] and
    defaulted to the US statutory ~21% where missing/nonsensical. Used to tax-
    adjust special items (core earnings) and to build NOPAT (ROIC)."""
    tax, pre = daily("incomeTaxExpense"), daily("pretaxIncome")
    if tax.empty or pre.empty:
        return pd.DataFrame()
    return _ratio(tax, pre.where(pre > 0)).clip(0.0, 0.5).fillna(default)


def _da_realism_fields(daily) -> dict:
    """#2 -- is reported depreciation believable given the asset base? Extending
    useful lives (a jump in implied life) or an aging base (high accumulated /
    gross PP&E) are classic earnings-quality tells the aggregate D&A line hides.
    `sbc_to_buyback` flags buybacks that merely offset option dilution."""
    F: dict[str, pd.DataFrame] = {}
    depamort = daily("depAmort")
    amort_intang = daily("amortizationIntangibles")
    ppe_gross = daily("ppeGross")
    accum_dep = daily("accumulatedDepreciation")
    # PP&E depreciation = D&A minus intangible amortization (fall back to full D&A
    # when the split is absent) so the useful-life read is about hard assets.
    depreciation = (depamort.sub(amort_intang, fill_value=0.0)
                    if not amort_intang.empty else depamort)

    if not ppe_gross.empty and not depreciation.empty:
        life = _ratio(ppe_gross, depreciation, positive_den=True)
        if life.notna().any().any():
            F["implied_useful_life"] = life
            luc = (life - life.shift(_YEAR)).replace([np.inf, -np.inf], np.nan)
            if luc.notna().any().any():
                F["useful_life_change"] = luc     # jump UP = lives extended = red flag
    if not accum_dep.empty and not ppe_gross.empty:
        age = _ratio(accum_dep, ppe_gross, positive_den=True)
        if age.notna().any().any():
            F["asset_age"] = age                  # high = old base -> capex catch-up ahead
    if not amort_intang.empty and not depamort.empty:
        ias = _ratio(amort_intang, depamort, positive_den=True)
        if ias.notna().any().any():
            F["intangible_amortization_share"] = ias
    sbc, buyback = daily("stockBasedComp"), daily("buybacks")
    if not sbc.empty and not buyback.empty:
        s2b = _ratio(sbc, buyback.abs(), positive_den=True)
        if s2b.notna().any().any():
            F["sbc_to_buyback"] = s2b             # >1 = buybacks don't even cover SBC
    return F


def _beneish_m_score(daily, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Beneish (1999) 8-variable earnings-manipulation model, each index computed
    as this year vs one year ago (trailing 252 trading days). Higher M = more
    likely a manipulator (the classic screen is M > -1.78). Missing index ->
    neutral (1.0; TATA -> 0), so M is defined wherever revenue+assets exist."""
    rev, assets = daily("totalRevenue"), daily("totalAssets")
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
        return _ratio(cur, prev.where(prev != 0))

    ar_sales = _ratio(ar, rev)
    dsri = ix(ar_sales, ar_sales.shift(_YEAR))                 # days sales in receivables
    gm = _ratio(gp, rev, positive_den=True)
    gmi = ix(gm.shift(_YEAR), gm)                              # gross-margin deterioration
    noncore = 1.0 - _ratio(ca.add(ppe, fill_value=0.0), assets, positive_den=True)
    aqi = ix(noncore, noncore.shift(_YEAR))                    # asset-quality (soft assets)
    sgi = ix(rev, rev.shift(_YEAR))                            # sales growth
    deprate = _ratio(dep, dep.add(ppe, fill_value=0.0), positive_den=True)
    depi = ix(deprate.shift(_YEAR), deprate)                   # slowing depreciation
    sgar = _ratio(sga, rev, positive_den=True)
    sgai = ix(sgar, sgar.shift(_YEAR))                         # SG&A efficiency
    lev = _ratio(ltd.add(cl, fill_value=0.0), assets, positive_den=True)
    lvgi = ix(lev, lev.shift(_YEAR))                           # leverage change
    tata = _ratio(ni.sub(ocf, fill_value=np.nan), assets)     # total accruals / assets

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
    return m.replace([np.inf, -np.inf], np.nan)


_NET_PENSION_TAGS = (
    "PensionAndOtherPostretirementDefinedBenefitPlansLiabilitiesNoncurrent",  # primary
    "DefinedBenefitPensionPlanLiabilitiesNoncurrent",                          # variant
)

# Footnote pension tags from the Financial Statement & NOTES sets (`notes_num`):
# the projected benefit obligation and fair value of plan assets, both INSTANT
# (balance-type) facts the primary statements never expose.
_FN_PBO_TAG = "DefinedBenefitPlanBenefitObligation"          # projected benefit obligation
_FN_PLAN_ASSETS_TAG = "DefinedBenefitPlanFairValueOfPlanAssets"


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
    Engine-side `WHERE tag IN (…)` when DB-backed, else a projected full read filtered in pandas.
    None if the table is absent/empty or no row matches."""
    cols = columns or _FACT_COLS
    store = context.store
    if hasattr(store, "exists") and not store.exists(table):
        return None
    engine = getattr(store, "engine", None)
    if engine is not None:
        sel = ", ".join(f'"{c}"' for c in cols)
        sql = text(f'SELECT {sel} FROM "{table}" WHERE tag IN :tags'
                   ).bindparams(bindparam("tags", expanding=True))
        with engine.connect() as conn:
            df = pd.read_sql(sql, conn, params={"tags": list(tags)})
    else:
        df = store.load(table, columns=cols)
        if df is not None and not df.empty and "tag" in df.columns:
            df = df[df["tag"].isin(tags)]
    return df.reset_index(drop=True) if df is not None and not df.empty else None


def load_pension_facts_scoped(context: Context) -> pd.DataFrame | None:
    """`pension_facts` restricted to the recognized net-liability tags the panel reads."""
    return load_tagged_facts(context, _PENSION_FACTS_TABLE, _NET_PENSION_TAGS)


def load_notes_num_scoped(context: Context) -> pd.DataFrame | None:
    """`notes_num` restricted to the footnote PBO + plan-asset tags the panel reads."""
    return load_tagged_facts(context, _NOTES_NUM_TABLE, (_FN_PBO_TAG, _FN_PLAN_ASSETS_TAG))


def _forensic_fields(daily, idx: pd.DatetimeIndex,
                     pension_deficit: pd.DataFrame | None = None) -> dict:
    """#5 -- accounting-quality / hidden-leverage red flags: working-capital days
    and their year-over-year drift (supplier-funded growth = rising DPO; channel
    stuffing = rising DSO), off-balance-sheet-INCLUSIVE net leverage (adds lease
    liabilities + pension deficit), and the Beneish M-score. All inputs already
    extracted; `pension_deficit` is the bulk Financial-Statement-Data-Sets frame."""
    F: dict[str, pd.DataFrame] = {}
    rev, cogs = daily("totalRevenue"), daily("costOfRevenue")
    ar, ap, inv = daily("accountsReceivable"), daily("accountsPayable"), daily("inventory")

    dso = _ratio(ar, rev, positive_den=True) * 365.0
    if dso.notna().any().any():
        F["dso"] = dso
        F["dso_change"] = (dso - dso.shift(_YEAR)).replace([np.inf, -np.inf], np.nan)
    dpo = _ratio(ap, cogs, positive_den=True) * 365.0
    if dpo.notna().any().any():
        F["dpo"] = dpo
        F["dpo_change"] = (dpo - dpo.shift(_YEAR)).replace([np.inf, -np.inf], np.nan)
    dio = _ratio(inv, cogs, positive_den=True) * 365.0
    if dio.notna().any().any():
        F["dio"] = dio
    if "dso" in F and "dpo" in F and "dio" in F:
        ccc = F["dso"].add(F["dio"], fill_value=np.nan).sub(F["dpo"], fill_value=np.nan)
        if ccc.notna().any().any():
            F["cash_conversion_cycle"] = ccc

    # off-BS-inclusive net leverage = (total debt + lease liabilities + pension deficit
    # - cash) / EBITDA. Pension deficit = underfunding only (negative funded status);
    # pension is absent until the tag is extracted, so this reduces to debt+leases then.
    debt = _combine_debt(daily("longTermDebt"), daily("shortTermDebt"), daily("totalLiabilities"))
    leases = daily("operatingLeaseLiability").add(daily("financeLeaseLiability"), fill_value=0.0)
    cash, ebitda = daily("cash"), daily("ebitda")
    if not debt.empty and not ebitda.empty:
        net_od = debt.add(leases, fill_value=0.0)
        # bulk Financial-Statement-Data-Sets pension deficit preferred (universe-wide);
        # the companyfacts `pensionDeficit` column fills any remaining gaps.
        pension = daily("pensionDeficit")
        if pension_deficit is not None and not pension_deficit.empty:
            pension = (pension_deficit.combine_first(pension) if not pension.empty
                       else pension_deficit)
        if not pension.empty:
            net_od = net_od.add(pension.clip(lower=0.0), fill_value=0.0)
        if not cash.empty:
            net_od = net_od.sub(cash, fill_value=0.0)
        nlev = _ratio(net_od, ebitda, positive_den=True)
        if nlev.notna().any().any():
            F["net_debt_incl_offbs_to_ebitda"] = nlev

    m = _beneish_m_score(daily, idx)
    if not m.empty and m.notna().any().any():
        F["beneish_m_score"] = m
    return F


def _digestion_fields(daily, fund_hist: pd.DataFrame, idx: pd.DatetimeIndex,
                      yoy_periods: int) -> dict:
    """#3 -- is bought growth being digested? ROIC WITH vs WITHOUT goodwill (the
    wedge = how much acquisitions drag returns), goodwill+intangibles balance-
    sheet weight (writedown exposure), and SG&A elasticity to revenue (<1 =
    synergies captured, ~1 = bolt-on with no integration)."""
    F: dict[str, pd.DataFrame] = {}
    oi, assets = daily("operatingIncome"), daily("totalAssets")
    equity, cash = daily("stockholdersEquity"), daily("cash")
    goodwill, intang = daily("goodwill"), daily("intangiblesExGoodwill")
    tax = _effective_tax_rate(daily)
    debt = _combine_debt(daily("longTermDebt"), daily("shortTermDebt"), pd.DataFrame())

    if not oi.empty and not equity.empty:
        nopat = oi * (1.0 - tax) if not tax.empty else oi
        ic = equity.copy()
        if not debt.empty:
            ic = ic.add(debt, fill_value=0.0)
        if not cash.empty:
            ic = ic.sub(cash, fill_value=0.0)
        roic_incl = _ratio(nopat, ic, positive_den=True)
        if roic_incl.notna().any().any():
            F["roic_incl_goodwill"] = roic_incl
            ic_ex = ic
            if not goodwill.empty:
                ic_ex = ic_ex.sub(goodwill, fill_value=0.0)
            if not intang.empty:
                ic_ex = ic_ex.sub(intang, fill_value=0.0)
            roic_ex = _ratio(nopat, ic_ex, positive_den=True)
            if roic_ex.notna().any().any():
                F["roic_ex_goodwill"] = roic_ex
                # incl - ex < 0 => goodwill/intangibles dilute returns (overpaid)
                F["goodwill_roic_drag"] = roic_incl.sub(roic_ex, fill_value=np.nan)

    if not goodwill.empty and not assets.empty:
        gi = goodwill.add(intang, fill_value=0.0)
        gta = _ratio(gi, assets, positive_den=True)
        if gta.notna().any().any():
            F["goodwill_intangibles_to_assets"] = gta
        gte = _ratio(goodwill, equity.where(equity > 0))
        if gte.notna().any().any():
            F["goodwill_to_equity"] = gte     # >1 => a writedown can wipe out book equity
        gw_imp = daily("goodwillImpairment")  # absent until split-out tag is extracted
        if not gw_imp.empty:
            gii = _ratio(gw_imp, assets, positive_den=True)
            if gii.notna().any().any():
                F["goodwill_impairment_intensity"] = gii   # writedown = overpayment admitted

    sga_g = _fiscal_change_to_daily(fund_hist, "sellingGeneralAdmin", idx, kind="pct", periods=yoy_periods)
    rev_g = _fiscal_change_to_daily(fund_hist, "totalRevenue", idx, kind="pct", periods=yoy_periods)
    if sga_g.notna().any().any() and rev_g.notna().any().any():
        el = _ratio(sga_g, rev_g.where(rev_g.abs() >= 0.02))   # guard ~flat-revenue blow-ups
        if el.notna().any().any():
            F["sga_elasticity"] = el
    return F


def _core_earnings_fields(daily, mcap: pd.DataFrame) -> dict:
    """#1 -- normalize earnings for non-recurring items (impairment + restructuring
    added back, gains on disposition removed) -> CORE margins/yield kept ALONGSIDE
    the reported ones (both versions live in the cube). `nonrecurring_pretax_share`
    is how transitory the quarter's profit is. The special-items pool widens once
    litigation / discontinued-ops / unusual tags are extracted (see backlog)."""
    F: dict[str, pd.DataFrame] = {}
    rev = daily("totalRevenue")
    if rev.empty:
        return F
    pretax, ni = daily("pretaxIncome"), daily("netIncome")
    oi, ebitda = daily("operatingIncome"), daily("ebitda")
    impair, restr, gains = daily("impairment"), daily("restructuring"), daily("gainOnDispositions")

    charges = impair.add(restr, fill_value=0.0)          # >=0 expense add-backs
    litig = daily("litigationExpense")                   # positive charge (add back); absent pre-refetch
    if not litig.empty:
        charges = charges.add(litig, fill_value=0.0)
    # signed gains removed from core (gain +, loss -): disposals, bargain purchase, net unusual
    for extra in ("gainOnSaleGeneric", "bargainPurchaseGain", "unusualItems"):
        g = daily(extra)
        if not g.empty:
            gains = gains.add(g, fill_value=0.0) if not gains.empty else g
    special = charges.sub(gains, fill_value=0.0)         # +net charges (reported depressed) / -net gains
    if special.empty or not special.notna().any().any():
        return F
    tax = _effective_tax_rate(daily)
    rev_pos = rev.where(rev > 0)

    if not pretax.empty:
        share = _ratio(special.abs(), pretax.abs(), positive_den=True)
        if share.notna().any().any():
            F["nonrecurring_pretax_share"] = share
    F["special_items_intensity"] = _ratio(special, rev_pos)    # signed: +ve => one-offs hurt reported

    core_ni = ni.add(special.mul(1.0 - tax) if not tax.empty else special, fill_value=0.0)
    # discontinued operations are net-of-tax and transitory -> removed from core directly
    disc = daily("discontinuedOps")
    if not core_ni.empty and not disc.empty:
        core_ni = core_ni.sub(disc, fill_value=0.0)
    if not core_ni.empty:
        F["core_profit_margin"] = _ratio(core_ni, rev_pos)     # vs reported profitMargins
        if mcap is not None and not mcap.empty:
            cey = _ratio(core_ni.where(core_ni > 0), mcap, positive_den=True)
            if cey.notna().any().any():
                F["core_earnings_yield"] = cey                 # vs reported earnings_yield
    if not oi.empty:
        F["core_operating_margin"] = _ratio(oi.add(charges, fill_value=0.0), rev_pos)
    if not ebitda.empty:
        F["adjusted_ebitda_margin"] = _ratio(
            ebitda.add(charges, fill_value=0.0).sub(gains, fill_value=0.0), rev_pos)
    return F


def _ai_leverage_fields(daily) -> dict:
    """#4 -- IT-MATURITY inputs for the AI-leverage score: how much a firm invests in
    its own software (the capability to deploy AI on its cost base). Populates once
    CapitalizedComputerSoftware* is extracted; empty before that. The OPPORTUNITY
    side (SG&A intensity = automatable admin/marketing, low revenue-per-employee =
    labor-heavy) and the sector-neutral score are assembled as a composite in
    build_cube.yml from existing peer-relative members (kept sector-neutral so it
    finds the best adopter in each industry, not just the hyperscalers)."""
    F: dict[str, pd.DataFrame] = {}
    soft, assets, rev = daily("capitalizedSoftware"), daily("totalAssets"), daily("totalRevenue")
    if not soft.empty and not assets.empty:
        si = _ratio(soft, assets, positive_den=True)
        if si.notna().any().any():
            F["capitalized_software_intensity"] = si
    if not soft.empty and not rev.empty:
        sr = _ratio(soft, rev.where(rev > 0))
        if sr.notna().any().any():
            F["software_to_revenue"] = sr
    return F


# --------------------------------------------------------------------------- #
# Derived characteristics                                                      #
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
) -> dict:
    """Build daily wide frames (date x ticker) for every characteristic.

    `yoy_periods` is how many filings span a year (4 for quarterly history,
    1 for annual); growth/trend features use it for a true YoY comparison.
    `intrinsic_cfg` overrides the DCF parameters for the intrinsic-value yield.
    """
    F: dict[str, pd.DataFrame] = {}
    intrinsic_cfg = intrinsic_cfg or {}

    _daily_cache: dict[str, pd.DataFrame] = {}

    def daily(field):
        # memoized: several fields are reused across blocks (and the new business-
        # quality helpers), so cache the pivot+ffill instead of recomputing it.
        if field not in _daily_cache:
            _daily_cache[field] = fundamentals_to_daily(fund_hist, field, idx)
        return _daily_cache[field]

    revenue = daily("totalRevenue")
    net_income = daily("netIncome")
    fcf = daily("freeCashflow")
    equity = daily("stockholdersEquity")
    ebitda = daily("ebitda")
    d2e = daily("debtToEquity")
    rnd = daily("researchAndDevelopment")
    # balance-sheet levels reused by BOTH the EV valuation and the distress block
    cash = daily("cash")
    long_debt = daily("longTermDebt")
    short_debt = daily("shortTermDebt")
    sbc = daily("stockBasedComp")

    # ---- Pension / OPEB overhang (debt-like): the recognized net UNDERFUNDED liability.
    # Point-in-time, gap-filled (combine_first, NOT a sum -> no OPEB double-count) across
    # three sources in order of directness:
    #   1) bulk Financial-Statement-Data-Sets recognized net liability (`pension_facts`),
    #   2) companyfacts `pensionDeficit`,
    #   3) footnote funded status = PBO - plan assets from the NOTES sets (`notes_num`).
    # Feeds the True EV + overhang ratio. The NOTES footnote (PBO / plan assets) ALSO
    # yields standalone funded-health features below (funded ratio, PBO / underfunding vs mcap).
    pbo = _notes_num_daily(notes_num, _FN_PBO_TAG, idx, instant=True)
    plan_assets = _notes_num_daily(notes_num, _FN_PLAN_ASSETS_TAG, idx, instant=True)
    fn_deficit = pd.DataFrame()
    if not pbo.empty and not plan_assets.empty:
        # funded status = plan assets - PBO; the deficit (underfunding) is the debt-like part.
        fn_deficit = pbo.sub(plan_assets).clip(lower=0.0)
        funded_ratio = _ratio(plan_assets, pbo, positive_den=True)   # 1.0 = fully funded, <1 under
        if funded_ratio.notna().any().any():
            F["pension_funded_ratio"] = funded_ratio

    pension_ret = _pension_deficit_daily(pension_facts, idx)
    for _src in (daily("pensionDeficit"), fn_deficit):
        if not _src.empty:
            pension_ret = pension_ret.combine_first(_src) if not pension_ret.empty else _src
    if not pension_ret.empty:
        pension_ret = pension_ret.clip(lower=0.0)          # underfunding only (>= 0)
        if pension_ret.notna().any().any():
            F["pension_retirement_liability"] = pension_ret

    # ---- Valuation yields (need a daily market cap) ----
    mcap = daily_market_cap(fund_hist, close) if close is not None else pd.DataFrame()
    if not mcap.empty:
        # Earnings/FCF/EV yields are only monotone as "cheapness" when the
        # NUMERATOR is positive: a negative E/P is not "cheap", it is a loss, so
        # ranking loss-makers by it is noise. Mask those to NaN (LightGBM handles
        # NaN; the peer-z/rank then only ranks names where the metric is defined),
        # and let the `profitable` / `fcf_positive` flags carry the regime instead.
        # Sales/price stays valid for everyone (revenue is always positive).
        F["earnings_yield"] = _ratio(net_income.where(net_income > 0), mcap, positive_den=True)
        F["sales_yield"] = _ratio(revenue, mcap, positive_den=True)
        F["book_yield"] = _ratio(equity.where(equity > 0), mcap, positive_den=True)
        F["fcf_yield"] = _ratio(fcf.where(fcf > 0), mcap, positive_den=True)
        # ---- True (fully-diluted) enterprise value; feeds every EV yield ----
        #   EV = fully-diluted mcap + total debt + leases + minority interest
        #        + pension/OPEB net deficit - cash - short-term investments
        # Prefer diluted shares x price for the equity claim (falls back to basic
        # mcap when diluted shares are absent); real debt columns preferred, with
        # debtToEquity*equity as a last-resort fallback.
        diluted = daily("dilutedShares")
        fd_mcap = mcap
        if close is not None and not diluted.empty and diluted.notna().any().any():
            cols = diluted.columns.intersection(close.columns)
            fd = (close[cols] * diluted[cols]).where(lambda x: x > 0)
            fd_mcap = fd.combine_first(mcap)          # diluted where available, else basic
        debt = _combine_debt(long_debt, short_debt, fallback=pd.DataFrame())
        if debt.empty and not d2e.empty and not equity.empty:
            debt = d2e.clip(lower=0.0) * equity.where(equity > 0)
        leases = daily("operatingLeaseLiability").add(daily("financeLeaseLiability"), fill_value=0.0)
        ev = _enterprise_value(fd_mcap, [debt, leases, daily("minorityInterest"), pension_ret],
                               [cash, daily("shortTermInvestments")])
        # Pension Overhang Leverage = pension/OPEB deficit / market cap (debt-like burden on
        # the equity value); higher = a bigger retirement obligation overhanging the stock.
        if not pension_ret.empty:
            pol = _ratio(pension_ret, mcap, positive_den=True)
            if pol.notna().any().any():
                F["pension_overhang_leverage"] = pol
        # NOTES footnote scale vs equity value: gross obligation (PBO) and the
        # underfunding, both relative to market cap. PBO/mcap flags rate/return
        # sensitivity even for FUNDED plans; underfunding/mcap is the cleaner deficit
        # burden (footnote-sourced, so it covers names the balance-sheet tag misses).
        if not pbo.empty:
            pbo_mc = _ratio(pbo, mcap, positive_den=True)
            if pbo_mc.notna().any().any():
                F["pbo_to_mcap"] = pbo_mc
        if not fn_deficit.empty:
            und_mc = _ratio(fn_deficit, mcap, positive_den=True)
            if und_mc.notna().any().any():
                F["pension_underfunding_to_mcap"] = und_mc
        if not ebitda.empty:
            F["ebitda_to_ev"] = _ratio(ebitda.where(ebitda > 0), ev, positive_den=True)
        # FCF/EV yield: cash the whole capital structure earns vs its total price.
        # The cleanest cross-sector cash-valuation yield, and it is exactly the
        # "Fully-Diluted FCF Yield" / energy FCF-EV yield (freeCashflow = OCF - capex).
        fcf_to_ev = _ratio(fcf.where(fcf > 0), ev, positive_den=True)
        if not fcf_to_ev.empty and fcf_to_ev.notna().any().any():
            F["fcf_to_ev"] = fcf_to_ev

        # ---- Altman Z (market-value variant): standard bankruptcy-risk screen ----
        #   Z = 1.2*WC/TA + 1.4*RE/TA + 3.3*EBIT/TA + 0.6*mcap/TL + 1.0*Sales/TA
        assets_z = daily("totalAssets")
        if not assets_z.empty:
            ta = assets_z.where(assets_z > 0)
            wc = daily("currentAssets").sub(daily("currentLiabilities"), fill_value=np.nan)
            z = (1.2 * _ratio(wc, ta) + 1.4 * _ratio(daily("retainedEarnings"), ta)
                 + 3.3 * _ratio(daily("operatingIncome"), ta)
                 + 0.6 * _ratio(mcap, daily("totalLiabilities"), positive_den=True)
                 + 1.0 * _ratio(revenue, ta))
            if not z.empty and z.notna().any().any():
                F["altman_z"] = z.replace([np.inf, -np.inf], np.nan)

        # ---- PEGY = P/E / (EPS growth% + dividend yield%) ----
        # trailing P/E; growth term PREFERS PROJECTED EPS growth (NTM/TTM-1 from the
        # analyst-estimate archive) and falls back to TTM realized net-income growth.
        pe = _ratio(mcap, net_income.where(net_income > 0), positive_den=True)
        growth_pct = None
        if earnings_history is not None and not earnings_history.empty:
            from src.data_aggregate.utils.earnings_features import ntm_ttm_eps  # lazy: avoid cycle
            ntm_e, ttm_e = ntm_ttm_eps(earnings_history, idx)
            if not ntm_e.empty and not ttm_e.empty:
                proj = _ratio(ntm_e, ttm_e.where(ttm_e > 0)) - 1.0        # projected EPS growth
                if proj.notna().any().any():
                    growth_pct = proj * 100.0
        if growth_pct is None:
            growth_pct = _fiscal_change_to_daily(fund_hist, "netIncome", idx,
                                                 kind="pct", periods=yoy_periods) * 100.0
        # div yield fills to 0 for non-payers, but GROWTH must be known or PEGY is
        # undefined (don't silently treat unknown growth as 0). This is the SEC
        # cash-flow (`dividendsPaid`) leg of the reconciled dividend yield — the
        # precise per-share/ex-date version is the standalone `dividend_yield`
        # feature in dividend_features.py (both agree; see its reconciliation note).
        div_yield_pct = _ratio(daily("dividendsPaid"), mcap, positive_den=True) * 100.0
        denom = (growth_pct + div_yield_pct.fillna(0.0)).where(lambda x: x > 0)
        pegy = _ratio(pe, denom)
        if not pegy.empty and pegy.notna().any().any():
            F["pegy"] = pegy.replace([np.inf, -np.inf], np.nan)

        # ---- REIT price multiples (gated on real-estate signature) ----
        re_gate = daily("realEstateNet").notna() | daily("rentalIncome").notna()
        depamort_ev = daily("depAmort")
        ffo = net_income.add(depamort_ev, fill_value=0.0).sub(
            daily("gainOnDispositions"), fill_value=0.0)
        ffo_yield = _ratio(ffo, mcap, positive_den=True)
        if not re_gate.empty:
            fy = ffo_yield.where(re_gate)
            if fy.notna().any().any():
                F["ffo_yield"] = fy                                   # FFO/price = 1 / P-FFO
            icr = _ratio(daily("operatingIncome").add(depamort_ev, fill_value=0.0), ev,
                         positive_den=True).where(re_gate)            # NOI(≈EBITDAre) / EV
            if icr.notna().any().any():
                F["implied_cap_rate"] = icr

        # ---- Energy EV/EBITDAX yield (gated on oil-gas property) ----
        energy_gate = daily("oilGasPropertyNet").notna()
        if not energy_gate.empty and energy_gate.any().any():
            ebitdax = (daily("operatingIncome").add(depamort_ev, fill_value=0.0)
                       .add(daily("explorationExpense"), fill_value=0.0))
            ex = _ratio(ebitdax.where(ebitdax > 0), ev, positive_den=True).where(energy_gate)
            if ex.notna().any().any():
                F["ebitdax_to_ev"] = ex

    # ---- Profitability / moat (raw ratios straight from the history) ----
    for field in ["grossMargins", "operatingMargins", "profitMargins",
                  "returnOnEquity", "debtToEquity", "revenueGrowth", "earningsGrowth"]:
        f = daily(field)
        if not f.empty:
            F[field] = f

    fcf_margin = _ratio(fcf, revenue, positive_den=True)
    if not fcf_margin.empty:
        F["fcf_margin"] = fcf_margin
    if not net_income.empty and not fcf.empty and not revenue.empty:
        cols = net_income.columns.intersection(fcf.columns)
        accr_num = net_income[cols] - fcf[cols]
        F["accruals"] = _ratio(accr_num, revenue, positive_den=True)

    # ---- Growth / trend / dilution from the fiscal series (year-over-year) ----
    fcf_growth = _fiscal_change_to_daily(fund_hist, "freeCashflow", idx,
                                         kind="pct", periods=yoy_periods)
    if fcf_growth.notna().any().any():
        F["fcf_growth"] = fcf_growth
    shares_growth = _fiscal_change_to_daily(fund_hist, "sharesOutstanding", idx,
                                            kind="pct", periods=yoy_periods)
    if shares_growth.notna().any().any():
        F["shares_growth"] = shares_growth
    gm_chg = _fiscal_change_to_daily(fund_hist, "grossMargins", idx,
                                     kind="diff", periods=yoy_periods)
    if gm_chg.notna().any().any():
        F["gross_margin_chg"] = gm_chg

    # ---- 5-YEAR MARGIN TREND: is the operating margin STRUCTURALLY expanding, not
    # just one good year. TTM operating income / revenue now vs ~5 trading years ago
    # (point-in-time, percentage-point change). Positive = durable margin expansion. ----
    op_margin = _ratio(daily("operatingIncome"), revenue, positive_den=True)
    if not op_margin.empty and op_margin.notna().any().any():
        om_5y = (op_margin - op_margin.shift(_FIVE_YEARS)).replace([np.inf, -np.inf], np.nan)
        if om_5y.notna().any().any():
            F["operating_margin_5y_chg"] = om_5y

    # ---- R&D intensity (only if collected) ----
    rd_intensity = _ratio(rnd, revenue, positive_den=True)
    if not rd_intensity.empty and rd_intensity.notna().any().any():
        F["rd_intensity"] = rd_intensity

    # ---- Reinvestment quality: depreciation & amortization vs capex ----
    # If D&A runs ABOVE capex (da_to_capex > 1) and, worse, is GROWING faster than
    # capex (da_minus_capex_growth > 0), the firm is consuming its asset base faster
    # than it reinvests -> aging PP&E / under-investment / a likely future capex
    # cliff, and reported earnings flattered by low capex. Both come straight from
    # the SEC fields already extracted (depAmort, capex); the model learns the sign.
    depamort = daily("depAmort")
    capex = daily("capex")
    if not depamort.empty and not capex.empty:
        da_to_capex = _ratio(depamort.abs(), capex.abs(), positive_den=True)
        if da_to_capex.notna().any().any():
            F["da_to_capex"] = da_to_capex
    da_growth = _fiscal_change_to_daily(fund_hist, "depAmort", idx,
                                        kind="pct", periods=yoy_periods)
    capex_growth = _fiscal_change_to_daily(fund_hist, "capex", idx,
                                           kind="pct", periods=yoy_periods)
    if da_growth.notna().any().any() and capex_growth.notna().any().any():
        F["da_minus_capex_growth"] = da_growth - capex_growth

    # ---- REGIME-ROBUST quality + state flags (loss-makers / growth names) ----
    # gross profitability (Novy-Marx) = gross profit / total assets. Defined and
    # monotone even for a net-loss-making firm, so it scores the growth /
    # unprofitable cohort where the earnings-based yields above are masked out.
    assets = daily("totalAssets")
    gm_lvl = F.get("grossMargins")
    if gm_lvl is not None and not revenue.empty and not assets.empty:
        cols = gm_lvl.columns.intersection(revenue.columns)
        gross_profit = gm_lvl[cols] * revenue[cols]           # grossMargins * revenue
        gp = _ratio(gross_profit, assets, positive_den=True)
        if not gp.empty and gp.notna().any().any():
            F["gross_profitability"] = gp

    # ---- A2: asset growth (Fama-French CMA "investment" factor). Firms that expand
    # the asset base aggressively subsequently UNDERperform (empire-building / over-
    # investment); sign is -1 in the model. ----
    asset_growth = _fiscal_change_to_daily(fund_hist, "totalAssets", idx,
                                           kind="pct", periods=yoy_periods)
    if asset_growth.notna().any().any():
        F["asset_growth"] = asset_growth

    # ---- A5: Rule of 40 (growth+profitability health) = TTM revenue-growth% + FCF-
    # margin%; >40 is elite (a fast grower OR a cash cow). Plus RPO growth = forward
    # bookings momentum (only defined for filers that report RPO -> tech-gated). ----
    rev_growth_pct = _fiscal_change_to_daily(fund_hist, "totalRevenue", idx,
                                             kind="pct", periods=yoy_periods) * 100.0
    fcf_margin_pct = _ratio(fcf, revenue, positive_den=True) * 100.0
    rule40 = (rev_growth_pct + fcf_margin_pct).replace([np.inf, -np.inf], np.nan)
    if rule40.notna().any().any():
        F["rule_of_40"] = rule40
    rpo_growth = _fiscal_change_to_daily(fund_hist, "remainingPerformanceObligation", idx,
                                         kind="pct", periods=yoy_periods)
    if rpo_growth.notna().any().any():
        F["rpo_growth"] = rpo_growth

    # ---- A3: Piotroski F-score (0-9). Nine fundamental-health binaries: profitability
    # (ROA>0, CFO>0, ΔROA>0, CFO/assets>ROA), lower leverage / better liquidity / no
    # dilution, and rising gross margin / asset turnover. Higher = stronger; scored only
    # where the core inputs (assets, NI, CFO) exist so a data-less name isn't a false 0. ----
    _oa = daily("totalAssets"); _ocf = daily("operatingCashFlow"); _sh = daily("sharesOutstanding")
    _roa = _ratio(net_income, _oa, positive_den=True)
    _cr = _ratio(daily("currentAssets"), daily("currentLiabilities"), positive_den=True)
    _lev = _ratio(long_debt, _oa, positive_den=True)
    _gm = daily("grossMargins")
    _turn = _ratio(revenue, _oa, positive_den=True)
    if not _oa.empty and not _ocf.empty and not net_income.empty:
        y = _YEAR
        parts = [
            (_roa > 0), (_ocf > 0), (_roa > _roa.shift(y)),
            (_ratio(_ocf, _oa, positive_den=True) > _roa),                 # accruals: cash > profit
            (_lev < _lev.shift(y)), (_cr > _cr.shift(y)),
            (_sh <= _sh.shift(y) * 1.001),                                 # no net dilution
            (_gm > _gm.shift(y)), (_turn > _turn.shift(y)),
        ]
        fscore = sum(p.astype("float64") for p in parts)
        gate = _oa.notna() & net_income.notna() & _ocf.notna()
        fscore = fscore.where(gate)
        if fscore.notna().any().any():
            F["piotroski_f_score"] = fscore

    # absolute 0/1 regime flags (emitted RAW, see _STATE_FIELDS): a NaN base ->
    # NaN flag (never a false 0), so "no data" is not read as "unprofitable".
    def _flag(base: pd.DataFrame, cond: pd.DataFrame) -> pd.DataFrame:
        return cond.astype(float).where(base.notna())

    if not net_income.empty:
        F["profitable"] = _flag(net_income, net_income > 0)
    if not fcf.empty:
        F["fcf_positive"] = _flag(fcf, fcf > 0)
    if not equity.empty:
        F["negative_equity"] = _flag(equity, equity <= 0)
    rev_growth_lvl = daily("revenueGrowth")
    if not rev_growth_lvl.empty:
        F["hyper_growth"] = _flag(rev_growth_lvl, rev_growth_lvl > _HYPER_GROWTH)

    # ---- LATEST-QUARTER momentum (discrete single quarter, not TTM) ----
    # captures acceleration/inflection that TTM smooths away. Needs the discrete
    # single-quarter columns emitted by the extractor.
    q_rev_yoy = _fiscal_apply_to_daily(fund_hist, "revenue_q", idx,
                                       lambda s: s.pct_change(yoy_periods))
    if q_rev_yoy.notna().any().any():
        F["q_rev_growth"] = q_rev_yoy
        # acceleration = this quarter's YoY minus the previous quarter's YoY
        F["rev_growth_accel"] = _fiscal_apply_to_daily(
            fund_hist, "revenue_q", idx,
            lambda s: s.pct_change(yoy_periods).diff(1))

    q_ni_yoy = _fiscal_apply_to_daily(fund_hist, "netIncome_q", idx,
                                      lambda s: s.pct_change(yoy_periods))
    if q_ni_yoy.notna().any().any():
        F["q_earnings_growth"] = q_ni_yoy

    # latest-quarter margin vs TTM margin = margin inflection
    rev_q = daily("revenue_q")
    ni_q = daily("netIncome_q")
    q_margin = _ratio(ni_q, rev_q, positive_den=True)
    if not q_margin.empty and "profitMargins" in F:
        cols = q_margin.columns.intersection(F["profitMargins"].columns)
        F["q_margin_vs_ttm"] = q_margin[cols] - F["profitMargins"][cols]

    # ---- YEARLY-TTM momentum (current TTM vs TTM one year ago) ----
    # Complements the single-quarter features: captures multi-quarter trend rather
    # than the most-recent quarter jolt, which is less noisy and works at longer
    # horizons. Uses the fiscal history of level columns so seasonality is removed
    # by construction (same quarter each year -> yoy_periods filings back).
    y_rev_growth = _fiscal_change_to_daily(fund_hist, "totalRevenue", idx,
                                           kind="pct", periods=yoy_periods)
    if y_rev_growth.notna().any().any():
        F["y_rev_growth"] = y_rev_growth
        F["y_rev_growth_accel"] = _fiscal_apply_to_daily(
            fund_hist, "totalRevenue", idx,
            lambda s, n=yoy_periods: s.pct_change(n).diff(1))

    y_earnings_growth = _fiscal_change_to_daily(fund_hist, "netIncome", idx,
                                                kind="pct", periods=yoy_periods)
    if y_earnings_growth.notna().any().any():
        F["y_earnings_growth"] = y_earnings_growth

    # YoY change in TTM profit margin (margin expansion / contraction trend)
    y_margin_chg = _fiscal_change_to_daily(fund_hist, "profitMargins", idx,
                                           kind="diff", periods=yoy_periods)
    if y_margin_chg.notna().any().any():
        F["y_margin_vs_ttm"] = y_margin_chg

    # ---- DISTRESS / SOLVENCY (can the firm service and roll its debt?) ----
    # debtToEquity is a book ratio that says nothing about debt-SERVICING ability;
    # these are the leverage / coverage / liquidity ratios credit desks watch.
    total_debt = _combine_debt(long_debt, short_debt, daily("totalLiabilities"))
    if not total_debt.empty and not ebitda.empty:
        cols = total_debt.columns.intersection(cash.columns) if not cash.empty else total_debt.columns
        net_debt = (total_debt[cols].sub(cash[cols], fill_value=0.0)
                    if not cash.empty else total_debt)
        # HIGH net-debt/EBITDA = more leveraged = worse (only meaningful for EBITDA>0)
        nd_ebitda = _ratio(net_debt, ebitda, positive_den=True)
        if not nd_ebitda.empty and nd_ebitda.notna().any().any():
            F["net_debt_to_ebitda"] = nd_ebitda
    interest = daily("interestExpense")
    if not ebitda.empty and not interest.empty:
        # HIGH coverage = safer. Interest is an expense (take abs to be sign-safe).
        cov = _ratio(ebitda, interest.abs(), positive_den=True)
        if not cov.empty and cov.notna().any().any():
            F["interest_coverage"] = cov
    cur_a, cur_l = daily("currentAssets"), daily("currentLiabilities")
    current_ratio = _ratio(cur_a, cur_l, positive_den=True)
    if not current_ratio.empty and current_ratio.notna().any().any():
        F["current_ratio"] = current_ratio
    if not cash.empty and not total_debt.empty:
        cash_to_debt = _ratio(cash, total_debt, positive_den=True)
        if not cash_to_debt.empty and cash_to_debt.notna().any().any():
            F["cash_to_debt"] = cash_to_debt

    # ---- REFINANCING RISK: near-term debt maturities vs the liquidity to cover them.
    # short-term debt / (cash + trailing free cash flow). HIGH (>>1) = the firm must
    # roll/refinance a big slug of debt it cannot self-fund -> exposed to rate spikes
    # / frozen credit markets. (short_debt / cash / fcf are hoisted above.) ----
    liquidity = cash.add(fcf.where(fcf > 0), fill_value=0.0)
    refi = _ratio(short_debt, liquidity, positive_den=True)
    if not refi.empty and refi.notna().any().any():
        F["refinancing_risk"] = refi

    # ---- MARKETING & SALES efficiency (operating leverage) ----
    sga = daily("sellingGeneralAdmin")
    sga_intensity = _ratio(sga, revenue, positive_den=True)
    if not sga_intensity.empty and sga_intensity.notna().any().any():
        F["sga_intensity"] = sga_intensity
    sga_growth = _fiscal_change_to_daily(fund_hist, "sellingGeneralAdmin", idx,
                                         kind="pct", periods=yoy_periods)
    rev_growth = _fiscal_change_to_daily(fund_hist, "totalRevenue", idx,
                                         kind="pct", periods=yoy_periods)
    if sga_growth.notna().any().any():
        F["sga_growth"] = sga_growth
        # operating leverage = sales growing FASTER than selling cost (scalable);
        # negative = growth is being "bought" with rising SG&A (margin risk ahead).
        if rev_growth.notna().any().any():
            cols = rev_growth.columns.intersection(sga_growth.columns)
            F["operating_leverage"] = rev_growth[cols] - sga_growth[cols]

    # ---- M&A footprint (organic vs inorganic growth; impairment risk) ----
    acq = daily("acquisitions")
    assets = daily("totalAssets")
    acq_den = assets if not assets.empty else revenue
    acq_intensity = _ratio(acq.abs() if not acq.empty else acq, acq_den, positive_den=True)
    if not acq_intensity.empty and acq_intensity.notna().any().any():
        F["acquisition_intensity"] = acq_intensity
    goodwill_growth = _fiscal_change_to_daily(fund_hist, "goodwill", idx,
                                              kind="pct", periods=yoy_periods)
    if goodwill_growth.notna().any().any():
        F["goodwill_growth"] = goodwill_growth

    # ---- STOCK-BASED COMPENSATION ("employee shares given") ----
    # shares_growth is NET of buybacks and can be masked; SBC is the GROSS give-away.
    # (sbc daily frame is hoisted above; it also feeds the EV calc.)
    sbc_intensity = _ratio(sbc, revenue, positive_den=True)
    if not sbc_intensity.empty and sbc_intensity.notna().any().any():
        F["sbc_intensity"] = sbc_intensity
    ocf = daily("operatingCashFlow")
    if not sbc.empty and not ocf.empty:
        # how much of reported operating cash flow is really non-cash comp
        sbc_to_ocf = _ratio(sbc, ocf, positive_den=True)
        if not sbc_to_ocf.empty and sbc_to_ocf.notna().any().any():
            F["sbc_to_ocf"] = sbc_to_ocf

    # ---- REFINED VALUATION-ENGINE RATIOS (elasticity / divergence / dilution) ----
    # Operating-leverage ELASTICITY = %ΔOperatingIncome / %ΔRevenue (>1 = scalable
    # model, exponential profit vs linear sales; <1 = diseconomies of scale). This is
    # the user's exact def, distinct from `operating_leverage` (revenue - SG&A growth).
    oi_growth = _fiscal_change_to_daily(fund_hist, "operatingIncome", idx, kind="pct", periods=yoy_periods)
    rev_growth_f = _fiscal_change_to_daily(fund_hist, "totalRevenue", idx, kind="pct", periods=yoy_periods)
    # guard: elasticity is meaningless (and explodes) when revenue is ~flat, so require
    # at least a 2% revenue move for the denominator.
    ol_el = _ratio(oi_growth, rev_growth_f.where(rev_growth_f.abs() >= 0.02))
    if not ol_el.empty and ol_el.notna().any().any():
        F["operating_leverage_elasticity"] = ol_el
    # Gross vs EBITDA margin-expansion divergence: gross margin expanding while EBITDA
    # margin lags = losing SG&A/overhead control; both expanding = true pricing power.
    gm = _ratio(daily("grossProfit"), revenue, positive_den=True)
    if gm.empty:
        gm = daily("grossMargins")
    em = _ratio(ebitda, revenue, positive_den=True)
    if not gm.empty and not em.empty:
        med = (gm - gm.shift(252)) - (em - em.shift(252))     # ~1y change divergence
        if med.notna().any().any():
            F["margin_expansion_delta"] = med.replace([np.inf, -np.inf], np.nan)
    # Net working-capital elasticity = %ΔNWC / %ΔRevenue (>1 = cash-hungry growth).
    cur_a2, cur_l2 = daily("currentAssets"), daily("currentLiabilities")
    if not cur_a2.empty and not cur_l2.empty and not revenue.empty:
        nwc = cur_a2.sub(cur_l2, fill_value=np.nan)
        prev_nwc = nwc.shift(252)
        nwc_g = _ratio(nwc - prev_nwc, prev_nwc, positive_den=True)
        prev_rev = revenue.shift(252)
        rev_g = _ratio(revenue - prev_rev, prev_rev, positive_den=True)
        nwc_el = _ratio(nwc_g, rev_g.where(rev_g.abs() >= 0.02))   # guard ~flat-revenue blow-ups
        if not nwc_el.empty and nwc_el.notna().any().any():
            F["nwc_elasticity"] = nwc_el
    # Fully-diluted shareholder dilution rate (YoY change in diluted shares).
    dil_growth = _fiscal_change_to_daily(fund_hist, "dilutedShares", idx, kind="pct", periods=yoy_periods)
    if dil_growth.notna().any().any():
        F["diluted_shares_growth"] = dil_growth
    # EBIT interest coverage = EBIT / interest (the user's def; complements the
    # EBITDA-based interest_coverage). LOW (< ~2x) = structural distress risk.
    eic = _ratio(daily("operatingIncome"), daily("interestExpense").abs(), positive_den=True)
    if not eic.empty and eic.notna().any().any():
        F["ebit_interest_coverage"] = eic

    # ---- INTRINSIC VALUE (two-stage DCF on TTM FCF) vs price ----
    if close is not None:
        iv = intrinsic_value_daily(fund_hist, close, idx, **intrinsic_cfg)
        iy = iv.get("yield")
        if iy is not None and not iy.empty and iy.notna().any().any():
            F["intrinsic_yield"] = iy

    # ---- BUSINESS-QUALITY blocks (all from tags already extracted) ----
    #   #2 D&A/SBC realism, #5 forensic red flags, #3 M&A digestion,
    #   #1 core/adjusted earnings (kept alongside the reported figures above).
    F.update(_da_realism_fields(daily))
    F.update(_forensic_fields(daily, idx, pension_ret))   # reuse the coalesced overhang pool
    F.update(_digestion_fields(daily, fund_hist, idx, yoy_periods))
    F.update(_core_earnings_fields(daily, mcap))
    F.update(_ai_leverage_fields(daily))

    return F


def _peer_relative(
    field_df: pd.DataFrame,
    peer_dict: dict,
    min_peers: int = 3,
    clip: float = 8.0,
) -> pd.DataFrame:
    """
    (stock - peer_weighted_mean) / peer_weighted_std, per date, per stock.
    Self excluded by construction of the peer dict. NaN-tolerant: peer stats use
    whichever peers have data on the date (weights renormalized).

    Robustness (critical): when only a couple of peers report or their values
    nearly coincide, the peer std collapses toward zero and the raw z-score
    explodes to ~1e13. We therefore (a) require at least `min_peers` peers with
    data on the date, (b) drop dates where the peer std is not strictly positive,
    and (c) winsorize the result to +-`clip` so a near-degenerate peer group can
    never dominate the model.
    """
    rel = pd.DataFrame(index=field_df.index, columns=field_df.columns, dtype="float64")
    for ticker, peers in peer_dict.items():
        if not peers or ticker not in field_df.columns:
            continue
        cols = [p for p in peers if p in field_df.columns]
        if len(cols) < min_peers:
            continue
        w = pd.Series({p: float(peers[p]) for p in cols}, dtype="float64")
        w = w / w.sum()
        peer_vals = field_df[cols]
        present = peer_vals.notna()
        n_present = present.sum(axis=1)
        wsum = present.mul(w, axis=1).sum(axis=1)
        valid = (n_present >= min_peers) & (wsum > 0)

        pmean = peer_vals.mul(w, axis=1).sum(axis=1, min_count=1).div(wsum.where(valid))
        var = (peer_vals.sub(pmean, axis=0) ** 2).mul(w, axis=1).sum(axis=1, min_count=1)
        pstd = np.sqrt(var.div(wsum.where(valid)))

        z = (field_df[ticker] - pmean) / pstd.where(pstd > 0)
        z = z.where(valid)
        rel[ticker] = z.clip(-clip, clip)
    return rel.replace([np.inf, -np.inf], np.nan)


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
        return pd.DataFrame(columns=["date", "ticker"])

    yoy_periods = _infer_yoy_periods(fundamentals_history)
    fields = _derived_fields(fundamentals_history, trading_index, stock_close,
                             yoy_periods=yoy_periods, intrinsic_cfg=intrinsic_cfg,
                             earnings_history=earnings_history, pension_facts=pension_facts,
                             notes_num=notes_num)

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
        s = fdf.stack()
        s.index.set_names(["date", "ticker"], inplace=True)
        long_frames.append(s.rename(f"f_{name}"))
    if not long_frames:
        return pd.DataFrame(columns=["date", "ticker"])
    # .copy() consolidates the many single-column blocks that concat(axis=1) leaves
    # behind, so the reset_index() column insert doesn't trip the "highly fragmented
    # DataFrame" PerformanceWarning once the panel has 100+ feature columns.
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
        s = zdf.stack()
        s.index.set_names(["date", "ticker"], inplace=True)
        long_frames.append(s.rename(f"f_{name}_vs_hist"))
    if not long_frames:
        return pd.DataFrame(columns=["date", "ticker"])
    # .copy() consolidates the many single-column blocks that concat(axis=1) leaves
    # behind, so the reset_index() column insert doesn't trip the "highly fragmented
    # DataFrame" PerformanceWarning once the panel has 100+ feature columns.
    return pd.concat(long_frames, axis=1).copy().reset_index()


def build_peer_relative_panel(fields: dict, peer_dict: dict) -> pd.DataFrame:
    """Turn a {name: daily wide frame} dict into the long feature panel, each
    characteristic expressed as `f_<name>_vs_peers` (peer-standardized) and
    `f_<name>_xs` (universe percentile). Shared by the fundamental and analyst
    feature builders."""
    if not fields:
        return pd.DataFrame(columns=["date", "ticker"])

    long_frames = []
    for name, fdf in fields.items():
        if fdf is None or fdf.empty:
            continue
        # peer z-score, then trim per-day cross-sectional 1%/99% outliers (the
        # percentile-rank `_xs` below is already outlier-proof, so it uses raw fdf).
        rel = _winsorize_xs(_peer_relative(fdf, peer_dict))
        s = rel.stack()
        s.index.set_names(["date", "ticker"], inplace=True)
        long_frames.append(s.rename(f"f_{name}_vs_peers"))

        xs = fdf.rank(axis=1, pct=True, method="average")
        s2 = xs.stack()
        s2.index.set_names(["date", "ticker"], inplace=True)
        long_frames.append(s2.rename(f"f_{name}_xs"))

    if not long_frames:
        return pd.DataFrame(columns=["date", "ticker"])
    # .copy() consolidates the many single-column blocks that concat(axis=1) leaves
    # behind, so the reset_index() column insert doesn't trip the "highly fragmented
    # DataFrame" PerformanceWarning once the panel has 100+ feature columns.
    return pd.concat(long_frames, axis=1).copy().reset_index()
