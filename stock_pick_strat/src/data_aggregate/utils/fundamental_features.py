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
    ebitda_to_ev    = ebitda / enterpriseValue (inverse EV/EBITDA), where
                      EV = marketCap + total debt + stock-based comp - cash.
                      SBC is added like debt ("future debt disguised as stock":
                      the equity claim serial diluters keep handing employees);
                      real debt columns are used, with debtToEquity*equity as a
                      fallback for histories that lack them.

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

from src.data_aggregate.utils.factors import fundamentals_to_daily, daily_market_cap
from src.data_aggregate.utils.intrinsic import intrinsic_value_daily


# Valuation yields that also get a self-history (mean-reversion) z-score, i.e.
# "cheap vs its OWN past" in addition to "cheap vs peers". High = cheaper than
# the firm's own norm -> classic valuation mean-reversion signal.
_MEAN_REVERSION_FIELDS = (
    "earnings_yield", "sales_yield", "book_yield",
    "fcf_yield", "ebitda_to_ev", "intrinsic_yield",
)
_HIST_WINDOW = 1260      # ~5 trading years of daily observations
_HIST_MIN_PERIODS = 252  # require >= 1y of history before emitting a z-score

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


def _enterprise_value(mcap: pd.DataFrame, debt: pd.DataFrame,
                      sbc: pd.DataFrame, cash: pd.DataFrame) -> pd.DataFrame:
    """EV = market cap + total debt + stock-based comp - cash, restricted to the
    tickers that have a market cap.

    Stock-based comp is added like debt: it is the equity claim the firm keeps
    handing employees -- "future debt disguised as stock" -- so a serial diluter
    has a larger true EV (and a lower EBITDA/EV yield) than its market cap alone
    suggests. NaN-tolerant: a missing component contributes 0."""
    ev = mcap.copy()
    for part, sign in ((debt, 1.0), (sbc, 1.0), (cash, -1.0)):
        if part is not None and not part.empty:
            aligned = part.reindex(columns=mcap.columns).fillna(0.0)
            ev = ev + sign * aligned
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
    return z.clip(-clip, clip).replace([np.inf, -np.inf], np.nan)


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
# Derived characteristics                                                      #
# --------------------------------------------------------------------------- #
def _derived_fields(
    fund_hist: pd.DataFrame,
    idx: pd.DatetimeIndex,
    close: pd.DataFrame | None,
    yoy_periods: int = 1,
    intrinsic_cfg: dict | None = None,
) -> dict:
    """Build daily wide frames (date x ticker) for every characteristic.

    `yoy_periods` is how many filings span a year (4 for quarterly history,
    1 for annual); growth/trend features use it for a true YoY comparison.
    `intrinsic_cfg` overrides the DCF parameters for the intrinsic-value yield.
    """
    F: dict[str, pd.DataFrame] = {}
    intrinsic_cfg = intrinsic_cfg or {}

    def daily(field):
        return fundamentals_to_daily(fund_hist, field, idx)

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
        if not ebitda.empty:
            # Enterprise value from the real balance sheet:
            #   EV = marketCap + total debt + stock-based comp - cash
            # SBC is added as future dilution ("debt disguised as stock"). Prefer
            # the real debt columns; older histories without them fall back to the
            # debtToEquity * equity approximation.
            debt = _combine_debt(long_debt, short_debt, fallback=pd.DataFrame())
            if debt.empty and not d2e.empty and not equity.empty:
                debt = d2e.clip(lower=0.0) * equity.where(equity > 0)
            ev = _enterprise_value(mcap, debt, sbc, cash)
            F["ebitda_to_ev"] = _ratio(ebitda.where(ebitda > 0), ev, positive_den=True)

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

    # ---- R&D intensity (only if collected) ----
    rd_intensity = _ratio(rnd, revenue, positive_den=True)
    if not rd_intensity.empty and rd_intensity.notna().any().any():
        F["rd_intensity"] = rd_intensity

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

    # ---- INTRINSIC VALUE (two-stage DCF on TTM FCF) vs price ----
    if close is not None:
        iv = intrinsic_value_daily(fund_hist, close, idx, **intrinsic_cfg)
        iy = iv.get("yield")
        if iy is not None and not iy.empty and iy.notna().any().any():
            F["intrinsic_yield"] = iy

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
                             yoy_periods=yoy_periods, intrinsic_cfg=intrinsic_cfg)

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
    return pd.concat(long_frames, axis=1).reset_index()


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
    return pd.concat(long_frames, axis=1).reset_index()


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
        rel = _peer_relative(fdf, peer_dict)
        s = rel.stack()
        s.index.set_names(["date", "ticker"], inplace=True)
        long_frames.append(s.rename(f"f_{name}_vs_peers"))

        xs = fdf.rank(axis=1, pct=True, method="average")
        s2 = xs.stack()
        s2.index.set_names(["date", "ticker"], inplace=True)
        long_frames.append(s2.rename(f"f_{name}_xs"))

    if not long_frames:
        return pd.DataFrame(columns=["date", "ticker"])
    return pd.concat(long_frames, axis=1).reset_index()
