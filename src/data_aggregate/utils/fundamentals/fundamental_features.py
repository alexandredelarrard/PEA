"""Point-in-time fundamental features.

The builder cleans each source once, converts filing data to memoized daily frames,
computes source-specific features, then emits peer, cross-sectional, self-history,
and raw regime views.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
import pandas as pd

from src.context import Context
from src.data_aggregate.utils.common import capital
from src.data_aggregate.utils.common.frames import ratio, sanitize
from src.data_aggregate.utils.common.panel import build_peer_relative_panel, mask_to_availability
from src.data_aggregate.utils.common.pit import (
    PitFrames,
    fiscal_change_to_daily,
    fundamentals_to_daily,
)
from src.data_aggregate.utils.common.sector_gates import family_tickers
from src.data_aggregate.utils.common.xs import (
    HIST_MIN_PERIODS,
    HIST_WINDOW,
    self_history_z,
)
from src.data_aggregate.utils.fundamentals.earnings_features import ntm_ttm_eps
from src.data_aggregate.utils.fundamentals.intrinsic import intrinsic_value_daily
from src.data_store.schema import Table, Tables

_FACT_COLUMNS = ("ticker", "tag", "ddate", "qtrs", "value", "filed")
_FUNDAMENTAL_NON_NUMERIC_COLUMNS = frozenset({"ticker", "as_of", "fiscal_end", "regime", "sector", "industry_group", "sub_industry"})
_MEAN_REVERSION_FIELDS = (
    "earnings_yield",
    "sales_yield",
    "book_yield",
    "fcf_yield",
    "ebitda_to_ev",
    "fcf_to_ev",
    "ffo_yield",
    "intrinsic_yield",
)
_STATE_FIELDS = ("profitable", "fcf_positive", "negative_equity", "hyper_growth")
_HYPER_GROWTH = 0.25
_YEAR = 252
_FIVE_YEARS = 5 * _YEAR
_NET_PENSION_TAGS = (
    "PensionAndOtherPostretirementDefinedBenefitPlansLiabilitiesNoncurrent",
    "DefinedBenefitPensionPlanLiabilitiesNoncurrent",
)
_FN_PBO_TAG = "DefinedBenefitPlanBenefitObligation"
_FN_PLAN_ASSETS_TAG = "DefinedBenefitPlanFairValueOfPlanAssets"


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def clean_fundamentals_history(fundamentals_history: pd.DataFrame) -> pd.DataFrame:
    """Normalize the merged consumer history once before feature computation."""
    missing = {"ticker", "as_of"} - set(fundamentals_history.columns)
    if missing:
        raise ValueError(f"{Tables.fundamentals_history} is missing required columns: {sorted(missing)}")

    cleaned = fundamentals_history.copy()
    cleaned["ticker"] = cleaned["ticker"].astype("string").str.strip().str.upper()
    cleaned["as_of"] = pd.to_datetime(cleaned["as_of"], errors="coerce")
    if "fiscal_end" in cleaned.columns:
        cleaned["fiscal_end"] = pd.to_datetime(cleaned["fiscal_end"], errors="coerce")
    numeric = [column for column in cleaned.columns if column not in _FUNDAMENTAL_NON_NUMERIC_COLUMNS]
    cleaned[numeric] = cleaned[numeric].apply(pd.to_numeric, errors="coerce")
    return cleaned.dropna(subset=["ticker", "as_of"]).sort_values(["ticker", "as_of"]).reset_index(drop=True)


def _clean_tagged_facts(facts: pd.DataFrame | None, universe: set[str]) -> pd.DataFrame | None:
    """Normalize one SEC facts source and restrict it to the panel universe."""
    if facts is None or facts.empty or not set(_FACT_COLUMNS).issubset(facts.columns):
        return None

    cleaned = cast(pd.DataFrame, facts.loc[:, list(_FACT_COLUMNS)].copy())
    cleaned["ticker"] = cleaned["ticker"].astype("string").str.strip().str.upper()
    cleaned["tag"] = cleaned["tag"].astype("string")
    cleaned["as_of"] = pd.to_datetime(cleaned["filed"], errors="coerce")
    cleaned["ddate"] = pd.to_datetime(cleaned["ddate"], errors="coerce")
    cleaned[["qtrs", "value"]] = cleaned[["qtrs", "value"]].apply(pd.to_numeric, errors="coerce")
    if universe:
        cleaned = cast(pd.DataFrame, cleaned.loc[cleaned["ticker"].isin(universe), :])
    cleaned = cleaned.dropna(subset=["ticker", "tag", "as_of", "value"])
    if cleaned.empty:
        return None
    return cleaned.sort_values(["ticker", "as_of", "ddate"]).reset_index(drop=True)


def _postprocess_feature_frames(features: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Apply output-wide numeric cleanup and storage dtype once."""
    return {
        name: frame.replace([np.inf, -np.inf], np.nan).sort_index().astype("float32") if not frame.empty else frame
        for name, frame in features.items()
    }


def _combine_debt(daily, long_debt: pd.DataFrame, short_debt: pd.DataFrame, fallback: pd.DataFrame) -> pd.DataFrame:
    """Prefer reconciled debt, then debt legs, then total liabilities."""
    borrowed = cast(pd.DataFrame | None, capital.borrowings(daily))
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
    """Combine market cap with available debt-like and liquid-asset components."""
    ev = mcap.copy()
    for part in additions:
        if part is not None and not part.empty:
            ev = ev + part.reindex(columns=mcap.columns).fillna(0.0)
    for part in subtractions:
        if part is not None and not part.empty:
            ev = ev - part.reindex(columns=mcap.columns).fillna(0.0)
    return ev


# --------------------------------------------------------------------------- #
# Business-quality helpers (all from tags ALREADY extracted -- no new SEC pull)
#   #2 D&A/SBC realism, #5 forensic, #3 M&A digestion, #1 core/adjusted earnings.
# Each returns a {name: daily wide frame} dict that _derived_fields merges into F,
# so every field auto-expands to f_<name>_vs_peers + f_<name>_xs downstream.
# `daily` is the shared PitFrames accessor (field -> date x ticker).
# --------------------------------------------------------------------------- #
def _nopat_tax_rate(daily, default: float = 0.21) -> pd.DataFrame:
    """Return the bounded tax rate used internally for NOPAT calculations."""
    tax, pre = daily("incomeTaxExpense"), daily("pretaxIncome")
    if tax.empty or pre.empty:
        return pd.DataFrame()
    return ratio(tax, pre.where(pre > 0)).clip(0.0, 0.5).fillna(default)


def _da_realism_fields(daily) -> dict:
    """Compare stock-based compensation with repurchases."""
    features: dict[str, pd.DataFrame] = {}
    sbc = daily("stockBasedComp")
    buyback = cast(pd.DataFrame | None, capital.share_repurchases(daily))  # positive magnitude; see capital.py
    if not sbc.empty and buyback is not None and not buyback.empty:
        s2b = ratio(sbc, buyback, positive_den=True)
        if s2b.notna().any().any():
            features["sbc_to_buyback"] = s2b  # >1 = buybacks don't even cover SBC
    return features


def _beneish_m_score(daily, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Compute the eight-variable Beneish score where revenue and assets exist."""
    rev = daily("totalRevenue")
    assets = cast(pd.DataFrame, capital.assets_ex_lease(daily))
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
    dsri = ix(ar_sales, ar_sales.shift(_YEAR))  # days sales in receivables
    gm = ratio(gp, rev, positive_den=True)
    gmi = ix(gm.shift(_YEAR), gm)  # gross-margin deterioration
    noncore = 1.0 - ratio(ca.add(ppe, fill_value=0.0), assets, positive_den=True)
    aqi = ix(noncore, noncore.shift(_YEAR))  # asset-quality (soft assets)
    sgi = ix(rev, rev.shift(_YEAR))  # sales growth
    deprate = ratio(dep, dep.add(ppe, fill_value=0.0), positive_den=True)
    depi = ix(deprate.shift(_YEAR), deprate)  # slowing depreciation
    sgar = ratio(sga, rev, positive_den=True)
    sgai = ix(sgar, sgar.shift(_YEAR))  # SG&A efficiency
    lev = ratio(ltd.add(cl, fill_value=0.0), assets, positive_den=True)
    lvgi = ix(lev, lev.shift(_YEAR))  # leverage change
    tata = ratio(ni.sub(ocf, fill_value=np.nan), assets)  # total accruals / assets

    terms = [
        (0.920, dsri, 1.0),
        (0.528, gmi, 1.0),
        (0.404, aqi, 1.0),
        (0.892, sgi, 1.0),
        (0.115, depi, 1.0),
        (-0.172, sgai, 1.0),
        (-0.327, lvgi, 1.0),
        (4.679, tata, 0.0),
    ]
    cols = sorted(set().union(*[set(t[1].columns) for t in terms if isinstance(t[1], pd.DataFrame) and not t[1].empty]))
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
    support = rev.reindex(index=idx, columns=cols).notna() & assets.reindex(index=idx, columns=cols).notna()
    return m.where(support)


def _tagged_fact_daily(
    facts: pd.DataFrame | None,
    tag: str,
    idx: pd.DatetimeIndex,
    *,
    name: str | None = None,
    instant: bool = True,
) -> pd.DataFrame:
    """Convert one already-cleaned SEC fact tag to a point-in-time daily frame."""
    if facts is None or facts.empty:
        return pd.DataFrame(index=idx)
    rows = facts[(facts["tag"] == tag) & ((facts["qtrs"].fillna(0) == 0) if instant else (facts["qtrs"] > 0))]
    if rows.empty:
        return pd.DataFrame(index=idx)
    field = name or tag
    return fundamentals_to_daily(rows.rename(columns={"value": field}), field, idx)


def _pension_deficit_daily(pension_facts: pd.DataFrame | None, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Recognized net pension deficit, preferring the broader primary tag."""
    primary = _tagged_fact_daily(pension_facts, _NET_PENSION_TAGS[0], idx, name="pension_deficit")
    variant = _tagged_fact_daily(pension_facts, _NET_PENSION_TAGS[1], idx, name="pension_deficit")
    prim, var = primary, variant
    if prim.empty:
        return var
    if var.empty:
        return prim
    return prim.combine_first(var)


def load_tagged_facts(
    context: Context,
    table: Table,
    tags: tuple[str, ...],
    columns: Sequence[str] | None = None,
    tickers: Sequence[str] | None = None,
) -> pd.DataFrame | None:
    """Read only the projected fact tags used by this panel."""
    where: dict[str, list[str]] = {"tag": list(tags)}
    if tickers:
        where["ticker"] = list(tickers)
    df = context.store.load(table, columns=columns or _FACT_COLUMNS, where=where, optional=True)
    return df.reset_index(drop=True) if df is not None else None


def load_pension_facts_scoped(context: Context, tickers: Sequence[str] | None = None) -> pd.DataFrame | None:
    """`pension_facts` restricted to the recognized net-liability tags the panel reads."""
    return load_tagged_facts(context, Tables.pension_facts, _NET_PENSION_TAGS, tickers=tickers)


def load_notes_num_scoped(context: Context, tickers: Sequence[str] | None = None) -> pd.DataFrame | None:
    """`notes_num` restricted to the footnote PBO + plan-asset tags the panel reads."""
    return load_tagged_facts(context, Tables.notes_num, (_FN_PBO_TAG, _FN_PLAN_ASSETS_TAG), tickers=tickers)


def _forensic_fields(daily, idx: pd.DatetimeIndex) -> dict:
    """Build working-capital timing and Beneish forensic fields."""
    features: dict[str, pd.DataFrame] = {}
    rev, cogs = daily("totalRevenue"), daily("costOfRevenue")
    ar, ap, inv = daily("accountsReceivable"), daily("accountsPayable"), daily("inventory")

    dso = ratio(ar, rev, positive_den=True) * 365.0
    if dso.notna().any().any():
        features["dso"] = dso
        features["dso_change"] = dso - dso.shift(_YEAR)
    dpo = ratio(ap, cogs, positive_den=True) * 365.0
    if dpo.notna().any().any():
        features["dpo"] = dpo
        features["dpo_change"] = dpo - dpo.shift(_YEAR)
    dio = ratio(inv, cogs, positive_den=True) * 365.0
    if dio.notna().any().any():
        features["dio"] = dio
    if "dso" in features and "dpo" in features and "dio" in features:
        ccc = features["dso"].add(features["dio"], fill_value=np.nan).sub(features["dpo"], fill_value=np.nan)
        if ccc.notna().any().any():
            features["cash_conversion_cycle"] = ccc

    m = _beneish_m_score(daily, idx)
    if not m.empty and m.notna().any().any():
        features["beneish_m_score"] = m
    return features


def _digestion_fields(daily, fund_hist: pd.DataFrame, idx: pd.DatetimeIndex, yoy_periods: int, operating_cash: frozenset[str] | None = None) -> dict:
    """Build acquisition-digestion, intangible exposure, and SG&A elasticity fields."""
    features: dict[str, pd.DataFrame] = {}
    # asset base EX the ASC-842 ROU asset, so the FY2019 adoption jump does not read as
    # balance-sheet growth (see `totalAssetsExLease` in the extractor).
    oi, assets = daily("operatingIncome"), capital.assets_ex_lease(daily)
    equity = daily("stockholdersEquity")
    intangibles = daily("intangibles")  # goodwill + other intangibles, COMBINED
    tax = _nopat_tax_rate(daily)

    if not oi.empty and not equity.empty:
        nopat = oi * (1.0 - tax) if not tax.empty else oi
        # invested capital now INCLUDES capitalized leases (shared definition), matching
        # how leases are already treated as debt in EV and in the leverage ratios.
        ic = cast(pd.DataFrame | None, capital.invested_capital(daily, operating_cash=operating_cash))
        roic_incl = ratio(nopat, ic, positive_den=True) if ic is not None else pd.DataFrame()
        if roic_incl.notna().any().any():
            features["roic_incl_intangibles"] = roic_incl
            if ic is not None and not intangibles.empty:
                roic_ex = ratio(nopat, ic.sub(intangibles, fill_value=0.0), positive_den=True)
                if roic_ex.notna().any().any():
                    features["roic_ex_intangibles"] = roic_ex
                    # incl - ex < 0 => acquired intangibles dilute returns (overpaid)
                    features["intangibles_roic_drag"] = roic_incl.sub(roic_ex, fill_value=np.nan)

    if not intangibles.empty and not assets.empty:
        gta = ratio(intangibles, assets, positive_den=True)
        if gta.notna().any().any():
            features["intangibles_to_assets"] = gta
        gte = ratio(intangibles, equity.where(equity > 0))
        if gte.notna().any().any():
            features["intangibles_to_equity"] = gte  # >1 => a writedown can wipe out book equity

    if isinstance(daily, PitFrames):
        sga_g = daily.change("sellingGeneralAdmin", periods=yoy_periods)
        rev_g = daily.change("totalRevenue", periods=yoy_periods)
    else:
        sga_g = fiscal_change_to_daily(fund_hist, "sellingGeneralAdmin", idx, kind="pct", periods=yoy_periods)
        rev_g = fiscal_change_to_daily(fund_hist, "totalRevenue", idx, kind="pct", periods=yoy_periods)
    if sga_g.notna().any().any() and rev_g.notna().any().any():
        el = ratio(sga_g, rev_g.where(rev_g.abs() >= 0.02))  # guard ~flat-revenue blow-ups
        if el.notna().any().any():
            features["sga_elasticity"] = el
    return features


def _per_share_and_profit_slice_fields(pit: PitFrames, close: pd.DataFrame | None, yoy_periods: int) -> dict:
    """Build per-share, dilution, dividend-growth, and NCI fields."""
    daily = pit
    features: dict[str, pd.DataFrame] = {}
    # (diluted - basic) / basic, already computed by the extractor on the periods where BOTH
    # counts are reported -- dividing the two independently forward-filled columns here would
    # compare a stale diluted count against a fresh basic one.
    overhang = daily("optionOverhang")
    if not overhang.empty and overhang.notna().any().any():
        features["option_overhang"] = overhang
    eps = daily("epsDiluted")
    if not eps.empty and close is not None:
        cols = eps.columns.intersection(close.columns)
        ey = ratio(eps[cols].where(eps[cols] > 0), close[cols], positive_den=True)
        if ey.notna().any().any():
            features["eps_yield"] = ey  # reported diluted EPS / price: E/P net of preferred
    dps_growth = pit.change("dividendsPerShare", periods=yoy_periods)
    if dps_growth.notna().any().any():
        features["dps_growth"] = dps_growth

    ni, nci = daily("netIncome"), daily("netIncomeToNci")
    if not nci.empty and not ni.empty:
        share = ratio(nci, ni.abs().where(ni.abs() > 0))
        if share.notna().any().any():
            features["nci_income_share"] = share
    return features


# --------------------------------------------------------------------------- #
# Per-block field builders                                                     #
#                                                                              #
# Each block consumes shared daily frames and returns {feature_name: date x ticker frame}.
# --------------------------------------------------------------------------- #
def _pension_pool(
    notes_num: pd.DataFrame | None, pension_facts: pd.DataFrame | None, idx: pd.DatetimeIndex
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return PBO, plan assets, and the best available net pension deficit."""
    pbo = _tagged_fact_daily(notes_num, _FN_PBO_TAG, idx)
    plan_assets = _tagged_fact_daily(notes_num, _FN_PLAN_ASSETS_TAG, idx)
    fn_deficit = pd.DataFrame()
    if not pbo.empty and not plan_assets.empty:
        # funded status = plan assets - PBO; the deficit (underfunding) is the debt-like part.
        fn_deficit = pbo.sub(plan_assets).clip(lower=0.0)

    pension_ret = _pension_deficit_daily(pension_facts, idx)
    if not fn_deficit.empty:
        pension_ret = pension_ret.combine_first(fn_deficit) if not pension_ret.empty else fn_deficit
    if not pension_ret.empty:
        pension_ret = pension_ret.clip(lower=0.0)  # underfunding only (>= 0)
    return pbo, plan_assets, pension_ret


def _pension_health_fields(pbo: pd.DataFrame, plan_assets: pd.DataFrame, pension_ret: pd.DataFrame) -> dict:
    """Build funded health and recognized-overhang features."""
    features: dict[str, pd.DataFrame] = {}
    if not pbo.empty and not plan_assets.empty:
        funded_ratio = ratio(plan_assets, pbo, positive_den=True)  # 1.0 = fully funded
        if funded_ratio.notna().any().any():
            features["pension_funded_ratio"] = funded_ratio
    if not pension_ret.empty and pension_ret.notna().any().any():
        features["pension_retirement_liability"] = pension_ret
    return features


def _pension_scale_fields(pension_ret: pd.DataFrame, pbo: pd.DataFrame, mcap: pd.DataFrame) -> dict:
    """Scale the net pension deficit and gross obligation by market cap."""
    features: dict[str, pd.DataFrame] = {}
    for name, src in (("pension_overhang_leverage", pension_ret), ("pbo_to_mcap", pbo)):
        if src is None or src.empty:
            continue
        r = ratio(src, mcap, positive_den=True)
        if r.notna().any().any():
            features[name] = r
    return features


def _valuation_yield_fields(mcap: pd.DataFrame, net_income: pd.DataFrame, revenue: pd.DataFrame, equity: pd.DataFrame, fcf: pd.DataFrame) -> dict:
    """Build positive-numerator valuation yields and a separate loss intensity."""
    return {
        "earnings_yield": ratio(net_income.where(net_income > 0), mcap, positive_den=True),
        "sales_yield": ratio(revenue, mcap, positive_den=True),
        "book_yield": ratio(equity.where(equity > 0), mcap, positive_den=True),
        "fcf_yield": ratio(fcf.where(fcf > 0), mcap, positive_den=True),
        "loss_intensity": ratio(-net_income.where(net_income < 0), mcap, positive_den=True),
    }


def _enterprise_value_frame(
    daily,
    close: pd.DataFrame | None,
    mcap: pd.DataFrame,
    equity: pd.DataFrame,
    d2e: pd.DataFrame,
    cash: pd.DataFrame,
    pension_ret: pd.DataFrame,
    operating_cash: frozenset[str] | None = None,
) -> pd.DataFrame:
    """Build fully diluted enterprise value from shared capital definitions."""
    diluted = daily("dilutedShares")
    fd_mcap = mcap
    if close is not None and not diluted.empty and diluted.notna().any().any():
        cols = diluted.columns.intersection(close.columns)
        fd = (close[cols] * diluted[cols]).where(lambda x: x > 0)
        fd_mcap = fd.combine_first(mcap)  # diluted where available, else basic
    debt = capital.total_debt(daily)
    if debt is None or debt.empty:
        debt = d2e.clip(lower=0.0) * equity.where(equity > 0) if not d2e.empty and not equity.empty else pd.DataFrame()
    liquid = capital.liquid_assets(daily, operating_cash=operating_cash)
    return _enterprise_value(
        fd_mcap,
        [debt, daily("minorityInterest"), daily("redeemableNCI"), daily("preferredEquity"), pension_ret],
        [liquid if liquid is not None else capital.drop_operating_cash(cash, operating_cash)],
    )


def _ev_yield_fields(ebitda: pd.DataFrame, fcf: pd.DataFrame, ev: pd.DataFrame) -> dict:
    """Build positive EBITDA and FCF yields against enterprise value."""
    features: dict[str, pd.DataFrame] = {}
    if not ebitda.empty:
        features["ebitda_to_ev"] = ratio(ebitda.where(ebitda > 0), ev, positive_den=True)
    fcf_to_ev = ratio(fcf.where(fcf > 0), ev, positive_den=True)
    if not fcf_to_ev.empty and fcf_to_ev.notna().any().any():
        features["fcf_to_ev"] = fcf_to_ev
    return features


def _altman_z_fields(daily, mcap: pd.DataFrame, revenue: pd.DataFrame) -> dict:
    """Build the market-value Altman Z score on the ex-lease asset base."""
    assets_z = capital.assets_ex_lease(daily)
    if assets_z.empty:
        return {}
    ta = assets_z.where(assets_z > 0)
    wc = daily("currentAssets").sub(daily("currentLiabilities"), fill_value=np.nan)
    z = (
        1.2 * ratio(wc, ta)
        + 1.4 * ratio(daily("retainedEarnings"), ta)
        + 3.3 * ratio(daily("operatingIncome"), ta)
        + 0.6 * ratio(mcap, daily("totalLiabilities"), positive_den=True)
        + 1.0 * ratio(revenue, ta)
    )
    if z.empty or not z.notna().any().any():
        return {}
    return {"altman_z": sanitize(z)}


def _pegy_fields(
    pit: PitFrames,
    mcap: pd.DataFrame,
    net_income: pd.DataFrame,
    yoy_periods: int,
    earnings_history: pd.DataFrame | None,
) -> dict:
    """Build PEGY, preferring projected EPS growth over realized growth."""
    daily = pit
    pe = ratio(mcap, net_income.where(net_income > 0), positive_den=True)
    growth_pct = None
    if earnings_history is not None and not earnings_history.empty:
        ntm_e, ttm_e = ntm_ttm_eps(earnings_history, pit.trading_index)
        if not ntm_e.empty and not ttm_e.empty:
            proj = ratio(ntm_e, ttm_e.where(ttm_e > 0)) - 1.0  # projected EPS growth
            if proj.notna().any().any():
                growth_pct = proj * 100.0
    if growth_pct is None:
        growth_pct = pit.change("netIncome", periods=yoy_periods) * 100.0
    div_yield_pct = ratio(daily("dividendsPaid"), mcap, positive_den=True) * 100.0
    denom = (growth_pct + div_yield_pct.fillna(0.0)).where(lambda x: x > 0)
    pegy = ratio(pe, denom)
    if pegy.empty or not pegy.notna().any().any():
        return {}
    return {"pegy": sanitize(pegy)}


def _reit_multiple_fields(daily, net_income: pd.DataFrame, mcap: pd.DataFrame, reit_tickers: set[str]) -> dict:
    """FFO yield, scoped directly to tickers in the GICS equity-REIT group."""
    if not reit_tickers:
        return {}
    ffo = net_income.add(daily("depAmort"), fill_value=0.0)
    fy = ratio(ffo, mcap, positive_den=True)
    fy.loc[:, ~fy.columns.isin(reit_tickers)] = np.nan
    return {"ffo_yield": fy} if fy.notna().any().any() else {}


def _profitability_level_fields(daily, revenue: pd.DataFrame, net_income: pd.DataFrame, fcf: pd.DataFrame) -> dict:
    """Build raw profitability, FCF margin, and accrual fields."""
    features: dict[str, pd.DataFrame] = {}
    for field in ["grossMargins", "operatingMargins", "profitMargins", "returnOnEquity", "debtToEquity", "revenueGrowth", "earningsGrowth"]:
        f = daily(field)
        if not f.empty:
            features[field] = f

    fcf_margin = ratio(fcf, revenue, positive_den=True)
    if not fcf_margin.empty:
        features["fcf_margin"] = fcf_margin
    if not net_income.empty and not fcf.empty and not revenue.empty:
        cols = net_income.columns.intersection(fcf.columns)
        accr_num = net_income[cols] - fcf[cols]
        features["accruals"] = ratio(accr_num, revenue, positive_den=True)
    return features


def _growth_trend_fields(pit: PitFrames, revenue: pd.DataFrame, rnd: pd.DataFrame, yoy_periods: int) -> dict:
    """Build fiscal growth, five-year margin trend, and R&D intensity fields."""
    daily = pit
    features: dict[str, pd.DataFrame] = {}
    for name, field in (("fcf_growth", "freeCashflow"), ("shares_growth", "sharesOutstanding")):
        chg = pit.change(field, periods=yoy_periods)
        if chg.notna().any().any():
            features[name] = chg
    gross_margin_chg = pit.change("grossMargins", kind="diff", periods=yoy_periods)
    if gross_margin_chg.notna().any().any():
        features["gross_margin_chg"] = gross_margin_chg

    op_margin = ratio(daily("operatingIncome"), revenue, positive_den=True)
    if not op_margin.empty and op_margin.notna().any().any():
        om_5y = sanitize(op_margin - op_margin.shift(_FIVE_YEARS))
        if om_5y.notna().any().any():
            features["operating_margin_5y_chg"] = om_5y

    rd_intensity = ratio(rnd, revenue, positive_den=True)
    if not rd_intensity.empty and rd_intensity.notna().any().any():
        features["rd_intensity"] = rd_intensity
    return features


def _reinvestment_fields(pit: PitFrames, yoy_periods: int) -> dict:
    """Compare depreciation and amortization with capex and its growth."""
    daily = pit
    features: dict[str, pd.DataFrame] = {}
    depamort, capex = daily("depAmort"), daily("capex")
    if not depamort.empty and not capex.empty:
        da_to_capex = ratio(depamort.abs(), capex.abs(), positive_den=True)
        if da_to_capex.notna().any().any():
            features["da_to_capex"] = da_to_capex
    da_growth = pit.change("depAmort", periods=yoy_periods)
    capex_growth = pit.change("capex", periods=yoy_periods)
    if da_growth.notna().any().any() and capex_growth.notna().any().any():
        features["da_minus_capex_growth"] = da_growth - capex_growth
    return features


def _quality_regime_fields(
    pit: PitFrames,
    revenue: pd.DataFrame,
    net_income: pd.DataFrame,
    fcf: pd.DataFrame,
    long_debt: pd.DataFrame,
    yoy_periods: int,
) -> dict:
    """Build quality fields that remain useful outside profitable regimes."""
    daily = pit
    features: dict[str, pd.DataFrame] = {}
    assets = capital.assets_ex_lease(daily)
    gm_lvl = daily("grossMargins")
    if not gm_lvl.empty and not revenue.empty and not assets.empty:
        cols = gm_lvl.columns.intersection(revenue.columns)
        gross_profit = gm_lvl[cols] * revenue[cols]  # grossMargins * revenue
        gp = ratio(gross_profit, assets, positive_den=True)
        if not gp.empty and gp.notna().any().any():
            features["gross_profitability"] = gp

    history = pit.history
    assets_field = "totalAssetsExLease" if history is not None and "totalAssetsExLease" in history.columns else "totalAssets"
    asset_growth = pit.change(assets_field, periods=yoy_periods)
    if asset_growth.notna().any().any():
        features["asset_growth"] = asset_growth

    rev_growth_pct = pit.change("totalRevenue", periods=yoy_periods) * 100.0
    fcf_margin_pct = ratio(fcf, revenue, positive_den=True) * 100.0
    rule40 = sanitize(rev_growth_pct + fcf_margin_pct)
    if rule40.notna().any().any():
        features["rule_of_40"] = rule40
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
            (_roa > 0),
            (_ocf > 0),
            (_roa > _roa.shift(y)),
            (ratio(_ocf, _oa, positive_den=True) > _roa),  # accruals: cash > profit
            (_lev < _lev.shift(y)),
            (_cr > _cr.shift(y)),
            (_sh <= _sh.shift(y) * 1.001),  # no net dilution
            (_gm > _gm.shift(y)),
            (_turn > _turn.shift(y)),
        ]
        fscore = cast(pd.DataFrame, sum(p.astype("float64") for p in parts))
        gate = _oa.notna() & net_income.notna() & _ocf.notna()
        fscore = fscore.where(gate)
        if fscore.notna().any().any():
            features["piotroski_f_score"] = fscore
    return features


def _state_flag_fields(daily, net_income: pd.DataFrame, fcf: pd.DataFrame, equity: pd.DataFrame) -> dict:
    """Build raw 0/1 regime flags while preserving missing inputs."""

    def _flag(base: pd.DataFrame, cond: pd.DataFrame) -> pd.DataFrame:
        return cond.astype(float).where(base.notna())

    features: dict[str, pd.DataFrame] = {}
    if not net_income.empty:
        features["profitable"] = _flag(net_income, net_income > 0)
    if not fcf.empty:
        features["fcf_positive"] = _flag(fcf, fcf > 0)
    if not equity.empty:
        features["negative_equity"] = _flag(equity, equity <= 0)
    rev_growth_lvl = daily("revenueGrowth")
    if not rev_growth_lvl.empty:
        features["hyper_growth"] = _flag(rev_growth_lvl, rev_growth_lvl > _HYPER_GROWTH)
    return features


def _quarter_momentum_fields(pit: PitFrames, yoy_periods: int) -> dict:
    """Build discrete-quarter growth, acceleration, and margin inflection."""
    daily = pit
    features: dict[str, pd.DataFrame] = {}
    q_rev_yoy = pit.applied("revenue_q", f"yoy_{yoy_periods}", lambda series: series.pct_change(yoy_periods))
    if q_rev_yoy.notna().any().any():
        features["q_rev_growth"] = q_rev_yoy
        # acceleration = this quarter's YoY minus the previous quarter's YoY
        features["rev_growth_accel"] = pit.applied(
            "revenue_q",
            f"yoy_accel_{yoy_periods}",
            lambda series: series.pct_change(yoy_periods).diff(1),
        )

    q_ni_yoy = pit.applied("netIncome_q", f"yoy_{yoy_periods}", lambda series: series.pct_change(yoy_periods))
    if q_ni_yoy.notna().any().any():
        features["q_earnings_growth"] = q_ni_yoy

    # latest-quarter margin vs TTM margin = margin inflection
    q_margin = ratio(daily("netIncome_q"), daily("revenue_q"), positive_den=True)
    profit_margins = daily("profitMargins")
    if not q_margin.empty and not profit_margins.empty:
        cols = q_margin.columns.intersection(profit_margins.columns)
        features["q_margin_vs_ttm"] = q_margin[cols] - profit_margins[cols]
    return features


def _yearly_ttm_momentum_fields(pit: PitFrames, yoy_periods: int) -> dict:
    """Build year-over-year TTM growth, acceleration, and margin change."""
    features: dict[str, pd.DataFrame] = {}
    y_rev_growth = pit.change("totalRevenue", periods=yoy_periods)
    if y_rev_growth.notna().any().any():
        features["y_rev_growth"] = y_rev_growth
        features["y_rev_growth_accel"] = pit.applied(
            "totalRevenue",
            f"yoy_accel_{yoy_periods}",
            lambda series: series.pct_change(yoy_periods).diff(1),
        )

    y_earnings_growth = pit.change("netIncome", periods=yoy_periods)
    if y_earnings_growth.notna().any().any():
        features["y_earnings_growth"] = y_earnings_growth

    # YoY change in TTM profit margin (margin expansion / contraction trend)
    y_margin_chg = pit.change("profitMargins", kind="diff", periods=yoy_periods)
    if y_margin_chg.notna().any().any():
        features["y_margin_vs_ttm"] = y_margin_chg
    return features


def _distress_fields(
    daily,
    ebitda: pd.DataFrame,
    cash: pd.DataFrame,
    fcf: pd.DataFrame,
    long_debt: pd.DataFrame,
    short_debt: pd.DataFrame,
    operating_cash: frozenset[str] | None = None,
) -> dict:
    """Build debt service, coverage, liquidity, and refinancing-risk fields."""
    features: dict[str, pd.DataFrame] = {}
    total_debt = _combine_debt(daily, long_debt, short_debt, daily("totalLiabilities"))
    if not total_debt.empty and not ebitda.empty:
        net_cash = capital.drop_operating_cash(cash, operating_cash)
        cols = total_debt.columns.intersection(net_cash.columns) if not net_cash.empty else total_debt.columns
        net_debt = total_debt[cols].sub(net_cash[cols], fill_value=0.0) if not net_cash.empty else total_debt
        # HIGH net-debt/EBITDA = more leveraged = worse (only meaningful for EBITDA>0)
        nd_ebitda = ratio(net_debt, ebitda, positive_den=True)
        if not nd_ebitda.empty and nd_ebitda.notna().any().any():
            features["net_debt_to_ebitda"] = nd_ebitda
    interest = daily("interestExpense")
    if not ebitda.empty and not interest.empty:
        # HIGH coverage = safer. Interest is an expense (take abs to be sign-safe).
        cov = ratio(ebitda, interest.abs(), positive_den=True)
        if not cov.empty and cov.notna().any().any():
            features["interest_coverage"] = cov
    current_ratio = ratio(daily("currentAssets"), daily("currentLiabilities"), positive_den=True)
    if not current_ratio.empty and current_ratio.notna().any().any():
        features["current_ratio"] = current_ratio
    if not cash.empty and not total_debt.empty:
        cash_to_debt = ratio(cash, total_debt, positive_den=True)
        if not cash_to_debt.empty and cash_to_debt.notna().any().any():
            features["cash_to_debt"] = cash_to_debt

    liquidity = cash.add(fcf.where(fcf > 0), fill_value=0.0)
    refi = ratio(short_debt, liquidity, positive_den=True)
    if not refi.empty and refi.notna().any().any():
        features["refinancing_risk"] = refi
    return features


def _sga_efficiency_fields(pit: PitFrames, revenue: pd.DataFrame, yoy_periods: int) -> dict:
    """Build SG&A intensity, growth, and operating leverage."""
    daily = pit
    features: dict[str, pd.DataFrame] = {}
    sga_intensity = ratio(daily("sellingGeneralAdmin"), revenue, positive_den=True)
    if not sga_intensity.empty and sga_intensity.notna().any().any():
        features["sga_intensity"] = sga_intensity
    sga_growth = pit.change("sellingGeneralAdmin", periods=yoy_periods)
    rev_growth = pit.change("totalRevenue", periods=yoy_periods)
    if sga_growth.notna().any().any():
        features["sga_growth"] = sga_growth
        if rev_growth.notna().any().any():
            cols = rev_growth.columns.intersection(sga_growth.columns)
            features["operating_leverage"] = rev_growth[cols] - sga_growth[cols]
    return features


def _ma_footprint_fields(pit: PitFrames, revenue: pd.DataFrame, yoy_periods: int) -> dict:
    """M&A footprint: organic vs inorganic growth, and impairment risk."""
    daily = pit
    features: dict[str, pd.DataFrame] = {}
    # Sharadar's `ncfbus`: net cash paid for acquisitions, stored outflow-negative, so the
    # magnitude is what "how acquisitive is this firm" means. 27,731 negative rows to 9,039
    # positive (a positive row is a net DISPOSAL year), hence `.abs()` rather than a floor:
    # both directions are M&A activity, and the intensity is unsigned by construction.
    acq = daily("businessAcquisitionsNet")
    assets = capital.assets_ex_lease(daily)
    acq_den = assets if not assets.empty else revenue
    acq_intensity = ratio(acq.abs() if not acq.empty else acq, acq_den, positive_den=True)
    if not acq_intensity.empty and acq_intensity.notna().any().any():
        features["acquisition_intensity"] = acq_intensity
    # YoY growth in the acquired-intangibles balance: the balance-sheet trace of M&A, and
    # the exposure a future writedown lands on. On Sharadar's COMBINED `intangibles` -- the
    # bare `goodwill` this read before is written by no producer, so it grew nothing.
    intangibles_growth = pit.change("intangibles", periods=yoy_periods)
    if intangibles_growth.notna().any().any():
        features["intangibles_growth"] = intangibles_growth
    return features


def _sbc_fields(daily, revenue: pd.DataFrame, sbc: pd.DataFrame) -> dict:
    """Build stock-based compensation intensity and cash-flow share."""
    features: dict[str, pd.DataFrame] = {}
    sbc_intensity = ratio(sbc, revenue, positive_den=True)
    if not sbc_intensity.empty and sbc_intensity.notna().any().any():
        features["sbc_intensity"] = sbc_intensity
    ocf = daily("operatingCashFlow")
    if not sbc.empty and not ocf.empty:
        sbc_to_ocf = ratio(sbc, ocf, positive_den=True)
        if not sbc_to_ocf.empty and sbc_to_ocf.notna().any().any():
            features["sbc_to_ocf"] = sbc_to_ocf
    return features


def _valuation_engine_fields(pit: PitFrames, revenue: pd.DataFrame, ebitda: pd.DataFrame, yoy_periods: int) -> dict:
    """Build operating, margin, working-capital, dilution, and coverage fields."""
    daily = pit
    features: dict[str, pd.DataFrame] = {}
    oi_growth = pit.change("operatingIncome", periods=yoy_periods)
    rev_growth_f = pit.change("totalRevenue", periods=yoy_periods)
    ol_el = ratio(oi_growth, rev_growth_f.where(rev_growth_f.abs() >= 0.02))
    if not ol_el.empty and ol_el.notna().any().any():
        features["operating_leverage_elasticity"] = ol_el

    gm = ratio(daily("grossProfit"), revenue, positive_den=True)
    if gm.empty:
        gm = daily("grossMargins")
    em = ratio(ebitda, revenue, positive_den=True)
    if not gm.empty and not em.empty:
        med = (gm - gm.shift(252)) - (em - em.shift(252))  # ~1y change divergence
        if med.notna().any().any():
            features["margin_expansion_delta"] = sanitize(med)

    cur_a2, cur_l2 = daily("currentAssets"), daily("currentLiabilities")
    if not cur_a2.empty and not cur_l2.empty and not revenue.empty:
        nwc = cur_a2.sub(cur_l2, fill_value=np.nan)
        prev_nwc = nwc.shift(252)
        nwc_g = ratio(nwc - prev_nwc, prev_nwc, positive_den=True)
        prev_rev = revenue.shift(252)
        rev_g = ratio(revenue - prev_rev, prev_rev, positive_den=True)
        nwc_el = ratio(nwc_g, rev_g.where(rev_g.abs() >= 0.02))
        if not nwc_el.empty and nwc_el.notna().any().any():
            features["nwc_elasticity"] = nwc_el

    dil_growth = pit.change("dilutedShares", periods=yoy_periods)
    if dil_growth.notna().any().any():
        features["diluted_shares_growth"] = dil_growth

    eic = ratio(daily("operatingIncome"), daily("interestExpense").abs(), positive_den=True)
    if not eic.empty and eic.notna().any().any():
        features["ebit_interest_coverage"] = eic
    return features


def _intrinsic_fields(
    fund_hist: pd.DataFrame, close: pd.DataFrame | None, idx: pd.DatetimeIndex, intrinsic_cfg: dict | None, level_factor: pd.DataFrame | None = None
) -> dict:
    """Build two-stage DCF yield, using default parameters when config is absent."""
    if close is None:
        return {}
    iy = intrinsic_value_daily(fund_hist, close, idx, level_factor=level_factor, **(intrinsic_cfg or {})).get("yield")
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
    pit: PitFrames | None = None,
) -> dict[str, pd.DataFrame]:
    """Compute source-specific daily feature frames from cleaned inputs."""
    if pit is None:
        fund_hist = clean_fundamentals_history(fund_hist)
        pit = PitFrames(fund_hist, idx, close, level_factor)
    else:
        pit.assert_matches(idx, close)
        if pit.history is None:
            raise ValueError(f"{Tables.fundamentals_history} is unavailable")
        fund_hist = pit.history

    daily = pit
    features: dict[str, pd.DataFrame] = {}
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
    universe = set(fund_hist["ticker"].astype(str)) if "ticker" in fund_hist.columns else set()
    operating_cash = frozenset(family_tickers(fund_hist, "bank") | family_tickers(fund_hist, "insurance"))
    reit_tickers = family_tickers(fund_hist, "reit")
    mcap = pit.market_cap

    # Root table: compute its features before moving to supplemental SEC facts.
    if not mcap.empty:
        features.update(_valuation_yield_fields(mcap, net_income, revenue, equity, fcf))
        features.update(_altman_z_fields(daily, mcap, revenue))
        features.update(_pegy_fields(pit, mcap, net_income, yoy_periods, earnings_history))
        features.update(_reit_multiple_fields(daily, net_income, mcap, reit_tickers))

    features.update(_profitability_level_fields(daily, revenue, net_income, fcf))
    features.update(_growth_trend_fields(pit, revenue, rnd, yoy_periods))
    features.update(_reinvestment_fields(pit, yoy_periods))
    features.update(_quality_regime_fields(pit, revenue, net_income, fcf, long_debt, yoy_periods))
    features.update(_state_flag_fields(daily, net_income, fcf, equity))
    features.update(_quarter_momentum_fields(pit, yoy_periods))
    features.update(_yearly_ttm_momentum_fields(pit, yoy_periods))
    features.update(_distress_fields(daily, ebitda, cash, fcf, long_debt, short_debt, operating_cash))
    features.update(_sga_efficiency_fields(pit, revenue, yoy_periods))
    features.update(_ma_footprint_fields(pit, revenue, yoy_periods))
    features.update(_sbc_fields(daily, revenue, sbc))
    features.update(_valuation_engine_fields(pit, revenue, ebitda, yoy_periods))
    features.update(_intrinsic_fields(fund_hist, close, idx, intrinsic_cfg, level_factor))
    features.update(_da_realism_fields(daily))
    features.update(_forensic_fields(daily, idx))
    features.update(_digestion_fields(daily, fund_hist, idx, yoy_periods, operating_cash))
    features.update(_per_share_and_profit_slice_fields(pit, close, yoy_periods))

    # Supplemental tables: clean once, build their own features, then compute the EV
    # features whose capital structure includes the pension deficit.
    pension_facts = _clean_tagged_facts(pension_facts, universe)
    notes_num = _clean_tagged_facts(notes_num, universe)
    pbo, plan_assets, pension_ret = _pension_pool(notes_num, pension_facts, idx)
    features.update(_pension_health_fields(pbo, plan_assets, pension_ret))
    if not mcap.empty:
        features.update(_pension_scale_fields(pension_ret, pbo, mcap))
        ev = _enterprise_value_frame(daily, close, mcap, equity, d2e, cash, pension_ret, operating_cash)
        features.update(_ev_yield_fields(ebitda, fcf, ev))

    return features


def _merge_feature_panels(panels: list[pd.DataFrame]) -> pd.DataFrame:
    """Combine non-empty long panels on their shared key."""
    indexed = [
        panel.set_index(["date", "ticker"]) for panel in panels if panel is not None and not panel.empty and list(panel.columns) != ["date", "ticker"]
    ]
    return pd.concat(indexed, axis=1).reset_index() if indexed else pd.DataFrame(columns=["date", "ticker"])


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
    columns. The input frames are the output of `self_history_z` (final signal),
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
    hist_window: int = HIST_WINDOW,
    hist_min_periods: int = HIST_MIN_PERIODS,
    earnings_history: pd.DataFrame | None = None,
    pension_facts: pd.DataFrame | None = None,
    notes_num: pd.DataFrame | None = None,
    level_factor: pd.DataFrame | None = None,
    pit: PitFrames | None = None,
    availability: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the long peer, cross-sectional, self-history, and regime panel."""

    if fundamentals_history is None or fundamentals_history.empty:
        raise ValueError("Need to build fundamentals_history table first")

    if pit is None:
        fundamentals_history = clean_fundamentals_history(fundamentals_history)
        pit = PitFrames(fundamentals_history, trading_index, stock_close, level_factor)
    else:
        pit.assert_matches(trading_index, stock_close)
    yoy_periods = pit.yoy_periods
    fields = _derived_fields(
        fund_hist=fundamentals_history,
        idx=trading_index,
        close=stock_close,
        yoy_periods=yoy_periods,
        intrinsic_cfg=intrinsic_cfg,
        earnings_history=earnings_history,
        pension_facts=pension_facts,
        notes_num=notes_num,
        level_factor=level_factor,
        pit=pit,
    )
    fields = _postprocess_feature_frames(fields)
    fields = {name: mask_to_availability(frame, availability) for name, frame in fields.items()}

    # regime state flags -> RAW `f_<name>`; everything else -> peer-relative.
    state_fields = {k: v for k, v in fields.items() if k in _STATE_FIELDS}
    peer_fields = {k: v for k, v in fields.items() if k not in _STATE_FIELDS}

    peer_panel = build_peer_relative_panel(peer_fields, peer_dict)

    # Self-history (mean-reversion) z-scores on the valuation yields only.
    hist_fields = {
        name: self_history_z(fields[name], window=hist_window, min_periods=hist_min_periods)
        for name in _MEAN_REVERSION_FIELDS
        if name in fields and fields[name] is not None and not fields[name].empty
    }
    hist_panel = build_self_history_panel(hist_fields)
    state_panel = build_state_panel(state_fields)

    return _merge_feature_panels([peer_panel, hist_panel, state_panel])
