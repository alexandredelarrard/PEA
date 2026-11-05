# ruff: noqa: N806
"""Point-in-time fundamental features with supported issuer-valued denominators.

Saved capitalization supplies issuer yields; vendor per-share inputs and comparison quantities keep distinct bases."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from src.constants.constants import SEC_NET_PENSION_LIABILITY_TAGS
from src.context import Context
from src.data_aggregate.utils.common import capital
from src.data_aggregate.utils.common.frames import ratio, sanitize
from src.data_aggregate.utils.common.pit import (
    fiscal_change_to_daily,
    fiscal_change_values,
    fiscal_prior_positions,
    fiscal_prior_to_daily,
    fiscal_prior_values,
    fiscal_values_to_daily,
    fundamentals_to_daily,
    infer_yoy_periods,
    null_invalid_gross_profit_sentinels,
)
from src.data_aggregate.utils.common.sector_gates import family_tickers, mask_columns, row_gate
from src.data_aggregate.utils.common.xs import (
    self_history_z,
)
from src.data_aggregate.utils.fundamentals.earnings_features import reported_events
from src.data_aggregate.utils.fundamentals.feature_views import build_fundamental_views
from src.data_aggregate.utils.fundamentals.history_features import reported_history_fields
from src.data_aggregate.utils.fundamentals.inputs import FundamentalFeatureSettings, FundamentalHistory, FundamentalSupplement
from src.data_aggregate.utils.fundamentals.intrinsic import intrinsic_value_daily
from src.data_aggregate.utils.price.frames import PriceFrames, comparison_share_history
from src.utils.price.adjustments import ShareBasis
from src.utils.price.per_share import ordinary_per_share_to_security, same_capital_scope, vendor_ordinary_per_share

_NOTES_NUM_TABLE = "notes_num"  # footnote NUMERIC facts (the panel reads 4 tags)
_FACT_COLS = ["ticker", "tag", "ddate", "qtrs", "value", "filed", "available_at", "adsh"]


FUNDAMENTAL_TRANSFORM_SEMANTICS = {
    "nci_income_share": "structural_zero",
    "rd_intensity": "structural_zero",
}
_HYPER_GROWTH = 0.25  # YoY revenue growth above which a name is "hyper-growth"
_QUARTERLY_MAX_AGE_DAYS = 185
_ANNUAL_MAX_AGE_DAYS = 460


@dataclass(frozen=True, slots=True)
class _FiscalView:
    """Reuse the comparison history and cached fiscal projections for one build."""

    comparison_history: pd.DataFrame
    trading_index: pd.DatetimeIndex
    yoy_periods: int
    daily: Callable[[str], pd.DataFrame]
    prior: Callable[..., pd.DataFrame] | None
    operating_cash: frozenset[str]


def _previous_fiscal_value(
    fund_hist: pd.DataFrame,
    values: pd.Series,
) -> pd.Series:
    """Previous fiscal quarter's row-aligned value, with a legacy row fallback."""
    if "fiscal_end" in fund_hist.columns and fund_hist["fiscal_end"].notna().any():
        return fiscal_prior_values(
            fund_hist,
            values,
            years=0,
            tolerance_days=120,
        )
    work = pd.DataFrame(
        {
            "ticker": fund_hist["ticker"].astype(str),
            "as_of": pd.to_datetime(fund_hist["as_of"], errors="coerce"),
            "value": values.to_numpy(),
            "position": np.arange(len(fund_hist)),
        }
    ).sort_values(["ticker", "as_of", "position"])
    work["prior"] = work.groupby("ticker", sort=False)["value"].shift(1)
    result = np.full(len(fund_hist), np.nan, dtype="float64")
    result[work["position"].to_numpy(dtype="int64")] = work["prior"].to_numpy(dtype="float64", na_value=np.nan)
    return pd.Series(result, index=fund_hist.index)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _combine_debt(daily, long_debt: pd.DataFrame, short_debt: pd.DataFrame, fallback: pd.DataFrame) -> pd.DataFrame:
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
    """Build enterprise value from supported equity and available senior claims/liquid assets."""
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
# Each returns a {name: daily wide frame} dict that _derived_fields merges into F;
# `feature_views` then applies the approved raw/self-history contract downstream.
# `daily` is the memoized accessor from _derived_fields (field -> date x ticker).
# --------------------------------------------------------------------------- #
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
    buyback = capital.share_repurchases(daily)  # positive magnitude; see capital.py
    if not sbc.empty and buyback is not None and not buyback.empty:
        s2b = ratio(sbc, buyback, positive_den=True)
        if s2b.notna().any().any():
            F["sbc_to_buyback"] = s2b  # >1 = buybacks don't even cover SBC
    return F


def _beneish_m_score(daily, idx: pd.DatetimeIndex, prior=None) -> pd.DataFrame:
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

    previous = prior if prior is not None else lambda _field, *_args: pd.DataFrame()
    prior_rev = previous("totalRevenue")
    prior_assets = previous("totalAssetsExLease")
    if prior_assets.empty:
        prior_assets = previous("totalAssets")
    ar_sales = ratio(ar, rev)
    prior_ar_sales = ratio(previous("accountsReceivable"), prior_rev)
    dsri = ix(ar_sales, prior_ar_sales)  # days sales in receivables
    gm = ratio(gp, rev, positive_den=True)
    prior_gm = ratio(previous("grossProfit"), prior_rev, positive_den=True)
    gmi = ix(prior_gm, gm)  # gross-margin deterioration
    noncore = 1.0 - ratio(ca.add(ppe, fill_value=0.0), assets, positive_den=True)
    prior_noncore = 1.0 - ratio(
        previous("currentAssets").add(previous("ppeNet"), fill_value=0.0),
        prior_assets,
        positive_den=True,
    )
    aqi = ix(noncore, prior_noncore)  # asset-quality (soft assets)
    sgi = ix(rev, prior_rev)  # sales growth
    deprate = ratio(dep, dep.add(ppe, fill_value=0.0), positive_den=True)
    prior_dep = previous("depAmort")
    prior_deprate = ratio(
        prior_dep,
        prior_dep.add(previous("ppeNet"), fill_value=0.0),
        positive_den=True,
    )
    depi = ix(prior_deprate, deprate)  # slowing depreciation
    sgar = ratio(sga, rev, positive_den=True)
    prior_sgar = ratio(previous("sellingGeneralAdmin"), prior_rev, positive_den=True)
    sgai = ix(sgar, prior_sgar)  # SG&A efficiency
    lev = ratio(ltd.add(cl, fill_value=0.0), assets, positive_den=True)
    prior_lev = ratio(
        previous("longTermDebt").add(previous("currentLiabilities"), fill_value=0.0),
        prior_assets,
        positive_den=True,
    )
    lvgi = ix(lev, prior_lev)  # leverage change
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
    return m.where(support).replace([np.inf, -np.inf], np.nan)


# Footnote pension tags from the NOTES sets (`notes_num`): the projected benefit
# obligation and fair value of plan assets, both INSTANT (balance-type) facts the
# primary statements never expose.
_FN_PBO_TAG = "DefinedBenefitPlanBenefitObligation"  # projected benefit obligation
_FN_PLAN_ASSETS_TAG = "DefinedBenefitPlanFairValueOfPlanAssets"


def _scope_to_universe(facts: pd.DataFrame | None, universe: set[str]) -> pd.DataFrame | None:
    """A bulk facts frame restricted to the tickers the fundamentals panel is being built
    for. A no-op when the universe is unknown, so a caller that hands over a frame without
    a `ticker` column still gets the old behaviour rather than an empty one."""
    if facts is None or facts.empty or not universe or "ticker" not in facts.columns:
        return facts
    return facts[facts["ticker"].astype(str).isin(universe)]


def _net_pension_daily(notes_num: pd.DataFrame | None, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Recognized net DB-pension liability from the two `notes_num` balance-sheet tags,
    point-in-time. The primary tag is preferred; the pension-only variant fills gaps.
    Empty frame if `notes_num` is unavailable."""
    primary = _notes_num_daily(notes_num, SEC_NET_PENSION_LIABILITY_TAGS[0], idx)
    variant = _notes_num_daily(notes_num, SEC_NET_PENSION_LIABILITY_TAGS[1], idx)
    if primary.empty:
        return variant
    if variant.empty:
        return primary
    return primary.combine_first(variant)


def _notes_num_daily(notes_num: pd.DataFrame | None, tag: str, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """One tag from the NOTES sets (`notes_num`) -> daily wide frame, point-in-time on
    the later of filing and archive availability, latest period-end (`ddate`) per
    filing; the later filing date, then the accession, breaks a tie between filings.
    Keeps the balance-type facts only (qtrs==0, e.g. PBO / plan assets). Empty frame if unavailable."""
    if notes_num is None or notes_num.empty or not {"tag", "ticker", "filed", "available_at", "adsh"} <= set(notes_num.columns):
        return pd.DataFrame(index=idx)
    d = notes_num[notes_num["tag"] == tag].copy()
    if d.empty:
        return pd.DataFrame(index=idx)
    filed = pd.to_datetime(d["filed"], errors="coerce")
    available = pd.to_datetime(d["available_at"], errors="coerce")
    valid_clock = filed.notna() & available.notna()
    d = d[valid_clock].copy()
    filed = filed[valid_clock]
    available = available[valid_clock]
    d["as_of"] = filed.where(filed >= available, available)
    d["value"] = pd.to_numeric(d.get("value"), errors="coerce")
    q = pd.to_numeric(d.get("qtrs"), errors="coerce").fillna(0)
    d = d[q == 0]
    d = d.dropna(subset=["as_of", "value", "ticker"])
    if d.empty:
        return pd.DataFrame(index=idx)
    d["filed"] = filed
    d = d.sort_values(["ticker", "as_of", "ddate", "filed", "adsh"]).rename(columns={"value": tag})
    return fundamentals_to_daily(
        d,
        tag,
        idx,
        max_age_days=_ANNUAL_MAX_AGE_DAYS,
    )


def load_notes_num_scoped(
    context: Context,
    *,
    tickers: tuple[str, ...] | None = None,
    since: pd.Timestamp | None = None,
) -> pd.DataFrame | None:
    """`notes_num` restricted to the 4 tags the pension features read (PBO, plan assets and
    the two recognized net-liability tags), the tag filter pushed down server-side in one
    read. None if the table is absent/empty, lacks `available_at`, or no row matches."""
    if "available_at" not in context.store.columns(_NOTES_NUM_TABLE):
        return None
    where: dict[str, object] = {"tag": [_FN_PBO_TAG, _FN_PLAN_ASSETS_TAG, *SEC_NET_PENSION_LIABILITY_TAGS]}
    if tickers:
        where["ticker"] = list(tickers)
    df = context.store.load(_NOTES_NUM_TABLE, columns=_FACT_COLS, where=where, since=since, optional=True)
    return df.reset_index(drop=True) if df is not None else None


def _forensic_fields(daily, idx: pd.DatetimeIndex, prior=None) -> dict:
    """#5 -- accounting-quality / hidden-leverage red flags: working-capital days
    and their year-over-year drift (supplier-funded growth = rising DPO; channel
    stuffing = rising DSO), off-balance-sheet-INCLUSIVE net leverage (adds lease
    liabilities + pension deficit), and the Beneish M-score. All inputs already
    extracted; `pension_deficit` is the recognized net pension liability."""
    F: dict[str, pd.DataFrame] = {}
    rev, cogs = daily("totalRevenue"), daily("costOfRevenue")
    ar, ap, inv = daily("accountsReceivable"), daily("accountsPayable"), daily("inventory")

    dso = ratio(ar, rev, positive_den=True) * 365.0
    if dso.notna().any().any():
        F["dso"] = dso
        if prior is not None:
            prior_dso = (
                ratio(
                    prior("accountsReceivable"),
                    prior("totalRevenue"),
                    positive_den=True,
                )
                * 365.0
            )
            F["dso_change"] = (dso - prior_dso).replace([np.inf, -np.inf], np.nan)
    dpo = ratio(ap, cogs, positive_den=True) * 365.0
    if dpo.notna().any().any():
        F["dpo"] = dpo
        if prior is not None:
            prior_dpo = (
                ratio(
                    prior("accountsPayable"),
                    prior("costOfRevenue"),
                    positive_den=True,
                )
                * 365.0
            )
            F["dpo_change"] = (dpo - prior_dpo).replace([np.inf, -np.inf], np.nan)
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
    m = _beneish_m_score(daily, idx, prior)
    if not m.empty and m.notna().any().any():
        F["beneish_m_score"] = m
    return F


def _digestion_fields(daily, fund_hist: pd.DataFrame, idx: pd.DatetimeIndex, yoy_periods: int, operating_cash: frozenset[str] | None = None) -> dict:
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
    the data supports, and it is the same deduction for every ticker, so the raw ratio and
    its self-history change stay interpretable."""
    F: dict[str, pd.DataFrame] = {}
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
            F["intangibles_to_equity"] = gte  # >1 => a writedown can wipe out book equity

    sga_g = fiscal_change_to_daily(
        fund_hist,
        "sellingGeneralAdmin",
        idx,
        kind="pct",
        periods=yoy_periods,
        years=1,
        max_age_days=_QUARTERLY_MAX_AGE_DAYS,
    )
    rev_g = fiscal_change_to_daily(
        fund_hist,
        "totalRevenue",
        idx,
        kind="pct",
        periods=yoy_periods,
        years=1,
        max_age_days=_QUARTERLY_MAX_AGE_DAYS,
    )
    if sga_g.notna().any().any() and rev_g.notna().any().any():
        el = ratio(sga_g, rev_g.where(rev_g.abs() >= 0.02))  # guard ~flat-revenue blow-ups
        if el.notna().any().any():
            F["sga_elasticity"] = el
    return F


def _financial_eps_yield(fund_hist: pd.DataFrame, prices: PriceFrames | None, idx: pd.DatetimeIndex, share_basis: ShareBasis | None) -> pd.DataFrame:
    """Normalize vendor-ordinary EPS on its public clock before using saved canonical quotes."""
    if prices is None or prices.close_pit is None or share_basis is None or "epsDiluted" not in fund_hist:
        return pd.DataFrame(index=idx)
    terminal = pd.Timestamp(idx.max())
    targets = pd.Series(terminal, index=fund_hist.index)
    values = vendor_ordinary_per_share(share_basis, fund_hist["epsDiluted"], fund_hist["ticker"], fund_hist["as_of"], targets)
    df_eps = fiscal_values_to_daily(fund_hist, values, idx, max_age_days=_QUARTERLY_MAX_AGE_DAYS, preserve_nulls=True)
    source = pd.Series(terminal, index=idx)
    destination = pd.Series(idx, index=idx)
    for ticker in df_eps:
        df_eps[ticker] = ordinary_per_share_to_security(share_basis, df_eps[ticker], pd.Series(str(ticker), index=idx), source, destination)
    return ratio(df_eps.where(df_eps.gt(0)), prices.close_pit.reindex(index=idx), positive_den=True)


def _per_share_and_profit_slice_fields(
    daily: Callable[[str], pd.DataFrame], comparison_history: pd.DataFrame, idx: pd.DatetimeIndex, yoy_periods: int, df_eps_yield: pd.DataFrame
) -> dict:
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
    if not df_eps_yield.empty and df_eps_yield.notna().any().any():
        F["eps_yield"] = df_eps_yield
    dps_growth = fiscal_change_to_daily(
        comparison_history, "dividendsPerShare", idx, kind="pct", periods=yoy_periods, years=1, max_age_days=_QUARTERLY_MAX_AGE_DAYS
    )
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
def _pension_pool(notes_num: pd.DataFrame | None, idx: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The three SHARED pension frames: (pbo, footnote deficit, coalesced overhang).

    Not features -- these feed the EV calculation, the forensic block and the scale
    features below, which is why they are computed once here rather than per block.

    The overhang is gap-filled (`combine_first`, NOT a sum -> no OPEB double-count)
    across two `notes_num` sources in order of directness:
      1) the recognized net liability (the two balance-sheet tags),
      2) the footnote funded status = PBO - plan assets."""
    pbo = _notes_num_daily(notes_num, _FN_PBO_TAG, idx)
    plan_assets = _notes_num_daily(notes_num, _FN_PLAN_ASSETS_TAG, idx)
    fn_deficit = pd.DataFrame()
    if not pbo.empty and not plan_assets.empty:
        # funded status = plan assets - PBO; the deficit (underfunding) is the debt-like part.
        fn_deficit = pbo.sub(plan_assets).clip(lower=0.0)

    pension_ret = _net_pension_daily(notes_num, idx)
    if not fn_deficit.empty:
        pension_ret = pension_ret.combine_first(fn_deficit) if not pension_ret.empty else fn_deficit
    if not pension_ret.empty:
        pension_ret = pension_ret.clip(lower=0.0)  # underfunding only (>= 0)
    return pbo, fn_deficit, pension_ret


def _pension_health_fields(daily, pbo: pd.DataFrame, notes_num: pd.DataFrame | None, idx: pd.DatetimeIndex, pension_ret: pd.DataFrame) -> dict:
    """Funded health straight off the footnote, plus the recognized overhang level.
    The NOTES footnote (PBO / plan assets) yields a standalone funded ratio that covers
    names the balance-sheet tag misses."""
    F: dict[str, pd.DataFrame] = {}
    plan_assets = _notes_num_daily(notes_num, _FN_PLAN_ASSETS_TAG, idx)
    if not pbo.empty and not plan_assets.empty:
        funded_ratio = ratio(plan_assets, pbo, positive_den=True)  # 1.0 = fully funded
        if funded_ratio.notna().any().any():
            F["pension_funded_ratio"] = funded_ratio
    if not pension_ret.empty and pension_ret.notna().any().any():
        F["pension_retirement_liability"] = pension_ret
    return F


def _pension_scale_fields(pension_ret: pd.DataFrame, pbo: pd.DataFrame, mcap: pd.DataFrame) -> dict:
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
    for name, src in (("pension_overhang_leverage", pension_ret), ("pbo_to_mcap", pbo)):
        if src is None or src.empty:
            continue
        r = ratio(src, mcap, positive_den=True)
        if r.notna().any().any():
            F[name] = r
    return F


def _valuation_yield_fields(
    mcap: pd.DataFrame, common_income: pd.DataFrame, consolidated_income: pd.DataFrame, revenue: pd.DataFrame, equity: pd.DataFrame, fcf: pd.DataFrame
) -> dict:
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
    profitable name into the same artificial raw state. Given its own column the model can
    learn the reversed sign directly instead of having it cancel inside `earnings_yield`."""
    return {
        "earnings_yield": ratio(common_income.where(common_income > 0), mcap, positive_den=True),
        "sales_yield": ratio(revenue, mcap, positive_den=True),
        "book_yield": ratio(equity.where(equity > 0), mcap, positive_den=True),
        "fcf_yield": ratio(fcf.where(fcf > 0), mcap, positive_den=True),
        "loss_intensity": ratio(
            -consolidated_income.where(consolidated_income < 0),
            mcap,
            positive_den=True,
        ),
    }


def _diluted_capital(fund_hist: pd.DataFrame, idx: pd.DatetimeIndex, prices: PriceFrames | None, df_cap: pd.DataFrame) -> pd.DataFrame:
    """Scale supported same-scope issuer cap by a co-observed vendor dilution ratio."""
    if prices is None or not {"dilutedShares", "sharesOutstanding"}.issubset(fund_hist):
        return df_cap
    counts = pd.to_numeric(fund_hist["sharesOutstanding"], errors="coerce")
    diluted = pd.to_numeric(fund_hist["dilutedShares"], errors="coerce")
    ratio_values = diluted / counts.where(counts.gt(0))
    ratio_values = ratio_values.where(np.isfinite(ratio_values) & ratio_values.gt(0))
    df_ratio = fiscal_values_to_daily(fund_hist, ratio_values, idx, max_age_days=_QUARTERLY_MAX_AGE_DAYS, preserve_nulls=True)
    df_supported = same_capital_scope(prices.close_pit, prices.shares_outstanding_pit, df_cap)
    df_diluted = df_cap * df_ratio.reindex_like(df_cap)
    return df_diluted.where(df_supported & np.isfinite(df_diluted) & df_diluted.gt(0)).combine_first(df_cap)


def _enterprise_value_frame(view: _FiscalView, mcap: pd.DataFrame, pension_ret: pd.DataFrame) -> pd.DataFrame:
    """Add senior claims and obligations, then subtract non-operating liquid assets."""
    daily = view.daily
    equity, d2e, cash = daily("stockholdersEquity"), daily("debtToEquity"), daily("cash")
    operating_cash = view.operating_cash
    debt = capital.total_debt(daily)
    if debt is None or debt.empty:
        debt = d2e.clip(lower=0.0) * equity.where(equity > 0) if not d2e.empty and not equity.empty else pd.DataFrame()
    liquid = capital.liquid_assets(daily, operating_cash=operating_cash)
    return _enterprise_value(
        mcap,
        [debt, daily("minorityInterest"), daily("redeemableNCI"), daily("preferredEquity"), pension_ret],
        [liquid if liquid is not None else capital.drop_operating_cash(cash, operating_cash)],
    )


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
    view: _FiscalView, fund_hist: pd.DataFrame, mcap: pd.DataFrame, net_income: pd.DataFrame, earnings_history: pd.DataFrame | None
) -> dict[str, pd.DataFrame]:
    """Value trailing earnings against then-public EPS growth and dividend yield."""
    daily, idx, yoy_periods = view.daily, view.trading_index, view.yoy_periods
    pe = ratio(mcap, net_income.where(net_income > 0), positive_den=True)
    if earnings_history is not None and not earnings_history.empty:
        df_reported = reported_events(earnings_history, idx)
        growth_pct = pd.DataFrame(index=idx, columns=mcap.columns, dtype="float64")
        for ticker, df_ticker in df_reported.groupby("ticker", sort=False):
            if ticker not in growth_pct.columns:
                continue
            current = df_ticker["eps_actual"]
            prior = current.shift(4)
            # shortcut: report quarters proxy fiscal quarters until the archive supplies fiscal-period keys.
            quarters = df_ticker["earnings_date"].dt.to_period("Q").astype("int64")
            # A missing quarter must not turn four older observations into a false YoY comparison.
            valid_growth = current.gt(0) & prior.gt(0) & quarters.sub(quarters.shift(4)).eq(4)
            growth = ((current / prior - 1.0) * 100.0).where(valid_growth).to_numpy()
            reports = df_ticker["earnings_date"].to_numpy(dtype="datetime64[ns]")
            positions = reports.searchsorted(idx.to_numpy(), side="left") - 1
            selected = positions.clip(min=0)
            fresh = (idx.to_numpy() - reports[selected]) <= pd.Timedelta(days=_QUARTERLY_MAX_AGE_DAYS)
            growth_pct[ticker] = np.where((positions >= 0) & fresh, growth[selected], np.nan)
    else:
        growth_pct = (
            fiscal_change_to_daily(fund_hist, "netIncome", idx, kind="pct", periods=yoy_periods, years=1, max_age_days=_QUARTERLY_MAX_AGE_DAYS)
            * 100.0
        )
    div_yield_pct = ratio(daily("dividendsPaid"), mcap, positive_den=True) * 100.0
    denom = (growth_pct + div_yield_pct.fillna(0.0)).where(lambda x: x > 0)
    pegy = ratio(pe, denom)
    if pegy.empty or not pegy.notna().any().any():
        return {}
    return {"pegy": sanitize(pegy)}


def _reit_multiple_fields(daily, fund_hist: pd.DataFrame, net_income: pd.DataFrame, mcap: pd.DataFrame, ev: pd.DataFrame) -> dict:
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


def _profitability_level_fields(daily, revenue: pd.DataFrame, net_income: pd.DataFrame, fcf: pd.DataFrame) -> dict:
    """Profitability / moat: raw ratios straight from the history, plus FCF margin and
    accruals (the net-income-vs-cash gap)."""
    F: dict[str, pd.DataFrame] = {}
    for field in ["grossMargins", "operatingMargins", "profitMargins", "returnOnEquity", "debtToEquity", "revenueGrowth"]:
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


def _growth_trend_fields(view: _FiscalView) -> dict[str, pd.DataFrame]:
    """Compare fiscal growth, five-year operating margins and research intensity."""
    daily, fund_hist, idx = view.daily, view.comparison_history, view.trading_index
    revenue, rnd = daily("totalRevenue"), daily("researchAndDevelopment")
    yoy_periods, prior = view.yoy_periods, view.prior
    F: dict[str, pd.DataFrame] = {}
    for name, field, kind in (
        ("fcf_growth", "freeCashflow", "pct"),
        ("shares_growth", "sharesOutstanding", "pct"),
        ("gross_margin_chg", "grossMargins", "diff"),
    ):
        chg = fiscal_change_to_daily(
            fund_hist,
            field,
            idx,
            kind=kind,
            periods=yoy_periods,
            years=1,
            max_age_days=_QUARTERLY_MAX_AGE_DAYS,
        )
        if chg.notna().any().any():
            F[name] = chg

    op_margin = ratio(daily("operatingIncome"), revenue, positive_den=True)
    if not op_margin.empty and op_margin.notna().any().any():
        if prior is not None:
            prior_op_margin = ratio(
                prior("operatingIncome", 5),
                prior("totalRevenue", 5),
                positive_den=True,
            )
            om_5y = sanitize(op_margin - prior_op_margin)
            if om_5y.notna().any().any():
                F["operating_margin_5y_chg"] = om_5y

    rd_intensity = ratio(rnd, revenue, positive_den=True)
    if not rd_intensity.empty and rd_intensity.notna().any().any():
        F["rd_intensity"] = rd_intensity
    return F


def _reinvestment_fields(daily, fund_hist: pd.DataFrame, idx: pd.DatetimeIndex, yoy_periods: int) -> dict:
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
    da_growth = fiscal_change_to_daily(fund_hist, "depAmort", idx, kind="pct", periods=yoy_periods, years=1, max_age_days=_QUARTERLY_MAX_AGE_DAYS)
    capex_growth = fiscal_change_to_daily(fund_hist, "capex", idx, kind="pct", periods=yoy_periods, years=1, max_age_days=_QUARTERLY_MAX_AGE_DAYS)
    if da_growth.notna().any().any() and capex_growth.notna().any().any():
        F["da_minus_capex_growth"] = da_growth - capex_growth
    return F


def _quality_regime_fields(view: _FiscalView, fund_hist: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Measure profitability, investment, cash generation and fiscal financial health."""
    daily, idx, yoy_periods, prior = view.daily, view.trading_index, view.yoy_periods, view.prior
    assert prior is not None
    revenue, net_income = daily("totalRevenue"), daily("netIncome")
    fcf, long_debt = daily("freeCashflow"), daily("longTermDebt")
    F: dict[str, pd.DataFrame] = {}
    assets = capital.assets_ex_lease(daily)
    gm_lvl = daily("grossMargins")
    if not gm_lvl.empty and not revenue.empty and not assets.empty:
        cols = gm_lvl.columns.intersection(revenue.columns)
        gross_profit = gm_lvl[cols] * revenue[cols]  # grossMargins * revenue
        gp = ratio(gross_profit, assets, positive_den=True)
        if not gp.empty and gp.notna().any().any():
            F["gross_profitability"] = gp

    _assets_field = "totalAssetsExLease" if "totalAssetsExLease" in fund_hist.columns else "totalAssets"
    asset_growth = fiscal_change_to_daily(
        fund_hist, _assets_field, idx, kind="pct", periods=yoy_periods, years=1, max_age_days=_QUARTERLY_MAX_AGE_DAYS
    )
    if asset_growth.notna().any().any():
        F["asset_growth"] = asset_growth

    rev_growth_pct = (
        fiscal_change_to_daily(fund_hist, "totalRevenue", idx, kind="pct", periods=yoy_periods, years=1, max_age_days=_QUARTERLY_MAX_AGE_DAYS) * 100.0
    )
    fcf_margin_pct = ratio(fcf, revenue, positive_den=True) * 100.0
    rule40 = sanitize(rev_growth_pct + fcf_margin_pct)
    if rule40.notna().any().any():
        F["rule_of_40"] = rule40
    _oa = capital.assets_ex_lease(daily)
    _ocf = daily("operatingCashFlow")
    df_source_shares = daily("_source_shares_outstanding")
    _sh = daily("sharesOutstanding").reindex_like(df_source_shares)
    _roa = ratio(net_income, _oa, positive_den=True)
    _cr = ratio(daily("currentAssets"), daily("currentLiabilities"), positive_den=True)
    _lev = ratio(long_debt, _oa, positive_den=True)
    _gm = daily("grossMargins")
    _turn = ratio(revenue, _oa, positive_den=True)
    if not _oa.empty and not _ocf.empty and not net_income.empty:
        assets_field = "totalAssetsExLease" if "totalAssetsExLease" in fund_hist.columns else "totalAssets"
        prior_assets = prior(assets_field)
        prior_income = prior("netIncome")
        prior_roa = ratio(prior_income, prior_assets, positive_den=True)
        prior_cr = ratio(
            prior("currentAssets"),
            prior("currentLiabilities"),
            positive_den=True,
        )
        prior_lev = ratio(prior("longTermDebt"), prior_assets, positive_den=True)
        prior_turn = ratio(prior("totalRevenue"), prior_assets, positive_den=True)
        df_prior_shares = prior("sharesOutstanding").reindex_like(df_source_shares)
        parts = [
            (_roa > 0),
            (_ocf > 0),
            (_roa > prior_roa),
            (ratio(_ocf, _oa, positive_den=True) > _roa),  # accruals: cash > profit
            (_lev < prior_lev),
            (_cr > prior_cr),
            (_sh <= df_prior_shares * 1.001),  # no net dilution on common comparison units
            (_gm > prior("grossMargins")),
            (_turn > prior_turn),
        ]
        fscore = cast(pd.DataFrame, sum(p.astype("float64") for p in parts))
        df_raw_pair = df_source_shares.notna() & prior("_source_shares_outstanding").reindex_like(df_source_shares).notna()
        df_converted_pair = _sh.notna() & df_prior_shares.notna()
        df_unknown_units = (df_raw_pair & ~df_converted_pair).reindex_like(fscore).fillna(False).astype(bool)
        gate = _oa.notna() & net_income.notna() & _ocf.notna()
        fscore = fscore.where(gate & ~df_unknown_units)
        if fscore.notna().any().any():
            F["piotroski_f_score"] = fscore
    return F


def _state_flag_fields(daily, net_income: pd.DataFrame, fcf: pd.DataFrame, equity: pd.DataFrame) -> dict:
    """Absolute 0/1 regime flags emitted as raw columns.

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


def _quarter_momentum_fields(daily, fund_hist: pd.DataFrame, idx: pd.DatetimeIndex, yoy_periods: int) -> dict:
    """LATEST-QUARTER momentum (discrete single quarter, not TTM): captures the
    acceleration / inflection that TTM smooths away. Needs the discrete single-quarter
    columns emitted by the extractor."""
    F: dict[str, pd.DataFrame] = {}
    q_rev_values = fiscal_change_values(
        fund_hist,
        "revenue_q",
        kind="pct",
        periods=yoy_periods,
        years=1,
    )
    q_rev_yoy = fiscal_values_to_daily(
        fund_hist,
        q_rev_values,
        idx,
        max_age_days=_QUARTERLY_MAX_AGE_DAYS,
    )
    if q_rev_yoy.notna().any().any():
        # acceleration = this quarter's YoY minus the previous quarter's YoY
        previous_growth = _previous_fiscal_value(fund_hist, q_rev_values)
        F["rev_growth_accel"] = fiscal_values_to_daily(
            fund_hist,
            q_rev_values - previous_growth,
            idx,
            max_age_days=_QUARTERLY_MAX_AGE_DAYS,
        )

    q_ni_values = fiscal_change_values(
        fund_hist,
        "netIncome_q",
        kind="pct",
        periods=yoy_periods,
        years=1,
    )
    q_ni_yoy = fiscal_values_to_daily(
        fund_hist,
        q_ni_values,
        idx,
        max_age_days=_QUARTERLY_MAX_AGE_DAYS,
    )
    if q_ni_yoy.notna().any().any():
        F["q_earnings_growth"] = q_ni_yoy

    # latest-quarter margin vs TTM margin = margin inflection
    q_margin = ratio(daily("netIncome_q"), daily("revenue_q"), positive_den=True)
    profit_margins = daily("profitMargins")
    if not q_margin.empty and not profit_margins.empty:
        cols = q_margin.columns.intersection(profit_margins.columns)
        F["q_margin_vs_ttm"] = q_margin[cols] - profit_margins[cols]
    return F


def _yearly_ttm_momentum_fields(fund_hist: pd.DataFrame, idx: pd.DatetimeIndex, yoy_periods: int) -> dict:
    """YEARLY-TTM momentum (current TTM vs TTM one year ago) and three-year revenue CAGR.

    Complements the single-quarter features: captures a multi-quarter trend rather than
    the most-recent quarter jolt, which is less noisy and works at longer horizons. Uses
    the fiscal history of level columns so seasonality is removed by construction (same
    fiscal quarter each year, matched by fiscal dates)."""
    F: dict[str, pd.DataFrame] = {}
    y_rev_growth = fiscal_change_to_daily(
        fund_hist, "totalRevenue", idx, kind="pct", periods=yoy_periods, years=1, max_age_days=_QUARTERLY_MAX_AGE_DAYS
    )
    if y_rev_growth.notna().any().any():
        F["y_rev_growth"] = y_rev_growth
        y_rev_values = fiscal_change_values(
            fund_hist,
            "totalRevenue",
            kind="pct",
            periods=yoy_periods,
            years=1,
        )
        F["y_rev_growth_accel"] = fiscal_values_to_daily(
            fund_hist,
            y_rev_values - _previous_fiscal_value(fund_hist, y_rev_values),
            idx,
            max_age_days=_QUARTERLY_MAX_AGE_DAYS,
        )

    signed_growth = fiscal_change_values(fund_hist, "netIncome", kind="signed_pct", periods=yoy_periods, years=1)
    F["y_earnings_growth_signed"] = fiscal_values_to_daily(fund_hist, signed_growth, idx, max_age_days=_QUARTERLY_MAX_AGE_DAYS, preserve_nulls=True)
    if "totalRevenue" in fund_hist.columns:
        revenue = pd.to_numeric(fund_hist["totalRevenue"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        prior_revenue = fiscal_prior_values(fund_hist, revenue, years=3).replace([np.inf, -np.inf], np.nan)
        cagr = (revenue.where(revenue > 0) / prior_revenue.where(prior_revenue > 0)).pow(1.0 / 3.0) - 1.0
        F["revenue_cagr_3y"] = fiscal_values_to_daily(
            fund_hist, cagr.replace([np.inf, -np.inf], np.nan), idx, max_age_days=_QUARTERLY_MAX_AGE_DAYS, preserve_nulls=True
        )

    # YoY change in TTM profit margin (margin expansion / contraction trend)
    y_margin_chg = fiscal_change_to_daily(
        fund_hist, "profitMargins", idx, kind="diff", periods=yoy_periods, years=1, max_age_days=_QUARTERLY_MAX_AGE_DAYS
    )
    if y_margin_chg.notna().any().any():
        F["y_margin_vs_ttm"] = y_margin_chg
    return F


def _cash_runway_fields(fund_hist: pd.DataFrame, idx: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    """Months of current liquidity at the trailing FCF burn rate, outside Financials."""
    if not {"cash", "freeCashflow", "sector"}.issubset(fund_hist.columns):
        return {}
    cash = pd.to_numeric(fund_hist["cash"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    fcf = pd.to_numeric(fund_hist["freeCashflow"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    nonfinancial = fund_hist["sector"].notna() & fund_hist["sector"].ne("") & ~row_gate(fund_hist, "financials")
    runway = (12.0 * cash.where(cash >= 0) / -fcf.where(fcf < 0)).where(nonfinancial)
    return {
        "cash_runway_months": fiscal_values_to_daily(
            fund_hist, runway.replace([np.inf, -np.inf], np.nan), idx, max_age_days=_QUARTERLY_MAX_AGE_DAYS, preserve_nulls=True
        )
    }


def _distress_fields(view: _FiscalView) -> dict[str, pd.DataFrame]:
    """Measure debt servicing and liquidity while separating cash ratios from netting."""
    daily, operating_cash = view.daily, view.operating_cash
    ebitda, cash, fcf = daily("ebitda"), daily("cash"), daily("freeCashflow")
    long_debt, short_debt = daily("longTermDebt"), daily("shortTermDebt")
    F: dict[str, pd.DataFrame] = {}
    total_debt = _combine_debt(daily, long_debt, short_debt, daily("totalLiabilities"))
    if not total_debt.empty and not ebitda.empty:
        net_cash = capital.drop_operating_cash(cash, operating_cash)
        cols = total_debt.columns.intersection(net_cash.columns) if not net_cash.empty else total_debt.columns
        net_debt = total_debt[cols].sub(net_cash[cols], fill_value=0.0) if not net_cash.empty else total_debt
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
    current_ratio = ratio(daily("currentAssets"), daily("currentLiabilities"), positive_den=True)
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


def _sga_efficiency_fields(daily, fund_hist: pd.DataFrame, idx: pd.DatetimeIndex, revenue: pd.DataFrame, yoy_periods: int) -> dict:
    """MARKETING & SALES efficiency (operating leverage).

    `operating_leverage` = sales growing FASTER than selling cost (scalable); negative
    means growth is being "bought" with rising SG&A (margin risk ahead)."""
    F: dict[str, pd.DataFrame] = {}
    sga_intensity = ratio(daily("sellingGeneralAdmin"), revenue, positive_den=True)
    if not sga_intensity.empty and sga_intensity.notna().any().any():
        F["sga_intensity"] = sga_intensity
    sga_growth = fiscal_change_to_daily(
        fund_hist, "sellingGeneralAdmin", idx, kind="pct", periods=yoy_periods, years=1, max_age_days=_QUARTERLY_MAX_AGE_DAYS
    )
    rev_growth = fiscal_change_to_daily(
        fund_hist, "totalRevenue", idx, kind="pct", periods=yoy_periods, years=1, max_age_days=_QUARTERLY_MAX_AGE_DAYS
    )
    if sga_growth.notna().any().any():
        F["sga_growth"] = sga_growth
        if rev_growth.notna().any().any():
            cols = rev_growth.columns.intersection(sga_growth.columns)
            F["operating_leverage"] = rev_growth[cols] - sga_growth[cols]
    return F


def _ma_footprint_fields(daily, fund_hist: pd.DataFrame, idx: pd.DatetimeIndex, revenue: pd.DataFrame, yoy_periods: int) -> dict:
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
    intangibles_growth = fiscal_change_to_daily(
        fund_hist, "intangibles", idx, kind="pct", periods=yoy_periods, years=1, max_age_days=_QUARTERLY_MAX_AGE_DAYS
    )
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


def _valuation_engine_fields(view: _FiscalView) -> dict[str, pd.DataFrame]:
    """Measure fiscal operating leverage, margin divergence, dilution and debt coverage."""
    daily, fund_hist, idx = view.daily, view.comparison_history, view.trading_index
    revenue, ebitda = daily("totalRevenue"), daily("ebitda")
    yoy_periods, prior = view.yoy_periods, view.prior
    F: dict[str, pd.DataFrame] = {}
    oi_growth = fiscal_change_to_daily(
        fund_hist, "operatingIncome", idx, kind="pct", periods=yoy_periods, years=1, max_age_days=_QUARTERLY_MAX_AGE_DAYS
    )
    rev_growth_f = fiscal_change_to_daily(
        fund_hist, "totalRevenue", idx, kind="pct", periods=yoy_periods, years=1, max_age_days=_QUARTERLY_MAX_AGE_DAYS
    )
    ol_el = ratio(oi_growth, rev_growth_f.where(rev_growth_f.abs() >= 0.02))
    if not ol_el.empty and ol_el.notna().any().any():
        F["operating_leverage_elasticity"] = ol_el

    gm = ratio(daily("grossProfit"), revenue, positive_den=True)
    if gm.empty:
        gm = daily("grossMargins")
    em = ratio(ebitda, revenue, positive_den=True)
    if not gm.empty and not em.empty and prior is not None:
        prior_gm = ratio(
            prior("grossProfit"),
            prior("totalRevenue"),
            positive_den=True,
        )
        if prior_gm.empty:
            prior_gm = prior("grossMargins")
        prior_em = ratio(
            prior("ebitda"),
            prior("totalRevenue"),
            positive_den=True,
        )
        med = (gm - prior_gm) - (em - prior_em)
        if med.notna().any().any():
            F["margin_expansion_delta"] = sanitize(med)

    cur_a2, cur_l2 = daily("currentAssets"), daily("currentLiabilities")
    if not cur_a2.empty and not cur_l2.empty and not revenue.empty and prior is not None:
        nwc = cur_a2.sub(cur_l2, fill_value=np.nan)
        prev_nwc = prior("currentAssets").sub(prior("currentLiabilities"), fill_value=np.nan)
        nwc_g = ratio(nwc - prev_nwc, prev_nwc, positive_den=True)
        prev_rev = prior("totalRevenue")
        rev_g = ratio(revenue - prev_rev, prev_rev, positive_den=True)
        nwc_el = ratio(nwc_g, rev_g.where(rev_g.abs() >= 0.02))
        if not nwc_el.empty and nwc_el.notna().any().any():
            F["nwc_elasticity"] = nwc_el

    dil_growth = fiscal_change_to_daily(
        fund_hist, "dilutedShares", idx, kind="pct", periods=yoy_periods, years=1, max_age_days=_QUARTERLY_MAX_AGE_DAYS
    )
    if dil_growth.notna().any().any():
        F["diluted_shares_growth"] = dil_growth

    eic = ratio(daily("operatingIncome"), daily("interestExpense").abs(), positive_den=True)
    if not eic.empty and eic.notna().any().any():
        F["ebit_interest_coverage"] = eic
    return F


def _intrinsic_fields(fund_hist: pd.DataFrame, prices: PriceFrames | None, idx: pd.DatetimeIndex, intrinsic_cfg: dict | None) -> dict:
    """Emit issuer DCF yield only when saved capitalization supports it."""
    if prices is None:
        return {}
    iy = intrinsic_value_daily(fund_hist, prices, idx, settings=intrinsic_cfg).get("yield")
    if iy is None or iy.empty or not iy.notna().any().any():
        return {}
    return {"intrinsic_yield": iy}


# --------------------------------------------------------------------------- #
# Characteristics                                                              #
# --------------------------------------------------------------------------- #
def _fiscal_accessors(
    fund_hist: pd.DataFrame,
    idx: pd.DatetimeIndex,
    yoy_periods: int,
) -> tuple[Callable[[str], pd.DataFrame], Callable[..., pd.DataFrame]]:
    """Memoize daily observations and then-public fiscal predecessors for one build."""
    _daily_cache: dict[str, pd.DataFrame] = {}
    _prior_cache: dict[tuple[str, int, int], pd.DataFrame] = {}
    _match_cache: dict[tuple[int, int], pd.Series] = {}

    def daily(field: str) -> pd.DataFrame:
        # memoized: several fields are reused across blocks (and the business-quality
        # helpers), so cache the pivot+ffill instead of recomputing it.
        if field not in _daily_cache:
            _daily_cache[field] = fundamentals_to_daily(
                fund_hist,
                field,
                idx,
                max_age_days=_QUARTERLY_MAX_AGE_DAYS,
            )
        return _daily_cache[field]

    def prior(field: str, years: int = 1, tolerance_days: int = 45) -> pd.DataFrame:
        if field not in fund_hist.columns:
            return pd.DataFrame(index=idx)
        key = (field, years, tolerance_days)
        if key not in _prior_cache:
            has_fiscal_end = "fiscal_end" in fund_hist.columns and fund_hist["fiscal_end"].notna().any()
            if not has_fiscal_end:
                work = pd.DataFrame(
                    {
                        "ticker": fund_hist["ticker"].astype(str),
                        "as_of": pd.to_datetime(fund_hist["as_of"], errors="coerce"),
                        "value": pd.to_numeric(fund_hist[field], errors="coerce"),
                        "position": np.arange(len(fund_hist)),
                    }
                ).sort_values(["ticker", "as_of", "position"])
                work["prior"] = work.groupby("ticker", sort=False)["value"].shift(periods=int(years * yoy_periods))
                values = np.full(len(fund_hist), np.nan, dtype="float64")
                values[work["position"].to_numpy(dtype="int64")] = work["prior"].to_numpy(dtype="float64", na_value=np.nan)
                _prior_cache[key] = fiscal_values_to_daily(
                    fund_hist,
                    pd.Series(values, index=fund_hist.index),
                    idx,
                    max_age_days=_QUARTERLY_MAX_AGE_DAYS,
                )
            else:
                match_key = (years, tolerance_days)
                if match_key not in _match_cache:
                    _match_cache[match_key] = fiscal_prior_positions(
                        fund_hist,
                        years=years,
                        tolerance_days=tolerance_days,
                    )
                _prior_cache[key] = fiscal_prior_to_daily(
                    fund_hist,
                    field,
                    idx,
                    years=years,
                    tolerance_days=tolerance_days,
                    max_age_days=_QUARTERLY_MAX_AGE_DAYS,
                    positions=_match_cache[match_key],
                )
        return _prior_cache[key]

    return daily, prior


def _pension_fields(view: _FiscalView, supplement: FundamentalSupplement) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """Scope supplemental facts before projecting pension health and obligations."""
    universe = set(view.comparison_history["ticker"].astype(str)) if "ticker" in view.comparison_history.columns else set()
    df_notes = _scope_to_universe(supplement.notes_num, universe)
    df_pbo, _fn_deficit, df_pension_ret = _pension_pool(df_notes, view.trading_index)
    fields = _pension_health_fields(view.daily, df_pbo, df_notes, view.trading_index, df_pension_ret)
    return fields, df_pbo, df_pension_ret


def _capital_fields(
    view: _FiscalView,
    df_history: pd.DataFrame,
    prices: PriceFrames,
    earnings: pd.DataFrame | None,
    df_pbo: pd.DataFrame,
    df_pension_ret: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Value supported public financials with saved issuer capitalization."""
    daily, idx = view.daily, view.trading_index
    df_mcap = pd.DataFrame(index=idx) if prices.mcap_pit is None else prices.mcap_pit.reindex(idx)
    df_eligible = daily("_source_shares_outstanding")
    df_mcap = df_mcap.where(df_eligible.reindex_like(df_mcap).gt(0) & np.isfinite(df_mcap) & df_mcap.gt(0))
    if df_mcap.empty:
        return {}

    # Preserve the saved-capital field-presence and public source-age gates.
    df_revenue, df_income = daily("totalRevenue"), daily("netIncome")
    df_fcf, df_equity = daily("freeCashflow"), daily("stockholdersEquity")
    fields = {"ocf_yield": ratio(daily("operatingCashFlow"), df_mcap, positive_den=True)}
    df_equity_value = _diluted_capital(df_history, idx, prices, df_mcap)
    df_ev = _enterprise_value_frame(view, df_equity_value, df_pension_ret)
    fields.update(_valuation_yield_fields(df_mcap, daily("netIncomeCommon"), df_income, df_revenue, df_equity, df_fcf))
    fields.update(_pension_scale_fields(df_pension_ret, df_pbo, df_mcap))
    fields.update(_ev_yield_fields(daily("ebitda"), df_fcf, df_ev))
    fields.update(_altman_z_fields(daily, df_mcap, df_revenue))
    fields.update(_pegy_fields(view, df_history, df_mcap, df_income, earnings))
    fields.update(_reit_multiple_fields(daily, df_history, df_income, df_mcap, df_ev))
    return fields


def _operating_fields(
    view: _FiscalView,
    df_history: pd.DataFrame,
    prices: PriceFrames,
    intrinsic_cfg: dict | None,
    share_basis: ShareBasis | None,
) -> dict[str, pd.DataFrame]:
    """Compose operating, quality and matched per-share financial characteristics."""
    daily, idx, yoy_periods = view.daily, view.trading_index, view.yoy_periods
    df_revenue, df_income, df_fcf = daily("totalRevenue"), daily("netIncome"), daily("freeCashflow")

    # Fiscal and operating features reuse the same daily/prior accessor caches.
    fields = _profitability_level_fields(daily, df_revenue, df_income, df_fcf)
    fields.update(_growth_trend_fields(view))
    fields.update(_reinvestment_fields(daily, df_history, idx, yoy_periods))
    fields.update(_quality_regime_fields(view, df_history))
    fields.update(_state_flag_fields(daily, df_income, df_fcf, daily("stockholdersEquity")))
    fields.update(_quarter_momentum_fields(daily, df_history, idx, yoy_periods))
    fields.update(_yearly_ttm_momentum_fields(df_history, idx, yoy_periods))
    fields.update(_cash_runway_fields(df_history, idx))
    fields.update(_distress_fields(view))
    fields.update(_sga_efficiency_fields(daily, df_history, idx, df_revenue, yoy_periods))
    fields.update(_ma_footprint_fields(daily, df_history, idx, df_revenue, yoy_periods))
    fields.update(_sbc_fields(daily, df_revenue, daily("stockBasedComp")))
    fields.update(_valuation_engine_fields(view))
    fields.update(_intrinsic_fields(df_history, prices, idx, intrinsic_cfg))

    # Preserve the source-unit gates on earnings and per-share comparisons.
    fields.update(_da_realism_fields(daily))
    fields.update(_forensic_fields(daily, idx, view.prior))
    fields.update(_digestion_fields(daily, df_history, idx, yoy_periods, view.operating_cash))
    df_eps_yield = _financial_eps_yield(df_history, prices, idx, share_basis)
    fields.update(_per_share_and_profit_slice_fields(daily, view.comparison_history, idx, yoy_periods, df_eps_yield))
    return fields


def _derived_fields(
    history: FundamentalHistory,
    prices: PriceFrames,
    supplement: FundamentalSupplement | None = None,
    settings: FundamentalFeatureSettings | None = None,
    yoy_periods: int = 1,
) -> dict[str, pd.DataFrame]:
    """Compose saved-capital, financial and comparison-unit characteristics."""
    df_history = history.normalized_share_history if history.normalized_share_history is not None else history.raw_history
    if df_history is None:
        raise ValueError("Need to build fundamentals_history table first")
    df_history = null_invalid_gross_profit_sentinels(df_history)
    idx = prices.trading_index
    supplement = supplement if supplement is not None else FundamentalSupplement()
    settings = settings if settings is not None else FundamentalFeatureSettings()

    # Keep raw source age distinct from normalized comparison quantities.
    df_comparison = comparison_share_history(
        df_history,
        history.share_basis,
        pd.Timestamp(idx.max()),
        count_fields=("sharesOutstanding", "dilutedShares"),
        per_share_fields=("dividendsPerShare",),
    )
    df_comparison["_source_shares_outstanding"] = df_history.get(
        "_source_shares_outstanding", df_history.get("sharesOutstanding", pd.Series(np.nan, index=df_history.index))
    )
    daily, prior = _fiscal_accessors(df_comparison, idx, yoy_periods)
    operating_cash = frozenset(family_tickers(df_history, "bank") | family_tickers(df_history, "insurance"))
    view = _FiscalView(df_comparison, idx, yoy_periods, daily, prior, operating_cash)

    # Scope supplemental facts once, then compose the unchanged financial blocks.
    fields, df_pbo, df_pension_ret = _pension_fields(view, supplement)
    fields.update(_capital_fields(view, df_history, prices, supplement.earnings_history, df_pbo, df_pension_ret))
    fields.update(_operating_fields(view, df_history, prices, settings.intrinsic_cfg, history.share_basis))
    return fields


def build_fundamental_feature_panel(
    history: FundamentalHistory,
    prices: PriceFrames,
    settings: FundamentalFeatureSettings | None = None,
    supplement: FundamentalSupplement | None = None,
    history_fields: dict[str, pd.DataFrame] | None = None,
    output_since: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Build financial feature views from explicit histories and saved capital."""
    df_history = history.normalized_share_history if history.normalized_share_history is not None else history.raw_history
    if df_history is None or df_history.empty:
        raise ValueError("Need to build fundamentals_history table first")
    settings = settings if settings is not None else FundamentalFeatureSettings()

    # Preserve financial cadence, mathematical ordering and persisted precision.
    fields = _derived_fields(history, prices, supplement, settings, infer_yoy_periods(df_history))
    fields.update(
        reported_history_fields(
            df_history,
            prices.trading_index,
            earnings_lags=settings.earnings_lags,
            earnings_scale_quarters=settings.earnings_scale_quarters,
            persistence_quarters=settings.persistence_quarters,
        )
    )
    fields = {key: (value.astype("float32") if isinstance(value, pd.DataFrame) and not value.empty else value) for key, value in fields.items()}

    # Apply the existing raw/self-history view semantics.
    semantics = {name: mode for name, mode in FUNDAMENTAL_TRANSFORM_SEMANTICS.items() if name in fields}
    return build_fundamental_views(
        fields,
        prices.peers,
        semantics=semantics,
        history_window=settings.history_window,
        history_min_periods=settings.history_min_periods,
        history_fields=history_fields,
        output_since=output_since,
    )
