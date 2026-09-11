"""
pay_features.py  (src/data_aggregate/utils/governance/pay_features.py)
----------------------------------------------------------------------
The three EXECUTIVE-COMPENSATION families: CEO pay level and TURNOVER-GUARDED growth, the
exact academic CEO Pay Slice, and deterministic pay-vs-performance misalignment.

Sources, and the grain each arrives on:

    def14a_llm              one row per annual proxy   -> the CEO numerator + their identity
    def14a_executive_comp   one row per NEO per FISCAL  -> the top-five denominator
                            YEAR per filing (~3 years
                            per NEO, Item 402(c))
    fundamentals_history    quarterly filings           -> the revenue leg
    close_total             daily                       -> the shareholder-return leg

⚠ WHY THE TURNOVER GUARD IS THE POINT OF FAMILY 5. A partial-year incoming CEO's package is
not organic pay growth. The guarded field ships under a NEW name (D13) rather than replacing
`ceo_pay_growth`, which was a live feature at the time.

⚠ THE LEGACY LEG IS NO LONGER UNGUARDED. Phase 1 (2026-09-08) gave `panel._ceo_pay_growth` the
same guard, so the two families now share ONE definition of "the CEO changed" --
`names.ceo_identity_changed` -- and cannot disagree about a transition. They still differ where
it matters: this one is a LOG growth with an adjacency guard and a both-sides-positive guard,
that one a PCT change with neither, bounded instead by a cross-sectional trim. Measured after
phase 1 they run at Pearson r = 0.516 and Spearman rho = 0.958 over 3.78M overlapping cells --
close to rank-identical, which is a question for the deduplication phase and not a defect.

⚠ IDENTITY COMES FROM `ceo_identity`, NEVER A RAW-STRING COMPARISON. Measured on the live
archive, comparing `ceo_name_proxy` as text manufactures **356 spurious turnovers out of
1,625 (21.9%)** purely from a filer's own spelling drift -- `Timothy D. Cook` -> `Timothy Cook`
-> `Tim Cook` is one CEO across three filings, `Juan R. Luciano` -> `J. R. LUCIANO` one more.
Each of those would null a legitimate growth observation. The shared key collapses them.

⚠ EVERY GROWTH LEG IS A LOG GROWTH, AND SO IS EVERY LEG SUBTRACTED FROM IT. The pay leg is
`log(comp_t / comp_{t-1})` because GPT S2.3 asks for log growth and it is symmetric where a
ratio is not; the revenue and return legs therefore go through `log1p` before the subtraction.
Mixing the two bases is a real error, not a rounding one: a doubled package is `+1.00` as a
percentage change and `+0.69` as a log growth, so `pay_return_gap` on mixed bases would read
+31pp of spurious misalignment on exactly the extreme observations the family exists to find.

TIME BASIS, and the one place this panel is NOT a step function. `ceo_comp_growth_1y` is
annual and forward-filled; `shareholder_return_1y` moves every day. Their difference therefore
drifts *within* a proxy year, which is the correct reading of the alignment question -- "pay set
last spring, against the stock's trailing year as of today" -- but it does mean family 7 is not
piecewise-constant like the rest of the governance panel.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_aggregate.utils.common.panel import peer_relative
from src.data_aggregate.utils.common.pit import (
    fiscal_change_to_daily,
    fundamentals_to_daily,
    infer_yoy_periods,
)
from src.data_aggregate.utils.common.xs import winsorize_xs
from src.data_aggregate.utils.governance.names import (
    ceo_identity_changed,
    ceo_identity_series,
    is_multi_name,
)
from src.data_aggregate.utils.governance.staleness import (
    LEVEL_MAX_AGE_DAYS, expire_event_fields, expire_level_fields,
)
from src.utils.names import person_key

#: Binary indicators. A flag ships RAW and is never peer-z-scored: the peer z of a Bernoulli
#: draw over ~7 peers divides by the standard deviation of that draw, so it answers "how many
#: of my peers ALSO did this" rather than "did this happen here" (the measured 32%-coverage
#: failure of the retired `f_founder_ceo_vs_peers`).
RAW_FLAG_FIELDS: frozenset[str] = frozenset({
    "ceo_turnover_flag", "pay_up_revenue_down", "pay_up_return_down",
})

#: ⚠ EMPTY, AND THAT IS A MEASURED RESULT THAT REFUTED THE PLAN. Phase 4 was written
#: expecting the two LEVELS to be peer-panelled -- "a $30M package is unremarkable for a
#: mega-cap bank and extreme for a regional utility" -- so the peer basket was supposed to say
#: which case a firm is in. Measured 2026-09-08 on the live archive, it does not:
#:
#:     field                  between-sector var    peer-basket R^2
#:     log_ceo_total_comp                   2.4%               2.4%
#:     ceo_pay_slice                        4.7%               4.1%
#:     -- against the floor the same rule kept in phase 3 --
#:     insider_ownership_pct (kept)         7.7%               n/a
#:     ceo_pay_ratio (kept)                12.2%               7.7%
#:     board_size (kept)                   12.4%              10.0%
#:     profitMargins (reference)             9.0%              13.6%
#:     totalRevenue (reference)              8.1%              30.0%
#:
#: TWO INDEPENDENT MEASURES AGREE, which is why this is a decision and not a coin toss. The
#: second one is the more direct: `peer-basket R^2` is the share of the day's cross-section
#: that a ticker's OWN basket mean explains, so it tests the embedding basket itself rather
#: than using sector as a proxy for it. A CEO package is idiosyncratic -- set by one board's
#: compensation committee against a peer group of its own choosing -- and dividing it by the
#: dispersion of seven business-text-similar companies standardizes it against noise.
#:
#: So EVERY field in this module ships RAW. The plan's PAY02 / PAY04 "peer gap" features
#: therefore do not exist as separate columns, which is the honest outcome: they would have
#: been `f_log_ceo_total_comp_vs_peers` and nothing measurable stood behind them.
PEER_RELATIVE_FIELDS: frozenset[str] = frozenset()

#: Fields that describe an EVENT and expire 548 days after the filing that produced them
#: (D21). The two pay LEVELS are excluded: a CEO's package and their share of the top five are
#: standing facts between proxies, where "pay grew 40% last year" stops being true about today.
EVENT_FIELDS: frozenset[str] = frozenset({
    "ceo_comp_growth_1y", "ceo_turnover_flag", "ceo_pay_slice_delta_1y",
    "pay_revenue_gap", "pay_return_gap",
    "pay_revenue_peer_misalignment", "pay_return_peer_misalignment",
    "pay_up_revenue_down", "pay_up_return_down",
    "pay_up_revenue_down_severity", "pay_up_return_down_severity",
})

#: Fields whose value depends on the WHOLE CROSS-SECTION on their date, not only on the
#: filer's own archive. Both are built as `winsorize_xs(peer_relative(pay)) -
#: winsorize_xs(peer_relative(prf))`, so a ticker's value moves when the peer basket changes
#: OR when the 1%/99% trim bound moves -- neither of which requires the ticker's own filings to
#: change at all.
#:
#: Declared because a per-ticker attribution check cannot otherwise tell them apart from a raw
#: characteristic. `reports/validate/governance/_scripts/21_fetch_attribution.py` gates on
#: "a raw feature moves only for a ticker whose own filings changed", and these two are exempt
#: from that gate BY CONSTRUCTION. Reading the exemption off the name would have been wrong:
#: they carry no `_vs_peers` suffix, and on the 2026-09-10 rebuild they moved for 424 and 417
#: untouched tickers respectively -- correctly.
CROSS_SECTIONAL_FIELDS: frozenset[str] = frozenset(
    {f"pay_{lab}_peer_misalignment" for lab in ("revenue", "return")}
)

#: Every field this module can emit. `_alignment_family` names its members from a LABEL
#: (`revenue` / `return`), so the two sets above can only be checked against the builder by
#: enumerating what it produces -- which is what caught `pay_up_stock_down`, a name the plan
#: used, that no code path ever produces, sitting in both sets while the two fields actually
#: built sat in neither.
ALL_FIELDS: frozenset[str] = frozenset(
    {"log_ceo_total_comp", "ceo_comp_growth_1y", "ceo_turnover_flag",
     "ceo_pay_slice", "ceo_pay_slice_delta_1y"}
    | {f"pay_{lab}_gap" for lab in ("revenue", "return")}
    | {f"pay_{lab}_peer_misalignment" for lab in ("revenue", "return")}
    | {f"pay_up_{lab}_down" for lab in ("revenue", "return")}
    | {f"pay_up_{lab}_down_severity" for lab in ("revenue", "return")}
)

#: The two pay LEVELS, on `LEVEL_MAX_AGE_DAYS` (1,095 days) rather than on no horizon at all.
#:
#: ⚠ The `EVENT_FIELDS` note above says these two are "standing facts between proxies", and
#: that is right -- it just is not an argument for keeping them forever. A package set in 2014
#: is no more evidence about today than a 2019 say-on-pay vote is; the two claims differ in
#: HALF-LIFE, which is what two horizons express and what excluding them from both did not.
#: Defined as the complement of `EVENT_FIELDS` so the two are exhaustive over `ALL_FIELDS`.
#:
#: These two carry the largest measured exposure of the twelve fields this change reaches:
#: 3,956 cells past 1,095 days (**0.15%**) and a maximum forward-fill age of 4,428 days --
#: 12.1 years of one proxy's number reported as current.
LEVEL_FIELDS: frozenset[str] = ALL_FIELDS - EVENT_FIELDS

#: A YoY pair must be two filings roughly ONE year apart. Proxy filing dates drift by weeks
#: between years, so the window is generous; what it rejects is the pair that straddles a
#: MISSING proxy, which `shift(1)` would otherwise label a one-year change across two years.
#: Expressed in days rather than calendar years because a filer moving from a January to a
#: December meeting would show a year difference of 0 on two genuinely adjacent proxies.
_ANNUAL_GAP_DAYS: tuple[float, float] = (250.0, 550.0)

#: The trailing window of the shareholder-return leg: 252 trading days = one year, on
#: `close_total` because it is the dividend-adjusted basis and the only correct one for a
#: RETURN (`close_split` would understate every dividend payer's performance and so overstate
#: its CEO's misalignment).
_RETURN_WINDOW = 252

#: A denominator built from fewer than three NEOs is not an executive pool -- it is the CEO and
#: perhaps a CFO, and the resulting "slice" is near 1.0 by construction. Measured cost after
#: `impute_exec_comp`: only 84 of 11,316 filings fall below three, so the gate is nearly free.
_MIN_TOP5_NEOS = 3

#: The academic CPS is the CEO's share of the top FIVE executives' total pay.
_TOP_N_NEOS = 5


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    """`df[col]` as float, or an all-NaN column when it is absent."""
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def prior_annual_leg(hist: pd.DataFrame, field: str, key: str = "ticker",
                     date: str = "as_of") -> pd.Series:
    """The value of `field` at each row's PREVIOUS filing -- but ONLY when that filing is about
    one year earlier. NaN otherwise.

    ⚠ THIS IS THE STRICT-LAG RULE, AND IT IS THE MOST LIKELY SILENT FAILURE IN THE WHOLE
    PHASE. When the prior year is MISSING the answer is NaN: never carried forward from Y-2,
    never zero-filled, never back-filled from Y. A bare `groupby.shift(1)` looks identical and
    is wrong -- it hands back Y-2's value with no marker, so a two-year change ships labelled as
    a one-year change and a two-year-old vote leaks into a pay feature. The hole has to stay
    visible, which is why the gap is checked and not assumed.

    Measured cost on the live proxy archive: of 8,914 rows whose ticker has ANY prior filing,
    **8,724 have one ~1 year earlier**, so the gap check rejects **190** -- of which **102**
    would otherwise have produced a growth number (the other 88 already had a missing or
    non-positive comp on one side). Those 102 are genuinely missing proxy years mid-history.

    `hist` must be sorted by `[key, date]` by the caller; the returned Series carries `hist`'s
    own index, so it aligns for assignment.
    """
    d = pd.to_datetime(hist[date], errors="coerce")
    g = hist.groupby(key, sort=False)
    prev = g[field].shift(1)
    gap = d.groupby(hist[key]).diff().dt.days
    return prev.where(gap.between(*_ANNUAL_GAP_DAYS))


def _comp_history(def14a: pd.DataFrame, tally: dict[str, int]) -> pd.DataFrame | None:
    """`[ticker, as_of, log_ceo_total_comp, ceo_comp_growth_1y, ceo_turnover_flag]` per filing.

    Per ticker, chronologically by `as_of`:

        growth_t   = log(comp_t / comp_{t-1})  iff both comps are > 0, the CEO identity is
                     KNOWN on both sides and unchanged, and the two filings are ~1y apart
        turnover_t = 1.0 when the identity changed between two KNOWN names, else 0.0

    ⚠ THE UNKNOWN CASE IS NaN, NEVER "no change". `impute_def14a` fills a `ceo_name_proxy`
    gap only when the key agrees on both sides (D27), leaving 15 disagreements NaN on purpose.
    Reading that NaN as continuity would compute pay growth straight across the AMD 2009 and
    CNC 2022 transitions this guard exists to catch.
    """
    if def14a is None or def14a.empty or "as_of" not in def14a.columns:
        return None
    if "ceo_total_comp" not in def14a.columns:
        return None

    h = pd.DataFrame({
        "ticker": def14a["ticker"],
        "as_of": pd.to_datetime(def14a["as_of"], errors="coerce"),
        "comp": _num(def14a, "ceo_total_comp"),
    })
    raw_names = (def14a["ceo_name_proxy"] if "ceo_name_proxy" in def14a.columns
                 else pd.Series(None, index=def14a.index, dtype="object"))
    h["name"] = raw_names.astype(object)
    h = h.dropna(subset=["ticker", "as_of"]).sort_values(["ticker", "as_of"])
    if h.empty:
        return None

    g = h.groupby("ticker", sort=False)
    # The prior-year comp comes from the STRICT-LAG primitive, so a missing proxy year is a
    # NaN here rather than a silently borrowed Y-2 package. `prev_ident` uses the plain shift
    # on purpose: identity continuity is a question about the previous FILING whatever its
    # date, and pairing it with the strict comp leg is what keeps a gap year out of the growth.
    prev_comp = prior_annual_leg(h, "comp")
    prev_any_comp = g["comp"].shift(1)

    # ONE definition of "the CEO changed", shared with `panel._ceo_pay_growth` (phase 1) so the
    # guarded field and the unguarded legacy leg can never disagree about a transition.
    changed = ceo_identity_changed(h["name"], h["ticker"])
    positive = (h["comp"] > 0) & (prev_comp > 0)
    known = changed.notna()
    unchanged = changed == 0.0
    adjacent = prev_comp.notna()

    out = h[["ticker", "as_of"]].copy()
    out["log_ceo_total_comp"] = np.log1p(h["comp"].where(h["comp"] > 0))
    out["ceo_comp_growth_1y"] = np.log(
        (h["comp"] / prev_comp).where(positive & unchanged))
    # The flag does NOT take the adjacency guard: a CEO change between two filings three years
    # apart is still a CEO change, only imprecisely dated. Requiring both names to be known is
    # what keeps it from reading a gap as continuity.
    out["ceo_turnover_flag"] = changed

    tally["comp observations"] = int(h["comp"].notna().sum())
    tally["with a prior-filing comp"] = int(prev_any_comp.notna().sum())
    tally["with a prior-filing comp ~1y earlier"] = int(prev_comp.notna().sum())
    tally["growth pairs kept"] = int(out["ceo_comp_growth_1y"].notna().sum())
    tally["growth nulled: CEO turnover"] = int((positive & known & ~unchanged & adjacent).sum())
    tally["growth nulled: CEO identity unknown"] = int((positive & ~known).sum())
    tally["growth nulled: filings not ~1y apart"] = int(
        ((h["comp"] > 0) & (prev_any_comp > 0) & ~adjacent).sum())
    tally["CEO turnovers detected"] = int((out["ceo_turnover_flag"] > 0).sum())
    tally["co-CEO cells (first person taken, D28)"] = int(
        raw_names.astype(object).map(is_multi_name).sum())
    return out.reset_index(drop=True)


def _top5_history(exec_comp: pd.DataFrame, tally: dict[str, int]) -> pd.DataFrame | None:
    """`[ticker, as_of, top5_neo_total_comp, n_neos_top5]` -- one row per FILING.

    Per `(ticker, accession_number)`: keep the rows of the filing's own LATEST `fiscal_year`,
    collapse duplicate people keeping their largest `total`, then sum the five largest.

    ⚠ THE LATEST FISCAL YEAR IS LOAD-BEARING. Item 402(c) requires three years, so this table
    carries ~3 rows per NEO per filing. Summing the whole table would put three years of pay in
    the denominator against one year in the numerator -- the exact defect
    `def14a/flatten.py::_latest_sct_rows` exists to prevent on the parent row, which is why the
    numerator and this denominator have to apply the same rule.
    """
    if exec_comp is None or exec_comp.empty:
        return None
    need = {"ticker", "accession_number", "fiscal_year", "name", "as_of"}
    if not need.issubset(exec_comp.columns) or "total" not in exec_comp.columns:
        return None

    e = pd.DataFrame({
        "ticker": exec_comp["ticker"],
        "accession_number": exec_comp["accession_number"],
        "as_of": pd.to_datetime(exec_comp["as_of"], errors="coerce"),
        "fiscal_year": _num(exec_comp, "fiscal_year"),
        "total": _num(exec_comp, "total"),
        "imputed": _num(exec_comp, "total_imputed").fillna(0.0),
        "reconciles": _num(exec_comp, "reconciles"),
        "name": exec_comp["name"].astype(object),
        "person": exec_comp["name"].astype(object).map(person_key),
    })
    e = e.dropna(subset=["ticker", "accession_number", "as_of", "total", "fiscal_year"])
    e = e[e["total"] > 0]
    if e.empty:
        return None

    filing = ["ticker", "accession_number"]
    latest = e.groupby(filing)["fiscal_year"].transform("max")
    e = e[e["fiscal_year"] == latest]

    # One row per NEO per filing, keeping the largest total.
    #
    # ⚠ KEYED ON THE RAW NAME, DELIBERATELY NOT ON `person_key`. The shared key exists to
    # reconcile a filer's spelling drift ACROSS years, and there is no drift to reconcile
    # WITHIN one filing's own table: the rows have already been narrowed to a single fiscal
    # year, so each NEO appears exactly once, in one spelling. Using the key here only adds its
    # collisions -- it is `lastname|firstinitial`, and measured on the live table it MERGES two
    # distinct NEOs in 136 filings (1.2%), losing 142 rows from the denominator. ADM is the
    # clean example: `M. D. Andreas` and `M. L. Andreas` are two people and both key to
    # `andreas|m`, so one of them silently vanishes and the CEO's slice inflates.
    e = e.sort_values("total", ascending=False)
    e = e.drop_duplicates(subset=filing + ["name"], keep="first")

    rank = e.groupby(filing)["total"].rank(method="first", ascending=False)
    top = e[rank <= _TOP_N_NEOS]
    agg = top.groupby(filing).agg(
        as_of=("as_of", "min"),
        top5_neo_total_comp=("total", "sum"),
        n_neos_top5=("total", "size"),
        imputed_share=("imputed", "mean"),
        reconciles_share=("reconciles", "mean"),
    ).reset_index()

    thin = agg["n_neos_top5"] < _MIN_TOP5_NEOS
    tally["CPS filings with >=1 NEO total"] = int(len(agg))
    tally[f"CPS filings rejected (< {_MIN_TOP5_NEOS} NEOs)"] = int(thin.sum())
    for label, col in (("imputed", "imputed_share"), ("reconciling", "reconciles_share")):
        share = agg.loc[~thin, col].mean()
        if pd.notna(share):
            tally[f"CPS denominator rows {label} (%)"] = int(round(100.0 * float(share)))

    agg = agg.loc[~thin, ["ticker", "as_of", "top5_neo_total_comp", "n_neos_top5"]]
    return agg.reset_index(drop=True) if not agg.empty else None


def _ceo_inside_own_denominator(def14a: pd.DataFrame, exec_comp: pd.DataFrame,
                                tally: dict[str, int]) -> None:
    """Count the filings whose CEO is NOT among the NEOs their own slice divides by.

    A `ceo_pay_slice` whose numerator sits outside its own denominator is an extraction defect,
    and the only way it becomes visible is by counting it. Matched on the shared `person_key`,
    whose measured ceiling here is **98.4%** against 91.6% on raw strings -- the 6.8pp is
    exactly what the phase-1 reconciliation buys this cross-check.
    """
    # A DIAGNOSTIC MUST NOT RAISE. It needs the accession key on BOTH sides, and a caller can
    # legitimately lack it -- the slice itself joins on `(ticker, as_of)`, so `accession_number`
    # is required by this cross-check alone. Missing it means "not measurable here", not
    # "crash the build".
    if (def14a is None or def14a.empty or exec_comp is None or exec_comp.empty
            or not {"accession_number", "ceo_name_proxy"}.issubset(def14a.columns)
            or not {"accession_number", "name"}.issubset(exec_comp.columns)):
        return
    ceo = pd.DataFrame({
        "accession_number": def14a["accession_number"],
        "ceo": ceo_identity_series(def14a["ceo_name_proxy"]),
    }).dropna(subset=["accession_number", "ceo"])
    if ceo.empty:
        return
    neos = pd.DataFrame({
        "accession_number": exec_comp["accession_number"],
        "person": exec_comp["name"].astype(object).map(person_key),
    }).dropna()
    present = set(zip(neos["accession_number"], neos["person"]))
    have_rows = set(neos["accession_number"])
    hit = [(a, c) in present for a, c in zip(ceo["accession_number"], ceo["ceo"])]
    missed = [a for a, h in zip(ceo["accession_number"], hit) if not h]
    # ⚠ THE TWO MISS KINDS ARE DIFFERENT FAULTS AND ARE COUNTED SEPARATELY. Measured
    # 2026-09-08: of 1,166 misses, **1,004 (86%) are filings with NO NEO rows at all** -- an
    # extraction-COVERAGE gap, nothing to match against -- and only 162 are filings whose SCT
    # was parsed but does not contain its own CEO. Reporting one number called "extraction
    # defect" overstated the real defect by six times. (The matcher is not the problem: keying
    # on `person_key`, on `person_key` with `Last, First` un-inverted, and on the surname alone
    # match 90.4% / 90.5% / 90.9% -- a 0.5pp spread, so no matcher choice moves this.)
    no_rows = sum(1 for a in missed if a not in have_rows)
    tally["CPS filings where the CEO is in its own top five"] = int(sum(hit))
    tally["CPS misses: the filing has NO NEO rows at all"] = int(no_rows)
    tally["CPS misses: NEOs parsed but the CEO is not among them"] = int(len(missed) - no_rows)


def _slice_history(comp_src: pd.DataFrame, top5: pd.DataFrame,
                   tally: dict[str, int]) -> pd.DataFrame | None:
    """`[ticker, as_of, ceo_pay_slice]`, the CEO's share of the top five, in `(0, 1]`.

    A slice above 1 means the CEO is not inside their own denominator, which is an extraction
    defect rather than a signal -- rejected, and COUNTED (never clipped to 1.0, which would
    disguise the defect as a maximally-dominant CEO).

    Joined on `(ticker, as_of)` rather than `accession_number` so the interim frame keeps the
    `[ticker, as_of, field...]` shape every point-in-time utility in this repo expects; the two
    key sets are asserted to agree in the tests rather than at runtime.
    """
    if top5 is None or top5.empty or comp_src is None or comp_src.empty:
        return None
    num = pd.DataFrame({
        "ticker": comp_src["ticker"],
        "as_of": pd.to_datetime(comp_src["as_of"], errors="coerce"),
        "ceo_total_comp": _num(comp_src, "ceo_total_comp"),
    }).dropna(subset=["ticker", "as_of", "ceo_total_comp"])
    if num.empty:
        return None

    m = num.merge(top5, on=["ticker", "as_of"], how="inner")
    if m.empty:
        tally["CPS numerator/denominator join misses"] = int(len(num))
        return None
    s = m["ceo_total_comp"] / m["top5_neo_total_comp"].where(m["top5_neo_total_comp"] > 0)
    kept = s.where((s > 0) & (s <= 1.0))
    tally["CPS slices computed"] = int(s.notna().sum())
    tally["CPS slices rejected (outside (0, 1])"] = int(s.notna().sum() - kept.notna().sum())
    out = m[["ticker", "as_of"]].copy()
    out["ceo_pay_slice"] = kept
    out = out.dropna(subset=["ceo_pay_slice"])
    return out.reset_index(drop=True) if not out.empty else None


def _log_growth(pct: pd.DataFrame) -> pd.DataFrame:
    """A percentage change re-expressed as a LOG growth, so it is subtractable from the pay leg.

    `where(pct > -1)` because a total wipe-out (-100%) has no finite log growth, and `log1p` of
    anything below -1 is undefined rather than merely extreme.
    """
    if pct is None or pct.empty:
        return pd.DataFrame()
    return np.log1p(pct.where(pct > -1.0)).replace([np.inf, -np.inf], np.nan)


def _flag_pair(pay: pd.DataFrame, perf: pd.DataFrame) -> pd.DataFrame:
    """`1.0` where pay rose while performance fell, `0.0` otherwise, NaN when either is NaN.

    NaN-preserving on purpose: `(a > 0) & (b < 0)` on a NaN gives False, which would report
    "no misalignment" for a company whose pay or performance is simply unknown.
    """
    both = pay.notna() & perf.notna()
    return ((pay > 0) & (perf < 0)).astype("float64").where(both)


def _severity(pay: pd.DataFrame, perf: pd.DataFrame) -> pd.DataFrame:
    """`max(pay growth, 0) * max(-performance, 0)` -- a ONE-SIDED severity, not an interaction.

    ⚠ A product is not a ratio and must not be sanitised as one: both legs are already finite
    growth rates, and it is the `clip(lower=0)` BEFORE the multiplication that makes the result
    "how badly pay rose while performance fell" rather than a sign-ambiguous cross term (where
    pay falling as performance falls would otherwise multiply to a positive severity).
    """
    both = pay.notna() & perf.notna()
    return (pay.clip(lower=0.0) * (-perf).clip(lower=0.0)).where(both)


def _aligned(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The two frames on their common tickers, or a pair of empties when they share none."""
    if a is None or a.empty or b is None or b.empty:
        return pd.DataFrame(), pd.DataFrame()
    cols = a.columns.intersection(b.columns)
    if len(cols) == 0:
        return pd.DataFrame(), pd.DataFrame()
    return a[cols], b[cols]


def _alignment_family(growth: pd.DataFrame, perf: pd.DataFrame, peer_dict: dict,
                      label: str) -> dict[str, pd.DataFrame]:
    """One pay-vs-performance leg: gap, peer misalignment, flag and severity."""
    pay, prf = _aligned(growth, perf)
    if pay.empty:
        return {}
    out = {
        f"pay_{label}_gap": pay - prf,
        f"pay_up_{label}_down": _flag_pair(pay, prf),
        f"pay_up_{label}_down_severity": _severity(pay, prf),
    }
    if peer_dict:
        # ⚠ The ONLY direct call to the low-level `peer_relative` in this module, and it is
        # deliberate: both legs are z-scores from the SAME basket on the SAME day, so their
        # difference is well-posed and answers GPT's preferred question -- "is pay growing
        # faster than peers' by MORE than performance is growing faster than peers' by".
        # The result is a difference of two z-scores and therefore ships RAW; peer-z-scoring it
        # a second time would relativize the same quantity twice.
        pz = winsorize_xs(peer_relative(pay, peer_dict))
        rz = winsorize_xs(peer_relative(prf, peer_dict))
        pz, rz = _aligned(pz, rz)
        if not pz.empty:
            out[f"pay_{label}_peer_misalignment"] = pz - rz
    return {k: v for k, v in out.items() if v is not None and not v.empty
            and v.notna().any().any()}


def pay_fields(
    def14a: pd.DataFrame | None,
    exec_comp: pd.DataFrame | None,
    fundamentals: pd.DataFrame | None,
    close_total: pd.DataFrame | None,
    peer_dict: dict,
    idx: pd.DatetimeIndex,
) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    """(daily wide frames keyed by feature name, data-quality tallies).

    EACH FAMILY DEGRADES INDEPENDENTLY, because the four sources accrue at different rates in a
    half-built database: no `exec_comp` -> no pay slice while pay growth still builds; no
    `close_total` -> no return-based features while the revenue ones still build. Every skip is
    recorded in the tally with its reason, so "no feature" is never indistinguishable from
    "feature built and empty".

    `exec_comp` is expected to have been through `impute_exec_comp` already -- the caller owns
    the clean-on-read, exactly as it owns `impute_def14a`.
    """
    tally: dict[str, int] = {}
    frames: dict[str, pd.DataFrame] = {}

    comp_hist = _comp_history(def14a, tally)
    if comp_hist is None:
        tally["skipped: no def14a ceo_total_comp -> no pay families at all"] = 1
        return {}, tally

    # ---- family 5: level, guarded growth, turnover ---- #
    family5: dict[str, pd.DataFrame] = {}
    for name in ("log_ceo_total_comp", "ceo_comp_growth_1y", "ceo_turnover_flag"):
        f = fundamentals_to_daily(comp_hist, name, idx)
        if not f.empty and f.notna().any().any():
            family5[name] = f
    family5, expiry5 = expire_event_fields(family5, comp_hist, EVENT_FIELDS)
    # `log_ceo_total_comp` is a LEVEL: a CEO's package is a standing fact between proxies, so it
    # is correctly off the 548-day event clock -- but "not an event" was allowed to mean "never
    # expires", and a 2014 package is not evidence about today either. 1,095 days.
    family5, expiry_lvl = expire_level_fields(family5, comp_hist, LEVEL_FIELDS)

    # ---- family 6: the exact CEO Pay Slice ---- #
    family6: dict[str, pd.DataFrame] = {}
    top5 = _top5_history(exec_comp, tally)
    if top5 is None:
        tally["skipped: no def14a_executive_comp -> no exact CEO Pay Slice"] = 1
    else:
        _ceo_inside_own_denominator(def14a, exec_comp, tally)
        slices = _slice_history(def14a, top5, tally)
        if slices is not None:
            level = fundamentals_to_daily(slices, "ceo_pay_slice", idx)
            if not level.empty and level.notna().any().any():
                family6["ceo_pay_slice"] = level
            delta = fiscal_change_to_daily(slices, "ceo_pay_slice", idx,
                                           kind="diff", periods=1)
            if not delta.empty and delta.notna().any().any():
                family6["ceo_pay_slice_delta_1y"] = delta
            # The delta's provenance is the LEVEL's filing -- the later leg is what made the
            # change knowable -- so it ages against `ceo_pay_slice`, a column that exists in
            # the history frame, not against a name that does not.
            family6, expiry6 = expire_event_fields(
                family6, slices, EVENT_FIELDS,
                sources={"ceo_pay_slice_delta_1y": "ceo_pay_slice"})
            expiry5.update(expiry6)
            # `ceo_pay_slice` is the other LEVEL -- the CEO's share of the top five is a
            # standing fact of the same filing, and it ages against its own history frame.
            family6, lvl6 = expire_level_fields(family6, slices, LEVEL_FIELDS)
            expiry_lvl.update(lvl6)

    # ---- family 7: pay vs performance ---- #
    growth = family5.get("ceo_comp_growth_1y", pd.DataFrame())
    family7: dict[str, pd.DataFrame] = {}
    if growth.empty:
        tally["skipped: no guarded pay growth -> no pay-vs-performance family"] = 1
    else:
        rev_growth = pd.DataFrame()
        if fundamentals is not None and not fundamentals.empty:
            rev_growth = _log_growth(fiscal_change_to_daily(
                fundamentals, "totalRevenue", idx, kind="pct",
                periods=infer_yoy_periods(fundamentals)))
        if rev_growth.empty:
            tally["skipped: no totalRevenue -> no revenue-based misalignment"] = 1
        else:
            family7.update(_alignment_family(growth, rev_growth, peer_dict, "revenue"))

        ret = pd.DataFrame()
        if close_total is not None and not close_total.empty:
            ret = _log_growth(close_total.pct_change(_RETURN_WINDOW))
        if ret.empty:
            tally["skipped: no close_total -> no return-based misalignment"] = 1
        else:
            family7.update(_alignment_family(growth, ret, peer_dict, "return"))

        # Family 7 is a DIFFERENCE of a pay leg against a performance leg, so its provenance is
        # the pay leg's filing: the return moves daily but the package it is measured against
        # was set at the proxy, and that is the date the 548-day horizon has to age.
        family7, expiry7 = expire_event_fields(
            family7, comp_hist, EVENT_FIELDS,
            sources={k: "ceo_comp_growth_1y" for k in family7})
        expiry5.update(expiry7)

    for name, (expired, before) in expiry5.items():
        if expired:
            tally[f"expired >548d: {name}"] = expired
            tally[f"non-null before expiry: {name}"] = before
    # Tallied under their OWN horizon, not folded into the 548-day lines: a reader comparing
    # two counts has to be able to see which clock each was measured on.
    for name, (expired, before) in expiry_lvl.items():
        if expired:
            tally[f"expired >{LEVEL_MAX_AGE_DAYS}d: {name}"] = expired
            tally[f"non-null before expiry: {name}"] = before

    frames.update(family5)
    frames.update(family6)
    frames.update(family7)
    return frames, tally
