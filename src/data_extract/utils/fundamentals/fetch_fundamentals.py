"""
fetch_fundamentals.py (src/data_extract/utils/fundamentals/fetch_fundamentals.py)
--------------------------------------------------------------------------------
Extracts ~10-year point-in-time quarterly fundamental history via SEC EDGAR 
`companyfacts` (`build_fundamentals_history_sec()`). Keyed on SEC filing date 
(`as_of`) to eliminate look-ahead bias in backtesting.

Core Data Logic:
- TTM Flows: Sums discrete Q1–Q3 and derived Q4 (FY - Q1 - Q2 - Q3).
- Market Cap & Valuation: Emits `sharesOutstanding` and `stockholdersEquity`. 
  Market Cap (`sharesOutstanding` * daily close) and B/P are calculated in factors layer.

Output Schema :
- Keys: `ticker`, `as_of` (filing date), `fiscal_end`
- Financial Ratios & Growth: `revenueGrowth`, `earningsGrowth`, `grossMargins`, `operatingMargins`, 
  `profitMargins`, `returnOnEquity`, `debtToEquity`
- Financial Levels: `cash`, `shortTermDebt`, `longTermDebt`, `totalLiabilities`, `currentAssets`, 
  `currentLiabilities`, `goodwill`, `totalAssets`, `stockholdersEquity`, `sharesOutstanding`
- TTM & Discrete Flows: `revenue(_q)`, `netIncome(_q)`, `ebitda(_q)`, `freeCashflow(_q)`, 
  `operatingCashFlow`, `researchAndDevelopment`, `sellingGeneralAdmin`, `stockBasedComp`, 
  `acquisitions`, `interestExpense`

Standardized GAAP Restatements (Emits Raw Metric + Adjustment Size):
- `cash`: Netted to unrestricted cash (tracks `restrictedCash`).
- `inventory` / `costOfRevenue`: Standardized from LIFO to FIFO (tracks `lifoReserve`).
- `operatingIncome`: Restores pre-2018 non-service pension costs to maintain ASU-2017-07 comparability (tracks `nonServicePensionCost`).
- `totalRevenue`: Nets pass-through excise/sales taxes (tracks `exciseTaxAdjustment`).
- `totalAssetsExLease`: Excludes ASC-842 ROU assets to prevent 2019 balance sheet distortion (tracks `operatingLeaseRouAsset`).
- `capexGlobal`: Includes both operating- and finance-lease additions.

Downstream Consumption:
Feeds `fundamental_features.py` (distress metrics, S&M efficiency, M&A growth, SBC intensity) 
and risk-neutral factor computations.
"""

import logging

import numpy as np
import pandas as pd
from src.constants.constants import PPE_NET_MIN_SHARE_OF_ROLLFORWARD
   
from src.data_extract.utils.fundamentals.fundamentals_tags import (
    ANNUAL_MAX_DAYS, ANNUAL_MIN_DAYS, ASU_2017_07_EFFECTIVE, CHARGE_FLOWS,
    DILUTED_SHARES_TAGS, EXTRA_FLOW_TAGS, EXTRA_STOCK_TAGS, FLOW_TAGS,
    GROSS_MARGIN_MAX, GROSS_MARGIN_MIN, 
    LATEST_DURATION_TAGS,
    SHARES_TAGS, STOCK_TAGS, TTM_QUARTERS,
)
from src.validate.fundamentals_validation import (
    apply_plausibility_guards,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Concept extraction                                                          #
# --------------------------------------------------------------------------- #
_CONCEPT_COLS = ["end", "start", "filed", "form", "fp", "val"]


def _extract_concept(section: dict, tag_candidates: list[str]) -> pd.DataFrame:
    """
    Observations for a logical concept as [end, start, filed, form, fp, val],
    COALESCED across all candidate tags rather than taking only the first one.

    Filers split the same economic concept across tags — over time (ASC-606
    revenue: `Revenues` pre-2018, `RevenueFromContractWithCustomer…` after) and
    by scope (`NetIncomeLoss` excl. NCI vs `ProfitLoss` incl. NCI; equity with /
    without NCI). Taking only the first present tag truncated history badly
    (e.g. CVX revenue from 2018, AVGO/CAT net income near-empty, JNJ/UNH ROE
    only a few years). We therefore union every candidate and, PER PERIOD
    (start, end), keep the highest-priority (earliest-listed) candidate that
    reported it — retaining all of that candidate's filings so the downstream
    earliest-disclosure / point-in-time logic is unchanged.
    """
    frames = []
    for prio, tag in enumerate(tag_candidates):
        if tag not in section:
            continue
        units = section[tag].get("units", {})
        # Preferred units first, then the unit with the MOST observations. The old
        # fallback took the FIRST unit key, which silently picked a stray: CSCO reports
        # EarningsPerShareDiluted under both 'pure' (4 facts) and 'USD/shares' (276), and
        # 'pure' came first -> 13 of 74 quarters of EPS instead of 73.
        unit_key = next((u for u in ("USD", "shares", "USD/shares") if u in units), None)
        if unit_key is None and units:
            unit_key = max(units, key=lambda u: len(units[u]))
        if unit_key is None:
            continue
        rows = [{"end": o.get("end"), "start": o.get("start"),
                 "filed": o.get("filed"), "form": o.get("form"),
                 "fp": o.get("fp"), "val": o.get("val"), "_prio": prio}
                for o in units[unit_key]]
        if rows:
            frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame(columns=_CONCEPT_COLS)

    df = pd.concat(frames, ignore_index=True)
    for c in ("end", "start", "filed"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    # per (start, end) period, keep only the best-priority candidate's rows
    df["_min_prio"] = df.groupby(["start", "end"], dropna=False)["_prio"].transform("min")
    df = df[df["_prio"] == df["_min_prio"]]
    return df[_CONCEPT_COLS].reset_index(drop=True)


def _annual_flow(df: pd.DataFrame) -> pd.DataFrame:
    """Keep annual (~365d duration) observations; first disclosure per fiscal end."""
    if df.empty:
        return df
    d = df.dropna(subset=["end", "start", "filed", "val"]).copy()
    dur = (d["end"] - d["start"]).dt.days
    d = d[(dur >= ANNUAL_MIN_DAYS) & (dur <= ANNUAL_MAX_DAYS)]
    if d.empty:
        return d
    # earliest filing that disclosed each fiscal-year-end value
    d = d.sort_values("filed").drop_duplicates(subset=["end"], keep="first")
    return d[["end", "filed", "val"]].sort_values("end").reset_index(drop=True)


def _quarterly_flow(df: pd.DataFrame) -> pd.DataFrame:
    """Discrete quarterly flow observations [end, filed, val].

    XBRL flow facts come in two shapes and this handles both:
      * DISCRETE "three-months-ended" (~90d) facts -> used directly (typical for
        income-statement items, giving Q1-Q3).
      * YEAR-TO-DATE cumulative facts (3M/6M/9M/FY, all sharing the fiscal-year
        `start`) -> typical for CASH-FLOW items; de-cumulated into quarters by
        differencing consecutive period ends within the fiscal year.
    Any fiscal-year end still missing a quarter (pure-discrete filers that never
    file a Q4 10-Q) gets Q4 DERIVED as FY - (Q1 + Q2 + Q3).

    `filed` on each quarter is the filing that made it computable (the later of
    the cumulatives involved), so nothing is stamped before it is public.
    """
    if df.empty:
        return df
    d = df.dropna(subset=["end", "start", "filed", "val"]).copy()
    d["dur"] = (d["end"] - d["start"]).dt.days
    d = d[(d["dur"] >= 45) & (d["dur"] <= ANNUAL_MAX_DAYS)]
    if d.empty:
        return pd.DataFrame(columns=["end", "filed", "val"])
    # earliest filing that disclosed each (start, end) period
    d = d.sort_values("filed").drop_duplicates(subset=["start", "end"], keep="first")

    # De-cumulate within each fiscal-year `start`: discrete = value - previous
    # cumulative (by end); the first (shortest) period is already discrete.
    d = d.sort_values(["start", "end"])
    grp = d.groupby("start", sort=False)
    disc = d["val"].astype(float) - grp["val"].shift(1)
    is_first = grp.cumcount() == 0
    disc = disc.where(~is_first, d["val"])
    implied = (d["end"] - grp["end"].shift(1)).dt.days
    implied = implied.where(~is_first, d["dur"])

    q = d.assign(val=disc, implied=implied)
    # keep quarter-length periods. Upper bound 120d (not 100) so 52/53-week retailers
    # with a 16-week fiscal Q1 (~112d, e.g. KR/COST/TGT) aren't dropped -> their flows
    # now form 4 quarters and TTM populates. 6-month YTD (181d) stays excluded.
    q = q[(q["implied"] >= 75) & (q["implied"] <= 120)]
    q = (q.sort_values("filed").drop_duplicates(subset=["end"], keep="first")
           [["end", "filed", "val"]])

    # Derive Q4 for any fiscal-year end not already covered above.
    a = d[(d["dur"] >= ANNUAL_MIN_DAYS) & (d["dur"] <= ANNUAL_MAX_DAYS)]
    a = (a.sort_values("filed").drop_duplicates(subset=["end"], keep="first")
           [["end", "filed", "val"]])
    q_ends = set(q["end"])
    derived = []
    for _, r in a.iterrows():
        fye = r["end"]
        if fye in q_ends:
            continue
        prior = q[(q["end"] > fye - pd.Timedelta(days=340))
                  & (q["end"] <= fye - pd.Timedelta(days=20))]
        if len(prior) == 3:
            derived.append({
                "end": fye,
                "filed": max(r["filed"], prior["filed"].max()),
                "val": r["val"] - prior["val"].sum(),
            })
    if derived:
        q = pd.concat([q, pd.DataFrame(derived)], ignore_index=True)
    return (q.sort_values("end").drop_duplicates(subset=["end"], keep="first")
              .reset_index(drop=True))


def _instant_stock(df: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time balance items: first disclosure per period end."""
    if df.empty:
        return df
    d = df.dropna(subset=["end", "filed", "val"]).copy()
    d = d.sort_values("filed").drop_duplicates(subset=["end"], keep="first")
    return d[["end", "filed", "val"]].sort_values("end").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Per-ticker fundamentals history builder                                      #
# --------------------------------------------------------------------------- #
# curated (always-checked) concepts + the "spine" whose period-ends define the grid
_CURATED_FLOWS = ["totalRevenue", "netIncome", "grossProfit", "costOfRevenue", "operatingIncome",
                  "depAmort", "operatingCashFlow", "capex", "researchAndDevelopment",
                  "sellingGeneralAdmin", "stockBasedComp", "acquisitions", "interestExpense"]
_CURATED_STOCKS = ["stockholdersEquity", "totalLiabilities", "longTermDebt", "cash",
                   "shortTermDebt", "currentAssets", "currentLiabilities", "goodwill", "totalAssets"]
_SPINE_FLOWS = ("totalRevenue", "netIncome", "operatingCashFlow")
_SPINE_STOCKS = ("totalAssets", "stockholdersEquity", "totalLiabilities")


# net-income-to-common: added to the netIncome coalesce ONLY for filers with no
# material preferred dividends (see `_net_income_tags`).
_NET_INCOME_TO_COMMON_TAG = "NetIncomeLossAvailableToCommonStockholdersBasic"
_TO_COMMON_MIN_OVERLAP = 4     # need this many primary-vs-to-common overlaps to judge
_TO_COMMON_TOL = 0.02          # per-period relative gap treated as "equal"
_TO_COMMON_MIN_MATCH = 0.90    # share of overlap periods that must match to trust it


def _concept_periods(gaap: dict, tag: str) -> dict[tuple, float]:
    """`{(start, end): val}` for one raw us-gaap tag (USD unit), for the preferred-
    dividend safety check — light-weight, no coalescing/date parsing."""
    units = gaap.get(tag, {}).get("units", {})
    uk = "USD" if "USD" in units else next(iter(units), None)
    if uk is None:
        return {}
    return {(o["start"], o["end"]): o["val"] for o in units[uk]
            if o.get("start") and o.get("end") and o.get("val") is not None}


def _net_income_tags(gaap: dict) -> list[str]:
    """netIncome coalesce list, extended with the net-income-to-common tag ONLY when
    it materially equals ProfitLoss/NetIncomeLoss on their overlapping periods — i.e.
    the filer has no meaningful preferred dividends. This recovers pre-2016 netIncome
    for no-preferred filers (e.g. WAT 2014-2015) without contaminating the YTD
    de-cumulation chain of preferred-paying REITs/banks/insurers (WELL, O, SPG, USB,
    VTR, ...), whose to-common figure is net of preferred dividends."""
    # Start WITHOUT the to-common tag. `FLOW_TAGS["netIncome"]` lists it as a candidate, so
    # starting from that list left the guard unable to reject anything -- it could only
    # re-append a tag that was already there, and every preferred-paying filer's netIncome
    # was silently net of preferred dividends regardless.
    tags = [t for t in FLOW_TAGS["netIncome"] if t != _NET_INCOME_TO_COMMON_TAG]
    common = _concept_periods(gaap, _NET_INCOME_TO_COMMON_TAG)
    if not common:
        return tags
    primary: dict[tuple, float] = {}
    for tag in ("NetIncomeLoss", "ProfitLoss"):
        for k, v in _concept_periods(gaap, tag).items():
            primary.setdefault(k, v)
    rel = [abs(common[k] - primary[k]) / abs(primary[k])
           for k in common if k in primary and primary[k]]
    if len(rel) >= _TO_COMMON_MIN_OVERLAP and \
            sum(r <= _TO_COMMON_TOL for r in rel) / len(rel) >= _TO_COMMON_MIN_MATCH:
        tags.append(_NET_INCOME_TO_COMMON_TAG)
    return tags


def _extract_all(gaap: dict, dei: dict) -> tuple[dict, dict, dict, pd.DataFrame, dict, pd.DataFrame]:
    """Raw us-gaap/dei -> per-concept quarterly flows, annual full-year fallbacks,
    instant balance-sheet levels, cover-page shares (dei) and the LATEST-value duration
    concepts (share counts, segment count) that must never be TTM-summed."""
    flow_tags = {**FLOW_TAGS, **EXTRA_FLOW_TAGS}
    flow_tags["netIncome"] = _net_income_tags(gaap)   # preferred-dividend-guarded fill
    raw = {k: _extract_concept(gaap, tags) for k, tags in flow_tags.items()}
    flows = {k: _quarterly_flow(v) for k, v in raw.items()}
    annuals = {k: _annual_flow(v) for k, v in raw.items()}      # full-year TTM fallback
    stocks = {k: _instant_stock(_extract_concept(gaap, tags))
              for k, tags in {**STOCK_TAGS, **EXTRA_STOCK_TAGS}.items()}
    _sh = SHARES_TAGS["sharesOutstanding"]
    shares = _instant_stock(_extract_concept(dei, _sh) if any(t in dei for t in _sh)
                            else _extract_concept(gaap, _sh))
    latest = {k: _instant_stock(_extract_concept(gaap, tags))
              for k, tags in LATEST_DURATION_TAGS.items()}
    return flows, annuals, stocks, shares, latest, _option_overhang(gaap)


def _spine_grid(flows: dict, stocks: dict) -> "pd.DatetimeIndex | None":
    """Fiscal-quarter row grid = UNION of period-ends across the core spine concepts,
    so a revenue-tag gap (banks ~2013, utilities ~2014) doesn't truncate the ticker."""
    ends: set = set()
    for k in _SPINE_FLOWS:
        s = flows.get(k)
        if s is not None and not s.empty:
            ends |= set(s["end"])
    for k in _SPINE_STOCKS:
        s = stocks.get(k)
        if s is not None and not s.empty:
            ends |= set(s["end"])
    return pd.DatetimeIndex(sorted(ends)) if ends else None


def _option_overhang(gaap: dict) -> pd.DataFrame:
    """(diluted - basic) / basic weighted-average share count, matched on the EXACT period
    (start AND end) both counts were reported for -> [end, filed, val].

    Matching on `end` alone is not enough: a filer tags both a three-month and a
    year-to-date count ending on the same date, and taking each tag's earliest-filed fact
    per `end` independently can pair a YTD basic against a quarterly diluted (BLDR). Matching
    on the period makes the wedge structurally non-negative, since diluted shares are basic
    plus dilutive potential (equal when antidilutive)."""
    b = _extract_concept(gaap, LATEST_DURATION_TAGS["basicShares"])
    d = _extract_concept(gaap, DILUTED_SHARES_TAGS)
    if b.empty or d.empty:
        return pd.DataFrame(columns=["end", "filed", "val"])
    j = b.merge(d, on=["start", "end"], suffixes=("_b", "_d"))
    j = j[(j["val_b"] > 0) & j["val_d"].notna()]
    if j.empty:
        return pd.DataFrame(columns=["end", "filed", "val"])
    j = j.assign(val=(j["val_d"] - j["val_b"]) / j["val_b"],
                 filed=j[["filed_b", "filed_d"]].max(axis=1))
    return _instant_stock(j[["end", "filed", "val"]])


def _assemble_base(ends, flows, annuals, stocks, shares, latest, overhang) -> pd.DataFrame:
    """Align every concept onto the quarter grid in ONE frame construction (a single
    dict -> one DataFrame, no repeated column inserts -> no pandas fragmentation).
    Exact-end join for flows/annuals/stocks; backward as-of (reindex-ffill) for the
    cover-page shares and the LATEST-value duration concepts (`latest`); balance-sheet
    levels carried forward across interim quarters; `as_of` = latest filing date among a
    row's concepts (point-in-time / leak-free)."""
    cols: dict[str, object] = {}

    def exact(key, src, filed=True):
        if src is None or src.empty:
            return
        s = src.drop_duplicates("end").set_index("end").reindex(ends)
        cols[key] = s["val"].to_numpy()
        if filed:
            cols[key + "_filed"] = s["filed"].to_numpy()

    def asof(key, src, filed=False):     # value is dated near the filing, not period end
        if src is None or src.empty:
            return
        s = src.drop_duplicates("end").sort_values("end").set_index("end")
        cols[key] = s["val"].reindex(ends, method="ffill").to_numpy()
        if filed:
            cols[key + "_filed"] = s["filed"].reindex(ends, method="ffill").to_numpy()

    flow_keys = _CURATED_FLOWS + list(EXTRA_FLOW_TAGS)
    for key in flow_keys:
        exact(key, flows.get(key))
    for key in flow_keys:                # annual full-year fallback (suffix `_ann`)
        a = annuals.get(key)
        if a is not None and not a.empty:
            cols[key + "_ann"] = a.drop_duplicates("end").set_index("end")["val"].reindex(ends).to_numpy()
    for key in _CURATED_STOCKS + list(EXTRA_STOCK_TAGS):
        exact(key, stocks.get(key))
    asof("sharesOutstanding", shares, filed=True)
    for key in LATEST_DURATION_TAGS:
        asof(key, latest.get(key))

    asof("optionOverhang", overhang)

    base = pd.DataFrame(cols, index=ends)
    level_cols = [k for k in (list(STOCK_TAGS) + list(EXTRA_STOCK_TAGS)) if k in base.columns]
    if level_cols:                       # carry point-in-time levels forward (~1y cap)
        base[level_cols] = base[level_cols].ffill(limit=4)
    filed_cols = [c for c in base.columns if c.endswith("_filed")]

    # ---- `as_of` = when this fiscal period's CORE financials became public -------------
    # It used to be MAX(filed) over ALL ~300 concepts on the row, which let ONE peripheral
    # straggler date the whole row. A filer routinely first tags a minor concept for an old
    # period only in the NEXT year's filing, as a prior-year comparative: measured on ATO's
    # 2024-06-30 row, 44 of 45 concepts were filed 2024-08-07 (+38d, the real 10-Q) while
    # `CommonStockDividendsPerShareDeclared` first appeared 2025-08-06 (+402d) -- and max()
    # handed the row that date. Across the table that produced a MEDIAN as_of lag of 383 days,
    # 52% of rows stamped >200d after their period end, and -- because the table is keyed
    # (ticker, as_of) -- 13.8% of rows out of chronological order for 493 of 498 tickers, so
    # every QoQ / TTM feature differenced non-adjacent quarters.
    #
    # The SPINE concepts are the right anchor: the row grid is built from their period-ends
    # (see `_spine_grid`), so the row exists precisely because they exist, and it is knowable
    # once they are all public. Per-row fallback to the old all-column max keeps a row that
    # somehow has no spine `filed` (never silently drops one).
    # `as_of` must equal THE FILING DATE of the periodic report that disclosed this period,
    # because that is exactly the condition prediction runs under: the daily job scores with
    # whatever a filing contains on the day it lands, and re-runs later when more is filed. Train
    # and serve therefore have to be built the same way -- anything that arrived AFTER the filing
    # is blanked below rather than back-dated into the row.
    #
    # The estimator is the MEDIAN of the spine concepts' filing dates, not the max or the min:
    #   * max  -> one straggler dates the whole row (ATO 2024-06-30: five spine concepts filed
    #             2024-08-07, `operatingCashFlow` first filed 2025-08-06 -> max = +402d),
    #   * min  -> an early earnings-release 8-K (these exist: 17 on ATO's core tags) dates the row
    #             before the 10-Q, so the leak guard would blank everything the 10-Q brought,
    #   * median over 6 concepts survives BOTH at once and lands on the real filing date.
    spine_filed = [f"{k}_filed" for k in (_SPINE_FLOWS + _SPINE_STOCKS)
                   if f"{k}_filed" in base.columns]
    if spine_filed:
        # median of datetimes: rank-based, so no interpolation between two dates
        core = base[spine_filed].apply(
            lambda r: r.dropna().sort_values().iloc[(r.notna().sum() - 1) // 2]
            if r.notna().any() else pd.NaT, axis=1)
        base["as_of"] = core.fillna(base[filed_cols].max(axis=1)) if filed_cols else core
    else:
        base["as_of"] = base[filed_cols].max(axis=1) if filed_cols else pd.NaT
    base = base[base["as_of"].notna()]

    # ---- no as_of may PRECEDE its own period end ---------------------------------------
    # The median survives one early earnings-release 8-K, but not several: ROP's 2009-12-31
    # row had enough spine concepts filed early that the MEDIAN itself landed 2009-11-02 --
    # 59 days BEFORE the quarter closed. That is a look-ahead leak, not a lag: the row
    # claims the full-year numbers were public while the year was still running.
    # Repair with the earliest spine filing that is actually >= the period end (the first
    # filing that COULD have disclosed a completed period); if a row has none, it is
    # dropped, because nothing in it is datable without inventing a disclosure date.
    if spine_filed:
        too_early = base["as_of"] < base.index.to_series()
        if too_early.any():
            ends = base.index.to_series()
            valid = base[spine_filed].where(base[spine_filed].ge(ends, axis=0))
            repaired = valid.min(axis=1)
            base.loc[too_early, "as_of"] = repaired[too_early]
            base = base[base["as_of"].notna() & (base["as_of"] >= base.index.to_series())]

    # ---- leak guard -------------------------------------------------------------------
    # Dating the row from the spine alone would otherwise expose a value that was NOT yet
    # public at `as_of` (exactly the straggler above). A concept filed after `as_of` is
    # blanked for THIS period; it is unaffected in later periods, whose as_of is past its
    # filing. So the row is both correctly dated AND still strictly point-in-time.
    for col in filed_cols:
        value_col = col[: -len("_filed")]
        if value_col in base.columns:
            not_yet = base[col].notna() & (base[col] > base["as_of"])
            if not_yet.any():
                base.loc[not_yet, value_col] = np.nan

    return base.rename_axis("end").reset_index().sort_values("end").reset_index(drop=True)


class TickerFundamentalsBuilder:
    """Builds one ticker's point-in-time QUARTERLY fundamentals history from its SEC
    companyfacts. Fundamentals are the backbone of the cube, so the build is split
    into small, testable stages — extract -> grid -> assemble -> derive:

        _extract_all    raw us-gaap/dei -> quarterly / annual / instant concept series
        _spine_grid     the fiscal-quarter row grid (union of core period-ends)
        _assemble_base  align concepts onto the grid in ONE frame + point-in-time as_of
        _derive_history TTM sums (annual fallback), margins/ratios, level reconstructions

    Flows are trailing-twelve-month sums of discrete quarters (annual full-year value
    as a fallback); balance-sheet items are the last filed level carried point-in-time;
    `as_of` is the latest filing date among a row's concepts, so nothing leaks.
    """

    def __init__(self, ticker: str, facts: dict, sector: str | None = None,
                 industry_group: str | None = None):
        self.ticker = ticker
        self.sector = sector
        self.industry_group = industry_group
        self._gaap = facts.get("facts", {}).get("us-gaap", {})
        self._dei = facts.get("facts", {}).get("dei", {})

    def build(self) -> pd.DataFrame:
        flows, annuals, stocks, shares, latest, overhang = _extract_all(self._gaap, self._dei)
        ends = _spine_grid(flows, stocks)
        if ends is None:
            return pd.DataFrame()
        base = _assemble_base(ends, flows, annuals, stocks, shares, latest, overhang)
        if base.empty:
            return pd.DataFrame()
        return _derive_history(base, self.ticker, self.sector, self.industry_group)


def build_ticker_history(ticker: str, facts: dict, sector: str | None = None,
                         industry_group: str | None = None) -> pd.DataFrame:
    """One row per FISCAL QUARTER, keyed on the filing date (`as_of`). Public entry
    point — delegates to TickerFundamentalsBuilder (see it for the full pipeline)."""
    return TickerFundamentalsBuilder(ticker, facts, sector, industry_group).build()


# --------------------------------------------------------------------------- #
# Derived history: TTM flows (annual fallback), margins/ratios, reconstructions #
# --------------------------------------------------------------------------- #
def _derive_history(base: pd.DataFrame, ticker: str, sector, industry_group) -> pd.DataFrame:
    """Turn the assembled quarter grid into the output history: TTM flow sums (with
    the annual full-year fallback), margins / ratios, and clean balance-sheet level
    reconstructions. Flows are seasonality-free trailing-twelve-month; balance-sheet
    items are point-in-time; `as_of` was already stamped as the latest filing date."""

    # ---- discrete quarterly numerics ----
    def col(name):
        # a concept a filer never reports isn't a column in the assembled frame -> NaN
        if name in base.columns:
            return pd.to_numeric(base[name], errors="coerce")
        return pd.Series(float("nan"), index=base.index)

    def ttm(s):
        # trailing 12 months = sum of the 4 most recent quarters
        return s.rolling(TTM_QUARTERS, min_periods=TTM_QUARTERS).sum()

    def ttm_a(key, charge=False):
        """TTM of the quarterly series, falling back to the forward-filled ANNUAL
        full-year value where quarters are unavailable (a filer's reported full year
        IS a trailing-twelve-month). Leak-free: the annual value was filed before the
        interim quarters it is carried into. Used for filers that report a flow only
        annually (no de-cumulable interim quarters)."""
        s = col(key)
        if charge:
            s = s.fillna(0.0)
        q = ttm(s)
        acol = key + "_ann"
        if acol in base.columns:
            ann = pd.to_numeric(base[acol], errors="coerce").ffill(limit=4)
            q = q.where(q.notna(), ann)
        return q

    # discrete SINGLE-QUARTER values (before rolling) -> "latest quarter" features
    rev_q = col("totalRevenue")
    ni_q = col("netIncome")
    oi_q = col("operatingIncome")
    da_q = col("depAmort")
    ocf_q = col("operatingCashFlow")
    capex_q = col("capex")
    ebitda_q = oi_q + da_q.fillna(0)
    # bottom-up single-quarter EBITDA for filers without an operating-income line
    ebitda_q = ebitda_q.where(ebitda_q.notna(),
                              ni_q + col("incomeTaxExpense").fillna(0)
                              + col("interestExpense").fillna(0) + da_q.fillna(0))
    fcf_q = ocf_q - capex_q.fillna(0)

    rev_ttm = ttm_a("totalRevenue")
    ni_ttm = ttm_a("netIncome")
    # EXCISE / SALES TAX collected: 19.5% of filers tag revenue under the INCLUDING-
    # assessed-tax element, which reports the tax they merely collect as their own top line
    # (tobacco, beverages, fuel distribution) -> revenue, margins and every price multiple
    # are non-comparable to peers. Net it off ONLY for the periods where the EXCLUDING
    # element is absent (i.e. the including-tag is what the coalesce used), so a filer that
    # reports the clean element is untouched and nothing is deducted twice. The size of the
    # adjustment is kept as `exciseTaxes` in its own right.
    _excise = ttm_a("exciseTaxes", charge=True)
    _rev_excl, _rev_incl = ttm_a("revenueExcludingAssessedTax"), ttm_a("revenueIncludingAssessedTax")
    _excise_adj = _excise.where(_rev_excl.isna() & _rev_incl.notna()).fillna(0.0)
    rev_ttm = rev_ttm - _excise_adj
    # Financials top line: the ASC-606 contract-revenue element tags only a FEE SLICE for
    # banks / insurers / asset managers, understating revenue many-fold (e.g. FITB $0.5B
    # vs true $8B, MET, AIG) -> nonsense margins. For the Financials sector ONLY, rebuild
    # revenue from its components and take the most complete signal — the fee slice is
    # always a subset, so max() never double-counts: net interest income + noninterest
    # income (banks), premiums earned + net investment income (insurers), or the
    # consolidated `Revenues` line (asset managers). Non-financials are untouched.
    if sector == "Financials":
        nii, noni = ttm_a("netInterestIncome"), ttm_a("noninterestIncome")
        prem, inv = ttm_a("premiumsEarned"), ttm_a("netInvestmentIncome")
        bank_rev = (nii.fillna(0) + noni.fillna(0)).where(nii.notna() | noni.notna())
        # insurer GAAP top line = premiums earned + net investment income + REALIZED
        # investment gains/losses (the third leg was missing). Gains are 0-filled, so a
        # quarter with none is unchanged.
        insurer_rev = ((prem.fillna(0) + inv.fillna(0)
                        + ttm_a("realizedInvestmentGains", charge=True).fillna(0))
                       .where(prem.notna() | inv.notna()))
        rev_ttm = pd.concat([rev_ttm, bank_rev, insurer_rev, ttm_a("revenuesTotal")],
                            axis=1).max(axis=1)
    elif sector == "Real Estate":
        # REITs tag rental income under the operating-LEASE elements (leases are outside
        # ASC-606), so the contract-revenue element the coalesce grabs is only a small FEE
        # slice (e.g. CPT $11M, EXR $52M vs ~$1B of rent). Take the larger of the coalesced
        # total and the rental line: rent is a subset, so max() never double-counts, and
        # REITs that DO tag rent under contract-revenue (e.g. ARE/PLD) keep their higher total.
        rev_ttm = pd.concat([rev_ttm, ttm_a("rentalIncome")], axis=1).max(axis=1)
    elif sector == "Energy":
        # E&P filers tag the top line as oil & gas revenue; integrated majors report a
        # fuller `Revenues`, which max() keeps (oil&gas is a subset -> no double-count).
        # Sector-gated so a non-energy filer consolidating an oil&gas portfolio company
        # (e.g. an asset manager) is untouched.
        rev_ttm = pd.concat([rev_ttm, ttm_a("oilGasRevenue")], axis=1).max(axis=1)
    gp_ttm = ttm_a("grossProfit")
    cor_ttm = ttm_a("costOfRevenue")
    oi_ttm = ttm_a("operatingIncome")
    da_ttm = ttm_a("depAmort")
    ocf_ttm = ttm_a("operatingCashFlow")
    capex_ttm = ttm_a("capex")
    rnd_ttm = ttm_a("researchAndDevelopment")
    sga_ttm = ttm_a("sellingGeneralAdmin")
    sbc_ttm = ttm_a("stockBasedComp")
    acq_ttm = ttm_a("acquisitions", charge=True)     # no acquisition this quarter = 0 (charge flow)
    int_ttm = ttm_a("interestExpense")
    eq = col("stockholdersEquity")          # instant (point-in-time), not summed
    liab = col("totalLiabilities")
    ltd = col("longTermDebt")
    # instant balance-sheet levels carried raw for the distress / M&A features
    cash = col("cash")
    std_debt = col("shortTermDebt")
    cur_a = col("currentAssets")
    cur_l = col("currentLiabilities")
    goodwill = col("goodwill")
    assets = col("totalAssets")

    # ---- CLEAN (unrestricted, investment-free) CASH -------------------------------- #
    # `cash` now coalesces only the unrestricted elements. Where a filer tags none of them
    # we DERIVE it from the broader total instead of silently accepting the broader number:
    #   cash = cash+restricted total - restricted     (95.6% of filers report the total)
    #   cash = cash+ST-investments   - ST investments
    # This is what stops EV from netting restricted cash, and stops short-term investments
    # being subtracted TWICE (once inside the cash total, once as their own EV term).
    # The CURRENT + NONCURRENT split is preferred over the single combined element: filers
    # keep the balance-sheet split current while the combined tag is often reported once and
    # then goes stale (TKO tags `RestrictedCash` four times ending 2024, but
    # `RestrictedCashCurrent` every quarter -- and its restricted cash is 54% of the total).
    _restr_split = (col("restrictedCashCurrent").fillna(0) + col("restrictedCashNoncurrent").fillna(0))
    _restr_split = _restr_split.where(col("restrictedCashCurrent").notna()
                                      | col("restrictedCashNoncurrent").notna())
    restricted = _restr_split.where(_restr_split.notna(), col("restrictedCash"))
    # Derive clean cash from the broad total ONLY when the restricted amount is known for
    # that period, or when the filer discloses no restricted cash ANYWHERE in its history
    # (then zero is safe). Netting an UNKNOWN restricted balance would silently reintroduce
    # exactly the overstatement this change removes, so we leave cash missing instead --
    # NaN is handled downstream, a wrong number is not.
    _no_restricted_ever = restricted.notna().sum() == 0
    _known = restricted.notna() | _no_restricted_ever
    cash = cash.where(cash.notna(),
                      (col("cashInclRestricted") - restricted.fillna(0.0)).where(_known))
    _sti = col("shortTermInvestments")
    cash = cash.where(cash.notna(),
                      (col("cashAndShortTermInvestments") - _sti.fillna(0.0)).where(_sti.notna()))
    cash = cash.clip(lower=0.0)             # a mis-tagged restricted amount can't make cash negative

    # ---- ASC-842 lease liabilities: BOTH legs, all three eras --------------------- #
    # total = the combined element when tagged, else current + noncurrent, else (pre-2019)
    # the capital-lease legs. The eras never overlap, so nothing is double counted.
    def _total_or_split(total_key: str, current_key: str, noncurrent_key: str) -> pd.Series:
        tot, cur, nc = col(total_key), col(current_key), col(noncurrent_key)
        split = (cur.fillna(0) + nc.fillna(0)).where(cur.notna() | nc.notna())
        return tot.where(tot.notna(), split)

    op_lease_liab = _total_or_split("operatingLeaseLiability", "operatingLeaseLiabilityCurrent",
                                   "operatingLeaseLiabilityNoncurrent")
    fin_lease_liab = _total_or_split("financeLeaseLiability", "financeLeaseLiabilityCurrent",
                                     "financeLeaseLiabilityNoncurrent")
    _cap_lease = (col("capitalLeaseObligationCurrent").fillna(0)
                  + col("capitalLeaseObligationNoncurrent").fillna(0))
    _cap_lease = _cap_lease.where(col("capitalLeaseObligationCurrent").notna()
                                 | col("capitalLeaseObligationNoncurrent").notna())
    fin_lease_liab = fin_lease_liab.where(fin_lease_liab.notna(), _cap_lease)

    # ---- the ASC-842 TOTAL-ASSETS break ------------------------------------------- #
    # Adopting ASC 842 in FY2019 put the operating-lease ROU asset ON the balance sheet, so
    # `totalAssets` jumps once for every lease-heavy filer with no change in the business.
    # Every assets-denominated ratio inherited that jump: asset growth (the Fama-French CMA
    # investment factor), asset turnover, gross profitability, accruals, Altman Z. The
    # break-free base removes the ROU asset in the post-adoption era (it is simply absent
    # before), and `operatingLeaseRouAsset` stays available as the lease-intensity signal.
    rou = col("operatingLeaseRouAsset")
    assets_ex_lease = assets - rou.fillna(0.0)

    # ---- LIFO -> FIFO normalization ----------------------------------------------- #
    # FIFO inventory = LIFO inventory + reserve; FIFO COGS = LIFO COGS - the reserve's
    # INCREASE over the year (a rising reserve means LIFO charged more than FIFO would).
    # Restated in place so a LIFO filer's inventory days / GMROI / gross margin sit on the
    # same basis as its FIFO peers; `lifoReserve` keeps the size of the adjustment.
    lifo = col("lifoReserve")
    inventory = col("inventory") + lifo.fillna(0.0)
    # a filer that discloses FIFO inventory directly needs no reconstruction
    inventory = inventory.where(inventory.notna(), col("inventoryFifoReported"))
    cor_ttm = cor_ttm - (lifo - lifo.shift(TTM_QUARTERS)).fillna(0.0)

    # ---- asset-retirement obligation total ---------------------------------------- #
    aro = col("assetRetirementObligation")
    _aro_split = (col("aroCurrent").fillna(0) + col("aroNoncurrent").fillna(0))
    _aro_split = _aro_split.where(col("aroCurrent").notna() | col("aroNoncurrent").notna())
    aro = aro.where(aro.notna(), _aro_split)

    # Derive total liabilities when a filer doesn't tag `Liabilities` as a single
    # element (e.g. LLY, AMD report only Assets + Equity + LiabilitiesAnd-
    # StockholdersEquity, leaving `Liabilities` absent). Accounting identity:
    # Liabilities = Assets - Equity. Only where both sides exist, so filers that
    # report none of them stay NaN.
    liab = liab.where(liab.notna(), assets - eq)

    # Total interest-bearing debt (universal): the single combined ST+LT tag when a
    # filer reports it (many banks / insurers), else long-term + short-term, else
    # notes payable (REITs). Its OWN pool, so short- and long-term are never merged
    # into longTermDebt.
    _debt_combined = col("debtCombined")
    _lt_st = (ltd.fillna(0) + std_debt.fillna(0)).where(ltd.notna() | std_debt.notna())
    total_debt = _debt_combined.where(_debt_combined.notna(), _lt_st)
    total_debt = total_debt.where(total_debt.notna(), col("notesPayable"))
    # ZERO debt vs MISSING debt — dissociate the two (they were both NaN before). When a filer
    # reports a balance sheet for a period (total assets or equity present) but tags NO interest-
    # bearing debt, debt is 0 on that date, not unknown. Applied PER PERIOD (so a name that is
    # debt-free only part of its history is 0 there, not NaN) and to BOTH legs + the total so the
    # stored columns stay consistent. EXCLUDED for Financials UNLESS debt-free across ALL history:
    # banks / insurers fund via deposits / FHLB advances / repos tagged outside our debt concepts,
    # so a missing debt tag there is not reliably zero and stays NaN.
    bs_present = assets.notna() | eq.notna()
    may_zero = (sector != "Financials") or (total_debt.notna().sum() == 0)
    if may_zero:
        total_debt = total_debt.where(~(bs_present & total_debt.isna()), 0.0)
        ltd = ltd.where(~(bs_present & ltd.isna()), 0.0)
        std_debt = std_debt.where(~(bs_present & std_debt.isna()), 0.0)

    gross_profit_ttm = gp_ttm.where(gp_ttm.notna(), rev_ttm - cor_ttm)
    # Gross margin, with an extraction-artifact guard: a value > 1 (negative COGS) or
    # below -200% only arises from a revenue/cost period-or-scope mismatch (truncated
    # revenue), never as a real gross margin, so it is nulled rather than shipped.
    gross_margin = (gross_profit_ttm / rev_ttm).where(rev_ttm > 0)
    gross_margin = gross_margin.where(
        (gross_margin >= GROSS_MARGIN_MIN) & (gross_margin <= GROSS_MARGIN_MAX))
    # Net margin, with a FINANCIALS-ONLY artifact guard: a bank/insurer/asset-manager net
    # income exceeding ~1.5x net revenue (or a loss beyond -200%) reflects consolidated-
    # fund NCI / one-time attribution (e.g. ARES pre-IPO), not a real margin. The bound
    # holds structurally in-sector, so it is nulled there; other sectors (biotech losses,
    # one-time gains) keep their genuine extreme margins.
    profit_margin = (ni_ttm / rev_ttm).where(rev_ttm > 0)
    if sector == "Financials":
        profit_margin = profit_margin.where((profit_margin >= -2.0) & (profit_margin <= 1.5))
    # Derive operating income when the filer doesn't tag OperatingIncomeLoss
    # (e.g. LLY, CVX, JNJ post-2015): operating income ≈ gross profit − SG&A − R&D.
    # Only where the components exist, so banks (no gross profit) stay NaN.
    oi_derived = gross_profit_ttm - sga_ttm.fillna(0) - rnd_ttm.fillna(0)
    oi_ttm = oi_ttm.where(oi_ttm.notna(), oi_derived)
    # REITs / integrated oil & gas tag no operating-income line and have no gross profit
    # (e.g. O, EQR, DVN, XOM), yet EBITDAre / EBITDAX need operating income -> bottom-up
    # EBIT = pre-tax income + interest expense. Gated to non-financials (banks / insurers
    # have their own operating-income proxies), so their operating margin stays N/A.
    if sector != "Financials":
        # some non-financials report "Total costs and expenses" (incl. DD&A) but no
        # operating-income line (integrated oil pre-restructuring, e.g. COP 2012-2016):
        # operating income = revenue - total operating costs. Preferred over the EBIT
        # proxy below because it excludes non-operating items. Guard: reject an
        # implausibly high (>60%) implied operating margin, which signals INCOMPLETELY
        # tagged costs (e.g. COP's pre-2012 integrated years) rather than real OI.
        oi_cae = rev_ttm - ttm_a("costsAndExpenses")
        oi_cae = oi_cae.where(oi_cae <= 0.60 * rev_ttm)
        oi_ttm = oi_ttm.where(oi_ttm.notna(), oi_cae)
        # last resort: bottom-up EBIT = pre-tax income + interest expense.
        oi_ttm = oi_ttm.where(oi_ttm.notna(), ttm_a("pretaxIncome") + int_ttm.fillna(0))

    # ---- ASU 2017-07: NON-SERVICE pension cost out of operating income ------------- #
    # Only SERVICE cost is compensation; interest cost, expected return on plan assets and
    # the amortizations are financing/actuarial items. From FY2018 the standard forces them
    # OUT of the operating subtotal — but the pre-2018 half of the history has them INSIDE
    # it, so a filer's own operating-margin series breaks at adoption (and cross-sectionally,
    # a big-pension industrial looks structurally less profitable than a peer with none).
    # Restated by ADDING the non-service cost back to pre-adoption operating income; from
    # FY2018 on there is nothing to add back, so the series is continuous either side.
    # Non-service = net periodic cost - service cost (the most robust form: one subtraction
    # of two well-tagged totals), falling back to the component build-up.
    non_service_pension = ttm_a("pensionNetPeriodicCost") - ttm_a("pensionServiceCost")
    _components = (ttm_a("pensionInterestCost").fillna(0)
                   - ttm_a("pensionExpectedReturn").fillna(0)
                   + ttm_a("pensionAmortPriorService").fillna(0)
                   + ttm_a("pensionAmortGainsLosses").fillna(0))
    _has_components = (ttm_a("pensionInterestCost").notna() | ttm_a("pensionExpectedReturn").notna())
    non_service_pension = non_service_pension.where(non_service_pension.notna(),
                                                    _components.where(_has_components))
    _pre_asu = base["end"] < pd.Timestamp(ASU_2017_07_EFFECTIVE)
    oi_ttm = oi_ttm + non_service_pension.fillna(0.0).where(_pre_asu, 0.0)

    # EBITDA = operating income + D&A, with a bottom-up fallback (net income + taxes
    # + interest + D&A) for filers that report no operating-income line, e.g.
    # integrated oil (XOM). All four inputs exist even when OI/gross profit don't.
    ebitda = oi_ttm + da_ttm.fillna(0)
    ebitda = ebitda.where(ebitda.notna(),
                          ni_ttm + ttm_a("incomeTaxExpense").fillna(0)
                          + int_ttm.fillna(0) + da_ttm.fillna(0))

    out = pd.DataFrame({
        "ticker": ticker,
        "as_of": base["as_of"].dt.date.astype(str),
        "fiscal_end": base["end"].dt.date.astype(str),
        "totalRevenue": rev_ttm,
        "netIncome": ni_ttm,
        "grossMargins": gross_margin,
        "operatingMargins": (oi_ttm / rev_ttm).where(rev_ttm > 0),
        "profitMargins": profit_margin,
        # ROE is defined whenever equity is non-zero; negative-equity firms (heavy
        # buybacks, e.g. VRSN/WYNN) get a (negative) ROE rather than being dropped.
        "returnOnEquity": (ni_ttm / eq).where(eq != 0),
        "debtToEquity": (ltd.where(ltd.notna(), liab) / eq).where(eq > 0),
        "ebitda": ebitda,
        # operatingIncome / depAmort / capex are emitted as raw TTM levels (not just
        # folded into margins / FCF) because the sector KPIs consume them directly:
        # EBITDAre = operatingIncome + depAmort (REIT), EBITDAX = + explorationExpense
        # (oil & gas), FFO uses depAmort, AFFO = FFO - capex, bank operating-income proxy.
        "operatingIncome": oi_ttm,
        "depAmort": da_ttm,
        "capex": capex_ttm,
        # GLOBAL capacity investment = cash capex + capacity funded via FINANCE leases (non-cash).
        # Captures how much a firm actually injects into capacity each period even when it leases
        # rather than buys (data centers -> MSFT ~+$6-9B/q, historically AMZN). NaN where cash capex
        # is unknown; = capex for non-lease filers (finance-lease term is 0-filled). NOTE `capex`
        # (and thus FCF) is unchanged -- capexGlobal is an ADDITIVE capacity measure, not cash-flow.
        # Now includes OPERATING-lease additions too (87.3% coverage). Previously only the
        # finance-lease leg was added, which made the measure asymmetric: a retailer or
        # airline that adds capacity through operating leases showed none of it, while a
        # hyperscaler leasing data centres under finance leases showed all of it.
        "capexGlobal": (capex_ttm + ttm_a("financeLeaseAdditions", charge=True)
                        + ttm_a("operatingLeaseAdditions", charge=True)),
        "freeCashflow": ocf_ttm - capex_ttm.fillna(0),
        "operatingCashFlow": ocf_ttm,
        "researchAndDevelopment": rnd_ttm,
        "stockholdersEquity": eq,
        "sharesOutstanding": col("sharesOutstanding"),
        # ---- raw levels / TTM flows for the refined features ----
        # (distress: cash + debt + current items + interest; S&M: SG&A;
        #  M&A: acquisitions + goodwill + assets; SBC: stockBasedComp)
        "cash": cash,
        "longTermDebt": ltd,
        "shortTermDebt": std_debt,
        "totalDebt": total_debt,
        "totalLiabilities": liab,
        "currentAssets": cur_a,
        "currentLiabilities": cur_l,
        "goodwill": goodwill,
        "totalAssets": assets,
        "sellingGeneralAdmin": sga_ttm,
        "stockBasedComp": sbc_ttm,
        "acquisitions": acq_ttm,
        "interestExpense": int_ttm,
        # discrete single-quarter values -> "latest quarter" momentum features
        "revenue_q": rev_q,
        "netIncome_q": ni_q,
        "ebitda_q": ebitda_q,
        "freeCashflow_q": fcf_q,
    })

    # Year-over-year growth on the TTM series (4 quarters back), so it is a true
    # annual comparison free of seasonality even at quarterly cadence.
    out["revenueGrowth"] = rev_ttm.pct_change(TTM_QUARTERS)
    out["earningsGrowth"] = ni_ttm.pct_change(TTM_QUARTERS)

    # ---- expanded raw coverage (universal §B + sector line items) ----
    # flows -> TTM (seasonality-free); balance-sheet items -> point-in-time level.
    # Assemble all extra columns at once (avoids DataFrame fragmentation).
    extra = {"costOfRevenue": cor_ttm, "grossProfit": gross_profit_ttm,
             "optionOverhang": col("optionOverhang"),
             **{k: col(k) for k in LATEST_DURATION_TAGS}}
    for key in EXTRA_FLOW_TAGS:
        extra[key] = ttm_a(key, charge=(key in CHARGE_FLOWS))
    for key in EXTRA_STOCK_TAGS:
        extra[key] = col(key)

    # --- clean reconstructions: fill each target level by DERIVING from its own
    #     components, never by coalescing a different pool (which would mix signals) ---
    _accum = extra["accumulatedDepreciation"]
    # gross PP&E for net-only filers:  gross = net + accumulated depreciation
    extra["ppeGross"] = extra["ppeGross"].where(extra["ppeGross"].notna(),
                                                extra["ppeNet"] + _accum)
    # ... and the SYMMETRIC repair, for gross-only filers and for the utilities whose
    # `PropertyPlantAndEquipmentNet` is only a minor non-utility COMPONENT. AEP tags its
    # rate base as `PublicUtilitiesPropertyPlantAndEquipment{Transmission,Distribution,
    # GenerationOrProcessing}` and leaves `PropertyPlantAndEquipmentNet` at $0.71bn against
    # $114bn of total assets and $120bn of gross PP&E -- a 99% understatement of the asset
    # base that feeds asset turnover, capex intensity and Altman Z. Rebuild net from the
    # roll-forward whenever it is missing, or implausibly small next to (gross - accum).
    _ppe_derived = extra["ppeGross"] - _accum
    _net_too_small = (extra["ppeNet"].notna() & _ppe_derived.notna() & (_ppe_derived > 0)
                      & (extra["ppeNet"] < _ppe_derived * PPE_NET_MIN_SHARE_OF_ROLLFORWARD))
    extra["ppeNet"] = extra["ppeNet"].where(extra["ppeNet"].notna() & ~_net_too_small,
                                            _ppe_derived)
    # When accumulated depreciation is unavailable (AEP stops tagging it after 2025, so the
    # 4-quarter ffill runs out) the roll-forward cannot be rebuilt -- but the component value
    # is still known-wrong. NULL it rather than ship a 99%-understated asset base: a missing
    # PP&E is handled by every downstream ratio, a wrong one silently is not.
    _net_vs_gross_only = (extra["ppeNet"].notna() & extra["ppeGross"].notna()
                          & (extra["ppeGross"] > 0)
                          & (extra["ppeNet"]
                             < extra["ppeGross"] * PPE_NET_MIN_SHARE_OF_ROLLFORWARD))
    extra["ppeNet"] = extra["ppeNet"].where(~_net_vs_gross_only)
    # net oil&gas property for E&P filers that tag only gross:  net = gross - accumulated
    extra["oilGasPropertyNet"] = extra["oilGasPropertyNet"].where(
        extra["oilGasPropertyNet"].notna(), extra["oilGasPropertyGross"] - _accum)
    # total deferred revenue: combined tag when present, else current + noncurrent
    # (no double-count: the split is used only where the combined tag is absent)
    _dr_split = extra["deferredRevenueCurrent"].fillna(0) + extra["deferredRevenueNoncurrent"].fillna(0)
    _dr_split = _dr_split.where(extra["deferredRevenueCurrent"].notna()
                               | extra["deferredRevenueNoncurrent"].notna())
    extra["deferredRevenue"] = extra["deferredRevenue"].where(extra["deferredRevenue"].notna(), _dr_split)
    # regulatory assets/liabilities total: combined tag when present, else current +
    # noncurrent (utilities split them) -> the full pool the utility KPIs need.
    for _tot, _cur, _nc in (("regulatoryAssets", "regulatoryAssetsCurrent", "regulatoryAssetsNoncurrent"),
                            ("regulatoryLiabilities", "regulatoryLiabilitiesCurrent", "regulatoryLiabilitiesNoncurrent")):
        _sp = extra[_cur].fillna(0) + extra[_nc].fillna(0)
        _sp = _sp.where(extra[_cur].notna() | extra[_nc].notna())
        extra[_tot] = extra[_tot].where(extra[_tot].notna(), _sp)

    # --- RESTATED / RECONSTRUCTED levels win over the raw single-tag reads above ---
    # (the `for key in EXTRA_STOCK_TAGS` loop assigned the raw tag value; these are the
    #  corrected versions built earlier in this function)
    extra["restrictedCash"] = restricted
    extra["operatingLeaseLiability"] = op_lease_liab      # combined | current+noncurrent
    extra["financeLeaseLiability"] = fin_lease_liab       # + pre-2019 capital-lease legs
    extra["inventory"] = inventory                        # FIFO-normalized (+ LIFO reserve)
    extra["assetRetirementObligation"] = aro              # combined | current+noncurrent
    # break-free asset base + the two lease/pension adjustment SIZES as their own signals
    extra["totalAssetsExLease"] = assets_ex_lease
    extra["nonServicePensionCost"] = non_service_pension
    extra["exciseTaxAdjustment"] = _excise_adj
    # gross finite-lived intangibles for filers that tag only net + accumulated amortization
    extra["intangiblesGross"] = extra["intangiblesGross"].where(
        extra["intangiblesGross"].notna(),
        extra["intangiblesExGoodwill"] + extra["intangiblesAccumAmort"])
    # total principal coming due within five years (the refinancing WALL)
    extra["debtMaturity5yTotal"] = sum(
        (extra[f"debtMaturity{y}y"].fillna(0.0) for y in range(1, 6)),
        start=pd.Series(0.0, index=out.index),
    ).where(pd.concat([extra[f"debtMaturity{y}y"].notna() for y in range(1, 6)], axis=1).any(axis=1))

    out = pd.concat([out, pd.DataFrame(extra, index=out.index)], axis=1)

    # sector tags so downstream KPIs can be computed / neutralized sector-relatively
    out["sector"] = sector
    out["industry_group"] = industry_group

    out = apply_plausibility_guards(out)
    return out.copy().reset_index(drop=True)
