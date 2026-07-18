"""
Fetch fundamental data per ticker.

THREE sources, in order of how much history they give you:

1. SEC EDGAR `companyfacts` (PRIMARY, free, no key) -> genuine ~10-year
   point-in-time history at QUARTERLY cadence. This is the one that lets you
   make size / value / quality risk-neutral over a real backtest, because every
   value comes with the FILING DATE (`filed`), so we key each row on the date
   the number actually became public -- no look-ahead. Flow items are stored as
   trailing-twelve-month (TTM) sums of discrete quarters (Q1-Q3 from the 10-Qs,
   Q4 derived as FY - Q1 - Q2 - Q3), refreshed every quarter. Built by
   `build_fundamentals_history_sec()`.

2. SimFin bulk CSV (free tier, ~10y) -> alternative if you prefer a single
   download. `load_simfin_bulk()`.

3. yfinance `.info` snapshot (free, current only) -> used to enrich the LATEST
   row with fields SEC doesn't carry cleanly (sector, industry, current
   marketCap). `fetch_snapshot()`.

Output schema (same `fundamentals_history.parquet` the cube reads):
    ticker, as_of (= filing date), fiscal_end,
    totalRevenue, netIncome, grossMargins, operatingMargins, profitMargins,
    returnOnEquity, debtToEquity, ebitda, freeCashflow, operatingCashFlow,
    researchAndDevelopment, revenueGrowth, earningsGrowth, sharesOutstanding,
    stockholdersEquity,
    cash, longTermDebt, shortTermDebt, totalLiabilities, currentAssets,
    currentLiabilities, goodwill, totalAssets           (raw balance-sheet levels),
    sellingGeneralAdmin, stockBasedComp, acquisitions, interestExpense  (TTM flows),
    revenue_q, netIncome_q, ebitda_q, freeCashflow_q  (discrete single-quarter)

The raw levels / extra TTM flows above feed the refined feature families in
fundamental_features.py: distress (net-debt/EBITDA, interest coverage, current
ratio, cash/debt), S&M efficiency (SG&A intensity + operating leverage), M&A
(acquisition intensity, goodwill growth) and stock-based-comp (SBC intensity,
SBC/OCF). All are TTM/point-in-time, keyed on the SEC filing date.

IMPORTANT for size / value: SEC gives shares and equity, not market cap
(market cap needs price). We store `sharesOutstanding`; compute
marketCap = sharesOutstanding * close in the factor layer (see the companion
change to factors.py). Store equity so book/price is available too.

Run:
    python -m data.fetch_fundamentals
"""
import json
import time
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf
from tqdm import tqdm

from src.constants.constants import SEC_COMPANYFACTS_URL
from src.context import Context
from src.data_extract.utils.common.sec_utils import sec_get, load_cik_mapping


# --------------------------------------------------------------------------- #
# SEC XBRL concept tags. Each logical field maps to a list of candidate us-gaap
# (or dei) tags; we take the first one a filer actually uses. Companies tag the
# same economic concept differently, so candidates matter.
# --------------------------------------------------------------------------- #
FLOW_TAGS = {   # income-statement / cash-flow items (duration facts, annual)
    "totalRevenue": ["RevenueFromContractWithCustomerExcludingAssessedTax",
                     "Revenues", "SalesRevenueNet"],
    "netIncome": ["NetIncomeLoss", "ProfitLoss"],
    "grossProfit": ["GrossProfit"],
    "costOfRevenue": ["CostOfGoodsAndServicesSold", "CostOfRevenue"],
    "operatingIncome": ["OperatingIncomeLoss"],
    "depAmort": ["DepreciationDepletionAndAmortization",
                 "DepreciationAmortizationAndAccretionNet",
                 "DepreciationAndAmortization"],
    "operatingCashFlow": ["NetCashProvidedByUsedInOperatingActivities",
                          "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment",
              "PaymentsToAcquireProductiveAssets"],
    "researchAndDevelopment": ["ResearchAndDevelopmentExpense",
                               "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost"],
    # ---- added for refined features (S&M efficiency, M&A, SBC, distress) ----
    "sellingGeneralAdmin": ["SellingGeneralAndAdministrativeExpense",
                            "GeneralAndAdministrativeExpense",
                            "SellingAndMarketingExpense"],
    "stockBasedComp": ["ShareBasedCompensation",
                       "AllocatedShareBasedCompensationExpense"],
    "acquisitions": ["PaymentsToAcquireBusinessesNetOfCashAcquired",
                     "PaymentsToAcquireBusinessesAndInterestInAffiliates"],
    "interestExpense": ["InterestExpense", "InterestAndDebtExpense",
                        "InterestExpenseNonoperating"],
}
STOCK_TAGS = {  # balance-sheet items (instant facts, point-in-time)
    "stockholdersEquity": ["StockholdersEquity",
                           "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "totalLiabilities": ["Liabilities"],
    "longTermDebt": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsAndShortTermInvestments"],
    # ---- added for refined features (distress / liquidity, M&A footprint) ----
    "shortTermDebt": ["DebtCurrent", "LongTermDebtCurrent", "ShortTermBorrowings"],
    "currentAssets": ["AssetsCurrent"],
    "currentLiabilities": ["LiabilitiesCurrent"],
    "goodwill": ["Goodwill"],
    "totalAssets": ["Assets"],
}
SHARES_TAGS = {  # tried under dei first, then us-gaap
    "sharesOutstanding": ["EntityCommonStockSharesOutstanding",
                          "CommonStockSharesOutstanding", "WeightedAverageNumberOfDilutedSharesOutstanding"],
}

ANNUAL_MIN_DAYS, ANNUAL_MAX_DAYS = 340, 380   # accept a fiscal year as ~365d
QUARTER_MIN_DAYS, QUARTER_MAX_DAYS = 80, 100   # accept a fiscal quarter as ~13 weeks
TTM_QUARTERS = 4                               # trailing-twelve-months = 4 quarters


def _today_iso() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _history_meta_path(context: Context):
    return context.paths["FUNDAMENTALS_HISTORY_PATH"].with_name(
        "fundamentals_history_meta.json",
    )


def _load_history_meta(context: Context) -> dict | None:
    path = _history_meta_path(context)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_history_meta(
    context: Context,
    history: pd.DataFrame,
    universe_size: int,
) -> None:
    meta = {
        "last_built": _today_iso(),
        "row_count": len(history),
        "ticker_count": int(history["ticker"].nunique()),
        "universe_size": universe_size,
    }
    _history_meta_path(context).write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )


def _load_existing_history(context: Context) -> pd.DataFrame | None:
    df = context.store.load("fundamentals_history")
    return None if df.empty else df


def _is_sec_history_up_to_date(context: Context, cik_mapping: pd.DataFrame) -> bool:
    """True when today's run already attempted the full CIK universe."""
    meta = _load_history_meta(context)
    if meta is None or meta.get("last_built") != _today_iso():
        return False
    if not context.store.exists("fundamentals_history"):
        return False
    return meta.get("universe_size", 0) >= len(cik_mapping)


def _tickers_to_process(
    context: Context,
    cik_mapping: pd.DataFrame,
    existing: pd.DataFrame | None,
) -> pd.DataFrame:
    """Return CIK rows that still need SEC history fetched."""
    if existing is None or existing.empty:
        return cik_mapping

    meta = _load_history_meta(context)
    if meta and meta.get("last_built") == _today_iso():
        have = set(existing["ticker"].unique())
        return cik_mapping[~cik_mapping["ticker"].isin(have)]

    # New calendar day (or no meta): refresh all tickers for new filings.
    return cik_mapping


# --------------------------------------------------------------------------- #
# SEC companyfacts fetch (cached)                                             #
# --------------------------------------------------------------------------- #
def _fetch_companyfacts(context: Context, cik: str) -> dict | None:
    cache_dir = context.paths["SEC_BULK_CACHE_DIR"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"companyfacts_CIK{cik}.json"

    if context.use_cache and cache.exists():
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except Exception:
            pass
    try:
        resp = sec_get(SEC_COMPANYFACTS_URL.format(cik=cik))
        data = resp.json()
        cache.write_text(json.dumps(data), encoding="utf-8")
        return data
    except Exception as e:
        print(f"companyfacts CIK{cik}: failed ({e})")
        return None


# --------------------------------------------------------------------------- #
# Concept extraction                                                          #
# --------------------------------------------------------------------------- #
def _extract_concept(section: dict, tag_candidates: list[str]) -> pd.DataFrame:
    """
    From one facts section (us-gaap or dei), return the observations for the
    first matching tag as a DataFrame [end, start, filed, form, fp, val].
    """
    for tag in tag_candidates:
        if tag not in section:
            continue
        units = section[tag].get("units", {})
        # prefer USD, then shares, then whatever exists
        unit_key = next((u for u in ("USD", "shares") if u in units),
                        next(iter(units), None))
        if unit_key is None:
            continue
        rows = []
        for obs in units[unit_key]:
            rows.append({
                "end": obs.get("end"), "start": obs.get("start"),
                "filed": obs.get("filed"), "form": obs.get("form"),
                "fp": obs.get("fp"), "val": obs.get("val"),
            })
        df = pd.DataFrame(rows)
        if df.empty:
            continue
        for c in ("end", "start", "filed"):
            df[c] = pd.to_datetime(df[c], errors="coerce")
        return df
    return pd.DataFrame(columns=["end", "start", "filed", "form", "fp", "val"])


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
    q = q[(q["implied"] >= 75) & (q["implied"] <= 100)]     # keep quarter-length only
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


def _series_on_ends(concept_df: pd.DataFrame, name: str) -> pd.DataFrame:
    return concept_df.rename(columns={"val": name, "filed": f"{name}_filed"})


def _merge_shares_asof(base: pd.DataFrame, shares: pd.DataFrame) -> pd.DataFrame:
    """Attach the most recent sharesOutstanding as of each fiscal-year `end`
    (backward as-of merge). Cover-page share counts are dated near the filing,
    not at fiscal-year end, so an exact `end` join misses almost everything."""
    if shares is None or shares.empty:
        base["sharesOutstanding"] = pd.NA
        base["sharesOutstanding_filed"] = pd.NaT
        return base
    s = shares.rename(columns={"val": "sharesOutstanding",
                               "filed": "sharesOutstanding_filed"})
    s = s[["end", "sharesOutstanding", "sharesOutstanding_filed"]].sort_values("end")
    left = base.sort_values("end")
    merged = pd.merge_asof(left, s, on="end", direction="backward")
    # restore original row order
    return merged.sort_values("end").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Build one ticker's QUARTERLY history (TTM levels) from its companyfacts      #
# --------------------------------------------------------------------------- #
def build_ticker_history(ticker: str, facts: dict) -> pd.DataFrame:
    """One row per FISCAL QUARTER, keyed on the filing date (`as_of`).

    Flow items (revenue, income, cash flows, R&D) are stored as
    TRAILING-TWELVE-MONTH (TTM) sums of the four most recent discrete quarters,
    which is the correct, seasonality-free level for valuation and margins and
    still refreshes every quarter. Balance-sheet items and shares are the
    point-in-time value at each quarter end. Growth is year-over-year (TTM vs 4
    quarters earlier). `as_of` is the latest filing date among the merged
    concepts, so a value is only visible once fully public (no look-ahead).
    """
    gaap = facts.get("facts", {}).get("us-gaap", {})
    dei = facts.get("facts", {}).get("dei", {})

    flows = {k: _quarterly_flow(_extract_concept(gaap, tags)) for k, tags in FLOW_TAGS.items()}
    stocks = {k: _instant_stock(_extract_concept(gaap, tags)) for k, tags in STOCK_TAGS.items()}
    shares = _instant_stock(_extract_concept(dei, SHARES_TAGS["sharesOutstanding"])
                            if any(t in dei for t in SHARES_TAGS["sharesOutstanding"])
                            else _extract_concept(gaap, SHARES_TAGS["sharesOutstanding"]))

    rev = flows.get("totalRevenue")
    if rev is None or rev.empty:
        return pd.DataFrame()

    # Base frame on the quarterly revenue ends; merge every other concept by end.
    base = _series_on_ends(rev, "totalRevenue")

    def merge_flow(base, key):
        s = flows.get(key)
        if s is None or s.empty:
            base[key] = pd.NA
            base[f"{key}_filed"] = pd.NaT
            return base
        return base.merge(_series_on_ends(s, key), on="end", how="left")

    for key in ["netIncome", "grossProfit", "costOfRevenue", "operatingIncome",
                "depAmort", "operatingCashFlow", "capex", "researchAndDevelopment",
                "sellingGeneralAdmin", "stockBasedComp", "acquisitions", "interestExpense"]:
        base = merge_flow(base, key)

    def merge_stock(base, key, src):
        if src is None or src.empty:
            base[key] = pd.NA
            base[f"{key}_filed"] = pd.NaT
            return base
        return base.merge(_series_on_ends(src, key), on="end", how="left")

    for key in ["stockholdersEquity", "totalLiabilities", "longTermDebt", "cash",
                "shortTermDebt", "currentAssets", "currentLiabilities",
                "goodwill", "totalAssets"]:
        base = merge_stock(base, key, stocks.get(key))

    # sharesOutstanding is a dei COVER-PAGE fact whose `end` is the cover date,
    # which almost never equals a period `end` -> an exact merge drops it for
    # ~95% of filers. Attach instead the most recent shares count as of each
    # quarter end (backward as-of merge), which keeps it point-in-time.
    base = _merge_shares_asof(base, shares)

    base = base.sort_values("end").reset_index(drop=True)

    # as_of = latest filing date among the merged concepts (ensures all public).
    filed_cols = [c for c in base.columns if c.endswith("_filed")]
    base["as_of"] = base[filed_cols].max(axis=1)
    base = base.dropna(subset=["as_of"]).sort_values("end").reset_index(drop=True)

    # ---- discrete quarterly numerics ----
    def col(name):
        return pd.to_numeric(base.get(name), errors="coerce")

    def ttm(s):
        # trailing 12 months = sum of the 4 most recent quarters
        return s.rolling(TTM_QUARTERS, min_periods=TTM_QUARTERS).sum()

    # discrete SINGLE-QUARTER values (before rolling) -> "latest quarter" features
    rev_q = col("totalRevenue")
    ni_q = col("netIncome")
    oi_q = col("operatingIncome")
    da_q = col("depAmort")
    ocf_q = col("operatingCashFlow")
    capex_q = col("capex")
    ebitda_q = oi_q + da_q.fillna(0)
    fcf_q = ocf_q - capex_q.fillna(0)

    rev_ttm = ttm(rev_q)
    ni_ttm = ttm(ni_q)
    gp_ttm = ttm(col("grossProfit"))
    cor_ttm = ttm(col("costOfRevenue"))
    oi_ttm = ttm(oi_q)
    da_ttm = ttm(da_q)
    ocf_ttm = ttm(ocf_q)
    capex_ttm = ttm(capex_q)
    rnd_ttm = ttm(col("researchAndDevelopment"))
    sga_ttm = ttm(col("sellingGeneralAdmin"))
    sbc_ttm = ttm(col("stockBasedComp"))
    acq_ttm = ttm(col("acquisitions"))
    int_ttm = ttm(col("interestExpense"))
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

    gross_profit_ttm = gp_ttm.where(gp_ttm.notna(), rev_ttm - cor_ttm)
    out = pd.DataFrame({
        "ticker": ticker,
        "as_of": base["as_of"].dt.date.astype(str),
        "fiscal_end": base["end"].dt.date.astype(str),
        "totalRevenue": rev_ttm,
        "netIncome": ni_ttm,
        "grossMargins": (gross_profit_ttm / rev_ttm).where(rev_ttm > 0),
        "operatingMargins": (oi_ttm / rev_ttm).where(rev_ttm > 0),
        "profitMargins": (ni_ttm / rev_ttm).where(rev_ttm > 0),
        "returnOnEquity": (ni_ttm / eq).where(eq > 0),
        "debtToEquity": (ltd.where(ltd.notna(), liab) / eq).where(eq > 0),
        "ebitda": oi_ttm + da_ttm.fillna(0),
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

    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Full universe historical build                                             #
# --------------------------------------------------------------------------- #
def build_fundamentals_history_sec(context: Context,
                                   cik_mapping: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Build the ~10-year point-in-time fundamentals history for the universe from
    SEC companyfacts and write it to FUNDAMENTALS_HISTORY_PATH.

    Incremental behaviour:
      - If history was already built today for the full universe, load and return.
      - If history exists from today but new tickers appeared, fetch only those.
      - On a new calendar day, refresh all tickers and merge with existing rows.
    """

    if _is_sec_history_up_to_date(context, cik_mapping):
        hist = context.store.load("fundamentals_history")
        print(
            f"SEC fundamentals history already up to date for {_today_iso()} "
            f"— skipping ({len(hist)} rows, {hist['ticker'].nunique()} tickers)"
        )
        return hist

    existing = _load_existing_history(context)
    to_process = _tickers_to_process(context, cik_mapping, existing)

    if to_process.empty and existing is not None and not existing.empty:
        _save_history_meta(context, existing, len(cik_mapping))
        print(
            f"SEC fundamentals history already up to date for {_today_iso()} "
            f"— skipping ({len(existing)} rows)"
        )
        return existing

    years = context.config.data_extract.years_history
    cutoff = pd.Timestamp.today() - pd.DateOffset(years=years)

    new_frames = []
    for _, r in tqdm(to_process.iterrows(), total=len(to_process),
                     desc="Building SEC fundamentals history"):
        cik, ticker = r["cik"], r["ticker"]
        facts = _fetch_companyfacts(context, cik)
        if not facts:
            continue
        hist = build_ticker_history(ticker, facts)
        if not hist.empty:
            new_frames.append(hist)

    parts = [df for df in (existing, *new_frames) if df is not None and not df.empty]
    if not parts:
        raise RuntimeError("No SEC fundamentals built — check CIK mapping / network.")

    out = pd.concat(parts, ignore_index=True)
    out["as_of_dt"] = pd.to_datetime(out["as_of"])
    out = out[out["as_of_dt"] >= cutoff].drop(columns=["as_of_dt"])
    out = out.drop_duplicates(subset=["ticker", "as_of"], keep="last")
    out = out.sort_values(["ticker", "as_of"]).reset_index(drop=True)

    # persist only the newly-built rows; the DB merges on (ticker, as_of)
    new = pd.concat(new_frames, ignore_index=True) if new_frames else pd.DataFrame()
    if not new.empty:
        context.store.save("fundamentals_history", new)
    _save_history_meta(context, out, len(cik_mapping))
    print(f"Saved {len(new)} new SEC fundamental rows "
          f"({out['ticker'].nunique()} tickers) to DB table 'fundamentals_history'")
    return out


# --------------------------------------------------------------------------- #
# yfinance current snapshot (enrichment only: sector/industry/current mktcap)  #
# --------------------------------------------------------------------------- #
SNAPSHOT_FIELDS = ["marketCap", "trailingPE", "forwardPE", "sector", "industry", "shortName"]


def _snapshot_up_to_date(context: Context, tickers: list[str]) -> pd.DataFrame | None:
    """Return the cached snapshot if it was already pulled today for the full
    requested universe, else None."""
    existing = context.store.load("fundamentals_latest")
    if existing.empty or "as_of" not in existing.columns:
        return None
    as_of = pd.to_datetime(existing["as_of"], errors="coerce").dt.strftime("%Y-%m-%d")
    if not (as_of == _today_iso()).all():
        return None
    if not set(tickers).issubset(set(existing["ticker"].unique())):
        return None
    return existing


def fetch_snapshot(context: Context, tickers: list[str], pause: float = 0.3) -> pd.DataFrame:
    cached = _snapshot_up_to_date(context, tickers)
    if cached is not None:
        print(f"Snapshot already pulled today for {len(cached)} tickers — skipping")
        return cached

    as_of = _today_iso()
    rows = []
    for tkr in tqdm(tickers, desc="Fetching current snapshot"):
        try:
            info = yf.Ticker(tkr).info
        except Exception as e:
            print(f"{tkr}: snapshot failed ({e})")
            continue
        row = {"ticker": tkr, "as_of": as_of}
        for f in SNAPSHOT_FIELDS:
            row[f] = info.get(f)
        rows.append(row)
        time.sleep(pause)
    return pd.DataFrame(rows)


def load_simfin_bulk(context: Context, csv_path: str) -> pd.DataFrame:
    """Alternative 10y source: SimFin free-tier bulk CSV. Adjust rename map to
    your export's header. Kept as a secondary option to the SEC builder."""
    df = pd.read_csv(csv_path, sep=";")
    rename_map = {
        "Ticker": "ticker", "Report Date": "as_of",
        "Revenue": "totalRevenue", "Net Income": "netIncome",
        "Shares (Diluted)": "sharesOutstanding",
        "Total Equity": "stockholdersEquity",
    }
    return df.rename(columns=rename_map)


# --------------------------------------------------------------------------- #
# Entry point (called by StepExtractAllData)                                  #
# --------------------------------------------------------------------------- #
def fetch_fundamentals(context: Context, tickers: list[str]):
    """
    Build the SEC 10-year history (primary) and a current snapshot (enrichment).
    SEC history is incremental: skips when already built today for the full
    universe; companyfacts JSONs are cached per CIK between runs.
    """
    cik_mapping = load_cik_mapping(context)
    history = build_fundamentals_history_sec(context, cik_mapping)

    # Latest-row enrichment with current market cap / sector for the screen.
    snapshot = fetch_snapshot(context, tickers)
    if not snapshot.empty:
        context.store.save("fundamentals_latest", snapshot)
    print(f"Saved current snapshot for {len(snapshot)} tickers to DB 'fundamentals_latest'")
    return history
