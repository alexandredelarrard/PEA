"""
Fetch fundamental data per ticker.

THREE sources, in order of how much history they give you:

1. SEC EDGAR `companyfacts` (PRIMARY, free, no key) -> genuine ~10-year
   point-in-time history. This is the one that lets you make size / value /
   quality risk-neutral over a real backtest, because every value comes with
   the FILING DATE (`filed`), so we key each row on the date the number
   actually became public -- no look-ahead. Built by
   `build_fundamentals_history_sec()`.

2. SimFin bulk CSV (free tier, ~10y) -> alternative if you prefer a single
   download. `load_simfin_bulk()`.

3. yfinance `.info` snapshot (free, current only) -> used to enrich the LATEST
   row with fields SEC doesn't carry cleanly (sector, industry, current
   marketCap). `fetch_snapshot()`.

Output schema (same `fundamentals_history.parquet` the cube reads):
    ticker, as_of (= filing date), fiscal_end,
    totalRevenue, netIncome, grossMargins, operatingMargins, profitMargins,
    returnOnEquity, debtToEquity, ebitda, freeCashflow,
    revenueGrowth, earningsGrowth, sharesOutstanding, stockholdersEquity

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

from src.context import Context
from src.data_extract.sec_utils import sec_get, load_cik_mapping


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
}
STOCK_TAGS = {  # balance-sheet items (instant facts, point-in-time)
    "stockholdersEquity": ["StockholdersEquity",
                           "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "totalLiabilities": ["Liabilities"],
    "longTermDebt": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsAndShortTermInvestments"],
}
SHARES_TAGS = {  # tried under dei first, then us-gaap
    "sharesOutstanding": ["EntityCommonStockSharesOutstanding",
                          "CommonStockSharesOutstanding", "WeightedAverageNumberOfDilutedSharesOutstanding"],
}

ANNUAL_MIN_DAYS, ANNUAL_MAX_DAYS = 340, 380   # accept a fiscal year as ~365d


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
    path = context.paths["FUNDAMENTALS_HISTORY_PATH"]
    if not path.exists():
        return None
    return pd.read_parquet(path)


def _is_sec_history_up_to_date(context: Context, cik_mapping: pd.DataFrame) -> bool:
    """True when today's run already attempted the full CIK universe."""
    meta = _load_history_meta(context)
    if meta is None or meta.get("last_built") != _today_iso():
        return False
    if not context.paths["FUNDAMENTALS_HISTORY_PATH"].exists():
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
        resp = sec_get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json")
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


def _instant_stock(df: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time balance items: first disclosure per period end."""
    if df.empty:
        return df
    d = df.dropna(subset=["end", "filed", "val"]).copy()
    d = d.sort_values("filed").drop_duplicates(subset=["end"], keep="first")
    return d[["end", "filed", "val"]].sort_values("end").reset_index(drop=True)


def _series_on_ends(concept_df: pd.DataFrame, name: str) -> pd.DataFrame:
    return concept_df.rename(columns={"val": name, "filed": f"{name}_filed"})


# --------------------------------------------------------------------------- #
# Build one ticker's annual history from its companyfacts                     #
# --------------------------------------------------------------------------- #
def build_ticker_history(ticker: str, facts: dict) -> pd.DataFrame:
    gaap = facts.get("facts", {}).get("us-gaap", {})
    dei = facts.get("facts", {}).get("dei", {})

    flows = {k: _annual_flow(_extract_concept(gaap, tags)) for k, tags in FLOW_TAGS.items()}
    stocks = {k: _instant_stock(_extract_concept(gaap, tags)) for k, tags in STOCK_TAGS.items()}
    shares = _instant_stock(_extract_concept(dei, SHARES_TAGS["sharesOutstanding"])
                            if any(t in dei for t in SHARES_TAGS["sharesOutstanding"])
                            else _extract_concept(gaap, SHARES_TAGS["sharesOutstanding"]))

    rev = flows.get("totalRevenue")
    if rev is None or rev.empty:
        return pd.DataFrame()

    # Base frame on fiscal-year ends from revenue; merge others by nearest end.
    base = _series_on_ends(rev, "totalRevenue")

    def merge_flow(base, key):
        s = flows.get(key)
        if s is None or s.empty:
            base[key] = pd.NA
            base[f"{key}_filed"] = pd.NaT
            return base
        return base.merge(_series_on_ends(s, key), on="end", how="left")

    for key in ["netIncome", "grossProfit", "costOfRevenue", "operatingIncome",
                "depAmort", "operatingCashFlow", "capex"]:
        base = merge_flow(base, key)

    def merge_stock(base, key, src):
        if src is None or src.empty:
            base[key] = pd.NA
            base[f"{key}_filed"] = pd.NaT
            return base
        return base.merge(_series_on_ends(src, key), on="end", how="left")

    for key in ["stockholdersEquity", "totalLiabilities", "longTermDebt", "cash"]:
        base = merge_stock(base, key, stocks.get(key))
    base = merge_stock(base, "sharesOutstanding", shares)

    base = base.sort_values("end").reset_index(drop=True)

    # as_of = latest filing date among the merged concepts (ensures all public).
    filed_cols = [c for c in base.columns if c.endswith("_filed")]
    base["as_of"] = base[filed_cols].max(axis=1)
    base = base.dropna(subset=["as_of"])

    # ---- derived fields ----
    def col(name):
        return pd.to_numeric(base.get(name), errors="coerce")

    rev_v = col("totalRevenue")
    ni = col("netIncome")
    gp = col("grossProfit")
    cor = col("costOfRevenue")
    oi = col("operatingIncome")
    da = col("depAmort")
    ocf = col("operatingCashFlow")
    capex = col("capex")
    eq = col("stockholdersEquity")
    liab = col("totalLiabilities")
    ltd = col("longTermDebt")

    gross_profit = gp.where(gp.notna(), rev_v - cor)
    out = pd.DataFrame({
        "ticker": ticker,
        "as_of": base["as_of"].dt.date.astype(str),
        "fiscal_end": base["end"].dt.date.astype(str),
        "totalRevenue": rev_v,
        "netIncome": ni,
        "grossMargins": (gross_profit / rev_v).where(rev_v > 0),
        "operatingMargins": (oi / rev_v).where(rev_v > 0),
        "profitMargins": (ni / rev_v).where(rev_v > 0),
        "returnOnEquity": (ni / eq).where(eq > 0),
        "debtToEquity": (ltd.where(ltd.notna(), liab) / eq).where(eq > 0),
        "ebitda": oi + da.fillna(0),
        "freeCashflow": ocf - capex.fillna(0),
        "stockholdersEquity": eq,
        "sharesOutstanding": col("sharesOutstanding"),
    })

    # YoY growth on the fiscal series (chronological).
    out["revenueGrowth"] = rev_v.pct_change()
    out["earningsGrowth"] = ni.pct_change()

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
    if cik_mapping is None:
        cik_mapping = load_cik_mapping(context)

    if _is_sec_history_up_to_date(context, cik_mapping):
        path = context.paths["FUNDAMENTALS_HISTORY_PATH"]
        hist = pd.read_parquet(path)
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

    path = context.paths["FUNDAMENTALS_HISTORY_PATH"]
    out.to_parquet(path, index=False)
    _save_history_meta(context, out, len(cik_mapping))
    print(f"Saved {len(out)} SEC fundamental rows "
          f"({out['ticker'].nunique()} tickers) to {path}")
    return out


# --------------------------------------------------------------------------- #
# yfinance current snapshot (enrichment only: sector/industry/current mktcap)  #
# --------------------------------------------------------------------------- #
SNAPSHOT_FIELDS = ["marketCap", "trailingPE", "forwardPE", "sector", "industry", "shortName"]


def fetch_snapshot(context: Context, tickers: list[str], pause: float = 0.3) -> pd.DataFrame:
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
    snap_path = context.paths["FUNDAMENTALS_SNAPSHOT_PATH"]
    snapshot.to_parquet(snap_path, index=False)
    print(f"Saved current snapshot for {len(snapshot)} tickers to {snap_path}")
    return history
