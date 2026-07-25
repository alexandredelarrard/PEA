"""
fetch_8k_items.py  (src/data_extract/utils/structure/fetch_8k_items.py)
-----------------------------------------------------------------------
8-K MATERIAL-EVENT ITEM CODES from EDGAR -> `sec_8k_items`. The item codes (e.g. "2.02,9.01") are a
STRUCTURED field of the submissions JSON (`filings.*.items`), so this needs NO document parsing and
is ~100% fill for post-2004 8-Ks — the highest-reliability SEC signal to add.

Per ticker: list the 8-K history via the shared `list_filings` (recent + archive pages, raw JSON
cached to data/sec_8k_cache/ BEFORE the DB), keep filings not already stored, and upsert per ticker.
Incremental & DAG-fast: the per-ticker `since = max(stored filing_date)` means a daily run fetches
only the recent submissions page (new 8-Ks), never re-walking the archives; a fresh table backfills
the full `years_history`. Deduped by accession so a re-run never double-counts.
"""
from __future__ import annotations

import logging

import pandas as pd
from tqdm import tqdm

from src.constants.constants import SEC_8K_CACHE_DIR, SEC_8K_FORMS, SEC_8K_ITEMS_TABLE
from src.context import Context
from src.data_extract.utils.common.edgar_fillings import list_filings
from src.data_extract.utils.common.sec_utils import load_cik_mapping

logger = logging.getLogger(__name__)
_TABLE = SEC_8K_ITEMS_TABLE
_COLS = ["ticker", "cik", "accession_number", "form", "filing_date",
         "period_of_report", "items", "n_items", "primary_document"]


def _existing(context: Context) -> tuple[set[str], dict[str, pd.Timestamp]]:
    """(accessions already stored, per-ticker max filing_date) — dedup + incremental cutoff."""
    try:
        df = context.store.load(_TABLE, columns=["ticker", "accession_number", "filing_date"])
    except Exception:                                   # table not created yet
        return set(), {}
    if df is None or df.empty:
        return set(), {}
    seen = set(df["accession_number"].dropna().astype(str))
    d = df.copy()
    d["filing_date"] = pd.to_datetime(d["filing_date"], errors="coerce")
    last = d.dropna(subset=["filing_date"]).groupby("ticker")["filing_date"].max().to_dict()
    return seen, last


def fetch_8k_items(context: Context, tickers: list[str], years: int | None = None) -> pd.DataFrame:
    """Build/refresh the 8-K item-code table, one ticker at a time (upsert per ticker so an
    interrupted run keeps its work). Returns the full table."""
    years = int(years if years is not None else context.config.data_extract.years_history)
    cache_dir = context.paths["DATA_STORE"] / SEC_8K_CACHE_DIR
    cik_map = load_cik_mapping(context)
    cik_map = cik_map[cik_map["ticker"].isin(tickers)]

    seen, last_by_ticker = _existing(context)
    total_new, touched, empty = 0, 0, 0
    for _, r in tqdm(cik_map.iterrows(), total=len(cik_map), desc="8-K items"):
        ticker, cik, company = r["ticker"], r["cik"], r.get("company_name", "")
        try:
            filings = list_filings(cik, SEC_8K_FORMS, years, company,
                                   since=last_by_ticker.get(ticker), cache_dir=cache_dir)
        except Exception as e:                          # noqa: BLE001
            context.log.warning("%s: 8-K filing list failed (%s)", ticker, e)
            continue
        if filings.empty:
            empty += 1
            continue
        new = filings[~filings["accession_number"].astype(str).isin(seen)].copy()
        if new.empty:
            continue
        new["ticker"] = ticker
        new["items"] = new["items"].fillna("").astype(str)
        new["n_items"] = new["items"].apply(lambda s: sum(1 for x in s.split(",") if x.strip()))
        out = new.reindex(columns=_COLS)
        out["filing_date"] = pd.to_datetime(out["filing_date"]).dt.normalize()
        saved = context.store.save(_TABLE, out)
        seen.update(out["accession_number"].astype(str))
        total_new += saved
        touched += 1

    context.log.info("8-K items: +%d filings across %d tickers (%d had no 8-K in window) -> '%s'",
                     total_new, touched, empty, _TABLE)
    out_all = context.store.load(_TABLE)
    return out_all
