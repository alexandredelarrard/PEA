"""
fetch_13d.py  (src/data_extract/utils/structure/fetch_13d.py)
-------------------------------------------------------------
SC 13D activist filings (+ amendments) about the subject company -> `sec_13d`. The event itself —
an activist crossing >5% WITH intent to influence, and every amendment — is the catalyst signal, and
it comes STRUCTURED from the subject company's EDGAR submissions JSON (SEC indexes 13D under the
subject CIK), so this needs NO document parsing and is ~100% event-fill.

Per ticker: list SC 13D / SC 13D/A via the shared `list_filings` (raw JSON cached to
data/sec_13d_cache/ before the DB), keep filings not already stored, upsert per ticker. Incremental
& DAG-fast via the per-ticker `since = max(stored filing_date)`; deduped by accession. (Filer name /
% owned live in the cover page and are a later best-effort doc-parse enhancement — the event/date is
the reliable core stored here.)
"""
from __future__ import annotations

import logging

import pandas as pd
from tqdm import tqdm

from src.constants.constants import SEC_13D_CACHE_DIR, SEC_13D_FORMS, SEC_13D_TABLE
from src.context import Context
from src.data_extract.utils.common.edgar_fillings import list_filings
from src.data_extract.utils.common.sec_utils import load_cik_mapping

logger = logging.getLogger(__name__)
_TABLE = SEC_13D_TABLE
_COLS = ["ticker", "cik", "accession_number", "form", "filing_date", "primary_document", "doc_url"]


def _existing(context: Context) -> tuple[set[str], dict[str, pd.Timestamp]]:
    """(accessions already stored, per-ticker max filing_date) — dedup + incremental cutoff."""
    try:
        df = context.store.load(_TABLE, columns=["ticker", "accession_number", "filing_date"])
    except Exception:
        return set(), {}
    if df is None or df.empty:
        return set(), {}
    seen = set(df["accession_number"].dropna().astype(str))
    d = df.copy()
    d["filing_date"] = pd.to_datetime(d["filing_date"], errors="coerce")
    last = d.dropna(subset=["filing_date"]).groupby("ticker")["filing_date"].max().to_dict()
    return seen, last


def fetch_13d(context: Context, tickers: list[str], years: int | None = None) -> pd.DataFrame:
    """Build/refresh the SC 13D activist-filing table, one ticker at a time (upsert per ticker).
    Returns the full table."""
    years = int(years if years is not None else context.config.data_extract.years_history)
    cache_dir = context.paths["DATA_STORE"] / SEC_13D_CACHE_DIR
    cik_map = load_cik_mapping(context)
    cik_map = cik_map[cik_map["ticker"].isin(tickers)]

    seen, last_by_ticker = _existing(context)
    total_new, touched, none_found = 0, 0, 0
    for _, r in tqdm(cik_map.iterrows(), total=len(cik_map), desc="SC 13D"):
        ticker, cik, company = r["ticker"], r["cik"], r.get("company_name", "")
        try:
            filings = list_filings(cik, SEC_13D_FORMS, years, company,
                                   since=last_by_ticker.get(ticker), cache_dir=cache_dir)
        except Exception as e:                          # noqa: BLE001
            context.log.warning("%s: SC 13D list failed (%s)", ticker, e)
            continue
        if filings.empty:
            none_found += 1
            continue
        new = filings[~filings["accession_number"].astype(str).isin(seen)].copy()
        if new.empty:
            continue
        new["ticker"] = ticker
        out = new.reindex(columns=_COLS)
        out["filing_date"] = pd.to_datetime(out["filing_date"]).dt.normalize()
        total_new += context.store.save(_TABLE, out)
        seen.update(out["accession_number"].astype(str))
        touched += 1

    context.log.info("SC 13D: +%d filings across %d tickers (%d had no 13D in window) -> '%s'",
                     total_new, touched, none_found, _TABLE)
    return context.store.load(_TABLE)
