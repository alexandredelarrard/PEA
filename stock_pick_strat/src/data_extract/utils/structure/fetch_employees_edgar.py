"""
fetch_employees_edgar.py  (src/data_extract/fetch_employees_edgar.py)
---------------------------------------------------------------------
FREE, full-history employee counts from SEC EDGAR 10-K (and optionally 10-Q)
body text -- a drop-in replacement for the FMP historical-employee-count fetch.

Output schema is IDENTICAL to fetch_employees.py, so employee_features.py works
unchanged:

    ticker | as_of (filing date) | period (report date) | employees | form_type

`as_of` is the SEC filing date (point-in-time / leak-free).

SPEED / INCREMENTAL
-------------------
Downloading a full 10-K per filing is the cost, so re-runs must avoid it:
  * Skip entirely when already built today for the full universe (meta sidecar).
  * Otherwise fetch ONLY filings after each ticker's last parsed `as_of` (`D`),
    i.e. the D..today window -- `list_filings(since=...)`. Already-parsed
    accessions are also skipped as a safety net.
  * Per-ticker work runs in a ThreadPoolExecutor; sec_get spaces request starts
    under SEC's 10 req/s limit, so the downloads overlap without breaching it.

Run:
    python -m src.data_extract.fetch_employees_edgar
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from src.context import Context
from src.data_extract.utils.common.sec_utils import (
    sec_get, load_cik_mapping, load_extract_meta, save_extract_meta, today_iso,
)
from src.data_extract.utils.common.edgar_fillings import list_filings
from src.data_extract.utils.common.edgar_extract import html_to_text, extract_employee_count

_DATA_COLUMNS = ["ticker", "as_of", "period", "employees", "form_type"]
_FORMS = ["10-K"]                     # add "10-Q" for quarterly headcount refreshes
_MAX_WORKERS = 8                      # concurrent tickers (rate-limited in sec_get)


def _load_existing(context: Context) -> pd.DataFrame | None:
    path = context.paths["EMPLOYEES_HISTORY_PATH"]
    return pd.read_parquet(path) if path.exists() else None


def _seen_accessions(existing: pd.DataFrame | None) -> set:
    if existing is None or existing.empty or "accession_number" not in existing.columns:
        return set()
    return set(existing["accession_number"].dropna())


def _last_asof_by_ticker(existing: pd.DataFrame | None) -> dict:
    """Max already-parsed filing date per ticker -> the incremental cutoff `D`.
    Tickers absent here (new to the universe) get their full history fetched."""
    if existing is None or existing.empty:
        return {}
    s = existing[["ticker", "as_of"]].copy()
    s["as_of"] = pd.to_datetime(s["as_of"], errors="coerce")
    s = s.dropna(subset=["as_of"])
    return s.groupby("ticker")["as_of"].max().to_dict()


def _is_up_to_date(context: Context, cik_map: pd.DataFrame) -> bool:
    """True when the history was already built today for the full universe."""
    path = context.paths["EMPLOYEES_HISTORY_PATH"]
    meta = load_extract_meta(path)
    if meta is None or meta.get("last_built") != today_iso() or not path.exists():
        return False
    return meta.get("universe_size", 0) >= len(cik_map)


def _employee_rows_for_ticker(context: Context, ticker: str, cik: str, company: str,
                              forms: list[str], years: int,
                              since: pd.Timestamp | None, seen: set) -> list[dict]:
    """One ticker's NEW employee-count rows (runs in a worker thread)."""
    try:
        filings = list_filings(cik, forms, years, company, since=since)
    except Exception as e:
        context.log.warning("%s: filing list failed (%s)", ticker, e)
        return []

    rows = []
    for _, f in filings.iterrows():
        if f["accession_number"] in seen:
            continue
        try:
            raw = sec_get(f["doc_url"]).text
            count = extract_employee_count(html_to_text(raw))
        except Exception as e:
            context.log.warning("%s %s: text fetch/parse failed (%s)",
                                ticker, f["filing_date"].date(), e)
            continue
        if count is None:
            continue
        rows.append({
            "ticker": ticker,
            "as_of": f["filing_date"],
            "period": pd.to_datetime(f.get("period_of_report"), errors="coerce"),
            "employees": count,
            "form_type": f["form"],
            "accession_number": f["accession_number"],
        })
    return rows


def fetch_employees_edgar(context: Context, tickers: list[str],
                          forms: list[str] | None = None, pause: float = 0.0) -> pd.DataFrame:
    """Build/refresh the EDGAR employee-count history. Incremental and skips a
    same-day rebuild. `pause` is accepted for backwards-compat but ignored --
    pacing is handled centrally by the rate limiter in sec_get."""
    forms = forms or _FORMS
    years = context.config.data_extract.years_history
    path = context.paths["EMPLOYEES_HISTORY_PATH"]

    cik_map = load_cik_mapping(context)
    cik_map = cik_map[cik_map["ticker"].isin(tickers)]

    existing = _load_existing(context)
    if _is_up_to_date(context, cik_map):
        context.log.info("EDGAR employees already up to date for %s — skipping (%d rows)",
                         today_iso(), 0 if existing is None else len(existing))
        return existing if existing is not None else pd.DataFrame(columns=_DATA_COLUMNS)

    seen = _seen_accessions(existing)
    last_asof = _last_asof_by_ticker(existing)

    new_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
        futures = {
            ex.submit(_employee_rows_for_ticker, context, r["ticker"], r["cik"],
                      r.get("company_name", ""), forms, years,
                      last_asof.get(r["ticker"]), seen): r["ticker"]
            for _, r in cik_map.iterrows()
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="EDGAR employee counts"):
            new_rows.extend(fut.result())

    new_df = pd.DataFrame(new_rows)
    parts = [d for d in (existing, new_df) if d is not None and not d.empty]
    if not parts:
        save_extract_meta(path, None, 0, len(cik_map))
        return pd.DataFrame(columns=_DATA_COLUMNS)

    out = pd.concat(parts, ignore_index=True)
    out["as_of"] = pd.to_datetime(out["as_of"]).dt.normalize()
    out["employees"] = pd.to_numeric(out["employees"], errors="coerce")
    out = out.dropna(subset=["as_of", "employees"])
    out = out[out["employees"] > 0]
    out = (out.sort_values(["ticker", "as_of"])
              .drop_duplicates(subset=["ticker", "as_of"], keep="last")
              .reset_index(drop=True))
    out.to_parquet(path, index=False)

    last_fd = out["as_of"].max()
    save_extract_meta(path, last_fd.date().isoformat() if pd.notna(last_fd) else None,
                      out["ticker"].nunique(), len(cik_map))
    context.log.info("EDGAR employees: %d rows, %d tickers (%d new filings parsed)",
                     len(out), out["ticker"].nunique(), len(new_df))
    # feature builder only needs _DATA_COLUMNS; accession kept for incremental dedup
    return out
