"""
fetch_employees_edgar.py  (src/data_extract/fetch_employees_edgar.py)
---------------------------------------------------------------------
FREE, full-history employee counts from SEC EDGAR 10-K (and optionally 10-Q)
body text -- a drop-in replacement for the FMP historical-employee-count fetch.

Output schema is IDENTICAL to fetch_employees.py, so employee_features.py works
unchanged:

    ticker | as_of (filing date) | period (report date) | employees | form_type

`as_of` is the SEC filing date (point-in-time / leak-free). Incremental: filings
already parsed (by accession) are skipped on re-run.

Run:
    python -m src.data_extract.fetch_employees_edgar
"""
from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from src.context import Context
from src.data_extract.utils.sec_utils import sec_get, load_cik_mapping
from src.data_extract.utils.edgar_fillings import list_filings
from src.data_extract.utils.edgar_extract import html_to_text, extract_employee_count

_DATA_COLUMNS = ["ticker", "as_of", "period", "employees", "form_type"]
_FORMS = ["10-K"]                     # add "10-Q" for quarterly headcount refreshes


def _load_existing(context: Context) -> pd.DataFrame | None:
    path = context.paths["EMPLOYEES_HISTORY_PATH"]
    return pd.read_parquet(path) if path.exists() else None


def _seen_accessions(existing: pd.DataFrame | None) -> set:
    if existing is None or existing.empty or "accession_number" not in existing.columns:
        return set()
    return set(existing["accession_number"].dropna())


def fetch_employees_edgar(context: Context, tickers: list[str],
                          forms: list[str] | None = None, pause: float = 0.0) -> pd.DataFrame:
    forms = forms or _FORMS
    years = context.config.data_extract.years_history
    cik_map = load_cik_mapping(context)
    cik_map = cik_map[cik_map["ticker"].isin(tickers)]

    existing = _load_existing(context)
    seen = _seen_accessions(existing)

    new_rows = []
    for _, r in tqdm(cik_map.iterrows(), total=len(cik_map), desc="EDGAR employee counts"):
        ticker, cik = r["ticker"], r["cik"]
        try:
            filings = list_filings(cik, forms, years, r.get("company_name", ""))
        except Exception as e:
            context.log.warning("%s: filing list failed (%s)", ticker, e)
            continue

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
            new_rows.append({
                "ticker": ticker,
                "as_of": f["filing_date"],
                "period": pd.to_datetime(f.get("period_of_report"), errors="coerce"),
                "employees": count,
                "form_type": f["form"],
                "accession_number": f["accession_number"],
            })

    new_df = pd.DataFrame(new_rows)
    parts = [d for d in (existing, new_df) if d is not None and not d.empty]
    if not parts:
        return existing if existing is not None else pd.DataFrame(columns=_DATA_COLUMNS)

    out = pd.concat(parts, ignore_index=True)
    out["as_of"] = pd.to_datetime(out["as_of"]).dt.normalize()
    out["employees"] = pd.to_numeric(out["employees"], errors="coerce")
    out = out.dropna(subset=["as_of", "employees"])
    out = out[out["employees"] > 0]
    out = (out.sort_values(["ticker", "as_of"])
              .drop_duplicates(subset=["ticker", "as_of"], keep="last")
              .reset_index(drop=True))
    out.to_parquet(context.paths["EMPLOYEES_HISTORY_PATH"], index=False)
    context.log.info("EDGAR employees: %d rows, %d tickers (%d new filings parsed)",
                     len(out), out["ticker"].nunique(), len(new_df))
    # feature builder only needs _DATA_COLUMNS; accession kept for incremental dedup
    return out