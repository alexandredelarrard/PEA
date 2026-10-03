"""List a company's filings of arbitrary form types over its full submissions history.

Reads the recent page and the older paginated `filings.files[]` pages of submissions/CIK{cik}.json,
so long histories are not truncated. On-demand filing discovery for the structure fetchers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.constants.constants import (
    SEC_ARCHIVES_BASE_URL,
    SEC_SUBMISSIONS_PAGE_URL,
    SEC_SUBMISSIONS_URL,
)
from src.context import Context
from src.data_extract.utils.common.sec_utils import sec_get
from src.utils.string import pad_cik


def archive_url(cik: str, accession: str, document: str) -> str:
    """Absolute EDGAR archives URL of one document inside a filing's accession folder."""
    return f"{SEC_ARCHIVES_BASE_URL}/{int(cik)}/{accession.replace('-', '')}/{document}"


def _rows_from_recent(block: dict, cik: str, company: str, forms: set, cutoff: pd.Timestamp) -> list[dict]:
    rows = []
    n = len(block.get("accessionNumber", []))
    for i in range(n):
        form = block["form"][i]
        if form not in forms:
            continue
        fdate = pd.Timestamp(block["filingDate"][i])
        if fdate < cutoff:
            continue
        acc = block["accessionNumber"][i]
        primary = block["primaryDocument"][i]
        rows.append(
            {
                "cik": cik,
                "company_name": company,
                "form": form,
                "filing_date": fdate,
                "period_of_report": block.get("reportDate", [None] * n)[i],
                "accession_number": acc,
                "primary_document": primary,
                # Old filings have an empty `primaryDocument`; the `<accession>.txt` full submission stands in.
                "doc_url": archive_url(cik, acc, primary or f"{acc}.txt"),
                # Fallback consumers retry when a named primary document is missing from the archive.
                "txt_url": archive_url(cik, acc, f"{acc}.txt"),
                # 8-K structured item codes (e.g. "2.02,9.01"); "" for forms without items
                "items": (block.get("items", [""] * n)[i] or ""),
            }
        )
    return rows


def _cache_json(cache_dir: Path | None, name: str, payload: dict) -> None:
    """Best-effort save of a raw submissions page before it is parsed; no-op when `cache_dir` is None."""
    if cache_dir is None:
        return
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / name).write_text(json.dumps(payload), encoding="utf-8")
    except Exception:  # caching is best-effort; never break extraction
        pass


def list_filings(
    context: Context,
    cik: str,
    forms: list[str],
    years: int,
    company_name: str = "",
    since: pd.Timestamp | str | None = None,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """All filings of `forms` for one CIK over the last `years` years, oldest first, incl. 8-K `items`.

    With `since` (a date already fully parsed) only filings strictly after it are returned; archive
    pages entirely before the cutoff are not downloaded. `cache_dir` keeps every raw page.
    """
    cik = pad_cik(cik)
    forms_set = set(forms)
    cutoff = pd.Timestamp.today() - pd.DateOffset(years=years)
    if since is not None:
        cutoff = max(cutoff, pd.Timestamp(since).normalize() + pd.Timedelta(days=1))

    data = sec_get(context, SEC_SUBMISSIONS_URL.format(cik=cik)).json()
    _cache_json(cache_dir, f"CIK{cik}_submissions.json", data)
    company = company_name or data.get("name", "")
    filings = data.get("filings", {})

    rows = _rows_from_recent(filings.get("recent", {}), cik, company, forms_set, cutoff)

    for f in filings.get("files", []):
        older_name = f.get("name")
        if not older_name:
            continue
        page_to = f.get("filingTo")
        if page_to and pd.Timestamp(page_to) < cutoff:
            continue
        try:
            page = sec_get(context, SEC_SUBMISSIONS_PAGE_URL.format(name=older_name)).json()
        except Exception:
            continue
        _cache_json(cache_dir, older_name, page)
        rows += _rows_from_recent(page, cik, company, forms_set, cutoff)

    df = pd.DataFrame(rows)
    if not df.empty:
        df["filing_date"] = pd.to_datetime(df["filing_date"]).dt.normalize()
        df = df.sort_values("filing_date").reset_index(drop=True)
    return df
