"""
edgar_fillings.py  (src/data_extract/utils/common/edgar_fillings.py)
-----------------------------------------------------------
List a company's filings of arbitrary form types over the full history,
INCLUDING the older paginated pages that submissions/CIK{cik}.json splits out
(the `filings.files[]` archives), so long histories are not truncated (covers
15y+). Shared, on-demand filing discovery for the structure fetchers (DEF 14A,
employees) -- there is no separate filing-index download.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.constants.constants import (
    SEC_ARCHIVES_BASE_URL, SEC_SUBMISSIONS_PAGE_URL, SEC_SUBMISSIONS_URL,
)
from src.data_extract.utils.common.sec_utils import sec_get


def _doc_url(cik: str, accession: str, primary_doc: str) -> str:
    acc_nodash = accession.replace("-", "")
    return f"{SEC_ARCHIVES_BASE_URL}/{int(cik)}/{acc_nodash}/{primary_doc}"


def _rows_from_recent(block: dict, cik: str, company: str, forms: set,
                      cutoff: pd.Timestamp) -> list[dict]:
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
        rows.append({
            "cik": cik, "company_name": company, "form": form,
            "filing_date": fdate, "period_of_report": block.get("reportDate", [None] * n)[i],
            "accession_number": acc, "primary_document": primary,
            "doc_url": _doc_url(cik, acc, primary),
            # 8-K structured item codes (e.g. "2.02,9.01"); "" for forms without items
            "items": (block.get("items", [""] * n)[i] or ""),
        })
    return rows


def _cache_json(cache_dir: Path | None, name: str, payload: dict) -> None:
    """Save a raw EDGAR submissions page to disk BEFORE it is parsed (reproducible re-parse /
    offline). No-op when `cache_dir` is None."""
    if cache_dir is None:
        return
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / name).write_text(json.dumps(payload), encoding="utf-8")
    except Exception:                       # caching is best-effort; never break extraction
        pass


def list_filings(cik: str, forms: list[str], years: int,
                 company_name: str = "", since: pd.Timestamp | str | None = None,
                 cache_dir: Path | None = None) -> pd.DataFrame:
    """All filings of `forms` for one CIK, across the recent page AND older
    archive pages. Returns columns incl. `items` (8-K structured item codes).

    Window: the last `years` years by default. When `since` is given (a date
    already fully parsed, `D`), only filings STRICTLY AFTER it are returned --
    this is the incremental path, so a re-run fetches just D..today instead of
    re-listing the whole history. Older archive pages entirely before the cutoff
    are skipped without being downloaded. When `cache_dir` is given, every raw
    submissions page is written there before parsing.
    """
    cik = str(cik).zfill(10)
    forms_set = set(forms)
    cutoff = pd.Timestamp.today() - pd.DateOffset(years=years)
    if since is not None:
        # strictly after the last date already parsed
        cutoff = max(cutoff, pd.Timestamp(since).normalize() + pd.Timedelta(days=1))

    data = sec_get(SEC_SUBMISSIONS_URL.format(cik=cik)).json()
    _cache_json(cache_dir, f"CIK{cik}_submissions.json", data)
    company = company_name or data.get("name", "")
    filings = data.get("filings", {})

    rows = _rows_from_recent(filings.get("recent", {}), cik, company, forms_set, cutoff)

    # older paginated archives
    for f in filings.get("files", []):
        older_name = f.get("name")
        if not older_name:
            continue
        # only fetch a page if its date range can overlap our window
        page_to = f.get("filingTo")
        if page_to and pd.Timestamp(page_to) < cutoff:
            continue
        try:
            page = sec_get(SEC_SUBMISSIONS_PAGE_URL.format(name=older_name)).json()
        except Exception:
            continue
        _cache_json(cache_dir, older_name, page)
        rows += _rows_from_recent(page, cik, company, forms_set, cutoff)

    df = pd.DataFrame(rows)
    if not df.empty:
        df["filing_date"] = pd.to_datetime(df["filing_date"]).dt.normalize()
        df = df.sort_values("filing_date").reset_index(drop=True)
    return df