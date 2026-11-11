"""List a company's filings of arbitrary form types over its full submissions history.

Reads the recent page and the older paginated `filings.files[]` pages of submissions/CIK{cik}.json,
so long histories are not truncated. Read by the DEF 14A text walk; `archive_url` is shared by the EDGAR driver.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from src.constants.constants import (
    SEC_ARCHIVES_BASE_URL,
    SEC_SUBMISSIONS_PAGE_URL,
    SEC_SUBMISSIONS_URL,
)
from src.context import Context
from src.data_extract.utils.common.sec_io import sec_get
from src.utils.string import pad_cik


def archive_url(cik: str, accession: str, document: str) -> str:
    """Absolute EDGAR archives URL of one document inside a filing's accession folder."""
    return f"{SEC_ARCHIVES_BASE_URL}/{int(cik)}/{accession.replace('-', '')}/{document}"


def _rows_from_recent(block: dict, forms: set, cutoff: pd.Timestamp) -> list[dict]:
    rows = []
    n = len(block.get("accessionNumber", []))
    for i in range(n):
        if block["form"][i] not in forms:
            continue
        if pd.Timestamp(block["filingDate"][i]) < cutoff:
            continue
        rows.append(
            {
                "accession_number": block["accessionNumber"][i],
                "primary_document": block["primaryDocument"][i],
            }
        )
    return rows


def list_filings(context: Context, cik: str, forms: Sequence[str], since: pd.Timestamp | str) -> pd.DataFrame:
    """Filings of `forms` for one CIK filed strictly after `since`: accession and primary document.

    Archive pages entirely before the cutoff are not downloaded. A page that cannot be read raises
    (`sec_io.TransientReadError` once retries are spent), so a listing is never silently short.
    """
    cik = pad_cik(cik)
    forms_set = set(forms)
    cutoff = pd.Timestamp(since).normalize() + pd.Timedelta(days=1)

    filings = sec_get(context, SEC_SUBMISSIONS_URL.format(cik=cik)).json().get("filings", {})
    rows = _rows_from_recent(filings.get("recent", {}), forms_set, cutoff)

    for f in filings.get("files", []):
        older_name = f.get("name")
        if not older_name:
            continue
        page_to = f.get("filingTo")
        if page_to and pd.Timestamp(page_to) < cutoff:
            continue
        rows += _rows_from_recent(sec_get(context, SEC_SUBMISSIONS_PAGE_URL.format(name=older_name)).json(), forms_set, cutoff)

    return pd.DataFrame(rows)
