"""SEC submissions metadata -> `sec_cik_profile`, one row per CIK of the bulk `submissions.zip`.

Metadata only: name, entity type, SIC, EIN, state of incorporation, the current tickers and exchanges, the dated former
names, the first and last filing dates and the Form 25, 15, 8-K12B and 8-K12G3 filings (form, date, accession). A CIK's
supplementary `CIK##########-submissions-NNN.json` pages are merged into its filing list. The zip is downloaded once per
run date into the local cache and read member by member.
"""

from __future__ import annotations

import json
import re
import zipfile
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.context import Context
from src.data_extract.utils.common.bulk_cache import is_cached
from src.data_extract.utils.common.sec_io import download
from src.data_store.schema import Tables

SUBMISSIONS_URL = "https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip"
_ZIP_NAME = "submissions.zip"
_DOWNLOAD_TIMEOUT = 600  # seconds between two received chunks
#: a CIK's main JSON or one of its supplementary filing pages
_MEMBER = re.compile(r"^CIK(\d{10})(?:-submissions-\d+)?\.json$")
#: the profile column each event form (and its amendment) is listed under
_EVENT_FORMS = {
    "form_25": frozenset({"25", "25/A", "25-NSE", "25-NSE/A"}),
    "form_15": frozenset({f"{prefix}-{rule}{amend}" for prefix in ("15", "15F") for rule in ("12B", "12G", "15D") for amend in ("", "/A")}),
    "form_8k12b": frozenset({"8-K12B", "8-K12B/A"}),
    "form_8k12g3": frozenset({"8-K12G3", "8-K12G3/A"}),
}
_EVENT_COLUMN = {form: column for column, forms in _EVENT_FORMS.items() for form in forms}


def download_submissions(context: Context, cache: Path, start_date: pd.Timestamp) -> Path:
    """The cached `submissions.zip`, downloaded again unless it was already fetched on `start_date`; raises when SEC refuses it."""
    path = cache / _ZIP_NAME

    # a zip fetched on the run date is reused (a retried task does not download 1.5 GB again)
    if is_cached(path) and datetime.fromtimestamp(path.stat().st_mtime).date() == start_date.date():
        return path

    # stream it through the SEC limiter (`.part` file renamed on success)
    status = download(context, SUBMISSIONS_URL, path, timeout=_DOWNLOAD_TIMEOUT)
    if status != 200:
        raise RuntimeError(f"{SUBMISSIONS_URL}: HTTP {status}; {Tables.sec_cik_profile} left unchanged")
    return path


def _json_list(values: Sequence[Any] | None) -> str | None:
    """A non-empty list as JSON text; None for an absent or empty one."""
    return json.dumps(list(values)) if values else None


def _former_names(entries: Sequence[Mapping[str, Any]] | None) -> str | None:
    """`formerNames` as JSON `[{name, from, to}]`, dates cut to ISO days; None when the CIK never changed name."""
    names = [{"name": e.get("name"), "from": str(e.get("from") or "")[:10] or None, "to": str(e.get("to") or "")[:10] or None} for e in entries or []]
    return _json_list(names)


def _filing_lists(pages: Sequence[Mapping[str, Any]]) -> tuple[list[str], list[str], list[str]]:
    """`(accessions, filing dates, forms)` over the recent block and every supplementary page."""
    accessions: list[str] = []
    dates: list[str] = []
    forms: list[str] = []
    for page in pages:
        accessions += [str(a) for a in page.get("accessionNumber") or []]
        dates += [str(d) for d in page.get("filingDate") or []]
        forms += [str(f) for f in page.get("form") or []]
    return accessions, dates, forms


def _event_filings(accessions: list[str], dates: list[str], forms: list[str]) -> dict[str, str | None]:
    """`{profile column: JSON [{form, filed, accession}] sorted by date}` for the four event forms; None when never filed."""
    events: dict[str, list[dict[str, str]]] = {column: [] for column in _EVENT_FORMS}
    for accession, filed, form in zip(accessions, dates, forms, strict=True):
        if form in _EVENT_COLUMN:
            events[_EVENT_COLUMN[form]].append({"form": form, "filed": filed, "accession": accession})
    return {column: _json_list(sorted(rows, key=lambda r: (r["filed"], r["accession"]))) for column, rows in events.items()}


def _profile_row(cik: str, main: Mapping[str, Any], pages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """One `sec_cik_profile` row (no `fetched_at`) from a CIK's main JSON and its supplementary pages."""
    accessions, dates, forms = _filing_lists([(main.get("filings") or {}).get("recent") or {}, *pages])
    return {
        "cik": cik,
        "name": main.get("name") or None,
        "entity_type": main.get("entityType") or None,
        "sic": main.get("sic") or None,
        "ein": main.get("ein") or None,
        "state_of_incorporation": main.get("stateOfIncorporation") or None,
        "tickers": _json_list(main.get("tickers")),
        "exchanges": _json_list(main.get("exchanges")),
        "former_names": _former_names(main.get("formerNames")),
        "first_filed": min(dates) if dates else None,
        "last_filed": max(dates) if dates else None,
        "n_filings": len(accessions),
        **_event_filings(accessions, dates, forms),
    }


def _cik_members(archive: zipfile.ZipFile) -> dict[str, list[str]]:
    """`{10-digit CIK: its member names}`; members that are not a CIK's JSON are ignored."""
    members: dict[str, list[str]] = {}
    for name in archive.namelist():
        match = _MEMBER.match(name)
        if match:
            members.setdefault(match.group(1), []).append(name)
    return members


def _read_profiles(archive: zipfile.ZipFile) -> tuple[dict[str, list[Any]], int]:
    """Profile columns over every CIK with a main JSON, and the count of CIKs skipped (no main JSON or unreadable)."""
    columns: dict[str, list[Any]] = {column: [] for column in Tables.sec_cik_profile.read_columns if column != "fetched_at"}
    skipped = 0
    for cik, names in sorted(_cik_members(archive).items()):
        main_name = f"CIK{cik}.json"
        try:
            main = json.loads(archive.read(main_name))
            row = _profile_row(cik, main, [json.loads(archive.read(name)) for name in sorted(names) if name != main_name])
        except (KeyError, ValueError):  # no main JSON, malformed JSON or unequal filing columns
            skipped += 1
            continue
        for column, value in row.items():
            columns[column].append(value)
    return columns, skipped


def parse_submissions(path: Path) -> tuple[pd.DataFrame, int]:
    """`submissions.zip` -> (`sec_cik_profile` frame, one row per CIK, `fetched_at` = the zip's download time; CIKs skipped)."""
    # read every CIK's main JSON and supplementary pages
    with zipfile.ZipFile(path) as archive:
        columns, skipped = _read_profiles(archive)

    # type the dates and the count, stamp the download time
    frame = pd.DataFrame(columns)
    for column in Tables.sec_cik_profile.date_type_cols:
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    frame["n_filings"] = frame["n_filings"].astype("int64")
    frame["fetched_at"] = pd.Timestamp(datetime.fromtimestamp(path.stat().st_mtime)).floor("s")
    return frame[list(Tables.sec_cik_profile.read_columns)], skipped


def refresh_sec_cik_profile(context: Context, cache: Path, start_date: pd.Timestamp) -> int:
    """Download `submissions.zip` (once per run date), parse every CIK and replace `sec_cik_profile`; returns the rows written."""
    # download
    path = download_submissions(context, cache, start_date)

    # parse
    profile, skipped = parse_submissions(path)

    # save
    written = context.store.replace(Tables.sec_cik_profile, profile)
    context.log.info(
        "%s: %s (%.0f MB) -> %d CIK row(s) written (%d skipped: no main JSON or unreadable)",
        Tables.sec_cik_profile,
        path.name,
        path.stat().st_size / 1e6,
        written,
        skipped,
    )
    return written
