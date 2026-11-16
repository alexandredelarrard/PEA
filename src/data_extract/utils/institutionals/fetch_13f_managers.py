"""
fetch_13f_managers.py (src/data_extract/utils/institutionals/fetch_13f_managers.py)
------------------------------------------------------------------------------------
The `sec_13f_manager_details` extract: one row per (cik, period) of every 13F filer since 1999-06-30, from the cached 13F
data-set covers (name, state, file number, report type) and, where no cover exists, from `sec13f_hr` marker rows named by
the cached EDGAR master index. The refresh window starts at the newest stored `first_filed_date` minus the table's overlap
(or at the oldest row still lacking cover evidence); only the data sets overlapping the window are opened. Rows are
diffed against the stored ones and only differences are saved, extract columns only. No SEC request is made.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import pandas as pd

from src.context import Context
from src.data_extract.utils.common.bulk_cache import ZipRead, read_zip_tables
from src.data_extract.utils.edgar_extract.edgar_index import entries
from src.data_extract.utils.institutionals.fetch_13f_zips import zip_submissions
from src.data_extract.utils.institutionals.manager_rows import (
    EXTRACT_COLUMNS,
    INDEX_SOURCE,
    candidate_rows,
    marker_manager_filings,
    rows_to_save,
    zip_manager_filings,
)
from src.data_extract.utils.zips.thirteen_f_datasets import DataSet, cached_data_sets, thirteen_f_cache
from src.data_store.schema import Resume, Tables
from src.utils.string import run_date

logger = logging.getLogger(__name__)

_SPECS = {
    "SUBMISSION.tsv": ZipRead(upper=True, usecols=frozenset({"ACCESSION_NUMBER", "FILING_DATE", "SUBMISSIONTYPE", "CIK", "PERIODOFREPORT"})),
    "COVERPAGE.tsv": ZipRead(
        upper=True,
        usecols=frozenset(
            {"ACCESSION_NUMBER", "REPORTCALENDARORQUARTER", "FILINGMANAGER_NAME", "FILINGMANAGER_STATEORCOUNTRY", "REPORTTYPE", "SECFILENUMBER"}
        ),
    ),
}
_MARKER_COLUMNS = ["cik", "period", "accession_number", "filing_date", "form"]
_MARKER_STATUSES = ["complete", "invalid"]


def _window_start(context: Context, data_sets: list[DataSet], full: bool) -> pd.Timestamp:
    """First filing date to read: the table's source start under `full` or an empty table, else the earlier of the newest
    stored `first_filed_date` minus the overlap and the oldest filing since the first data set that still has index evidence."""
    table = Tables.sec_13f_manager_details
    resume = cast(Resume, table.resume)
    floor = pd.Timestamp(cast(str, resume.source_start))
    frontier = None if full else context.store.max_date(table, "first_filed_date")
    if frontier is None:
        return floor
    starts = [frontier - pd.Timedelta(days=resume.overlap_days)]
    if data_sets:
        df_lacking = context.store.load(
            table, columns=["first_filed_date"], where={"source": INDEX_SOURCE}, since=data_sets[0].start, date_col="first_filed_date", optional=True
        )
        if df_lacking is not None:
            starts.append(pd.to_datetime(df_lacking["first_filed_date"]).min())
    return max(floor, min(starts))


def _read_zip_filings(cache: Path, data_sets: list[DataSet], since: pd.Timestamp, until: pd.Timestamp) -> tuple[pd.DataFrame, list[str]]:
    """Filings of every cached data set overlapping `[since, until]` and the names of the ZIPs read; an unreadable ZIP is
    logged and skipped."""
    frames: list[pd.DataFrame] = []
    names: list[str] = []
    for data_set in (d for d in data_sets if d.end >= since and d.start <= until):
        tables = read_zip_tables(cache / data_set.filename, _SPECS, on_corrupt="skip")
        if not tables:
            logger.warning("13F managers: data set %s is unreadable; skipped", data_set.name)
            continue
        frames.append(
            zip_manager_filings(zip_submissions(tables["SUBMISSION.tsv"], tables["COVERPAGE.tsv"]), tables["COVERPAGE.tsv"], data_set.filename)
        )
        names.append(data_set.filename)
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["accession_number"])), names


def _read_index_filings(context: Context, df_zip: pd.DataFrame, since: pd.Timestamp, until: pd.Timestamp) -> pd.DataFrame:
    """Filings of `sec13f_hr` markers filed in `[since, until]` that no cover describes, named by the cached index."""
    df_markers = context.store.load(
        Tables.sec13f_hr,
        columns=_MARKER_COLUMNS,
        where={"cusip": "_empty", "parse_status": _MARKER_STATUSES},
        markers=True,
        since=since,
        until=until,
        date_col="filing_date",
        optional=True,
    )
    if df_markers is None:
        return pd.DataFrame()
    df_markers = df_markers.loc[~df_markers["accession_number"].isin(set(df_zip["accession_number"]))]
    forms = cast(Resume, Tables.sec13f_hr.resume).forms
    df_index = entries(context, set(df_markers["cik"]), forms, since=since) if not df_markers.empty else pd.DataFrame()
    return marker_manager_filings(df_markers, df_index) if not df_markers.empty else pd.DataFrame()


def _save_changed(context: Context, df_rows: pd.DataFrame, full: bool) -> int:
    """Merge the candidate rows with the stored ones, save the extract columns of the rows that differ, return their count."""
    # load: the stored rows of the candidate keys (every row under `full`)
    keys = df_rows[["cik", "period"]]
    df_stored = context.store.load(Tables.sec_13f_manager_details, columns=list(EXTRACT_COLUMNS), where=None if full else keys, optional=True)

    # compute: only the differences
    df_changed = rows_to_save(df_rows, df_stored)

    # save: extract columns only; the profile columns stay as the profile step wrote them
    if not df_changed.empty:
        context.store.save(Tables.sec_13f_manager_details, df_changed)
    return len(df_changed)


def fetch_manager_details(context: Context, full: bool, as_of: pd.Timestamp | None) -> int:
    """Refresh `sec_13f_manager_details`; returns the rows written. `full` re-reads every cached data set and marker."""
    # load: the window, then its evidence (cover pages, marker rows and index names)
    until = run_date(as_of)
    data_sets = cached_data_sets(thirteen_f_cache(context))
    since = _window_start(context, data_sets, full)
    df_zip, read = _read_zip_filings(thirteen_f_cache(context), data_sets, since, until)
    df_index = _read_index_filings(context, df_zip, since, until)

    # log: the window and what it holds
    logger.info(
        "13F managers: filings from %s; %d data set(s) read: %s; %d cover filing(s), %d index filing(s)",
        since.date(),
        len(read),
        ", ".join(read),
        len(df_zip),
        len(df_index),
    )
    df_filings = pd.concat([df_zip, df_index], ignore_index=True)
    if df_filings.empty:
        return 0

    # clean + merge: filings to one row per (cik, period), ZIP evidence first
    floor = pd.Timestamp(cast(str, cast(Resume, Tables.sec13f_hr.resume).coverage_start))
    df_rows = candidate_rows(df_filings, floor, until)
    if df_rows.empty:
        return 0

    # save: only the changed rows
    changed = _save_changed(context, df_rows, full)
    logger.info("13F managers: %d candidate row(s), %d changed", len(df_rows), changed)
    return changed
