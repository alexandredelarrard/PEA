"""
fetch_13f_managers.py (src/data_extract/utils/institutionals/fetch_13f_managers.py)
------------------------------------------------------------------------------------
The `sec_13f_manager_details` extract: one row per (cik, period) of every 13F filer since 1999-06-30, from the cached 13F
data-set covers (name, state, file number, report type) and, where no cover exists, from `sec13f_hr` marker rows named by
the cached EDGAR master index. The refresh window starts at the newest stored `first_filed_date` minus the table's overlap
(or at the oldest row still lacking cover evidence), never before the second-newest cached data set; only the data sets
overlapping the window are opened, and older index rows are revisited by a forced full run only. Rows are
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
from src.data_store.schema import EMPTY_MARKER, Resume, Tables
from src.utils.string import run_date

logger = logging.getLogger(__name__)


def _window_start(context: Context, data_sets: list[DataSet], full: bool) -> pd.Timestamp:
    """First filing date to read: the table's source start under `full` or an empty table, else the earlier of the newest
    stored `first_filed_date` minus the overlap and the oldest index-evidence row filed since the second-newest data set,
    never before that data set's start."""
    # load: the frontier, then only the index-evidence rows filed since the anchor data set (nothing under `full`)
    table = Tables.sec_13f_manager_details
    resume = cast(Resume, table.resume)
    floor = pd.Timestamp(cast(str, resume.source_start))
    anchor = data_sets[-2 if len(data_sets) > 1 else 0].start if data_sets else None
    frontier = None if full else context.store.max_date(table, "first_filed_date")
    df_lacking = None
    if frontier is not None and anchor is not None:
        df_lacking = context.store.load(
            table, columns=["first_filed_date"], where={"source": INDEX_SOURCE}, since=anchor, date_col="first_filed_date", optional=True
        )
    if frontier is None:
        return floor

    # compute: the newest filing minus the overlap, then the oldest index-evidence row since the anchor data set
    starts = [frontier - pd.Timedelta(days=resume.overlap_days)]
    if anchor is None:
        return max(floor, starts[0])
    if df_lacking is not None and not df_lacking.empty:
        starts.append(pd.to_datetime(df_lacking["first_filed_date"]).min())
    return max(floor, anchor if len(data_sets) > 1 else floor, min(starts))


def _read_zip_tables(cache: Path, data_sets: list[DataSet], since: pd.Timestamp, until: pd.Timestamp) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    """`(ZIP name, SUBMISSION, COVERPAGE)` of every cached data set overlapping `[since, until]`; an unreadable ZIP is logged and skipped."""
    specs = {
        "SUBMISSION.tsv": ZipRead(upper=True, usecols=frozenset({"ACCESSION_NUMBER", "FILING_DATE", "SUBMISSIONTYPE", "CIK", "PERIODOFREPORT"})),
        "COVERPAGE.tsv": ZipRead(
            upper=True,
            usecols=frozenset(
                {"ACCESSION_NUMBER", "REPORTCALENDARORQUARTER", "FILINGMANAGER_NAME", "FILINGMANAGER_STATEORCOUNTRY", "REPORTTYPE", "SECFILENUMBER"}
            ),
        ),
    }
    reads: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
    for data_set in (d for d in data_sets if d.end >= since and d.start <= until):
        tables = read_zip_tables(cache / data_set.filename, specs, on_corrupt="skip")
        if not tables:
            logger.warning("13F managers: data set %s is unreadable; skipped", data_set.name)
            continue
        reads.append((data_set.filename, tables["SUBMISSION.tsv"], tables["COVERPAGE.tsv"]))
    return reads


def _read_index_markers(context: Context, df_zip: pd.DataFrame, since: pd.Timestamp, until: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`sec13f_hr` markers filed in `[since, until]` that no cover describes, and the cached-index rows naming them."""
    # One bounded read of the markers: the table has no cover evidence to select on, and markers are one row per filing.
    df_markers = context.store.load(
        Tables.sec13f_hr,
        columns=["cik", "period", "accession_number", "filing_date", "form"],
        where={"cusip": EMPTY_MARKER, "parse_status": ["complete", "invalid"]},
        markers=True,
        since=since,
        until=until,
        date_col="filing_date",
        optional=True,
    )
    if df_markers is None:
        return pd.DataFrame(), pd.DataFrame()
    df_markers = df_markers.loc[~df_markers["accession_number"].isin(set(df_zip["accession_number"]))]
    if df_markers.empty:
        return df_markers, pd.DataFrame()
    forms = cast(Resume, Tables.sec13f_hr.resume).forms
    return df_markers, entries(context, set(df_markers["cik"]), forms, since=since)


def _load_stored(context: Context, df_rows: pd.DataFrame, full: bool) -> pd.DataFrame | None:
    """The stored extract columns of the candidates' keys, read over their period range (everything under `full`)."""
    df_stored = context.store.load(
        Tables.sec_13f_manager_details,
        columns=list(EXTRACT_COLUMNS),
        since=None if full else df_rows["period"].min(),
        until=None if full else df_rows["period"].max(),
        optional=True,
    )
    if df_stored is None:
        return None
    return df_stored.assign(period=pd.to_datetime(df_stored["period"]).dt.normalize()).merge(df_rows[["cik", "period"]], on=["cik", "period"])


def fetch_manager_details(context: Context, full: bool, as_of: pd.Timestamp | None) -> int:
    """Refresh `sec_13f_manager_details`; returns the rows written. `full` re-reads every cached data set and marker."""
    # load: the window, then its evidence (cover pages, marker rows and index names)
    until = run_date(as_of)
    cache = thirteen_f_cache(context)
    data_sets = cached_data_sets(cache)
    since = _window_start(context, data_sets, full)
    zip_reads = _read_zip_tables(cache, data_sets, since, until)
    frames = [zip_manager_filings(zip_submissions(sub, cover), cover, name) for name, sub, cover in zip_reads]
    df_zip = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["accession_number"])
    df_markers, df_names = _read_index_markers(context, df_zip, since, until)

    # clean: the cover filings and the marker filings named by the index
    df_index = marker_manager_filings(df_markers, df_names) if not df_markers.empty else pd.DataFrame()
    logger.info(
        "13F managers: filings from %s; %d data set(s) read: %s; %d cover filing(s), %d index filing(s)",
        since.date(),
        len(zip_reads),
        ", ".join(name for name, _, _ in zip_reads),
        len(df_zip),
        len(df_index),
    )
    df_filings = pd.concat([df_zip, df_index], ignore_index=True)
    if df_filings.empty:
        return 0

    # merge: filings to one row per (cik, period), ZIP evidence first
    floor = pd.Timestamp(cast(str, cast(Resume, Tables.sec13f_hr.resume).coverage_start))
    df_rows = candidate_rows(df_filings, floor, until)
    if df_rows.empty:
        return 0

    # compute: only the rows that differ from the stored ones
    df_changed = rows_to_save(df_rows, _load_stored(context, df_rows, full))

    # save: extract columns only; the profile columns stay as the profile step wrote them
    if not df_changed.empty:
        context.store.save(Tables.sec_13f_manager_details, df_changed)
    logger.info("13F managers: %d candidate row(s), %d changed", len(df_rows), len(df_changed))
    return len(df_changed)
