"""
fetch_13f_managers.py (src/data_extract/utils/institutionals/fetch_13f_managers.py)
------------------------------------------------------------------------------------
The `sec_13f_manager_details` extract: one row per (cik, period) of every 13F filer since 1999-06-30, from the cached 13F
data-set covers (name, state, file number, report type) and, where no cover exists, from `sec13f_hr` marker rows named by
the cached EDGAR master index. The shared Window resolves the configured filing tail and a named bounded recovery for missing cover evidence in configured cached units.
Only overlapping archives are opened. Rows are
diffed against the stored ones and only differences are saved, extract columns only. No SEC request is made.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import pandas as pd

from src.context import Context
from src.data_extract.utils.common.bulk_cache import ZipRead, read_zip_tables
from src.data_extract.utils.common.resume import source_plan
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
from src.utils.data_policy import window_policy
from src.utils.window import Window, WindowPlan, WindowPolicy, WindowRun, resolve_window

logger = logging.getLogger(__name__)


def _manager_window(context: Context, data_sets: list[DataSet]) -> WindowPlan | None:
    """Resolve the configured tail and explicit missing-cover recovery without capping the tail at an archive."""
    table = Tables.sec_13f_manager_details
    policy = window_policy(context.config, table, context.run.as_of)
    if policy is None:
        return None

    # Load the in-scope frontier, and only pending cover clocks in the configured cached-unit recovery.
    frontier = context.store.max_date(table, "first_filed_date", where={"first_filed_date": policy.requested})
    units = [unit for unit in data_sets if unit.end >= policy.requested.start and unit.start <= policy.requested.end]
    count = int(context.config.data.tables[table.name].reprocess.cover_recovery.cached_units)
    anchor = max(policy.requested.start, units[-min(count, len(units))].start) if units else None
    df_pending = None
    if frontier is not None and anchor is not None and context.run.mode == "incremental":
        df_pending = context.store.load(
            table,
            columns=["first_filed_date"],
            where={"source": INDEX_SOURCE, "first_filed_date": Window(anchor, policy.requested.end)},
            optional=True,
        )

    # Normal overlap is resolved once. Missing covers may select the explicitly configured recovery scope.
    scope = (("source", tuple([INDEX_SOURCE, *(unit.filename for unit in data_sets)])),)
    plan = source_plan(context, table, frontier, scope)
    if plan is None or df_pending is None or df_pending.empty:
        return plan
    first_pending = pd.to_datetime(df_pending["first_filed_date"]).min()
    if pd.isna(first_pending) or first_pending >= plan.write.start:
        return plan
    recovery = Window(first_pending, policy.requested.end)
    run = WindowRun("recovery", "configured pending 13F manager cover evidence", context.run.instant)
    context.log.info("13F manager cover recovery: requested %s, effective %s, cached units %d", policy.requested, recovery, count)
    return resolve_window(WindowPolicy(recovery), run, frontier, pd.DatetimeIndex([]), scope)


def _read_zip_tables(cache: Path, data_sets: list[DataSet], window: Window) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    """`(ZIP name, SUBMISSION, COVERPAGE)` of every cached data set overlapping `window`; an unreadable ZIP is logged and skipped."""
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
    for data_set in (d for d in data_sets if d.end >= window.start and d.start <= window.end):
        tables = read_zip_tables(cache / data_set.filename, specs, on_corrupt="skip")
        if not tables:
            logger.warning("13F managers: data set %s is unreadable; skipped", data_set.name)
            continue
        reads.append((data_set.filename, tables["SUBMISSION.tsv"], tables["COVERPAGE.tsv"]))
    return reads


def _read_index_markers(context: Context, df_zip: pd.DataFrame, window: Window) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`sec13f_hr` markers filed inside `window` that no cover describes, and the cached-index rows naming them."""
    # One bounded read of the markers: the table has no cover evidence to select on, and markers are one row per filing.
    df_markers = context.store.load(
        Tables.sec13f_hr,
        columns=["cik", "period", "accession_number", "filing_date", "form"],
        where={"cusip": EMPTY_MARKER, "parse_status": ["complete", "invalid"]},
        markers=True,
        since=window.start,
        until=window.end,
        date_col="filing_date",
        optional=True,
    )
    if df_markers is None:
        return pd.DataFrame(), pd.DataFrame()
    df_markers = df_markers.loc[~df_markers["accession_number"].isin(set(df_zip["accession_number"]))]
    if df_markers.empty:
        return df_markers, pd.DataFrame()
    forms = cast(Resume, Tables.sec13f_hr.resume).forms
    return df_markers, entries(context, set(df_markers["cik"]), forms, window)


def fetch_manager_details(context: Context) -> int:
    """Refresh manager metadata inside its shared configured Window, preserving profile columns."""
    # load: the window, then its evidence (cover pages, marker rows and index names)
    cache = thirteen_f_cache(context)
    data_sets = cached_data_sets(cache)
    plan = _manager_window(context, data_sets)
    if plan is None:
        return 0
    window = plan.write
    zip_reads = _read_zip_tables(cache, data_sets, window)
    frames = [zip_manager_filings(zip_submissions(sub, cover), cover, name) for name, sub, cover in zip_reads]
    df_zip = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["accession_number"])
    df_markers, df_names = _read_index_markers(context, df_zip, window)

    # clean: the cover filings and the marker filings named by the index
    df_index = marker_manager_filings(df_markers, df_names) if not df_markers.empty else pd.DataFrame()
    logger.info(
        "13F managers: filings from %s; %d data set(s) read: %s; %d cover filing(s), %d index filing(s)",
        window.start.date(),
        len(zip_reads),
        ", ".join(name for name, _, _ in zip_reads),
        len(df_zip),
        len(df_index),
    )
    df_filings = pd.concat([df_zip, df_index], ignore_index=True)
    if df_filings.empty:
        return 0

    # merge: filings to one row per (cik, period), ZIP evidence first
    floor = pd.Timestamp(context.config.data.tables[Tables.sec_13f_manager_details.name].coverage_start)
    df_rows = candidate_rows(df_filings, floor, window.end)
    df_rows = df_rows[df_rows["first_filed_date"].between(window.start, window.end)]
    if df_rows.empty:
        return 0

    # compute: only the rows that differ from the stored ones; a missing table is created whole (extract and profile columns) first
    df_stored = context.store.load(
        Tables.sec_13f_manager_details,
        columns=list(EXTRACT_COLUMNS),
        where=df_rows[["cik", "period"]],
        optional=True,
    )
    df_changed = rows_to_save(df_rows, df_stored)
    if not df_changed["first_filed_date"].between(window.start, window.end).all():
        raise ValueError("Manager metadata correction crosses its Window; use explicit bounded historical recovery")

    # save: extract columns only; the profile columns stay as the profile step wrote them
    if not df_changed.empty:
        context.store.create_declared(Tables.sec_13f_manager_details)
        context.store.save(Tables.sec_13f_manager_details, df_changed)
    logger.info("13F managers: %d candidate row(s), %d changed", len(df_rows), len(df_changed))
    return len(df_changed)
