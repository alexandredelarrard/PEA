"""
fetch_13f_zips.py (src/data_extract/utils/institutionals/fetch_13f_zips.py)
----------------------------------------------------------------------------
Readers of the cached SEC Form 13F data-set ZIPs (the 13F-HR(/A) submissions of a window and the complete source
packets of chosen accessions, built by the EDGAR walk's `filing_vintage`) and the `zips-ingest --only 13f` ingest:
every 13F-HR(/A) accession of each cached window that `sec13f_hr` does not hold is saved through the walk's writer, and
a packet failing validation is recorded as an `invalid` marker. Archives stay read-only.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path
from typing import cast

import pandas as pd
from tqdm import tqdm

from src.context import Context
from src.data_extract.utils.common.bulk_cache import ZipRead, read_zip_tables
from src.data_extract.utils.institutionals.fetch_13f import (
    INVALID_STATUS,
    _pick,
    _Saver,
    _ticker_map,
    completed_accessions,
    filing_vintage,
    save_hr,
)
from src.data_extract.utils.zips.thirteen_f_datasets import cached_data_sets, thirteen_f_cache
from src.data_store.schema import Resume, Tables
from src.utils.string import pad_cik_series

logger = logging.getLogger(__name__)

_CHUNK_ROWS = 500_000
#: Pending accessions decoded, CUSIP-mapped and saved together: one INFOTABLE scan and one OpenFIGI pass per batch.
_BATCH_ACCESSIONS = 2_000

_INFO_COLS = frozenset({"ACCESSION_NUMBER", "CUSIP", "VALUE", "SSHPRNAMT", "SSHPRNAMTTYPE", "PUTCALL"})
_HEADS = {"SUBMISSION.tsv": ZipRead(upper=True), "COVERPAGE.tsv": ZipRead(upper=True)}
#: The marker statuses the ingest counts as stored; the walk counts `complete` only.
_STORED_STATUSES = ["complete", INVALID_STATUS]


def _filed_in(chunk: pd.DataFrame, accessions: frozenset[str]) -> pd.Series:
    return _pick(chunk, "ACCESSION_NUMBER").astype("string").str.strip().isin(accessions)


def zip_submissions(df_sub: pd.DataFrame, df_cover: pd.DataFrame) -> pd.DataFrame:
    """One row per 13F-HR(/A) accession: padded `cik`, `filing_date`, `period`, the cover page's
    `report_period` (else `period`) and `is_amendment`."""
    form = _pick(df_sub, "SUBMISSIONTYPE").astype("string").str.strip().str.upper()
    df_out = pd.DataFrame(
        {
            "accession": _pick(df_sub, "ACCESSION_NUMBER").astype("string").str.strip(),
            "cik": pad_cik_series(_pick(df_sub, "CIK")),
            "filing_date": pd.to_datetime(_pick(df_sub, "FILING_DATE"), format="mixed", errors="coerce").dt.normalize(),
            "period": pd.to_datetime(_pick(df_sub, "PERIODOFREPORT"), format="mixed", errors="coerce").dt.normalize(),
            "is_amendment": form.str.endswith("/A").fillna(False).astype(bool),
        }
    )[form.isin(cast(Resume, Tables.sec13f_hr.resume).forms).to_numpy()]
    report = pd.Series(dtype="datetime64[ns]")
    if not df_cover.empty:
        dates = pd.to_datetime(_pick(df_cover, "REPORTCALENDARORQUARTER"), format="mixed", errors="coerce").dt.normalize()
        report = pd.Series(dates.to_numpy(), index=_pick(df_cover, "ACCESSION_NUMBER").astype("string").str.strip()).dropna()
        report = report[~report.index.duplicated()]
    df_out["report_period"] = df_out["accession"].map(report).fillna(df_out["period"])
    return df_out.dropna(subset=["filing_date"]).reset_index(drop=True)


def cached_filing_vintages(
    path: Path,
    accessions: list[str],
    *,
    on_invalid: Callable[[str, ValueError], None] | None = None,
    heads: Mapping[str, pd.DataFrame] | None = None,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Read selected complete accessions before any latest-only filtering; archives stay read-only. `heads` holds the
    SUBMISSION and COVERPAGE tables already read from `path` (read here when None)."""
    if not accessions:
        return []
    wanted = frozenset(accessions)
    specs = {
        "SUMMARYPAGE.tsv": ZipRead(upper=True),
        "INFOTABLE.tsv": ZipRead(
            upper=True,
            chunksize=_CHUNK_ROWS,
            keep=partial(_filed_in, accessions=wanted),
            usecols=_INFO_COLS | frozenset({"NAMEOFISSUER", "TITLEOFCLASS"}),
        ),
    }
    read = read_zip_tables(path, specs if heads is not None else {**_HEADS, **specs}, on_corrupt="skip")
    if not read:
        raise ValueError(f"Unverifiable cached 13F archive: {path.name}")
    tables = {**(heads or {}), **read}
    submissions = zip_submissions(tables["SUBMISSION.tsv"], tables["COVERPAGE.tsv"]).set_index("accession")
    covers = tables["COVERPAGE.tsv"].set_index("ACCESSION_NUMBER")
    summaries = tables["SUMMARYPAGE.tsv"].set_index("ACCESSION_NUMBER")
    info_table = tables["INFOTABLE.tsv"]
    groups = info_table.groupby("ACCESSION_NUMBER", sort=False).indices
    packets: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for accession in accessions:
        try:
            if accession not in submissions.index or accession not in covers.index or accession not in summaries.index:
                raise ValueError(f"Incomplete SEC cover metadata: {accession}")
            submission, cover, summary = submissions.loc[accession], covers.loc[accession], summaries.loc[accession]
            info = info_table.iloc[groups.get(accession, [])]
            kind = str(cover["AMENDMENTTYPE"]) if submission.is_amendment else "original"
            restricted = str(summary.get("ISCONFIDENTIALOMITTED")).strip().lower() in {"true", "1", "y"}
            packets.append(
                filing_vintage(
                    info,
                    cik=str(submission.cik),
                    accession_number=accession,
                    period=submission.report_period,
                    filing_date=submission.filing_date,
                    amendment_type=kind,
                    source=path.name,
                    declared_rows=int(summary["TABLEENTRYTOTAL"]),
                    declared_value=float(summary["TABLEVALUETOTAL"]),
                    restricted=restricted,
                    report_type=str(cover["REPORTTYPE"]),
                    amendment_number=int(cover["AMENDMENTNO"]) if pd.notna(cover["AMENDMENTNO"]) else 0,
                )
            )
        except (ValueError, KeyError, TypeError) as error:
            if on_invalid is None:
                raise ValueError(f"Unverifiable SEC packet {accession}: {error}") from error
            on_invalid(accession, ValueError(str(error)))
    return packets


def _decode_batch(context: Context, path: Path, heads: Mapping[str, pd.DataFrame], df_batch: pd.DataFrame, saver: _Saver, marked: pd.Series) -> None:
    """Decode one batch of pending accessions, map its new CUSIPs in one pass and save each packet on its own: a checksum
    conflict is logged by accession; an invalid packet is logged and, unless `marked` already holds it, saved as an
    `invalid` marker (cusip = ticker = sentinel, the reason in `value_unit_evidence`)."""
    # load: the batch's complete packets; invalid ones are collected by accession
    invalid: dict[str, str] = {}
    packets = cached_filing_vintages(
        path, df_batch["accession"].tolist(), on_invalid=lambda accession, error: invalid.update({accession: str(error)}), heads=heads
    )

    # compute: one CUSIP-map pass for every packet of the batch
    if packets:
        _ticker_map(context, pd.DataFrame({"cusip": pd.concat([book["cusip"] for _, book in packets], ignore_index=True)}), saver.walk)

    # save: each packet atomically; a conflict never aborts the batch
    for manifest, book in packets:
        try:
            saver.save_packet(manifest, book)
        except ValueError as error:
            logger.error("13F ingest %s: accession %s NOT saved: %s", path.name, manifest.iloc[0]["accession_number"], error)

    # save: one invalid marker per newly invalid accession
    for accession, reason in invalid.items():
        logger.error("13F ingest %s: accession %s is invalid, marked and skipped until -F: %s", path.name, accession, reason)
    df_new = df_batch.loc[df_batch["accession"].isin(set(invalid)) & ~df_batch["accession"].isin(marked)]
    empty = cast(tuple[str, str], Tables.sec13f_hr.empty_marker)[1]
    df_markers = pd.DataFrame(
        {
            "cik": df_new["cik"],
            "period": df_new["report_period"],
            "filing_date": df_new["filing_date"],
            "accession_number": df_new["accession"],
            "form": df_new["is_amendment"].map({True: "13F-HR/A", False: "13F-HR"}),
            "cusip": empty,
            "ticker": empty,
            "parse_status": INVALID_STATUS,
            "source": path.name,
            "value_unit_evidence": df_new["accession"].map(invalid),
        }
    )
    save_hr(context, df_markers)


def _ingest_window(context: Context, path: Path, saver: _Saver, seen: set[str], df_homes: pd.DataFrame) -> None:
    """Save the pending 13F-HR(/A) accessions of one cached window in batches of `_BATCH_ACCESSIONS`. Pending = not in
    `seen` (stored, or decoded earlier in the run) and not stored under another (cik, period) in `df_homes`; `seen`
    gains every accession decoded. A window with none pending never reads its INFOTABLE."""
    # load: the window's submissions and cover pages, once
    heads = read_zip_tables(path, _HEADS, on_corrupt="skip")
    if not heads:
        logger.error("13F ingest %s: unreadable archive; skipped", path.name)
        return

    # clean: the 13F-HR(/A) accessions not stored, in walk order (filing date, amendment last, accession)
    df_sub = zip_submissions(heads["SUBMISSION.tsv"], heads["COVERPAGE.tsv"])
    df_pending = df_sub.loc[~df_sub["accession"].isin(seen)].sort_values(["filing_date", "is_amendment", "accession"])

    # clean: an accession stored under another (cik, period) is never written a second time
    homes = pd.MultiIndex.from_frame(df_homes[["accession_number", "cik", "period"]])
    own = pd.MultiIndex.from_frame(df_pending[["accession", "cik", "report_period"]]).isin(homes)
    elsewhere = df_pending["accession"].isin(df_homes["accession_number"]) & ~own
    for accession in df_pending.loc[elsewhere, "accession"]:
        logger.warning("13F ingest %s: accession %s is stored under another (cik, period); not written", path.name, accession)
    df_pending = df_pending.loc[~elsewhere]
    seen.update(df_pending["accession"])
    if df_pending.empty:
        logger.info("13F ingest %s: no pending accession; INFOTABLE not read", path.name)
        return

    # compute + save: the pending accessions, batch by batch
    for start in range(0, len(df_pending), _BATCH_ACCESSIONS):
        _decode_batch(context, path, heads, df_pending.iloc[start : start + _BATCH_ACCESSIONS], saver, df_homes["accession_number"])


def ingest_13f_datasets(context: Context, full: bool) -> int:
    """Save every 13F-HR(/A) accession of each cached 13F data set that `sec13f_hr` does not hold (complete or invalid
    marker), oldest window first, through the walk's writer; returns position rows saved. `full` re-decodes every
    accession: one stored under the same (cik, period) is re-verified by checksum, one stored under another is logged."""
    # load: the stored accessions to skip, or under `full` every stored marker's (cik, period)
    empty = cast(tuple[str, str], Tables.sec13f_hr.empty_marker)[1]
    seen = set() if full else set(completed_accessions(context, _STORED_STATUSES))
    df_stored = (
        context.store.load(
            Tables.sec13f_hr,
            columns=["accession_number", "cik", "period"],
            where={"cusip": empty, "parse_status": _STORED_STATUSES},
            markers=True,
            optional=True,
        )
        if full
        else None
    )

    # clean: one (accession, cik, period) key per stored marker
    columns = ["accession_number", "cik", "period"]
    df_homes = pd.DataFrame(columns=columns) if df_stored is None else df_stored.assign(period=pd.to_datetime(df_stored["period"]).dt.normalize())

    # compute + save: each cached window in turn, one saver (and CUSIP map) for the run
    cache = thirteen_f_cache(context)
    data_sets = cached_data_sets(cache)
    if not data_sets:
        logger.warning("13F ingest: no 13F data set is cached; run `zips-download --only 13f` first")
        return 0
    saver = _Saver(context, _BATCH_ACCESSIONS)
    for data_set in tqdm(data_sets, desc="13F data sets (ingest)"):
        _ingest_window(context, cache / data_set.filename, saver, seen, df_homes)

    # log
    logger.info("13F ingest: saved %d row(s) from %d cached data set(s) to %s", saver.walk.hr_saved, len(data_sets), Tables.sec13f_hr)
    return saver.walk.hr_saved
