"""
fetch_13f_managers.py (src/data_extract/utils/institutionals/fetch_13f_managers.py)
------------------------------------------------------------------------------------
Per-CIK immutable-accession catch-up for every ever-listed roster manager. Complete source packets
are saved atomically; originals and amendments remain separate. Only a completed accession is skipped.
Both existing destinations preserve source versions.
"""

from __future__ import annotations

import json
import logging
from functools import partial
from pathlib import Path
from typing import cast
from zipfile import ZipFile

import pandas as pd

from src.context import Context
from src.data_extract.utils.common.bulk_cache import ZipRead, read_zip_tables
from src.data_extract.utils.common.edgar_driver import FilingStamp
from src.data_extract.utils.common.parallel_fetch import run_per_ticker
from src.data_extract.utils.common.resume import DONE_PER_KEY, stored_accessions
from src.data_extract.utils.common.sec_io import company, company_filings, configure
from src.data_extract.utils.institutionals.fetch_13f import _IMPLIED_PRICE_BAND, ReadFailure, _suspect_prices, read_filing_vintage, save_13f_filing
from src.data_extract.utils.institutionals.fetch_13f_backfill import _submissions, cached_filing_vintages
from src.data_store.schema import Resume, Tables
from src.utils.string import pad_cik
from src.utils.superinvestor_roster import roster_cik_union, roster_map_as_of

logger = logging.getLogger(__name__)


class SuperinvestorRosterEmptyError(RuntimeError):
    """`superinvestor_roster` holds no CIK; the roster is the walk's entire input, so the run stops."""


def _listed_filings(cik: str, since: pd.Timestamp) -> list[tuple[FilingStamp, pd.Timestamp]]:
    """`(stamp, period)` for the CIK's 13F-HR filings whose period is on/after `since`, oldest
    first by (filed, amendment last, accession) -- edgartools lists newest first; a null or
    unparseable period is skipped."""
    listing = company_filings(company(cik), cast(Resume, Tables.sec13f_manager_holdings.resume).forms) or []
    stamps = sorted((FilingStamp.of(f, cik) for f in listing), key=lambda s: (s.filed, s.is_amendment, s.accession_number))
    periods = [pd.to_datetime(s.period_of_report, errors="coerce") for s in stamps]
    return [(s, period.normalize()) for s, period in zip(stamps, periods, strict=True) if pd.notna(period) and period >= since]


def _pending_filings(context: Context, cik: str, listed: list[tuple[FilingStamp, pd.Timestamp]]) -> list[tuple[FilingStamp, pd.Timestamp]]:
    """Only a verified completed accession can remove that accession from the pending set."""
    completed = stored_accessions(context, Tables.sec13f_manager_holdings, [cik], DONE_PER_KEY, where={"parse_status": "complete"})[cik]
    return [(stamp, period) for stamp, period in listed if stamp.accession_number not in completed]


def _log_read_failure(context: Context, cik: str, stamp: FilingStamp, period: pd.Timestamp, failure: ReadFailure) -> None:
    """WARNING for a transient failure (only this accession remains pending), ERROR for a deterministic one."""
    if failure.transient:
        context.log.warning(
            "13F managers: CIK %s %s (period %s) failed transiently (%s); this accession is retried next run; verified sibling filings are saved",
            cik,
            stamp.accession_number,
            period.date(),
            failure.reason,
        )
        return
    context.log.error(
        "13F managers: CIK %s %s (period %s) is unreadable (%s); the filing is skipped and the quarter's readable filings are saved",
        cik,
        stamp.accession_number,
        period.date(),
        failure.reason,
    )


def _catch_up_cik(label: str, cik: str, *, context: Context, since: pd.Timestamp) -> tuple[int, int, int]:
    """Save each complete pending accession independently; failed siblings remain pending.
    Returns (rows saved, suspect-price rows, transient failures)."""
    frames: list[pd.DataFrame] = []
    held_periods: list[pd.Timestamp] = []
    for stamp, period in _pending_filings(context, cik, _listed_filings(cik, since)):
        rows = read_filing_vintage(stamp)
        if isinstance(rows, ReadFailure):
            _log_read_failure(context, cik, stamp, period, rows)
        else:
            manifest, book = rows
            save_13f_filing(context.store, manifest, book)
            if not book.empty:
                frames.append(book)
        if isinstance(rows, ReadFailure) and rows.transient:
            held_periods.append(period)
    if not frames:
        return 0, 0, len(held_periods)
    book = pd.concat(frames, ignore_index=True)
    return len(book), _suspect_prices(book), len(held_periods)


def _warn_empty_books(context: Context, empty: list[str], n_ciks: int) -> None:
    """Warn on book-less roster CIKs that produced no rows, naming those with `sec13f_hr`
    history: those were throttled or failed transiently and must be re-run; the rest never filed."""
    if not empty:
        return
    known_filers = set(context.store.distinct(Tables.sec13f_hr, "cik", where={"cik": [c.lstrip("0") for c in empty] + empty}))
    known = {c for c in empty if c in known_filers or c.lstrip("0") in known_filers}
    context.log.warning(
        "13F managers: %d/%d roster CIK(s) with no stored book produced NO rows; %d of them have "
        "history in %s and must be re-run (throttling or a transient listing failure, not an "
        "absent filer): %s",
        len(empty),
        n_ciks,
        len(known),
        Tables.sec13f_hr,
        ", ".join(sorted(known)) or "none",
    )


def fetch_13f_managers(context: Context, years_history: int = 15) -> int:
    """Catch every roster CIK's book up with every listed filing it does not show yet; returns rows
    saved. `years_history` bounds by PERIOD. A failed CIK counts as zero rows. Raises
    `SuperinvestorRosterEmptyError` when the roster holds no CIK."""
    context.ensure_edgar_identity()
    configure(context)
    ciks = sorted(roster_cik_union(context))
    if not ciks:
        raise SuperinvestorRosterEmptyError(
            "superinvestor_roster is empty -- run `data_extract superinvestors -F` first. "
            "The roster is this walk's entire scope; there is nothing to fetch without it."
        )
    names = roster_map_as_of(context)  # latest snapshot; only used for log lines
    with_book = set(context.store.distinct(Tables.sec13f_manager_holdings, "cik"))
    since = pd.Timestamp.today().normalize() - pd.DateOffset(years=years_history)
    logger.info("13F managers: %d roster CIK(s), %d with a stored book, periods from %s", len(ciks), len(with_book & set(ciks)), since.date())

    worker = partial(_catch_up_cik, context=context, since=since)
    df_scope = pd.DataFrame({"cik_label": [names.get(c, c) for c in ciks], "cik": ciks})
    guarded = run_per_ticker(df_scope, worker, desc="13F manager books", log=context.log, key_cols=("cik_label", "cik"))
    results = [result or (0, 0, 0) for result in guarded]
    saved = sum(n for n, _, _ in results)
    suspect = sum(s for _, s, _ in results)
    failed = [(c, f) for c, (_, _, f) in zip(ciks, results, strict=True) if f]

    _warn_empty_books(context, [c for c, (n, _, _) in zip(ciks, results, strict=True) if n == 0 and c not in with_book], len(ciks))
    if failed:
        logger.warning(
            "13F managers: %d filing read(s) failed transiently across %d CIK(s); those accessions are retried next run: %s",
            sum(f for _, f in failed),
            len(failed),
            ", ".join(c for c, _ in failed),
        )
    if suspect:
        logger.warning(
            "13F managers: %d/%d saved rows imply a share price outside %s -- check "
            "source units/positions; aggregate P1 validates against report-date quotes",
            suspect,
            saved,
            _IMPLIED_PRICE_BAND,
        )
    logger.info("13F managers: saved %d row(s) across %d manager(s) -> %s", saved, len(ciks), Tables.sec13f_manager_holdings)
    return saved


def pending_vintages(context: Context, cik: str, accessions: list[str]) -> list[str]:
    """A later restatement never marks an unpreserved original accession complete."""
    done = stored_accessions(context, Tables.sec13f_manager_holdings, [cik], DONE_PER_KEY, where={"parse_status": "complete"})[cik]
    return [accession for accession in accessions if accession not in done]


def recover_13f_vintages(
    context: Context,
    *,
    ciks: list[str],
    start_period: str,
    end_period: str,
    filed_through: str,
    accessions: list[str],
    max_filings: int,
    dry_run: bool = True,
    archives: list[Path] | None = None,
) -> dict[str, object]:
    """Bounded source recovery into the two existing tables; legacy rows cannot certify an accession."""
    if not ciks or not accessions or max_filings < 1:
        raise ValueError("Version recovery requires explicit CIKs, accessions and a positive filing cap")
    ciks = [pad_cik(cik) for cik in ciks]
    first, last, through = (pd.Timestamp(value) for value in (start_period, end_period, filed_through))
    if first > last:
        raise ValueError("Recovery report-period bounds are inverted")
    selected: list[dict[str, object]] = []
    stamps: dict[str, FilingStamp] = {}
    paths: dict[str, Path] = {}
    if archives:
        for path in archives:
            tables = read_zip_tables(
                path, {"SUBMISSION.tsv": ZipRead(upper=True), "COVERPAGE.tsv": ZipRead(upper=True)}, on_corrupt="skip", log=context.log
            )
            if not tables:
                raise ValueError(f"Unverifiable recovery metadata: {path}")
            for row in _submissions(tables["SUBMISSION.tsv"], tables["COVERPAGE.tsv"]).itertuples():
                if row.accession in accessions and row.cik in ciks and first <= row.report_period <= last and row.filing_date <= through:
                    selected.append(
                        {
                            "accession": row.accession,
                            "cik": row.cik,
                            "period": str(row.report_period.date()),
                            "filed": str(row.filing_date.date()),
                            "source": path.name,
                        }
                    )
                    paths[row.accession] = path
    else:
        configure(context)
        for cik in ciks:
            for stamp, period in _listed_filings(cik, first):
                if stamp.accession_number in accessions and period <= last and stamp.filed <= through:
                    selected.append(
                        {
                            "accession": stamp.accession_number,
                            "cik": cik,
                            "period": str(period.date()),
                            "filed": str(stamp.filed.date()),
                            "source": stamp.doc_url,
                        }
                    )
                    stamps[stamp.accession_number] = stamp
    if len({row["accession"] for row in selected}) != len(selected):
        raise ValueError("Recovery sources repeat an accession; select one source per accession")
    if {row["accession"] for row in selected} != set(accessions):
        raise ValueError("Some requested accessions are missing or outside the explicit recovery scope")
    done = {cik: set(accessions) - set(pending_vintages(context, cik, accessions)) for cik in ciks}
    broad_done = stored_accessions(context, Tables.sec13f_hr, ciks, DONE_PER_KEY, where={"parse_status": "complete"})
    pending = [row for row in selected if row["accession"] not in (done[str(row["cik"])] & broad_done[str(row["cik"])])]
    if len(pending) > max_filings:
        raise ValueError(f"Recovery has {len(pending)} pending accessions, exceeding cap {max_filings}")
    byte_count = None
    if archives:
        byte_count = 0
        for path in set(paths[str(row["accession"])] for row in pending):
            with ZipFile(path) as archive:
                byte_count += sum(member.file_size for member in archive.infolist() if member.filename.upper().endswith("INFOTABLE.TSV"))
    preview: dict[str, object] = {
        "scope": selected,
        "pending_accessions": len(pending),
        "completed_accessions": len(selected) - len(pending),
        "archive_scan_bytes": byte_count,
        "expected_document_downloads": 0 if archives else 2 * len(pending),
        "dry_run": dry_run,
        "saved_rows": 0,
    }
    context.log.info("13F version recovery preview: %s", json.dumps(preview, sort_keys=True))
    if dry_run:
        return preview
    saved = 0
    cached_packets: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    if archives:
        for path in sorted(set(paths[str(row["accession"])] for row in pending)):
            wanted = [str(row["accession"]) for row in pending if paths[str(row["accession"])] == path]
            for manifest, holdings in cached_filing_vintages(path, wanted):
                cached_packets[str(manifest.iloc[0].accession_number)] = (manifest, holdings)
    for row in sorted(pending, key=lambda item: (str(item["filed"]), str(item["accession"]))):
        accession = str(row["accession"])
        # Each accession commits separately; a failed batch resumes its remaining accession gaps.
        packet = cached_packets[accession] if archives else read_filing_vintage(stamps[accession])
        if isinstance(packet, ReadFailure):
            raise RuntimeError(f"13F recovery stopped at {accession}: {packet.reason}")
        manifest, holdings = packet
        saved += save_13f_filing(context.store, manifest, holdings)
        cmap = context.store.load(Tables.cusip_ticker_map, columns=["cusip", "ticker"], optional=True)
        broad = holdings.assign(ticker=Tables.sec13f_hr.empty_marker[1])
        if cmap is not None:
            mapping = cmap.drop_duplicates("cusip").set_index("cusip").ticker
            broad["ticker"] = broad.cusip.map(mapping).fillna(Tables.sec13f_hr.empty_marker[1])
        save_13f_filing(context.store, manifest, broad, table=Tables.sec13f_hr)
    preview["saved_rows"] = saved
    return preview
