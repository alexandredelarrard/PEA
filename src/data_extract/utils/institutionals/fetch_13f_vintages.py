"""
fetch_13f_vintages.py (src/data_extract/utils/institutionals/fetch_13f_vintages.py)
------------------------------------------------------------------------------------
Manual, bounded recovery of named 13F accessions into `sec13f_hr` (`thirteen-f-vintages`): an original filing a later
amendment superseded before versions were kept, read from EDGAR or a cached 13F data-set archive. Complete source
packets are saved atomically; originals and amendments remain separate. Only a completed accession is skipped.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast
from zipfile import ZipFile

import pandas as pd

from src.context import Context
from src.data_extract.utils.common.bulk_cache import ZipRead, read_zip_tables
from src.data_extract.utils.common.resume import DONE_PER_KEY, stored_accessions
from src.data_extract.utils.common.sec_io import company, company_filings, configure
from src.data_extract.utils.edgar_extract.edgar_driver import FilingStamp
from src.data_extract.utils.institutionals.fetch_13f import (
    ReadFailure,
    read_filing_vintage,
    save_hr_filing,
)
from src.data_extract.utils.institutionals.fetch_13f_backfill import _submissions, cached_filing_vintages
from src.data_store.schema import Resume, Tables
from src.utils.identity_tables import load_cusip_map
from src.utils.string import pad_cik


def _listed_filings(cik: str, since: pd.Timestamp) -> list[tuple[FilingStamp, pd.Timestamp]]:
    """`(stamp, period)` for the CIK's 13F-HR filings whose period is on/after `since`, oldest
    first by (filed, amendment last, accession) -- edgartools lists newest first; a null or
    unparseable period is skipped."""
    listing = company_filings(company(cik), cast(Resume, Tables.sec13f_hr.resume).forms) or []
    stamps = sorted((FilingStamp.of(f, cik) for f in listing), key=lambda s: (s.filed, s.is_amendment, s.accession_number))
    periods = [pd.to_datetime(s.period_of_report, errors="coerce") for s in stamps]
    return [(s, period.normalize()) for s, period in zip(stamps, periods, strict=True) if pd.notna(period) and period >= since]


def pending_vintages(context: Context, cik: str, accessions: list[str]) -> list[str]:
    """A later restatement never marks an unpreserved original accession complete."""
    done = stored_accessions(context, Tables.sec13f_hr, [cik], DONE_PER_KEY, where={"parse_status": "complete"})[cik]
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
    """Bounded source recovery into `sec13f_hr`; legacy rows cannot certify an accession."""
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
            tables = read_zip_tables(path, {"SUBMISSION.tsv": ZipRead(upper=True), "COVERPAGE.tsv": ZipRead(upper=True)}, on_corrupt="skip")
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
    done = stored_accessions(context, Tables.sec13f_hr, ciks, DONE_PER_KEY, where={"parse_status": "complete"})
    pending = [row for row in selected if row["accession"] not in done[str(row["cik"])]]
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
    cmap = load_cusip_map(context.store)
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
        saved += save_hr_filing(context.store, manifest, holdings, cmap)
    preview["saved_rows"] = saved
    return preview
