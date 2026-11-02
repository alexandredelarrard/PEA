"""Pension facts from the SEC Financial Statement Data Sets (quarterly bulk zips of primary-statement XBRL).

Each quarter's `sub.txt` + `num.txt` is cached locally; consolidated rows (no `segments` member, no `coreg`) for
the curated pension tags are joined to `sub` for cik / form / filed, mapped to the universe ticker whose lineage window
of the filer CIK holds the filing date (consolidating) and upserted to
`pension_facts`, one row per (cik, tag, ddate, qtrs, quarter), latest filed wins, stamped with the archive's
point-in-time `available_at`. The quarters parsed come from `resume.archive_worklist`. Footnote pension detail
comes from `fetch_financial_notes.py`.
"""

from __future__ import annotations

import logging
from collections.abc import Collection
from datetime import date
from pathlib import Path
from typing import cast

import pandas as pd
from tqdm import tqdm

from src.context import Context
from src.data_extract.utils.common.bulk_cache import (
    DEFAULT_TIMEOUT,
    ZipRead,
    cache_dir,
    cached_periods,
    is_cached,
    period_end,
    quarter_periods,
    read_zip_tables,
    stored_period_clock,
)
from src.data_extract.utils.common.identity import Identity, load_identity, tickers_for_ciks
from src.data_extract.utils.common.resume import archive_periods, archive_worklist
from src.data_extract.utils.zips.zip_archive import ArchiveSource, ensure_archive, reparse_clock
from src.data_store.schema import Resume, Tables
from src.utils.string import pad_cik_series, run_date

logger = logging.getLogger(__name__)

_CHUNK = 500_000

# Curated defined-benefit tags: the recognized net liability first, then coverage/detail variants.
_PENSION_TAGS = frozenset(
    {
        "PensionAndOtherPostretirementDefinedBenefitPlansLiabilitiesNoncurrent",
        "PensionAndOtherPostretirementDefinedBenefitPlansLiabilitiesCurrent",
        "PensionAndOtherPostretirementDefinedBenefitPlansLiabilities",
        "DefinedBenefitPensionPlanLiabilitiesNoncurrent",
        "LiabilityPensionAndOtherPostretirementAndPostemploymentBenefitPlansNoncurrent",
        "DefinedBenefitPlanFundedStatusOfPlanAmount",
        "DefinedBenefitPlanBenefitObligation",
        "DefinedBenefitPlanFairValueOfPlanAssets",
        "DefinedBenefitPlanAccumulatedBenefitObligation",
    }
)
_ZIP_SPECS = {
    "sub.txt": ZipRead(usecols=frozenset({"adsh", "cik", "name", "form", "period", "fy", "fp", "filed"})),
    "num.txt": ZipRead(keep=lambda chunk: chunk["tag"].isin(_PENSION_TAGS), chunksize=_CHUNK),
}
_OUT_COLS = ["cik", "ticker", "tag", "ddate", "qtrs", "uom", "value", "adsh", "filed", "form", "fy", "fp", "quarter", "available_at"]

SEC_FINSTMT_URL_TEMPLATE = "https://www.sec.gov/files/dera/data/financial-statement-data-sets/{quarter}.zip"
STATEMENTS_ARCHIVE = ArchiveSource(
    label="finstmt",
    file_template="{period}.zip",
    url_template=SEC_FINSTMT_URL_TEMPLATE,
    observed_from=date.fromisoformat(cast(str, cast(Resume, Tables.pension_facts.resume).observed_from)),
    timeout=DEFAULT_TIMEOUT,
)


# --------------------------------------------------------------------------- #
# Pure parse (unit-tested)                                                       #
# --------------------------------------------------------------------------- #
def _join_pension(num: pd.DataFrame, sub: pd.DataFrame) -> pd.DataFrame:
    """Filtered num rows (pension tags) + sub -> tidy company-level facts. Pure."""
    if num is None or num.empty or sub is None or sub.empty:
        return pd.DataFrame()
    seg = num.get("segments", pd.Series("", index=num.index)).astype("string").fillna("").str.strip()
    coreg = num.get("coreg", pd.Series("", index=num.index)).astype("string").fillna("").str.strip()
    n = pd.DataFrame(
        {
            "adsh": num["adsh"],
            "tag": num["tag"],
            "ddate": pd.to_datetime(num["ddate"], format="%Y%m%d", errors="coerce"),
            "qtrs": pd.to_numeric(num["qtrs"], errors="coerce"),
            "uom": num["uom"],
            "value": pd.to_numeric(num["value"], errors="coerce"),
        }
    )
    # pension tags (re-checked so the join stays pure), consolidated parent-company facts only
    n = n[n["tag"].isin(_PENSION_TAGS) & (seg == "") & (coreg == "")].dropna(subset=["value", "ddate"])
    if n.empty:
        return pd.DataFrame()
    s = pd.DataFrame(
        {
            "adsh": sub["adsh"],
            "cik": pad_cik_series(sub["cik"]).astype("string"),
            "form": sub.get("form"),
            "fy": sub.get("fy"),
            "fp": sub.get("fp"),
            "filed": pd.to_datetime(sub["filed"], format="%Y%m%d", errors="coerce"),
        }
    )
    return n.merge(s, on="adsh", how="inner")


# --------------------------------------------------------------------------- #
# IO: cache/download + incremental state                                        #
# --------------------------------------------------------------------------- #


def _read_pension_facts(path: Path) -> pd.DataFrame | None:
    """`sub.txt` joined to the pension rows of `num.txt`; None if the zip is corrupt (deleted) or lacks a member."""
    tables = read_zip_tables(path, _ZIP_SPECS, on_corrupt="delete", log=logger)
    if not tables:
        return None
    return _join_pension(tables["num.txt"], tables["sub.txt"])


def _save_pension_quarter(context: Context, identity: Identity, path: Path, quarter: str, available_at: date, tickers: list[str]) -> int:
    """One cached quarter's pension facts of `tickers`, latest-filed per fact, upserted; returns rows saved."""
    facts = _read_pension_facts(path)
    if facts is None or facts.empty:
        return 0
    facts["ticker"] = tickers_for_ciks(identity, facts["cik"], facts["filed"], "consolidating")
    facts = facts[facts["ticker"].isin(tickers)]
    if facts.empty:
        return 0
    # keep the latest-filed value per (cik, tag, period-end, duration)
    facts = facts.sort_values("filed").drop_duplicates(subset=["cik", "tag", "ddate", "qtrs"], keep="last")
    facts["quarter"] = quarter
    facts["available_at"] = available_at
    return context.store.save(Tables.pension_facts, facts[[c for c in _OUT_COLS if c in facts.columns]])


def statement_quarters(years_history: int, as_of: pd.Timestamp) -> list[str]:
    """The quarters of the `years_history` window ending `as_of`, from the table's source start."""
    first_year = pd.Timestamp(cast(str, cast(Resume, Tables.pension_facts.resume).source_start)).year
    return quarter_periods(years_history + 1, first_year, as_of)


def download_financial_statements(context: Context, cache: Path, quarters: list[str]) -> int:
    """Cache the zips of `quarters` no stored `pension_facts` row carries; returns the zips available. Writes no table."""
    listed, stored = archive_periods(context, (Tables.pension_facts,), quarters)
    return sum(ensure_archive(context, cache, STATEMENTS_ARCHIVE, q) is not None for q in listed if q not in stored)


def fetch_financial_statements(
    context: Context,
    tickers: list[str],
    years_history: int,
    reparse: bool = False,
    as_of: pd.Timestamp | None = None,
    *,
    all_tickers: Collection[str] | None = None,
) -> int:
    """Extract universe pension facts over `years_history` into `pension_facts`; returns rows upserted.

    Quarters missing from the table are parsed for every ticker, and cached quarters again for a new
    ticker with no row. `reparse` re-reads every stored quarter too (after a parse
    change). Cached quarters cost no network.
    """

    identity = load_identity(context)
    cache = cache_dir(context, context.config.local.paths.financial_statements)
    start_date = run_date(as_of)
    periods = statement_quarters(years_history, start_date)
    work = archive_worklist(
        context, (Tables.pension_facts,), periods, cached_periods(cache), tickers, start_date, full=reparse, all_tickers=all_tickers
    )

    saved = 0
    for q, keys in tqdm(list(work.units()), desc="financial-statement data sets"):
        found = ensure_archive(context, cache, STATEMENTS_ARCHIVE, q)
        if found is None:
            continue
        path, clock = found
        stored = stored_period_clock(context, (Tables.pension_facts,), q, column="quarter")
        available_at = clock if period_end(q) < STATEMENTS_ARCHIVE.observed_from else stored or clock
        if available_at is None:
            logger.warning("finstmt %s: archive clock unavailable -> leaving quarter un-ingested", q)
            continue
        saved += _save_pension_quarter(context, identity, path, q, available_at, list(keys))

    logger.info("pension_facts: upserted %d rows (%d pending + %d rescanned quarters)", saved, len(work.pending), len(work.rescan))
    return saved


def reparse_financial_statements(context: Context, tickers: list[str], *, identity: Identity, run_date: pd.Timestamp) -> int:
    """Re-read every cached statement zip up to `run_date`'s year for `tickers` only (a lineage expansion) through
    `identity`; returns rows upserted.

    No download: the next regular run keeps its own resume.
    """
    cache = cache_dir(context, context.config.local.paths.financial_statements)
    first_year = pd.Timestamp(cast(str, cast(Resume, Tables.pension_facts.resume).source_start)).year
    quarters = [q for q in quarter_periods(run_date.year - first_year + 1, first_year) if is_cached(cache / f"{q}.zip")]
    observed_from = date.fromisoformat(cast(str, cast(Resume, Tables.pension_facts.resume).observed_from))
    saved = 0
    for q in quarters:
        path = cache / f"{q}.zip"
        available_at = reparse_clock(context, (Tables.pension_facts,), q, path, observed_from, column="quarter")
        if available_at is None:
            logger.warning("finstmt %s: archive clock unavailable -> not re-parsed", q)
            continue
        saved += _save_pension_quarter(context, identity, path, q, available_at, tickers)
    logger.info("pension_facts: re-parsed %d cached quarter(s) for %d ticker(s), %d row(s) upserted", len(quarters), len(tickers), saved)
    return saved
