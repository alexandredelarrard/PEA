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
    ZipRead,
    archive_available_at,
    cache_dir,
    cached_periods,
    ensure_zip,
    is_cached,
    period_end,
    quarter_periods,
    read_zip_tables,
    stored_period_clock,
)
from src.data_extract.utils.common.identity import Identity, load_identity, tickers_for_ciks
from src.data_extract.utils.common.resume import archive_worklist
from src.data_store.schema import Resume, Tables
from src.utils.string import pad_cik_series

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

    run_date = pd.Timestamp(as_of if as_of is not None else pd.Timestamp.today()).normalize()
    first_year = pd.Timestamp(cast(str, cast(Resume, Tables.pension_facts.resume).source_start)).year
    periods = quarter_periods(years_history + 1, first_year, run_date)
    observed_from = date.fromisoformat(cast(str, cast(Resume, Tables.pension_facts.resume).observed_from))
    work = archive_worklist(
        context, (Tables.pension_facts,), periods, cached_periods(cache), tickers, run_date, full=reparse, all_tickers=all_tickers
    )

    saved = 0
    for q, keys in tqdm(list(work.units()), desc="financial-statement data sets"):
        cached_path = cache / f"{q}.zip"
        downloaded = not is_cached(cached_path)
        path = ensure_zip(context, cached_path, SEC_FINSTMT_URL_TEMPLATE.format(quarter=q), label=f"finstmt {q}", log=logger)
        if path is None:
            continue
        stored = stored_period_clock(context, (Tables.pension_facts,), q, column="quarter")
        end = period_end(q)
        clock = archive_available_at(end, path, observed_from=observed_from, downloaded=downloaded)
        available_at = clock if end < observed_from else stored or clock
        if available_at is None:
            logger.warning("finstmt %s: archive clock unavailable -> leaving quarter un-ingested", q)
            continue
        saved += _save_pension_quarter(context, identity, path, q, available_at, list(keys))

    logger.info("pension_facts: upserted %d rows (%d pending + %d rescanned quarters)", saved, len(work.pending), len(work.rescan))
    return saved


def reparse_financial_statements(context: Context, tickers: list[str]) -> int:
    """Re-read every cached statement zip for `tickers` only (a lineage expansion); returns rows upserted.

    No download: the next regular run keeps its own resume.
    """
    identity = load_identity(context)
    cache = cache_dir(context, context.config.local.paths.financial_statements)
    first_year = pd.Timestamp(cast(str, cast(Resume, Tables.pension_facts.resume).source_start)).year
    quarters = [q for q in quarter_periods(pd.Timestamp.today().year - first_year + 1, first_year) if is_cached(cache / f"{q}.zip")]
    observed_from = date.fromisoformat(cast(str, cast(Resume, Tables.pension_facts.resume).observed_from))
    saved = 0
    for q in quarters:
        path = cache / f"{q}.zip"
        available_at = stored_period_clock(context, (Tables.pension_facts,), q, column="quarter") or archive_available_at(
            period_end(q), path, observed_from=observed_from, downloaded=False
        )
        if available_at is None:
            logger.warning("finstmt %s: archive clock unavailable -> not re-parsed", q)
            continue
        saved += _save_pension_quarter(context, identity, path, q, available_at, tickers)
    logger.info("pension_facts: re-parsed %d cached quarter(s) for %d ticker(s), %d row(s) upserted", len(quarters), len(tickers), saved)
    return saved
