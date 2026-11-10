"""SEC Financial Statement and Notes data sets -> `notes_num` (footnote pension numerics) and `notes_text` (note prose).

Each period zip (`.tsv` members: sub, num, txt) is cached locally and parsed for universe filers, keeping only
curated tags on consolidated facts (`dimn == 0`, no `coreg`); text is stored raw, no NLP. A filing belongs to the
ticker whose lineage window of its filer CIK holds the filing date (consolidating). Rows carry their archive
`period` and point-in-time `available_at` clock. A period counts as done only when both tables hold it; the work
list comes from `resume.archive_worklist`. Zips are large, so the window is `years_history + 1` years (`StepExtractZips`).
`download_notes_periods` caches the zips whose cover-page `dei:TradingSymbol` facts the identity step has not
captured yet (`utils/identity/evidence.py`); the notes parse reads its zips itself.
"""

from __future__ import annotations

import logging
import re
from datetime import date
from functools import partial
from pathlib import Path
from typing import cast

import pandas as pd
from tqdm import tqdm

from src.constants.constants import SEC_NET_PENSION_LIABILITY_TAGS
from src.context import Context
from src.data_extract.utils.common.bulk_cache import (
    ZipRead,
    archive_available_at,
    cache_dir,
    cached_periods,
    ensure_zip,
    quarter_periods,
    read_zip_tables,
    stored_period_clock,
)
from src.data_extract.utils.common.identity import Identity, load_identity, tickers_for_ciks
from src.data_extract.utils.common.incremental import stored_values
from src.data_extract.utils.common.resume import archive_worklist, period_end
from src.data_extract.utils.common.sec_io import sec_get
from src.data_extract.utils.identity.symbol_tenure import captured_dei_periods
from src.data_extract.utils.zips.zip_archive import ArchiveSource, ZipScope, ensure_archive, reparse_clock
from src.data_store.schema import Resume, Table, Tables
from src.utils.string import pad_cik_series, run_date

logger = logging.getLogger(__name__)

CHUNK_ROWS = 500_000
_LANDING_URL = "https://www.sec.gov/data-research/sec-markets-data/financial-statement-notes-data-sets"
_NOTES_TABLES = (Tables.notes_num, Tables.notes_text)

# Curated footnote pension tags (undimensioned totals); discount-rate tags are percentages, not USD. The two recognized
# net-liability tags (`SEC_NET_PENSION_LIABILITY_TAGS`) are kept only on the filing's balance sheet (`pre.tsv` stmt BS): a footnote
# line (OPEB-only, a sub-plan) carries the same undimensioned tag with another fact.
_NOTES_NUM_TAGS = frozenset(
    {
        "DefinedBenefitPlanBenefitObligation",  # PBO
        "DefinedBenefitPlanFairValueOfPlanAssets",  # plan assets (FV)
        "DefinedBenefitPlanAccumulatedBenefitObligation",  # ABO
        "DefinedBenefitPlanFundedStatusOfPlanAmount",  # funded status (rare; else computed)
        "DefinedBenefitPlanNetPeriodicBenefitCost",  # net periodic cost
        "DefinedBenefitPlanServiceCost",  # service cost (operating piece)
        "DefinedBenefitPlanInterestCost",  # interest cost
        "DefinedBenefitPlanExpectedReturnOnPlanAssets",  # expected return on assets
        "DefinedBenefitPlanContributionsByEmployer",  # employer cash contributions
        "DefinedBenefitPlanExpectedFutureBenefitPaymentsNextTwelveMonths",  # near-term cash outflow
        "DefinedBenefitPlanAssumptionsUsedCalculatingNetPeriodicBenefitCostDiscountRate",
        "DefinedBenefitPlanWeightedAverageAssumptionsUsedCalculatingBenefitObligationDiscountRate",
    }
) | frozenset(SEC_NET_PENSION_LIABILITY_TAGS)

# High-signal note TextBlock elements, a few variants per theme.
_NOTES_TEXT_TAGS = frozenset(
    {
        # pension / retirement
        "PensionAndOtherPostretirementBenefitPlansFullDisclosureTextBlock",
        "DefinedBenefitPlanDisclosureTextBlock",
        "CompensationAndEmployeeBenefitPlansTextBlock",
        # revenue recognition
        "RevenueFromContractWithCustomerTextBlock",
        "RevenueRecognitionPolicyTextBlock",
        "RevenueRecognitionTextBlock",
        # commitments / litigation
        "CommitmentsAndContingenciesDisclosureTextBlock",
        "LegalMattersAndContingenciesTextBlock",
        # segment
        "SegmentReportingDisclosureTextBlock",
        # risk / going concern
        "SubstantialDoubtAboutGoingConcernTextBlock",
        "ConcentrationRiskDisclosureTextBlock",
        # critical accounting estimates / significant policies
        "SignificantAccountingPoliciesTextBlock",
        "OrganizationConsolidationAndPresentationOfFinancialStatementsDisclosureAndSignificantAccountingPoliciesTextBlock",
        "UseOfEstimates",
    }
)

_SUB_USECOLS = frozenset({"adsh", "cik", "form", "fy", "fp", "filed"})
_NUM_USECOLS = frozenset({"adsh", "tag", "ddate", "qtrs", "uom", "dimn", "coreg", "footnote", "value"})
_TXT_USECOLS = frozenset({"adsh", "tag", "ddate", "qtrs", "dimn", "coreg", "escaped", "txtlen", "footnote", "value"})
_PRE_USECOLS = frozenset({"adsh", "tag", "stmt"})
_FACT_PK = ["adsh", "tag", "ddate", "qtrs"]
_NOTES_ZIP_NAME = re.compile(r"(\d{4}(?:q[1-4]|_\d{2}))_notes\.zip")
_NUM_OUT = ["cik", "ticker", "adsh", "tag", "ddate", "qtrs", "uom", "value", "footnote", "form", "fy", "fp", "filed", "period", "available_at"]
_TXT_OUT = [
    "cik",
    "ticker",
    "adsh",
    "tag",
    "ddate",
    "qtrs",
    "txtlen",
    "escaped",
    "value",
    "footnote",
    "form",
    "fy",
    "fp",
    "filed",
    "period",
    "available_at",
]

SEC_FINNOTES_URL_TEMPLATE = "https://www.sec.gov/files/dera/data/financial-statement-notes-data-sets/{period}_notes.zip"
NOTES_ARCHIVE = ArchiveSource(
    label="notes",
    file_template="{period}_notes.zip",
    url_template=SEC_FINNOTES_URL_TEMPLATE,
    observed_from=date.fromisoformat(cast(str, cast(Resume, Tables.notes_num.resume).observed_from)),
    timeout=600,
)


# --------------------------------------------------------------------------- #
# Period list (rolling quarterly <-> monthly)                                   #
# --------------------------------------------------------------------------- #
def _period_year(tag: str) -> int:
    """'2024q1' -> 2024 ; '2025_07' -> 2025."""
    return int(tag[:4])


def _scrape_available_periods(context: Context) -> list[str] | None:
    """Available period tags scraped from the SEC landing page; None on any failure (caller falls back to the generator)."""
    try:
        r = sec_get(context, _LANDING_URL, timeout=60)
    except Exception:  # noqa: BLE001 (best-effort)
        return None
    tags = re.findall(r"/(\d{4}(?:q[1-4]|_\d{2}))_notes\.zip", r.text)
    return sorted(set(tags)) or None


def _generate_periods(years_history: int, today: pd.Timestamp | None = None) -> list[str]:
    """Candidate tags: quarterly over the window plus the last 14 months as monthly (404s are skipped downstream)."""
    today = run_date(today)
    first_year = min(pd.Timestamp(cast(str, cast(Resume, table.resume).source_start)).year for table in _NOTES_TABLES)
    quarterly = quarter_periods(years_history, first_year, today)
    monthly = [f"{month.year}_{month.month:02d}" for month in pd.period_range(end=today, periods=14, freq="M")]
    return sorted(set(quarterly + monthly))


def notes_periods(context: Context, years_history: int, today: pd.Timestamp | None = None) -> list[str]:
    """Period tags to fetch, newest last, filtered to the year window."""
    today = run_date(today)
    first_year = min(pd.Timestamp(cast(str, cast(Resume, table.resume).source_start)).year for table in _NOTES_TABLES)
    start_year = max(first_year, today.year - years_history)
    avail = _scrape_available_periods(context)
    tags = avail if avail is not None else _generate_periods(years_history, today)
    return sorted(t for t in tags if _period_year(t) >= start_year)


# --------------------------------------------------------------------------- #
# Pure parse (unit-tested)                                                       #
# --------------------------------------------------------------------------- #
def _sub_meta(sub: pd.DataFrame, identity: Identity, universe: set[str]) -> pd.DataFrame:
    """sub.tsv -> [adsh, cik, ticker, form, fy, fp, filed] for filings a universe ticker's lineage window holds. Pure."""
    if sub is None or sub.empty:
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
    s["ticker"] = tickers_for_ciks(identity, s["cik"], s["filed"], "consolidating")
    return s[s["ticker"].isin(universe)]


def _join_notes_num(num: pd.DataFrame, sub_meta: pd.DataFrame) -> pd.DataFrame:
    """Filtered num rows (pension tags, dimn==0) + sub_meta -> tidy facts. Pure."""
    if num is None or num.empty or sub_meta is None or sub_meta.empty:
        return pd.DataFrame()
    n = pd.DataFrame(
        {
            "adsh": num["adsh"],
            "tag": num["tag"],
            "ddate": pd.to_datetime(num["ddate"], format="%Y%m%d", errors="coerce"),
            "qtrs": pd.to_numeric(num["qtrs"], errors="coerce"),
            "uom": num.get("uom"),
            "value": pd.to_numeric(num["value"], errors="coerce"),
            "footnote": num.get("footnote"),
        }
    ).dropna(subset=["value", "ddate"])
    return n.merge(sub_meta, on="adsh", how="inner") if not n.empty else pd.DataFrame()


def _join_notes_text(txt: pd.DataFrame, sub_meta: pd.DataFrame) -> pd.DataFrame:
    """Filtered txt rows (high-signal tags, dimn==0) + sub_meta -> tidy text. Pure."""
    if txt is None or txt.empty or sub_meta is None or sub_meta.empty:
        return pd.DataFrame()
    t = pd.DataFrame(
        {
            "adsh": txt["adsh"],
            "tag": txt["tag"],
            "ddate": pd.to_datetime(txt["ddate"], format="%Y%m%d", errors="coerce"),
            "qtrs": pd.to_numeric(txt["qtrs"], errors="coerce"),
            "txtlen": pd.to_numeric(txt.get("txtlen", pd.Series(pd.NA, index=txt.index)), errors="coerce"),
            "escaped": txt.get("escaped"),
            "value": txt["value"].astype("string"),
            "footnote": txt.get("footnote"),
        }
    ).dropna(subset=["value", "ddate"])
    t = t[t["value"].str.strip() != ""]
    return t.merge(sub_meta, on="adsh", how="inner") if not t.empty else pd.DataFrame()


# --------------------------------------------------------------------------- #
# IO: cache/download + incremental state                                        #
# --------------------------------------------------------------------------- #
def _periods_missing_available_at(context: Context) -> set[str]:
    """Stored archive periods with no availability clock in either destination."""
    missing: set[str] = set()
    for table in _NOTES_TABLES:
        columns = set(context.store.columns(table))
        if "period" not in columns:
            continue
        where = {"available_at": None} if "available_at" in columns else None
        missing.update(str(value) for value in context.store.distinct(table, "period", where=where))
    return missing


def _repair_period_available_at(context: Context, period: str, available_at: date, *, overwrite: bool = False) -> int:
    """Patch clocks using only each table's PK plus available_at."""
    repaired = 0
    for table in _NOTES_TABLES:
        columns = set(context.store.columns(table))
        if "period" not in columns:
            continue
        where: dict[str, object] = {"period": period}
        if "available_at" in columns and not overwrite:
            where["available_at"] = None
        rows = context.store.load(
            table,
            columns=[*table.pk, "period"],
            where=where,
            optional=True,
        )
        if rows is None or rows.empty:
            continue
        patch = rows[list(table.pk)].drop_duplicates().copy()
        patch["available_at"] = available_at
        repaired += context.store.save(table, patch)
    return repaired


def _consolidated_rows(chunk: pd.DataFrame, adsh_set: set[str], tags: frozenset[str]) -> pd.Series:
    """Rows of universe filings with a curated tag, undimensioned (dimn==0) and no co-registrant."""
    dimn = pd.to_numeric(chunk.get("dimn", pd.Series(pd.NA, index=chunk.index)), errors="coerce")
    coreg = chunk.get("coreg", pd.Series("", index=chunk.index)).astype("string").fillna("").str.strip()
    return chunk["adsh"].isin(adsh_set) & chunk["tag"].isin(tags) & (dimn == 0) & (coreg == "")


def _balance_sheet_rows(chunk: pd.DataFrame, adsh_set: set[str]) -> pd.Series:
    """`pre.tsv` rows presenting a net-liability tag of a universe filing on the balance sheet (stmt BS)."""
    return chunk["adsh"].isin(adsh_set) & chunk["tag"].isin(SEC_NET_PENSION_LIABILITY_TAGS) & (chunk["stmt"] == "BS")


def _on_balance_sheet(num: pd.DataFrame, pre: pd.DataFrame, zip_name: str) -> pd.DataFrame:
    """`num` without the net-liability rows the filing does not present on its balance sheet.

    A footnote line (OPEB-only, a sub-plan) carries the same undimensioned tag with a different fact, so a
    net-liability row survives only when `pre` lists its (adsh, tag) with stmt BS. A zip without `pre.tsv`
    keeps no net-liability row (warned), never an unchecked one. Pure besides the warning.
    """
    if num.empty:
        return num
    net = num["tag"].isin(SEC_NET_PENSION_LIABILITY_TAGS)
    if not net.any():
        return num
    if "adsh" not in pre.columns:
        logger.warning("%s: no pre.tsv, dropping the %d net-liability rows (balance-sheet presentation unknown)", zip_name, net.sum())
        return num[~net].reset_index(drop=True)
    shown = pd.MultiIndex.from_frame(pre[["adsh", "tag"]].drop_duplicates())
    keep = ~net | pd.MultiIndex.from_frame(num[["adsh", "tag"]]).isin(shown)
    return num[keep].reset_index(drop=True)


def _read_notes(path: Path, identity: Identity, universe: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One notes zip -> `(num facts, text)` for universe filings; num/txt stream in chunks keeping consolidated curated rows.

    The net-liability tags need a balance-sheet presentation in `pre.tsv` (`_on_balance_sheet`).
    """
    empty = (pd.DataFrame(), pd.DataFrame())
    subs = read_zip_tables(path, {"sub.tsv": ZipRead(usecols=_SUB_USECOLS)}, on_corrupt="delete")
    if not subs:
        return empty
    sub_meta = _sub_meta(subs["sub.tsv"], identity, universe)
    if sub_meta.empty:
        return empty
    adsh_set = set(sub_meta["adsh"])
    specs = {
        member: ZipRead(usecols=usecols, keep=partial(_consolidated_rows, adsh_set=adsh_set, tags=tags), chunksize=CHUNK_ROWS, skip_bad_lines=True)
        for member, usecols, tags in (("num.tsv", _NUM_USECOLS, _NOTES_NUM_TAGS), ("txt.tsv", _TXT_USECOLS, _NOTES_TEXT_TAGS))
    }
    specs["pre.tsv"] = ZipRead(
        usecols=_PRE_USECOLS, keep=partial(_balance_sheet_rows, adsh_set=adsh_set), chunksize=CHUNK_ROWS, required=False, skip_bad_lines=True
    )
    tables = read_zip_tables(path, specs, on_corrupt="delete")
    if not tables:
        return empty
    num = _on_balance_sheet(tables["num.tsv"], tables["pre.tsv"], path.name)
    return _join_notes_num(num, sub_meta), _join_notes_text(tables["txt.tsv"], sub_meta)


def _repair_stored_clocks(context: Context, cache: Path, overwrite: bool) -> dict[str, date]:
    """Stamp `available_at` on stored periods lacking it (all stored periods if `overwrite`); returns the clock per period."""
    periods = _periods_missing_available_at(context)
    if overwrite:
        periods |= stored_values(context, _NOTES_TABLES, "period")
    clocks: dict[str, date] = {}
    observed_from = date.fromisoformat(cast(str, cast(Resume, Tables.notes_num.resume).observed_from))
    for period in sorted(periods):
        end = period_end(period)
        inferred = archive_available_at(end, cache / f"{period}_notes.zip", observed_from=observed_from, downloaded=False)
        available_at = inferred if end < observed_from or overwrite else stored_period_clock(context, _NOTES_TABLES, period) or inferred
        if available_at is None:
            logger.warning("notes %s: no archive availability metadata -> leaving rows unavailable", period)
            continue
        repaired = _repair_period_available_at(context, period, available_at, overwrite=overwrite)
        clocks[period] = available_at
        logger.info("notes %s: repaired available_at=%s on %d stored rows", period, available_at, repaired)
    return clocks


def _save_period(context: Context, table: Table, frame: pd.DataFrame, columns: list[str], period: str, available_at: date) -> int:
    """Latest-filed row per fact, stamped with its archive period and clock, upserted to `table`."""
    if frame.empty:
        return 0
    frame = frame.sort_values("filed").drop_duplicates(subset=_FACT_PK, keep="last")
    frame["period"] = period
    frame["available_at"] = available_at
    return context.store.save(table, frame[[c for c in columns if c in frame.columns]])


# --------------------------------------------------------------------------- #
# Download (the cover-page capture is `utils/identity/evidence.py`)          #
# --------------------------------------------------------------------------- #
def cached_notes_periods(cache: Path) -> list[str]:
    """Period tags of the notes zips cached under `cache`, oldest first."""
    return sorted(match.group(1) for path in cache.glob("*_notes.zip") if (match := _NOTES_ZIP_NAME.fullmatch(path.name)))


def download_notes_periods(context: Context, cache: Path, periods: list[str]) -> int:
    """Cache the notes zips of `periods` whose cover-page symbols are not captured yet; returns the zips available.

    Resume reads the `dei` periods stored in `symbol_tenure` (no marker file). Writes no table.
    """
    captured = captured_dei_periods(context)
    pending = [period for period in periods if period not in captured]
    return sum(
        ensure_zip(context, cache / f"{period}_notes.zip", SEC_FINNOTES_URL_TEMPLATE.format(period=period), label=f"notes {period}", timeout=600)
        is not None
        for period in pending
    )


def repair_notes_availability(context: Context, cache: Path) -> int:
    """Re-stamp every stored period's `available_at` clock; metadata only, never a download or parse. Returns 0."""
    _repair_stored_clocks(context, cache, overwrite=True)
    return 0


def ingest_financial_notes(context: Context, cache: Path, scope: ZipScope, periods: list[str]) -> int:
    """Parse the work list's notes periods into `notes_num` and `notes_text`; returns rows upserted.

    Periods missing from either table are parsed for every ticker, and cached periods again for a new ticker
    with no row. `scope.full` re-reads every stored period too (after a parse change).
    """
    period_clocks = _repair_stored_clocks(context, cache, overwrite=False)
    identity = load_identity(context)
    work = archive_worklist(
        context,
        _NOTES_TABLES,
        periods,
        cached_periods(cache, suffix="_notes.zip"),
        scope.tickers,
        scope.run_date,
        full=scope.full,
    )
    units = dict(work.units())
    n_num = n_txt = 0
    for period in tqdm(periods, desc="SEC financial-statement notes"):
        # every stored period validates its immutable clock, including those skipped below
        available_at = period_clocks.get(period) or stored_period_clock(context, _NOTES_TABLES, period)
        if period not in units:
            continue
        found = ensure_archive(context, cache, NOTES_ARCHIVE, period)
        if found is None:
            continue
        path, clock = found
        available_at = available_at or clock
        if available_at is None:
            logger.warning("notes %s: archive clock unavailable -> skipping rows until a later retry", period)
            continue
        num, txt = _read_notes(path, identity, set(units[period]))
        n_num += _save_period(context, Tables.notes_num, num, _NUM_OUT, period, available_at)
        n_txt += _save_period(context, Tables.notes_text, txt, _TXT_OUT, period, available_at)

    logger.info("notes: upserted %d num + %d text rows (%d pending + %d rescanned periods)", n_num, n_txt, len(work.pending), len(work.rescan))
    return n_num + n_txt


def reparse_financial_notes(context: Context, tickers: list[str], *, identity: Identity) -> int:
    """Re-read every cached notes zip for `tickers` only (a lineage expansion) through `identity`; returns rows upserted.

    No download: the next regular run keeps its own resume.
    """
    cache = cache_dir(context, context.config.local.paths.financial_notes)
    saved = 0
    observed_from = date.fromisoformat(cast(str, cast(Resume, Tables.notes_num.resume).observed_from))
    periods = cached_notes_periods(cache)
    for period in periods:
        path = cache / f"{period}_notes.zip"
        available_at = reparse_clock(context, _NOTES_TABLES, period, path, observed_from)
        if available_at is None:
            logger.warning("notes %s: archive clock unavailable -> not re-parsed", period)
            continue
        num, txt = _read_notes(path, identity, set(tickers))
        saved += _save_period(context, Tables.notes_num, num, _NUM_OUT, period, available_at)
        saved += _save_period(context, Tables.notes_text, txt, _TXT_OUT, period, available_at)
    logger.info("notes: re-parsed %d cached period(s) for %d ticker(s), %d row(s) upserted", len(periods), len(tickers), saved)
    return saved
