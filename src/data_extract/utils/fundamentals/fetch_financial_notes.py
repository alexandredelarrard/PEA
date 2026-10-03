"""SEC Financial Statement and Notes data sets -> `notes_num` (footnote pension numerics) and `notes_text` (note prose).

Each period zip (`.tsv` members: sub, num, txt) is cached locally and parsed for universe filers, keeping only
curated tags on consolidated facts (`dimn == 0`, no `coreg`); text is stored raw, no NLP. Rows carry their archive
`period` and point-in-time `available_at` clock. Incremental: a stored period is skipped unless the universe gained
tickers or `reparse` is set. Zips are large, so the window is the dedicated `notes_years_history` knob.
"""

from __future__ import annotations

import logging
import re
from datetime import date
from functools import partial
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.context import Context
from src.data_extract.utils.common.bulk_cache import (
    ZipRead,
    archive_available_at,
    cache_dir,
    ensure_zip,
    is_cached,
    mark_processed,
    pending_periods,
    period_end,
    quarter_periods,
    read_zip_tables,
    stored_period_clock,
)
from src.data_extract.utils.common.incremental import stored_values
from src.data_extract.utils.common.registrant import Registrant, drop_rows_outside_segment, load_registrants
from src.data_extract.utils.common.run_manifest import record_run
from src.data_extract.utils.common.sec_utils import cik_to_ticker, load_cik_mapping
from src.data_store.schema import Table, Tables
from src.utils.string import pad_cik_series

logger = logging.getLogger(__name__)

_CHUNK = 500_000
_LANDING_URL = "https://www.sec.gov/data-research/sec-markets-data/financial-statement-notes-data-sets"
# Archives covering September 2026 onward carry an observed availability clock, older ones an estimate.
_OBSERVED_FROM = date(2026, 9, 1)
_NOTES_TABLES = (Tables.notes_num, Tables.notes_text)

# Curated footnote pension tags (undimensioned totals); discount-rate tags are percentages, not USD.
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
)

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
_FACT_PK = ["adsh", "tag", "ddate", "qtrs"]
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
SEC_FINNOTES_FIRST_YEAR = 2009  # earliest notes data set (2009q1)


# --------------------------------------------------------------------------- #
# Period list (rolling quarterly <-> monthly)                                   #
# --------------------------------------------------------------------------- #
def _period_year(tag: str) -> int:
    """'2024q1' -> 2024 ; '2025_07' -> 2025."""
    return int(tag[:4])


def _scrape_available_periods(context: Context) -> list[str] | None:
    """Available period tags scraped from the SEC landing page; None on any failure (caller falls back to the generator)."""
    try:
        r = context.sec_session.get(_LANDING_URL, timeout=60)
        if r.status_code != 200:
            return None
        tags = re.findall(r"/(\d{4}(?:q[1-4]|_\d{2}))_notes\.zip", r.text)
        return sorted(set(tags)) or None
    except Exception:  # noqa: BLE001 (best-effort)
        return None


def _generate_periods(years_history: int, today: pd.Timestamp | None = None) -> list[str]:
    """Candidate tags: quarterly over the window plus the last 14 months as monthly (404s are skipped downstream)."""
    today = (today or pd.Timestamp.today()).normalize()
    quarterly = quarter_periods(years_history, SEC_FINNOTES_FIRST_YEAR, today)
    monthly = [f"{month.year}_{month.month:02d}" for month in pd.period_range(end=today, periods=14, freq="M")]
    return sorted(set(quarterly + monthly))


def _notes_periods(context: Context, years_history: int, today: pd.Timestamp | None = None) -> list[str]:
    """Period tags to fetch, newest last, filtered to the year window."""
    today = (today or pd.Timestamp.today()).normalize()
    start_year = max(SEC_FINNOTES_FIRST_YEAR, today.year - years_history)
    avail = _scrape_available_periods(context)
    tags = avail if avail is not None else _generate_periods(years_history, today)
    return sorted(t for t in tags if _period_year(t) >= start_year)


# --------------------------------------------------------------------------- #
# Pure parse (unit-tested)                                                       #
# --------------------------------------------------------------------------- #
def _sub_meta(sub: pd.DataFrame, cik2tkr: dict[str, str], universe: set[str], registrants: dict[str, Registrant]) -> pd.DataFrame:
    """sub.tsv -> [adsh, cik, ticker, form, fy, fp, filed] for UNIVERSE filers only. Pure."""
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
    s["ticker"] = s["cik"].map(cik2tkr)
    s = s[s["ticker"].isin(universe)]
    # consolidating table: a predecessor CIK's row must also fall in that CIK's dated segment (see `FORM_POLICY`)
    return drop_rows_outside_segment(s, cik_col="cik", ticker_col="ticker", filed_col="filed", registrants=registrants)


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


def _read_notes(path: Path, cik2tkr: dict[str, str], universe: set[str], registrants: dict[str, Registrant]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One notes zip -> `(num facts, text)` for universe filings; num/txt stream in chunks keeping consolidated curated rows."""
    empty = (pd.DataFrame(), pd.DataFrame())
    subs = read_zip_tables(path, {"sub.tsv": ZipRead(usecols=_SUB_USECOLS)}, on_corrupt="delete", log=logger)
    if not subs:
        return empty
    sub_meta = _sub_meta(subs["sub.tsv"], cik2tkr, universe, registrants)
    if sub_meta.empty:
        return empty
    adsh_set = set(sub_meta["adsh"])
    specs = {
        member: ZipRead(usecols=usecols, keep=partial(_consolidated_rows, adsh_set=adsh_set, tags=tags), chunksize=_CHUNK, skip_bad_lines=True)
        for member, usecols, tags in (("num.tsv", _NUM_USECOLS, _NOTES_NUM_TAGS), ("txt.tsv", _TXT_USECOLS, _NOTES_TEXT_TAGS))
    }
    tables = read_zip_tables(path, specs, on_corrupt="delete", log=logger)
    if not tables:
        return empty
    return _join_notes_num(tables["num.tsv"], sub_meta), _join_notes_text(tables["txt.tsv"], sub_meta)


def _repair_stored_clocks(context: Context, cache: Path, overwrite: bool) -> dict[str, date]:
    """Stamp `available_at` on stored periods lacking it (all stored periods if `overwrite`); returns the clock per period."""
    periods = _periods_missing_available_at(context)
    if overwrite:
        periods |= stored_values(context, _NOTES_TABLES, "period")
    clocks: dict[str, date] = {}
    for period in sorted(periods):
        end = period_end(period)
        inferred = archive_available_at(end, cache / f"{period}_notes.zip", observed_from=_OBSERVED_FROM, downloaded=False)
        available_at = inferred if end < _OBSERVED_FROM or overwrite else stored_period_clock(context, _NOTES_TABLES, period) or inferred
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


def fetch_financial_notes(
    context: Context,
    tickers: list[str],
    years_history: int = 15,
    reparse: bool = False,
    repair_availability: bool = False,
) -> int:
    """Fetch the notes data sets over `years_history` into `notes_num` and `notes_text`; returns rows upserted.

    A stored period is skipped unless the universe gained tickers. `reparse` re-reads the WHOLE cached window
    (needed after a registrant/CIK-resolution change; never a partial suffix). `repair_availability` only
    repairs stored `available_at` clocks and returns 0 before any download or parse.
    """

    cikmap = load_cik_mapping(context)
    cik2tkr = cik_to_ticker(cikmap, config_dir=str(context.config_dir))
    registrants = load_registrants(str(context.config_dir))
    cache = cache_dir(context, context.config.local.paths.financial_notes)

    period_clocks = _repair_stored_clocks(context, cache, overwrite=repair_availability)
    if repair_availability:
        return 0  # Metadata-only mode: never download or reparse a ZIP.

    periods = _notes_periods(context, years_history + 1)
    pending = set(pending_periods(context, cache, _NOTES_TABLES, periods, tickers, reparse=reparse))
    n_num = n_txt = 0
    for period in tqdm(periods, desc="SEC financial-statement notes"):
        # every stored period validates its immutable clock, including those skipped below
        available_at = period_clocks.get(period) or stored_period_clock(context, _NOTES_TABLES, period)
        if period not in pending:
            continue

        archive_path = cache / f"{period}_notes.zip"
        downloaded = not is_cached(archive_path)
        path = ensure_zip(context, archive_path, SEC_FINNOTES_URL_TEMPLATE.format(period=period), label=f"notes {period}", timeout=600, log=logger)
        if path is None:
            continue
        available_at = available_at or archive_available_at(period_end(period), path, observed_from=_OBSERVED_FROM, downloaded=downloaded)
        if available_at is None:
            logger.warning("notes %s: archive clock unavailable -> skipping rows until a later retry", period)
            continue

        num, txt = _read_notes(path, cik2tkr, set(tickers), registrants)
        n_num += _save_period(context, Tables.notes_num, num, _NUM_OUT, period, available_at)
        n_txt += _save_period(context, Tables.notes_text, txt, _TXT_OUT, period, available_at)

    mark_processed(cache, Tables.notes_num, tickers)
    logger.info("notes: upserted %d num + %d text rows (%d periods scanned)", n_num, n_txt, len(periods))
    record_run(context, Tables.notes_num, len(tickers), n_num)
    record_run(context, Tables.notes_text, len(tickers), n_txt)
    return n_num + n_txt
