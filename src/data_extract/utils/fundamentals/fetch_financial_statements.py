"""
fetch_financial_statements.py  (src/data_extract/utils/fundamentals/fetch_financial_statements.py)
-------------------------------------------------------------------------------------------------
Pension facts from the SEC "Financial Statement Data Sets" (free quarterly bulk
TSV zips of the flattened primary-statement XBRL). This is the PRIMARY source for
pensions: `companyfacts` only surfaces the tags a filer happens to expose, and the
Sharadar-first `fundamentals_history` carries no pension column at all, so the bulk
num/sub sets are what give the recognized net defined-benefit liability across the
universe. Measured: 6,244 rows over 125 tickers, median 17 years each. The footnote
funded status in `notes_num` (see `fetch_financial_notes.py`) is the second source
and the only other one; together they reach 199 of 489 tickers.

Each quarter's zip carries:
  * sub.txt   adsh -> cik, name, form, period, fy, fp, filed
  * num.txt   adsh, tag, version, ddate (period end), qtrs (0 = instant/balance
              sheet), uom, segments, coreg, value

We keep the CONSOLIDATED company-level rows (no dimensional `segments` member, no
`coreg`) for a curated set of pension tags, join to sub for cik / form / filed,
map to our tickers, and upsert to `pension_facts` (one row per company / tag /
period-end / duration / ZIP quarter). The tag list is easily extended. The footnote PBO / plan-asset
detail from the Financial Statement AND Notes sets is already wired -- separately, in
`fetch_financial_notes.py` (`notes_num` / `notes_text`).

"""

from __future__ import annotations

import logging
from datetime import date
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
from src.data_extract.utils.common.registrant import drop_rows_outside_segment, load_registrants
from src.data_extract.utils.common.run_manifest import record_run
from src.data_extract.utils.common.sec_utils import cik_to_ticker, load_cik_mapping
from src.data_store.schema import Tables
from src.utils.string import pad_cik_series

logger = logging.getLogger(__name__)

_CHUNK = 500_000

# Curated defined-benefit pension tags. The first is the recognized NET deficit
# (balance-sheet, the debt-like obligation that feeds the cube's pension overhang);
# the rest add coverage / detail where filers report them. Extend freely.
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
SEC_FINSTMT_FIRST_YEAR = 2009
# Archives covering September 2026 onward carry an observed availability clock, older ones an estimate.
_OBSERVED_FROM = date(2026, 9, 1)


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
    # pension tags only (defensive: real path pre-filters, but keep the join pure),
    # consolidated parent-company fact only (drop dimensional members / co-registrants)
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
    """sub.txt plus the pension-tag rows of num.txt (streamed in chunks), joined. None when the zip
    is corrupt (deleted for re-download) or lacks either member."""
    tables = read_zip_tables(path, _ZIP_SPECS, on_corrupt="delete", log=logger)
    if not tables:
        return None
    return _join_pension(tables["num.txt"], tables["sub.txt"])


def fetch_financial_statements(context: Context, tickers: list[str], years_history: int = 15, reparse: bool = False) -> int:
    """Download (cached) the Financial Statement Data Sets over `years_history`,
    extract pension facts for the universe, upsert to `pension_facts`. Returns the
    number of rows upserted.

    ⚠ `reparse` RE-READS EVERY CACHED PERIOD, and it exists because the incremental test
    cannot see a resolution change. That test is "did the ticker universe gain members?" --
    and a registrant-register change gains none: the same 491 tickers resolve through MORE
    CIKs. Without this flag the recovered predecessor rows would never be parsed.

    ⚠ A PARTIAL RE-PARSE IS WORSE THAN EITHER STATE ALONE. It leaves the oldest periods
    carrying the old resolution while the rest carry the new, and nothing downstream can tell
    that from a real coverage cliff. So this re-reads the whole window, not a suffix of it.

    Cached periods cost no network; a newly published quarter may still be downloaded.
    """

    cikmap = load_cik_mapping(context)
    cik2tkr = cik_to_ticker(cikmap, config_dir=str(context.config_dir))
    registrants = load_registrants(str(context.config_dir))
    cache = cache_dir(context, context.config.local.paths.financial_statements)

    periods = quarter_periods(years_history + 1, SEC_FINSTMT_FIRST_YEAR)
    pending = pending_periods(context, cache, Tables.pension_facts, periods, tickers, reparse=reparse, column="quarter")

    saved = 0
    for q in tqdm(pending, desc="financial-statement data sets"):
        cached_path = cache / f"{q}.zip"
        downloaded = not is_cached(cached_path)
        path = ensure_zip(context, cached_path, SEC_FINSTMT_URL_TEMPLATE.format(quarter=q), label=f"finstmt {q}", log=logger)
        if path is None:
            continue
        stored = stored_period_clock(context, (Tables.pension_facts,), q, column="quarter")
        end = period_end(q)
        clock = archive_available_at(end, path, observed_from=_OBSERVED_FROM, downloaded=downloaded)
        available_at = clock if end < _OBSERVED_FROM else stored or clock
        if available_at is None:
            logger.warning("finstmt %s: archive clock unavailable -> leaving quarter un-ingested", q)
            continue
        facts = _read_pension_facts(path)
        if facts is None or facts.empty:
            continue
        facts["ticker"] = facts["cik"].map(cik2tkr)
        facts = facts[facts["ticker"].isin(tickers)]
        # `pension_facts` is CONSOLIDATING: a predecessor CIK resolves to the ticker, but only
        # for the dates that registrant actually owned. See `FORM_POLICY`.
        facts = drop_rows_outside_segment(facts, cik_col="cik", ticker_col="ticker", filed_col="filed", registrants=registrants)
        if facts.empty:
            continue
        # keep the latest-filed value per (cik, tag, period-end, duration)
        facts = facts.sort_values("filed").drop_duplicates(subset=["cik", "tag", "ddate", "qtrs"], keep="last")
        facts["quarter"] = q
        facts["available_at"] = available_at
        saved += context.store.save(Tables.pension_facts, facts[[c for c in _OUT_COLS if c in facts.columns]])

    mark_processed(cache, Tables.pension_facts, tickers)
    logger.info("pension_facts: upserted %d rows (%d quarters scanned)", saved, len(periods))
    record_run(context, Tables.pension_facts, len(tickers), saved)
    return saved
