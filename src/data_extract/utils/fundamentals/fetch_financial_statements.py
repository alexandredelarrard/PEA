"""
fetch_financial_statements.py  (src/data_extract/utils/fundamentals/fetch_financial_statements.py)
-------------------------------------------------------------------------------------------------
Pension facts from the SEC "Financial Statement Data Sets" (free quarterly bulk
TSV zips of the flattened primary-statement XBRL). This is the "other source" for
pensions: `companyfacts` only surfaces the tags a filer happens to expose, so our
`pensionDeficit` coverage was thin; the bulk num/sub sets give the recognized net
defined-benefit liability across the whole universe in one pull.

Each quarter's zip carries:
  * sub.txt   adsh -> cik, name, form, period, fy, fp, filed
  * num.txt   adsh, tag, version, ddate (period end), qtrs (0 = instant/balance
              sheet), uom, segments, coreg, value

We keep the CONSOLIDATED company-level rows (no dimensional `segments` member, no
`coreg`) for a curated set of pension tags, join to sub for cik / form / filed,
map to our tickers, and upsert to `pension_facts` (one row per company / tag /
period-end / duration). The tag list is easily extended. The footnote PBO / plan-asset
detail from the Financial Statement AND Notes sets is already wired -- separately, in
`fetch_financial_notes.py` (`notes_num` / `notes_text`).

"""
from __future__ import annotations

import logging
import zipfile
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.constants.constants import SEC_FINSTMT_URL_TEMPLATE, SEC_FINSTMT_FIRST_YEAR
from src.context import Context
from src.data_extract.utils.common.bulk_cache import (
    cache_dir, ensure_zip, quarter_periods,
)
from src.data_extract.utils.common.sec_utils import (
    load_cik_mapping, bulk_ingested_quarters, load_processed_universe,
    save_processed_universe)

logger = logging.getLogger(__name__)

_TABLE = "pension_facts"
_CHUNK = 500_000

# Curated defined-benefit pension tags. The first is the recognized NET deficit
# (balance-sheet, the debt-like obligation that feeds `pensionDeficit`); the rest
# add coverage / detail where filers report them. Extend freely.
_PENSION_TAGS = frozenset({
    "PensionAndOtherPostretirementDefinedBenefitPlansLiabilitiesNoncurrent",
    "PensionAndOtherPostretirementDefinedBenefitPlansLiabilitiesCurrent",
    "PensionAndOtherPostretirementDefinedBenefitPlansLiabilities",
    "DefinedBenefitPensionPlanLiabilitiesNoncurrent",
    "LiabilityPensionAndOtherPostretirementAndPostemploymentBenefitPlansNoncurrent",
    "DefinedBenefitPlanFundedStatusOfPlanAmount",
    "DefinedBenefitPlanBenefitObligation",
    "DefinedBenefitPlanFairValueOfPlanAssets",
    "DefinedBenefitPlanAccumulatedBenefitObligation",
})
_OUT_COLS = ["cik", "ticker", "tag", "ddate", "qtrs", "uom", "value",
             "adsh", "filed", "form", "fy", "fp", "quarter"]


# --------------------------------------------------------------------------- #
# Pure parse (unit-tested)                                                       #
# --------------------------------------------------------------------------- #
def _join_pension(num: pd.DataFrame, sub: pd.DataFrame) -> pd.DataFrame:
    """Filtered num rows (pension tags) + sub -> tidy company-level facts. Pure."""
    if num is None or num.empty or sub is None or sub.empty:
        return pd.DataFrame()
    seg = num.get("segments", pd.Series("", index=num.index)).astype("string").fillna("").str.strip()
    coreg = num.get("coreg", pd.Series("", index=num.index)).astype("string").fillna("").str.strip()
    n = pd.DataFrame({
        "adsh": num["adsh"],
        "tag": num["tag"],
        "ddate": pd.to_datetime(num["ddate"], format="%Y%m%d", errors="coerce"),
        "qtrs": pd.to_numeric(num["qtrs"], errors="coerce"),
        "uom": num["uom"],
        "value": pd.to_numeric(num["value"], errors="coerce"),
    })
    # pension tags only (defensive: real path pre-filters, but keep the join pure),
    # consolidated parent-company fact only (drop dimensional members / co-registrants)
    n = n[n["tag"].isin(_PENSION_TAGS) & (seg == "") & (coreg == "")].dropna(
        subset=["value", "ddate"])
    if n.empty:
        return pd.DataFrame()
    s = pd.DataFrame({
        "adsh": sub["adsh"],
        "cik": sub["cik"].astype("string").str.zfill(10),
        "form": sub.get("form"),
        "fy": sub.get("fy"),
        "fp": sub.get("fp"),
        "filed": pd.to_datetime(sub["filed"], format="%Y%m%d", errors="coerce"),
    })
    return n.merge(s, on="adsh", how="inner")


# --------------------------------------------------------------------------- #
# IO: cache/download + incremental state                                        #
# --------------------------------------------------------------------------- #


def _read_pension_facts(path: Path) -> pd.DataFrame | None:
    """Read sub.txt fully + stream num.txt in chunks keeping only pension tags
    (num.txt is ~100MB+ uncompressed), then join. Returns None on a corrupt zip."""
    try:
        with zipfile.ZipFile(path) as z:
            names = {n.lower(): n for n in z.namelist()}
            if "sub.txt" not in names or "num.txt" not in names:
                return None
            sub = pd.read_csv(z.open(names["sub.txt"]), sep="\t", dtype=str, low_memory=False,
                              usecols=lambda c: c in ("adsh", "cik", "name", "form",
                                                      "period", "fy", "fp", "filed"))
            keep: list[pd.DataFrame] = []
            with z.open(names["num.txt"]) as fh:
                for chunk in pd.read_csv(fh, sep="\t", dtype=str, low_memory=False,
                                         chunksize=_CHUNK):
                    m = chunk["tag"].isin(_PENSION_TAGS)
                    if m.any():
                        keep.append(chunk.loc[m])
    except zipfile.BadZipFile:
        logger.warning("finstmt %s: corrupt zip -> deleting so it re-downloads", path.name)
        path.unlink(missing_ok=True)
        return None
    num = pd.concat(keep, ignore_index=True) if keep else pd.DataFrame()
    return _join_pension(num, sub)


def fetch_financial_statements(context: Context, tickers: list[str]) -> int:
    """Download (cached) the Financial Statement Data Sets over `years_history`,
    extract pension facts for the universe, upsert to `pension_facts`. Returns the
    number of rows upserted."""

    store = context.store
    cikmap = load_cik_mapping(context)
    cik2tkr = ({c: str(t).upper() for c, t in zip(cikmap["cik"], cikmap["ticker"])}
               if not cikmap.empty and "ticker" in cikmap.columns else {})
    years_history = context.config.data_extract.years_history + 1
    cache = cache_dir(context, "sec_financial_statements")

    tickers = {str(t).upper() for t in tickers}          # universe as an uppercased set
    done_q = bulk_ingested_quarters(store, _TABLE)
    new_tickers = tickers - load_processed_universe(cache, _TABLE)   # empty once converged
    if new_tickers:
        logger.info("finstmt: %d new/changed tickers -> re-parsing cached quarters",
                    len(new_tickers))

    saved = 0
    for q in tqdm(quarter_periods(years_history, SEC_FINSTMT_FIRST_YEAR), desc="financial-statement data sets"):
        if q in done_q and not new_tickers:
            continue
        path = ensure_zip(cache / f"{q}.zip",
                          SEC_FINSTMT_URL_TEMPLATE.format(quarter=q),
                          label=f"finstmt {q}", log=logger)
        if path is None:
            continue
        facts = _read_pension_facts(path)
        if facts is None or facts.empty:
            continue
        facts["ticker"] = facts["cik"].map(cik2tkr)
        facts = facts[facts["ticker"].isin(tickers)]
        if facts.empty:
            continue
        # keep the latest-filed value per (cik, tag, period-end, duration)
        facts = (facts.sort_values("filed")
                 .drop_duplicates(subset=["cik", "tag", "ddate", "qtrs"], keep="last"))
        facts["quarter"] = q
        saved += store.save(_TABLE, facts[[c for c in _OUT_COLS if c in facts.columns]])

    save_processed_universe(cache, _TABLE, tickers)   # so a converged re-run skips
    logger.info("pension_facts: upserted %d rows (%d quarters scanned)",
                   saved, len(quarter_periods(years_history, SEC_FINSTMT_FIRST_YEAR)))
    return saved
