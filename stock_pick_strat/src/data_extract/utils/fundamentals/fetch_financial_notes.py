"""
fetch_financial_notes.py  (src/data_extract/utils/fundamentals/fetch_financial_notes.py)
----------------------------------------------------------------------------------------
SEC "Financial Statement AND Notes" data sets — the heavier sibling of the plain
Financial Statement Data Sets (fetch_financial_statements.py). Where the plain sets
carry only the PRIMARY-statement XBRL, the NOTES sets ALSO carry the footnote-level
facts, both:

  * NUMERIC (num.tsv) — the footnote pension roll-forward the primary statements
    never expose: projected benefit obligation (PBO), fair value of plan assets,
    accumulated benefit obligation (ABO), service / interest cost, employer
    contributions, discount-rate assumption. `companyfacts` only surfaces these
    where a filer tags them UNdimensioned, so coverage there is patchy (PBO ~24%,
    plan assets ~42% of our universe); the bulk sets give the same undimensioned
    totals across the whole universe in one pull, and we UNION them with the
    existing pension_facts / companyfacts data at the feature layer.
  * TEXT (txt.tsv) — the narrative note prose (pension, revenue recognition,
    commitments / litigation, segment, going-concern / concentration risk,
    critical accounting policies). Stored raw for LATER embedding / sentiment;
    no NLP is done here. Scoped to a curated high-signal tag set (not every
    boilerplate policy block) to keep the table lean.

Each period's zip carries (.tsv, note the extension differs from the plain sets'
.txt): sub.tsv (adsh -> cik, form, fy, fp, filed), num.tsv (adsh, tag, ddate,
qtrs, uom, dimh, dimn, coreg, value, footnote, ...), txt.tsv (adsh, tag, ddate,
qtrs, dimn, coreg, escaped, txtlen, value [the text], ...).

We keep the CONSOLIDATED / undimensioned company-level fact only (`dimn == 0`, no
`coreg`) — dimn>0 rows are pension-vs-OPEB / asset-category breakdowns we skip for
now (a dim.tsv join to sum members is a documented follow-up). Rows are joined to
sub for cik / form / filed, mapped to our tickers, and upserted per filing to
`notes_num` (numeric) and `notes_text` (text).

Rolling granularity: the SEC now consolidates months into a quarter after ~1 year,
so at any instant only the last ~12 months exist as monthly ("YYYY_MM") and older
data as quarterly ("YYYYqQ"); the two spans never overlap. We take the authoritative
period list from the landing page (falling back to a deterministic generator +
404-skip), so we only request files that exist.

Incremental: zips cached under data/sec_financial_notes/ and only downloaded when
missing; a period already in the DB is skipped unless the universe gained tickers
(then cached zips are re-parsed, no re-download). Upsert de-dupes by filing.

WARNING: these files are ~300-450MB EACH (~30x the other sources). At
notes_years_history=15 the full back-fill is ~26GB of cached zips — scoped by the
dedicated `data_extract.notes_years_history` config knob.
"""
from __future__ import annotations

import logging
import re
import zipfile
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm
from sqlalchemy import text

from src.constants.constants import (
    SEC_FINNOTES_URL_TEMPLATE, SEC_FINNOTES_FIRST_YEAR)
from src.context import Context
from src.data_extract.utils.common.bulk_cache import (
    cache_dir, ensure_zip, ingested_periods,
)
from src.data_extract.utils.common.sec_utils import (
    load_cik_mapping, load_processed_universe, save_processed_universe)

logger = logging.getLogger(__name__)

_NUM_TABLE = "notes_num"
_TXT_TABLE = "notes_text"
_CHUNK = 500_000
_LANDING_URL = ("https://www.sec.gov/data-research/sec-markets-data/"
                "financial-statement-notes-data-sets")

# Curated footnote NUMERIC pension tags (undimensioned totals). Superset of the
# balance-sheet net-liability tags in pension_facts — adds the footnote detail
# only the NOTES sets carry. Discount-rate tags are percentages (uom != USD).
_NOTES_NUM_TAGS = frozenset({
    "DefinedBenefitPlanBenefitObligation",                          # PBO
    "DefinedBenefitPlanFairValueOfPlanAssets",                      # plan assets (FV)
    "DefinedBenefitPlanAccumulatedBenefitObligation",              # ABO
    "DefinedBenefitPlanFundedStatusOfPlanAmount",                  # funded status (rare; else computed)
    "DefinedBenefitPlanNetPeriodicBenefitCost",                    # net periodic cost
    "DefinedBenefitPlanServiceCost",                               # service cost (operating piece)
    "DefinedBenefitPlanInterestCost",                              # interest cost
    "DefinedBenefitPlanExpectedReturnOnPlanAssets",                # expected return on assets
    "DefinedBenefitPlanContributionsByEmployer",                   # employer cash contributions
    "DefinedBenefitPlanExpectedFutureBenefitPaymentsNextTwelveMonths",  # near-term cash outflow
    "DefinedBenefitPlanAssumptionsUsedCalculatingNetPeriodicBenefitCostDiscountRate",
    "DefinedBenefitPlanWeightedAverageAssumptionsUsedCalculatingBenefitObligationDiscountRate",
})

# High-signal NOTES narrative text blocks (TextBlock XBRL elements). Stored raw for
# later embedding / sentiment; a few variants per theme for coverage.
_NOTES_TEXT_TAGS = frozenset({
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
})

_NUM_USECOLS = {"adsh", "tag", "ddate", "qtrs", "uom", "dimn", "coreg", "footnote", "value"}
_TXT_USECOLS = {"adsh", "tag", "ddate", "qtrs", "dimn", "coreg", "escaped", "txtlen",
                "footnote", "value"}
_NUM_PK = ["adsh", "tag", "ddate", "qtrs"]
_TXT_PK = ["adsh", "tag", "ddate", "qtrs"]
_NUM_OUT = ["cik", "ticker", "adsh", "tag", "ddate", "qtrs", "uom", "value",
            "footnote", "form", "fy", "fp", "filed", "period"]
_TXT_OUT = ["cik", "ticker", "adsh", "tag", "ddate", "qtrs", "txtlen", "escaped",
            "value", "footnote", "form", "fy", "fp", "filed", "period"]


# --------------------------------------------------------------------------- #
# Period list (rolling quarterly <-> monthly)                                   #
# --------------------------------------------------------------------------- #
def _period_year(tag: str) -> int:
    """'2024q1' -> 2024 ; '2025_07' -> 2025."""
    return int(tag[:4])


def _scrape_available_periods() -> list[str] | None:
    """Authoritative available-file list from the landing page (robust to the
    rolling quarterly<->monthly boundary). None on any failure -> caller falls
    back to the deterministic generator."""
    try:
        r = requests.get(_LANDING_URL, headers=_sec_headers(), timeout=60)
        if r.status_code != 200:
            return None
        tags = re.findall(r"/(\d{4}(?:q[1-4]|_\d{2}))_notes\.zip", r.text)
        return sorted(set(tags)) or None
    except Exception:                                   # noqa: BLE001 (best-effort)
        return None


def _generate_periods(years_history: int, today: pd.Timestamp | None = None) -> list[str]:
    """Deterministic candidate tags: quarterly for every year in the window PLUS
    the last 14 months as monthly. 404s are skipped by the downloader, so
    over-generating the recent edge is harmless."""
    today = (today or pd.Timestamp.today()).normalize()
    start_year = max(SEC_FINNOTES_FIRST_YEAR, today.year - years_history)
    out: list[str] = []
    for y in range(start_year, today.year + 1):
        out += [f"{y}q{q}" for q in range(1, 5)]
    m = today.replace(day=1)
    for _ in range(14):
        out.append(f"{m.year}_{m.month:02d}")
        m = (m - pd.Timedelta(days=1)).replace(day=1)
    return sorted(set(out))


def _notes_periods(years_history: int, today: pd.Timestamp | None = None) -> list[str]:
    """Period tags to fetch, newest last, filtered to the year window."""
    today = (today or pd.Timestamp.today()).normalize()
    start_year = max(SEC_FINNOTES_FIRST_YEAR, today.year - years_history)
    avail = _scrape_available_periods()
    tags = avail if avail is not None else _generate_periods(years_history, today)
    return sorted(t for t in tags if _period_year(t) >= start_year)


# --------------------------------------------------------------------------- #
# Pure parse (unit-tested)                                                       #
# --------------------------------------------------------------------------- #
def _sub_meta(sub: pd.DataFrame, cik2tkr: dict[str, str], universe: set[str]) -> pd.DataFrame:
    """sub.tsv -> [adsh, cik, ticker, form, fy, fp, filed] for UNIVERSE filers only. Pure."""
    if sub is None or sub.empty:
        return pd.DataFrame()
    s = pd.DataFrame({
        "adsh": sub["adsh"],
        "cik": sub["cik"].astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(10),
        "form": sub.get("form"),
        "fy": sub.get("fy"),
        "fp": sub.get("fp"),
        "filed": pd.to_datetime(sub["filed"], format="%Y%m%d", errors="coerce"),
    })
    s["ticker"] = s["cik"].map(cik2tkr)
    return s[s["ticker"].isin(universe)]


def _join_notes_num(num: pd.DataFrame, sub_meta: pd.DataFrame) -> pd.DataFrame:
    """Filtered num rows (pension tags, dimn==0) + sub_meta -> tidy facts. Pure."""
    if num is None or num.empty or sub_meta is None or sub_meta.empty:
        return pd.DataFrame()
    n = pd.DataFrame({
        "adsh": num["adsh"],
        "tag": num["tag"],
        "ddate": pd.to_datetime(num["ddate"], format="%Y%m%d", errors="coerce"),
        "qtrs": pd.to_numeric(num["qtrs"], errors="coerce"),
        "uom": num.get("uom"),
        "value": pd.to_numeric(num["value"], errors="coerce"),
        "footnote": num.get("footnote"),
    }).dropna(subset=["value", "ddate"])
    return n.merge(sub_meta, on="adsh", how="inner") if not n.empty else pd.DataFrame()


def _join_notes_text(txt: pd.DataFrame, sub_meta: pd.DataFrame) -> pd.DataFrame:
    """Filtered txt rows (high-signal tags, dimn==0) + sub_meta -> tidy text. Pure."""
    if txt is None or txt.empty or sub_meta is None or sub_meta.empty:
        return pd.DataFrame()
    t = pd.DataFrame({
        "adsh": txt["adsh"],
        "tag": txt["tag"],
        "ddate": pd.to_datetime(txt["ddate"], format="%Y%m%d", errors="coerce"),
        "qtrs": pd.to_numeric(txt["qtrs"], errors="coerce"),
        "txtlen": pd.to_numeric(txt.get("txtlen"), errors="coerce"),
        "escaped": txt.get("escaped"),
        "value": txt["value"].astype("string"),
        "footnote": txt.get("footnote"),
    }).dropna(subset=["value", "ddate"])
    t = t[t["value"].str.strip() != ""]
    return t.merge(sub_meta, on="adsh", how="inner") if not t.empty else pd.DataFrame()


# --------------------------------------------------------------------------- #
# IO: cache/download + incremental state                                        #
# --------------------------------------------------------------------------- #


def _chunk_filter(z: zipfile.ZipFile, name: str, adsh_set: set[str],
                  tags: frozenset[str], usecols: set[str]) -> pd.DataFrame:
    """Stream a huge .tsv member in chunks, keeping only universe filings, curated
    tags and undimensioned/consolidated rows (dimn==0, no coreg)."""
    keep: list[pd.DataFrame] = []
    with z.open(name) as fh:
        for chunk in pd.read_csv(fh, sep="\t", dtype=str, low_memory=False,
                                 chunksize=_CHUNK, on_bad_lines="skip",
                                 usecols=lambda c: c in usecols):
            dimn = pd.to_numeric(chunk.get("dimn"), errors="coerce")
            coreg = (chunk.get("coreg", pd.Series("", index=chunk.index))
                     .astype("string").fillna("").str.strip())
            m = chunk["adsh"].isin(adsh_set) & chunk["tag"].isin(tags) & (dimn == 0) & (coreg == "")
            if m.any():
                keep.append(chunk.loc[m])
    return pd.concat(keep, ignore_index=True) if keep else pd.DataFrame()


def _read_notes(path: Path, cik2tkr: dict[str, str],
                universe: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One notes zip -> (tidy num facts, tidy text) for the universe. Reads sub.tsv
    fully (small) to resolve universe filings, then streams num/txt in chunks."""
    try:
        with zipfile.ZipFile(path) as z:
            names = {n.lower(): n for n in z.namelist()}
            if not {"sub.tsv", "num.tsv", "txt.tsv"} <= set(names):
                return pd.DataFrame(), pd.DataFrame()
            sub = pd.read_csv(z.open(names["sub.tsv"]), sep="\t", dtype=str, low_memory=False,
                              usecols=lambda c: c in ("adsh", "cik", "form", "fy", "fp", "filed"))
            sub_meta = _sub_meta(sub, cik2tkr, universe)
            if sub_meta.empty:
                return pd.DataFrame(), pd.DataFrame()
            adsh_set = set(sub_meta["adsh"])
            num = _chunk_filter(z, names["num.tsv"], adsh_set, _NOTES_NUM_TAGS, _NUM_USECOLS)
            txt = _chunk_filter(z, names["txt.tsv"], adsh_set, _NOTES_TEXT_TAGS, _TXT_USECOLS)
    except zipfile.BadZipFile:
        logger.warning("notes %s: corrupt zip -> deleting so it re-downloads", path.name)
        path.unlink(missing_ok=True)
        return pd.DataFrame(), pd.DataFrame()
    return _join_notes_num(num, sub_meta), _join_notes_text(txt, sub_meta)


def fetch_financial_notes(context: Context, tickers: list[str]) -> int:
    """Download (cached) the SEC Financial Statement & Notes data sets over
    `notes_years_history`, extract footnote pension NUMERICS -> `notes_num` and
    high-signal note TEXT -> `notes_text` for the universe. Returns total rows
    upserted (num + text). Incremental: a period already in the DB is skipped
    (no re-download) unless the universe gained tickers (then cached zips are
    re-parsed, no re-download)."""
    store = context.store
    cikmap = load_cik_mapping(context)
    cik2tkr = ({c: str(t).upper() for c, t in zip(cikmap["cik"], cikmap["ticker"])}
               if not cikmap.empty and "ticker" in cikmap.columns else {})
    de = context.config.data_extract
    years_history = getattr(de, "notes_years_history", de.years_history) + 1
    cache = cache_dir(context, "sec_financial_notes")
    universe = {str(t).upper() for t in tickers}

    done = ingested_periods(context, (_NUM_TABLE, _TXT_TABLE))
    new_tickers = universe - load_processed_universe(cache, _NUM_TABLE)   # empty once converged
    if new_tickers:
        logger.info("notes: %d new/changed tickers -> re-parsing cached files", len(new_tickers))

    n_num = n_txt = 0
    periods = _notes_periods(years_history)
    for period in tqdm(periods, desc="SEC financial-statement notes"):
        if period in done and not new_tickers:
            continue
        path = ensure_zip(cache / f"{period}_notes.zip",
                          SEC_FINNOTES_URL_TEMPLATE.format(period=period),
                          label=f"notes {period}", timeout=600, log=logger)
        if path is None:
            continue
        num, txt = _read_notes(path, cik2tkr, universe)
        if not num.empty:
            num = num.sort_values("filed").drop_duplicates(subset=_NUM_PK, keep="last")
            num["period"] = period
            n_num += store.save(_NUM_TABLE, num[[c for c in _NUM_OUT if c in num.columns]])
        if not txt.empty:
            txt = txt.sort_values("filed").drop_duplicates(subset=_TXT_PK, keep="last")
            txt["period"] = period
            n_txt += store.save(_TXT_TABLE, txt[[c for c in _TXT_OUT if c in txt.columns]])

    save_processed_universe(cache, _NUM_TABLE, universe)   # so a converged re-run skips
    logger.info("notes: upserted %d num + %d text rows (%d periods scanned)",
                   n_num, n_txt, len(periods))
    return n_num + n_txt
