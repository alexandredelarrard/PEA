"""
fetch_13f.py (src/data_extract/utils/fetch_13f.py)
---------------------------------------------------
Extracts institutional holdings from SEC Form 13F quarterly bulk TSV datasets
(`SUBMISSION` and `INFOTABLE`). Reconciles holdings to tickers exclusively via 
CUSIP (OpenFIGI mapping) rather than unstandardized issuer names.

Data Grain & Instrument Breakdown:
- One record per (Manager, CUSIP, Quarter), categorized to isolate long equity 
  from derivatives/debt:
  • Long Stock: Standard equity (`SSHPRNAMTTYPE=SH`, no Put/Call)
  • Options: Split into Call vs. Put exposure (`PUTCALL`)
  • Debt / Principal: Fixed income holdings (`SSHPRNAMTTYPE=PRN`)
  • Residual / Other: Unclassified or malformed rows

Key Guardrails:
- Dollar Value Scaling: Automatically normalizes reported `VALUE` magnitudes 
  ($thousands pre-Jan 2023 vs. whole dollars post-Jan 2023) based on filing date.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.constants.constants import SEC_FORM13F_URL_DICT, UNIVERSE_TABLE
from src.context import Context
from src.data_extract.utils.common.bulk_cache import (
    cache_dir, ensure_zip, ingested_periods, read_zip_members,
)
from src.data_extract.utils.prices.fetch_cusip_map import build_cusip_ticker_map
from src.data_extract.utils.common.run_manifest import record_run
from src.data_extract.utils.common.sec_utils import (
    load_processed_universe, save_processed_universe)

logger = logging.getLogger(__name__)

# SEC changed the 13F VALUE unit from $thousands to $ones with the amendment
# effective 2023-01-03; scale by filing_date so pre/post-2023 values are comparable.
_VALUE_DOLLARS_FROM = pd.Timestamp("2023-01-01")
_VALUE_COLS = ["shares_value", "call_value", "put_value", "debt_value", "other_value"]

def _pick(df: pd.DataFrame, *candidates: str) -> pd.Series:
    """Return the first present column (case-insensitive) among candidates."""
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return df[lower[cand.lower()]]
    return pd.Series([pd.NA] * len(df), index=df.index)


# --------------------------------------------------------------------------- #
# Classification: split each holding into stock / call / put / debt / other    #
# --------------------------------------------------------------------------- #
def _classify_holdings(infotable: pd.DataFrame) -> pd.DataFrame:
    """One typed row per INFOTABLE line: the value/shares land in exactly one
    bucket keyed on PUTCALL (Put/Call) and SSHPRNAMTTYPE (SH shares / PRN debt).
    A blank type with no put/call is treated as long stock (the overwhelmingly
    common case, and what older datasets omit)."""
    putcall = _pick(infotable, "PUTCALL", "putCall").astype("string").str.strip().str.upper().fillna("")
    amttype = _pick(infotable, "SSHPRNAMTTYPE", "sshPrnamtType").astype("string").str.strip().str.upper().fillna("")
    amt = pd.to_numeric(_pick(infotable, "SSHPRNAMT", "sshPrnamt", "shares"), errors="coerce").fillna(0.0)
    val = pd.to_numeric(_pick(infotable, "VALUE", "value"), errors="coerce").fillna(0.0)

    is_call = putcall == "CALL"
    is_put = putcall == "PUT"
    opt = is_call | is_put
    is_debt = (~opt) & (amttype == "PRN")
    is_stock = (~opt) & (amttype.isin(["SH", ""]))     # SH or blank -> long equity
    is_other = ~(is_call | is_put | is_debt | is_stock)

    z = pd.Series(0.0, index=infotable.index)
    return pd.DataFrame({
        "accession": _pick(infotable, "ACCESSION_NUMBER", "accession_number"),
        # canonical 9-char CUSIP (restore any dropped leading zero) so the map skip
        # and the holdings<->ticker merge use ONE form (see fetch_cusip_map.normalize_cusip)
        "cusip": _pick(infotable, "CUSIP", "cusip").astype(str).str.strip().str.upper().str.zfill(9),
        "shares": amt.where(is_stock, z),        "shares_value": val.where(is_stock, z),
        "call_shares": amt.where(is_call, z),    "call_value": val.where(is_call, z),
        "put_shares": amt.where(is_put, z),      "put_value": val.where(is_put, z),
        "debt_prn": amt.where(is_debt, z),       "debt_value": val.where(is_debt, z),
        "other_value": val.where(is_other, z),
    })


def _join_13f(submission: pd.DataFrame, infotable: pd.DataFrame) -> pd.DataFrame:
    """SUBMISSION + typed INFOTABLE -> one row per manager x security x quarter
    with the type breakdown. Pure."""
    sub = pd.DataFrame({
        "accession": _pick(submission, "ACCESSION_NUMBER", "accession_number"),
        "cik": _pick(submission, "CIK", "cik"),
        "filing_date": pd.to_datetime(_pick(submission, "FILING_DATE", "filing_date"),
                                      format="mixed", errors="coerce"),
        "period": pd.to_datetime(_pick(submission, "PERIODOFREPORT", "period_of_report",
                                       "periodofreport"), format="mixed", errors="coerce"),
    })
    typed = _classify_holdings(infotable)
    # combine a manager's multiple lines for the same security (e.g. split sub-accounts)
    agg = typed.groupby(["accession", "cusip"], as_index=False).sum(numeric_only=True)
    holdings = agg.merge(sub, on="accession", how="inner")

    # normalize VALUE units to raw dollars (thousands before the 2023 amendment)
    scale = np.where(holdings["filing_date"].fillna(pd.Timestamp("2000-01-01"))
                     < _VALUE_DOLLARS_FROM, 1000.0, 1.0)
    for c in _VALUE_COLS:
        holdings[c] = holdings[c] * scale
    holdings["value_usd"] = holdings["shares_value"]     # long-equity value (feature-layer key)

    return holdings.dropna(subset=["cusip", "cik", "period"])


def _period_names(years_history: int, today: pd.Timestamp | None = None) -> list[str]:
    """Data-set base names (no extension) for every filing window in range, e.g. '2025q2'.
    Only quarters whose END date has passed are included, because SEC publishes a data set
    only after the quarter closes. Pure/deterministic (pass `today` in tests).

    `pd.to_datetime("2026q3")` resolves to the quarter's START (2026-07-01), so comparing
    that against `today` admitted the current, not-yet-published quarter from its first day
    and spent a guaranteed 404 on every run. Anchor on the quarter END instead."""
    today = (today or pd.Timestamp.today()).normalize()
    names = []
    for y in range(today.year - years_history, today.year + 1):
        if y >= 2013:  # SEC started filing 13F data in 2013 q2
            for quarter in range(1, 5):
                tag = f"{y}q{quarter}"
                quarter_end = pd.Period(tag, freq="Q").end_time.normalize()
                if tag == "2013q1" or quarter_end > today:
                    continue
                names.append(tag)
    return names


# --------------------------------------------------------------------------- #
# Zip cache: download once to disk, reuse thereafter                           #
# --------------------------------------------------------------------------- #


def fetch_13f(context: Context) -> pd.DataFrame:
    """Download (once, cached) the 13F data sets, split by holding type, map to
    tickers, keep the universe, and store.

    INCREMENTAL: a quarter already ingested into `sec13f_hr` is skipped
    ENTIRELY (no re-download, no re-parse of the cached zip, no re-ingest) unless the
    ticker universe grew (then cached zips are re-parsed to back-fill new names). The
    CUSIP->ticker map is built only over the NEW quarters' cusips (and itself skips
    already-attempted cusips), so a converged re-run does almost no work."""
    
    store = context.store
    universe = set(store.load(UNIVERSE_TABLE, columns=["ticker"])["ticker"])
    years_history = context.config.data_extract.years_history + 1
    cache = cache_dir(context, "SEC_13F_INSIDERS_DIR")

    done = ingested_periods(context, "sec13f_hr", "quarter")
    new_tickers = universe - load_processed_universe(cache, "sec13f_hr")
    if new_tickers:
        logger.info("13F: %d new/changed tickers -> re-parsing cached quarters", len(new_tickers))

    quarter_frames: dict[str, pd.DataFrame] = {}
    for tag in tqdm(_period_names(years_history), desc="13F data sets"):
        if tag in done and not new_tickers:
            continue                                   # downloaded + ingested already -> skip
        path = ensure_zip(cache / f"{tag}_form13f.zip",
                          SEC_FORM13F_URL_DICT.get(tag),
                          label=f"13F {tag}", timeout=180, log=logger)
        if path is None:
            continue
        got = read_zip_members(path, ("SUBMISSION.tsv", "INFOTABLE.tsv"), log=logger)
        if got is None:
            continue
        h = _join_13f(got["SUBMISSION.tsv"], got["INFOTABLE.tsv"])
        if not h.empty:
            h["quarter"] = tag
            quarter_frames[tag] = h

    cols = ["cik", "period", "filing_date", "ticker", "cusip", "shares", "value_usd",
            "call_shares", "call_value", "put_shares", "put_value",
            "debt_prn", "debt_value", "other_value", "quarter"]
    if not quarter_frames:
        save_processed_universe(cache, "sec13f_hr", universe)
        logger.info("13F sec13f_hr already up to date (no new quarters).")
        record_run(context, "sec13f_hr", len(universe), 0)
        return pd.DataFrame(columns=cols)

    all_cusips = sorted(set().union(*(set(h["cusip"].unique()) for h in quarter_frames.values())))
    cmap = build_cusip_ticker_map(context, all_cusips)

    saved, saved_frames = 0, []
    for tag, h in quarter_frames.items():
        out = h.merge(cmap, on="cusip", how="inner")
        out = out[out["ticker"].isin(universe)]
        if not out.empty:
            keep = out[[c for c in cols if c in out.columns]]
            saved += store.save("sec13f_hr", keep)
            saved_frames.append(keep)
    save_processed_universe(cache, "sec13f_hr", universe)
    logger.warning("Saved %d 13F holding rows across %d new quarter(s) to 'sec13f_hr'",
                   saved, len(quarter_frames))
    record_run(context, "sec13f_hr", len(universe), saved)
    return pd.concat(saved_frames, ignore_index=True) if saved_frames else pd.DataFrame(columns=cols)
