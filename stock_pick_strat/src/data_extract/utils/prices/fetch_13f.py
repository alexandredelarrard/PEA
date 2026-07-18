"""
fetch_13f.py  (src/data_extract/utils/fetch_13f.py)
---------------------------------------------------
Institutional 13F holdings from the SEC "Form 13F Data Sets" (free quarterly bulk
TSV zips). Each quarter's zip has SUBMISSION (accession -> CIK, filing_date,
period) and INFOTABLE (one row per holding: CUSIP, VALUE, SSHPRNAMT, plus the two
columns that classify the holding: SSHPRNAMTTYPE = SH|PRN, PUTCALL = blank|Put|Call).

Reconciliation to a ticker is by **CUSIP** (via OpenFIGI, see fetch_cusip_map) —
never the free-text NAMEOFISSUER, which varies by filer.

Each manager x security x quarter holding is split by type so long equity is not
contaminated by options / bonds:
    shares / value_usd  -> LONG STOCK only (SSHPRNAMTTYPE=SH, no put/call)
    call_shares / call_value
    put_shares  / put_value        (the bearish / "sell-side" exposure)
    debt_prn    / debt_value       (SSHPRNAMTTYPE=PRN)
    other_value                    (residual / malformed rows)

VALUE units: 13F reported VALUE in $THOUSANDS before the Jan-2023 SEC amendment,
in whole DOLLARS after -> scaled per filing_date so magnitudes align across eras.

Zips are cached under data/sec_bulk_cache/form13f/ and only re-downloaded when
missing. Network/zip IO is isolated in `_ensure_zip`/`_read_zip`; the parse/join
(`_classify_holdings`, `_join_13f`) is pure and unit-tested.
"""
from __future__ import annotations

from tkinter.constants import Y
import zipfile
from pathlib import Path
import logging

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from src.constants.constants import SEC_FORM13F_URL_DICT
from src.context import Context
from src.data_extract.utils.prices.fetch_cusip_map import build_cusip_ticker_map

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": "stock_pick_strat/1.0 (research; valar_analytics@gmail.com)"}

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
        "cusip": _pick(infotable, "CUSIP", "cusip").astype(str).str.strip().str.upper(),
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
    """Data-set base names (no extension) for every filing window in range, e.g.
    '01jun2025-31aug2025' from 2024. Only windows whose end date has passed are included.
    Pure/deterministic (pass `today` in tests)."""
    today = (today or pd.Timestamp.today()).normalize()
    names = []
    for y in range(today.year - years_history, today.year + 1):
        if y >= 2013: # SEC started filing 13F data in 2013 q2
            for quarter in range(1,5):
                names.append(f"{y}q{quarter}")
    return names


# --------------------------------------------------------------------------- #
# Zip cache: download once to disk, reuse thereafter                           #
# --------------------------------------------------------------------------- #
def _cache_dir(context: Context) -> Path:
    d = context.paths["SEC_BULK_CACHE_DIR"] / "form13f"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _ensure_zip(name: str, cache_dir: Path) -> Path | None:
    """Return the local path to this window's zip, downloading it once if absent.
    Streams to a .part file and renames on success so a partial download is never
    mistaken for a cached one."""
    path = cache_dir / f"{name}_form13f.zip"
    if path.exists() and path.stat().st_size > 0:
        return path
    try:
        r = requests.get(SEC_FORM13F_URL_DICT.get(name), headers=_HEADERS,
                         timeout=180, stream=True)
    except Exception as e:
        logger.warning(f"13F {name} download failed: {e}")
        return None
    if r.status_code != 200:
        logger.warning(f"13F {name} download failed: {r.status_code}")
        return None
    tmp = path.with_suffix(".part")
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    tmp.replace(path)
    return path


def _read_zip(path: Path) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Read SUBMISSION + INFOTABLE tsvs from a cached zip on disk."""
    try:
        with zipfile.ZipFile(path) as z:
            names = {n.upper(): n for n in z.namelist()}
            if "SUBMISSION.TSV" not in names or "INFOTABLE.TSV" not in names:
                return None
            sub = pd.read_csv(z.open(names["SUBMISSION.TSV"]), sep="\t", dtype=str, low_memory=False)
            info = pd.read_csv(z.open(names["INFOTABLE.TSV"]), sep="\t", dtype=str, low_memory=False)
        return sub, info
    except zipfile.BadZipFile:
        logger.warning(f"13F {path.name}: corrupt zip — deleting so it re-downloads next run")
        path.unlink(missing_ok=True)
        return None


def fetch_13f(context: Context) -> pd.DataFrame:
    """Download (once, cached) the 13F data sets, split by holding type, map to
    tickers, keep the universe, and store."""
    universe = set(context.store.load("sp500_tickers", columns=["ticker"])["ticker"])
    years_history = context.config.data_extract.years_history + 1
    cache_dir = _cache_dir(context)

    frames = []
    for tag in tqdm(_period_names(years_history), desc="13F data sets"):
        path = _ensure_zip(tag, cache_dir)
        if path is None:
            continue
        got = _read_zip(path)
        if got is not None:
            frames.append(_join_13f(*got))

    cols = ["cik", "period", "filing_date", "ticker", "cusip", "shares", "value_usd",
            "call_shares", "call_value", "put_shares", "put_value",
            "debt_prn", "debt_value", "other_value"]
    if not frames:
        logger.warning("No 13F data downloaded.")
        return pd.DataFrame(columns=cols)

    raw = pd.concat(frames, ignore_index=True)
    cmap = build_cusip_ticker_map(context, raw["cusip"].unique().tolist())
    raw = raw.merge(cmap, on="cusip", how="inner")
    out = raw[raw["ticker"].isin(universe)].reset_index(drop=True)
    context.store.save("institutional_holdings", out)
    logger.warning(f"Saved {len(out)} 13F holding rows ({out['ticker'].nunique()} tickers) "
          f"to DB table 'institutional_holdings'")
    return out
