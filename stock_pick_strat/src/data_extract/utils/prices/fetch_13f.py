"""
fetch_13f.py  (src/data_extract/utils/fetch_13f.py)
---------------------------------------------------
Institutional 13F holdings from the SEC "Form 13F Data Sets" (free quarterly bulk
TSV zips). Each quarter's zip contains SUBMISSION (accession -> CIK, filing_date,
period) and INFOTABLE (accession -> CUSIP, value, shares). We join them, map CUSIP
-> ticker (OpenFIGI, see fetch_cusip_map), keep the S&P 500 universe, and save
manager-grain long rows [cik, period, filing_date, ticker, cusip, shares, value_usd].

Column names in the SEC datasets are normalized tolerantly (`_pick`), since SEC
occasionally tweaks headers -- verify against a downloaded quarter if a column is
missing. Network/zip IO is isolated in `_download_quarter`; the parse/join
(`_join_13f`) is pure and unit-tested.
"""
from __future__ import annotations

import io
import zipfile

import pandas as pd
import requests

from src.context import Context
from src.data_extract.utils.prices.fetch_cusip_map import build_cusip_ticker_map

_BASE = "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/{name}_form13f.zip"
_HEADERS = {
    "User-Agent": "stock_pick_strat/1.0 (research; valar_analytics@gmail.com)"}

# SEC 13F data-set zips are named by the 3-month FILING-RECEIPT window (NOT calendar
# quarters), start-date to end-date, e.g. `01jun2025-31aug2025_form13f.zip`. The four
# windows per year are Dec-Feb, Mar-May, Jun-Aug, Sep-Nov (13F for the Dec-31 period
# is filed by mid-Feb, hence the one-month shift). Format: DDMMMYYYY-DDMMMYYYY (lower).
_WINDOWS = [(12, -1, 2), (3, 0, 5), (6, 0, 8), (9, 0, 11)]  # (start_month, yr_delta, end_month)


def _pick(df: pd.DataFrame, *candidates: str) -> pd.Series:
    """Return the first present column (case-insensitive) among candidates."""
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return df[lower[cand.lower()]]
    return pd.Series([pd.NA] * len(df), index=df.index)


def _join_13f(submission: pd.DataFrame, infotable: pd.DataFrame) -> pd.DataFrame:
    """Join SUBMISSION + INFOTABLE on accession -> manager-grain holdings. Pure."""
    sub = pd.DataFrame({
        "accession": _pick(submission, "ACCESSION_NUMBER", "accession_number"),
        "cik": _pick(submission, "CIK", "cik"),
        "filing_date": pd.to_datetime(_pick(submission, "FILING_DATE", "filing_date"),
                                      errors="coerce"),
        "period": pd.to_datetime(_pick(submission, "PERIODOFREPORT", "period_of_report",
                                       "periodofreport"), errors="coerce"),
    })
    info = pd.DataFrame({
        "accession": _pick(infotable, "ACCESSION_NUMBER", "accession_number"),
        "cusip": _pick(infotable, "CUSIP", "cusip").astype(str).str.strip().str.upper(),
        "value_usd": pd.to_numeric(_pick(infotable, "VALUE", "value"), errors="coerce"),
        "shares": pd.to_numeric(_pick(infotable, "SSHPRNAMT", "sshPrnamt", "shares"),
                                errors="coerce"),
    })
    holdings = info.merge(sub, on="accession", how="inner")
    # 13F VALUE historically in $1000s; normalize to raw dollars
    holdings["value_usd"] = holdings["value_usd"] * 1000.0
    return holdings.dropna(subset=["cusip", "cik", "period"])


def _period_names(years_history: int, today: pd.Timestamp | None = None) -> list[str]:
    """Data-set base names (no extension) for every filing window in range, e.g.
    '01jun2025-31aug2025'. Only windows whose end date has passed are included.
    Pure/deterministic (pass `today` in tests)."""
    today = (today or pd.Timestamp.today()).normalize()
    names = []
    for y in range(today.year - years_history, today.year + 1):
        for start_month, yr_delta, end_month in _WINDOWS:
            start = pd.Timestamp(year=y + yr_delta, month=start_month, day=1)
            end = pd.Timestamp(year=y, month=end_month, day=1) + pd.offsets.MonthEnd(0)
            if end > today:
                continue
            names.append(f"{start.strftime('%d%b%Y')}-{end.strftime('%d%b%Y')}".lower())
    return names


def _download_quarter(name: str) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Download+unzip one window's SUBMISSION and INFOTABLE tsvs. Network IO."""
    r = requests.get(_BASE.format(name=name), headers=_HEADERS, timeout=120)
    if r.status_code != 200:
        return None
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        names = {n.upper(): n for n in z.namelist()}
        if "SUBMISSION.TSV" not in names or "INFOTABLE.TSV" not in names:
            return None
        sub = pd.read_csv(z.open(names["SUBMISSION.TSV"]), sep="\t", dtype=str, low_memory=False)
        info = pd.read_csv(z.open(names["INFOTABLE.TSV"]), sep="\t", dtype=str, low_memory=False)
    return sub, info


def fetch_13f(context: Context) -> pd.DataFrame:
    """Download the 13F data sets, map to tickers, keep the universe, and cache."""
    universe = set(pd.read_csv(context.paths["TICKERS_PATH"])["ticker"])
    years_history = context.config.data_extract.years_history
    frames = []
    for tag in _period_names(years_history):
        try:
            got = _download_quarter(tag)
        except Exception as e:
            print(f"13F {tag} download failed: {e}")
            continue
        if got is None:
            continue
        frames.append(_join_13f(*got))
    if not frames:
        print("No 13F data downloaded.")
        return pd.DataFrame(columns=["cik", "period", "filing_date", "ticker",
                                     "cusip", "shares", "value_usd"])

    raw = pd.concat(frames, ignore_index=True)
    cmap = build_cusip_ticker_map(context, raw["cusip"].unique().tolist())
    raw = raw.merge(cmap, on="cusip", how="inner")
    out = raw[raw["ticker"].isin(universe)].reset_index(drop=True)
    out.to_parquet(context.paths["INSTITUTIONAL_HOLDINGS_PATH"], index=False)
    print(f"Saved {len(out)} 13F holding rows ({out['ticker'].nunique()} tickers) "
          f"to {context.paths['INSTITUTIONAL_HOLDINGS_PATH']}")
    return out
