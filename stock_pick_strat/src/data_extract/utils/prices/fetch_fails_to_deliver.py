"""
fetch_fails_to_deliver.py  (src/data_extract/utils/prices/fetch_fails_to_deliver.py)
------------------------------------------------------------------------------------
SEC Fails-to-Deliver (FTD) settlement-fails signal. When a trade doesn't settle
on time (the seller fails to deliver shares by T+1/T+2), the CNS position is a
"fail to deliver" — persistently high fails are a well-known signal of settlement
stress / naked-short pressure and correlate with squeeze risk. Complements the
FINRA RegSHO short-VOLUME signal (`short_interest`); combined at the feature layer.

Source: semi-monthly text files (free, no auth), each ~2 weeks of DAILY fails:
    SETTLEMENT DATE | CUSIP | SYMBOL | QUANTITY (FAILS) | DESCRIPTION | PRICE
`{period}` = "YYYYMMa" (settlement dates 1-15) / "YYYYMMb" (16-end). Grain is
ticker x settlement date -> stored long [date, ticker, fails_quantity, fails_value].

POINT-IN-TIME: FTD files are published with a lag (~1-2 months), so a backtest that
forward-fills from the settlement date is conservative (the data is only public later).

Kept in its OWN table (not merged into `short_interest`): FTD's lagged, semi-monthly
files would pollute short_interest's global-max-date incremental and break its
backfill. Incremental here is FILE-level: a semi-monthly file already in the DB is
skipped (no re-download) unless the ticker universe grew (then cached files are
re-parsed, no re-download). Zips cached under data/sec_fails_to_deliver/.
"""
from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

import pandas as pd
import requests
from sqlalchemy import text
from tqdm import tqdm

from src.constants.constants import SEC_FTD_URL_TEMPLATE, SEC_FTD_FIRST_YEAR
from src.context import Context
from src.data_extract.utils.common.sec_utils import (
    load_processed_universe, save_processed_universe)

logger = logging.getLogger(__name__)

_HEADERS = {"User-Agent": "stock_pick_strat/1.0 (research; valar_analytics@gmail.com)"}
_TABLE = "fails_to_deliver"
_OUT_COLS = ["ticker", "date", "fails_quantity", "fails_value", "period"]


def _periods(years_history: int, today: pd.Timestamp | None = None) -> list[str]:
    """Semi-monthly file tags ('YYYYMMa'/'YYYYMMb') from the data-set era to now."""
    today = (today or pd.Timestamp.today()).normalize()
    out: list[str] = []
    for y in range(today.year - years_history, today.year + 1):
        if y < SEC_FTD_FIRST_YEAR:
            continue
        for m in range(1, 13):
            if (y, m) > (today.year, today.month):
                break
            out += [f"{y}{m:02d}a", f"{y}{m:02d}b"]
    return out


# --------------------------------------------------------------------------- #
# Pure parse (unit-tested)                                                       #
# --------------------------------------------------------------------------- #
def _parse_ftd(raw: str) -> pd.DataFrame:
    """One semi-monthly FTD file -> [date, ticker, fails_quantity, fails_value],
    summed per (date, ticker). Pure. PRICE '.' (N/A) -> fails_value NaN."""
    if not raw or "|" not in raw:
        return pd.DataFrame(columns=["date", "ticker", "fails_quantity", "fails_value"])
    df = pd.read_csv(io.StringIO(raw), sep="|", dtype=str, engine="python", on_bad_lines="skip")
    cols = {c.strip().upper(): c for c in df.columns}

    def col(name: str) -> pd.Series:
        return df[cols[name]] if name in cols else pd.Series(pd.NA, index=df.index)

    price = pd.to_numeric(
        col("PRICE").astype("string").str.strip().where(lambda s: s != "."), errors="coerce")
    qty = pd.to_numeric(col("QUANTITY (FAILS)"), errors="coerce")
    out = pd.DataFrame({
        "date": pd.to_datetime(col("SETTLEMENT DATE"), format="%Y%m%d", errors="coerce"),
        "ticker": col("SYMBOL").astype("string").str.upper()
                  .str.replace(".", "-", regex=False).str.strip(),
        "fails_quantity": qty,
        "fails_value": qty * price,
    }).dropna(subset=["date", "ticker"])
    out = out[out["ticker"] != ""]
    if out.empty:
        return out
    # min_count=1 so a (date, ticker) with only N/A prices keeps fails_value = NaN (not 0)
    return (out.groupby(["date", "ticker"], as_index=False)[["fails_quantity", "fails_value"]]
            .sum(min_count=1))


# --------------------------------------------------------------------------- #
# IO: cache/download + incremental state                                        #
# --------------------------------------------------------------------------- #
def _cache_dir(context: Context) -> Path:
    d = context.paths["DATA_STORE"] / "sec_fails_to_deliver"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _ensure_zip(period: str, cache_dir: Path) -> Path | None:
    path = cache_dir / f"cnsfails{period}.zip"
    if path.exists() and path.stat().st_size > 0:
        return path
    url = SEC_FTD_URL_TEMPLATE.format(period=period)
    try:
        r = requests.get(url, headers=_HEADERS, timeout=180, stream=True)
    except Exception as e:
        logger.warning("FTD %s download failed: %s", period, e)
        return None
    if r.status_code != 200:                       # not-yet-published / missing semi-month
        logger.info("FTD %s not available (HTTP %s)", period, r.status_code)
        return None
    tmp = path.with_suffix(".part")
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    tmp.replace(path)
    return path


def _read_zip(path: Path) -> str | None:
    try:
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            return z.open(names[0]).read().decode("latin-1", errors="replace") if names else None
    except zipfile.BadZipFile:
        logger.warning("FTD %s: corrupt zip -> deleting so it re-downloads", path.name)
        path.unlink(missing_ok=True)
        return None


def _ingested_periods(store) -> set[str]:
    if not store.exists(_TABLE):
        return set()
    with store.engine.connect() as c:
        return set(pd.read_sql(text(f'SELECT DISTINCT period FROM "{_TABLE}"'), c)
                   ["period"].dropna())


def fetch_fails_to_deliver(context: Context, tickers: list[str]) -> int:
    """Download (cached) the semi-monthly SEC Fails-to-Deliver files over
    `years_history`, keep the universe, upsert to `fails_to_deliver`. Returns rows
    upserted. Incremental: a file already in the DB is skipped (no re-download)
    unless the universe gained tickers (then cached files are re-parsed)."""
    store = context.store
    universe = {str(t).upper() for t in tickers}
    years_history = context.config.data_extract.years_history + 1
    cache_dir = _cache_dir(context)

    done = _ingested_periods(store)
    new_tickers = universe - load_processed_universe(cache_dir, _TABLE)   # empty once converged
    if new_tickers:
        logger.info("FTD: %d new/changed tickers -> re-parsing cached files", len(new_tickers))

    saved = 0
    for period in tqdm(_periods(years_history), desc="SEC fails-to-deliver"):
        if period in done and not new_tickers:
            continue
        path = _ensure_zip(period, cache_dir)
        if path is None:
            continue
        raw = _read_zip(path)
        if raw is None:
            continue
        df = _parse_ftd(raw)
        df = df[df["ticker"].isin(universe)]
        if df.empty:
            continue
        df["period"] = period
        saved += store.save(_TABLE, df[[c for c in _OUT_COLS if c in df.columns]])

    save_processed_universe(cache_dir, _TABLE, universe)   # so a converged re-run skips
    logger.warning("fails_to_deliver: upserted %d rows (%d files scanned)",
                   saved, len(_periods(years_history)))
    return saved
