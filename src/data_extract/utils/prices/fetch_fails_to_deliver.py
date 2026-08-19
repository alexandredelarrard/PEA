"""
fetch_fails_to_deliver.py (src/data_extract/utils/prices/fetch_fails_to_deliver.py)
------------------------------------------------------------------------------------
Extracts SEC Fails-to-Deliver (FTD) settlement-fail data from semi-monthly SEC 
text releases (`YYYYMMa` / `YYYYMMb`). Serves as a key signal for settlement 
stress and short-squeeze risk, complementing FINRA RegSHO short-volume data.

Data Grain:
- Daily records stored long as (date, ticker, fails_quantity, fails_value).

Key Guardrails & Architecture:
- Point-in-Time Lag: Released with a 1–2 month publication delay; backtests 
  must account for reporting lag to avoid look-ahead bias.
- Isolated Table: Stored in its own table separate from `short_interest` to 
  prevent lagged semi-monthly releases from corrupting incremental sync dates.
- File-Level Incremental Sync: Caches raw files in `data/sec_fails_to_deliver/` 
  and skips re-downloading existing files unless the ticker universe expands.
"""

from __future__ import annotations

import io
import logging

import pandas as pd
from tqdm import tqdm

from src.data_store.schema import Tables
from src.constants.constants import (
    SEC_FTD_URL_TEMPLATE, SEC_FTD_LEGACY_URL_TEMPLATE,
    SEC_FTD_LEGACY_LAST_PERIOD, SEC_FTD_FIRST_YEAR)
from src.context import Context
from src.data_extract.utils.common.bulk_cache import (
    cache_dir, ensure_zip, ingested_periods, read_zip_text,
)
from src.data_extract.utils.common.run_manifest import record_run
from src.data_extract.utils.common.sec_utils import (
    load_processed_universe)

logger = logging.getLogger(__name__)

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

def _period_urls(period: str) -> tuple[str, ...]:
    """Download URL(s) for a semi-monthly period, path chosen by date: the FOIA
    'legacy' path for <= 2017-06a, the current path for >= 2017-06b. The other path
    is tried as a fallback (boundary / occasional re-issued files live on both).
    Fixed-width 'YYYYMMx' tags sort chronologically, so a string compare is safe."""
    modern = SEC_FTD_URL_TEMPLATE.format(period=period)
    legacy = SEC_FTD_LEGACY_URL_TEMPLATE.format(period=period)
    return (legacy, modern) if period <= SEC_FTD_LEGACY_LAST_PERIOD else (modern, legacy)


def fetch_fails_to_deliver(context: Context, tickers: list[str], years_history:int = 15) -> int:
    """Download (cached) the semi-monthly SEC Fails-to-Deliver files over
    `years_history`, keep the universe, upsert to `fails_to_deliver`. Returns rows
    upserted. Incremental: a file already in the DB is skipped (no re-download)
    unless the universe gained tickers (then cached files are re-parsed)."""

    cache = cache_dir(context, "sec_fails_to_deliver")
    done = ingested_periods(context, "fails_to_deliver")
    new_tickers = set(tickers) - load_processed_universe(cache, Tables.fails_to_deliver)   # empty once converged
    if new_tickers:
        logger.info("FTD: %d new/changed tickers -> re-parsing cached files", len(new_tickers))

    saved = 0
    for period in tqdm(_periods(years_history +1), desc="SEC fails-to-deliver"):
        if period in done and not new_tickers:
            continue
        path = ensure_zip(cache / f"cnsfails{period}.zip", _period_urls(period),
                          label=f"FTD {period}", timeout=180, log=logger)
        if path is None:
            continue
        raw = read_zip_text(path, log=logger)
        if raw is None:
            continue
        df = _parse_ftd(raw)
        df = df[df["ticker"].isin(tickers)]
        if df.empty:
            continue
        df["period"] = period
        context.store.save(Tables.fails_to_deliver, df[[c for c in _OUT_COLS if c in df.columns]])
        saved +=df.shape[0]

    logger.info(f"fails_to_deliver completed ({len(_periods(years_history))} files scanned) +{saved}")
    record_run(context, Tables.fails_to_deliver, len(tickers), saved)
