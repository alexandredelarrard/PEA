"""
fetch_short_interest.py (src/data_extract/utils/institutionals/fetch_short_interest.py)
---------------------------------------------------------------------------------
FINRA RegSHO consolidated daily short-sale VOLUME (`CNMSshvol` files; not reported short interest).
Each day-file line is stored as filed in `sec_short_volume_security`, keyed on the symbol as filed (case kept:
`BACpB` is a preferred, `BRK/A` a class), source columns only. A line is kept when its FTD spelling (`finra_key`) is a
symbol of a stored FTD line, or its symbol is a universe ticker or an equity lineage symbol of a universe company
(`equity_symbol_rows`: not noise, not the Notes cover page alone), the symbols the cube's lineage fallback reads. The
institutionals cube maps the lines to companies and sums the classes (`security_series.short_volume_series`). One file
covers every ticker, so the day files read are the business days from the earlier of the table's latest day and today
less the table's overlap, to today; `full` starts at the table's source start. FINRA day files are not cached: a symbol
newly in scope gets its history from `short-interest --full`. Missing the Lit exchange volumes.
"""

from __future__ import annotations

import io
import logging
import time
from collections.abc import Collection, Sequence
from typing import cast

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from src.constants.constants import BROWSER_HEADERS, DATE_FORMAT_COMPACT
from src.context import Context
from src.data_extract.utils.common.identity_store import load_stored_lineage
from src.data_store.schema import Resume, Tables
from src.utils.security_series import equity_symbol_rows, finra_key
from src.utils.string import run_date, squash, yahoo_symbol

_URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{yyyymmdd}.txt"

#: Column order of `sec_short_volume_security`: the day-file columns as filed.
RAW_COLUMNS = ("date", "source_symbol", "market", "short_volume", "short_exempt_volume", "total_volume")
_VOLUMES = ["short_volume", "short_exempt_volume", "total_volume"]

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Pure parse and scope (unit-tested)                                            #
# --------------------------------------------------------------------------- #
def _parse_regsho(text: str) -> pd.DataFrame:
    """Validate the entire served file before retaining its raw symbols and quantities."""
    lines = text.strip().splitlines()
    header = "Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market"
    legacy = "Date|Symbol|ShortVolume|TotalVolume|Market"
    supplied = lines[0].strip().lstrip("﻿") if lines else ""
    if len(lines) < 2 or supplied not in (header, legacy) or not lines[-1].strip().isdigit():
        raise ValueError("RegSHO missing header or record-count trailer")
    records = lines[1:-1]
    if int(lines[-1]) != len(records) or any(len(row.split("|")) != len(supplied.split("|")) for row in records):
        raise ValueError("RegSHO record-count or line-width mismatch")
    df = pd.read_csv(io.StringIO("\n".join([supplied, *records])), sep="|", dtype=str, keep_default_na=False)
    if "ShortExemptVolume" not in df:
        df["ShortExemptVolume"] = pd.NA
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df["Date"].str.strip(), format="%Y%m%d", errors="coerce"),
            "source_symbol": df["Symbol"].str.strip(),
            "market": df["Market"].str.strip() if "Market" in df.columns else "",
            **{
                column: pd.to_numeric(df[name], errors="coerce")
                for column, name in zip(_VOLUMES, ("ShortVolume", "ShortExemptVolume", "TotalVolume"), strict=True)
            },
        }
    )
    validated = _VOLUMES if supplied == header else ["short_volume", "total_volume"]
    quantities = out[validated].to_numpy(dtype=float)
    if out["date"].isna().any() or out["source_symbol"].eq("").any() or not np.isfinite(quantities).all() or (quantities < 0).any():
        raise ValueError("RegSHO invalid date, symbol or finite nonnegative quantity")
    if out["short_volume"].gt(out["total_volume"]).any():
        raise ValueError("RegSHO short volume exceeds reported total volume")
    if supplied == header and out["short_exempt_volume"].gt(out["short_volume"]).any():
        raise ValueError("RegSHO exempt volume exceeds its short-volume total")
    out = out.groupby(["date", "source_symbol"], as_index=False, sort=False).agg(
        market=("market", "first"),
        short_volume=("short_volume", "sum"),
        short_exempt_volume=("short_exempt_volume", "sum"),
        total_volume=("total_volume", "sum"),
    )
    if supplied == legacy:
        out["short_exempt_volume"] = np.nan
    if not np.isfinite(out[validated].to_numpy(dtype=float)).all():
        raise ValueError("RegSHO grouped source quantities overflowed")
    return out[list(RAW_COLUMNS)].reset_index(drop=True)


def short_scope(lineage: pd.DataFrame | None, ftd_symbols: Collection[str], tickers: Collection[str]) -> tuple[frozenset[str], frozenset[str]]:
    """`(FTD keys, symbols)` a stored line matches: the squashed symbols of the stored FTD lines (matched on `finra_key`),
    and the universe tickers plus the equity lineage symbols of the universe companies (matched on `yahoo_symbol`)."""
    keys = frozenset(key for key in map(squash, ftd_symbols) if key)
    lineage_symbols = set(equity_symbol_rows(lineage)["symbol"].map(yahoo_symbol)) if lineage is not None and not lineage.empty else set()
    return keys, frozenset(lineage_symbols | set(tickers)) - {""}


# --------------------------------------------------------------------------- #
# IO                                                                             #
# --------------------------------------------------------------------------- #
def _fetch_day(
    day: pd.Timestamp,
    session: requests.Session | None = None,
) -> str | None:
    """Fetch one date, optionally reusing a run-scoped HTTP connection pool."""
    url = _URL.format(yyyymmdd=day.strftime(DATE_FORMAT_COMPACT))
    r = session.get(url, headers=BROWSER_HEADERS, timeout=30) if session is not None else requests.get(url, headers=BROWSER_HEADERS, timeout=30)
    return r.text if r.status_code == 200 else None


def _plan_days(context: Context, full: bool, as_of: pd.Timestamp | None) -> pd.DatetimeIndex:
    """The business days to read, up to today: from the earlier of the table's latest day and today less its overlap
    (one file serves every ticker, so the latest day is table-wide), or from the source start when `full` or empty."""
    resume = cast(Resume, Tables.sec_short_volume_security.resume)
    start_date = run_date(as_of)
    latest = context.store.max_date(Tables.sec_short_volume_security)
    start = (
        pd.Timestamp(resume.source_start)
        if full or latest is None
        else min(pd.Timestamp(latest), start_date - pd.Timedelta(days=resume.overlap_days))
    )
    return pd.bdate_range(start, start_date)


def _validated_regsho_day(day: pd.Timestamp, session: requests.Session) -> pd.DataFrame | None:
    text = _fetch_day(day, session)
    if not text:
        return None
    lines = _parse_regsho(text)
    if not lines["date"].eq(day.normalize()).all():
        raise ValueError("RegSHO source date does not match requested day")
    return lines


def _read_regsho_days(
    days: Sequence[pd.Timestamp], keys: frozenset[str], symbols: frozenset[str], pause: float
) -> tuple[pd.DataFrame, list[pd.Timestamp], list[str]]:
    """Stage fully validated source units; failed units never enter the write."""
    frames: list[pd.DataFrame] = []
    successful: list[pd.Timestamp] = []
    errors: list[str] = []
    with requests.Session() as session:
        for day in tqdm(days, "short_interest OffExchange - fetch RegSHO"):
            try:
                lines = _validated_regsho_day(day, session)
            except Exception as exc:
                logger.error("RegSHO %s failed: %s", day.date(), exc)
                errors.append(f"{day.date()} ({exc})")
                continue
            if lines is None:
                continue
            successful.append(day)
            in_scope = lines["source_symbol"].map(finra_key).isin(keys) | lines["source_symbol"].map(yahoo_symbol).isin(symbols)
            frames.append(lines.loc[in_scope])
            time.sleep(pause)
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(RAW_COLUMNS))
    return raw, successful, errors


def fetch_short_interest(
    context: Context,
    tickers: list[str],
    pause: float = 0.05,
    full: bool = False,
    as_of: pd.Timestamp | None = None,
) -> None:
    """Fetch the `_plan_days` RegSHO day files and store their in-scope lines as filed (a keyed upsert).
    `full` aborts before any write on a fetch error or when no day is served."""
    # load: the lineage and the stored FTD symbols the scope is built from; the planned days
    lineage = load_stored_lineage(context)
    ftd = Tables.sec_fails_to_deliver_security
    ftd_symbols = [str(s) for s in context.store.distinct(ftd, "source_symbol") if s is not None] if context.store.exists(ftd) else []
    days = list(_plan_days(context, full, as_of))
    keys, symbols = short_scope(lineage, ftd_symbols, tickers)
    logger.info(f"Fetching {len(days)} RegSHO day-file(s) for {len(tickers)} tickers ({len(keys)} FTD key(s), {len(symbols)} lineage symbol(s))")
    raw, successful, errors = _read_regsho_days(days, keys, symbols, pause)

    if full and (errors or not successful):
        raise RuntimeError(f"RegSHO full: {len(errors)} fetch error(s) {errors[:5]}, {len(successful)} served; aborting before any write")

    written = context.store.save(Tables.sec_short_volume_security, raw[list(RAW_COLUMNS)]) if not raw.empty else 0
    logger.info(
        f"Saved {written} RegSHO line(s) of {raw['source_symbol'].nunique()} symbol(s) over {len(successful)} day(s) to '{Tables.sec_short_volume_security}'"
    )
