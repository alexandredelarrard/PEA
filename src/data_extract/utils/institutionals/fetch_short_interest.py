"""
fetch_short_interest.py (src/data_extract/utils/institutionals/fetch_short_interest.py)
---------------------------------------------------------------------------------
FINRA RegSHO consolidated daily short-sale VOLUME (`CNMSshvol` files; not reported short interest).
Each day-file line is stored as filed in `sec_short_volume_security`, keyed on the symbol as filed (case kept:
`BACpB` is a preferred, `BRK/A` a class), source columns only. A line is kept when its symbol is an equity lineage
symbol of a universe company (`equity_symbol_rows`: not noise, not the Notes cover page alone) and its day lies inside
that symbol's lineage window or inside the span of the stored FTD lines of that symbol (a reused symbol brings no other
issuer's days), or, undated, when its FTD spelling (`finra_key`) is a symbol of a stored FTD line or a universe ticker
that is no lineage symbol (a CUSIP-6 family class or preferred). The institutionals cube maps the lines to companies and sums the classes (`security_series.short_volume_series`). One file
covers every ticker, so the day files read are the business days from the earlier of the table's latest day and today
less the table's overlap, to today; `full`, an empty table and the first run of a new half-month (today's half-month,
1st-15th or 16th-end, later than the latest stored day's) start at the table's source start, a keyed upsert. FINRA day
files are not cached, so that half-monthly full refresh is how a symbol newly in scope gets its history. Missing the
Lit exchange volumes.
"""

from __future__ import annotations

import io
import logging
import time
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from src.constants.constants import BROWSER_HEADERS, DATE_FORMAT_COMPACT
from src.context import Context
from src.data_extract.utils.common.identity_store import load_stored_lineage
from src.data_extract.utils.common.security_lines import in_window, symbol_windows
from src.data_store.schema import Resume, Tables
from src.utils.security_series import finra_key
from src.utils.string import run_date, squash

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


@dataclass(frozen=True)
class ShortScope:
    """What a FINRA line matches: a lineage symbol inside one of its `windows`, or one of the undated `keys`."""

    windows: pd.DataFrame
    keys: frozenset[str]


def short_scope(lineage: pd.DataFrame | None, ftd_spans: pd.DataFrame, tickers: Collection[str]) -> ShortScope:
    """The scope of the FINRA lines. A lineage symbol is dated: the windows of the equity lineage symbols of the universe
    companies, plus each stored FTD symbol's span of trade dates (`ftd_spans`: `source_symbol`, `first`, `last`; those
    lines are in scope by construction). The other stored FTD symbols and universe tickers are undated keys."""
    windows = symbol_windows(lineage) if lineage is not None and not lineage.empty else pd.DataFrame(columns=["key", "valid_from", "valid_to"])
    spans = pd.DataFrame(
        {
            "key": ftd_spans["source_symbol"].map(squash),
            "valid_from": pd.to_datetime(ftd_spans["first"]),
            "valid_to": pd.to_datetime(ftd_spans["last"]) + pd.Timedelta(days=1),
        }
    )
    keys = {key for key in [*spans["key"], *map(squash, tickers)] if key} - set(windows["key"])
    return ShortScope(windows=pd.concat([windows, spans.loc[spans["key"].ne("")]], ignore_index=True), keys=frozenset(keys))


def in_short_scope(lines: pd.DataFrame, scope: ShortScope) -> pd.Series:
    """Per FINRA line, whether its symbol (as filed or in its FTD spelling) has a scope window covering the line's day,
    or its FTD spelling is an undated scope key."""
    finra = lines["source_symbol"].map(finra_key)
    filed = lines["source_symbol"].map(squash)
    dated = in_window(filed, lines["date"], scope.windows) | in_window(finra, lines["date"], scope.windows)
    return dated | finra.isin(scope.keys)


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


def _half_month(day: pd.Timestamp) -> tuple[int, int, bool]:
    """The half-month of `day`: (year, month, whether it is the 16th or later)."""
    return day.year, day.month, day.day > 15


def _plan_days(context: Context, full: bool, as_of: pd.Timestamp | None) -> pd.DatetimeIndex:
    """The business days to read, up to today: from the earlier of the table's latest day and today less its overlap
    (one file serves every ticker, so the latest day is table-wide), or from the source start when `full`, when the table
    is empty, or on the first run of a half-month (today's half-month is later than the latest stored day's), so a
    symbol newly in scope gets its history."""
    # load: the table's latest stored day
    resume = cast(Resume, Tables.sec_short_volume_security.resume)
    start_date = run_date(as_of)
    latest = context.store.max_date(Tables.sec_short_volume_security)

    # plan: a full refresh from the source start, else the incremental days
    refresh = latest is not None and _half_month(start_date) > _half_month(pd.Timestamp(latest))
    if refresh and not full:
        logger.info(
            "RegSHO: first run of a new half-month (latest stored day %s): full refresh from %s", pd.Timestamp(latest).date(), resume.source_start
        )
    if full or refresh or latest is None:
        return pd.bdate_range(pd.Timestamp(resume.source_start), start_date)
    return pd.bdate_range(min(pd.Timestamp(latest), start_date - pd.Timedelta(days=resume.overlap_days)), start_date)


def _validated_regsho_day(day: pd.Timestamp, session: requests.Session) -> pd.DataFrame | None:
    text = _fetch_day(day, session)
    if not text:
        return None
    lines = _parse_regsho(text)
    if not lines["date"].eq(day.normalize()).all():
        raise ValueError("RegSHO source date does not match requested day")
    return lines


def _read_regsho_days(days: Sequence[pd.Timestamp], scope: ShortScope, pause: float) -> tuple[pd.DataFrame, list[pd.Timestamp], list[str]]:
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
            frames.append(lines.loc[in_short_scope(lines, scope)])
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
    # load: the lineage and the stored FTD symbols' trade-date spans the scope is built from; the planned days
    lineage = load_stored_lineage(context)
    spans = {"first": ("min", "trade_date"), "last": ("max", "trade_date")}
    ftd_spans = context.store.summarize(Tables.sec_fails_to_deliver_security, "source_symbol", spans)
    days = list(_plan_days(context, full, as_of))
    scope = short_scope(lineage, ftd_spans, tickers)
    logger.info(
        f"Fetching {len(days)} RegSHO day-file(s) for {len(tickers)} tickers ({len(scope.keys)} undated key(s), {len(scope.windows)} dated window(s))"
    )
    raw, successful, errors = _read_regsho_days(days, scope, pause)

    if full and (errors or not successful):
        raise RuntimeError(f"RegSHO full: {len(errors)} fetch error(s) {errors[:5]}, {len(successful)} served; aborting before any write")

    written = context.store.save(Tables.sec_short_volume_security, raw[list(RAW_COLUMNS)]) if not raw.empty else 0
    logger.info(
        f"Saved {written} RegSHO line(s) of {raw['source_symbol'].nunique()} symbol(s) over {len(successful)} day(s) to '{Tables.sec_short_volume_security}'"
    )
