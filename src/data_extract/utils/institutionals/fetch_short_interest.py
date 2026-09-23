"""
fetch_short_interest.py (src/data_extract/utils/institutionals/fetch_short_interest.py)
---------------------------------------------------------------------------------
FINRA RegSHO CONSOLIDATED short-sale volume (`CNMSshvol` daily files, free, no auth). This is
short-selling PRESSURE (daily short vs total volume) -- a proxy for short interest, NOT reported
short interest. Saved long [date, ticker, short_volume, total_volume]; each day's file is
disseminated the next morning, so the aggregation step lags it one trading day (point-in-time).

Missing the Lit exchange short volumes from NYSE / Nasdaq and CBOE equities.

⚠ THE CDN KEEPS A ROLLING ~8-YEAR WINDOW. There is no deep history to backfill, and this is a
RETENTION limit at the source, not a gap in this fetcher -- so no future reader should spend a
day trying. Probed at the URL pattern below on 2026-09-08:

    20100415 403 · 20130415 403 · 20160415 403 · 20170103 403 · 20180112 403 · 20180712 403
    20180731 403 · 20180801 200 · 20180814 200 · 20190701 200 · ... · 20260901 200

Binary-searched to the day: last 403 is 2018-07-31, first 200 is 2018-08-01 -- a boundary on a
month start, ~8.10 years before the probe date, which is why it MOVES FORWARD. Rows already
stored below it cannot be re-fetched if lost.

The stored `min(date)` is 2017-12-29, which is NOT the history start: 20171229 is a lone file
that survives outside the window (probed 200 while every other 2017 and early-2018 date returns
403). Treating it as a floor would claim eight months of coverage that do not exist.

NAMING: the table is `sec_short_interest` but it holds short-sale VOLUME. The misnomer is a live
table with consumers, so it is fixed at the feature level (`ic_shortvol_*`), not here.
"""

from __future__ import annotations

import io
import logging
import time

import pandas as pd
import requests
from tqdm import tqdm

from src.constants.constants import _HEADERS, DATE_FORMAT_COMPACT
from src.context import Context
from src.data_extract.utils.common.identity import (
    Identity,
    load_identity,
    log_symbol_resolutions,
    resolve_symbol_rows,
)
from src.data_extract.utils.common.run_manifest import record_run
from src.data_store.errors import TableEmptyError
from src.data_store.schema import Tables

_URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{yyyymmdd}.txt"

logger = logging.getLogger(__name__)


def _today() -> pd.Timestamp:
    """Normalised current day, isolated for deterministic window tests."""
    return pd.Timestamp.today().normalize()


def _parse_regsho(text: str) -> pd.DataFrame:
    """Parse one RegSHO file while retaining its historical source symbol."""

    if not text or "|" not in text:
        return pd.DataFrame(columns=["date", "source_symbol", "short_volume", "total_volume"])

    df = pd.read_csv(io.StringIO(text), sep="|")
    df = df[df.get("Symbol").notna()] if "Symbol" in df.columns else df.iloc[0:0]
    if df.empty:
        return pd.DataFrame(columns=["date", "source_symbol", "short_volume", "total_volume"])
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df["Date"].astype(str), format="%Y%m%d", errors="coerce"),
            "source_symbol": (df["Symbol"].astype(str).str.upper().str.replace(".", "-", regex=False).str.strip()),
            "short_volume": pd.to_numeric(df["ShortVolume"], errors="coerce"),
            "total_volume": pd.to_numeric(df["TotalVolume"], errors="coerce"),
        }
    ).dropna(subset=["date", "source_symbol"])
    return out.groupby(["date", "source_symbol"], as_index=False)[["short_volume", "total_volume"]].sum()


def _fetch_day(
    day: pd.Timestamp,
    session: requests.Session | None = None,
) -> str | None:
    """Fetch one date, optionally reusing a run-scoped HTTP connection pool."""
    url = _URL.format(yyyymmdd=day.strftime(DATE_FORMAT_COMPACT))
    r = session.get(url, headers=_HEADERS, timeout=30) if session is not None else requests.get(url, headers=_HEADERS, timeout=30)
    return r.text if r.status_code == 200 else None


def _resume_day(context: Context, years_history: int = 15, full: bool = False) -> pd.Timestamp:
    """The first day to download: the day after the GLOBAL stored max, or the full
    `years_history` window on a cold table.

    Global and not per-ticker on purpose. A RegSHO day-file carries every symbol at once, so
    one lagging ticker would drag the whole download back to its own last date and re-fetch
    days already stored for all the others.
    """

    today = _today()
    stored_max = context.store.max_date(Tables.short_interest)
    if stored_max is None or full:
        return today - pd.DateOffset(years=years_history)
    return stored_max + pd.Timedelta(days=1)


def _canonicalise_regsho(
    context: Context,
    frame: pd.DataFrame,
    identity: Identity,
    universe: frozenset[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resolve historical RegSHO symbols and aggregate to `(ticker, date)`."""
    accepted, unresolved = resolve_symbol_rows(identity, frame, universe)
    log_symbol_resolutions(context, "RegSHO", accepted, unresolved, universe=universe)
    if accepted.empty:
        return pd.DataFrame(columns=["date", "ticker", "short_volume", "total_volume"]), unresolved
    grouped = accepted.groupby(["ticker", "date"], as_index=False)[["short_volume", "total_volume"]].sum()
    return grouped[["date", "ticker", "short_volume", "total_volume"]], unresolved


def _stored_rows(
    context: Context,
    *,
    until: pd.Timestamp | None = None,
    dates: list[pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """Load a projected, date-scoped RegSHO slice; empty on a cold table."""
    if not context.store.exists(Tables.short_interest):
        return pd.DataFrame(columns=["date", "ticker", "short_volume", "total_volume"])
    kwargs: dict[str, object] = {
        "columns": ["date", "ticker", "short_volume", "total_volume"],
    }
    if until is not None:
        kwargs["until"] = until
    if dates is not None:
        if not dates:
            return pd.DataFrame(columns=["date", "ticker", "short_volume", "total_volume"])
        kwargs["where"] = {"date": dates}
    try:
        loaded = context.store.load(Tables.short_interest, **kwargs)
    except TableEmptyError:
        return pd.DataFrame(columns=["date", "ticker", "short_volume", "total_volume"])
    loaded["date"] = pd.to_datetime(loaded["date"])
    return loaded


def _reconcile_legacy(
    context: Context,
    legacy: pd.DataFrame,
    identity: Identity,
    universe: frozenset[str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Relabel proven legacy rows, remove outsiders and preserve unresolved keys."""
    if legacy.empty:
        return legacy, {"retained": 0, "relabelled": 0, "removed": 0, "unresolved": 0}
    source = legacy.rename(columns={"ticker": "source_symbol"})
    accepted, unresolved = resolve_symbol_rows(identity, source, universe)
    log_symbol_resolutions(context, "RegSHO legacy", accepted, unresolved, universe=universe)
    relabelled = int((accepted["source_symbol"] != accepted["ticker"]).sum())
    removed = int(unresolved["resolution_verdict"].eq("entity_not_in_universe").sum())
    preserve = unresolved[~unresolved["resolution_verdict"].eq("entity_not_in_universe")].copy()
    preserve["ticker"] = preserve["source_symbol"]
    columns = ["date", "ticker", "short_volume", "total_volume"]
    combined = pd.concat([accepted[columns], preserve[columns]], ignore_index=True)
    if not combined.empty:
        combined = combined.groupby(["ticker", "date"], as_index=False)[["short_volume", "total_volume"]].sum()
    stats = {
        "retained": len(combined),
        "relabelled": relabelled,
        "removed": removed,
        "unresolved": len(preserve),
    }
    return combined[columns], stats


def _validate_full_frame(frame: pd.DataFrame, universe: frozenset[str]) -> None:
    """Validate the reconciled replacement before the one destructive write."""
    if frame.empty:
        raise ValueError("RegSHO full refresh staged no rows")
    if frame[["ticker", "date"]].isna().any().any():
        raise ValueError("RegSHO full refresh staged a null key")
    duplicate = frame.duplicated(["ticker", "date"], keep=False)
    if duplicate.any():
        raise ValueError(f"RegSHO full refresh staged {int(duplicate.sum())} duplicate key row(s)")
    outside = sorted(set(frame["ticker"]) - set(universe))
    if outside:
        raise ValueError(f"RegSHO full refresh staged non-universe ticker(s): {outside}")


def fetch_short_interest(
    context: Context,
    tickers: list[str],
    years_history: int = 15,
    pause: float = 0.05,
    full: bool = False,
    identity: Identity | None = None,
) -> None:
    """Resolve RegSHO point-in-time; full mode preserves unrecoverable stored dates."""

    today = _today()
    days = pd.bdate_range(_resume_day(context, years_history, full), today)
    logger.info(f"Fetching {len(days)} RegSHO day-file(s) for {len(tickers)} tickers")
    resolver = identity or load_identity(context)
    universe = frozenset(str(ticker).strip().upper() for ticker in tickers)
    candidates = resolver.candidate_symbols(universe)

    frames: list[pd.DataFrame] = []
    successful_days: list[pd.Timestamp] = []
    failed_days: list[pd.Timestamp] = []
    session = requests.Session()
    for day in tqdm(days, "short_interest OffExchange - fetch RegSHO"):
        try:
            text = _fetch_day(day, session)
        except Exception as e:  # one bad day must not abort the run
            logger.error(f"RegSHO {day.date()} failed: {e}")
            failed_days.append(day)
            continue
        if not text:
            failed_days.append(day)
            continue
        successful_days.append(day)
        df_day = _parse_regsho(text)
        df_day = df_day[df_day["source_symbol"].isin(candidates)]
        if not df_day.empty:
            frames.append(df_day)
        time.sleep(pause)
    session.close()

    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["date", "source_symbol", "short_volume", "total_volume"])
    fresh, unresolved = _canonicalise_regsho(context, raw, resolver, universe)

    if not full:
        context.store.save(Tables.short_interest, fresh)
        logger.info(f"Saved {len(fresh)} new short-volume rows to DB table " f"'{Tables.short_interest}'")
        logger.info(f"RegSHO: {len(unresolved)} unresolved raw row(s) excluded")
        record_run(context, Tables.short_interest, len(tickers), len(fresh))
        return

    if not successful_days:
        raise RuntimeError("RegSHO full refresh retrieved no source date; preserving the table by aborting")
    earliest = min(successful_days)
    legacy = _stored_rows(context, until=earliest - pd.Timedelta(days=1))
    corrected_legacy, legacy_stats = _reconcile_legacy(context, legacy, resolver, universe)
    failed_stored = _stored_rows(context, dates=[day for day in failed_days if day >= earliest])
    complete = pd.concat([corrected_legacy, failed_stored, fresh], ignore_index=True)
    if not complete.empty:
        complete = complete.groupby(["ticker", "date"], as_index=False)[["short_volume", "total_volume"]].sum()
    _validate_full_frame(complete, universe)
    written = context.store.replace(Tables.short_interest, complete)

    logger.info(
        f"RegSHO full: retained={legacy_stats['retained']} "
        f"relabelled={legacy_stats['relabelled']} removed={legacy_stats['removed']} "
        f"legacy_unresolved={legacy_stats['unresolved']} refreshed={len(fresh)} "
        f"fresh_unresolved={len(unresolved)} preserved_failed_date_rows={len(failed_stored)}"
    )
    record_run(context, Tables.short_interest, len(tickers), written, is_full_rescan=True)
