"""
fetch_short_interest.py (src/data_extract/utils/institutionals/fetch_short_interest.py)
---------------------------------------------------------------------------------
FINRA RegSHO consolidated daily short-sale VOLUME (`CNMSshvol` files) -> `short_interest`
[date, ticker, short_volume, total_volume]; despite the table name it is not reported short
interest. Each day's file is disseminated the next morning, so aggregation lags it one trading day.
One day file covers every symbol, so a run reads the union of the sessions any universe key needs
(`resume.series_windows`: forward overlap, new keys in full until their days reach the source start) plus the missing DAYS: calendar sessions
inside the stored span on which no key has a row. A day with rows for some keys was read; `repair`
re-reads those per-key gaps once.
The CDN keeps only a rolling ~8-year window, so stored rows older than it cannot be re-fetched;
`full` mode therefore preserves stored dates the source no longer serves. Only an unscoped `full` run
replaces the table; a scoped one (`-t`) re-reads its tickers' days and upserts them, touching no other key.
Missing the Lit exchange short volumes from NYSE / Nasdaq and CBOE equities.
"""

from __future__ import annotations

import io
import logging
import time
from typing import Any, cast

import pandas as pd
import requests
from tqdm import tqdm

from src.constants.constants import BROWSER_HEADERS, DATE_FORMAT_COMPACT
from src.context import Context
from src.data_extract.utils.common.identity import (
    Identity,
    load_identity,
    log_symbol_resolutions,
    resolve_symbol_rows,
)
from src.data_extract.utils.common.resume import document_floor, series_windows, session_dates, trading_calendar
from src.data_extract.utils.common.sessions import last_completed_session
from src.data_store.errors import TableEmptyError
from src.data_store.schema import Tables
from src.utils.universe import load_universe_tickers

_URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{yyyymmdd}.txt"

logger = logging.getLogger(__name__)


def _parse_regsho(text: str) -> pd.DataFrame:
    """Parse one RegSHO file while retaining its historical source symbol."""

    if not text or "|" not in text:
        return pd.DataFrame(columns=["date", "source_symbol", "short_volume", "total_volume"])

    df = pd.read_csv(io.StringIO(text), sep="|")
    df = df[df["Symbol"].notna()] if "Symbol" in df.columns else df.iloc[0:0]
    if df.empty:
        return pd.DataFrame(columns=["date", "source_symbol", "short_volume", "total_volume"])
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df["Date"].astype(str), format="%Y%m%d", errors="coerce"),
            "source_symbol": (df["Symbol"].astype(str).str.upper().str.replace(".", "-", regex=False).str.replace("/", "-", regex=False).str.strip()),
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
    r = session.get(url, headers=BROWSER_HEADERS, timeout=30) if session is not None else requests.get(url, headers=BROWSER_HEADERS, timeout=30)
    return r.text if r.status_code == 200 else None


def _missing_days(context: Context, calendar: pd.DatetimeIndex, floor: pd.Timestamp) -> pd.DatetimeIndex:
    """Calendar sessions inside the stored span (from `floor`) on which no key has a row: day files never stored."""
    stored = pd.DatetimeIndex(pd.to_datetime(context.store.distinct(Tables.short_interest, "date"))).normalize()
    if stored.empty:
        return pd.DatetimeIndex([])
    span = calendar[(calendar >= max(stored.min(), floor)) & (calendar <= stored.max())]
    return span.difference(stored)


def _plan_days(
    context: Context, tickers: list[str], years_history: int, full: bool, as_of: pd.Timestamp | None, *, repair: bool = False
) -> pd.DatetimeIndex:
    """The day files to read: every key's windows plus the never-stored days, as trading sessions (business
    days past the calendar). `repair` adds each key's own interior gaps (a one-time pass, not nightly)."""
    run_date = pd.Timestamp(as_of if as_of is not None else pd.Timestamp.today()).normalize()
    until = last_completed_session(as_of)
    calendar = trading_calendar(context)
    work = series_windows(
        context, Tables.short_interest, tickers, run_date, until=until, years_history=years_history, full=full, calendar=calendar if repair else None
    )
    days = _missing_days(context, calendar, document_floor(Tables.short_interest, run_date, years_history))
    for since, end, _keys in work.groups():
        days = days.union(session_dates(calendar, since, end))
    return days


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
        loaded = context.store.load(Tables.short_interest, **cast(dict[str, Any], kwargs))
    except TableEmptyError:
        return pd.DataFrame(columns=["date", "ticker", "short_volume", "total_volume"])
    assert loaded is not None
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
    proven_exclusions = {"entity_not_in_universe", "redundant_share_class"}
    removed_mask = unresolved["resolution_verdict"].isin(proven_exclusions)
    removed = int(removed_mask.sum())
    preserve = unresolved[~removed_mask].copy()
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
    years_history: int,
    pause: float = 0.05,
    full: bool = False,
    identity: Identity | None = None,
    as_of: pd.Timestamp | None = None,
    repair: bool = False,
) -> None:
    """Resolve RegSHO point-in-time; an unscoped full run replaces the table and preserves unrecoverable stored dates,
    a scoped one upserts its own tickers only; `repair` re-reads per-key gap days."""

    universe = frozenset(str(ticker).strip().upper() for ticker in tickers)
    days = _plan_days(context, sorted(universe), years_history, full, as_of, repair=repair)
    logger.info(f"Fetching {len(days)} RegSHO day-file(s) for {len(tickers)} tickers")

    resolver = identity or load_identity(context)
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

    df_raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["date", "source_symbol", "short_volume", "total_volume"])
    df_fresh, df_unresolved = _canonicalise_regsho(context, df_raw, resolver, universe)

    scoped = full and bool(set(load_universe_tickers(context)) - universe)
    if not full or scoped:
        context.store.save(Tables.short_interest, df_fresh)
        logger.info(f"Saved {len(df_fresh)} new short-volume rows to DB table '{Tables.short_interest}'")
        logger.info(f"RegSHO: {len(df_unresolved)} unresolved raw row(s) excluded; {len(failed_days)} day file(s) not served")
        return

    if not successful_days:
        raise RuntimeError("RegSHO full refresh retrieved no source date; preserving the table by aborting")
    earliest = min(successful_days)
    df_legacy = _stored_rows(context, until=earliest - pd.Timedelta(days=1))
    df_corrected_legacy, legacy_stats = _reconcile_legacy(context, df_legacy, resolver, universe)
    df_failed_stored = _stored_rows(context, dates=[day for day in failed_days if day >= earliest])
    df_complete = pd.concat([df_corrected_legacy, df_failed_stored, df_fresh], ignore_index=True)
    if not df_complete.empty:
        df_complete = df_complete.groupby(["ticker", "date"], as_index=False)[["short_volume", "total_volume"]].sum()
    _validate_full_frame(df_complete, universe)
    written = context.store.replace(Tables.short_interest, df_complete)

    logger.info(
        f"RegSHO full: retained={legacy_stats['retained']} "
        f"relabelled={legacy_stats['relabelled']} removed={legacy_stats['removed']} "
        f"legacy_unresolved={legacy_stats['unresolved']} refreshed={len(df_fresh)} "
        f"fresh_unresolved={len(df_unresolved)} preserved_failed_date_rows={len(df_failed_stored)}"
    )
    logger.info(f"RegSHO full: {written} row(s) written to '{Tables.short_interest}'")
