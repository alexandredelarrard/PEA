"""Historical earnings surprises (consensus EPS estimate vs reported EPS) per quarter -> `earnings_surprises`.

Source is `yfinance.get_earnings_dates()`. The next, not-yet-reported date arrives as a forward row with a NaN
actual; the upsert on `(ticker, earnings_date)` overwrites it once the actual is reported.
"""

from __future__ import annotations

import time
from typing import cast

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

from src.context import Context
from src.data_extract.utils.common.rate_limit import call_with_retries
from src.data_extract.utils.common.run_manifest import record_run
from src.data_store.schema import Tables

_RENAME = {
    "EPS Estimate": "eps_estimate",
    "Reported EPS": "eps_actual",
    "Surprise(%)": "surprise_pct",
}
_COLUMNS = ["ticker", "earnings_date", "eps_estimate", "eps_actual", "surprise_pct"]

# a stale ticker (no new row within this many days) is re-pulled with this limit
_RECENT_LIMIT = 8

# rows before this date are sporadic and almost always missing, so they are dropped
MIGRATION_DATE = "2002-10-01"


def _download_one(ticker: str, limit: int) -> pd.DataFrame | None:
    """One ticker's earnings-date table normalized to `_COLUMNS`, retried with backoff on 429; None if empty."""
    raw = call_with_retries(lambda: yf.Ticker(ticker).get_earnings_dates(limit=limit), retries=3, base_wait=10.0, label=f"earnings {ticker}")
    if raw is None or raw.empty:
        return None

    df = raw.reset_index()
    date_col = "Earnings Date" if "Earnings Date" in df.columns else df.columns[0]
    df = df.rename(columns={date_col: "earnings_date", **_RENAME})
    for c in ("eps_estimate", "eps_actual", "surprise_pct"):
        if c not in df.columns:
            df[c] = np.nan
    df["ticker"] = ticker
    df["earnings_date"] = pd.to_datetime(df["earnings_date"], utc=True).dt.tz_localize(None).dt.normalize()
    return df[_COLUMNS]


def _plan_fetch(tickers: list[str], existing: pd.DataFrame | None, full_limit: int, refetch_window_days: int) -> list[tuple[str, int]]:
    """`(ticker, limit)` fetch plan: full pull for unseen tickers, `_RECENT_LIMIT` once the stored forward date has passed.

    Tickers with no stored forward row fall back to `refetch_window_days` since the last reported date; the rest
    are skipped.
    """
    last_reported: dict[str, pd.Timestamp] = {}
    next_expected: dict[str, pd.Timestamp] = {}
    if existing is not None and not existing.empty:
        reported = existing.dropna(subset=["eps_actual"])
        if not reported.empty:
            last_reported = cast(dict[str, pd.Timestamp], reported.groupby("ticker")["earnings_date"].max().to_dict())
        next_expected = cast(dict[str, pd.Timestamp], existing.groupby("ticker")["earnings_date"].max().to_dict())

    today = pd.Timestamp.today().normalize()
    plan = []
    for t in tickers:
        last = last_reported.get(t)
        if last is None:
            plan.append((t, full_limit))  # never seen -> full pull
            continue
        nxt = next_expected.get(t, last)
        if nxt > last:  # a forward earnings date is known
            if nxt <= today:  # ... and it has passed -> new quarter due
                plan.append((t, _RECENT_LIMIT))
        elif (today - last).days > refetch_window_days:  # no forward date -> staleness window
            plan.append((t, _RECENT_LIMIT))
        # else: next earnings still in the future / already current -> skip
    return plan


def fetch_earnings_surprises(
    context: Context,
    tickers: list[str],
    years_history: int = 15,
    pause: float = 0.3,
    refetch_window_days: int = 95,  # > one quarter; fallback only when no forward date is known
) -> None:
    """Incrementally refresh `earnings_surprises` for `tickers`; per-ticker failures are logged and skipped."""

    existing = context.store.load(Tables.earnings_surprises, columns=["ticker", "earnings_date", "eps_actual"], optional=True)
    if existing is not None:
        existing["earnings_date"] = pd.to_datetime(existing["earnings_date"], format="%Y-%m-%d")

    full_limit = (years_history + 1) * 4
    plan = _plan_fetch(tickers, existing, full_limit, refetch_window_days)
    context.log.info("Earnings surprises: %d/%d tickers to fetch (%d already current)", len(plan), len(tickers), len(tickers) - len(plan))

    new_frames = []
    empty, failed = [], []
    for tkr, limit in tqdm(plan, desc="Fetching earnings-surprise history"):
        try:
            df = _download_one(tkr, limit)
        except Exception as e:  # noqa: BLE001 - network/parse issues are per-ticker
            context.log.warning("%s: earnings history failed (%s)", tkr, e)
            failed.append(tkr)
            continue
        if df is not None:
            new_frames.append(df)
        else:
            empty.append(tkr)  # Yahoo returned no calendar (genuine gap)
        time.sleep(pause)

    if empty or failed:
        context.log.warning(
            "Earnings: %d empty (no Yahoo calendar) + %d failed after retries out of %d fetched. Empty e.g.: %s",
            len(empty),
            len(failed),
            len(plan),
            empty[:15],
        )

    parts = [df for df in (existing, *new_frames) if df is not None and not df.empty]
    if not parts:
        context.log.warning("No earnings-surprise data available (nothing fetched, no cache).")
        record_run(context, Tables.earnings_surprises, len(tickers), 0)

    # upsert on (ticker, earnings_date): a now-filled actual overwrites its forward-estimate row
    if new_frames:
        new = pd.concat(new_frames, ignore_index=True)[_COLUMNS]
        new = new.loc[new["earnings_date"] >= MIGRATION_DATE].reset_index(drop=True)
        context.store.save(Tables.earnings_surprises, new)
        context.log.info(f"Saved {len(new)} new earnings rows for {new['ticker'].nunique()} tickers to DB")
        record_run(context, Tables.earnings_surprises, len(tickers), len(new))
