"""
Historical earnings surprises = the market's FORWARD EPS expectation vs what the
company actually delivered, per quarter, going back years.

This is the free, genuinely-historical answer to "what did the market think about
future earnings, in the past?". `yfinance.get_earnings_dates()` returns, for each
past earnings date, the consensus **EPS Estimate**, the **Reported EPS**, and the
**Surprise(%)** -- often close to the full `years_history` window for large,
long-listed S&P 500 names. It also returns the NEXT (not-yet-reported) date with
its estimate and a NaN actual: that row is the live forward EPS.

    earnings_date | eps_estimate | eps_actual | surprise_pct
    2026-08-04    | 1.61         | NaN        | NaN            <- forward (upcoming)
    2026-05-05    | 1.29         | 1.37       | 5.82           <- reported (beat)
    ...

INCREMENTAL: the history parquet is keyed on (ticker, earnings_date). On each run
we only fetch tickers that are (a) missing entirely, or (b) stale -- their most
recent known earnings date is older than `refetch_window_days` (a new quarter has
likely been reported since). Up-to-date tickers are skipped, and a stale ticker
is re-pulled with a small limit just to append the newest quarters and fill in
the actual for a row that was previously a forward estimate.

Run:
    python -m src.data_extract.fetch_earnings_surprises
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

from src.context import Context
from src.data_extract.utils.common.rate_limit import call_with_retries

_RENAME = {
    "EPS Estimate": "eps_estimate",
    "Reported EPS": "eps_actual",
    "Surprise(%)": "surprise_pct",
}
_COLUMNS = ["ticker", "earnings_date", "eps_estimate", "eps_actual", "surprise_pct"]

# a stale ticker (no new row within this many days) is re-pulled with this limit
_RECENT_LIMIT = 8


def _download_one(ticker: str, limit: int) -> pd.DataFrame | None:
    """One ticker's earnings-date table normalized to `_COLUMNS`; None if empty.

    yfinance's earnings-dates endpoint is aggressively rate-limited (429), which
    used to make a ~fixed subset of tickers fail on every run and be silently
    skipped. We now wait + retry on 429 (exponential backoff) so throttled tickers
    recover; a genuine empty (Yahoo has no calendar for the name) still returns
    None after the retries.
    """
    raw = call_with_retries(
        lambda: yf.Ticker(ticker).get_earnings_dates(limit=limit),
        retries=3, base_wait=10.0, label=f"earnings {ticker}")
    if raw is None or raw.empty:
        return None

    df = raw.reset_index()
    date_col = "Earnings Date" if "Earnings Date" in df.columns else df.columns[0]
    df = df.rename(columns={date_col: "earnings_date", **_RENAME})
    for c in ("eps_estimate", "eps_actual", "surprise_pct"):
        if c not in df.columns:
            df[c] = np.nan
    df["ticker"] = ticker
    df["earnings_date"] = (
        pd.to_datetime(df["earnings_date"], utc=True).dt.tz_localize(None).dt.normalize()
    )
    return df[_COLUMNS]


def _plan_fetch(tickers: list[str], existing: pd.DataFrame | None,
                full_limit: int, refetch_window_days: int) -> list[tuple[str, int]]:
    """(ticker, limit) list: full pull for unseen tickers, small pull for stale
    ones, nothing for up-to-date tickers."""
    last_seen: dict[str, pd.Timestamp] = {}
    if existing is not None and not existing.empty:
        reported = existing.dropna(subset=["eps_actual"])
        if not reported.empty:
            last_seen = reported.groupby("ticker")["earnings_date"].max().to_dict()

    today = pd.Timestamp.today().normalize()
    plan = []
    for t in tickers:
        last = last_seen.get(t)
        if last is None:
            plan.append((t, full_limit))
        elif (today - last).days > refetch_window_days:
            plan.append((t, _RECENT_LIMIT))
        # else: already current -> skip
    return plan


def fetch_earnings_surprises(
    context: Context,
    tickers: list[str],
    pause: float = 0.3,
    refetch_window_days: int = 80,
) -> pd.DataFrame:
    """Build/refresh the incremental earnings-surprise history and save it to
    EARNINGS_SURPRISES_PATH. Returns the full merged history."""
    log = context.log
    path = context.paths["EARNINGS_SURPRISES_PATH"]
    existing = pd.read_parquet(path) if path.exists() else None
    if existing is not None and not existing.empty:
        existing["earnings_date"] = pd.to_datetime(existing["earnings_date"]).dt.normalize()

    full_limit = int(context.config.data_extract.years_history) * 4 + 4
    plan = _plan_fetch(tickers, existing, full_limit, refetch_window_days)
    log.info("Earnings surprises: %d/%d tickers to fetch (%d already current)",
             len(plan), len(tickers), len(tickers) - len(plan))

    new_frames = []
    empty, failed = [], []
    for tkr, limit in tqdm(plan, desc="Fetching earnings-surprise history"):
        try:
            df = _download_one(tkr, limit)
        except Exception as e:  # noqa: BLE001 - network/parse issues are per-ticker
            log.warning("%s: earnings history failed (%s)", tkr, e)
            failed.append(tkr)
            continue
        if df is not None:
            new_frames.append(df)
        else:
            empty.append(tkr)          # Yahoo returned no calendar (genuine gap)
        time.sleep(pause)
    if empty or failed:
        log.warning("Earnings: %d empty (no Yahoo calendar) + %d failed after retries "
                    "out of %d fetched. Empty e.g.: %s", len(empty), len(failed),
                    len(plan), empty[:15])

    parts = [df for df in (existing, *new_frames) if df is not None and not df.empty]
    if not parts:
        log.warning("No earnings-surprise data available (nothing fetched, no cache).")
        return existing if existing is not None else pd.DataFrame(columns=_COLUMNS)

    out = pd.concat(parts, ignore_index=True)[_COLUMNS]
    # keep="last" so a freshly fetched row (actual now filled) beats the old
    # forward-estimate row for the same (ticker, earnings_date).
    out = (out.sort_values(["ticker", "earnings_date"])
              .drop_duplicates(subset=["ticker", "earnings_date"], keep="last")
              .reset_index(drop=True))

    out.to_parquet(path, index=False)
    reported = int(out["eps_actual"].notna().sum())
    log.info("Saved %d earnings rows (%d reported, %d forward) for %d tickers to %s",
             len(out), reported, len(out) - reported, out["ticker"].nunique(), path)
    return out
