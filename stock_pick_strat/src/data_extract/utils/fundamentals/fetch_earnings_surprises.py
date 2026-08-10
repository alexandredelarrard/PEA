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

INCREMENTAL (DB table `earnings_surprises`, keyed on (ticker, earnings_date)): each run
only fetches tickers that are (a) missing entirely, or (b) due -- their next earnings
date (the forward row yfinance returns with eps_actual = NaN) has already passed, so
the actual should now be available. Tickers whose next earnings is still in the future
are skipped (nothing new to fetch yet); a due ticker is re-pulled with a small limit to
append the new quarter and fill the actual for the prior forward-estimate row.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

from src.context import Context
from src.data_extract.utils.common.rate_limit import call_with_retries
from src.data_extract.utils.common.run_manifest import record_run

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
    """(ticker, limit) list: full pull for unseen tickers; a small pull only for
    tickers whose NEXT earnings date has already passed (a new quarter is due);
    nothing for tickers whose next earnings is still in the future.

    yfinance returns the upcoming (not-yet-reported) date as a forward row
    (eps_actual = NaN), so its max earnings_date is the next-expected date. Gating on
    that — instead of a fixed staleness window shorter than the ~91-day quarterly
    cycle — stops ~30% of names being re-pulled needlessly in the ~10-day gap before
    they report (they already have full history; there is simply nothing new yet).
    Tickers with no known forward date fall back to the staleness window."""
    last_reported: dict[str, pd.Timestamp] = {}
    next_expected: dict[str, pd.Timestamp] = {}
    if existing is not None and not existing.empty:
        reported = existing.dropna(subset=["eps_actual"])
        if not reported.empty:
            last_reported = reported.groupby("ticker")["earnings_date"].max().to_dict()
        next_expected = existing.groupby("ticker")["earnings_date"].max().to_dict()

    today = pd.Timestamp.today().normalize()
    plan = []
    for t in tickers:
        last = last_reported.get(t)
        if last is None:
            plan.append((t, full_limit))            # never seen -> full pull
            continue
        nxt = next_expected.get(t, last)
        if nxt > last:                              # a forward earnings date is known
            if nxt <= today:                        # ... and it has passed -> new quarter due
                plan.append((t, _RECENT_LIMIT))
        elif (today - last).days > refetch_window_days:   # no forward date -> staleness window
            plan.append((t, _RECENT_LIMIT))
        # else: next earnings still in the future / already current -> skip
    return plan


def fetch_earnings_surprises(
    context: Context,
    tickers: list[str],
    pause: float = 0.3,
    refetch_window_days: int = 95,      # > one quarter; fallback only when no forward date is known
) -> pd.DataFrame:
    """Build/refresh the incremental earnings-surprise history and upsert it into the
    `earnings_surprises` DB table. Returns the full merged history."""
    log = context.log
    # earnings are an equity concept — drop non-equity instruments (indices / futures /
    # FX from other_tickers, e.g. ^VIX, CL=F, USDEUR=X) that never return a calendar and
    # would otherwise be re-attempted every run as "missing".
    tickers = [t for t in tickers if not any(c in t for c in ("^", "="))]
    existing = context.store.load("earnings_surprises", optional=True)
    if existing is not None:
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
        record_run(context, "earnings_surprises", len(tickers), 0)
        return existing if existing is not None else pd.DataFrame(columns=_COLUMNS)

    out = pd.concat(parts, ignore_index=True)[_COLUMNS]
    # keep="last" so a freshly fetched row (actual now filled) beats the old
    # forward-estimate row for the same (ticker, earnings_date).
    out = (out.sort_values(["ticker", "earnings_date"])
              .drop_duplicates(subset=["ticker", "earnings_date"], keep="last")
              .reset_index(drop=True))

    # upsert the freshly-fetched rows; the DB merges on (ticker, earnings_date),
    # so a now-filled actual overwrites the old forward-estimate row.
    new = pd.concat(new_frames, ignore_index=True)[_COLUMNS] if new_frames else pd.DataFrame()
    if not new.empty:
        context.store.save("earnings_surprises", new)
    reported = int(out["eps_actual"].notna().sum())
    log.info("Saved %d new earnings rows (history %d rows, %d reported) for %d tickers to DB",
             len(new), len(out), reported, out["ticker"].nunique())
    record_run(context, "earnings_surprises", len(tickers), len(new))
    return out
