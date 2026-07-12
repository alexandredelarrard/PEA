"""
Analyst estimates and estimate REVISIONS per ticker.

READ THIS FIRST — free-data ceiling:
There is no free, ToS-compliant source for a true 10-YEAR ARCHIVE of
analyst estimates and their revisions over time (that's exactly what
paid data like I/B/E/S, Refinitiv, FactSet sell). What yfinance gives you
for free is:
  - Current consensus estimates (EPS/revenue) for the next few quarters/years
  - A short revision TREND: how estimates have moved over the last
    7/30/60/90 days (`eps_trend`, `eps_revisions`) — recent momentum only,
    not history
  - `recommendations`: analyst rating changes, typically covering the last
    couple of years, not 10

So this script does two things:
  1. Pulls everything available TODAY (`fetch_snapshot`) — current
     estimates, short-term revision trend, and available rating history.
  2. Appends today's pull to a running history file
     (`analyst_estimates_history.parquet`), exactly like fetch_fundamentals
     does — so if you run this monthly/quarterly, you build your own real
     multi-year archive going forward. That's the only free way to get
     genuine history here; there's no shortcut for the past.

If you need real historical analyst estimates NOW (not built up over time),
that requires a paid provider — flagging so you don't spend hours trying to
find a free 10-year archive that doesn't exist.

Run:
    python -m data.fetch_analyst_estimates
"""
import time
from datetime import datetime

import pandas as pd
import yfinance as yf
from tqdm import tqdm

from src.context import Context

def _safe_get_df(ticker_obj, attr):
    try:
        val = getattr(ticker_obj, attr)
        return val if isinstance(val, pd.DataFrame) and not val.empty else None
    except Exception:
        return None


def fetch_snapshot(context: Context, tickers: list[str], pause: float = 0.3) -> pd.DataFrame:
    rows = []
    as_of = datetime.utcnow().date().isoformat()

    for tkr in tqdm(tickers, desc="Fetching analyst estimates"):
        t = yf.Ticker(tkr)
        row = {"ticker": tkr, "as_of": as_of}

        # Current consensus estimates (next quarter / next year)
        est = _safe_get_df(t, "earnings_estimate")
        if est is not None:
            for period in est.index:
                if "avg" in est.columns:
                    row[f"eps_est_{period}"] = est.loc[period, "avg"]
                if "numberOfAnalysts" in est.columns:
                    row[f"eps_est_{period}_n_analysts"] = est.loc[period, "numberOfAnalysts"]

        rev_est = _safe_get_df(t, "revenue_estimate")
        if rev_est is not None and "avg" in rev_est.columns:
            for period in rev_est.index:
                row[f"rev_est_{period}"] = rev_est.loc[period, "avg"]

        # Revision trend: how estimates moved over last 7/30/60/90 days
        trend = _safe_get_df(t, "eps_trend")
        if trend is not None:
            for period in trend.index:
                for col in ["current", "7daysAgo", "30daysAgo", "60daysAgo", "90daysAgo"]:
                    if col in trend.columns:
                        row[f"eps_trend_{period}_{col}"] = trend.loc[period, col]

        revisions = _safe_get_df(t, "eps_revisions")
        if revisions is not None:
            for period in revisions.index:
                for col in ["upLast7days", "upLast30days", "downLast7days", "downLast30days"]:
                    if col in revisions.columns:
                        row[f"eps_revisions_{period}_{col}"] = revisions.loc[period, col]

        # Recommendation summary (current snapshot: strongBuy/buy/hold/sell/strongSell counts)
        rec = _safe_get_df(t, "recommendations")
        if rec is not None and len(rec) > 0:
            latest = rec.iloc[0]
            for col in ["strongBuy", "buy", "hold", "sell", "strongSell"]:
                if col in latest.index:
                    row[f"rec_{col}"] = latest[col]

        # Price targets (current snapshot only)
        try:
            pt = t.analyst_price_targets
            if pt:
                row["price_target_mean"] = pt.get("mean")
                row["price_target_high"] = pt.get("high")
                row["price_target_low"] = pt.get("low")
                row["price_target_current"] = pt.get("current")
        except Exception:
            pass

        rows.append(row)
        time.sleep(pause)

    return pd.DataFrame(rows)


def append_to_history(context: Context, snapshot: pd.DataFrame):
    if context.paths["ANALYST_ESTIMATES_HISTORY_PATH"].exists():
        hist = pd.read_parquet(context.paths["ANALYST_ESTIMATES_HISTORY_PATH"])
        hist = pd.concat([hist, snapshot], ignore_index=True)
        hist = hist.drop_duplicates(subset=["ticker", "as_of"], keep="last")
    else:
        hist = snapshot
    hist.to_parquet(context.paths["ANALYST_ESTIMATES_HISTORY_PATH"], index=False)


def fetch_analyst_estimates(context: Context, tickers: list[str], pause: float = 0.3) -> pd.DataFrame:
    snapshot = fetch_snapshot(context, tickers, pause)
    snapshot.to_parquet(context.paths["ANALYST_ESTIMATES_PATH"], index=False)
    append_to_history(context, snapshot)
    return snapshot
