"""
Historical earnings surprises (actual vs. estimated EPS) per ticker.

Unlike analyst estimate HISTORY (see fetch_analyst_estimates.py), this one
is genuinely available for free: yfinance's `get_earnings_dates()` returns
past reported quarters with EPS estimate, actual, and surprise % — often
covering close to the full 10-year window for large, long-listed S&P 500
names (smaller/newer listings will have less). No paid data needed here.

Run:
    python -m data.fetch_earnings_surprises
"""
import time
import pandas as pd
import yfinance as yf
from tqdm import tqdm

from src.context import Context

# yfinance limits results per call; ~4 earnings/year * years + buffer
def fetch_earnings_surprises(context: Context, tickers: list[str], pause: float = 0.3) -> pd.DataFrame:
    frames = []
    for tkr in tqdm(tickers, desc="Fetching earnings surprise history"):
        try:
            df = yf.Ticker(tkr).get_earnings_dates(limit=context.config.data_extract.years_history * 4)
        except Exception as e:
            print(f"{tkr}: failed ({e})")
            continue

        if df is None or df.empty:
            continue

        df = df.reset_index().rename(columns={
            "Earnings Date": "earnings_date",
            "EPS Estimate": "eps_estimate",
            "Reported EPS": "eps_actual",
            "Surprise(%)": "surprise_pct",
        })
        df["ticker"] = tkr
        frames.append(df)
        time.sleep(pause)

    if not frames:
        raise RuntimeError("No earnings surprise data downloaded.")

    out = pd.concat(frames, ignore_index=True)
    out["earnings_date"] = pd.to_datetime(out["earnings_date"], utc=True).dt.tz_localize(None)
    out = out.dropna(subset=["eps_actual"])  # keep only reported (not future/scheduled) quarters
    return out


def fetch_earnings_surprises(context: Context, tickers: list[str], pause: float = 0.3) -> pd.DataFrame:
    frames = []
    for tkr in tqdm(tickers, desc="Fetching earnings surprise history"):
        try:
            df = yf.Ticker(tkr).get_earnings_dates(limit=context.config.data_extract.years_history * 4)
        except Exception as e:
            print(f"{tkr}: failed ({e})")
            continue

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.reset_index().rename(columns={
        "Earnings Date": "earnings_date",
        "EPS Estimate": "eps_estimate",
        "Reported EPS": "eps_actual",
        "Surprise(%)": "surprise_pct",
    })
    df["ticker"] = tkr
    frames.append(df)
    time.sleep(pause)

    if not frames:
        raise RuntimeError("No earnings surprise data downloaded.")

    out = pd.concat(frames, ignore_index=True)
    out["earnings_date"] = pd.to_datetime(out["earnings_date"], utc=True).dt.tz_localize(None)
    out = out.dropna(subset=["eps_actual"])  # keep only reported (not future/scheduled) quarters
    out.to_parquet(context.paths["EARNINGS_SURPRISES_PATH"], index=False)
    print(f"Saved {len(out)} earnings surprise rows to {context.paths["EARNINGS_SURPRISES_PATH"]}")
    return out