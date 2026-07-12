"""
Fetch the current S&P 500 constituent list and daily OHLCV price history
for each ticker, via yfinance (free, no API key).

Run:
    python -m data.fetch_prices

Notes:
  - The S&P 500 list is scraped from Wikipedia (the standard, widely-used
    source for this). This reflects *today's* constituents — it does not
    reconstruct historical index membership, so there's slight survivorship
    bias in any backtest using it. Fine for a starting point.
  - yfinance can rate-limit / hiccup on big batch downloads. We download in
    chunks with retries and cache to parquet so re-runs are cheap.
  - Re-runs are incremental: only missing date ranges are downloaded and
    appended to the existing parquet file.
"""
import io
import time

import pandas as pd
import requests
import yfinance as yf
from tqdm import tqdm

from src.context import Context

_WIKI_HEADERS = {
    "User-Agent": "stock_pick_strat/1.0 (https://github.com; research@example.com)",
}


def get_sp500_tickers(context: Context) -> list[str]:
    """Scrape current S&P 500 tickers + sector info from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url, headers=_WIKI_HEADERS, timeout=30)
    response.raise_for_status()
    tables = pd.read_html(io.StringIO(response.text))
    df = tables[0]
    df = df.rename(columns={
        "Symbol": "ticker",
        "Security": "name",
        "GICS Sector": "sector",
        "GICS Sub-Industry": "sub_industry",
    })
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)  # yfinance format, e.g. BRK.B -> BRK-B
    tickers_path = context.paths["TICKERS_PATH"]
    df[["ticker", "name", "sector", "sub_industry"]].to_csv(tickers_path, index=False)
    print(f"Saved {len(df)} tickers to {tickers_path}")
    return df["ticker"].tolist()


def _normalize_prices(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]
    out = out.rename(columns={"index": "date"})
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    return out


def _load_existing_prices(path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return _normalize_prices(df)


def _history_start(years_history: int) -> pd.Timestamp:
    return pd.Timestamp.today().normalize() - pd.DateOffset(years=years_history)


def _is_up_to_date(max_date: pd.Timestamp, today: pd.Timestamp) -> bool:
    return max_date >= today - pd.Timedelta(days=1)


def _tickers_needing_download(
    existing: pd.DataFrame | None,
    tickers: list[str],
    years_history: int,
) -> dict[str, tuple[pd.Timestamp, pd.Timestamp] | None]:
    """Return per-ticker (start, end) download windows; None means up to date."""
    today = pd.Timestamp.today().normalize()
    required_start = _history_start(years_history)
    plans: dict[str, tuple[pd.Timestamp, pd.Timestamp] | None] = {}

    for tkr in tickers:
        if existing is None or tkr not in existing["ticker"].values:
            plans[tkr] = (required_start, today)
            continue

        ticker_data = existing.loc[existing["ticker"] == tkr, "date"]
        max_date = ticker_data.max()
        min_date = ticker_data.min()

        if _is_up_to_date(max_date, today) and min_date <= required_start:
            plans[tkr] = None
            continue

        start = required_start if min_date > required_start else max_date + pd.Timedelta(days=1)
        if start > today:
            plans[tkr] = None
        else:
            plans[tkr] = (start, today)

    return plans


def _chunk_response_to_frames(data: pd.DataFrame, chunk: list[str]) -> list[pd.DataFrame]:
    frames = []
    if isinstance(data.columns, pd.MultiIndex):
        for tkr in chunk:
            if tkr not in data.columns.get_level_values(0):
                continue
            sub = data[tkr].dropna(how="all").reset_index()
            sub["ticker"] = tkr
            frames.append(sub)
    elif len(chunk) == 1:
        sub = data.dropna(how="all").reset_index()
        sub["ticker"] = chunk[0]
        frames.append(sub)
    return frames


def _download_price_chunk(
    chunk: list[str],
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
    period: str | None,
    pause: float,
) -> list[pd.DataFrame]:
    for attempt in range(3):
        try:
            kwargs = {
                "interval": "1d",
                "group_by": "ticker",
                "auto_adjust": True,
                "threads": True,
                "progress": False,
            }
            if period is not None:
                kwargs["period"] = period
            else:
                kwargs["start"] = start.strftime("%Y-%m-%d")
                kwargs["end"] = (end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

            data = yf.download(chunk, **kwargs)
            return _chunk_response_to_frames(data, chunk)
        except Exception as e:
            print(f"Chunk {chunk[0]}..{chunk[-1]} attempt {attempt + 1} failed: {e}")
            time.sleep(pause * (attempt + 1))
    print(f"Skipping chunk {chunk[0]}..{chunk[-1]} after 3 failed attempts")
    return []


def _download_prices(
    plans: dict[str, tuple[pd.Timestamp, pd.Timestamp] | None],
    years_history: int,
    chunk_size: int,
    pause: float,
) -> pd.DataFrame:
    tickers_to_fetch = [t for t, window in plans.items() if window is not None]
    if not tickers_to_fetch:
        return pd.DataFrame()

    required_start = _history_start(years_history)
    full_tickers = [t for t in tickers_to_fetch if plans[t][0] == required_start]
    incremental_tickers = [t for t in tickers_to_fetch if plans[t][0] != required_start]

    full_period = f"{years_history}y"
    frames: list[pd.DataFrame] = []

    for i in tqdm(range(0, len(full_tickers), chunk_size), desc="Downloading full price history"):
        chunk = full_tickers[i:i + chunk_size]
        frames.extend(_download_price_chunk(chunk, None, None, full_period, pause))
        time.sleep(pause)

    for i in tqdm(range(0, len(incremental_tickers), chunk_size), desc="Downloading incremental prices"):
        chunk = incremental_tickers[i:i + chunk_size]
        chunk_start = min(plans[t][0] for t in chunk)
        chunk_end = max(plans[t][1] for t in chunk)
        frames.extend(_download_price_chunk(chunk, chunk_start, chunk_end, None, pause))
        time.sleep(pause)

    if not frames:
        return pd.DataFrame()

    return _normalize_prices(pd.concat(frames, ignore_index=True))


def _merge_prices(
    existing: pd.DataFrame | None,
    new: pd.DataFrame,
    years_history: int,
) -> pd.DataFrame:
    parts = [df for df in (existing, new) if df is not None and not df.empty]
    if not parts:
        raise RuntimeError("No price data available — check network / yfinance status.")

    out = pd.concat(parts, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out = out.drop_duplicates(subset=["ticker", "date"], keep="last")
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)

    cutoff = _history_start(years_history)
    out = out[out["date"] >= cutoff].reset_index(drop=True)
    return out


def fetch_price_history(
    context: Context,
    tickers: list[str],
    chunk_size: int = 50,
    pause: float = 2.0,
) -> pd.DataFrame:
    """Download daily OHLCV, incrementally appending to the cached parquet file."""
    path = context.paths["PRICES_PATH"]
    years_history = context.config.data_extract.years_history

    existing = _load_existing_prices(path)
    plans = _tickers_needing_download(existing, tickers, years_history)
    tickers_to_fetch = [t for t, window in plans.items() if window is not None]

    if not tickers_to_fetch:
        print(f"Price history already up to date ({len(existing)} rows) — {path}")
        return existing

    print(
        f"Downloading prices for {len(tickers_to_fetch)}/{len(tickers)} tickers "
        f"({len(tickers) - len(tickers_to_fetch)} already up to date)"
    )
    new = _download_prices(plans, years_history, chunk_size, pause)
    out = _merge_prices(existing, new, years_history)
    out.to_parquet(path, index=False)
    print(f"Saved {len(out)} price rows to {path}")
    return out
