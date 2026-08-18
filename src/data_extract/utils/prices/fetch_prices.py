"""
fetch_prices.py  (src/data_extract/utils/prices/fetch_prices.py)
------------------------------------------------------------------
Daily OHLCV price history per ticker via yfinance (free, no API key), upserted
into the `prices` table. The universe itself is scraped elsewhere
(`fetch_tickers.py`); this module only fetches bars.

Notes:
  - yfinance can rate-limit / hiccup on big batch downloads, so we download in
    chunks with retries and upsert, which makes re-runs cheap.
  - Re-runs are incremental: `resume_since` (utils/common/incremental.py) finds
    the oldest per-ticker last-extracted date across the whole batch, and every
    ticker is re-fetched forward from that ONE shared date. Some tickers get
    redundant rows this way, but that is cheap and the upsert no-ops them --
    simpler than planning a bespoke window per ticker.
  - OHLCV only. Dividends are a SEPARATE fetcher (`fetch_dividends.py`, called
    alongside this one by `StepExtractPrices`) with its own resume window, since
    ex-dates are quarterly and sparse where bars are daily. `download_ohlcv` is
    the shared entry point -- one yfinance response serves both.
  - The market/macro series (`other_tickers`: SPY, ^VIX, oil/gold, FX) are just
    this function over that list: ordinary OHLCV rows in `prices`, never part of
    the equity universe and never passed to `fetch_dividends`.
"""
import time

import pandas as pd
import yfinance as yf
from tqdm import tqdm
import logging

from src.data_store.schema import Tables
from src.data_extract.utils.common.incremental import resume_since
from src.data_extract.utils.common.run_manifest import record_run
from src.constants.constants import (DATE_FORMAT, NO_VOLUME_TICKERS, PRELISTING_VOLUME_RATIO, PRELISTING_ZERO_VOLUME_SHARE)
from src.context import Context

logger = logging.getLogger(__name__)


def _normalize_prices(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]
    out = out.rename(columns={"index": "date"})
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    return out


def _prelisting_cutoff(frame: pd.DataFrame) -> pd.Timestamp | None:
    """Last date of a ticker's SYNTHETIC pre-listing block, or None when it has none.

    yfinance back-fills a US symbol with whatever line preceded it: AMCR carries Amcor's
    ASX quote before the June-2019 NYSE listing, SW carries Smurfit Kappa's before
    July 2024, VRT carries the GS Acquisition SPAC trust (~$9.9 flat) before the
    Feb-2020 merger. Those bars are flat and mostly zero-volume, so they inject zero
    realised volatility and fake zero returns into vol / beta / correlation / momentum.
    Measured on the live table: AMCR 1,371 of 3,569 rows zero-volume (all <= 2019-06-10),
    SW 2,041 of 3,771 (all <= 2024-07-05, i.e. 86% of its stored history).

    Two independent tells, either sufficient, both an order of magnitude clear of the
    nearest false positive:
      * the zero-volume SHARE of [first bar .. last zero-volume bar] is high -- AMCR
        77.1%, SW 62.7%, HWM 94.6%, versus PFG / AMD / XEL / IBKR / DXCM / HUBB / SBAC /
        WTW / DOC / CNC / GEN / CHD / ERIE / VST at <= 2.7% (isolated vendor glitches,
        which must NOT be trimmed),
      * the first-year median VOLUME is a negligible fraction of the ticker's own
        full-history median -- VRT 0.17%, whose zero-volume share is only 3.6%, versus
        the tightest genuine listings NCLH 2.9% / ARES 2.85% / SMCI 3.9% / CRH 4.2%.

    Only a contiguous PREFIX is ever returned, so trimming can never punch an interior
    hole in the middle of a ticker's otherwise-contiguous history.
    """
    if frame.empty or "volume" not in frame.columns:
        return None
    f = frame.sort_values("date")
    volume = pd.to_numeric(f["volume"], errors="coerce")
    zero = volume.fillna(0) <= 0
    if not zero.any():
        return None

    last_zero = f.loc[zero, "date"].max()
    window = f["date"] <= last_zero
    if window.sum() and zero[window].mean() >= PRELISTING_ZERO_VOLUME_SHARE:
        return last_zero

    # Flat SPAC-trust / stub regime that still records token volume: compare the first
    # year against the ticker's own long-run level, so the test is scale-free.
    first_year = f["date"] <= f["date"].min() + pd.DateOffset(years=1)
    early, overall = volume[first_year].median(), volume.median()
    if overall and overall > 0 and early / overall < PRELISTING_VOLUME_RATIO:
        return last_zero
    return None


def trim_prelisting_bars(prices: pd.DataFrame) -> pd.DataFrame:
    """Drop each equity ticker's synthetic pre-listing prefix (see `_prelisting_cutoff`).

    Tickers whose volume is legitimately always zero are exempt: FX has no exchange
    volume, so `USDEUR=X` is 100% zero-volume and would otherwise be erased entirely.
    Pure; safe to re-apply (a trimmed frame has no qualifying prefix left)."""
    if prices is None or prices.empty or "ticker" not in prices.columns:
        return prices
    drop = pd.Series(False, index=prices.index)
    for ticker, group in prices.groupby("ticker", sort=False):
        if str(ticker).upper() in NO_VOLUME_TICKERS:
            continue
        cutoff = _prelisting_cutoff(group)
        if cutoff is not None:
            drop.loc[group.index[group["date"] <= cutoff]] = True
    if not drop.any():
        return prices
    return prices.loc[~drop].reset_index(drop=True)


def _is_up_to_date(since: pd.Timestamp, today: pd.Timestamp) -> bool:
    # The freshest bar we could reasonably have is the previous *business* day
    # (today's close may not be published yet by yfinance). Using a business-day
    # offset stops weekends/Mondays from re-downloading Friday-anchored data:
    #   Sun/Sat/Mon -> Fri, Wed -> Tue, etc.
    last_expected = today - pd.tseries.offsets.BDay(1)
    return since >= last_expected


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
    start: pd.Timestamp,
    end: pd.Timestamp,
    pause: float,
) -> list[pd.DataFrame]:
    for attempt in range(3):
        try:
            data = yf.download(
                chunk,
                start=start.strftime(DATE_FORMAT),
                end=(end + pd.Timedelta(days=1)).strftime(DATE_FORMAT),
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                actions=True,          # also return Dividends / Stock Splits
                threads=True,
                progress=False,
            )
            return _chunk_response_to_frames(data, chunk)
        except Exception as e:
            logger.error(f"Chunk {chunk[0]}..{chunk[-1]} attempt {attempt + 1} failed: {e}")
            time.sleep(pause * (attempt + 1))
    logger.info(f"Skipping chunk {chunk[0]}..{chunk[-1]} after 3 failed attempts")
    return []


def download_ohlcv(
    tickers: list[str],
    since: pd.Timestamp,
    until: pd.Timestamp,
    chunk_size: int = 50,
    pause: float = 2.0,
    desc: str = "Downloading prices",
) -> pd.DataFrame:
    """Chunked yfinance pull over [since, until] -> one normalized long frame
    [date, ticker, OHLCV, dividends, stock splits, ...]. Shared with
    `fetch_dividends`, which needs the same `actions=True` response for its
    ex-dates; empty frame when every chunk failed."""
    frames: list[pd.DataFrame] = []
    for i in tqdm(range(0, len(tickers), chunk_size), desc=desc):
        chunk = tickers[i:i + chunk_size]
        frames.extend(_download_price_chunk(chunk, since, until, pause))
        time.sleep(pause)

    if not frames:
        return pd.DataFrame()

    return _normalize_prices(pd.concat(frames, ignore_index=True))


# yfinance actions=True columns to DROP from the OHLCV `prices` table: dividends are
# saved separately (see fetch_dividends.py); "capital gains" is a mutual-fund/ETF
# distribution field (~99% empty for equities, unused) — keep prices clean OHLCV.
_ACTION_COLS = ["dividends", "stock splits", "capital gains"]


def fetch_price_history(
    context: Context,
    tickers: list[str],
    years_history: int,
    chunk_size: int = 50,
    pause: float = 2.0,
) -> pd.DataFrame:
    """Download daily OHLCV for `tickers`, incrementally upserting into the `prices`
    DB table, and return only the freshly-downloaded rows.

    Resumes from `resume_since` -- the oldest per-ticker last-extracted date across the
    whole batch -- and re-fetches every ticker forward from that ONE shared date; a
    ticker already current just gets redundant rows, no-opped by the upsert, while a
    ticker with no rows yet gets `years_history` of backfill. Dividends are NOT written
    here: `fetch_dividends` is its own fetcher, called alongside this one."""

    today = pd.Timestamp.today().normalize()
    since = resume_since(context, Tables.prices, tickers, years_history)

    if not tickers or _is_up_to_date(since, today):
        logger.info(f"Price history already up to date — DB table '{Tables.prices}'")
        record_run(context, Tables.prices, len(tickers), 0)
        return pd.DataFrame()

    logger.info(f"Downloading prices for {len(tickers)} tickers since {since.date()}")
    df_prices = download_ohlcv(tickers, since, today, chunk_size, pause)

    # keep the prices table a clean OHLCV frame (drop the action columns)
    df_prices = df_prices.drop(columns=_ACTION_COLS, errors="ignore")
    # Drop the synthetic pre-listing prefix BEFORE the upsert, so a full-history pull
    # never writes another ticker's predecessor line into `prices`. Applied to the
    # freshly-downloaded frame only: an incremental tail carries no prefix.
    df_prices = trim_prelisting_bars(df_prices)

    # upsert the freshly-downloaded delta; the DB merges on (ticker, date)
    context.store.save(Tables.prices, df_prices)
    logger.info(f"Saved {len(df_prices)} new price rows to DB table '{Tables.prices}'")
    record_run(context, Tables.prices, len(tickers), len(df_prices))

    return df_prices
