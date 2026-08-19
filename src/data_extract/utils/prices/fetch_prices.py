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
  - EQUITY ONLY. The market/macro series (SPY, ^VIX, oil/gold/energy, FX) used to be
    extra OHLCV rows in `prices`, fetched by this function over `other_tickers`.
    They are now rows in `prices_macro` (`fetch_macro.py`) -- close-only from
    yfinance, or a FRED level for FX -- so `prices` holds the analysed universe and
    nothing else, which is what let the cube drop its `cube_part_market` firewall
    and three `drop(columns=[market])` guards.
"""
import time

import pandas as pd
import yfinance as yf
from tqdm import tqdm
import logging

from src.data_store.schema import Tables
from src.data_extract.utils.common.incremental import resume_since
from src.data_extract.utils.common.run_manifest import record_run
from src.constants.constants import DATE_FORMAT
from src.context import Context

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# PRICE PRE-LISTING TRIM                                                       #
# --------------------------------------------------------------------------- #
# yfinance back-fills a US ticker with its predecessor line (AMCR's ASX quote
# pre-2019, SW's Smurfit Kappa quote pre-2024) or its SPAC trust (VRT before the
# Feb-2020 merger). Those bars are flat and mostly zero-volume, so they inject
# zero realised vol and fake zero returns into beta / correlation / momentum.
# Two independent tells, either of which marks the pre-window as synthetic:
#   * zero-volume share in [first_bar .. last_zero_volume_bar] >= 20%
#     (AMCR 77.1%, SW 62.7%, HWM 94.6% vs PFG/AMD/XEL/IBKR/... all <= 2.7%),
#   * first-year median volume < 1% of the ticker's full-history median volume
#     (VRT 0.17% vs the tightest true listing, NCLH 2.9% / ARES 2.85% / SMCI 3.9%).
# Both thresholds sit an order of magnitude away from the nearest false positive.
PRELISTING_ZERO_VOLUME_SHARE = 0.2
PRELISTING_VOLUME_RATIO = 0.01

# No NO_VOLUME_TICKERS exemption any more. It existed because a quoted index or FX pair carries
# 100% zero volume (no exchange volume exists) and this trim would have erased its whole history
# -- but those series are no longer fetched through here at all (`fetch_macro.py` pulls ^VIX and
# the commodity/energy legs close-only and skips the trim, which is an equity-listing heuristic;
# FX is a FRED level). Everything reaching this function is an equity, where zero volume really
# does mean a synthetic bar.


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

    EQUITIES ONLY -- there is no volume-less exemption list, because the volume-less series
    (FX, ^VIX) do not come through this function any more (see the module docstring).
    Pure; safe to re-apply (a trimmed frame has no qualifying prefix left)."""
    if prices is None or prices.empty or "ticker" not in prices.columns:
        return prices
    drop = pd.Series(False, index=prices.index)
    for ticker, group in prices.groupby("ticker", sort=False):
        cutoff = _prelisting_cutoff(group)
        if cutoff is not None:
            drop.loc[group.index[group["date"] <= cutoff]] = True
    if not drop.any():
        return prices
    return prices.loc[~drop].reset_index(drop=True)

def _is_up_to_date(since: pd.Timestamp, today: pd.Timestamp) -> bool:
    return since >= today - pd.tseries.offsets.BDay(1)

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
    actions: False
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
                actions=actions,          # also return Dividends / Stock Splits
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

    actions= False
    if 'dividend' in desc.lower():
        actions = True

    frames: list[pd.DataFrame] = []
    for i in tqdm(range(0, len(tickers), chunk_size), desc=desc):
        chunk = tickers[i:i + chunk_size]
        
        frames.extend(_download_price_chunk(chunk, since, until, pause, actions))
        time.sleep(pause)

    if not frames:
        return pd.DataFrame()

    return _normalize_prices(pd.concat(frames, ignore_index=True))


def fetch_price_history(
    context: Context,
    tickers: list[str],
    years_history: int,
    chunk_size: int = 50,
    pause: float = 1,
) -> None:
    """Download daily OHLCV for `tickers`, incrementally upserting into the `prices`
    DB table, and return only the freshly-downloaded rows."""

    today = pd.Timestamp.today().normalize()
    since = resume_since(context, Tables.prices, tickers, years_history)

    if _is_up_to_date(since, today):
        logger.info(f"Price history already up to date — DB table '{Tables.prices}'")
        record_run(context, Tables.prices, len(tickers), 0)
        return pd.DataFrame()

    logger.info(f"Downloading prices for {len(tickers)} tickers since {since.date()}")
    df_prices = download_ohlcv(tickers, since, today, chunk_size, pause)

    # Drop the synthetic pre-listing prefix BEFORE the upsert, so a full-history pull
    # never writes another ticker's predecessor line into `prices`. Applied to the
    # freshly-downloaded frame only: an incremental tail carries no prefix.
    df_prices = trim_prelisting_bars(df_prices)

    # upsert the freshly-downloaded delta; the DB merges on (ticker, date)
    context.store.save(Tables.prices, df_prices)
    logger.info(f"Saved {len(df_prices)} new price rows to DB table '{Tables.prices}'")
    record_run(context, Tables.prices, len(tickers), len(df_prices))
