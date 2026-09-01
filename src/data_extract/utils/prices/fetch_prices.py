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


def _normalize_prices(df: pd.DataFrame, auto_adjust: bool) -> pd.DataFrame:
    """Lowercase the yfinance columns and name the two price bases explicitly.

    ⚠ The bare name `close` is NEVER emitted, on either path. `close` silently changing
    meaning is exactly the bug class this module exists to remove: a reader that wants the
    total-return series and gets the split-adjusted one computes PRICE returns and is wrong
    by every dividend ever paid (MO reads 1.24x over the sample where the truth is 20.2x).
    A missed reader must raise `KeyError`, not quietly return the wrong number.

    `auto_adjust=False` (the EQUITY path) returns `Close` -- split-adjusted only, the same
    basis as Sharadar's `price`, agreeing to the cent -- and `Adj Close`, that series further
    reduced for every dividend paid after the date. They map to `close_split` and
    `close_total`. Both are written from ONE response in ONE upsert, so they cannot drift.

    `auto_adjust=True` (the MACRO path) returns a single already-total-return `Close`, which
    becomes `close_total`. `SPY` is stored as `equity_tr` and consumed as a RETURN, so it
    must stay on that basis -- see `fetch_macro._fetch_price_leg`."""
    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]
    out = out.rename(columns={"index": "date"})
    out["date"] = pd.to_datetime(out["date"], format="%Y-%m-%d")

    if auto_adjust:
        return out.rename(columns={"close": "close_total"})

    if "adj close" not in out.columns:
        # Refuse rather than fall back. A single-column write here is precisely the silent
        # failure the two-column design exists to prevent, and it would be indistinguishable
        # from a correct table until a label came out on the wrong basis.
        raise RuntimeError(
            "yfinance returned no 'Adj Close' under auto_adjust=False -- refusing to write a "
            f"single-basis price frame. Columns present: {sorted(out.columns)}")
    return out.rename(columns={"close": "close_split", "adj close": "close_total"})


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
    actions: bool,
    auto_adjust: bool,
) -> list[pd.DataFrame]:
    """One retried yfinance call. `auto_adjust` has NO DEFAULT on purpose: it selects the
    adjustment basis of everything downstream, and a default is how that choice gets made
    silently by whoever adds the next call site."""
    for attempt in range(3):
        try:
            data = yf.download(
                chunk,
                start=start.strftime(DATE_FORMAT),
                end=(end + pd.Timedelta(days=1)).strftime(DATE_FORMAT),
                interval="1d",
                group_by="ticker",
                auto_adjust=auto_adjust,
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
    *,
    auto_adjust: bool,
    actions: bool = True,
) -> pd.DataFrame:
    """Chunked yfinance pull over [since, until] -> one normalized long frame
    [date, ticker, open/high/low, close_split and/or close_total, volume, dividends,
    stock splits]. Empty frame when every chunk failed.

    ⚠ `auto_adjust` is KEYWORD-ONLY and REQUIRED, because this function is SHARED and its
    three callers need different bases:

      * `fetch_price_history` / `fetch_splits` -> `auto_adjust=False`, giving both `Close`
        (split-adjusted, the market-cap basis) and `Adj Close` (total return).
      * `fetch_macro._fetch_price_leg`         -> `auto_adjust=True`. `SPY` is stored as
        `equity_tr` and feeds the L/S benchmark leg, `beta_market` and `fwd_market` inside
        every label; `XLE` pays ~3%. Flipping either to a PRICE return corrupts all of that.

    It used to be a hard-coded `True` here and `actions` was inferred from the `desc` STRING
    ("dividend" in desc), which meant the basis and the action columns were both decided by a
    progress-bar label. Both are now explicit arguments."""
    frames: list[pd.DataFrame] = []
    for i in tqdm(range(0, len(tickers), chunk_size), desc=desc):
        chunk = tickers[i:i + chunk_size]
        frames.extend(_download_price_chunk(chunk, since, until, pause, actions, auto_adjust))
        time.sleep(pause)

    if not frames:
        return pd.DataFrame()

    return _normalize_prices(pd.concat(frames, ignore_index=True), auto_adjust)


def tickers_needing_repull(context: Context, tickers: list[str]) -> list[str]:
    """Tickers whose stored history is on a STALE adjustment basis, because a split has an
    ex-date after their last stored bar.

    Split adjustment is RETROACTIVE: the day a stock splits, every prior bar Yahoo serves is
    restated, but nothing in an incremental upsert ever revisits them. So the table ends up
    interleaving two vintages inside one ticker. This is live today on MNST, which split
    2026-07-20 and whose July/August bars alternate between ~97 and ~47 -- day-to-day returns
    of +-95% that never happened, flowing straight into momentum, vol, betas and the labels.

    Without this trigger, EVERY future splitter re-corrupts the table the same way, and the
    one-off `--full` re-download buys only a clean snapshot. Empty list when `prices_splits`
    has no rows yet (P2 not run), so this degrades to today's behaviour rather than failing."""
    splits = context.store.load(Tables.prices_splits, columns=["ticker", "date"],
                                where={"ticker": tickers}, optional=True)
    if splits is None or splits.empty:
        return []
    last_bar = context.store.max_date_by(Tables.prices, "ticker", "date")
    if not last_bar:
        return []

    splits = splits.copy()
    splits["date"] = pd.to_datetime(splits["date"])
    stale = {
        ticker for ticker, event in zip(splits["ticker"], splits["date"])
        if ticker in last_bar and event > pd.Timestamp(last_bar[ticker])
    }
    return sorted(stale)


def fetch_price_history(
    context: Context,
    tickers: list[str],
    years_history: int,
    chunk_size: int = 50,
    pause: float = 1,
    full: bool = False,
) -> None:
    """Download daily OHLCV for `tickers`, upserting into the `prices` DB table.

    Two price columns are written from ONE response: `close_split` (split-adjusted only --
    the basis market cap, EV, dividend yield and ATR need, because on it the future-split
    factor cancels exactly against Sharadar's `sharesbas`) and `close_total` (further reduced
    for later dividends -- the basis returns, momentum, betas and labels need).

    Windows, widest first:
      * `full=True`          -- the whole `years_history` window for every ticker. Needed
        because a split restates history retroactively and an incremental tail never revisits
        it, so the stored table interleaves adjustment vintages.
      * a pending split      -- the same full window, for those tickers only, whatever `full`
        says (see `tickers_needing_repull`).
      * otherwise            -- `resume_since`, the shared per-batch frontier.

    The two windows are TWO CALLS rather than a per-ticker `since`, which keeps `resume_since`
    and its one-shared-date contract untouched."""
    today = pd.Timestamp.today().normalize()
    window_start = today - pd.DateOffset(years=years_history)

    repull = [] if full else tickers_needing_repull(context, tickers)
    incremental = [t for t in tickers if t not in set(repull)]

    batches: list[tuple[list[str], pd.Timestamp, str]] = []
    if full:
        batches.append((tickers, window_start, "full history"))
    else:
        if repull:
            logger.info("%d ticker(s) split after their last stored bar -- re-pulling their "
                        "full history to clear the stale adjustment basis: %s",
                        len(repull), ", ".join(repull))
            batches.append((repull, window_start, "post-split re-pull"))
        if incremental:
            since = resume_since(context, Tables.prices, incremental, years_history)
            if _is_up_to_date(since, today) and not repull:
                logger.info(f"Price history already up to date — DB table '{Tables.prices}'")
                record_run(context, Tables.prices, len(tickers), 0)
                return
            if not _is_up_to_date(since, today):
                batches.append((incremental, since, "incremental"))

    total = 0
    for batch, since, label in batches:
        logger.info("Downloading prices for %d tickers since %s (%s)",
                    len(batch), since.date(), label)
        # actions=False keeps `prices` clean OHLCV: no `dividends` / `stock splits` column
        # can reach the upsert. Both price bases arrive regardless -- `auto_adjust=False`
        # returns `Close` AND `Adj Close` on its own (verified: AAPL 2020-07-31 -> 106.26 and
        # 102.795). Ex-dates and split events have their own fetchers with their own sparse
        # resume frontiers.
        df_prices = download_ohlcv(batch, since, today, chunk_size, pause,
                                   auto_adjust=False, actions=False)

        # Drop the synthetic pre-listing prefix BEFORE the upsert, so a full-history pull
        # never writes another ticker's predecessor line into `prices`. It matters most on
        # exactly these wide windows: an incremental tail carries no prefix.
        df_prices = trim_prelisting_bars(df_prices)

        # upsert the freshly-downloaded delta; the DB merges on (ticker, date)
        context.store.save(Tables.prices, df_prices)
        total += len(df_prices)
        logger.info("Saved %d price rows to DB table '%s' (%s)",
                    len(df_prices), Tables.prices, label)

    record_run(context, Tables.prices, len(tickers), total)
