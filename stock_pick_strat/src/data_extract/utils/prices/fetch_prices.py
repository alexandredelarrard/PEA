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

from src.data_extract.utils.common.gics import industry_group
from src.data_extract.utils.common.run_manifest import record_run
from src.constants.constants import (
    DATE_FORMAT, NO_VOLUME_TICKERS, PRELISTING_VOLUME_RATIO,
    PRELISTING_ZERO_VOLUME_SHARE, UNIVERSE_TABLE
)
from src.context import Context

_WIKI_HEADERS = {
    "User-Agent": "stock_pick_strat/1.0 (https://github.com; research@example.com)",
}


def _dedupe_share_classes(df: pd.DataFrame) -> pd.DataFrame:
    """Drop redundant dual-class listings (e.g. GOOG vs GOOGL, FOX vs FOXA, NWS vs
    NWSA): both share one CIK. Keep ONE row per CIK — the LONGEST symbol, which is
    the voting/Class-A line (GOOGL, FOXA, NWSA) rather than the non-voting Class-C
    (GOOG, FOX, NWS). Rows without a CIK are kept as-is."""
    if "cik" not in df.columns:
        return df
    has_cik = df[df["cik"].notna() & (df["cik"].astype(str).str.strip() != "")].copy()
    no_cik = df[~df.index.isin(has_cik.index)]
    has_cik["_len"] = has_cik["ticker"].str.len()
    kept = (has_cik.sort_values(["cik", "_len", "ticker"], ascending=[True, False, True])
            .drop_duplicates("cik", keep="first").drop(columns="_len"))
    dropped = sorted(set(has_cik["ticker"]) - set(kept["ticker"]))
    if dropped:
        print(f"Deduplicated {len(dropped)} redundant share-class tickers: {dropped}")
    return pd.concat([kept, no_cik], ignore_index=True).sort_values("ticker").reset_index(drop=True)


def get_sp500_tickers(context: Context) -> list[str]:
    """Scrape current S&P 500 tickers + sector info from Wikipedia. Adds the GICS
    industry group (24-level, for sector-neutral construction) and deduplicates
    dual-class share listings."""

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
        "CIK": "cik",
    })
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)  # yfinance format, e.g. BRK.B -> BRK-B
    if "cik" in df.columns:
        df["cik"] = df["cik"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)
    
    # GICS industry group (24) from sub-industry, sector fallback -> sector neutrality
    tick_redundant = context.config.data_extract.redundant_ticks
    df["industry_group"] = [industry_group(s, sec)
                            for s, sec in zip(df["sub_industry"], df["sector"])]
    df = _dedupe_share_classes(df)

    keep = [c for c in ["ticker", "name", "sector", "industry_group", "sub_industry", "cik"]
            if c in df.columns]
    df = df.loc[~df['ticker'].isin(tick_redundant)].reset_index(drop=True)
    context.store.save(UNIVERSE_TABLE, df[keep])
    print(f"Saved {len(df)} tickers to DB table {UNIVERSE_TABLE}")
    return df["ticker"].tolist()


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
    hole for `_interior_gap_start` to chase in a re-download loop.
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


def _load_existing_prices(context: Context) -> pd.DataFrame | None:
    df = context.store.load("prices")
    if df.empty:
        return None
    return _normalize_prices(df)


def _history_start(years_history: int) -> pd.Timestamp:
    return pd.Timestamp.today().normalize() - pd.DateOffset(years=years_history)


def _is_up_to_date(max_date: pd.Timestamp, today: pd.Timestamp) -> bool:
    # The freshest bar we could reasonably have is the previous *business* day
    # (today's close may not be published yet by yfinance). Using a business-day
    # offset stops weekends/Mondays from re-downloading Friday-anchored data:
    #   Sun/Sat/Mon -> Fri, Wed -> Tue, etc.
    last_expected = today - pd.tseries.offsets.BDay(1)
    return max_date >= last_expected


def _trading_calendar(existing: pd.DataFrame, min_coverage: float = 0.5) -> pd.DatetimeIndex:
    """Reference set of trading dates: dates present for at least `min_coverage`
    of the tickers in the file. Robust to any single ticker's interior gaps (a
    date is 'real' because MOST names traded it), so it can catch a benchmark
    (e.g. SPY) that is itself missing an interior window even though the stocks
    have it."""
    counts = existing.groupby("date")["ticker"].nunique()
    n_tickers = existing["ticker"].nunique()
    return counts.index[counts >= min_coverage * max(1, n_tickers)]


def _interior_gap_start(ticker_dates: pd.Series,
                        calendar: pd.DatetimeIndex) -> pd.Timestamp | None:
    """Earliest reference trading date the ticker is MISSING inside its own span
    [min, max]. None if the ticker has no interior hole. The incremental updater
    only appends past max_date, so without this an interior hole (e.g. a partial
    yfinance response saved once) is never revisited and drops those dates for
    the whole universe when SPY defines the cube calendar."""
    if calendar is None or len(calendar) == 0 or ticker_dates.empty:
        return None
    lo, hi = ticker_dates.min(), ticker_dates.max()
    expected = calendar[(calendar >= lo) & (calendar <= hi)]
    missing = expected.difference(pd.DatetimeIndex(ticker_dates.unique()))
    return missing.min() if len(missing) else None


def _tickers_needing_download(
    existing: pd.DataFrame | None,
    tickers: list[str],
    years_history: int,
) -> dict[str, tuple[pd.Timestamp, pd.Timestamp] | None]:
    """Return per-ticker (start, end) download windows; None means up to date.

    Backfill logic: `required_start` (today - N years) frequently lands on a
    weekend/holiday, and young tickers (recent IPOs) simply have no data that
    far back -- so a per-ticker `min_date <= required_start` test would force a
    perpetual full re-download. Instead we decide whether the *whole file*
    reaches back to the window (its global min is within a business-day grace of
    the cutoff). If it does, a ticker's own earlier-than-cutoff gap is just its
    IPO age and we don't re-pull. A real backward backfill (e.g. after raising
    `years_history`) is triggered only when the file itself doesn't reach back.
    """
    today = pd.Timestamp.today().normalize()
    required_start = _history_start(years_history)
    # First plausible bar on/after the cutoff, plus grace for a holiday-adjacent
    # weekend right at the boundary.
    backfill_floor = required_start + pd.tseries.offsets.BDay(3)

    global_min = None if existing is None or existing.empty else existing["date"].min()
    history_reaches_back = global_min is not None and global_min <= backfill_floor
    # trading calendar to detect interior holes (see _interior_gap_start)
    calendar = (_trading_calendar(existing)
                if existing is not None and not existing.empty else None)

    plans: dict[str, tuple[pd.Timestamp, pd.Timestamp] | None] = {}
    for tkr in tickers:
        if existing is None or tkr not in existing["ticker"].values:
            plans[tkr] = (required_start, today)
            continue

        ticker_data = existing.loc[existing["ticker"] == tkr, "date"]
        max_date = ticker_data.max()
        min_date = ticker_data.min()

        # Only backfill earlier bars when the dataset as a whole is short of the
        # window (not merely because this ticker is younger than the window).
        need_backfill = (not history_reaches_back) and (min_date > backfill_floor)
        # An interior hole is never healed by the tail-append path below, so
        # widen the window back to the first missing trading day when present.
        gap_start = _interior_gap_start(ticker_data, calendar)

        if need_backfill:
            plans[tkr] = (required_start, today)
        elif gap_start is not None:
            # one window that both heals the interior hole and refreshes the tail;
            # _merge_prices dedups keep="last", so re-fetched bars fill the gap
            plans[tkr] = (gap_start, today)
        elif not _is_up_to_date(max_date, today):
            start = max_date + pd.Timedelta(days=1)
            plans[tkr] = (start, today) if start <= today else None
        else:
            plans[tkr] = None

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
                "actions": True,          # also return Dividends / Stock Splits
                "threads": True,
                "progress": False,
            }
            if period is not None:
                kwargs["period"] = period
            else:
                kwargs["start"] = start.strftime(DATE_FORMAT)
                kwargs["end"] = (end + pd.Timedelta(days=1)).strftime(DATE_FORMAT)

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


# yfinance actions=True columns to DROP from the OHLCV `prices` table: dividends are
# saved separately (see _save_dividends); "capital gains" is a mutual-fund/ETF
# distribution field (~99% empty for equities, unused) — keep prices clean OHLCV.
_ACTION_COLS = ["dividends", "stock splits", "capital gains"]


def _extract_dividends(long_prices: pd.DataFrame | None) -> pd.DataFrame:
    """Pull the (raw, pre-adjust) cash dividends out of a normalized price frame
    that was downloaded with actions=True -> long [date, ticker, dividend], only
    the nonzero ex-dates. Pure. Empty if no dividends column."""
    cols = {"date", "ticker", "dividends"}
    if long_prices is None or long_prices.empty or not cols.issubset(long_prices.columns):
        return pd.DataFrame(columns=["date", "ticker", "dividend"])
    d = long_prices[["date", "ticker", "dividends"]].copy()
    d["dividends"] = pd.to_numeric(d["dividends"], errors="coerce")
    d = d[d["dividends"] > 0].rename(columns={"dividends": "dividend"})
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    return d.reset_index(drop=True)


def _save_dividends(context: Context, new_prices: pd.DataFrame) -> None:
    """Dividends piggy-back on the SAME yfinance price download (actions=True), so
    there is no separate dividend run. Accumulate nonzero ex-dates into the
    dividends parquet the aggregator already reads."""
    div = _extract_dividends(new_prices)
    if div.empty:
        return
        
    # upsert on (ticker, date) — the DB merges with any prior dividend rows
    n = context.store.save("dividends", div)
    print(f"Saved {n} dividend rows to DB table 'dividends' (from the price download)")


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
    download_dividends: bool = True,
) -> pd.DataFrame:
    """Download daily OHLCV for `tickers`, incrementally upserting into the `prices`
    DB table. `download_dividends=False` skips the dividend side-table — used for the
    market/macro price tickers (`fetch_market_prices`), which carry no equity data."""
    years_history = context.config.data_extract.years_history

    existing = _load_existing_prices(context)
    plans = _tickers_needing_download(existing, tickers, years_history)
    tickers_to_fetch = [t for t, window in plans.items() if window is not None]

    if not tickers_to_fetch:
        n = 0 if existing is None else len(existing)
        print(f"Price history already up to date ({n} rows) — DB table 'prices'")
        record_run(context, "prices", len(tickers), 0)
        return existing

    print(
        f"Downloading prices for {len(tickers_to_fetch)}/{len(tickers)} tickers "
        f"({len(tickers) - len(tickers_to_fetch)} already up to date)"
    )
    new = _download_prices(plans, years_history, chunk_size, pause)

    # dividends piggy-back on the SAME download (actions=True); skipped for the
    # market/macro tickers, which are not part of the equity universe
    if download_dividends:
        _save_dividends(context, new)

    # keep the prices table a clean OHLCV frame (drop the action columns)
    if not new.empty:
        new = new.drop(columns=_ACTION_COLS, errors="ignore")
        # Drop the synthetic pre-listing prefix BEFORE the upsert, so a full-history
        # pull never writes another ticker's predecessor line into `prices`. Applied to
        # the freshly-downloaded frame only: an incremental tail carries no prefix, and
        # the dividend extraction above already ran on the untrimmed frame (a dividend
        # paid by the predecessor entity is still a real ex-date for this symbol).
        new = trim_prelisting_bars(new)
    out = _merge_prices(existing, new, years_history)
    # upsert only the freshly-downloaded delta; the DB merges on (ticker, date)
    if not new.empty:
        context.store.save("prices", new)
    print(f"Saved {len(new)} new price rows to DB table 'prices' "
          f"(table now spans {len(out)} rows in memory)")
    record_run(context, "prices", len(tickers), len(new))

    return out


def fetch_market_prices(
    context: Context,
    chunk_size: int = 50,
    pause: float = 2.0,
) -> pd.DataFrame | None:
    """Fetch OHLCV (open/close/volume) for the benchmark + macro price tickers
    (`data_extract.other_tickers`: SPY, ^VIX, oil/gold, FX) into the `prices` table.

    These are NOT part of the equity universe: no fundamentals / behavioral / SEC data
    and NO dividends — they exist only so the cube can read the market-beta benchmark
    and the commodity / currency factor series. Kept in `prices` (not `macro`) because
    they are per-ticker OHLCV, which is exactly the price pipeline's shape; the cube
    already restricts FEATURES to the `sp500_tickers` universe, so these never become
    peers or feature rows."""
    others = list(context.config.data_extract.other_tickers)
    if not others:
        return None
    print(f"Fetching market/macro price series (OHLCV, no dividends): {others}")
    return fetch_price_history(context, tickers=others, chunk_size=chunk_size,
                               pause=pause, download_dividends=False)
