"""
fetch_prices.py  (src/data_extract/utils/prices/fetch_prices.py)
------------------------------------------------------------------
Daily OHLCV, cash dividends and share splits per equity ticker from ONE yfinance `actions=True`
download per window group, split into `prices`, `prices_dividends` and `prices_splits`.

Windows are planned by `resume.series_windows` from the tables' own rows. `download_ohlcv` is the
shared chunked entry point (the macro fetcher calls it on the total-return basis). Market/macro
series are not fetched here; they live in `prices_macro` (`fetch_macro.py`).
`prices` holds the universe AND the current secondary share classes of its companies (`security_master` rows
with role `secondary_class` and an open end: BRK-A, GOOG, LEN-B, ...), each under its own Yahoo symbol; every
reader of `prices` filters on the universe. Dividends and splits are stored for the universe only.
"""

import logging
import time
from typing import cast

import pandas as pd
import yfinance as yf
from tqdm import tqdm

from src.constants.constants import DATE_FORMAT, SECONDARY_CLASS
from src.context import Context
from src.data_extract.utils.common.resume import document_floor, series_windows, trading_calendar
from src.data_extract.utils.common.sessions import last_completed_session
from src.data_extract.utils.prices.fetch_dividends import _extract_dividends
from src.data_extract.utils.prices.fetch_splits import _extract_splits
from src.data_store.schema import Tables
from src.utils.string import yahoo_symbol

logger = logging.getLogger(__name__)

#: yfinance action columns: kept for the dividend and split rows, never written to `prices`.
_ACTION_COLUMNS = ("dividends", "stock splits", "capital gains")

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
            f"single-basis price frame. Columns present: {sorted(out.columns)}"
        )
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
    volume = cast(pd.Series, pd.to_numeric(f["volume"], errors="coerce"))
    zero = volume.fillna(0) <= 0
    if not zero.any():
        return None

    last_zero_value = cast(pd.Series, f.loc[zero, "date"]).max()
    if pd.isna(last_zero_value):
        return None
    last_zero = cast(pd.Timestamp, last_zero_value)
    dates = cast(pd.Series, f["date"])
    window = dates <= last_zero
    if window.sum() and zero[window].mean() >= PRELISTING_ZERO_VOLUME_SHARE:
        return last_zero

    # Flat SPAC-trust / stub regime that still records token volume: compare the first
    # year against the ticker's own long-run level, so the test is scale-free.
    first_year = dates <= dates.min() + pd.DateOffset(years=1)
    early, overall = volume.loc[first_year].median(), volume.median()
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
    for _ticker, group in prices.groupby("ticker", sort=False):
        cutoff = _prelisting_cutoff(group)
        if cutoff is not None:
            drop.loc[group.index[group["date"] <= cutoff]] = True
    if not drop.any():
        return prices
    return prices.loc[~drop].reset_index(drop=True)


def _chunk_response_to_frames(data: pd.DataFrame, chunk: list[str]) -> list[pd.DataFrame]:
    frames = []
    if isinstance(data.columns, pd.MultiIndex):
        served = set(data.columns.get_level_values(0))
        missing = [t for t in chunk if t not in served]
        if missing:
            # WARN, do not pass over it. A ticker yfinance declines to serve used to vanish
            # here without a trace, so a truncated response was indistinguishable from a
            # complete one and only surfaced three tables downstream as a thin cross-section.
            # This is the single line that made the 45-of-491 day invisible at fetch time.
            logger.warning(
                "yfinance returned no data for %d of %d tickers in chunk %s..%s -- their bars for this window are MISSING, not empty: %s",
                len(missing),
                len(chunk),
                chunk[0],
                chunk[-1],
                ", ".join(missing),
            )
        for tkr in chunk:
            if tkr not in served:
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
            data = cast(
                pd.DataFrame,
                yf.download(
                    chunk,
                    start=start.strftime(DATE_FORMAT),
                    end=(end + pd.Timedelta(days=1)).strftime(DATE_FORMAT),
                    interval="1d",
                    group_by="ticker",
                    auto_adjust=auto_adjust,
                    actions=actions,  # also return Dividends / Stock Splits
                    threads=True,
                    progress=False,
                ),
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
    stock splits]. Empty frame when every chunk failed; `actions` adds the action columns.

    `auto_adjust` is keyword-only and required because the callers need different bases:
    `fetch_prices_and_actions` passes False (split-adjusted `Close`, the market-cap basis, plus
    `Adj Close`, total return); `fetch_macro._fetch_price_leg` passes True, because its `SPY` leg
    is stored as `equity_tr` and feeds the L/S benchmark, `beta_market` and every label's `fwd_market`."""
    frames: list[pd.DataFrame] = []
    for i in tqdm(range(0, len(tickers), chunk_size), desc=desc):
        chunk = tickers[i : i + chunk_size]
        frames.extend(_download_price_chunk(chunk, since, until, pause, actions, auto_adjust))
        time.sleep(pause)

    if not frames:
        return pd.DataFrame()

    return _normalize_prices(pd.concat(frames, ignore_index=True), auto_adjust)


def secondary_class_symbols(context: Context, companies: list[str]) -> dict[str, str]:
    """`{Yahoo symbol: canonical company}` of the companies' secondary classes that `security_master` holds as current.

    A class whose every row has an end is no longer listed and is not fetched (yfinance has no history for it).
    """
    rows = context.store.load(
        Tables.security_master,
        columns=["canonical_company", "market_symbol", "valid_to"],
        where={"lineage_role": SECONDARY_CLASS, "canonical_company": companies},
        optional=True,
    )
    if rows is None or rows.empty:
        return {}
    rows = rows.assign(symbol=rows["market_symbol"].map(yahoo_symbol), current=rows["valid_to"].isna())
    current = cast(pd.DataFrame, rows.groupby(["symbol", "canonical_company"], as_index=False)["current"].any())
    triples = zip(current["symbol"].astype(str), current["canonical_company"].astype(str), current["current"].astype(bool), strict=True)
    listed = {symbol: company for symbol, company, is_current in triples if is_current and symbol not in companies}
    delisted = sorted(set(current.loc[~current["current"].astype(bool), "symbol"].astype(str)) - set(listed))
    if delisted:
        logger.info("%d secondary class(es) no longer listed, not fetched: %s", len(delisted), ", ".join(delisted))
    return dict(sorted(listed.items()))


def _split_repulls(df_splits: pd.DataFrame, last: dict[str, pd.Timestamp], full_keys: set[str]) -> list[str]:
    """Keys whose response carries a split after their last stored bar: split adjustment restates
    every earlier bar, so their stored history is on a stale basis. Keys already pulled in full are skipped."""
    if df_splits.empty:
        return []
    stale = {
        str(ticker)
        for ticker, day in zip(df_splits["ticker"], pd.to_datetime(df_splits["date"]), strict=True)
        if str(ticker) in last and str(ticker) not in full_keys and day > last[str(ticker)]
    }
    return sorted(stale)


def _price_rows(df_raw: pd.DataFrame) -> pd.DataFrame:
    """The `prices` rows of an `actions=True` response: the action columns dropped, rows with no bar
    dropped (what an `actions=False` response holds), then the synthetic pre-listing prefix trimmed."""
    if df_raw.empty:
        return df_raw
    df_bars = df_raw.drop(columns=[c for c in _ACTION_COLUMNS if c in df_raw.columns])
    bar_columns = [c for c in df_bars.columns if c not in ("date", "ticker")]
    return trim_prelisting_bars(df_bars.dropna(subset=bar_columns, how="all").reset_index(drop=True))


def _download_groups(groups: list[tuple[pd.Timestamp, pd.Timestamp, list[str]]], chunk_size: int, pause: float, label: str) -> pd.DataFrame:
    """One `actions=True` download per window group, concatenated; duplicate `(ticker, date)` rows keep the last."""
    frames = []
    for since, until, keys in groups:
        logger.info("Downloading prices and actions for %d ticker(s) over %s .. %s (%s)", len(keys), since.date(), until.date(), label)
        frames.append(download_ohlcv(keys, since, until, chunk_size, pause, desc=f"Downloading prices ({label})", auto_adjust=False, actions=True))
    frames = [df for df in frames if not df.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["ticker", "date"], keep="last").reset_index(drop=True)


def fetch_prices_and_actions(
    context: Context,
    tickers: list[str],
    years_history: int,
    *,
    full: bool = False,
    as_of: pd.Timestamp | None = None,
    chunk_size: int = 50,
    pause: float = 1.0,
) -> None:
    """`prices`, `prices_dividends` and `prices_splits` from ONE yfinance `actions=True` download per window group.

    Windows come from `series_windows` over `prices` and `prices_dividends` (each key from its own last
    date minus the overlap, holes, new keys in full) and end at the last completed session. The companies'
    current secondary classes (`secondary_class_symbols`) are fetched alongside; a class with no stored bar
    takes the whole window. A key whose response holds a split after its last stored bar is re-pulled over
    the whole window before saving."""
    run_date = pd.Timestamp(as_of if as_of is not None else pd.Timestamp.today()).normalize()
    until = last_completed_session(as_of)
    calendar = trading_calendar(context)
    plan = {"until": until, "years_history": years_history, "full": full, "calendar": calendar}
    owners = secondary_class_symbols(context, tickers)
    if owners:
        logger.info("%d current secondary class(es) fetched with their companies: %s", len(owners), ", ".join(owners))
    symbols = [*tickers, *owners]
    floor = document_floor(Tables.prices, run_date, years_history)
    prices_work = series_windows(context, Tables.prices, symbols, run_date, **plan)
    # A secondary class is never a new universe key, so one with no stored bar takes the whole window here.
    for symbol in owners:
        if symbol not in prices_work.last:
            prices_work.windows[symbol] = [(floor, until)]
    work = prices_work.merge(series_windows(context, Tables.dividends, tickers, run_date, **plan))
    df_raw = _download_groups(work.groups(), chunk_size, pause, "resume")

    full_keys = {key for key, spans in work.windows.items() if spans and spans[0][0] <= floor}
    repull = _split_repulls(_extract_splits(df_raw), prices_work.last, full_keys)
    if repull:
        logger.info("%d ticker(s) split after their last stored bar; re-pulling their whole window: %s", len(repull), ", ".join(repull))
        df_raw = pd.concat(
            [df_raw[~df_raw["ticker"].isin(repull)], _download_groups([(floor, until, repull)], chunk_size, pause, "post-split re-pull")],
            ignore_index=True,
        )

    n_prices = context.store.save(Tables.prices, _price_rows(df_raw))
    # Dividends and splits are kept for the universe only; a secondary class lands in `prices` alone.
    df_universe = df_raw[df_raw["ticker"].isin(set(tickers))] if not df_raw.empty else df_raw
    n_dividends = context.store.save(Tables.dividends, _extract_dividends(df_universe))
    df_splits = _extract_splits(df_universe)
    n_splits = context.store.save(Tables.prices_splits, df_splits) if not df_splits.empty else 0
    served = set(df_raw["ticker"]) if not df_raw.empty else set()
    missing = sorted(key for key, spans in work.windows.items() if spans and key not in served)
    logger.info(
        "Saved %d price, %d dividend and %d split row(s) for %d ticker(s) in %d window group(s); key classes %s",
        n_prices,
        n_dividends,
        n_splits,
        len(symbols),
        len(work.groups()),
        {cls: sum(1 for c in work.key_class.values() if c == cls) for cls in sorted(set(work.key_class.values()))},
    )
    if missing:
        logger.warning("No bars returned for %d ticker(s) with a window; they are re-listed next run: %s", len(missing), ", ".join(missing))
