"""The four single-threaded Sharadar fetchers.

`fetch_sharadar_tickers` (entity dimension; must run first, it supplies the USD check), `fetch_sharadar_fundamentals`
(SF1, one request per (ticker, dimension)), `fetch_sharadar_actions` (market-wide, full replace) and `fetch_sharadar_sp500`
(market-wide, resumed).
`predecessor_vendor_tickers` adds the vendor tickers carrying a register predecessor CIK's own series and the cited
`vendor_series_overrides` of `security_master_manual.json`.
SF1 resumes per TICKER from its own last filing `date` minus the contract overlap (`resume.series_windows`); `sp500`
from the table's last date minus the overlap; `actions` is pulled whole each run. `lastupdated` is not a watermark, so a Sharadar
restatement older than the overlap needs `--full`. A failed page fails its ticker (or table) whole; nothing partial is saved.
"""

from __future__ import annotations

from typing import Any, cast

import pandas as pd
from tqdm import tqdm

from src.constants.constants import DATE_FORMAT, SHARADAR_BASE_URL, SHARADAR_SF1_COLUMNS
from src.context import Context
from src.data_extract.utils.common.resume import document_floor, series_windows
from src.data_extract.utils.fundamentals_sharadar.client import (
    PAGE_LIMIT,
    NotEntitledError,
    SharadarRequestError,
    canonical_symbols,
    cast_value_columns,
    coerce_date_columns,
    sharadar_get,
)
from src.data_store.schema import Resume, Table, Tables
from src.utils.polite_http import sleep_pace
from src.utils.predecessor_series import load_predecessor_series
from src.utils.string import run_date

# As-reported dimensions only: point-in-time and immutable (MR* rows restate in place).
SHARADAR_DIMENSIONS = ("ARQ", "ARY", "ART")


def _table_since(context: Context, table: Table, floor: pd.Timestamp, full: bool) -> str:
    """The `date.gte` bound of a market-wide table: its last stored date minus the contract overlap, or `floor` when cold or `full`.

    Always explicit: the API's default lower bound is one year ago."""
    stored_max = None if full else context.store.max_date(table)
    if stored_max is None:
        return floor.strftime(DATE_FORMAT)
    overlap = pd.Timedelta(days=table.resume.overlap_days if table.resume is not None else 0)
    return max(stored_max - overlap, floor).strftime(DATE_FORMAT)


def _usd_roster(context: Context) -> dict[str, str]:
    """`{ticker: currency}` from `sharadar_tickers`, preferring the live (not delisted) row; raises if empty."""
    frame = context.store.load(Tables.sharadar_tickers, columns=["ticker", "currency", "isdelisted"])
    if frame is None or frame.empty:
        raise RuntimeError(
            f"{Tables.sharadar_tickers} is empty -- run the tickers fetch first. The "
            f"fundamentals fetch reads `currency` from it to assert USD (D20), and only 8 "
            f"of Sharadar's money columns are USD-converted, so a non-USD row mixes units "
            f"INSIDE ITSELF."
        )
    frame = frame.assign(_live=(frame["isdelisted"].astype(str).str.upper() != "Y"))
    frame = frame.sort_values("_live", ascending=False).drop_duplicates("ticker")
    return dict(zip(frame["ticker"].astype(str), frame["currency"].astype(str), strict=False))


def predecessor_vendor_tickers(context: Context, tickers: list[str]) -> list[str]:
    """The vendor tickers to fetch beside `tickers`, stored under their own vendor ticker."""
    return sorted({s.vendor_ticker for s in load_predecessor_series(context, tickers)})


# --------------------------------------------------------------------------- #
# 1. tickers -- the entity dimension (must run first)                          #
# --------------------------------------------------------------------------- #
def fetch_sharadar_tickers(context: Context) -> None:
    """Full refresh of `sharadar_tickers` (SF1 coverage rows); no date to resume on and its fields mutate."""
    try:
        frame = sharadar_get(context, "tickers", keep_default_na=False, **cast(dict[str, Any], {"table": "fundamentals"}))
    except SharadarRequestError as error:
        context.log.warning("Sharadar tickers: %s; %s left unchanged", error, Tables.sharadar_tickers)
        return
    if frame.empty:
        context.log.warning("Sharadar tickers: no rows returned; %s left unchanged", Tables.sharadar_tickers)
        return
    frame = coerce_date_columns(frame, Tables.sharadar_tickers.date_type_cols)
    frame["permaticker"] = pd.to_numeric(frame["permaticker"], errors="coerce").astype("Int64")
    written = context.store.save(Tables.sharadar_tickers, frame)
    context.log.info(
        "Sharadar tickers: %d rows -> %s (%d USD, %d non-USD)",
        written,
        Tables.sharadar_tickers,
        int((frame["currency"] == "USD").sum()),
        int((frame["currency"] != "USD").sum()),
    )


# --------------------------------------------------------------------------- #
# 2. fundamentals (SF1)                                                        #
# --------------------------------------------------------------------------- #
def _sf1_pages(context: Context, symbol: str, since: str, pace: float) -> list[pd.DataFrame]:
    """The non-empty SF1 pages of `symbol` since `since`, one request per dimension; raises `NotEntitledError`."""
    frames: list[pd.DataFrame] = []
    for dimension in SHARADAR_DIMENSIONS:
        df_page = sharadar_get(
            context,
            "fundamentals",
            expect_columns=SHARADAR_SF1_COLUMNS,
            ticker=symbol,
            dimension=dimension,
            sort="date.asc",
            **cast(dict[str, Any], {"date.gte": since}),
        )
        if not df_page.empty:
            frames.append(df_page)
        sleep_pace(pace, SHARADAR_BASE_URL)
    return frames


def fetch_sharadar_fundamentals(
    context: Context, tickers: list[str], *, years_history: int, full: bool = False, as_of: pd.Timestamp | None = None
) -> None:
    """SF1 for `tickers` x `SHARADAR_DIMENSIONS` -> `fundamentals_sharadar`, each ticker from its own window.

    Each ticker is requested under its trading symbol (`Universe.trading_symbols`) and stored under the ticker; a
    ticker with no window (ended, with stored rows, run past its grace end plus overlap) is skipped. Non-USD filers are skipped, not written. A ticker the subscription does not cover (`NotEntitledError`) is
    counted, not retried. A ticker with a failed page saves nothing, is named in the coverage log, and is
    fetched again next run from the same window.
    """
    currencies = _usd_roster(context)
    start_date = run_date(as_of)
    work = series_windows(
        context, Tables.sharadar_fundamentals, tickers, start_date, until=start_date, years_history=years_history, full=full, calendar=None
    )
    pace = float(context.config.data_extract.sharadar_request_pace)
    context.log.info(
        "Sharadar SF1: %d ticker(s) x %d dimension(s); key classes %s (full=%s, window=%dy)",
        len(tickers),
        len(SHARADAR_DIMENSIONS),
        {cls: sum(1 for c in work.key_class.values() if c == cls) for cls in sorted(set(work.key_class.values()))},
        full,
        years_history,
    )

    entitled: list[str] = []
    denied: list[str] = []
    non_usd: list[str] = []
    failed: list[str] = []
    total_rows = 0
    symbols = context.universe.trading_symbols
    for ticker in tqdm(tickers, desc="Downloading tickers"):
        # an ended ticker with stored rows has no window once the run is past its grace end plus overlap
        if not work.windows.get(ticker):
            context.log.debug("Sharadar SF1: %s has no window (ended)", ticker)
            continue

        # the trading symbol in vendor spelling (`BRK.B`) for both the request and the roster lookup; the wrong one returns 0 rows
        symbol = symbols.get(ticker, ticker).replace("-", ".")
        currency = currencies.get(symbol)
        if currency is not None and currency != "USD":
            # refuse to write: only some SF1 columns are USD-converted, so the row would mix units
            non_usd.append(ticker)
            context.log.warning(
                "Sharadar SF1: %s reports in %s, not USD -- NOT WRITTEN. "
                "Only 8 SF1 columns are USD-converted, so the row would mix "
                "units within itself (D20).",
                ticker,
                currency,
            )
            continue

        since = work.windows[ticker][0][0].strftime(DATE_FORMAT)
        try:
            frames = _sf1_pages(context, symbol, since, pace)
        except NotEntitledError:
            denied.append(ticker)
            continue
        except SharadarRequestError as error:
            failed.append(ticker)
            context.log.warning("Sharadar SF1: %s failed (%s); nothing saved for it this run", ticker, error)
            continue

        entitled.append(ticker)
        if not frames:
            context.log.debug("Sharadar SF1: %s up to date (since %s)", ticker, since)
            continue

        df_sf1 = pd.concat(frames, ignore_index=True)
        # relabel to the repo's canonical ticker, which every downstream join keys on
        df_sf1["ticker"] = ticker
        # cast before the first write: `ensure_table` would type an all-None object column as TEXT
        df_sf1 = cast_value_columns(df_sf1)
        df_sf1 = coerce_date_columns(df_sf1, Tables.sharadar_fundamentals.date_type_cols)
        total_rows += context.store.save(Tables.sharadar_fundamentals, df_sf1)

    context.log.info(
        "Sharadar SF1: %d entitled, %d not entitled (403), %d failed; %d rows written to %s",
        len(entitled),
        len(denied),
        len(failed),
        total_rows,
        Tables.sharadar_fundamentals,
    )
    if failed:
        context.log.warning("Sharadar SF1: %d ticker(s) failed and are retried next run -> %s", len(failed), ", ".join(failed))
    if denied:
        context.log.info("Sharadar SF1: not entitled -> %s", ", ".join(denied[:20]) + (" ..." if len(denied) > 20 else ""))
    if non_usd:
        context.log.warning("Sharadar SF1: %d non-USD filer(s) skipped -> %s", len(non_usd), ", ".join(non_usd))


# --------------------------------------------------------------------------- #
# 3. actions (full replace) / 4. sp500 (resumed on the table's global max date) #
# --------------------------------------------------------------------------- #
def _clean_dated_frame(frame: pd.DataFrame, table: Table) -> pd.DataFrame:
    """A market-wide response in the repo's spelling and types: dates, share-class tickers (these tables join on repo
    tickers) and numeric values."""
    frame = coerce_date_columns(frame, table.date_type_cols)
    if "ticker" in frame.columns:
        frame["ticker"] = canonical_symbols(frame["ticker"])
    if "value" in frame.columns:
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce").astype("float64")
    return frame


def _pull_actions(context: Context, since: pd.Timestamp, until: pd.Timestamp) -> pd.DataFrame:
    """`actions` rows dated `since`..`until`, one request per date window that fits one page.

    Windows start quarterly and a full page halves its window, so the API's per-query offset cap, and offset pages that
    repeat or skip rows tied on `date`, are never relied on. Raises `SharadarRequestError` when a page fails and
    `RuntimeError` when a row falls outside its requested window.
    """
    pace = float(context.config.data_extract.sharadar_request_pace)
    day = pd.Timedelta(days=1)
    edges = sorted({since, *pd.date_range(since, until, freq="QS"), until + day})
    pending = [(start, end - day) for start, end in zip(edges, edges[1:], strict=False)]
    frames: list[pd.DataFrame] = []
    while pending:
        start, end = pending.pop()
        bounds = {"date.gte": start.strftime(DATE_FORMAT), "date.lte": end.strftime(DATE_FORMAT)}
        frame = sharadar_get(context, "actions", keep_default_na=False, sort="date.asc", **cast(dict[str, Any], bounds))
        sleep_pace(pace, SHARADAR_BASE_URL)
        dates = pd.to_datetime(frame["date"]) if not frame.empty else pd.Series(dtype="datetime64[ns]")
        if not dates.between(start, end).all():
            raise RuntimeError(f"Sharadar actions: rows outside the requested window {bounds}; the date filter was not applied")
        if len(frame) >= PAGE_LIMIT and end > start:
            middle = start + pd.Timedelta(days=(end - start).days // 2)
            pending += [(start, middle), (middle + day, end)]
            continue
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_sharadar_actions(context: Context, *, years_history: int, as_of: pd.Timestamp | None = None) -> int:
    """Corporate actions -> `sharadar_actions`, market-wide: the whole window is pulled and replaces the table.

    `ticker` is in the PK, so a re-pull is a replace, not an upsert. A failed, empty or truncated pull (fewer rows than
    90 % of the stored table) leaves the table unchanged. Returns the rows written.
    """
    # load: the stored size the pull is checked against
    stored = context.store.row_count(Tables.sharadar_actions) if context.store.exists(Tables.sharadar_actions) else 0

    # pull the whole window, window by window
    start_date = run_date(as_of)
    since = document_floor(Tables.sharadar_actions, start_date, years_history)
    try:
        frame = _pull_actions(context, since, start_date)
    except SharadarRequestError as error:
        context.log.warning("Sharadar actions: %s; %s left unchanged", error, Tables.sharadar_actions)
        return 0
    if frame.empty:
        context.log.warning("Sharadar actions: no rows %s..%s; %s left unchanged", since.date(), start_date.date(), Tables.sharadar_actions)
        return 0

    # clean: repo spelling and types, one row per primary key (COPY refuses a duplicate key)
    frame = _clean_dated_frame(frame, Tables.sharadar_actions)
    pk = list(Tables.sharadar_actions.pk)
    duplicated = int(frame.duplicated(pk).sum())
    frame = frame.drop_duplicates(pk, keep="last")

    # check: a truncated feed never replaces the table
    if len(frame) < 0.9 * stored:
        context.log.warning("Sharadar actions: %d row(s) pulled vs %d stored; %s left unchanged", len(frame), stored, Tables.sharadar_actions)
        return 0

    # replace
    written = context.store.replace(Tables.sharadar_actions, frame)
    context.log.info(
        "Sharadar actions: %d row(s) %s..%s replace %d stored (%d duplicate key row(s) dropped; actions: %s)",
        written,
        since.date(),
        start_date.date(),
        stored,
        duplicated,
        frame["action"].value_counts().to_dict(),
    )
    return written


def fetch_sharadar_sp500(context: Context, *, full: bool = False) -> None:
    """S&P 500 membership events; cold or `full` pulls from the table's source start."""
    floor = pd.Timestamp(cast(str, cast(Resume, Tables.sharadar_sp500.resume).source_start))
    since = _table_since(context, Tables.sharadar_sp500, floor, full)
    try:
        frame = sharadar_get(context, "sp500", keep_default_na=False, sort="date.asc", **cast(dict[str, Any], {"date.gte": since}))
    except SharadarRequestError as error:
        context.log.warning("Sharadar sp500: %s; %s left unchanged", error, Tables.sharadar_sp500)
        return
    if frame.empty:
        context.log.info("Sharadar sp500: no rows since %s; %s already current", since, Tables.sharadar_sp500)
        return
    written = context.store.save(Tables.sharadar_sp500, _clean_dated_frame(frame, Tables.sharadar_sp500))
    context.log.info(
        "Sharadar sp500: %d row(s) since %s -> %s (actions: %s)", written, since, Tables.sharadar_sp500, frame["action"].value_counts().to_dict()
    )
