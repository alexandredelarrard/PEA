"""The four single-threaded Sharadar fetchers, all resumable from the DB.

`fetch_sharadar_tickers` (entity dimension; must run first, it supplies the USD check), `fetch_sharadar_fundamentals`
(SF1, one request per (ticker, dimension)), `fetch_sharadar_actions` and `fetch_sharadar_sp500` (market-wide).
SF1 resumes per TICKER from its own last filing `date` minus the contract overlap (`resume.series_windows`); the two
market-wide tables from the table's last date minus the overlap. `lastupdated` is not a watermark, so a Sharadar
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
    NotEntitledError,
    SharadarRequestError,
    canonical_symbols,
    cast_value_columns,
    coerce_date_columns,
    sharadar_get,
    vendor_symbol,
)
from src.data_store.schema import Table, Tables
from src.utils.polite_http import sleep_pace

# As-reported dimensions only: point-in-time and immutable (MR* rows restate in place).
SHARADAR_DIMENSIONS = ("ARQ", "ARY", "ART")

# History floor for `sharadar_sp500`; membership is always pulled at full depth.
SHARADAR_SP500_FIRST_DATE = "1990-01-01"


def _pace(context: Context) -> float:
    return float(context.config.data_extract.sharadar_request_pace)


def _run_date(as_of: pd.Timestamp | None) -> pd.Timestamp:
    return pd.Timestamp(as_of if as_of is not None else pd.Timestamp.today()).normalize()


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

    Non-USD filers are skipped, not written. A ticker the subscription does not cover (`NotEntitledError`) is
    counted, not retried. A ticker with a failed page saves nothing, is named in the coverage log, and is
    fetched again next run from the same window.
    """
    currencies = _usd_roster(context)
    run_date = _run_date(as_of)
    work = series_windows(
        context, Tables.sharadar_fundamentals, tickers, run_date, until=run_date, years_history=years_history, full=full, calendar=None
    )
    pace = _pace(context)
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

    for ticker in tqdm(tickers, desc="Downloading tickers"):
        # vendor spelling (`BRK.B`) for both the request and the roster lookup; the wrong one returns 0 rows
        symbol = vendor_symbol(ticker)
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
# 3. actions / 4. sp500 -- market-wide, resumed on the table's global max date  #
# --------------------------------------------------------------------------- #
def _fetch_dated_table(context: Context, table: Table, endpoint: str, since: str) -> None:
    """Shared body for the two market-wide, date-resumed side tables."""
    try:
        frame = sharadar_get(context, endpoint, keep_default_na=False, sort="date.asc", **cast(dict[str, Any], {"date.gte": since}))
    except SharadarRequestError as error:
        context.log.warning("Sharadar %s: %s; %s left unchanged", endpoint, error, table)
        return
    if frame.empty:
        context.log.info("Sharadar %s: no rows since %s; %s already current", endpoint, since, table)
        return
    frame = coerce_date_columns(frame, table.date_type_cols)
    # share classes to the repo's spelling, since these tables join on repo tickers
    if "ticker" in frame.columns:
        frame["ticker"] = canonical_symbols(frame["ticker"])
    if "value" in frame.columns:
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce").astype("float64")
    written = context.store.save(table, frame)
    context.log.info("Sharadar %s: %d row(s) since %s -> %s (actions: %s)", endpoint, written, since, table, frame["action"].value_counts().to_dict())


def fetch_sharadar_actions(context: Context, *, years_history: int, full: bool = False, as_of: pd.Timestamp | None = None) -> None:
    """Corporate actions -> `sharadar_actions`, market-wide, from the table's last date minus the overlap."""
    floor = document_floor(Tables.sharadar_actions, _run_date(as_of), years_history)
    _fetch_dated_table(context, Tables.sharadar_actions, "actions", _table_since(context, Tables.sharadar_actions, floor, full))


def fetch_sharadar_sp500(context: Context, *, full: bool = False) -> None:
    """S&P 500 membership events -> `sharadar_sp500`; cold or `full` pulls from `SHARADAR_SP500_FIRST_DATE`."""
    floor = pd.Timestamp(SHARADAR_SP500_FIRST_DATE)
    _fetch_dated_table(context, Tables.sharadar_sp500, "sp500", _table_since(context, Tables.sharadar_sp500, floor, full))
