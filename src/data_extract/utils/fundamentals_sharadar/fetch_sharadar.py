"""The four single-threaded Sharadar fetchers, all resumable from the DB.

`fetch_sharadar_tickers` (entity dimension; must run first, it supplies the USD check), `fetch_sharadar_fundamentals`
(SF1, one request per (ticker, dimension)), `fetch_sharadar_actions` and `fetch_sharadar_sp500` (market-wide).
SF1 resumes per TICKER from the max stored filing `date` (safe: an ARY row is filed with the 10-K, never after the
ticker-wide watermark). `lastupdated` is not a watermark, so a Sharadar restatement needs `--full`.
"""

from __future__ import annotations

from typing import Any, cast

import pandas as pd
from tqdm import tqdm

from src.constants.constants import DATE_FORMAT, SHARADAR_BASE_URL, SHARADAR_SF1_COLUMNS
from src.context import Context
from src.data_extract.utils.common.run_manifest import record_run
from src.data_extract.utils.fundamentals_sharadar.client import (
    NotEntitledError,
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


def _cold_start(years_history: int) -> pd.Timestamp:
    return pd.Timestamp.today().normalize() - pd.DateOffset(years=int(years_history))


def _since(stored_max: pd.Timestamp | None, years_history: int, full: bool) -> str:
    """The `date.gte` bound: the day after `stored_max`, or the `years_history` window when cold or `full`.

    Always explicit: the API's default lower bound is one year ago.
    """
    if full or stored_max is None:
        return _cold_start(years_history).strftime(DATE_FORMAT)
    return (stored_max + pd.Timedelta(days=1)).strftime(DATE_FORMAT)


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
    frame = sharadar_get(context, "tickers", keep_default_na=False, **cast(dict[str, Any], {"table": "fundamentals"}))
    if frame is None or frame.empty:
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
    # market-wide, so ticker_count=0
    record_run(context, Tables.sharadar_tickers, 0, written, is_full_rescan=True)


# --------------------------------------------------------------------------- #
# 2. fundamentals (SF1)                                                        #
# --------------------------------------------------------------------------- #
def fetch_sharadar_fundamentals(context: Context, tickers: list[str], *, years_history: int, full: bool = False) -> None:
    """SF1 for `tickers` x `SHARADAR_DIMENSIONS` -> `fundamentals_sharadar`, resumed per ticker.

    Non-USD filers are skipped, not written. A ticker the subscription does not cover (`NotEntitledError`) is
    counted, not retried.
    """
    currencies = _usd_roster(context)
    resume = context.store.max_date_by(Tables.sharadar_fundamentals, "ticker")
    pace = _pace(context)
    context.log.info(
        "Sharadar SF1: %d ticker(s) x %d dimension(s); %d already have stored rows (full=%s, window=%dy)",
        len(tickers),
        len(SHARADAR_DIMENSIONS),
        len(resume),
        full,
        years_history,
    )

    entitled: list[str] = []
    denied: list[str] = []
    non_usd: list[str] = []
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

        since = _since(resume.get(ticker), years_history, full)
        frames: list[pd.DataFrame] = []
        try:
            for dimension in SHARADAR_DIMENSIONS:
                page = sharadar_get(
                    context,
                    "fundamentals",
                    expect_columns=SHARADAR_SF1_COLUMNS,
                    ticker=symbol,
                    dimension=dimension,
                    sort="date.asc",
                    **cast(dict[str, Any], {"date.gte": since}),
                )
                if page is not None and not page.empty:
                    frames.append(page)
                sleep_pace(pace, SHARADAR_BASE_URL)
        except NotEntitledError:
            denied.append(ticker)
            continue

        entitled.append(ticker)
        if not frames:
            context.log.debug("Sharadar SF1: %s up to date (since %s)", ticker, since)
            continue

        frame = pd.concat(frames, ignore_index=True)
        # relabel to the repo's canonical ticker, which every downstream join keys on
        frame["ticker"] = ticker
        # cast before the first write: `ensure_table` would type an all-None object column as TEXT
        frame = cast_value_columns(frame)
        frame = coerce_date_columns(frame, Tables.sharadar_fundamentals.date_type_cols)
        total_rows += context.store.save(Tables.sharadar_fundamentals, frame)

    context.log.info(
        "Sharadar SF1: %d entitled, %d not entitled (403); %d rows written to %s",
        len(entitled),
        len(denied),
        total_rows,
        Tables.sharadar_fundamentals,
    )
    record_run(context, Tables.sharadar_fundamentals, len(tickers), total_rows, is_full_rescan=full)
    if denied:
        context.log.info("Sharadar SF1: not entitled -> %s", ", ".join(denied[:20]) + (" ..." if len(denied) > 20 else ""))
    if non_usd:
        context.log.warning("Sharadar SF1: %d non-USD filer(s) skipped -> %s", len(non_usd), ", ".join(non_usd))


# --------------------------------------------------------------------------- #
# 3. actions / 4. sp500 -- market-wide, resumed on the table's global max date  #
# --------------------------------------------------------------------------- #
def _fetch_dated_table(context: Context, table: Table, endpoint: str, since: str, *, full: bool = False) -> None:
    """Shared body for the two market-wide, date-resumed side tables."""
    frame = sharadar_get(context, endpoint, keep_default_na=False, sort="date.asc", **cast(dict[str, Any], {"date.gte": since}))
    if frame is None:
        context.log.warning("Sharadar %s: request failed; %s left unchanged", endpoint, table)
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
    # market-wide, so ticker_count=0
    record_run(context, table, 0, written, is_full_rescan=full)


def fetch_sharadar_actions(context: Context, *, years_history: int, full: bool = False) -> None:
    """Corporate actions -> `sharadar_actions`, market-wide, resumed from the table's global max date."""
    since = _since(context.store.max_date(Tables.sharadar_actions), years_history, full)
    _fetch_dated_table(context, Tables.sharadar_actions, "actions", since, full=full)


def fetch_sharadar_sp500(context: Context, *, full: bool = False) -> None:
    """S&P 500 membership events -> `sharadar_sp500`; cold or `full` pulls from `SHARADAR_SP500_FIRST_DATE`."""
    stored_max = context.store.max_date(Tables.sharadar_sp500)
    since = SHARADAR_SP500_FIRST_DATE if full or stored_max is None else (stored_max + pd.Timedelta(days=1)).strftime(DATE_FORMAT)
    _fetch_dated_table(context, Tables.sharadar_sp500, "sp500", since, full=full)
