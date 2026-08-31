"""
fetch_sharadar.py  (src/data_extract/utils/fundamentals_sharadar/fetch_sharadar.py)
------------------------------------------------------------------------------------
The four Sharadar fetchers, all resumable from the DB:

  * `fetch_sharadar_tickers`      -- the entity dimension. MUST RUN FIRST: the fundamentals
                                     fetch reads `currency` from it to enforce the USD
                                     assertion, and refuses to run if it is empty.
  * `fetch_sharadar_fundamentals` -- SF1, all 112 columns, one request per (ticker, dimension).
  * `fetch_sharadar_actions`      -- dividends, splits, spinoffs, acquisitions, relations.
  * `fetch_sharadar_sp500`        -- index membership events.

SINGLE-THREADED, deliberately. Sharadar documents no rate limit anywhere (their whole doc set
has zero hits for rate limit / throttle / 429 / concurrent), so there is no measured budget to
spend; and `store.ensure_table` is a check-then-create with no lock, so threaded writers on a
COLD table race the CREATE and the losers silently lose rows. If parallelism is ever added,
the first write per table must be serialised with a `threading.Lock` + a `created` set exactly
as `data_extract/utils/common/edgar_driver.py` already does.

RESUME (D13) is per TICKER, not per (ticker, dimension), and that is safe rather than merely
convenient: `date` is the FILING date, and Sharadar publishes the ARY row for a fiscal year on
the 10-K filing date, which is always later than the latest 10-Q. So a ticker-wide watermark
can never sit past an ARY row that has not been fetched. Measured 2026-08-26 -- AAPL
ARY max 2025-10-31 vs ticker-wide 2026-07-31; JPM 2026-02-13 vs 2026-08-06; HD 2026-03-18 vs
2026-05-27. `lastupdated` is deliberately NOT used as a watermark: it is a per-ticker
REPROCESSING stamp, not a per-row change stamp (AAPL 2026-07-31 vs GS 2026-08-04), so a
Sharadar restatement is picked up by `-F/--full`, not by an incremental run.
"""
from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from src.constants.constants import (
    DATE_FORMAT, SHARADAR_BASE_URL, SHARADAR_DIMENSIONS, SHARADAR_SF1_COLUMNS,
    SHARADAR_SP500_FIRST_DATE,
)
from src.context import Context
from src.data_extract.utils.common.run_manifest import record_run
from src.data_store.schema import Table, Tables
from src.data_extract.utils.fundamentals_sharadar.client import (
    NotEntitled, cast_value_columns, coerce_date_columns, sharadar_get,
)
from src.utils.polite_http import sleep_pace


def _pace(context: Context) -> float:
    return float(context.config.data_extract.sharadar_request_pace)


def _cold_start(years_history: int) -> pd.Timestamp:
    return pd.Timestamp.today().normalize() - pd.DateOffset(years=int(years_history))


def _since(stored_max: pd.Timestamp | None, years_history: int, full: bool) -> str:
    """The `date.gte` for one entity: the day after what is stored, or the configured window
    on a cold table / a `--full` run. Always explicit -- the API defaults `from` to
    "1 year ago", so an omitted bound silently truncates history."""
    if full or stored_max is None:
        return _cold_start(years_history).strftime(DATE_FORMAT)
    return (stored_max + pd.Timedelta(days=1)).strftime(DATE_FORMAT)


def _usd_roster(context: Context) -> dict[str, str]:
    """`{ticker: currency}` from `sharadar_tickers`, preferring the live (not delisted) row.

    Raises if the dimension is empty: the USD assertion (D20) cannot be enforced without it,
    and writing unasserted rows is exactly the failure this guard exists to prevent.
    """
    frame = context.store.load(Tables.sharadar_tickers,
                               columns=["ticker", "currency", "isdelisted"])
    if frame is None or frame.empty:
        raise RuntimeError(
            f"{Tables.sharadar_tickers} is empty -- run the tickers fetch first. The "
            f"fundamentals fetch reads `currency` from it to assert USD (D20), and only 8 "
            f"of Sharadar's money columns are USD-converted, so a non-USD row mixes units "
            f"INSIDE ITSELF.")
    frame = frame.assign(_live=(frame["isdelisted"].astype(str).str.upper() != "Y"))
    frame = frame.sort_values("_live", ascending=False).drop_duplicates("ticker")
    return dict(zip(frame["ticker"].astype(str), frame["currency"].astype(str)))


# --------------------------------------------------------------------------- #
# 1. tickers -- the entity dimension (must run first)                          #
# --------------------------------------------------------------------------- #
def fetch_sharadar_tickers(context: Context) -> None:
    """Full refresh of `sharadar_tickers`, filtered to the rows describing SF1 coverage.

    A full refresh rather than an incremental one: there is no date column to resume on, the
    whole dimension is ~17.8k rows, and `isdelisted` / `lastquarter` MUTATE, so an
    append-only view of it would go stale silently.
    """
    frame = sharadar_get(context, "tickers", keep_default_na=False,
                         **{"table": "fundamentals"})
    if frame is None or frame.empty:
        context.log.warning("Sharadar tickers: no rows returned; %s left unchanged",
                            Tables.sharadar_tickers)
        return
    frame = coerce_date_columns(frame, Tables.sharadar_tickers.date_type_cols)
    frame["permaticker"] = pd.to_numeric(frame["permaticker"], errors="coerce").astype("Int64")
    written = context.store.save(Tables.sharadar_tickers, frame)
    context.log.info("Sharadar tickers: %d rows -> %s (%d USD, %d non-USD)",
                     written, Tables.sharadar_tickers,
                     int((frame["currency"] == "USD").sum()),
                     int((frame["currency"] != "USD").sum()))
    # Market-wide (a full refresh of the whole dimension, not scoped to our universe) --
    # `ticker_count=0` records that rather than the ~17.8k SF1-covered rows.
    record_run(context, Tables.sharadar_tickers, 0, written, is_full_rescan=True)


# --------------------------------------------------------------------------- #
# 2. fundamentals (SF1)                                                        #
# --------------------------------------------------------------------------- #
def fetch_sharadar_fundamentals(context: Context, tickers: list[str], *,
                                years_history: int, full: bool = False) -> None:
    """SF1 for `tickers` x `SHARADAR_DIMENSIONS` -> `fundamentals_sharadar`.

    One request per (ticker, dimension). A ticker the subscription does not cover costs
    exactly ONE request and is counted, not retried -- see `client.NotEntitled`.
    """
    currencies = _usd_roster(context)
    resume = context.store.max_date_by(Tables.sharadar_fundamentals, "ticker")
    pace = _pace(context)
    context.log.info("Sharadar SF1: %d ticker(s) x %d dimension(s); %d already have stored "
                     "rows (full=%s, window=%dy)", len(tickers), len(SHARADAR_DIMENSIONS),
                     len(resume), full, years_history)

    entitled: list[str] = []
    denied: list[str] = []
    non_usd: list[str] = []
    total_rows = 0

    for ticker in tqdm(tickers, desc="Downloading tickers"):
        currency = currencies.get(ticker)
        if currency is not None and currency != "USD":
            # Refuse to write rather than write-and-flag: only 8 of the 112 columns are
            # USD-converted, so the row would mix units inside itself and no downstream
            # consumer could unmix them.
            non_usd.append(ticker)
            context.log.warning("Sharadar SF1: %s reports in %s, not USD -- NOT WRITTEN. "
                                "Only 8 SF1 columns are USD-converted, so the row would mix "
                                "units within itself (D20).", ticker, currency)
            continue

        since = _since(resume.get(ticker), years_history, full)
        frames: list[pd.DataFrame] = []
        try:
            for dimension in SHARADAR_DIMENSIONS:
                page = sharadar_get(
                    context, "fundamentals", expect_columns=SHARADAR_SF1_COLUMNS,
                    ticker=ticker, dimension=dimension, sort="date.asc",
                    **{"date.gte": since})
                if page is not None and not page.empty:
                    frames.append(page)
                sleep_pace(pace, SHARADAR_BASE_URL)
        except NotEntitled:
            denied.append(ticker)
            continue

        entitled.append(ticker)
        if not frames:
            context.log.debug("Sharadar SF1: %s up to date (since %s)", ticker, since)
            continue

        frame = pd.concat(frames, ignore_index=True)
        # Cast BEFORE the first write: `ensure_table` types the table off the first frame,
        # and an all-None object column would become TEXT for every later ticker.
        frame = cast_value_columns(frame)
        frame = coerce_date_columns(frame, Tables.sharadar_fundamentals.date_type_cols)
        total_rows += context.store.save(Tables.sharadar_fundamentals, frame)

    context.log.info("Sharadar SF1: %d entitled, %d not entitled (403); %d rows written to %s",
                     len(entitled), len(denied), total_rows, Tables.sharadar_fundamentals)
    record_run(context, Tables.sharadar_fundamentals, len(tickers), total_rows, is_full_rescan=full)
    if denied:
        context.log.info("Sharadar SF1: not entitled -> %s",
                         ", ".join(denied[:20]) + (" ..." if len(denied) > 20 else ""))
    if non_usd:
        context.log.warning("Sharadar SF1: %d non-USD filer(s) skipped -> %s",
                            len(non_usd), ", ".join(non_usd))


# --------------------------------------------------------------------------- #
# 3. actions / 4. sp500 -- market-wide, resumed on the table's global max date  #
# --------------------------------------------------------------------------- #
def _fetch_dated_table(context: Context, table: Table, endpoint: str,
                       since: str, *, full: bool = False) -> None:
    """Shared body for the two market-wide, date-resumed side tables."""
    frame = sharadar_get(context, endpoint, keep_default_na=False,
                         sort="date.asc", **{"date.gte": since})
    if frame is None:
        context.log.warning("Sharadar %s: request failed; %s left unchanged", endpoint, table)
        return
    if frame.empty:
        context.log.info("Sharadar %s: no rows since %s; %s already current",
                         endpoint, since, table)
        return
    frame = coerce_date_columns(frame, table.date_type_cols)
    if "value" in frame.columns:
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce").astype("float64")
    written = context.store.save(table, frame)
    context.log.info("Sharadar %s: %d row(s) since %s -> %s (actions: %s)",
                     endpoint, written, since, table,
                     frame["action"].value_counts().to_dict())
    # Market-wide (one request covers every ticker), so ticker_count=0.
    record_run(context, table, 0, written, is_full_rescan=full)


def fetch_sharadar_actions(context: Context, *, years_history: int,
                           full: bool = False) -> None:
    """Corporate actions -> `sharadar_actions`. Market-wide, so resume is the table's GLOBAL
    max date: one request covers every ticker, and a per-ticker frontier would only re-pull
    days already held."""
    since = _since(context.store.max_date(Tables.sharadar_actions), years_history, full)
    _fetch_dated_table(context, Tables.sharadar_actions, "actions", since, full=full)


def fetch_sharadar_sp500(context: Context, *, full: bool = False) -> None:
    """S&P 500 membership events -> `sharadar_sp500`.

    No `years_history`: membership history is only useful at FULL depth (the survivorship-bias
    fix needs 1992 onward), and the whole table is ~3.3k rows, so the cold pull is one request.
    """
    stored_max = context.store.max_date(Tables.sharadar_sp500)
    since = (SHARADAR_SP500_FIRST_DATE if full or stored_max is None
             else (stored_max + pd.Timedelta(days=1)).strftime(DATE_FORMAT))
    _fetch_dated_table(context, Tables.sharadar_sp500, "sp500", since, full=full)
