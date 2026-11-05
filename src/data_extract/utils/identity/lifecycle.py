"""Lifecycle stage of the identity build: each universe company's first price date, its one end date and its status.

`ended_on` is the last day the company's shares traded (D-4): the `lastpricedate` of its Sharadar permaticker once
Sharadar marks it delisted, corroborated by the last stored price (A-3: more than 5 sessions apart is an action item).
A ticker change is not an end (D-6): the company keeps its key and trades under `trading_symbol` (`renamed`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data_extract.utils.identity.symbol_history import trading_symbols
from src.utils.identity_flags import FLAG_COLUMNS

logger = logging.getLogger(__name__)

#: The `sharadar_tickers` columns the stage reads, and the stored price span of each ticker (`store.summarize`).
LIFE_COLUMNS = ("permaticker", "ticker", "firstpricedate", "lastpricedate", "isdelisted", "lastupdated")
PRICE_SPAN = {"price_first": ("min", "date"), "price_last": ("max", "date")}
#: One row per universe ticker: the identity columns of `universe_tickers`.
TABLE_COLUMNS = ("ticker", "company_id", "trading_symbol", "first_price_date", "ended_on", "status")
STATUS_ACTIVE, STATUS_RENAMED, STATUS_ENDED = "active", "renamed", "ended"
#: A-3: Sharadar's end and the last stored price further apart than this many sessions is an action item.
MAX_SESSIONS_APART = 5
_LIFE = ("permaticker", "sharadar_first", "sharadar_last", "delisted")


@dataclass(frozen=True)
class Lifecycle:
    """The identity columns of each universe ticker (`TABLE_COLUMNS`) and the end-date items (`FLAG_COLUMNS`)."""

    rows: pd.DataFrame
    items: pd.DataFrame


def universe_permatickers(universe: pd.DataFrame) -> list[int]:
    """The Sharadar permatickers of the universe rows (none for a row without one)."""
    values = pd.to_numeric(universe.reindex(columns=["permaticker"])["permaticker"], errors="coerce").dropna()
    return sorted({int(value) for value in values})


def ended_dates(universe: pd.DataFrame, sharadar: pd.DataFrame | None) -> dict[str, pd.Timestamp]:
    """`{universe ticker: ended_on}` of the ended companies (D-4), read before the stamp passes so the CIK, symbol and
    CUSIP stages close the company's open rows at that end (REQ-006)."""
    # each universe ticker's permaticker with its Sharadar dates
    frame = universe.reindex(columns=["ticker", "permaticker"]).assign(
        permaticker=lambda f: pd.to_numeric(f["permaticker"], errors="coerce").astype("Int64")
    )
    frame = frame.merge(_sharadar_life(sharadar), on="permaticker", how="left")

    # the ended ones
    ended = _end_of(frame)
    return {str(ticker): pd.Timestamp(end) for ticker, end in zip(frame["ticker"], ended, strict=True) if pd.notna(end)}


def company_lifecycle(universe: pd.DataFrame, sharadar: pd.DataFrame | None, prices: pd.DataFrame | None, lineage: pd.DataFrame) -> Lifecycle:
    """First price date, end date, status, trading symbol and company id of each universe ticker, with the end-date items."""
    # clean: each permaticker's Sharadar dates, each ticker's stored price span, company id and trading symbol
    life, spans = _sharadar_life(sharadar), _price_spans(prices)
    ids, trading = _company_ids(lineage), trading_symbols(lineage)[["ticker", "trading_symbol"]]

    # merge: the universe tickers with them
    frame = universe.reindex(columns=["ticker", "permaticker"]).assign(
        permaticker=lambda f: pd.to_numeric(f["permaticker"], errors="coerce").astype("Int64")
    )
    frame = frame.merge(ids, on="ticker", how="left").merge(trading, on="ticker", how="left")
    frame = frame.merge(life, on="permaticker", how="left").merge(spans, on="ticker", how="left")

    # compute the identity columns, then check Sharadar's end against the last stored price
    rows = _lifecycle_rows(frame.assign(ended_on=_end_of(frame)))
    return Lifecycle(rows=rows, items=_end_disputes(frame, rows))


def log_lifecycle(lifecycle: Lifecycle) -> None:
    """One INFO line: tickers per status, the renamed and the ended ones."""
    rows = lifecycle.rows
    renamed = rows[rows["status"].eq(STATUS_RENAMED)]
    ended = rows[rows["status"].eq(STATUS_ENDED)]
    logger.info(
        "lifecycle: %d universe ticker(s): %s; renamed %s; ended %s; %d end-date item(s)",
        len(rows),
        ", ".join(f"{count} {status}" for status, count in rows["status"].value_counts().sort_index().items()),
        ", ".join(f"{t} -> {s}" for t, s in zip(renamed["ticker"], renamed["trading_symbol"], strict=True)) or "none",
        ", ".join(f"{t} {d.date()}" for t, d in zip(ended["ticker"], ended["ended_on"], strict=True)) or "none",
        len(lifecycle.items),
    )


# Flat helpers
def _sharadar_life(sharadar: pd.DataFrame | None) -> pd.DataFrame:
    """Per permaticker: the earliest `firstpricedate`, and the `lastpricedate` and delisted flag of its newest row."""
    if sharadar is None or sharadar.empty:
        return pd.DataFrame({column: pd.Series(dtype="Int64" if column == "permaticker" else object) for column in _LIFE})
    rows = sharadar.assign(
        permaticker=pd.to_numeric(sharadar["permaticker"], errors="coerce").astype("Int64"),
        first=pd.to_datetime(sharadar["firstpricedate"]),
        last=pd.to_datetime(sharadar["lastpricedate"]),
        updated=pd.to_datetime(sharadar["lastupdated"]),
    )
    newest = rows.sort_values(["permaticker", "last", "updated"], kind="mergesort", na_position="first").groupby("permaticker").tail(1)
    first = rows.groupby("permaticker", as_index=False)["first"].min().rename(columns={"first": "sharadar_first"})
    out = newest[["permaticker", "last", "isdelisted"]].rename(columns={"last": "sharadar_last", "isdelisted": "delisted"})
    return first.merge(out, on="permaticker", how="left")[list(_LIFE)]


def _price_spans(prices: pd.DataFrame | None) -> pd.DataFrame:
    """`[ticker, price_first, price_last]`, dates parsed (empty without stored prices)."""
    if prices is None or prices.empty:
        return pd.DataFrame(
            {"ticker": pd.Series(dtype=object), "price_first": pd.Series(dtype="datetime64[ns]"), "price_last": pd.Series(dtype="datetime64[ns]")}
        )
    return pd.DataFrame(
        {
            "ticker": prices["ticker"].astype(str),
            "price_first": pd.to_datetime(prices["price_first"]),
            "price_last": pd.to_datetime(prices["price_last"]),
        }
    )


def _company_ids(lineage: pd.DataFrame) -> pd.DataFrame:
    """`[ticker, company_id]` of each universe ticker in the lineage."""
    rows = lineage[lineage["canonical_ticker"].notna()]
    return rows[["canonical_ticker", "company_id"]].drop_duplicates("canonical_ticker").rename(columns={"canonical_ticker": "ticker"})


def _lifecycle_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """The identity columns: first price = the earlier of Sharadar's and the stored one; the end only once delisted."""
    first = pd.concat([pd.to_datetime(frame["sharadar_first"]), frame["price_first"]], axis=1).min(axis=1)
    ended = frame["ended_on"]
    trading = frame["trading_symbol"].fillna(frame["ticker"]).astype(str)
    status = np.select([ended.notna().to_numpy(), trading.ne(frame["ticker"]).to_numpy()], [STATUS_ENDED, STATUS_RENAMED], STATUS_ACTIVE)
    company = frame["company_id"].astype(object).where(frame["company_id"].notna(), None)
    values = (frame["ticker"].astype(str), company, trading, first, ended, pd.Series(status, index=frame.index))
    return pd.DataFrame(dict(zip(TABLE_COLUMNS, values, strict=True))).reset_index(drop=True)


def _end_of(frame: pd.DataFrame) -> pd.Series:
    """The end date of each row: Sharadar's `lastpricedate` of the newest row, only once it is delisted (D-4)."""
    return pd.to_datetime(frame["sharadar_last"]).where(frame["delisted"].eq("Y"))


def _end_disputes(frame: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """One action item per ended ticker whose last stored price is more than `MAX_SESSIONS_APART` sessions from its end."""
    ended = rows["ended_on"].reset_index(drop=True)
    last = frame["price_last"].reset_index(drop=True)
    both = ended.notna() & last.notna()
    low = np.minimum(ended[both], last[both]).to_numpy().astype("datetime64[D]")
    high = np.maximum(ended[both], last[both]).to_numpy().astype("datetime64[D]")
    sessions = pd.Series(np.busday_count(low, high), index=ended[both].index)
    far = sessions[sessions > MAX_SESSIONS_APART].index
    suggested = "check the last trading day (Sharadar lastpricedate) against the stored prices; refresh the stale source"
    items = [
        (
            "end_date_disputed",
            True,
            str(rows.at[i, "ticker"]),
            "",
            f"Sharadar lastpricedate {ended[i].date()} (delisted), last stored price {last[i].date()}: {sessions[i]} session(s) apart",
            suggested,
            "",
        )
        for i in far
    ]
    return pd.DataFrame(items, columns=list(FLAG_COLUMNS))
