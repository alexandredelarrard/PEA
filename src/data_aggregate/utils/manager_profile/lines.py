"""One report quarter of 13F books reduced to the original accession's cleaned, weighted lines."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.data_aggregate.utils.sec_thirteen.value_basis import consensus_prices, repair_value_basis
from src.data_store.schema import EMPTY_MARKER

logger = logging.getLogger(__name__)

#: Columns of `sec13f_hr` the profile reads (markers carry `form`).
LINE_COLUMNS = (
    "cik",
    "period",
    "filing_date",
    "accession_number",
    "ticker",
    "cusip",
    "issuer_name",
    "title_of_class",
    "position_type",
    "shares",
    "value_usd",
    "call_value",
    "put_value",
    "debt_value",
    "form",
)
#: Line kind -> the column holding its value (common lines never add `call_value` / `put_value`).
_VALUE_COLUMN = {"common": "value_usd", "debt": "debt_value", "call": "call_value", "put": "put_value"}
_BOOK_KEYS = ["cik", "period", "filing_date", "accession_number"]


def _original_book(raw: pd.DataFrame) -> pd.DataFrame:
    """Lines of the original 13F-HR accession per (cik, period), the earliest accession when no original marker exists."""
    # Normalise the dates and split markers from lines.
    df = raw.assign(period=pd.to_datetime(raw["period"]), filing_date=pd.to_datetime(raw["filing_date"]))
    is_marker = df["cusip"].eq(EMPTY_MARKER)
    markers, lines = df[is_marker], df[~is_marker]

    # Original marker first; filers with none fall back to their earliest accession.
    order = ["filing_date", "accession_number"]
    original = markers[markers["form"].eq("13F-HR")].sort_values(order).drop_duplicates(["cik", "period"])
    fallback = lines.sort_values(order).drop_duplicates(["cik", "period"])
    chosen = pd.concat([original, fallback]).drop_duplicates(["cik", "period"])[["cik", "period", "accession_number"]]

    # Keep the chosen accession's lines, dropping filings dated before their period end.
    book = lines.merge(chosen, on=["cik", "period", "accession_number"])
    book = book[book["filing_date"] >= book["period"]].copy()
    book["position_type"] = book["position_type"].fillna("common").str.lower()
    return book


def _kind_values(book: pd.DataFrame) -> pd.Series:
    """Each line's value, from the column of its kind (common lines never add `call_value` / `put_value`)."""
    value = pd.Series(np.nan, index=book.index)
    for name, column in _VALUE_COLUMN.items():
        value = value.where(book["position_type"] != name, book[column])
    return value


def _book_scale(book: pd.DataFrame, register: pd.DataFrame) -> pd.Series:
    """Each line's book unit factor from the repair register (1 for a book without one)."""
    factor = book[_BOOK_KEYS].merge(register[[*_BOOK_KEYS, "factor"]], on=_BOOK_KEYS, how="left", validate="many_to_one")["factor"]
    return pd.Series(factor.to_numpy(), index=book.index).fillna(1.0)


def _oversized_revaluations(book: pd.DataFrame, repaired: pd.DataFrame, value: pd.Series, scale: pd.Series) -> pd.Series:
    """Equity lines re-valued upward in a book whose re-valued lines together exceed its whole filed total (book's repaired unit).

    `value` and `scale` are the book's per-line filed values and unit factors. A line re-valued from shares x consensus price that had
    a filed value points to a wrong share count when the re-valued lines are worth more than everything the filer reported; lines the
    repair lowered or kept are left alone.
    """
    # The book's filed total in the repaired unit.
    filed = value.where(value.gt(0), 0.0)
    total = filed.groupby([book[column] for column in _BOOK_KEYS]).transform("sum") * scale

    # Re-valued amount per book against that total.
    revalued = repaired["value_basis_value_rebuilt"] & ~repaired["value_basis_missing_value"]
    amount = repaired["value_usd"].where(revalued, 0.0).groupby([book.loc[repaired.index, column] for column in _BOOK_KEYS]).transform("sum")
    return (
        revalued & amount.gt(total.loc[repaired.index]) & repaired["value_usd"].gt(book.loc[repaired.index, "value_usd"] * scale.loc[repaired.index])
    )


def _apply_units(book: pd.DataFrame, repaired: pd.DataFrame, scale: pd.Series) -> pd.DataFrame:
    """Equity lines take the repaired value and shares; debt and option values take their book's unit factor `scale`."""
    out = book.copy()
    out.loc[repaired.index, ["value_usd", "shares"]] = repaired[["value_usd", "shares"]]
    for column in ("debt_value", "call_value", "put_value"):
        out[column] = out[column] * scale
    return out


def _clean_lines(book: pd.DataFrame, value: pd.Series, prefixes: pd.Series) -> pd.DataFrame:
    """Keep positive values, take a missing ticker from the 8-character CUSIP prefix map, sum duplicates per (cik, kind, ticker or cusip) and weight by book total."""
    # One value per line, from the column of its kind.
    kind = book["position_type"]
    ticker = book["ticker"].mask(book["ticker"].isin(["", EMPTY_MARKER]))
    ticker = ticker.fillna(book["cusip"].str[:8].map(prefixes))
    lines = book.assign(kind=kind, value=value, ticker=ticker, key=ticker.fillna(book["cusip"]))
    lines = lines[lines["kind"].isin(_VALUE_COLUMN) & lines["value"].gt(0)]

    # Sum duplicate lines; shares stay unknown when no row of the position reports them.
    keys = ["cik", "period", "kind", "key"]
    grouped = lines.groupby(keys, sort=False)
    out = grouped.agg(
        ticker=("ticker", "first"),
        cusip=("cusip", "first"),
        issuer_name=("issuer_name", "first"),
        title_of_class=("title_of_class", "first"),
        filing_date=("filing_date", "first"),
        value=("value", "sum"),
    )
    out = out.assign(shares=grouped["shares"].sum(min_count=1)).reset_index()
    out["weight"] = out["value"] / out.groupby(["cik", "period"])["value"].transform("sum")
    return out[[*keys, "ticker", "cusip", "issuer_name", "title_of_class", "filing_date", "shares", "value", "weight"]]


def clean_book(raw: pd.DataFrame, cfg: DictConfig, prefixes: pd.Series) -> pd.DataFrame:
    """Cleaned lines (kind, key, ticker, cusip, issuer_name, title_of_class, filing_date, shares, value, weight) of every manager's original accession in `raw`.

    `prefixes` (`reference.prefix_tickers`) names the ticker of a CUSIP carried without one. Equity values take the cube's two-pass
    book-unit repair; lines it re-values above their whole filed book (a garbage `shares` column) are dropped.
    """
    # Original accession lines only.
    book = _original_book(raw)

    # Book-unit repair of the equity lines: consensus, repair, consensus on the repaired lines, repair.
    equity = book[book["position_type"].eq("common")]
    min_filers = int(cfg.price_check_min_filers)
    once, _ = repair_value_basis(equity, None, consensus=consensus_prices(equity, min_filers))
    repaired, register = repair_value_basis(equity, None, consensus=consensus_prices(once, min_filers))
    scale = _book_scale(book, register)
    oversized = _oversized_revaluations(book, repaired, _kind_values(book), scale)
    repaired.loc[oversized, "value_usd"] = np.nan
    nulled = int((equity["value_usd"].gt(0) & repaired["value_usd"].isna()).sum())
    logger.info(
        "13F profile lines: %d books, unit moved in %d, abstained in %d, %d equity lines nulled (%d re-valued above their whole book)",
        len(register),
        int(register["factor"].ne(1.0).sum()),
        int(register["status"].eq("abstained").sum()),
        nulled,
        int(oversized.sum()),
    )

    # Positive values, one row per position, weights over the book total.
    applied = _apply_units(book, repaired, scale)
    return _clean_lines(applied, _kind_values(applied), prefixes)
