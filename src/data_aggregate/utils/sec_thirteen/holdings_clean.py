"""Shared 13F normalization, stored-ticker attachment, publication preparation and value repair."""

from __future__ import annotations

import logging
from collections.abc import Generator, Mapping, Sequence
from contextlib import closing
from typing import Any, cast

import numpy as np
import pandas as pd

from src.data_aggregate.utils.sec_thirteen.value_basis import log_register, prepare_public_value_quotes, repair_value_basis
from src.data_store.schema import Tables

logger = logging.getLogger(__name__)


#: Coerced to a midnight `datetime64` when present. `filing_date` is OPTIONAL on both tables
#: (it is in `sec13f_hr.optional_columns`), which is why this is a membership test and not a
#: bare list comprehension -- reading `h["filing_date"]` on a frame that lacks it raises
#: `KeyError` before the assignment can create it.
_DATES = ("period", "filing_date")


def clean_filing_dates(df_filings: pd.DataFrame, df_holdings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize source dates without changing accession grain or missing quantities."""
    for df in (df_filings, df_holdings):
        for column in _DATES:
            df[column] = pd.to_datetime(df[column])
    return df_filings, df_holdings


def attach_tickers(holdings: pd.DataFrame, universe: Sequence[str] | None = None) -> pd.DataFrame:
    """Each position's ticker as stored at extraction (U-67); none on the unmapped sentinel, nor outside `universe`
    when given. Unmapped and off-universe positions stay in the book."""
    stored = holdings["ticker"].astype(object)
    keep = stored.ne(cast(tuple[str, str], Tables.sec13f_hr.empty_marker)[1]) if universe is None else stored.isin(set(universe))
    return holdings.assign(ticker=stored.where(keep, None))


def clean_public_holdings(
    df_holdings: pd.DataFrame,
    valuation_close: pd.DataFrame | None,
    value_basis: Mapping[str, Any] | None,
    *,
    period_quotes: pd.Series | None = None,
) -> pd.DataFrame:
    """Repair raw accession values before reconciliation, retaining unsupported rows."""
    if df_holdings.empty:
        return df_holdings
    df_holdings, df_register = repair_value_basis(df_holdings, valuation_close, value_basis, period_quotes=period_quotes)
    log_register(df_register, float(df_holdings.value_usd.sum()))
    if "value_basis_excluded" in df_holdings:
        df_holdings.loc[df_holdings.value_basis_excluded, ["shares", "value_usd"]] = np.nan
    return df_holdings


def clean_public_batches(
    packets: Generator[tuple[pd.DataFrame, pd.DataFrame]],
    universe: Sequence[str] | None,
    valuation_close: pd.DataFrame | None,
    value_basis: Mapping[str, Any] | None,
) -> Generator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Prepare shared quotes once, attach stored tickers and repair all completed books per batch."""
    period_quotes = prepare_public_value_quotes(valuation_close, value_basis)
    with closing(packets):
        for df_filings, df_raw in packets:
            df_holdings = attach_tickers(df_raw, universe)
            df_holdings = clean_public_holdings(df_holdings, valuation_close, value_basis, period_quotes=period_quotes)
            del df_raw
            yield df_filings, df_holdings
            del df_filings, df_holdings
