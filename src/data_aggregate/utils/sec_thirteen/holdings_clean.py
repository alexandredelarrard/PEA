"""Shared 13F date normalization, stored-ticker attachment and per-batch value repair."""

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


#: Coerced to a midnight `datetime64` when present; `filing_date` is optional on `sec13f_hr`.
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


def _null_excluded_amounts(df_holdings: pd.DataFrame) -> pd.DataFrame:
    """Null the shares and dollars of rows the value repair excluded; the rows stay in the book."""
    if "value_basis_excluded" in df_holdings:
        df_holdings.loc[df_holdings.value_basis_excluded, ["shares", "value_usd"]] = np.nan
    return df_holdings


def clean_public_batches(
    packets: Generator[tuple[pd.DataFrame, pd.DataFrame]],
    universe: Sequence[str] | None,
    valuation_close: pd.DataFrame | None,
    value_basis: Mapping[str, Any] | None,
) -> Generator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Prepare shared quotes once, then per batch attach stored tickers and repair raw accession values before
    reconciliation, retaining unsupported rows."""
    period_quotes = prepare_public_value_quotes(valuation_close, value_basis)
    with closing(packets):
        for df_filings, df_raw in packets:
            # Stored tickers, then one vector repair per non-empty batch; an empty batch passes through.
            df_holdings = attach_tickers(df_raw, universe)
            del df_raw
            df_holdings, df_register = (
                (df_holdings, pd.DataFrame())
                if df_holdings.empty
                else repair_value_basis(df_holdings, valuation_close, value_basis, period_quotes=period_quotes)
            )

            # Report the unit decisions, then drop the excluded amounts.
            log_register(df_register, float(df_holdings.value_usd.sum()) if not df_register.empty else 0.0)
            df_holdings = _null_excluded_amounts(df_holdings)
            yield df_filings, df_holdings
            del df_filings, df_holdings
