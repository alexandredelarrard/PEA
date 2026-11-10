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


#: Coerced to a midnight `datetime64` on both manifests and positions.
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
            # Stored tickers, then one vector repair per batch; an empty batch passes through unrepaired.
            df_holdings, df_register = repair_value_basis(attach_tickers(df_raw, universe), valuation_close, value_basis, period_quotes=period_quotes)
            del df_raw

            # Report the unit decisions, then null the excluded amounts; the rows stay in the book.
            log_register(df_register, float(df_holdings.value_usd.sum()))
            excluded = df_holdings.get("value_basis_excluded", pd.Series(False, index=df_holdings.index))
            df_holdings.loc[excluded, ["shares", "value_usd"]] = np.nan
            yield df_filings, df_holdings
            del df_filings, df_holdings
