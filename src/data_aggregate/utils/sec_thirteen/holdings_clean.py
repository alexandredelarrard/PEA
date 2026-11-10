"""Shared 13F date normalization and per-batch company tagging and value repair."""

from __future__ import annotations

import logging
from collections.abc import Generator, Mapping, Sequence
from contextlib import closing
from typing import Any

import numpy as np
import pandas as pd

from src.data_aggregate.utils.sec_thirteen.value_basis import log_register, prepare_public_value_quotes, repair_value_basis
from src.utils.identity_tables import CusipTags, tag_company_lines

logger = logging.getLogger(__name__)


#: Coerced to a midnight `datetime64` on both manifests and positions.
_DATES = ("period", "filing_date")


def clean_filing_dates(df_filings: pd.DataFrame, df_holdings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize source dates without changing accession grain or missing quantities."""
    for df in (df_filings, df_holdings):
        for column in _DATES:
            df[column] = pd.to_datetime(df[column])
    return df_filings, df_holdings


def clean_public_batches(
    packets: Generator[tuple[pd.DataFrame, pd.DataFrame]],
    tags: CusipTags,
    universe: Sequence[str] | None,
    valuation_close: pd.DataFrame | None,
    value_basis: Mapping[str, Any] | None,
) -> Generator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Prepare shared quotes once, then per batch tag each filed row and repair raw accession values before
    reconciliation, retaining unsupported rows. `tag_company_lines` tags every row (the stored ticker is not read),
    one row per stored row so the resolver's count checks hold: a company's share classes in canonical-class shares
    (x the dated ratio, before the value repair) with its canonical CUSIP in `COMPANY_CUSIP`. No ticker where nothing
    tags a row, nor outside `universe` when given; unmapped and off-universe positions stay in the book."""
    period_quotes = prepare_public_value_quotes(valuation_close, value_basis)
    members = None if universe is None else set(universe)
    with closing(packets):
        for df_filings, df_raw in packets:
            # Tags and canonical-class shares, then one vector repair per batch; an empty batch passes through unrepaired.
            df_tagged = tag_company_lines(df_raw, tags.lines, tags.cmap)
            tickers = df_tagged["ticker"].astype(object)
            keep = tickers.notna() if members is None else tickers.isin(members)
            df_holdings, df_register = repair_value_basis(
                df_tagged.assign(ticker=tickers.where(keep, None)), valuation_close, value_basis, period_quotes=period_quotes
            )
            del df_raw, df_tagged

            # Report the unit decisions, then null the excluded amounts; the rows stay in the book.
            log_register(df_register, float(df_holdings.value_usd.sum()))
            excluded = df_holdings.get("value_basis_excluded", pd.Series(False, index=df_holdings.index))
            df_holdings.loc[excluded, ["shares", "value_usd"]] = np.nan
            yield df_filings, df_holdings
            del df_filings, df_holdings
