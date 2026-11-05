"""Shared 13F normalization, dated CUSIP attribution, publication preparation and value repair."""

from __future__ import annotations

import logging
from collections.abc import Generator, Mapping, Sequence
from contextlib import closing
from typing import Any

import numpy as np
import pandas as pd

from src.constants.constants import SEC_13F_FILING_LAG_DAYS
from src.data_aggregate.utils.common.class_lines import dated_tickers
from src.data_aggregate.utils.common.data_utils import to_day
from src.data_aggregate.utils.sec_thirteen.value_basis import log_register, prepare_public_value_quotes, repair_value_basis

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


def attach_tickers(holdings: pd.DataFrame, windows: pd.DataFrame | None, universe: Sequence[str] | None = None) -> pd.DataFrame:
    """Attach the dated canonical ticker of each CUSIP (`cusip_windows`), retaining unmapped whole-book positions."""
    if windows is None or windows.empty:
        return holdings.assign(ticker=pd.NA)
    tickers = dated_tickers(holdings["cusip"], holdings["period"], windows)
    return holdings.assign(ticker=tickers if universe is None else tickers.where(tickers.isin(set(universe))))


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
    df_holdings, df_register = repair_value_basis(df_holdings, valuation_close, period_quotes=period_quotes, **dict(value_basis or {}))
    log_register(df_register, float(df_holdings.value_usd.sum()))
    if "value_basis_excluded" in df_holdings:
        df_holdings.loc[df_holdings.value_basis_excluded, ["shares", "value_usd"]] = np.nan
    return df_holdings


def clean_public_batches(
    packets: Generator[tuple[pd.DataFrame, pd.DataFrame]],
    windows: pd.DataFrame | None,
    universe: Sequence[str] | None,
    valuation_close: pd.DataFrame | None,
    value_basis: Mapping[str, Any] | None,
) -> Generator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Prepare shared quotes once and attach tickers and repair all completed books per batch."""
    period_quotes = prepare_public_value_quotes(valuation_close, value_basis)
    with closing(packets):
        for df_filings, df_raw in packets:
            df_holdings = clean_public_holdings(attach_tickers(df_raw, windows, universe), valuation_close, value_basis, period_quotes=period_quotes)
            del df_raw
            yield df_filings, df_holdings
            del df_filings, df_holdings


def _apply_filing_band(h: pd.DataFrame, band: tuple[int, int]) -> pd.DataFrame:
    """Keep legacy rows within the declared deadline band, including unknown dates."""
    early, late = band
    if "filing_date" not in h.columns or "period" not in h.columns:
        logger.info("13F clean: no `filing_date`/`period` -> the [-%s, +%s] filing band CANNOT be applied on this read; every row kept", early, late)
        return h

    lag = (h["filing_date"] - (h["period"] + pd.Timedelta(days=SEC_13F_FILING_LAG_DAYS))).dt.days
    unknown = lag.isna()
    drop = ~lag.between(-early, late) & ~unknown
    if not drop.any():
        return h

    value = pd.to_numeric(h["value_usd"], errors="coerce") if "value_usd" in h.columns else None
    total = float(value.sum()) if value is not None else 0.0
    logger.info(
        "13F filing band [-%sd, +%sd] vs deadline: %s of %s row(s) dropped (%.3f%%) "
        "across %s filer(s), carrying %.3f%% of as-filed value (PRE unit-repair, so "
        "that share is only indicative); %s early, %s late, %s row(s) undated and kept",
        early,
        late,
        f"{int(drop.sum()):,}",
        f"{len(h):,}",
        100.0 * float(drop.mean()),
        f"{h.loc[drop, 'cik'].nunique():,}" if "cik" in h.columns else "?",
        100.0 * float(value[drop].sum()) / total if (value is not None and total > 0) else 0.0,
        f"{int((lag < -early).sum()):,}",
        f"{int((lag > late).sum()):,}",
        f"{int(unknown.sum()):,}",
    )
    return h[~drop]


def clean_holdings(
    holdings: pd.DataFrame,
    *,
    key: Sequence[str],
    filing_band: tuple[int, int] | None = None,
) -> pd.DataFrame:
    """Normalize legacy keys, then apply the filing band and amendment order.

    This latest-only adapter does not certify immutable filing completeness."""
    h = holdings.copy()
    for col in _DATES:
        if col in h.columns:
            h[col] = to_day(h[col])
    for col in ("shares", "value_usd", "call_value", "put_value"):
        h[col] = pd.to_numeric(h[col], errors="coerce") if col in h.columns else pd.Series(np.nan if col == "value_usd" else 0.0, index=h.index)
        if col == "shares":
            h[col] = h[col].fillna(0.0)

    h = h.dropna(subset=list(key))
    if filing_band is not None:
        h = _apply_filing_band(h, filing_band)

    if "filing_date" in h.columns:
        h = h.sort_values("filing_date")
    else:
        logger.info(
            "13F clean: no `filing_date` column -> amendments cannot be ordered, so `keep='last'` falls back to source order for %s duplicate key(s)",
            f"{int(h.duplicated(list(key)).sum()):,}",
        )
    return h.drop_duplicates(list(key), keep="last")
