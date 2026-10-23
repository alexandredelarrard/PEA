"""Shared 13F normalization, CUSIP mapping, publication preparation and value repair."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
import pandas as pd

from src.constants.constants import F13_SETTLE_TRADING_DAYS, SEC_13F_FILING_LAG_DAYS
from src.data_aggregate.utils.common.availability import availability_date
from src.data_aggregate.utils.common.data_utils import to_day
from src.data_aggregate.utils.common.typing import frame_column
from src.data_aggregate.utils.sec_thirteen.value_basis import log_register, repair_value_basis
from src.utils.institutional_vintages import filing_public_dates
from src.utils.string import pad_cik

logger = logging.getLogger(__name__)


#: The DEFAULT numeric set: `sec13f_hr`'s four legs. Coerced to numeric and zero-filled when
#: present, CREATED as 0.0 when not -- `_quarter_features` sums `call_value` / `put_value`
#: unconditionally, and on that table a missing option leg means "no options reported", never
#: "unknown".
#:
#: ⚠ THE ELITE TABLE PASSES ITS OWN TWO, and that is not symmetry-breaking for its own sake.
#: `sec13f_manager_holdings` classifies positions with `position_type`, so the cleaner already
#: filters options out by row; manufacturing zero-valued `call_value` / `put_value` columns
#: there would add two all-zero columns to the conviction frame that nothing reads. Measured:
#: passing the 13F default unchanged widened `prim.super_conviction` from 13 to 15 columns.
_NUMERIC_13F = ("shares", "value_usd", "call_value", "put_value")

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


def prepare_publications(df_filings: pd.DataFrame, trading_index: pd.DatetimeIndex, settle_days: int) -> pd.DataFrame:
    """Prepare normalized public event dates without changing filing-version policy."""
    df_manifest = df_filings.copy()
    df_manifest["period"] = pd.to_datetime(df_manifest["period"])
    df_manifest["_day"] = filing_public_dates(df_manifest, trading_index, settle_days)
    return df_manifest.loc[df_manifest["_day"].notna()]


def legacy_manager_public_dates(state: pd.DataFrame) -> pd.Series:
    """Use the existing later of filing date and deadline for legacy manager states."""
    deadline = state["period"] + pd.Timedelta(days=SEC_13F_FILING_LAG_DAYS)
    filed = state.get("filing_date")
    if filed is None:
        return deadline
    return pd.concat([deadline, pd.to_datetime(filed)], axis=1).max(axis=1)


def stamp_legacy_publication(
    holdings: pd.DataFrame, trading_index: pd.DatetimeIndex, *, settle_trading_days: int = F13_SETTLE_TRADING_DAYS
) -> pd.DataFrame:
    """Retain legacy late revisions; remove periods not yet public on the calendar."""
    h = holdings.copy()
    h["first_pub"] = availability_date(frame_column(h, "period"), trading_index, settle_trading_days=settle_trading_days)
    if "filing_date" not in h.columns:
        logger.info(
            "13F availability: no `filing_date` column -> `as_of` is the period's "
            "availability date for every row; the cutoff CANNOT be applied on this read, so "
            "a late filing is inside the first publication",
        )
        h["as_of"] = h["first_pub"]
    else:
        filing_date = frame_column(h, "filing_date")
        first_pub = frame_column(h, "first_pub")
        later = filing_date.notna() & first_pub.notna() & (filing_date > first_pub)
        h["as_of"] = first_pub.where(~later, filing_date)
    unavailable = frame_column(h, "as_of").isna()
    if unavailable.any():
        # An unpublishable predecessor must not enter the next quarter's differences.
        logger.info(
            "13F availability: %s row(s) across period(s) %s have no availability date inside the trading calendar (it ends %s) -> not emitted",
            f"{int(unavailable.sum()):,}",
            sorted({str(pd.Timestamp(p).date()) for p in cast(pd.Series, h.loc[unavailable, "period"]).unique()})[:6],
            str(pd.Timestamp(cast(Any, pd.DatetimeIndex(trading_index).max())).date()) if len(trading_index) else "(empty)",
        )
        h = cast(pd.DataFrame, h.loc[~unavailable])
    return cast(pd.DataFrame, h)


def attach_tickers(holdings: pd.DataFrame, cusip_map: pd.DataFrame | None, universe: Sequence[str] | None = None) -> pd.DataFrame:
    """Map universe tickers while retaining unmapped whole-book positions."""
    df_holdings = holdings.copy()
    if cusip_map is None or cusip_map.empty:
        df_holdings["ticker"] = pd.NA
        return df_holdings
    df_map = cusip_map.dropna(subset=["cusip", "ticker"]).drop_duplicates("cusip")
    if universe is not None:
        df_map = df_map[df_map["ticker"].isin(set(universe))]
    return df_holdings.merge(df_map[["cusip", "ticker"]], on="cusip", how="left")


def clean_public_holdings(df_holdings: pd.DataFrame, valuation_close: pd.DataFrame | None, value_basis: Mapping[str, float] | None) -> pd.DataFrame:
    """Repair raw accession values before reconciliation, retaining unsupported rows."""
    if df_holdings.empty:
        return df_holdings
    df_holdings, df_register = repair_value_basis(df_holdings, valuation_close, **dict(value_basis or {}))
    log_register(df_register, float(df_holdings.value_usd.sum()))
    if "value_basis_excluded" in df_holdings:
        df_holdings.loc[df_holdings.value_basis_excluded, ["shares", "value_usd"]] = np.nan
    return df_holdings


def _repair_legacy_values(df_holdings: pd.DataFrame, valuation_close: pd.DataFrame | None, value_basis: Mapping[str, float] | None) -> pd.DataFrame:
    """Preserve legacy whole-row exclusions and its as-filed repair summary."""
    value_before = float(df_holdings["value_usd"].sum())
    df_holdings, df_register = repair_value_basis(df_holdings, valuation_close, **dict(value_basis or {}))
    log_register(df_register, value_before)
    if "value_basis_excluded" in df_holdings:
        df_holdings = df_holdings.loc[~df_holdings["value_basis_excluded"]]
    return df_holdings


def clean_legacy_institutional_holdings(
    df_holdings: pd.DataFrame, valuation_close: pd.DataFrame | None, value_basis: Mapping[str, float] | None
) -> pd.DataFrame:
    """Prepare the existing broad no-manifest contract without certifying vintages."""
    df_holdings = clean_holdings(df_holdings, key=("ticker", "cik", "period"), filing_band=(45, 60))
    return _repair_legacy_values(df_holdings, valuation_close, value_basis)


def clean_legacy_manager_holdings(
    df_holdings: pd.DataFrame,
    cusip_map: pd.DataFrame | None,
    universe: Sequence[str],
    valuation_close: pd.DataFrame | None,
    value_basis: Mapping[str, float] | None,
) -> pd.DataFrame:
    """Prepare the existing elite no-manifest grain, common rows and value basis."""
    df_holdings = attach_tickers(df_holdings, cusip_map, universe)
    df_holdings = clean_holdings(df_holdings, key=("cik", "period", "cusip"), numeric=("shares", "value_usd"), common_only=True, pad_ciks=True)
    return _repair_legacy_values(df_holdings, valuation_close, value_basis)


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
    numeric: Sequence[str] = _NUMERIC_13F,
    common_only: bool = False,
    pad_ciks: bool = False,
    filing_band: tuple[int, int] | None = None,
) -> pd.DataFrame:
    """Normalize legacy keys, then apply the filing band and amendment order.

    This latest-only adapter does not certify immutable filing completeness."""
    h = holdings.copy()
    if pad_ciks:
        h["cik"] = h["cik"].map(pad_cik)

    for col in _DATES:
        if col in h.columns:
            h[col] = to_day(h[col])
    for col in numeric:
        h[col] = pd.to_numeric(h[col], errors="coerce") if col in h.columns else pd.Series(np.nan if col == "value_usd" else 0.0, index=h.index)
        if col == "shares":
            h[col] = h[col].fillna(0.0)

    h = h.dropna(subset=list(key))
    if common_only and "position_type" in h.columns:
        h = h[h["position_type"].astype(str).str.lower() == "common"]
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
