"""Dated capitalization evidence and canonical outstanding-count projection."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.utils.cutover_continuity import PredecessorSeries
from src.utils.price.adjustments import ShareBasis
from src.utils.price.levels import convert_quantities, security_basis, vendor_identity_support, vendor_observation_factor

logger = logging.getLogger(__name__)


def canonical_source_support(tickers: pd.Series, dates: pd.Series, predecessors: tuple[PredecessorSeries, ...]) -> pd.Series:
    """Refuse current-ticker financial evidence while a different predecessor owns the quote."""
    if not tickers.index.equals(dates.index):
        raise ValueError("Source ticker and owner dates must be index-aligned")
    stamps = pd.to_datetime(dates, errors="coerce")
    supported = tickers.notna() & stamps.notna()
    for owner in predecessors:
        if owner.ticker == owner.vendor_ticker:
            continue
        selected = tickers.eq(owner.ticker)
        if owner.valid_from is not None:
            selected &= stamps.ge(owner.valid_from)
        if owner.valid_to is not None:
            selected &= stamps.lt(owner.valid_to)
        supported &= ~selected
    return supported


def _restore_observation_counts(df_observations: pd.DataFrame, basis: ShareBasis) -> pd.DataFrame:
    """Restore a matched raw vintage or retain a corroborated original count, allowing one share of rounding."""
    count = pd.to_numeric(df_observations["sharesOutstanding"], errors="coerce")
    original = pd.to_numeric(df_observations["sharesOutstandingPit"], errors="coerce")
    identity = vendor_identity_support(basis, df_observations["ticker"], df_observations["as_of"])
    supported = identity & np.isfinite(count) & count.gt(0) & np.isclose(original, count, rtol=0, atol=1.0)
    if {"sharesbas", "lastupdated"}.issubset(df_observations):
        raw = pd.to_numeric(df_observations["sharesbas"], errors="coerce")
        observed = vendor_observation_factor(basis, df_observations["ticker"], df_observations["as_of"], df_observations["lastupdated"])
        restored = raw / observed
        supported |= np.isfinite(restored) & restored.gt(0) & np.isclose(original, restored, rtol=0, atol=1.0)
        matched = np.isclose(count, raw, rtol=1e-12, atol=0) & count.gt(0) & observed.notna()
        original = original.where(~matched, restored)
        supported |= matched
    valid = supported & np.isfinite(original) & original.gt(0) & df_observations["source_owner_supported"]
    if (~valid).any():
        logger.warning(
            "PIT share counts unsupported for %d public observations; original units, source vintage or owner are uncorroborated", int((~valid).sum())
        )
    df_observations["sharesOutstandingPit"] = original.where(valid)
    return df_observations


def capitalization_observations(
    df_history: pd.DataFrame,
    df_vendor: pd.DataFrame | None,
    calendar: pd.DatetimeIndex,
    basis: ShareBasis,
    predecessors: tuple[PredecessorSeries, ...] = (),
) -> pd.DataFrame:
    """Attach a supported coefficient to the exact public share observation it values.

    Unversioned nonunit coefficients start strictly after lastupdated and expire at
    the next public observation. Prices and reference caps never fit the coefficient.
    """
    df_observations = df_history.copy()
    df_observations["as_of"] = pd.to_datetime(df_observations["as_of"], errors="coerce")
    df_observations["fiscal_end"] = pd.to_datetime(df_observations["fiscal_end"], errors="coerce")
    if df_observations[["ticker", "as_of"]].isna().any().any() or df_observations.duplicated(["ticker", "as_of"]).any():
        raise ValueError("Public share observations require unique, valid ticker/as_of keys")
    df_observations["capitalization_factor"] = np.nan
    df_observations["capitalization_available"] = pd.NaT
    df_observations["source_owner_supported"] = canonical_source_support(df_observations["ticker"], df_observations["as_of"], predecessors)
    required = ("ticker", "date", "reportperiod", "sharesbas")
    if df_vendor is None or not set(required).issubset(df_vendor.columns):
        return _restore_observation_counts(df_observations, basis)

    # Match the producer's same-date greatest-fiscal-period collapse before joining.
    optional = ("sharefactor", "lastupdated")
    df_raw = df_vendor[[*required, *(field for field in optional if field in df_vendor)]].copy()
    if "sharefactor" not in df_raw:
        df_raw["sharefactor"] = np.nan
    if "lastupdated" not in df_raw:
        df_raw["lastupdated"] = pd.NaT
    for field in ("date", "reportperiod", "lastupdated"):
        df_raw[field] = pd.to_datetime(df_raw[field], errors="coerce")
    df_raw = df_raw.dropna(subset=["ticker", "date", "reportperiod"])
    if df_raw.duplicated(["ticker", "date", "reportperiod"]).any():
        raise ValueError("Conflicting or repeated raw capitalization observation keys")
    df_raw = df_raw.sort_values(["ticker", "date", "reportperiod"]).drop_duplicates(["ticker", "date"], keep="last")
    df_observations = df_observations.drop(columns=["capitalization_factor", "capitalization_available"]).merge(
        df_raw, how="left", left_on=["ticker", "as_of", "fiscal_end"], right_on=["ticker", "date", "reportperiod"], validate="one_to_one"
    )
    count = pd.to_numeric(df_observations["sharesOutstanding"], errors="coerce")
    vendor_count = pd.to_numeric(df_observations["sharesbas"], errors="coerce")
    coefficient = pd.to_numeric(df_observations["sharefactor"], errors="coerce")
    units = basis.factor(df_observations["ticker"], df_observations["as_of"], "units")
    matched = np.isclose(count, vendor_count, rtol=1e-12, atol=0) & count.gt(0)
    supported = matched & np.isfinite(coefficient) & coefficient.gt(0) & df_observations["source_owner_supported"]
    # A mixed ADR/class coefficient needs a separate decomposition, never a second U.
    supported &= units.eq(1) | coefficient.eq(1)
    dated = coefficient.ne(1) | units.ne(1)
    positions = calendar.searchsorted(df_observations["lastupdated"].dt.normalize(), side="right")
    ready = pd.Series(pd.NaT, index=df_observations.index, dtype="datetime64[ns]")
    in_calendar = positions < len(calendar)
    ready.loc[in_calendar] = calendar.take(positions[in_calendar]).to_numpy()
    public = df_observations["as_of"]
    ready = pd.concat([public, ready], axis=1).max(axis=1).where(ready.notna() & df_observations["lastupdated"].notna())
    df_observations["capitalization_factor"] = coefficient.where(supported)
    df_observations["capitalization_available"] = public.where(~dated, ready).where(supported)
    return _restore_observation_counts(df_observations, basis)


def project_security_shares(df_observations: pd.DataFrame, basis: ShareBasis, df_basis: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carry the latest public ordinary count onto each date's quoted-security units by `F_d / F_s`.

    `df_basis` is the forward basis F on the output grid. The coefficient follows that same observation;
    an unsupported successor ends its predecessor's support instead of being forward-filled through.
    """
    index = pd.DatetimeIndex(df_basis.index)
    df_shares = pd.DataFrame(np.nan, index=index, columns=df_basis.columns, dtype="float64")
    df_scale = df_shares.copy()
    if df_observations.empty or index.empty:
        return df_shares, df_scale

    # Each observation in its own date's quoted units, with that date's forward basis.
    df_source = df_observations.sort_values(["ticker", "as_of"]).copy()
    original = pd.to_numeric(df_source["sharesOutstandingPit"], errors="coerce")
    df_source["quoted_count"] = original.where(original.gt(0)) / basis.factor(df_source["ticker"], df_source["as_of"], "units")
    df_source["source_basis"] = security_basis(basis, df_source["ticker"], df_source["as_of"])

    # Latest observation at or before each date, converted onto that date's basis.
    stamps = index.to_numpy(dtype="datetime64[ns]")
    for ticker, df_group in df_source.groupby("ticker", sort=False):
        if ticker not in df_basis.columns:
            continue
        source_positions = np.searchsorted(df_group["as_of"].to_numpy(dtype="datetime64[ns]"), stamps, side="right") - 1
        have_source = source_positions >= 0
        selected = np.maximum(source_positions, 0)
        quantities = np.where(have_source, df_group["quoted_count"].to_numpy(dtype="float64")[selected], np.nan)
        source = df_group["source_basis"].to_numpy(dtype="float64")[selected]
        coefficient = df_group["capitalization_factor"].to_numpy(dtype="float64")[selected]
        ready = df_group["capitalization_available"].to_numpy(dtype="datetime64[ns]")[selected]
        df_shares[ticker] = convert_quantities(quantities, source, df_basis[ticker].to_numpy(dtype="float64"))
        df_scale[ticker] = np.where(have_source & (stamps >= ready), coefficient, np.nan)
    return df_shares, df_scale
