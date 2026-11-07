"""One resolved price basis and its quoted-price, share-count and issuer-cap outputs."""

from __future__ import annotations

import logging
from typing import cast

import numpy as np
import pandas as pd

from src.data_aggregate.utils.price.capitalization import canonical_source_support, capitalization_observations, project_security_shares
from src.data_aggregate.utils.price.checks import verified_level_support, verified_price_factor_support
from src.data_aggregate.utils.price.inputs import PriceInputs, merge_price_actions
from src.utils.price.adjustments import ShareBasis, resolve_share_events, suffix_factor
from src.utils.price.levels import convert_quantities, security_basis, validate_quantity_events

logger = logging.getLogger(__name__)


def resolve_price_basis(df_close: pd.DataFrame, inputs: PriceInputs, bugfix: dict) -> ShareBasis:
    """Resolve the full dated evidence once using repaired, already-loaded anchors."""
    names = list(df_close.columns)
    df_actions = merge_price_actions(inputs, names)
    anchors = [
        (ticker, anchor)
        for ticker, policy in bugfix.get("share_events", {}).items()
        if ticker in names
        for anchor in policy["anchors"]
        if "quoted_price" in anchor
    ]
    df_quotes = None
    if anchors:
        dates = pd.DatetimeIndex([anchor["date"] for _, anchor in anchors])
        df_covered = df_close.loc[df_close.index.isin(dates), sorted({ticker for ticker, _ in anchors})]
        stacked = cast(pd.Series, df_covered.rename_axis(index="date", columns="ticker").stack(future_stack=True))
        df_quotes = stacked.dropna().rename("close_split").reset_index()
    return resolve_share_events(df_actions, inputs.splits, bugfix, shares=inputs.vendor, quotes=df_quotes, tickers=names)


def pit_fields(
    df_close: pd.DataFrame, df_factor: pd.DataFrame, basis: ShareBasis, inputs: PriceInputs, bugfix: dict, df_vendor: pd.DataFrame | None
) -> dict[str, pd.DataFrame]:
    """Compute quoted PIT prices, the forward basis F and public security counts without enlarging class scope."""
    # Forward basis F on the price grid; null where quote units are unsupported.
    index = pd.DatetimeIndex(df_close.index)
    unknown = sorted(inputs.unsupported_units & set(df_close.columns))
    df_basis = security_basis_frame(basis, index, df_close.columns)
    df_basis.loc[:, unknown] = np.nan

    # Quoted security prices where the level and vendor price factors are verified.
    df_quote = as_traded_close(df_close, df_factor, basis)
    assert df_quote is not None
    supported = verified_level_support(df_close * df_factor, df_vendor, bugfix)
    supported &= verified_price_factor_support(df_close, basis)
    df_quote = df_quote.where(supported & df_close.notna())
    if inputs.splits is None:
        df_quote[:] = np.nan
        logger.warning("PIT quotes unavailable: the Yahoo adjustment source is unavailable")

    # Public counts carried onto each date's units by F, refused where a predecessor owns the quote.
    df_shares = pd.DataFrame(np.nan, index=index, columns=df_close.columns, dtype="float64")
    df_scale = df_shares.copy()
    if inputs.shares is not None:
        df_observations = capitalization_observations(inputs.shares, inputs.vendor, basis, inputs.predecessors)
        df_shares, df_scale = project_security_shares(df_observations, basis, df_basis)
    for ticker in sorted({owner.ticker for owner in inputs.predecessors} & set(df_close.columns)):
        names = pd.Series(ticker, index=index)
        df_shares[ticker] = df_shares[ticker].where(canonical_source_support(names, pd.Series(index, index=index), inputs.predecessors))
    if unknown:
        logger.warning("PIT share units unsupported for %s; source observations remain unchanged", ", ".join(unknown))
    # The vendor class coefficient puts the count in canonical-class equivalents (BRK-B, V), so mcap = close x count.
    df_shares = df_shares.where(df_close.notna()) * df_scale
    df_cap = df_quote * df_shares

    # Saved F is 1 on each ticker's first bar; consumers use only its ratios, which this leaves unchanged.
    df_basis = df_basis / df_basis.where(df_close.notna()).bfill().iloc[0]
    return {"close_pit": df_quote, "shares_outstanding_pit": df_shares, "mcap_pit": df_cap, "security_basis": df_basis}


def as_traded_volume(df_volume: pd.DataFrame, df_basis: pd.DataFrame, basis: ShareBasis) -> pd.DataFrame:
    """Vendor volume carried from the terminal basis back to each date's by `F_d / F_T`.

    Null where the Yahoo tape restatement is not proven equal to that basis change.
    """
    supported = compatible_tape_support(basis, df_basis)
    return cast(pd.DataFrame, convert_quantities(df_volume, df_basis.iloc[-1], df_basis)).where(supported)


def security_basis_frame(basis: ShareBasis, index: pd.DatetimeIndex, columns: pd.Index) -> pd.DataFrame:
    """`security_basis` on a date x ticker grid."""
    keys = pd.MultiIndex.from_product([index, columns])
    values = security_basis(basis, pd.Series(keys.get_level_values(1).astype(str)), pd.Series(keys.get_level_values(0)))
    return pd.DataFrame(values.to_numpy().reshape(len(index), len(columns)), index=index, columns=columns)


def as_traded_close(
    close_split: pd.DataFrame | None,
    level_factor: pd.DataFrame | None,
    split_events: pd.DataFrame | ShareBasis,
) -> pd.DataFrame | None:
    """Restore quoted-security prices using vendor restatements and evidenced quote units."""
    if close_split is None or level_factor is None:
        return None
    df_close = close_split * level_factor.reindex_like(close_split)
    if isinstance(split_events, ShareBasis):
        basis = split_events
        events = basis.vendor
    else:
        basis = None
        events = split_events
    factors = suffix_factor(events, list(df_close.columns), df_close.index.to_numpy(dtype="datetime64[ns]"))
    for ticker, factor in factors.items():
        df_close[ticker] *= factor
    if basis is not None:
        for ticker in basis.policies.keys() & set(df_close.columns):
            df_close[ticker] *= basis.factor(pd.Series(ticker, index=df_close.index), pd.Series(df_close.index, index=df_close.index), "units")
    return df_close.where(np.isfinite(df_close) & df_close.gt(0))


def _tape_events(basis: ShareBasis) -> pd.DataFrame:
    """The genuine events the Yahoo volume tape is restated by: those its own split ledger also dates.

    Yahoo scales volume for the splits it records, never for the spin-off price factors it files as splits.
    """
    ledger = basis.price[["ticker", "date"]].assign(date=lambda df: pd.to_datetime(df["date"]).dt.normalize()).drop_duplicates()
    events = basis.actual.assign(date=pd.to_datetime(basis.actual["date"]).dt.normalize())
    return events.merge(ledger, on=["ticker", "date"], how="inner")


def compatible_tape_support(basis: ShareBasis, df_basis: pd.DataFrame) -> pd.DataFrame:
    """True where the Yahoo tape restatement after each date equals `F_T / F_d`, the change `volume_pit` undoes."""
    index = pd.DatetimeIndex(df_basis.index)
    df_supported = pd.DataFrame(False, index=index, columns=df_basis.columns)
    if not basis.actual_available:
        return df_supported
    validate_quantity_events(basis.actual)
    tape = suffix_factor(_tape_events(basis), list(df_basis.columns), index.to_numpy(dtype="datetime64[ns]"))
    df_growth = df_basis.iloc[-1] / df_basis
    for ticker in df_basis.columns:
        restatement = tape.get(str(ticker), np.ones(len(index)))
        growth = df_growth[ticker].to_numpy(dtype="float64")
        df_supported[ticker] = np.isfinite(growth) & (growth > 0) & np.isclose(restatement, growth, rtol=1e-12, atol=0)
    return df_supported
