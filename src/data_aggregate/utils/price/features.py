"""One resolved price basis and its quoted-price, share-count and issuer-cap outputs."""

from __future__ import annotations

import logging
from typing import cast

import numpy as np
import pandas as pd

from src.data_aggregate.utils.price.checks import verified_level_support, verified_price_factor_support
from src.data_aggregate.utils.price.inputs import PriceInputs, merge_price_actions
from src.utils.price.adjustments import ShareBasis, resolve_share_events
from src.utils.price.capitalization import canonical_source_support, capitalization_observations, project_security_shares
from src.utils.price.levels import as_traded_close

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
    """Compute quoted PIT prices and public security counts without enlarging class scope."""
    df_quote = as_traded_close(df_close, df_factor, basis)
    assert df_quote is not None
    supported = verified_level_support(df_close * df_factor, df_vendor, bugfix)
    supported &= verified_price_factor_support(df_close, basis)
    df_quote = df_quote.where(supported & df_close.notna())
    if inputs.splits is None:
        df_quote[:] = np.nan
        logger.warning("PIT quotes unavailable: the Yahoo adjustment source is unavailable")
    df_shares = pd.DataFrame(np.nan, index=df_close.index, columns=df_close.columns, dtype="float64")
    df_scale = df_shares.copy()
    if inputs.shares is not None:
        df_observations = capitalization_observations(inputs.shares, inputs.vendor, pd.DatetimeIndex(df_close.index), basis, inputs.predecessors)
        df_shares, df_scale = project_security_shares(df_observations, basis, pd.DatetimeIndex(df_close.index), df_close.columns)
    for ticker in sorted({owner.ticker for owner in inputs.predecessors} & set(df_close.columns)):
        names = pd.Series(ticker, index=df_close.index)
        dates = pd.Series(df_close.index, index=df_close.index)
        df_shares[ticker] = df_shares[ticker].where(canonical_source_support(names, dates, inputs.predecessors))
    unknown = sorted(inputs.unsupported_units & set(df_close.columns))
    if unknown:
        df_shares.loc[:, unknown] = np.nan
        logger.warning("PIT share units unsupported for %s; source observations remain unchanged", ", ".join(unknown))
    df_shares = df_shares.where(df_close.notna())
    df_cap = df_quote * df_shares * df_scale
    return {"close_pit": df_quote, "shares_outstanding_pit": df_shares, "mcap_pit": df_cap}
