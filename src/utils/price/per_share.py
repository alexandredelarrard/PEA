"""Supported per-share price pairs and quote/count capitalization coverage."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.price.adjustments import ShareBasis
from src.utils.price.levels import rebase_security_quantities


def eps_basis_supported(tickers: pd.Series, dates: pd.Series, basis: ShareBasis | None) -> pd.Series:
    """Keep archive EPS only when its source-date quote unit is established as ordinary."""
    if basis is None:
        return pd.Series(False, index=dates.index)
    return basis.factor(tickers, dates, "units").eq(1.0)


def eps_price(df_close: pd.DataFrame, df_factor: pd.DataFrame | None, basis: ShareBasis | None) -> pd.DataFrame:
    """Pair archive EPS with the published adjusted price level on supported daily units."""
    if df_factor is None or basis is None:
        return pd.DataFrame(np.nan, index=df_close.index, columns=df_close.columns)
    df_factor = df_factor.reindex_like(df_close)
    df_price = (df_close * df_factor).where(np.isfinite(df_factor) & df_factor.gt(0))
    dates = pd.Series(df_price.index, index=df_price.index)
    for ticker in df_price:
        names = pd.Series(str(ticker), index=df_price.index)
        df_price[ticker] = df_price[ticker].where(eps_basis_supported(names, dates, basis))
    return df_price.where(np.isfinite(df_price) & df_price.gt(0))


def vendor_ordinary_per_share(
    basis: ShareBasis, values: pd.Series, tickers: pd.Series, source_dates: pd.Series, target_dates: pd.Series
) -> pd.Series:
    """Express signed vendor per-share values in target-date ordinary units, with no quote-unit dependency."""
    if not basis.actual_available:
        return pd.Series(np.nan, index=values.index)
    quantities = basis.rebase(pd.Series(1.0, index=values.index), tickers, source_dates, target_dates)
    multiplier = basis.factor(tickers, source_dates, "vendor") / quantities
    converted = pd.to_numeric(values, errors="coerce") * multiplier
    return converted.where(np.isfinite(multiplier) & multiplier.gt(0) & np.isfinite(converted))


def ordinary_per_share_to_security(
    basis: ShareBasis, values: pd.Series, tickers: pd.Series, source_dates: pd.Series, target_dates: pd.Series
) -> pd.Series:
    """Convert signed ordinary per-share values inversely to target-date security quantities."""
    if not basis.actual_available:
        return pd.Series(np.nan, index=values.index)
    ones = pd.Series(1.0, index=values.index)
    multiplier = rebase_security_quantities(basis, ones, tickers, source_dates, target_dates, ones)
    converted = pd.to_numeric(values, errors="coerce") / multiplier
    return converted.where(np.isfinite(multiplier) & multiplier.gt(0) & np.isfinite(converted))


def same_capital_scope(df_quote: pd.DataFrame | None, df_count: pd.DataFrame | None, df_cap: pd.DataFrame) -> pd.DataFrame:
    """Identify supported same-scope quote/count/cap identities without filling any input."""
    if df_quote is None or df_count is None:
        return pd.DataFrame(False, index=df_cap.index, columns=df_cap.columns)
    quotes = df_quote.reindex_like(df_cap).to_numpy(dtype=float)
    counts = df_count.reindex_like(df_cap).to_numpy(dtype=float)
    caps = df_cap.to_numpy(dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        product = quotes * counts
    supported = np.isfinite(quotes) & (quotes > 0) & np.isfinite(counts) & (counts > 0) & np.isfinite(caps) & (caps > 0)
    equal = np.isclose(product, caps, rtol=1e-12, atol=0)
    return pd.DataFrame(supported & equal, index=df_cap.index, columns=df_cap.columns)
