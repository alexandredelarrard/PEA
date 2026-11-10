"""As-of lookups on saved date x ticker grids, and the terminal-unit factor of the saved basis."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd


def period_close(close: pd.DataFrame, periods: pd.Series, max_age_days: int) -> pd.Series:
    """Last observed valuation-day quote, bounded by age and never from the future."""
    wanted = pd.DatetimeIndex(sorted(periods.dropna().unique()))
    df_quotes = close.sort_index().where(np.isfinite(close) & close.gt(0))
    union = df_quotes.index.union(wanted)
    df_asof = df_quotes.reindex(union).ffill().reindex(wanted)
    for ticker in df_quotes:
        observed = pd.Series(df_quotes.index, index=df_quotes.index).where(df_quotes[ticker].notna())
        last = observed.reindex(union).ffill().reindex(wanted)
        df_asof[ticker] = df_asof[ticker].where((pd.Series(wanted, index=wanted) - last).le(pd.Timedelta(days=max_age_days)))
    result = cast(pd.Series, df_asof.stack(future_stack=True))
    result.index = result.index.set_names(["period", "ticker"])
    return result


def lookup_asof(df_values: pd.DataFrame, tickers: pd.Series, dates: pd.Series) -> pd.Series:
    """Per (ticker, date), the value saved on the last grid date at or before the date; a null there, or a missing date, stays null."""
    index = pd.DatetimeIndex(df_values.index)
    stamps = pd.to_datetime(dates).to_numpy(dtype="datetime64[ns]")
    rows = index.searchsorted(stamps, side="right") - 1
    cols = df_values.columns.get_indexer(tickers.astype(str))
    found = (rows >= 0) & (cols >= 0) & ~np.isnat(stamps)
    values = df_values.to_numpy(dtype="float64")[np.maximum(rows, 0), np.maximum(cols, 0)]
    return pd.Series(np.where(found, values, np.nan), index=dates.index)


def terminal_basis_factor(df_basis: pd.DataFrame) -> pd.DataFrame:
    """`F_T / F_d` on the saved grid: carries a quantity dated d into each ticker's last saved security units.

    `df_basis` is the saved forward basis F (`security_basis`). F_T is each ticker's last saved value; a null F_d stays null.
    """
    df_sorted = df_basis.sort_index()
    return df_sorted.rtruediv(df_sorted.ffill().iloc[-1], axis="columns")


def terminal_rows(df_rows: pd.DataFrame, columns: tuple[str, ...], df_factor: pd.DataFrame) -> pd.DataFrame:
    """Dated source rows with `columns` carried into terminal units, `q * F_T / F_d` as of each row's `date`."""
    factor = lookup_asof(df_factor, df_rows["ticker"], df_rows["date"])
    return df_rows.assign(**{column: df_rows[column] * factor for column in columns})
