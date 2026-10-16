"""Report-associated EPS expectations and surprises, observable from the next session.

The archive has one terminal consensus per report, not historical forecast vintages.
Unknown split units are masked; PEGY uses reported actuals independently.
"""

from __future__ import annotations

import logging
from typing import cast

import numpy as np
import pandas as pd

from src.data_aggregate.utils.common.frames import ratio
from src.data_aggregate.utils.fundamentals.feature_views import build_fundamental_views
from src.utils.price_adjustments import ShareBasis

logger = logging.getLogger(__name__)
SURPRISE_WINDOW = 4
_QUARTERLY_MAX_AGE_DAYS = 185


def reported_events(hist: pd.DataFrame, trading_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Return finite actual-bearing reports, with next-session availability in `as_of`."""
    df_reports = hist.copy()
    df_reports["earnings_date"] = pd.to_datetime(df_reports["earnings_date"], errors="coerce").dt.normalize()
    for field in ("eps_estimate", "eps_actual", "surprise_pct"):
        values = df_reports[field] if field in df_reports else pd.Series(np.nan, index=df_reports.index)
        df_reports[field] = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    df_reports = df_reports.dropna(subset=["ticker", "earnings_date", "eps_actual"])
    df_reports = df_reports.sort_values(["ticker", "earnings_date"], kind="stable").drop_duplicates(["ticker", "earnings_date"], keep="last")
    positions = trading_index.searchsorted(df_reports["earnings_date"], side="right")
    df_reports = df_reports.loc[positions < len(trading_index)].copy()
    df_reports["as_of"] = trading_index.take(positions[positions < len(trading_index)])
    return df_reports.reset_index(drop=True)


def _report_state_to_daily(
    df_reports: pd.DataFrame,
    value_fields: list[str],
    idx: pd.DatetimeIndex,
    *,
    max_age_days: int | None = _QUARTERLY_MAX_AGE_DAYS,
) -> dict[str, pd.DataFrame]:
    """Select entire report states, preserving invalid cells rather than reviving older values."""
    fields = {field: pd.DataFrame(index=idx, columns=df_reports["ticker"].unique(), dtype="float64") for field in value_fields}
    for ticker, df_ticker in df_reports.groupby("ticker", sort=False):
        reports = df_ticker["earnings_date"].to_numpy(dtype="datetime64[ns]")
        positions = reports.searchsorted(idx.to_numpy(), side="left") - 1
        selected = positions.clip(min=0)
        valid = positions >= 0
        if max_age_days is not None:
            valid &= (idx.to_numpy() - reports[selected]) <= pd.Timedelta(days=max_age_days)
        for field in value_fields:
            fields[field][ticker] = np.where(valid, df_ticker[field].to_numpy(dtype="float64")[selected], np.nan)
    return fields


def _supported_eps_basis(df_reports: pd.DataFrame, share_basis: ShareBasis | None) -> pd.Series:
    """Reject report EPS whose split vintage or quoted-security units are unsupported."""
    if share_basis is None:
        return pd.Series(False, index=df_reports.index)
    dates = df_reports["earnings_date"]
    units = share_basis.factor(df_reports["ticker"], dates, "units")
    df_splits = share_basis.actual
    if df_splits.empty:
        return units.eq(1.0)
    df_splits = df_splits.loc[pd.to_numeric(df_splits["value"], errors="coerce").ne(1.0)]
    if df_splits.empty:
        return units.eq(1.0)
    last_split = pd.to_datetime(df_splits["date"]).groupby(df_splits["ticker"]).max()
    boundary = pd.to_datetime(df_reports["ticker"].map(last_split))
    # shortcut: the archive has no EPS split-vintage evidence; retain only post-last-split reports.
    return units.eq(1.0) & (boundary.isna() | dates.ge(boundary))


def days_since_earnings(df: pd.DataFrame, idx: pd.DatetimeIndex, cap_days: int = 180) -> pd.DataFrame:
    """Calendar age of the latest actual-bearing report, visible next session and capped."""
    df_reports = reported_events(df, idx)
    df_reports["report_ordinal"] = (df_reports["earnings_date"] - pd.Timestamp("1970-01-01")).dt.days
    daily = _report_state_to_daily(df_reports, ["report_ordinal"], idx, max_age_days=None)["report_ordinal"]
    current = pd.Series((idx - pd.Timestamp("1970-01-01")).days, index=idx)
    return daily.rsub(current, axis=0).clip(lower=0, upper=cap_days)


def _derived_earnings_fields(
    hist: pd.DataFrame,
    idx: pd.DatetimeIndex,
    close: pd.DataFrame,
    level_factor: pd.DataFrame | None = None,
    *,
    share_basis: ShareBasis | None = None,
) -> dict[str, pd.DataFrame]:
    """Build observed report expectations; unsupported EPS bases never enter price ratios."""
    df_reports = reported_events(hist, idx)
    safe = _supported_eps_basis(df_reports, share_basis)
    logger.info("Report EPS share basis: supported=%s, masked_unknown_units=%s", int(safe.sum()), int((~safe).sum()))
    df_reports["reported_expected_eps_yield"] = df_reports["eps_estimate"].where(safe)
    grouped = df_reports.groupby("ticker", sort=False)
    prior = grouped["eps_actual"].shift()
    prior_safe = safe.groupby(df_reports["ticker"]).shift().fillna(False).astype(bool)
    # shortcut: report calendar quarters proxy fiscal keys until the archive exposes fiscal periods.
    quarter = df_reports["earnings_date"].dt.to_period("Q").astype("int64")
    adjacent = quarter.sub(quarter.groupby(df_reports["ticker"]).shift()).eq(1)
    df_reports["reported_expected_eps_growth_qoq"] = (df_reports["eps_estimate"] / prior.where(prior.gt(0)) - 1).where(safe & prior_safe & adjacent)
    trailing = pd.Series(0.0, index=df_reports.index)
    complete = safe.copy()
    for lag in range(1, 4):
        preceding = grouped["eps_actual"].shift(lag)
        trailing += preceding
        complete &= safe.groupby(df_reports["ticker"]).shift(lag).fillna(False).astype(bool)
        complete &= quarter.sub(quarter.groupby(df_reports["ticker"]).shift(lag)).eq(lag)
    df_reports["reported_expected_ttm_yield"] = (df_reports["eps_estimate"] + trailing).where(complete)
    df_reports["eps_surprise_last"] = df_reports["surprise_pct"]
    df_surprises = df_reports.dropna(subset=["surprise_pct"])
    df_reports["eps_surprise_4q_avg"] = df_surprises.groupby("ticker", sort=False)["surprise_pct"].transform(
        lambda values: values.rolling(SURPRISE_WINDOW, min_periods=1).mean()
    )
    names = [
        "reported_expected_eps_yield",
        "reported_expected_ttm_yield",
        "reported_expected_eps_growth_qoq",
        "eps_surprise_last",
        "eps_surprise_4q_avg",
    ]
    df_reports[names] = df_reports[names].replace([np.inf, -np.inf], np.nan)
    fields = {name: values.reindex(columns=close.columns) for name, values in _report_state_to_daily(df_reports, names, idx).items()}
    price = close.reindex(idx)
    if level_factor is not None:
        price = price.mul(level_factor.reindex_like(price))
    if share_basis is not None:
        for ticker in share_basis.policies.keys() & set(price.columns):
            units = share_basis.factor(pd.Series(ticker, index=idx), pd.Series(idx, index=idx), "units")
            price[ticker] = price[ticker].where(units.eq(1.0))
    for name in ("reported_expected_eps_yield", "reported_expected_ttm_yield"):
        fields[name] = ratio(fields[name], price, positive_den=True)
    return fields


def build_earnings_feature_panel(
    earnings_history: pd.DataFrame | None,
    peer_dict: dict,
    trading_index: pd.DatetimeIndex,
    stock_close: pd.DataFrame | None = None,
    level_factor: pd.DataFrame | None = None,
    history_fields: dict[str, pd.DataFrame] | None = None,
    output_since: pd.Timestamp | None = None,
    *,
    share_basis: ShareBasis | None = None,
) -> pd.DataFrame:
    """Long-format report features using the approved raw/self-history view contract."""
    if earnings_history is None or earnings_history.empty or stock_close is None or "earnings_date" not in earnings_history.columns:
        return pd.DataFrame(columns=["date", "ticker"])
    fields = _derived_earnings_fields(earnings_history, trading_index, stock_close, level_factor, share_basis=share_basis)
    panel = build_fundamental_views(fields, peer_dict, history_fields=history_fields, output_since=output_since)
    days = days_since_earnings(earnings_history, trading_index)
    if output_since is not None:
        days = days.loc[days.index >= pd.Timestamp(output_since)]
    if not days.empty and days.notna().any().any():
        values = cast(pd.Series, days.stack(future_stack=True)).dropna()
        values.index = values.index.set_names(["date", "ticker"])
        df_long = values.rename("f_days_since_earnings").reset_index()
        panel = df_long if panel.empty else panel.merge(df_long, on=["date", "ticker"], how="outer")
    return panel
