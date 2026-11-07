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
from src.data_aggregate.utils.price.frames import PriceFrames
from src.utils.price.adjustments import ShareBasis
from src.utils.price.per_share import eps_basis_supported, eps_price

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
    safe = eps_basis_supported(df_reports["ticker"], df_reports["earnings_date"], share_basis)
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
    price = eps_price(close.reindex(idx), level_factor, share_basis)
    for name in ("reported_expected_eps_yield", "reported_expected_ttm_yield"):
        fields[name] = ratio(fields[name], price, positive_den=True)
    return fields


def build_earnings_feature_panel(
    earnings_history: pd.DataFrame | None,
    prices: PriceFrames,
    history_fields: dict[str, pd.DataFrame] | None = None,
    output_since: pd.Timestamp | None = None,
    share_basis: ShareBasis | None = None,
) -> pd.DataFrame:
    """Build report features from saved prices and the dated quantity basis."""
    if earnings_history is None or earnings_history.empty or prices.close_split is None or "earnings_date" not in earnings_history.columns:
        return pd.DataFrame(columns=["date", "ticker"])
    fields = _derived_earnings_fields(earnings_history, prices.trading_index, prices.close_split, prices.level_factor, share_basis=share_basis)
    panel = build_fundamental_views(fields, prices.peers, history_fields=history_fields, output_since=output_since)
    days = days_since_earnings(earnings_history, prices.trading_index)
    if output_since is not None:
        days = days.loc[days.index >= pd.Timestamp(output_since)]
    if not days.empty and days.notna().any().any():
        values = cast(pd.Series, days.stack(future_stack=True)).dropna()
        values.index = values.index.set_names(["date", "ticker"])
        df_long = values.rename("f_days_since_earnings").reset_index()
        panel = df_long if panel.empty else panel.merge(df_long, on=["date", "ticker"], how="outer")
    return panel
