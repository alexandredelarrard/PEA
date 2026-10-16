"""Quarter-based earnings and cash-flow histories replayed at publication events."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_aggregate.utils.common.capital import assets_ex_lease


def _quarter_features(
    fiscal_ends: np.ndarray,
    quarters: np.ndarray,
    earnings_lags: int,
    scale_quarters: int,
    persistence_quarters: int,
) -> np.ndarray:
    """Evaluate the latest known fiscal quarter without counting amendments as periods."""
    # Restrict to a contiguous suffix; a missing fiscal quarter is never a shorter lag.
    gaps = np.diff(fiscal_ends).astype("timedelta64[D]").astype(int)
    breaks = np.flatnonzero((gaps < 60) | (gaps > 120))
    if len(breaks):
        quarters = quarters[breaks[-1] + 1 :]
    n = len(quarters)
    values = np.full(earnings_lags + 4, np.nan)
    income = quarters[:, 0]
    changes = np.full(n, np.nan)
    changes[4:] = income[4:] - income[:-4]
    for lag in range(min(earnings_lags, n)):
        end = n - lag - 1
        prior = changes[max(0, end - scale_quarters) : end]
        if len(prior) == scale_quarters and np.isfinite(prior).all():
            scale = prior.std(ddof=1)
            if scale > 0:
                values[lag] = changes[end] / scale
    streak = changes[-earnings_lags:][::-1]
    nonpositive = np.flatnonzero(streak <= 0)
    observed = streak[: nonpositive[0] + 1] if len(nonpositive) else streak
    if np.isfinite(observed).all() and (len(nonpositive) or len(streak) == earnings_lags):
        values[earnings_lags] = nonpositive[0] if len(nonpositive) else earnings_lags
    assets = np.where(quarters[:, 3] > 0, quarters[:, 3], np.nan)
    ttm_income = quarters[:, 1]
    if n >= 5:
        values[earnings_lags + 1] = (ttm_income[-1] - ttm_income[-5]) / ((assets[-1] + assets[-5]) / 2)
    if n >= persistence_quarters:
        roa = (ttm_income / assets)[-persistence_quarters:]
        fcf = quarters[-persistence_quarters:, 2]
        if np.isfinite(roa).all():
            values[earnings_lags + 2] = roa.std(ddof=1)
        if np.isfinite(fcf).all():
            values[earnings_lags + 3] = (fcf > 0).mean()
    values[~np.isfinite(values)] = np.nan
    return values


def reported_history_fields(
    history: pd.DataFrame,
    trading_index: pd.DatetimeIndex,
    *,
    earnings_lags: int = 8,
    earnings_scale_quarters: int = 8,
    persistence_quarters: int = 8,
    max_age_days: int = 185,
) -> dict[str, pd.DataFrame]:
    """Replay distinct quarters and prospective revisions, preserving invalid new states."""
    if earnings_lags != 8 or persistence_quarters != 8 or not 2 <= earnings_scale_quarters <= 8:
        raise ValueError("the feature schema requires eight earnings lags, eight-quarter persistence and a scale of two to eight quarters")
    names = [f"earnings_sue_lag{lag}" for lag in range(earnings_lags)]
    names += ["earnings_improvement_streak", "profit_improvement_to_assets", "roa_vol_8q", "fcf_positive_fraction_8q"]
    required = {"ticker", "as_of", "fiscal_end"}
    if history.empty or not required.issubset(history):
        return {}
    financial_fields = ("netIncome_q", "netIncome", "freeCashflow", "totalAssetsExLease", "totalAssets", "operatingLeaseRouAsset")
    work = history[["ticker", "as_of", "fiscal_end", *[field for field in financial_fields if field in history]]].copy()
    for key in ("as_of", "fiscal_end"):
        work[key] = pd.to_datetime(work[key], errors="coerce").dt.normalize()
    for key in financial_fields:
        work[key] = pd.to_numeric(work[key], errors="coerce") if key in work else np.nan
    work["assets"] = assets_ex_lease(lambda field: work[field])
    work = work.dropna(subset=list(required)).sort_values(["ticker", "as_of", "fiscal_end"], kind="stable")
    columns = sorted(work["ticker"].astype(str).unique())
    result = {name: pd.DataFrame(np.nan, index=trading_index, columns=columns) for name in names}
    source_fields = ["netIncome_q", "netIncome", "freeCashflow", "assets"]
    lookback = max(earnings_lags + earnings_scale_quarters + 4, persistence_quarters)
    # Replay filing events in small arrays; the large daily projection is vectorized.
    for ticker, rows in work.groupby("ticker", sort=False):
        state: dict[np.datetime64, np.ndarray] = {}
        as_of = rows["as_of"].to_numpy(dtype="datetime64[ns]")
        fiscal_ends = rows["fiscal_end"].to_numpy(dtype="datetime64[ns]")
        source = rows[source_fields].to_numpy(dtype=float)
        starts = np.r_[0, np.flatnonzero(as_of[1:] != as_of[:-1]) + 1]
        ends = np.r_[starts[1:], len(rows)]
        snapshots = []
        for start, end in zip(starts, ends, strict=True):
            state.update(zip(fiscal_ends[start:end], source[start:end], strict=True))
            periods = sorted(state)[-lookback:]
            quarters = np.asarray([state[period] for period in periods])
            snapshots.append(_quarter_features(np.asarray(periods), quarters, earnings_lags, earnings_scale_quarters, persistence_quarters))
        dates = pd.DatetimeIndex(as_of[starts])
        positions = dates.searchsorted(trading_index, side="right") - 1
        valid = positions >= 0
        valid &= (trading_index - dates.take(positions.clip(min=0))) <= pd.Timedelta(days=max_age_days)
        daily = np.full((len(trading_index), len(names)), np.nan)
        daily[valid] = np.asarray(snapshots)[positions[valid]]
        for offset, name in enumerate(names):
            result[name][str(ticker)] = daily[:, offset]
    return result
