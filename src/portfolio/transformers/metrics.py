"""Comparable net performance, measured from the initial capital before costs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.portfolio.utils.contracts import Settings, Simulation


def metrics(result: Simulation, settings: Settings) -> dict[str, Any]:
    daily = result.daily
    intervals = daily.index.to_series().diff().dt.days.iloc[1:].to_numpy()
    returns = daily["return"].iloc[1:].to_numpy()
    risk_free = (1 + settings.cash_rate) ** (intervals / 365) - 1
    excess = returns - risk_free
    std = float(np.std(excess, ddof=1)) if len(excess) > 1 else 0.0
    days = (daily.index[-1] - daily.index[0]).days
    drawdown = daily.nav / daily.nav.cummax() - 1
    months = pd.period_range(daily.index[0], daily.index[-1], freq="M")
    monthly = pd.Series(0, index=months, dtype=int)
    categories: dict[str, int] = {}
    if not result.orders.empty:
        counts = result.orders.groupby(pd.to_datetime(result.orders.date).dt.to_period("M")).size()
        monthly.loc[counts.index] = counts
        categories = {str(k): int(v) for k, v in result.orders.change.value_counts().items()}
    return {
        "status": result.status,
        "reason": result.reason,
        "start": str(daily.index[0].date()),
        "end": str(daily.index[-1].date()),
        "sessions": len(daily),
        "total_return": float(daily.nav.iloc[-1] / 100 - 1),
        "annual_return": float((daily.nav.iloc[-1] / 100) ** (365 / days) - 1) if days else None,
        "volatility": float(np.std(returns, ddof=1) * np.sqrt(252)) if len(returns) > 1 else None,
        "sharpe": float(np.mean(excess) / std * np.sqrt(252)) if std > 1e-14 else None,
        "max_drawdown": float(drawdown.min()),
        "fees": float(daily.fee.sum()),
        "cash_interest": float(daily.interest.sum()),
        "position_changes": len(result.orders),
        "changes_per_month": float(monthly.mean()),
        "monthly_changes": {str(k): int(v) for k, v in monthly.items()},
        "change_types": categories,
        "turnover_notional": float(result.orders.notional.abs().sum()) if not result.orders.empty else 0.0,
        "ls_pnl": float(daily.ls_pnl.sum()),
        "sp_pnl": float(daily.sp_pnl.sum()),
        "ls_fees": float(daily.ls_fee.sum()),
        "sp_fees": float(daily.sp_fee.sum()),
        "partial_months": [str(months[0]), str(months[-1])],
    }
