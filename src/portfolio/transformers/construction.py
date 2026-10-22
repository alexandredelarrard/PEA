"""Point-in-time sector quotas and diagonal-risk long/short weights."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.portfolio.utils.contracts import IncompleteScenario, MarketInputs, Settings


def risk_panels(data: MarketInputs, settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trailing windows include only returns observed by each signal close."""
    variance = data.returns.rolling(settings.risk_window, min_periods=settings.risk_window).var()
    sector_returns = data.returns.T.groupby(data.sectors).mean().T
    momentum = (1 + sector_returns).rolling(settings.momentum_window, min_periods=settings.momentum_window).apply(np.prod, raw=True) - 1
    return variance, momentum


def construct(
    data: MarketInputs,
    date: pd.Timestamp,
    n: int,
    variance: pd.DataFrame,
    momentum: pd.DataFrame,
) -> pd.Series:
    """N names per side in total; sector weights sum to zero and gross to one."""
    if date not in data.scores.index or date not in variance.index or date not in momentum.index:
        raise IncompleteScenario(f"{date.date()}: missing signal/risk history")
    frame = pd.DataFrame({"score": data.scores.loc[date], "variance": variance.loc[date], "sector": data.sectors, "quote": data.quotes.loc[date]})
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    frame = frame.loc[(frame.variance > 0) & (frame.quote > 0)]
    capacity = frame.groupby("sector").size().floordiv(2)
    trend = momentum.loc[date].replace([np.inf, -np.inf], np.nan).dropna()
    sectors = sorted((s for s in capacity.index if capacity[s] > 0 and s in trend.index), key=lambda s: (-trend[s], str(s)))[:n]
    if sum(int(capacity[s]) for s in sectors) < n:
        raise IncompleteScenario(f"{date.date()}: insufficient sector capacity for {n} names per side")
    trend = trend.loc[sectors]
    ranks = trend.rank(method="average")
    tilt = 1 + ((ranks - 1) / (len(ranks) - 1) if len(ranks) > 1 else ranks * 0)
    budgets = tilt / tilt.sum()
    quotas = {s: 1 for s in sectors}
    for _ in range(n - len(sectors)):
        available = [s for s in sectors if quotas[s] < capacity[s]]
        chosen = min(available, key=lambda s: (-(n * budgets[s] - quotas[s]), str(s)))
        quotas[chosen] += 1
    weights = pd.Series(0.0, index=frame.index)
    for sector in sectors:
        group = frame.loc[frame.sector == sector].sort_index().sort_values("score", kind="stable")
        k = quotas[sector]
        for names, sign in ((group.index[:k], -1), (group.index[-k:], 1)):
            inv = 1 / frame.loc[names, "variance"]
            weights.loc[names] = sign * budgets[sector] / 2 * inv / inv.sum()
    weights = weights.loc[weights != 0]
    assert len(weights) == 2 * n and np.isclose(weights.abs().sum(), 1)
    return weights
