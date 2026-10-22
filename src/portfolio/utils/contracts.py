"""Validated inputs and results shared by the backtest transformations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig


@dataclass(frozen=True)
class Settings:
    start: pd.Timestamp
    end: pd.Timestamp | None
    allocation: tuple[float, float, float]
    allocation_frequency: str
    cash_rate: float
    momentum_window: int
    risk_window: int
    commission_bps: float
    spread_bps: float
    sell_fee_bps: float

    @classmethod
    def from_config(cls, config: DictConfig) -> Settings:
        """Reject invalid budgets and costs at the configuration boundary."""
        p = config.portfolio
        allocation = tuple(float(p.allocation[key]) for key in ("long_short", "sp500", "cash"))
        if not all(np.isfinite(v) and v >= 0 for v in allocation) or not np.isclose(sum(allocation), 1, atol=1e-10, rtol=0):
            raise ValueError("Portfolio allocations must be nonnegative and sum to one")
        settings = cls(
            pd.Timestamp(p.start),
            pd.Timestamp(p.end) if p.end is not None else None,
            (allocation[0], allocation[1], allocation[2]),
            str(p.allocation_frequency),
            float(p.cash_rate),
            int(p.momentum_window),
            int(p.risk_window),
            float(p.commission_bps),
            float(p.spread_bps),
            float(p.sell_fee_bps),
        )
        if pd.isna(settings.start) or (settings.end is not None and (pd.isna(settings.end) or settings.end <= settings.start)):
            raise ValueError("Invalid backtest dates")
        if settings.allocation_frequency not in {"daily", "weekly", "monthly"}:
            raise ValueError("Unknown allocation frequency")
        if not np.isfinite(settings.cash_rate) or settings.cash_rate <= -1:
            raise ValueError("Cash rate must be finite and greater than -1")
        if min(settings.momentum_window, settings.risk_window) < 2:
            raise ValueError("Risk and momentum windows need at least two sessions")
        if not all(np.isfinite(v) and 0 <= v < 10000 for v in (settings.commission_bps, settings.spread_bps, settings.sell_fee_bps)):
            raise ValueError("Invalid proportional transaction costs")
        return settings


@dataclass
class MarketInputs:
    scores: pd.DataFrame
    quotes: pd.DataFrame
    returns: pd.DataFrame
    share_basis: pd.DataFrame
    market: pd.Series
    sectors: pd.Series
    metadata: dict[str, Any]


@dataclass
class Simulation:
    n: int
    frequency: str
    daily: pd.DataFrame
    orders: pd.DataFrame
    positions: pd.DataFrame
    status: str
    reason: str | None


class IncompleteScenario(ValueError):  # noqa: N818 -- scenario state, not a missing-data exception exposed to users
    """A required observation is unavailable; never substitute a zero return."""
