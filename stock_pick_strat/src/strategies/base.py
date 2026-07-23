"""
base.py  (src/strategies/base.py)
---------------------------------
Common interface for a STRATEGY step. Each strategy is self-contained (it does NOT depend on
any other strategy or on a backtest step) and follows the same flow in `run(inputs)`:
  1. read its own config (`configs/strategy/strategy_*.yml`)
  2. read the PortfolioInputs handed down by the portfolio (capital, target vol, window, fees)
  3. read the data it needs (DB)
  4. predict the underlying signal (L/S: model scores; trend: vol-normalized forecasts;
     long_book: risk-parity target weights)
  5. construct the book + optimize weights per its config
  6. compute per-day P&L / positions / metrics -> StrategyResult

The portfolio step then blends each sleeve's `StrategyResult.returns` (risk-parity / ERC) and
allocates capital dynamically.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd
from omegaconf import DictConfig

from src.context import Context


@dataclass(frozen=True)
class PortfolioInputs:
    """What the portfolio layer hands down to each strategy sleeve."""
    capital: float = 1_000_000.0          # notional allocated to the sleeve (for $ P&L / sizing)
    target_vol: float = 0.10              # per-sleeve reference annual vol the sleeve targets
    start: pd.Timestamp | None = None      # backtest window start (shared across sleeves)
    end: pd.Timestamp | None = None        # backtest window end
    fee_bps: float = 2.0                   # default trading cost (bps) unless the sleeve overrides
    spread_bps: float = 8.0
    risk_free_rate: float = 0.02
    analysis: bool = False                 # if True, each sleeve saves its analysis/plots


@dataclass
class StrategyResult:
    """One sleeve's backtest output."""
    name: str
    returns: pd.Series                     # daily NET returns (fractional), date-indexed
    metrics: dict                          # ann_return / ann_vol / sharpe / max_drawdown
    positions: pd.DataFrame | None = None  # per-instrument / per-asset weights (optional)
    extra: dict = field(default_factory=dict)   # sleeve-specific diagnostics (leverage, cash, ...)


class Strategy(ABC):
    """Base class for a self-contained strategy sleeve."""

    name: str = "strategy"

    def __init__(self, context: Context, config: DictConfig) -> None:
        self._context = context
        self._config = config
        self._log = logging.getLogger(__name__)

    @abstractmethod
    def run(self, inputs: PortfolioInputs) -> StrategyResult:
        """Read config + inputs + data, predict, construct, and return the sleeve's P&L."""
        raise NotImplementedError
