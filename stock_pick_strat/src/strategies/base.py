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
    trades: pd.DataFrame | None = None     # per-(day, instrument) trade blotter ($ traded/fee/spread)
    extra: dict = field(default_factory=dict)   # sleeve-specific diagnostics (leverage, cash, ...)
    # The EXACT panels `trades` was built from, so a caller can rebuild the blotter at a
    # different capital. `positions` is not a substitute: long_book reports its pre-leverage
    # allocation there while trading the levered panel, and ls_equity reports None. The daily
    # `strategy` ledger needs the traded panel re-sized by the portfolio's per-sleeve ERC
    # weight x leverage, which is time-varying -- so it must re-run the blotter on
    # `book_weights * factor(t)` rather than scale the resulting $ figures (share counts are
    # a non-linear function of a time-varying capital).
    book_weights: pd.DataFrame | None = None   # date x instrument FRACTIONAL weights, as traded
    book_prices: pd.DataFrame | None = None    # date x instrument price/level panel used to price it


class Strategy(ABC):
    """Base class for a self-contained strategy sleeve."""

    name: str = "strategy"
    # Key of this sleeve's own block in the merged config (configs/strategy/<config_key>.yml).
    # Declared here rather than hardcoded inside each `run` so a caller that needs a sleeve's
    # settings without instantiating it -- the daily ledger reads each sleeve's fee/spread --
    # can find them: the name ("ls_equity") does not map mechanically to the key ("strategy_ls").
    config_key: str = "strategy"

    def __init__(self, context: Context, config: DictConfig) -> None:
        self._context = context
        self._config = config
        self._log = logging.getLogger(__name__)

    @property
    def config(self):
        """This sleeve's own config block."""
        return self._config[self.config_key]

    @abstractmethod
    def run(self, inputs: PortfolioInputs) -> StrategyResult:
        """Read config + inputs + data, predict, construct, and return the sleeve's P&L."""
        raise NotImplementedError
