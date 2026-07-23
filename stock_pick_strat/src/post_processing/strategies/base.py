"""
base.py  (src/post_processing/strategies/base.py)
-------------------------------------------------
Common interface for a backtest STRATEGY sleeve. Each strategy runs its own prediction /
weighting / inputs internally and exposes ONE daily NET return stream (its own book) via
`returns()`, plus optional per-instrument `positions()` for inspection. The portfolio
backtest (`StepPortfolioBacktest`) collects each sleeve's `returns()` and blends them
(risk-parity / ERC across sleeves) into one book with a global vol target + leverage.

Keeping every sleeve behind the same tiny interface is what lets the orchestrator treat
L/S equity, the multi-asset long book and the trend/CTA sleeve uniformly.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import pandas as pd
from omegaconf import DictConfig

from src.context import Context


class Strategy(ABC):
    """A single backtest sleeve producing a daily net-return stream.

    Subclasses set `name` and implement `returns()`. `positions()` is optional (returns the
    per-instrument or per-asset weight panel when the sleeve exposes one, else None)."""

    name: str = "strategy"

    def __init__(self, context: Context, config: DictConfig) -> None:
        self._context = context
        self._config = config
        self._log = logging.getLogger(__name__)

    @abstractmethod
    def returns(self) -> pd.Series:
        """Date-indexed daily NET returns of this sleeve's own book (net of its own costs)."""
        raise NotImplementedError

    def positions(self) -> pd.DataFrame | None:
        """Optional per-instrument / per-asset weight panel (date x instrument), for inspection."""
        return None
