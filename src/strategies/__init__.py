"""Self-contained strategy sleeves, each exposing the common `Strategy.run(inputs)` interface."""

from src.strategies.base import PortfolioInputs, Strategy, StrategyResult
from src.strategies.step_eq_long_only import EqLongOnlyStrategy
from src.strategies.step_ls import LongShortStrategy

# name -> class, for the portfolio step to build the configured sleeves
STRATEGY_REGISTRY: dict[str, type[Strategy]] = {
    LongShortStrategy.name: LongShortStrategy,  # "ls_equity"
    EqLongOnlyStrategy.name: EqLongOnlyStrategy,  # "eq_long_only"
}

__all__ = [
    "Strategy",
    "PortfolioInputs",
    "StrategyResult",
    "LongShortStrategy",
    "EqLongOnlyStrategy",
    "STRATEGY_REGISTRY",
]
