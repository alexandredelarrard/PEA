"""Self-contained strategy sleeves, each exposing the common `Strategy.run(inputs)` interface."""
from src.strategies.base import Strategy, PortfolioInputs, StrategyResult
from src.strategies.step_ls import LongShortStrategy
from src.strategies.step_long_book import LongBookStrategy
from src.strategies.step_trend import TrendCTAStrategy
from src.strategies.step_eq_long_only import EqLongOnlyStrategy

# name -> class, for the portfolio step to build the configured sleeves
STRATEGY_REGISTRY: dict[str, type[Strategy]] = {
    LongShortStrategy.name: LongShortStrategy,       # "ls_equity"
    LongBookStrategy.name: LongBookStrategy,         # "long_book"
    TrendCTAStrategy.name: TrendCTAStrategy,          # "trend_cta"
    EqLongOnlyStrategy.name: EqLongOnlyStrategy,      # "eq_long_only"
}

__all__ = ["Strategy", "PortfolioInputs", "StrategyResult", "LongShortStrategy",
           "LongBookStrategy", "TrendCTAStrategy", "EqLongOnlyStrategy", "STRATEGY_REGISTRY"]
