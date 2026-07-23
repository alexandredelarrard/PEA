"""Self-contained strategy sleeves, each exposing the common `Strategy.run(inputs)` interface."""
from src.strategies.base import Strategy, PortfolioInputs, StrategyResult
from src.strategies.step_ls import LongShortStrategy
from src.strategies.step_long_book import LongBookStrategy
from src.strategies.step_trend import TrendCTAStrategy

# name -> class, for the portfolio step to build the configured sleeves
STRATEGY_REGISTRY: dict[str, type[Strategy]] = {
    LongShortStrategy.name: LongShortStrategy,       # "ls_equity"
    LongBookStrategy.name: LongBookStrategy,         # "long_book"
    TrendCTAStrategy.name: TrendCTAStrategy,          # "trend_cta"
}

__all__ = ["Strategy", "PortfolioInputs", "StrategyResult", "LongShortStrategy",
           "LongBookStrategy", "TrendCTAStrategy", "STRATEGY_REGISTRY"]
