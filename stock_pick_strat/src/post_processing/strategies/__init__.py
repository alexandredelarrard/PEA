"""Backtest strategy sleeves, each exposing the common `Strategy.returns()` interface."""
from src.post_processing.strategies.base import Strategy
from src.post_processing.strategies.strategy_ls import LongShortStrategy
from src.post_processing.strategies.strategy_long_book import LongBookStrategy
from src.post_processing.strategies.strategy_trend import TrendCTAStrategy

# name -> class, for the orchestrator to build the configured sleeves
STRATEGY_REGISTRY: dict[str, type[Strategy]] = {
    LongShortStrategy.name: LongShortStrategy,       # "ls_equity"
    LongBookStrategy.name: LongBookStrategy,         # "long_book"
    TrendCTAStrategy.name: TrendCTAStrategy,          # "trend_cta"
}

__all__ = ["Strategy", "LongShortStrategy", "LongBookStrategy", "TrendCTAStrategy",
           "STRATEGY_REGISTRY"]
