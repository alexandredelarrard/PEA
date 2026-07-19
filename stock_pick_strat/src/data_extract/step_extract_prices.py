"""
step_extract_prices.py  (src/data_extract/step_extract_prices.py)
-----------------------------------------------------------------
Price / stock-market data extraction:
  * price history (+ dividends, from the same yfinance download)
  * short interest
  * 13F institutional holdings
"""
from omegaconf import DictConfig

from src.context import Context
from src.utils.step import Step
from src.data_extract.utils.prices.fetch_prices import fetch_price_history
from src.data_extract.utils.prices.fetch_short_interest import fetch_short_interest
from src.data_extract.utils.prices.fetch_13f import fetch_13f
from src.data_extract.utils.fundamentals.fetch_macro import fetch_macro


class StepExtractPrices(Step):

    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)

    def run(self, tickers: list[str]) -> None:
      
        # prices + dividends come from ONE yfinance download (actions=True):
        # fetch_price_history writes both prices.parquet and dividends.parquet.
        fetch_price_history(self._context, tickers=tickers)
        fetch_short_interest(self._context, tickers=tickers)
        fetch_macro(self._context)

        # 13F institutional holdings (SEC bulk + OpenFIGI cusip map; slow one-off)
        fetch_13f(self._context)
