"""
step_extract_fundamentals.py  (src/data_extract/step_extract_fundamentals.py)
-----------------------------------------------------------------------------
Fundamentals / financials extraction:
  * company fundamentals (balance sheet, income, cash flow)
  * earnings surprises
  * macro series (economy-wide context)
"""
from omegaconf import DictConfig

from src.context import Context
from src.utils.step import Step
from src.data_extract.utils.fundamentals.fetch_fundamentals import fetch_fundamentals
from src.data_extract.utils.fundamentals.fetch_earnings_surprises import fetch_earnings_surprises
from src.data_extract.utils.fundamentals.fetch_macro import fetch_macro


class StepExtractFundamentals(Step):

    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)

    def run(self, tickers: list[str]) -> None:
        fetch_fundamentals(self._context, tickers=tickers)
        fetch_earnings_surprises(self._context, tickers=tickers)
        fetch_macro(self._context)
