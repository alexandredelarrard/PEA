"""
step_extract_fundamentals.py  (src/data_extract/step_extract_fundamentals.py)
-----------------------------------------------------------------------------
Fundamentals / financials extraction:
  * company fundamentals (balance sheet, income, cash flow)
  * earnings surprises
  * macro series (economy-wide context)

Disabled (FMP paid API / no historical version yet): analyst grades, analyst
actions, executive compensation, FMP estimates, yfinance analyst estimates.
"""
from omegaconf import DictConfig

from src.context import Context
from src.utils.step import Step
from src.data_extract.utils.fundamentals.fetch_fundamentals import fetch_fundamentals
from src.data_extract.utils.fundamentals.fetch_earnings_surprises import fetch_earnings_surprises
from src.data_extract.utils.fundamentals.fetch_macro import fetch_macro
from src.data_extract.utils.fundamentals.fetch_analyst_estimates import fetch_analyst_estimates
from src.data_extract.utils.fundamentals.fetch_fmp_history import (
    fetch_analyst_grades,
    fetch_analyst_actions,
    fetch_exec_comp,
    fetch_estimates,
)


class StepExtractFundamentals(Step):

    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)

    def run(self, tickers: list[str]) -> None:
        fetch_fundamentals(self._context, tickers=tickers)
        fetch_earnings_surprises(self._context, tickers=tickers)
        fetch_macro(self._context)

        # no historical version yet - FMP API paying
        # fetch_analyst_grades(self._context, tickers=tickers)
        # fetch_analyst_actions(self._context, tickers=tickers)
        # fetch_exec_comp(self._context, tickers=tickers)
        # fetch_estimates(self._context, tickers=tickers)
        # fetch_analyst_estimates(self._context, tickers=tickers)
