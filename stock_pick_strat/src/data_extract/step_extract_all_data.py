from omegaconf import DictConfig

from src.context import Context
from src.utils.step import Step
from src.data_extract.utils.fetch_prices import fetch_price_history, get_sp500_tickers
from src.data_extract.utils.fetch_fundamentals import fetch_fundamentals
from src.data_extract.utils.fetch_macro import fetch_macro
from src.data_extract.utils.fetch_news import fetch_news
from src.data_extract.utils.fetch_sec_filings import fetch_sec_filings
from src.data_extract.utils.fetch_earnings_surprises import fetch_earnings_surprises
from src.data_extract.utils.fetch_analyst_estimates import fetch_analyst_estimates
from src.data_extract.utils.fetch_management import fetch_management
from src.data_extract.utils.fetch_employees import fetch_employees
from src.data_extract.utils.fetch_fmp_history import (
    fetch_analyst_grades,
    fetch_analyst_actions,
    fetch_exec_comp,
    fetch_estimates,
)

class StepExtractAllData(Step):

    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)

    def run(self):

        tickers = get_sp500_tickers(self._context)
        tickers = tickers + self._config.data_extract.other_tickers
        
        fetch_price_history(self._context, tickers=tickers)
        fetch_fundamentals(self._context, tickers=tickers)
        fetch_macro(self._context)
        fetch_earnings_surprises(self._context, tickers=tickers)
        fetch_employees(self._context, tickers=tickers)
        fetch_analyst_grades(self._context, tickers=tickers)
        fetch_analyst_actions(self._context, tickers=tickers)
        fetch_exec_comp(self._context, tickers=tickers)
        fetch_estimates(self._context, tickers=tickers)
        fetch_sec_filings(self._context)

        # no historical version yet
        # fetch_news(self._context, tickers=tickers)
        # fetch_analyst_estimates(self._context, tickers=tickers)
        # fetch_management(self._context, tickers=tickers)
