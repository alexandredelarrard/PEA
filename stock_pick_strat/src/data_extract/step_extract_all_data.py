from omegaconf import DictConfig

from src.context import Context
from src.utils.step import Step
from src.data_extract.utils.fetch_prices import fetch_price_history, get_sp500_tickers
from src.data_extract.utils.fetch_dividends import fetch_dividends
from src.data_extract.utils.fetch_wiki_pageviews import fetch_wiki_pageviews
from src.data_extract.utils.fetch_google_trends import fetch_google_trends
from src.data_extract.utils.fetch_13f import fetch_13f
from src.data_extract.utils.fetch_short_interest import fetch_short_interest
from src.data_extract.utils.fetch_fundamentals import fetch_fundamentals
from src.data_extract.utils.fetch_macro import fetch_macro
from src.data_extract.utils.fetch_news import fetch_news
from src.data_extract.utils.fetch_sec_filings import fetch_sec_filings
from src.data_extract.utils.fetch_earnings_surprises import fetch_earnings_surprises
from src.data_extract.utils.fetch_analyst_estimates import fetch_analyst_estimates
from src.data_extract.utils.fetch_management_edgar import fetch_management_edgar
from src.data_extract.utils.fetch_employees_edgar import fetch_employees_edgar
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
        fetch_dividends(self._context, tickers=tickers)
        fetch_short_interest(self._context, tickers=tickers)
        fetch_fundamentals(self._context, tickers=tickers)
        fetch_macro(self._context)
        fetch_earnings_surprises(self._context, tickers=tickers)
        fetch_employees_edgar(self._context, tickers=tickers)
        fetch_management_edgar(self._context, tickers=tickers)
        fetch_sec_filings(self._context)

        # Retail-attention alt-data (slow / rate-limited -> opt-in; the cube step
        # picks up whatever parquet exists). Wikipedia is reliable & daily; Google
        # Trends needs `pip install pytrends` and self-skips if absent.
        fetch_wiki_pageviews(self._context, tickers=tickers)
        fetch_google_trends(self._context, tickers=tickers)
        # 13F institutional holdings (SEC bulk + OpenFIGI cusip map; slow one-off)
        fetch_13f(self._context)

        # no historical version yet - FMP API paying
        # fetch_analyst_grades(self._context, tickers=tickers)
        # fetch_analyst_actions(self._context, tickers=tickers)
        # fetch_exec_comp(self._context, tickers=tickers)
        # fetch_estimates(self._context, tickers=tickers)
        # fetch_news(self._context, tickers=tickers)
        # fetch_analyst_estimates(self._context, tickers=tickers)
        # fetch_management(self._context, tickers=tickers)
