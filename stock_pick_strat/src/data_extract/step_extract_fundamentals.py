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
from src.data_extract.utils.prices.fetch_insider_transactions import fetch_insider_transactions
from src.data_extract.utils.fundamentals.fetch_financial_statements import fetch_financial_statements


class StepExtractFundamentals(Step):

    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)

    def run(self, tickers: list[str]) -> None:
        fetch_fundamentals(self._context, tickers=tickers)
        fetch_earnings_surprises(self._context, tickers=tickers)

        # SEC bulk quarterly data sets (zips cached under data/sec_bulk_cache/,
        # incremental by quarter + new-ticker back-fill -- see the fetchers).
        # Pension facts from the Financial Statement Data Sets (num/sub XBRL):
        fetch_financial_statements(self._context, tickers=tickers)

        # Officer / director / 10%-owner transactions from the Insider Transactions sets:
        fetch_insider_transactions(self._context, tickers=tickers)

        # TODO: fail to deliver stock
        # https://www.sec.gov/data-research/sec-markets-data/fails-deliver-data

        # TODO: financial-statement NOTES data sets (footnote PBO / plan assets /
        # funded status) for fuller pension detail -- heavier (~1GB+/qtr) follow-up.
        # https://www.sec.gov/data-research/sec-markets-data/financial-statement-notes-data-sets
