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
from src.data_extract.utils.fundamentals.fetch_financial_notes import fetch_financial_notes


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

        # Footnote (notes) pension detail + note TEXT from the Financial Statement
        # AND Notes data sets -> notes_num / notes_text. Heavy (~26GB back-fill at
        # notes_years_history=15); own cache dir + config knob. (fetch_fails_to_deliver
        # lives in StepExtractPrices, alongside the other price/settlement signals.)
        fetch_financial_notes(self._context, tickers=tickers)
