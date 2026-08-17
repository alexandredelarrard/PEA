"""
step_extract_fundamentals.py  (src/data_extract/step_extract_fundamentals.py)
-----------------------------------------------------------------------------
Fundamentals / financials extraction:
  * company fundamentals (balance sheet, income, cash flow) -- edgartools per-filing
    walk (fetch_fundamentals_edgar.py) -> fundamentals_facts (raw, accession-grain,
    amendment-aware), then derived into the unchanged fundamentals_history shape
    (fundamentals_derive.py) for every existing downstream consumer.
  * earnings surprises
  * macro series (economy-wide context)
"""
from omegaconf import DictConfig

from src.context import Context
from src.utils.step import Step
from src.data_extract.utils.fundamentals.fetch_fundamentals_edgar import fetch_fundamentals_edgartools
from src.data_extract.utils.fundamentals.fundamentals_derive import rebuild_fundamentals_history
from src.data_extract.utils.fundamentals.fetch_earnings_surprises import fetch_earnings_surprises
from src.data_extract.utils.prices.fetch_insider_transactions import fetch_insider_transactions
from src.data_extract.utils.fundamentals.fetch_financial_notes import fetch_financial_notes


class StepExtractFundamentals(Step):

    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)

    def run(self, tickers: list[str]) -> None:

        # Fundamentals via edgartools: raw per-filing facts -> fundamentals_facts,
        # then derived into fundamentals_history
        fetch_fundamentals_edgartools(self._context, tickers=tickers)
        rebuild_fundamentals_history(self._context, tickers)

        # earnings surprises
        fetch_earnings_surprises(self._context, tickers=tickers)

        # Officer / director / 10%-owner transactions from the Insider Transactions sets:
        fetch_insider_transactions(self._context, tickers=tickers)

        # Footnote (notes) pension detail + note TEXT from the Financial Statement
        # AND Notes data sets -> notes_num / notes_text. Heavy (~26GB back-fill at
        # notes_years_history=15); own cache dir + config knob. (fetch_fails_to_deliver
        # lives in StepExtractPrices, alongside the other price/settlement signals.)
        fetch_financial_notes(self._context, tickers=tickers)
