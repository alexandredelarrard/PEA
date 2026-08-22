"""
step_extract_fundamentals.py  (src/data_extract/step_extract_fundamentals.py)
-----------------------------------------------------------------------------
Fundamentals / financials extraction:
  * company fundamentals (balance sheet, income, cash flow) -- REBUILD IN PROGRESS, see
    reports/planning/active-tasks/2026-08-21-fundamentals-rebuild-plan.md. The
    linkbase-driven fetcher (`fetch_fundamentals_sec.py`), the publication-event history
    build (`build_history.py`) and `FundamentalsValidator` are wired back in at Phase 3/5/7.
  * earnings surprises
  * insider transactions
  * footnote (notes) pension detail + note text
"""

from omegaconf import DictConfig

from src.context import Context
from src.utils.step import Step
from src.data_extract.utils.fundamentals.fetch_earnings_surprises import fetch_earnings_surprises
from src.data_extract.utils.prices.fetch_insider_transactions import fetch_insider_transactions
from src.data_extract.utils.fundamentals.fetch_financial_notes import fetch_financial_notes


class StepExtractFundamentals(Step):

    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)

    def run(self, tickers: list[str]) -> None:

        # earnings surprises
        fetch_earnings_surprises(self._context, tickers=tickers)

        # Officer / director / 10%-owner transactions from the Insider Transactions sets:
        fetch_insider_transactions(self._context, tickers=tickers)

        # Footnote (notes) pension detail + note TEXT from the Financial Statement
        # AND Notes data sets -> notes_num / notes_text. Heavy (~26GB back-fill at
        # notes_years_history=15); own cache dir + config knob. (fetch_fails_to_deliver
        # lives in StepExtractPrices, alongside the other price/settlement signals.)
        fetch_financial_notes(self._context, tickers=tickers)
