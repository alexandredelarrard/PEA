"""Company-structure extraction: DEF 14A governance (ECD XBRL plus the LLM pass), 10-K/10-Q narrative text,
and Item 5.07 shareholder-vote tallies. Downloading fetchers share the window resolved here; the vote parser
downloads nothing. `StepExtractInstitutionals` (which stores `sec_8k`) must run before this step.
"""

from omegaconf import DictConfig

from src.context import Context
from src.data_extract.utils.common.edgar_driver import run_edgar_fetch
from src.data_extract.utils.structure.def14a import fetch_def14a_llm
from src.data_extract.utils.structure.fetch_def14a_edgar import DEF14A_EDGAR_FETCH
from src.data_extract.utils.structure.fetch_filing_text import FILING_TEXT_FETCH
from src.data_extract.utils.structure.votes import fetch_8k_votes_llm
from src.utils.step import Step


class StepExtractStructure(Step):
    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)

    def run(self, tickers: list[str]) -> None:

        years_history = int(self._context.config.data_extract.years_history)

        run_edgar_fetch(self._context, tickers=tickers, years_history=years_history, fetch=FILING_TEXT_FETCH)

        fetch_def14a_llm(self._context, self._config, tickers=tickers)
        run_edgar_fetch(self._context, tickers=tickers, years_history=years_history, fetch=DEF14A_EDGAR_FETCH)

        # Last: it reads `sec_8k` Item 5.07 and the `def14a_*` tables (nominee roles), so both must be current.
        fetch_8k_votes_llm(self._context, self._config, tickers=tickers)
