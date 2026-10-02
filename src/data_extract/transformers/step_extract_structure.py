"""
step_extract_structure.py (src/data_extract/transformers/step_extract_structure.py)
-----------------------------------------------------------------------------------
Company-structure extraction: DEF 14A governance (deterministic via edgartools, plus the LLM
pass for narrative fields), 10-K/10-Q narrative text, and the Item 5.07 shareholder-vote
tallies. The window is resolved here and passed to every DOWNLOADING fetcher, so they discover
filings over the same history; the vote parser takes no window because it downloads nothing.

⚠ `fetch_8k_votes_llm` reads the `sec_8k` narratives, which `StepExtractInstitutionals` now
owns and stores. That step must therefore run BEFORE this one -- see `StepExtractAllData`.
8-K events and SC 13D activist stakes moved there with it.
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

        # LAST on purpose: it reads `sec_8k` (item 5.07) for its input and the three
        # `def14a_*` tables for the nominee role map, so both must be current first.
        # `sec_8k` is filled by StepExtractInstitutionals, which runs before this step.
        fetch_8k_votes_llm(self._context, self._config, tickers=tickers)
