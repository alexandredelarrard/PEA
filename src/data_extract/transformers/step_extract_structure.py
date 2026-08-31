"""
step_extract_structure.py (src/data_extract/transformers/step_extract_structure.py)
-----------------------------------------------------------------------------------
Company-structure extraction: 8-K events, SC 13D activist stakes, DEF 14A governance
(deterministic via edgartools, plus the LLM pass for narrative fields) and 10-K/10-Q
narrative text. The window is resolved here and passed to every fetcher, so all four
discover filings over the same history.
"""

from omegaconf import DictConfig

from src.context import Context
from src.data_extract.utils.structure.fetch_13d_edgar import fetch_13d_edgar
from src.data_extract.utils.structure.fetch_8k_edgar import fetch_8k_edgar
from src.data_extract.utils.structure.fetch_def14a_edgar import fetch_def14a_edgar
from src.data_extract.utils.structure.fetch_def14a_llm import fetch_def14a_llm
from src.data_extract.utils.structure.fetch_filing_text import fetch_filing_text
from src.utils.step import Step


class StepExtractStructure(Step):

    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)

    def run(self, tickers: list[str]) -> None:

        years_history = int(self._context.config.data_extract.years_history)

        # fetch_8k_edgar(self._context, tickers=tickers, years_history=years_history)
        # fetch_13d_edgar(self._context, tickers=tickers, years_history=years_history)
        # fetch_filing_text(self._context, tickers=tickers, years_history=years_history)

        fetch_def14a_llm(self._context, tickers=tickers,
                         model=self._config.data_extract.llm_model)
        # fetch_def14a_edgar(self._context, tickers=tickers, years_history=years_history)
