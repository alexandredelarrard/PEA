"""
step_extract_structure.py (src/data_extract/transformers/step_extract_structure.py)
-----------------------------------------------------------------------------------
Company-structure extraction: 8-K events, SC 13D activist stakes, DEF 14A governance
(deterministic via edgartools, plus the LLM pass for narrative fields), 10-K/10-Q
narrative text, and the Item 5.07 shareholder-vote tallies. The window is resolved here
and passed to every DOWNLOADING fetcher, so they discover filings over the same history;
the vote parser takes no window because it downloads nothing -- it reads the 8-K
narratives the first fetcher has already stored.
"""

from omegaconf import DictConfig

from src.context import Context
from src.data_extract.utils.structure.fetch_13d_edgar import fetch_13d_edgar
from src.data_extract.utils.structure.fetch_8k_edgar import fetch_8k_edgar
from src.data_extract.utils.structure.votes import fetch_8k_votes_llm
from src.data_extract.utils.structure.fetch_def14a_edgar import fetch_def14a_edgar
from src.data_extract.utils.structure.def14a import fetch_def14a_llm
from src.data_extract.utils.structure.fetch_filing_text import fetch_filing_text
from src.utils.step import Step


class StepExtractStructure(Step):

    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)

    def run(self, tickers: list[str]) -> None:

        years_history = int(self._context.config.data_extract.years_history)

        fetch_8k_edgar(self._context, tickers=tickers, years_history=years_history)
        fetch_13d_edgar(self._context, tickers=tickers, years_history=years_history)
        fetch_filing_text(self._context, tickers=tickers, years_history=years_history)

        fetch_def14a_llm(self._context, self._config, tickers=tickers)
        fetch_def14a_edgar(self._context, tickers=tickers, years_history=years_history)

        # LAST on purpose: it reads `sec_8k` (item 5.07) for its input and the three
        # `def14a_*` tables for the nominee role map, so both must be current first.
        fetch_8k_votes_llm(self._context, self._config, tickers=tickers)
