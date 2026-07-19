"""
step_extract_structure.py  (src/data_extract/step_extract_structure.py)
-----------------------------------------------------------------------
Company-structure extraction (workforce, management, governance, filings):
  * employee counts        (10-K text, EDGAR)
  * management / officers   (10-K + DEF 14A, EDGAR)
  * DEF 14A governance      (directors, compensation, ownership; LLM-parsed)
  * SEC filings index
"""
from omegaconf import DictConfig

from src.context import Context
from src.utils.step import Step
from src.data_extract.utils.structure.fetch_employees_edgar import fetch_employees_edgar
from src.data_extract.utils.structure.fetch_management_edgar import fetch_management_edgar
from src.data_extract.utils.structure.fetch_def14a_llm import fetch_def14a_llm
from src.data_extract.utils.structure.fetch_sec_filings import fetch_sec_filings


class StepExtractStructure(Step):

    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)

    def run(self, tickers: list[str]) -> None:
        # fetch_employees_edgar(self._context, tickers=tickers)
        # fetch_management_edgar(self._context, tickers=tickers)
        fetch_def14a_llm(self._context, tickers=tickers,
                        model=self._config.data_extract.llm_model)
        # fetch_sec_filings(self._context)
