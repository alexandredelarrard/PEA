"""
step_extract_structure.py  (src/data_extract/step_extract_structure.py)
-----------------------------------------------------------------------
Company-structure extraction (governance):
  * DEF 14A governance      (directors, compensation, ownership, executive pay;
                             LLM-parsed — this fully replaces the old EDGAR
                             officer/insider regex extraction)

Employee counts USED to be extracted here from 10-K body text into their own
`employees_history` table. They are now a `fundamentals_facts` field
(`fundamentals_employees.py`), parsed from the same 10-K the fundamentals walk
already opens, and surface as the `employees` column of `fundamentals_history`.

The sub-fetcher discovers its SEC filings on demand via
`edgar_fillings.list_filings` (full 15y+ history) -- there is no separate
filing-index download step.
"""

from omegaconf import DictConfig

from src.context import Context
from src.utils.step import Step
from src.data_extract.utils.structure.fetch_def14a_llm import fetch_def14a_llm


class StepExtractStructure(Step):

    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)

    def run(self, tickers: list[str]) -> None:
        fetch_def14a_llm(self._context, tickers=tickers,
                        model=self._config.data_extract.llm_model)
