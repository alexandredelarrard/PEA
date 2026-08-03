"""
step_extract_structure.py  (src/data_extract/step_extract_structure.py)
-----------------------------------------------------------------------
Company-structure extraction (governance):
  * DEF 14A governance      (directors, compensation, ownership, executive pay;
                             LLM-parsed — this fully replaces the old EDGAR
                             officer/insider regex extraction)
  * DEF 14A structured      (deterministic complement to the above: PEO/NEO
                             pay-vs-performance, CEO pay ratio, audit fees,
                             per-NEO/per-director compensation tables,
                             beneficial ownership, voting proposals -- via
                             edgartools' typed `ProxyStatement`, zero LLM cost;
                             see fetch_def14a_edgar.py's module docstring)

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
from src.data_extract.utils.structure.fetch_def14a_edgar import fetch_def14a_edgar
from src.data_extract.utils.structure.fetch_def14a_llm import fetch_def14a_llm
from src.data_extract.utils.structure.fetch_8k_edgar import fetch_8k_edgar
from src.data_extract.utils.structure.fetch_13d_edgar import fetch_13d_edgar
from src.data_extract.utils.structure.fetch_filing_text import fetch_filing_text

class StepExtractStructure(Step):

    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)

    def run(self, tickers: list[str]) -> None:

        # fetch 13d
        fetch_13d_edgar(self._context, tickers=tickers)

        # 8k edgar
        fetch_8k_edgar(self._context, tickers=tickers)

        # fetch filing text
        fetch_filing_text(self._context, tickers=tickers)

        # fetch def14a 
        fetch_def14a_edgar(self._context, tickers=tickers)

        # LLM + edgar 
        fetch_def14a_llm(self._context, tickers=tickers,
                                model=self._config.data_extract.llm_model)
        
