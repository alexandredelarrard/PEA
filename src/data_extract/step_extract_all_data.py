"""
step_extract_all_data.py  (src/data_extract/step_extract_all_data.py)
---------------------------------------------------------------------
Super step orchestrating the four data-extraction sub-steps. Resolves the
ticker universe once — from the `sp500_tickers` table (the single entry point;
seeded via the S&P 500 scraper only when empty) — and hands it to each sub-step:

  1. prices        — price history (+dividends), short interest, 13F holdings
  2. fundamentals  — fundamentals, earnings surprises, macro
  3. structure     — employees, management, DEF 14A governance, SEC filings
  4. behavioral    — Wikipedia pageviews (+Google Trends, news)
"""

from omegaconf import DictConfig

from src.data_store.schema import Tables
from src.context import Context
from src.utils.step import Step
from src.utils.universe import load_universe_tickers
from src.data_extract.utils.prices.fetch_tickers import get_sp500_tickers
from src.data_extract.transformers.step_extract_prices import StepExtractPrices
from src.data_extract.transformers.step_extract_fundamentals import StepExtractFundamentals
from src.data_extract.transformers.step_extract_structure import StepExtractStructure
from src.data_extract.transformers.step_extract_behavioral import StepExtractBehavioral

class StepExtractAllData(Step):

    def __init__(self, context: Context, config: DictConfig):

        super().__init__(context=context, config=config)
        
        self._prices = StepExtractPrices(context=context, config=config)
        self._fundamentals = StepExtractFundamentals(context=context, config=config)
        self._structure = StepExtractStructure(context=context, config=config)
        self._behavioral = StepExtractBehavioral(context=context, config=config)

    def _resolve_tickers(self) -> list[str]:
        if self._context.store.row_count(Tables.sp500_tickers) == 0:
            get_sp500_tickers(self._context)           
            self._log.info("Extracted ticker list")
        universe = load_universe_tickers(self._context)
        self._log.info(f"Equity universe: {len(universe)} tickers from {Tables.sp500_tickers}")
        return universe

    def run(self) -> None:
        tickers = self._resolve_tickers()

        # self._prices.run(tickers=tickers)
        self._structure.run(tickers=tickers)
        # self._fundamentals.run(tickers=tickers)
        # self._behavioral.run(tickers=tickers)
       
        