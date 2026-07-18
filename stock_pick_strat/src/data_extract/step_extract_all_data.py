"""
step_extract_all_data.py  (src/data_extract/step_extract_all_data.py)
---------------------------------------------------------------------
Super step orchestrating the four data-extraction sub-steps. Resolves the
ticker universe once and hands it to each sub-step in turn:

  1. prices        — price history (+dividends), short interest, 13F holdings
  2. fundamentals  — fundamentals, earnings surprises, macro
  3. structure     — employees, management, DEF 14A governance, SEC filings
  4. behavioral    — Wikipedia pageviews (+Google Trends, news)
"""

from omegaconf import DictConfig

from src.context import Context
from src.utils.step import Step
from src.data_extract.utils.prices.fetch_prices import get_sp500_tickers
from src.data_extract.step_extract_prices import StepExtractPrices
from src.data_extract.step_extract_fundamentals import StepExtractFundamentals
from src.data_extract.step_extract_structure import StepExtractStructure
from src.data_extract.step_extract_behavioral import StepExtractBehavioral


class StepExtractAllData(Step):

    def __init__(self, context: Context, config: DictConfig):

        super().__init__(context=context, config=config)
        
        self._prices = StepExtractPrices(context=context, config=config)
        self._fundamentals = StepExtractFundamentals(context=context, config=config)
        self._structure = StepExtractStructure(context=context, config=config)
        self._behavioral = StepExtractBehavioral(context=context, config=config)

    def _resolve_tickers(self) -> list[str]:
        tickers = get_sp500_tickers(self._context)
        return tickers + self._config.data_extract.other_tickers

    def run(self) -> None:
        tickers = self._resolve_tickers()

        # self._prices.run(tickers=tickers)
        self._fundamentals.run(tickers=tickers)
        # self._structure.run(tickers=tickers)
        # self._behavioral.run(tickers=tickers)

        # TODO: extract disclosures from SEC filings and notes 
        # https://www.sec.gov/data-research/sec-markets-data/financial-statement-notes-data-sets

        # TODO: fail to deliver stock 
        # https://www.sec.gov/data-research/sec-markets-data/fails-deliver-data     

        # TODO: senior executive & insiders transactions
        # https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets

    def analysis(self) -> None:
        import pandas as pd
        
        df = pd.read_sql('fundamentals_history', self._context.store.engine)
        x = df.isnull().groupby(df['ticker']).sum()
        x['max'] = x.max(axis=1)
        for col in x.columns : 
            x[col] /= (0.001+ x['max'])
        print(x.mean(axis=1).mean())
        x.to_csv('sanity_check_financials_missing_infos.csv')