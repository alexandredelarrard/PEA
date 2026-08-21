"""
step_extract_prices.py  (src/data_extract/step_extract_prices.py)
-----------------------------------------------------------------
Price / stock-market data extraction:
  * price history (daily OHLCV) -- the EQUITY universe only
  * dividends (ex-dates; its own fetcher, own resume window)
  * macro / market series (SPY, VIX, oil, gold, energy, FX + FRED) -> `prices_macro`
  * short interest (FINRA RegSHO short volume)
  * fails-to-deliver (SEC settlement fails)
  * 13F institutional holdings

The two windows live here, side by side: equities get `years_history`, the macro table gets
the deeper `macro_years_history` its sleeve backtests need. Both are passed INTO the
fetchers rather than read from config inside them.
"""
from omegaconf import DictConfig

from src.context import Context
from src.utils.step import Step
from src.data_extract.utils.prices.fetch_prices import fetch_price_history
from src.data_extract.utils.prices.fetch_dividends import fetch_dividends
from src.data_extract.utils.prices.fetch_short_interest import fetch_short_interest
from src.data_extract.utils.prices.fetch_fails_to_deliver import fetch_fails_to_deliver
from src.data_extract.utils.prices.fetch_superinvestors import build_superinvestors_json
from src.data_extract.utils.prices.fetch_macro import fetch_macro
from src.data_extract.utils.prices.fetch_13f import fetch_13f


class StepExtractPrices(Step):

    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)
        self.config = self._context.config

    def run(self, tickers: list[str]) -> None:

        years_history = self.config.data_extract.years_history
        years_macro = self.config.data_extract.macro_years_history

        # Prices and dividends are separate fetchers with separate resume windows
        # (daily bars vs quarterly ex-dates). Both get the EQUITY universe only
        fetch_price_history(self._context, tickers=tickers, years_history=years_history)
        fetch_dividends(self._context, tickers=tickers, years_history=years_history)

        # MARKET + MACRO series -> `prices_macro`: the yfinance legs (SPY / VIX / oil / gold
        # / energy / FX, close only) plus the FRED levels and the derived spreads and 10Y
        # TODO: 5 days off because FRED only downloads weekly -> use yfinance 
        fetch_macro(self._context, years_history=years_macro)

        # shorting stock 
        fetch_short_interest(self._context, tickers=tickers, years_history=years_history)

        # 13F institutional holdings (edgartools by filing date + OpenFIGI cusip map). Resumes
        # from max(filing_date) in sec13f_hr, so a routine run reads only the new filings.
        fetch_13f(self._context, tickers=tickers, years_history=years_history)

        # failing to give a stock in time
        fetch_fails_to_deliver(self._context, tickers=tickers, years_history=years_history)

        # Superinvestors roster: curated top managers (Dataroma) -> CIK subset JSON,
        build_superinvestors_json(self._context)