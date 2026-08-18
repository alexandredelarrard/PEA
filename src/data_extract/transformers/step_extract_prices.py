"""
step_extract_prices.py  (src/data_extract/step_extract_prices.py)
-----------------------------------------------------------------
Price / stock-market data extraction:
  * price history (daily OHLCV)
  * dividends (ex-dates; its own fetcher, own resume window)
  * short interest (FINRA RegSHO short volume)
  * fails-to-deliver (SEC settlement fails)
  * 13F institutional holdings
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
from src.data_extract.utils.prices.fetch_macro_assets import fetch_macro_assets
from src.data_extract.utils.prices.fetch_13f import fetch_13f


class StepExtractPrices(Step):

    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)

    def run(self, tickers: list[str]) -> None:

        years_history = self._context.config.data_extract.years_history

        # Prices and dividends are separate fetchers with separate resume windows
        # (daily bars vs quarterly ex-dates). Dividends get the EQUITY universe only.
        fetch_price_history(self._context, tickers=tickers, years_history=years_history)
        fetch_dividends(self._context, tickers=tickers, years_history=years_history)

        # benchmark + commodity/FX OHLCV (for the market-beta + factor panel),
        others = list(self._context.config.data_extract.other_tickers)
        fetch_price_history(self._context, tickers=others, years_history=years_history)

        # MARKET + MACRO data — NOT part of the equity universe, no features built on
        # then FRED macro series (yields, VIX, credit spread, breakevens).
        # Long-history multi-asset ALLOCATION series (FRED, since ~1995): equity /
        # gold / 10Y bond TR / cash / FX for the risk-parity + trend sleeve backtest.
        fetch_macro(self._context)
        fetch_macro_assets(self._context)

        # 13F institutional holdings (SEC bulk + OpenFIGI cusip map; slow one-off)
        # TODO : update it to be daily extract from edgar tool  Form 3/4/5's own SUBMISSION.FILING_DATE
        # NOT to stay the quarterly zip download
        fetch_13f(self._context)

        # shorting stock 
        fetch_short_interest(self._context, tickers=tickers, years_history=years_history)

        # failing to give a stock in time
        fetch_fails_to_deliver(self._context, tickers=tickers)

        # Superinvestors roster: curated top managers (Dataroma) -> CIK subset JSON,
        # ranked by 13F AUM, for the elite "smart-money" features. Best-effort: an
        # external (Dataroma) failure must never break the price extraction.
        build_superinvestors_json(self._context)