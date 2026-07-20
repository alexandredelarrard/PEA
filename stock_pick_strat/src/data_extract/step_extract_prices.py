"""
step_extract_prices.py  (src/data_extract/step_extract_prices.py)
-----------------------------------------------------------------
Price / stock-market data extraction:
  * price history (+ dividends, from the same yfinance download)
  * short interest (FINRA RegSHO short volume)
  * fails-to-deliver (SEC settlement fails)
  * 13F institutional holdings
"""
from omegaconf import DictConfig

from src.context import Context
from src.utils.step import Step
from src.data_extract.utils.prices.fetch_prices import fetch_price_history, fetch_market_prices
from src.data_extract.utils.prices.fetch_short_interest import fetch_short_interest
from src.data_extract.utils.prices.fetch_fails_to_deliver import fetch_fails_to_deliver
from src.data_extract.utils.prices.fetch_13f import fetch_13f
from src.data_extract.utils.prices.fetch_superinvestors import build_superinvestors_json
from src.data_extract.utils.fundamentals.fetch_macro import fetch_macro


class StepExtractPrices(Step):

    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)

    def run(self, tickers: list[str]) -> None:
        # `tickers` is the EQUITY universe (sp500_tickers) only — the benchmark/macro
        # price series (other_tickers) are handled in the market/macro pull below.

        # prices + dividends come from ONE yfinance download (actions=True):
        # fetch_price_history writes both the prices and dividends tables.
        fetch_price_history(self._context, tickers=tickers)
        fetch_short_interest(self._context, tickers=tickers)
        fetch_fails_to_deliver(self._context, tickers=tickers)

        # MARKET + MACRO data — NOT part of the equity universe, no features built on
        # them: benchmark + commodity/FX OHLCV (for the market-beta + factor panel),
        # then FRED macro series (yields, VIX, credit spread, breakevens).
        fetch_market_prices(self._context)
        fetch_macro(self._context)

        # 13F institutional holdings (SEC bulk + OpenFIGI cusip map; slow one-off)
        fetch_13f(self._context)

        # Superinvestors roster: curated top managers (Dataroma) -> CIK subset JSON,
        # ranked by 13F AUM, for the elite "smart-money" features. Best-effort: an
        # external (Dataroma) failure must never break the price extraction.
        try:
            build_superinvestors_json(self._context)
        except Exception as e:                                   # noqa: BLE001
            self._log.warning("Superinvestors roster refresh skipped: %s", e)
