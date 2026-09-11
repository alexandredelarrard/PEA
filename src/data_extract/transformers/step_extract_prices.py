"""
step_extract_prices.py  (src/data_extract/step_extract_prices.py)
-----------------------------------------------------------------
Price / stock-market data extraction:
  * price history (daily OHLCV) -- the EQUITY universe only
  * dividends (ex-dates; its own fetcher, own resume window)
  * splits (ex-dates; same shape as dividends) -> `prices_splits`
  * macro / market series (SPY, VIX, oil, gold, energy, FX + FRED) -> `prices_macro`

The two windows live here, side by side: equities get `years_history`, the macro table gets
the deeper `macro_years_history` its sleeve backtests need. Both are passed INTO the
fetchers rather than read from config inside them.

Ownership and flow -- 13F, superinvestors, insiders, 13D, 8-K, short interest,
fails-to-deliver -- are NOT here: they are `StepExtractInstitutionals`, which mirrors the
cube's `institutionals` part.
"""
from omegaconf import DictConfig

from src.context import Context
from src.utils.step import Step
from src.data_extract.utils.prices.fetch_prices import fetch_price_history
from src.data_extract.utils.prices.fetch_dividends import fetch_dividends
from src.data_extract.utils.prices.fetch_splits import fetch_splits
from src.data_extract.utils.prices.fetch_macro import fetch_macro


class StepExtractPrices(Step):

    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)
        self.config = self._context.config

    def run(self, tickers: list[str]) -> None:

        years_history = self.config.data_extract.years_history
        years_macro = self.config.data_extract.macro_years_history

        # Splits FIRST: `fetch_price_history` reads `prices_splits` to decide which tickers
        # need their whole history re-pulled because a split restated it retroactively. Run
        # the other way round and a fresh split is only acted on the NEXT night.
        fetch_splits(self._context, tickers=tickers, years_history=years_history)

        # Prices and dividends are separate fetchers with separate resume windows
        # (daily bars vs quarterly ex-dates). Both get the EQUITY universe only
        fetch_price_history(self._context, tickers=tickers, years_history=years_history)
        fetch_dividends(self._context, tickers=tickers, years_history=years_history)

        # MARKET + MACRO series -> `prices_macro`: the yfinance legs (SPY / VIX / oil / gold
        # / energy / FX, close only) plus the FRED levels and the derived spreads and 10Y
        # TODO: 5 days off because FRED only downloads weekly -> use yfinance
        # 2003 for breakeaven 10Y, oil/gold/fx 2001, rest is 1995
        fetch_macro(self._context, years_history=years_macro)
