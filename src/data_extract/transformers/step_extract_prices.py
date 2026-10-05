"""
step_extract_prices.py  (src/data_extract/step_extract_prices.py)
-----------------------------------------------------------------
Price / stock-market data extraction:
  * price history (daily OHLCV), dividend and split ex-dates -- the EQUITY universe only,
    from one yfinance download per window group -> `prices`, `prices_dividends`, `prices_splits`
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
from src.data_extract.utils.prices.fetch_macro import fetch_macro
from src.data_extract.utils.prices.fetch_prices import fetch_prices_and_actions
from src.utils.step import Step


class StepExtractPrices(Step):
    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)
        self.config = self._context.config

    def run(self, tickers: list[str]) -> None:

        years_history = self.config.data_extract.years_history
        years_macro = self.config.data_extract.macro_years_history

        # Prices, dividends and splits share one download; a split found in it re-pulls that ticker in full.
        fetch_prices_and_actions(self._context, tickers=tickers, years_history=years_history)

        # MARKET + MACRO series -> `prices_macro`: the yfinance legs (SPY / VIX / oil / gold
        # / energy / FX, close only) plus the FRED levels and the derived spreads and 10Y
        # TODO: 5 days off because FRED only downloads weekly -> use yfinance
        # 2003 for breakeaven 10Y, oil/gold/fx 2001, rest is 1995
        fetch_macro(self._context, years_history=years_macro)
