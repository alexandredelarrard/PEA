"""
step_extract_prices.py  (src/data_extract/step_extract_prices.py)
-----------------------------------------------------------------
Price / stock-market data extraction:
  * price history (daily OHLCV), dividend and split ex-dates -- the EQUITY universe only,
    through one shared yfinance session -> `prices_splits`, `prices_dividends`, `prices`
  * macro / market series (SPY, VIX, oil, gold, energy, FX + FRED) -> `prices_macro`
  * earnings surprises -> `earnings_surprises` (`run_earnings_surprises`, after the prices)

The two windows live here, side by side: equities get `years_history`, the macro table gets
the deeper `macro_years_history` its sleeve backtests need. Both are passed INTO the
fetchers rather than read from config inside them.

Ownership and flow -- 13F, superinvestors, short interest -- are `StepExtractInstitutionals`;
the insider, 13D/13G and 8-K walks are `StepExtractEdgar`; fails-to-deliver is `StepExtractZips`.
"""

from omegaconf import DictConfig

from src.context import Context
from src.data_extract.utils.prices.fetch_earnings_surprises import fetch_earnings_surprises
from src.data_extract.utils.prices.fetch_macro import fetch_macro
from src.data_extract.utils.prices.fetch_prices import YahooSession, fetch_dividends, fetch_price_actions, fetch_prices
from src.utils.step import Step


class StepExtractPrices(Step):
    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)
        self.config = self._context.config

    def run_earnings_surprises(self, tickers: list[str]) -> None:
        """Earnings surprises over the equity window -> `earnings_surprises`."""
        fetch_earnings_surprises(self._context, tickers=tickers, years_history=int(self.config.data_extract.years_history))

    def run(self, tickers: list[str]) -> None:

        years_history = self.config.data_extract.years_history
        years_macro = self.config.data_extract.macro_years_history

        # One yfinance session serves the three fetchers. Splits come first: their windows are planned from the
        # stored `prices`, which `fetch_prices` then advances. A split or dividend found re-pulls that ticker in full.
        session = YahooSession()
        fetch_price_actions(self._context, session, tickers=tickers, years_history=years_history)
        fetch_dividends(self._context, session, tickers=tickers, years_history=years_history)
        fetch_prices(self._context, session, tickers=tickers, years_history=years_history)

        # MARKET + MACRO series -> `prices_macro`: the yfinance legs (SPY / VIX / oil / gold
        # / energy / FX, close only) plus the FRED levels and the derived spreads and 10Y
        # TODO: 5 days off because FRED only downloads weekly -> use yfinance
        # 2003 for breakeaven 10Y, oil/gold/fx 2001, rest is 1995
        fetch_macro(self._context, years_history=years_macro)

        self.run_earnings_surprises(tickers)
