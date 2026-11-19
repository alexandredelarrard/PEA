"""
step_extract_prices.py  (src/data_extract/step_extract_prices.py)
-----------------------------------------------------------------
Price / stock-market data extraction:
  * price history (daily OHLCV), dividend and split ex-dates -- the EQUITY universe only,
    through one shared yfinance session -> `prices_splits`, `prices_dividends`, `prices`
  * macro / market series (SPY, VIX, oil, gold, energy, FX + FRED) -> `prices_macro`
  * earnings surprises -> `earnings_surprises` (`run_earnings_surprises`, after the prices)

Source dates and rewrite windows belong to data.yml and Context.run.

Ownership and flow -- 13F, superinvestors, short interest -- are `StepExtractInstitutionals`;
the insider, 13D/13G and 8-K walks are `StepExtractEdgar`; fails-to-deliver is `StepExtractZips`.
"""

from src.data_extract.utils.prices.fetch_earnings_surprises import fetch_earnings_surprises
from src.data_extract.utils.prices.fetch_macro import fetch_macro
from src.data_extract.utils.prices.fetch_prices import YahooSession, fetch_dividends, fetch_price_actions, fetch_prices
from src.utils.step import Step


class StepExtractPrices(Step):
    def run_earnings_surprises(self, tickers: list[str]) -> None:
        """Check each ticker using the configured source window and shared run mode."""
        fetch_earnings_surprises(self._context, tickers=tickers)

    def run(self, tickers: list[str]) -> None:

        # One yfinance session serves the three fetchers. Splits come first: their windows are planned from the
        # stored `prices`, which `fetch_prices` then advances. A split or dividend found re-pulls that ticker in full.
        session = YahooSession(symbols=self._context.universe.trading_symbols)
        fetch_price_actions(self._context, session, tickers=tickers)
        fetch_dividends(self._context, session, tickers=tickers)
        fetch_prices(self._context, session, tickers=tickers)

        # MARKET + MACRO series -> `prices_macro`: the yfinance legs (SPY / VIX / oil / gold
        # / energy / FX, close only) plus the FRED levels and the derived spreads and 10Y
        # TODO: 5 days off because FRED only downloads weekly -> use yfinance
        # 2003 for breakeaven 10Y, oil/gold/fx 2001, rest is 1995
        fetch_macro(self._context)

        self.run_earnings_surprises(tickers)
