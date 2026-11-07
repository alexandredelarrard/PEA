"""Bounded, read-only loading of the saved market observations and quantity basis."""

from __future__ import annotations

from typing import Any

import pandas as pd
from omegaconf import DictConfig

from src.constants.constants_price import MACRO_MARKET_SERIES
from src.context import Context
from src.data_store.schema import Tables
from src.portfolio.utils.contracts import MarketInputs, Settings


def _wide(frame: pd.DataFrame, field: str) -> pd.DataFrame:
    return frame.pivot(index="date", columns="ticker", values=field).sort_index().astype(float)


def load_market(context: Context, config: DictConfig, settings: Settings, metadata: dict[str, Any]) -> MarketInputs:
    """Read the saved quotes, total returns and forward quantity basis `security_basis` F."""
    names = context.universe.tickers
    end = settings.end if settings.end is not None else pd.Timestamp.today().normalize()

    # Calendar-day buffer covers the configured full trading-session windows without downloading history.
    since = settings.start - pd.Timedelta(days=3 * max(settings.momentum_window, settings.risk_window) + 30)

    # Load published prices with their saved quantity basis once.
    df_prices = context.store.load(
        Tables.cube_part_prices,
        columns=["date", "ticker", "close_pit", "close_total", "security_basis"],
        where={"ticker": names},
        since=since,
        until=end,
    )
    df_macro = context.store.load(
        Tables.prices_macro, columns=["date", "ticker", "close"], where={"ticker": MACRO_MARKET_SERIES}, since=since, until=end
    )
    sectors = pd.Series(context.universe.sector_map, name="sector")

    if df_macro is None or df_macro.empty:
        raise ValueError("No stored S&P total-return proxy")
    if df_prices is None or df_prices.empty:
        raise ValueError("No published PIT prices for the backtest; rebuild the price cube")

    # Normalize the published observations without changing source support.
    df_macro = df_macro.assign(date=pd.to_datetime(df_macro.date))
    df_prices = df_prices.assign(date=pd.to_datetime(df_prices.date))

    # The backtest stops at the last session where stocks and the S&P benchmark are both stored.
    last = min(df_prices.date.max(), df_macro.date.max())
    context.log.info("Backtest data frontier %s (prices %s, benchmark %s)", last.date(), df_prices.date.max().date(), df_macro.date.max().date())
    df_macro, df_prices = (frame[frame.date <= last] for frame in (df_macro, df_prices))

    # Saved quote and basis holes remain unknown; F only changes holding units.
    df_quotes = _wide(df_prices, "close_pit")
    df_returns = _wide(df_prices, "close_total").pct_change(fill_method=None)
    df_basis = _wide(df_prices, "security_basis")

    # Align the independent market and current sector inputs to the selected securities.
    market = df_macro.set_index("date")["close"].sort_index().astype(float)
    sectors = sectors.reindex(df_quotes.columns)
    return MarketInputs(pd.DataFrame(), df_quotes, df_returns, df_basis, market, sectors, metadata)
