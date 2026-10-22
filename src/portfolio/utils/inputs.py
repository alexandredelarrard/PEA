"""Bounded, read-only loading of existing market observations and quote bases."""

from __future__ import annotations

from typing import Any

import pandas as pd
from omegaconf import DictConfig

from src.constants.constants_price import MACRO_MARKET_SERIES
from src.context import Context
from src.data_store.schema import Tables
from src.portfolio.utils.contracts import MarketInputs, Settings
from src.utils.predecessor_series import load_level_actions
from src.utils.price_adjustments import load_bugfix
from src.utils.price_basis import load_share_basis
from src.utils.universe import load_universe_tickers


def _wide(frame: pd.DataFrame, field: str) -> pd.DataFrame:
    return frame.pivot(index="date", columns="ticker", values=field).sort_index().astype(float)


def load_market(context: Context, config: DictConfig, settings: Settings, metadata: dict[str, Any]) -> MarketInputs:
    """Reuse stored split/total-return prices and the shared corporate-action resolver."""
    names = load_universe_tickers(context)
    end = settings.end if settings.end is not None else pd.Timestamp.today().normalize()
    # Calendar-day buffer covers the configured full trading-session windows without downloading history.
    since = settings.start - pd.Timedelta(days=3 * max(settings.momentum_window, settings.risk_window) + 30)
    prices = context.store.load(
        Tables.prices, columns=["date", "ticker", "close_split", "close_total"], where={"ticker": names}, since=since, until=end
    )
    if prices is None or prices.empty:
        raise ValueError("No stored stock prices for the backtest")
    prices = prices.assign(date=pd.to_datetime(prices.date))
    level_table = Tables.cube if "level_factor" in context.store.columns(Tables.cube) else Tables.cube_part_prices
    levels = context.store.load(level_table, columns=["date", "ticker", "level_factor"], where={"ticker": names}, since=since, until=end)
    if levels is None or levels.empty:
        raise ValueError("Missing stored level factors; quoted execution prices cannot be reconstructed")
    levels = levels.assign(date=pd.to_datetime(levels.date))
    prices = prices.merge(levels, on=["date", "ticker"], how="left", validate="one_to_one")
    actions = load_level_actions(context, names)
    splits = context.store.load(Tables.prices_splits, columns=["date", "ticker", "ratio"], where={"ticker": names}, optional=True)
    basis = load_share_basis(context, names, load_bugfix(context.config_dir), actions, splits, quotes=prices)
    vendor = basis.factor(prices.ticker, prices.date, "vendor")
    units = basis.factor(prices.ticker, prices.date, "units")
    prices["quote"] = prices.close_split * prices.level_factor * vendor * units
    prices["share_basis"] = basis.factor(prices.ticker, prices.date, "actual") * units
    quotes = _wide(prices, "quote")
    returns = _wide(prices, "close_total").pct_change(fill_method=None)
    macro = context.store.load(
        Tables.prices_macro, columns=["date", "ticker", "close"], where={"ticker": MACRO_MARKET_SERIES}, since=since, until=end
    )
    if macro is None or macro.empty:
        raise ValueError("No stored S&P total-return proxy")
    macro = macro.assign(date=pd.to_datetime(macro.date))
    market = macro.set_index("date")["close"].sort_index().astype(float)
    universe = context.store.load(Tables.sp500_tickers, columns=["ticker", "sector"], where={"ticker": names})
    if universe is None or universe.empty:
        raise ValueError("No current sector classification")
    sectors = universe.set_index("ticker")["sector"].reindex(quotes.columns)
    return MarketInputs(pd.DataFrame(), quotes, returns, _wide(prices, "share_basis"), market, sectors, metadata)
