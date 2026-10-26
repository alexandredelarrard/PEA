"""Bounded, read-only loading of existing market observations and quote bases."""

from __future__ import annotations

from typing import Any

import pandas as pd
from omegaconf import DictConfig

from src.constants.constants import SHARADAR_ACTION_SPINOFF, SHARADAR_ACTION_SPLIT
from src.constants.constants_price import MACRO_MARKET_SERIES
from src.context import Context
from src.data_store.schema import Tables
from src.portfolio.utils.contracts import MarketInputs, Settings
from src.utils.cutover_continuity import predecessor_actions
from src.utils.predecessor_series import load_predecessor_series
from src.utils.price.adjustments import load_bugfix, quantity_basis
from src.utils.price.levels import security_quantity_basis
from src.utils.universe import load_universe_tickers


def _wide(frame: pd.DataFrame, field: str) -> pd.DataFrame:
    return frame.pivot(index="date", columns="ticker", values=field).sort_index().astype(float)


def load_market(context: Context, config: DictConfig, settings: Settings, metadata: dict[str, Any]) -> MarketInputs:
    """Read published quotes and returns with the independent holding-quantity basis."""
    names = load_universe_tickers(context)
    end = settings.end if settings.end is not None else pd.Timestamp.today().normalize()
    series = load_predecessor_series(context, names)
    action_names = sorted(set(names) | {item.vendor_ticker for item in series})

    # Calendar-day buffer covers the configured full trading-session windows without downloading history.
    since = settings.start - pd.Timedelta(days=3 * max(settings.momentum_window, settings.risk_window) + 30)

    # Load published prices and complete sparse quantity events once each.
    df_prices = context.store.load(
        Tables.cube_part_prices, columns=["date", "ticker", "close_pit", "close_total"], where={"ticker": names}, since=since, until=end
    )
    df_splits = context.store.load(Tables.prices_splits, columns=["date", "ticker", "ratio"], where={"ticker": names}, optional=True)
    df_actions = context.store.load(
        Tables.sharadar_actions,
        columns=["ticker", "date", "action", "value"],
        where={"ticker": action_names, "action": [SHARADAR_ACTION_SPLIT, SHARADAR_ACTION_SPINOFF]},
        optional=True,
    )
    df_macro = context.store.load(
        Tables.prices_macro, columns=["date", "ticker", "close"], where={"ticker": MACRO_MARKET_SERIES}, since=since, until=end
    )
    df_universe = context.store.load(Tables.sp500_tickers, columns=["ticker", "sector"], where={"ticker": names})

    if df_universe is None or df_universe.empty:
        raise ValueError("No current sector classification")
    if df_macro is None or df_macro.empty:
        raise ValueError("No stored S&P total-return proxy")
    if df_prices is None or df_prices.empty:
        raise ValueError("No published PIT prices for the backtest; rebuild the price cube")

    # Normalize dates and preserve the difference between absent and empty event feeds.
    df_macro = df_macro.assign(date=pd.to_datetime(df_macro.date))
    df_prices = df_prices.assign(date=pd.to_datetime(df_prices.date))
    if df_splits is None and context.store.exists(Tables.prices_splits):
        df_splits = pd.DataFrame(columns=["ticker", "date", "ratio"])
    if df_actions is None and context.store.exists(Tables.sharadar_actions):
        df_actions = pd.DataFrame(columns=["ticker", "date", "action", "value"])
    if df_actions is not None:
        df_owners = df_actions.loc[df_actions["ticker"].isin({item.vendor_ticker for item in series})]
        df_actions = predecessor_actions(df_actions.loc[df_actions["ticker"].isin(names)], df_owners, series)

    # Saved quote holes remain unknown; quantity events only change holding units.
    basis = quantity_basis(df_actions, df_splits, load_bugfix(context.config_dir), names)
    df_quotes = _wide(df_prices, "close_pit")
    df_returns = _wide(df_prices, "close_total").pct_change(fill_method=None)
    df_basis = security_quantity_basis(basis, pd.DatetimeIndex(df_quotes.index), df_quotes.columns)

    # Align the independent market and current sector inputs to the selected securities.
    market = df_macro.set_index("date")["close"].sort_index().astype(float)
    sectors = df_universe.set_index("ticker")["sector"].reindex(df_quotes.columns)
    return MarketInputs(pd.DataFrame(), df_quotes, df_returns, df_basis, market, sectors, metadata)
