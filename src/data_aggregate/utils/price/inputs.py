"""Visible scoped loading for one coherent historical price candidate."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from src.constants.constants import SHARADAR_ACTION_SPINOFF, SHARADAR_ACTION_SPLIT
from src.constants.constants_price import MACRO_MARKET_SERIES
from src.context import Context
from src.data_aggregate.utils.common.peers_io import load_peers
from src.data_store.schema import Tables
from src.utils.cutover_continuity import PredecessorSeries, predecessor_actions, predecessor_series, register_windows
from src.utils.macro import load_macro_series
from src.utils.predecessor_series import load_vendor_series, with_overrides

logger = logging.getLogger(__name__)
PRICE_COLS = ["date", "open", "high", "low", "close_split", "close_total", "volume", "ticker"]


@dataclass(frozen=True)
class PriceInputs:
    """Observed bars and sparse evidence held only for the current price run."""

    bars: pd.DataFrame
    market: pd.Series
    peers: dict[str, dict[str, float]]
    vendor: pd.DataFrame | None
    shares: pd.DataFrame | None
    actions: pd.DataFrame | None
    splits: pd.DataFrame | None
    predecessors: tuple[PredecessorSeries, ...]
    unsupported_units: frozenset[str]


def _identity_inputs(context: Context, tickers: list[str]) -> tuple[tuple[PredecessorSeries, ...], pd.DataFrame | None]:
    """Read predecessor identity metadata once before combining source scopes."""
    store = context.store
    columns = ["canonical_ticker", "cik", "role", "valid_from", "valid_to", "sources"]
    df_lineage = (
        store.load(Tables.entity_lineage, columns=columns, where={"role": "cik_window"}, optional=True)
        if "role" in store.columns(Tables.entity_lineage)
        else None
    )
    security_columns = ["ticker", "secfilings", "lastquarter"]
    if "category" in store.columns(Tables.sharadar_tickers):
        security_columns.append("category")
    df_security = store.load(Tables.sharadar_tickers, columns=security_columns, optional=True)
    derived = () if df_lineage is None or df_security is None else predecessor_series(df_security, register_windows(df_lineage), tickers)
    declared = tuple(item for item in load_vendor_series(context.config_dir) if item.ticker in tickers)
    return with_overrides(derived, declared), df_security


def _unsupported_units(df_actions: pd.DataFrame | None, df_security: pd.DataFrame | None, tickers: list[str], bugfix: dict) -> frozenset[str]:
    """Refuse unestablished ADS units without inventing an ordinary-to-quote ratio."""
    unknown: set[str] = set()
    if df_actions is not None:
        unknown.update(df_actions.loc[df_actions["action"].eq("adrratiosplit") & df_actions["ticker"].isin(tickers), "ticker"])
    if df_security is not None and "category" in df_security:
        adr = df_security["category"].astype("string").str.contains(r"\bADR\b|\bADS\b", case=False, na=False)
        unknown.update(df_security.loc[adr & df_security["ticker"].isin(tickers), "ticker"])
    unknown.difference_update(ticker for ticker, policy in bugfix.get("share_events", {}).items() if "quote_units" in policy)
    return frozenset(unknown)


def load_price_inputs(context: Context, tickers: list[str], bugfix: dict) -> PriceInputs:
    """Read projected sources once, retaining full actions, anchors and public share seeds."""
    store = context.store
    predecessors, df_security = _identity_inputs(context, tickers)
    action_tickers = sorted(set(tickers) | {item.vendor_ticker for item in predecessors})
    actions_present, splits_present = store.exists(Tables.sharadar_actions), store.exists(Tables.prices_splits)
    df_bars = store.load(Tables.prices, columns=PRICE_COLS, where={"ticker": tickers})
    market = load_macro_series(store, MACRO_MARKET_SERIES)
    df_actions = store.load(
        Tables.sharadar_actions,
        columns=["ticker", "date", "action", "value"],
        where={"ticker": action_tickers, "action": [SHARADAR_ACTION_SPLIT, SHARADAR_ACTION_SPINOFF, "adrratiosplit"]},
        optional=True,
    )
    df_splits = store.load(Tables.prices_splits, columns=["ticker", "date", "ratio"], where={"ticker": tickers}, optional=True)
    wanted = ["ticker", "date", "reportperiod", "sharesbas", "sharefactor", "price", "lastupdated"]
    available = set(store.columns(Tables.sharadar_fundamentals))
    df_vendor = store.load(
        Tables.sharadar_fundamentals,
        columns=[field for field in wanted if field in available] or wanted,
        where={"ticker": tickers, "dimension": "ARQ"},
        optional=True,
    )
    df_shares = store.load(
        Tables.fundamentals_history,
        columns=["ticker", "as_of", "fiscal_end", "sharesOutstanding", "sharesOutstandingPit"],
        where={"ticker": tickers},
        optional=True,
    )
    peers = load_peers(context)
    if df_bars is None or df_bars.empty or market is None:
        raise ValueError("Price creation requires observed Yahoo bars and the authoritative market calendar")
    if not peers:
        raise ValueError("Price creation requires an existing peer cache; run data_peers first")
    if df_actions is None and actions_present:
        df_actions = pd.DataFrame(columns=["ticker", "date", "action", "value"])
    if df_splits is None and splits_present:
        df_splits = pd.DataFrame(columns=["ticker", "date", "ratio"])
    unknown = _unsupported_units(df_actions, df_security, tickers, bugfix)
    return PriceInputs(df_bars, market, peers, df_vendor, df_shares, df_actions, df_splits, predecessors, unknown)


def merge_price_actions(inputs: PriceInputs, tickers: list[str]) -> pd.DataFrame | None:
    """Choose dated predecessor action owners from the already-loaded source frame."""
    if inputs.actions is None:
        return None
    df_actions = inputs.actions.loc[inputs.actions["action"].isin([SHARADAR_ACTION_SPLIT, SHARADAR_ACTION_SPINOFF])].copy()
    df_canonical = df_actions.loc[df_actions["ticker"].isin(tickers)].copy()
    df_owners = df_actions.loc[df_actions["ticker"].isin({item.vendor_ticker for item in inputs.predecessors})].copy()
    stored = set(df_owners["ticker"])
    for item in inputs.predecessors:
        if item.vendor_ticker not in stored:
            logger.warning(
                "level factor: no selected predecessor split/spinoff rows for %s of %s; canonical actions are kept", item.vendor_ticker, item.ticker
            )
    return predecessor_actions(df_canonical, df_owners, inputs.predecessors)
