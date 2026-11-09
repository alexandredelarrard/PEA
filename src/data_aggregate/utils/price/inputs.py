"""Visible scoped loading for one coherent historical price candidate."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from src.constants.constants import SHARADAR_ACTION_SPINOFF, SHARADAR_ACTION_SPLIT
from src.constants.constants_price import MACRO_MARKET_SERIES
from src.context import Context
from src.data_aggregate.utils.common.peers_io import load_peers
from src.data_aggregate.utils.price.observations import fundamental_share_observations
from src.data_store.schema import Tables
from src.utils.cutover_continuity import PredecessorSeries, predecessor_actions
from src.utils.macro import load_macro_series
from src.utils.price.adjustments import ShareBasis, unsupported_units

logger = logging.getLogger(__name__)
PRICE_COLS = ["date", "open", "high", "low", "close_split", "close_total", "volume", "ticker"]


def load_fundamental_share_observations(
    context: Context, history: pd.DataFrame, basis: ShareBasis, predecessors: tuple[PredecessorSeries, ...]
) -> pd.DataFrame:
    """Read one scoped raw representation projection for an ephemeral financial unit view."""
    wanted = ["ticker", "date", "reportperiod", "calendardate", "lastupdated", "sharesbas", "shareswa", "shareswadil", "dps"]
    available = set(context.store.columns(Tables.sharadar_fundamentals_facts))
    missing = pd.Series(pd.NaT, index=history.index, dtype="datetime64[ns]")
    dates = pd.to_datetime(history.get("as_of", missing), errors="coerce")
    fiscal = pd.to_datetime(history.get("fiscal_end", missing), errors="coerce")
    raw = None
    if dates.notna().any() and fiscal.notna().any():
        raw = context.store.load(
            Tables.sharadar_fundamentals_facts,
            columns=[field for field in wanted if field in available] or wanted,
            where={"ticker": sorted(set(history["ticker"].astype(str))), "dimension": "ARQ"},
            since=fiscal.min() - pd.DateOffset(years=1),
            until=dates.max(),
            optional=True,
        )
    return fundamental_share_observations(history, raw, basis, predecessors)


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


def _owner_series(df_owners: pd.DataFrame | None) -> tuple[PredecessorSeries, ...]:
    """The owner-table windows whose traded symbol is another security; a ticker with no such row owns itself."""
    if df_owners is None:
        return ()
    df_owners = df_owners.loc[df_owners["traded_symbol"].ne(df_owners["ticker"])]
    starts, ends = pd.to_datetime(df_owners["valid_from"]), pd.to_datetime(df_owners["valid_to"])
    rows = zip(df_owners["ticker"], df_owners["traded_symbol"], df_owners["issuer_cik"], starts, ends, strict=True)
    return tuple(PredecessorSeries(ticker, symbol, cik, start, None if pd.isna(end) else end) for ticker, symbol, cik, start, end in rows)


def load_price_inputs(context: Context, tickers: list[str], bugfix: dict) -> PriceInputs:
    """Read projected sources once, retaining full actions, anchors and public share seeds."""
    store = context.store
    df_owners = store.load(
        Tables.price_owner_windows,
        columns=["ticker", "traded_symbol", "issuer_cik", "valid_from", "valid_to"],
        where={"ticker": tickers},
        optional=True,
    )
    df_security = store.load(Tables.sharadar_tickers, columns=["ticker", "category"], where={"ticker": tickers}, optional=True)
    predecessors = _owner_series(df_owners)
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
    available = set(store.columns(Tables.sharadar_fundamentals_facts))
    df_vendor = store.load(
        Tables.sharadar_fundamentals_facts,
        columns=[field for field in wanted if field in available] or wanted,
        where={"ticker": tickers, "dimension": "ARQ"},
        optional=True,
    )
    df_shares = store.load(
        Tables.sharadar_fundamentals_history,
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
    unknown = unsupported_units(df_actions, df_security, tickers, bugfix)
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
