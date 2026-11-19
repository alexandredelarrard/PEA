"""Visible scoped loading for one coherent historical price candidate."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from src.constants.constants import SHARADAR_ACTION_SPINOFF, SHARADAR_ACTION_SPLIT
from src.constants.constants_price import MACRO_MARKET_SERIES
from src.context import Context
from src.data_aggregate.utils.common.inputs import input_windows, load_source, source_read_scope
from src.data_aggregate.utils.common.peers_io import load_peers
from src.data_aggregate.utils.price.observations import fundamental_share_observations
from src.data_store.schema import Tables
from src.utils.cutover_continuity import PredecessorSeries, predecessor_actions
from src.utils.price.adjustments import ShareBasis, unsupported_units
from src.utils.publication import point_in_time
from src.utils.window import Window

logger = logging.getLogger(__name__)
PRICE_COLS = ["date", "open", "high", "low", "close_split", "close_total", "volume", "ticker"]


def load_fundamental_share_observations(
    context: Context, history: pd.DataFrame, basis: ShareBasis, predecessors: tuple[PredecessorSeries, ...], window: Window | None
) -> pd.DataFrame:
    """Read one scoped raw representation projection for an ephemeral financial unit view."""
    wanted = ["ticker", "date", "reportperiod", "calendardate", "lastupdated", "sharesbas", "shareswa", "shareswadil", "dps"]
    available = set(context.store.columns(Tables.sharadar_fundamentals_facts))
    missing = pd.Series(pd.NaT, index=history.index, dtype="datetime64[ns]")
    dates = pd.to_datetime(history.get("_source_as_of", history.get("as_of", missing)), errors="coerce")
    raw = None
    if dates.notna().any():
        raw = context.store.load(
            Tables.sharadar_fundamentals_facts,
            columns=[field for field in wanted if field in available] or wanted,
            where={"ticker": sorted(set(history["ticker"].astype(str))), "dimension": "ARQ"},
            **source_read_scope(context, Tables.sharadar_fundamentals_facts, window),
            optional=True,
        )
    df_original = history.assign(as_of=dates)
    df_observed = fundamental_share_observations(df_original, raw, basis, predecessors)
    df_observed["as_of"] = history["as_of"]
    return df_observed


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
    # All root inputs keep complete anchors, while missing map entries still fail.
    windows = input_windows(context, Tables.cube_part_prices, Window(context.run.as_of, context.run.as_of))
    for table in (Tables.price_owner_windows, Tables.sharadar_tickers):
        if windows[table] is not None:
            raise ValueError(f"{Tables.cube_part_prices.name}/{table.name}: complete anchor metadata requires null warmup")
    finite = [table.name for table, window in windows.items() if window is not None]
    if finite:
        raise ValueError(f"{Tables.cube_part_prices.name}: cumulative anchors require null warmup for {finite}")
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
    df_bars = load_source(context, Tables.prices, tickers, columns=PRICE_COLS, window=windows[Tables.prices])
    df_macro = load_source(
        context, Tables.prices_macro, [MACRO_MARKET_SERIES], columns=["date", "ticker", "close"], window=windows[Tables.prices_macro]
    )
    market = None if df_macro is None else df_macro.loc[df_macro.ticker.eq(MACRO_MARKET_SERIES)].set_index("date")["close"].sort_index()
    df_actions = load_source(
        context, Tables.sharadar_actions, action_tickers, columns=["ticker", "date", "action", "value"], window=windows[Tables.sharadar_actions]
    )
    if df_actions is not None:
        df_actions = df_actions.loc[df_actions.action.isin([SHARADAR_ACTION_SPLIT, SHARADAR_ACTION_SPINOFF, "adrratiosplit"])]
    df_splits = load_source(context, Tables.prices_splits, tickers, columns=["ticker", "date", "ratio"], window=windows[Tables.prices_splits])
    wanted = ["ticker", "date", "dimension", "reportperiod", "sharesbas", "sharefactor", "price", "lastupdated"]
    available = set(store.columns(Tables.sharadar_fundamentals_facts))
    df_vendor = load_source(
        context,
        Tables.sharadar_fundamentals_facts,
        tickers,
        columns=[field for field in wanted if field in available] or wanted,
        window=windows[Tables.sharadar_fundamentals_facts],
    )
    if df_vendor is not None:
        df_vendor = df_vendor.loc[df_vendor.dimension.eq("ARQ")]
    df_shares = load_source(
        context,
        Tables.sharadar_fundamentals_history,
        tickers,
        columns=["ticker", "as_of", "fiscal_end", "sharesOutstanding", "sharesOutstandingPit"],
        window=windows[Tables.sharadar_fundamentals_history],
    )
    if df_shares is not None and market is not None:
        df_shares = point_in_time(df_shares, context.config, Tables.sharadar_fundamentals_history, pd.DatetimeIndex(market.index)).rename(
            columns={"as_of": "_available_on"}
        )
        df_shares["as_of"] = df_shares["_source_as_of"]
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
