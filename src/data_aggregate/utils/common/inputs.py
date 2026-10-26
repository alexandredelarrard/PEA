"""Shared source and share-basis inputs for the capital disclosure cube builders."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import pandas as pd

from src.constants.constants import SHARADAR_ACTION_SPINOFF, SHARADAR_ACTION_SPLIT
from src.context import Context
from src.data_aggregate.utils.common.peers_io import load_peers_or_raise
from src.data_aggregate.utils.price.clean import apply_volume_scale
from src.data_aggregate.utils.price.frames import PriceFrames, load_price_frames
from src.data_store.schema import Table, Tables
from src.data_store.store import DataStore
from src.utils.cutover_continuity import predecessor_actions
from src.utils.predecessor_series import load_predecessor_series
from src.utils.price.adjustments import ShareBasis, quantity_basis
from src.utils.string import yahoo_symbol

logger = logging.getLogger(__name__)


def load_full_price_frames(
    store: DataStore,
    context: Context,
    fields: Sequence[str],
) -> PriceFrames:
    """Load full-calendar price inputs from the already-persisted peer dependency."""
    peers = load_peers_or_raise(context)
    return load_price_frames(
        store,
        peers=peers,
        fields=fields,
        since=None,
    )


def load_source(store: DataStore, table: Table, universe: Sequence[str] | None = None) -> pd.DataFrame | None:
    """Stream one projected source payload in its economic scope."""
    filters = {table.ticker_col: sorted(set(universe))} if universe is not None and table.ticker_col else None
    chunks = list(store.iter_load(table, project=True, where=filters))
    df = pd.concat(chunks, ignore_index=True) if chunks else None
    if df is None or df.empty:
        logger.warning("%s is absent or empty -> its features are skipped.", table.name)
        return None
    logger.info("Loaded %s: %s rows x %s cols", table.name, len(df), len(df.columns))
    return df


def load_share_inputs(
    context: Context,
    frames: PriceFrames,
    bugfix: dict,
    class_lines: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, ShareBasis, pd.DataFrame | None, pd.DataFrame]:
    """Load sparse quantity evidence and class tape; reuse the price owner's saved quotes."""
    store = context.store
    names = list(frames.universe)
    series = load_predecessor_series(context, names)
    class_names = set(class_lines["market_symbol"].map(yahoo_symbol)) if class_lines is not None else set()
    shares = store.load(
        Tables.fundamentals_history, columns=["ticker", "as_of", "sharesOutstanding", "sharesOutstandingPit"], where={"ticker": names}, optional=True
    )
    splits = store.load(Tables.prices_splits, columns=["ticker", "date", "ratio"], where={"ticker": names}, optional=True)
    actions = store.load(
        Tables.sharadar_actions,
        columns=["ticker", "date", "action", "value"],
        where={"ticker": sorted(set(names) | {s.vendor_ticker for s in series}), "action": [SHARADAR_ACTION_SPLIT, SHARADAR_ACTION_SPINOFF]},
        optional=True,
    )
    raw = (
        store.load(
            Tables.prices,
            columns=["ticker", "date", "volume"],
            where={"ticker": sorted(class_names)},
            optional=True,
        )
        if class_names
        else None
    )

    if splits is None and store.exists(Tables.prices_splits):
        splits = pd.DataFrame(columns=["ticker", "date", "ratio"])
    if actions is None and store.exists(Tables.sharadar_actions):
        actions = pd.DataFrame(columns=["ticker", "date", "action", "value"])
    if actions is not None and series:
        owners = actions.loc[actions["ticker"].isin({s.vendor_ticker for s in series})]
        actions = predecessor_actions(actions.loc[actions["ticker"].isin(names)], owners, series)
    basis = quantity_basis(actions, splits, bugfix, names)
    bars = (
        raw.loc[raw["ticker"].isin(class_names), ["ticker", "date", "volume"]]
        if raw is not None
        else pd.DataFrame(columns=["ticker", "date", "volume"])
    )
    bars = apply_volume_scale(bars, bugfix)
    return shares, splits, basis, frames.close_pit, bars
