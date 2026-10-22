"""Shared source and share-basis inputs for the capital disclosure cube builders."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import pandas as pd

from src.constants.constants import SHARADAR_ACTION_SPINOFF, SHARADAR_ACTION_SPLIT
from src.context import Context
from src.data_aggregate.utils.common.level_basis import apply_volume_scale, as_traded_close
from src.data_aggregate.utils.common.peers_io import load_peers_or_raise
from src.data_aggregate.utils.common.price_frames import PriceFrames, load_price_frames
from src.data_store.schema import Table, Tables
from src.data_store.store import DataStore
from src.utils.cutover_continuity import predecessor_actions
from src.utils.predecessor_series import load_predecessor_series
from src.utils.price_adjustments import ShareBasis, resolve_share_events
from src.utils.price_basis import prepare_quote_anchors
from src.utils.string import normalise_ticker, yahoo_symbol

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


def load_source(store: DataStore, table: Table, universe: Sequence[str] | None = None, where: dict[str, Any] | None = None) -> pd.DataFrame | None:
    """Stream one projected source payload in its economic scope."""
    filters = dict(where or {})
    if universe is not None and table.ticker_col:
        filters[table.ticker_col] = sorted(set(universe))
    chunks = list(store.iter_load(table, project=True, where=filters or None))
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
    """Read each basis source once, combining class volume and all share/quote anchors."""
    store = context.store
    names = list(frames.universe)
    series = load_predecessor_series(context, names)
    policies = {ticker: policy for ticker, policy in bugfix.get("share_events", {}).items() if ticker in names}
    anchors = [(ticker, anchor) for ticker, policy in policies.items() for anchor in policy["anchors"]]
    quote_anchors = [(ticker, anchor) for ticker, anchor in anchors if "quoted_price" in anchor]
    class_names = set(class_lines["market_symbol"].map(yahoo_symbol)) if class_lines is not None else set()
    quote_names = {ticker for ticker, _ in quote_anchors}
    raw_names = sorted(class_names | quote_names)
    vendor_names = sorted({ticker for ticker, _ in anchors})
    boundaries = [pd.Timestamp(a["date"]) for _, a in quote_anchors]
    boundaries += [pd.Timestamp(e["date"]) for t in quote_names for e in bugfix.get("return_seams", {}).get(t, []) if "fields" not in e]
    boundaries += [pd.Timestamp(e["before"]) for t in quote_names for e in bugfix.get("split_vintage", {}).get(t, [])]
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
            columns=["ticker", "date", "close_split", "volume"],
            where={"ticker": raw_names},
            since=min(boundaries) - pd.Timedelta(days=7) if boundaries and not class_names else None,
            until=max(boundaries) + pd.Timedelta(days=7) if boundaries and not class_names else None,
            optional=True,
        )
        if raw_names
        else None
    )
    vendor = (
        store.load(
            Tables.sharadar_fundamentals,
            columns=["ticker", "date", "sharesbas", "price"],
            where={"ticker": vendor_names, "dimension": "ARQ"},
            optional=True,
        )
        if vendor_names
        else None
    )

    if splits is None and store.exists(Tables.prices_splits):
        splits = pd.DataFrame(columns=["ticker", "date", "ratio"])
    if actions is None and store.exists(Tables.sharadar_actions):
        actions = pd.DataFrame(columns=["ticker", "date", "action", "value"])
    if actions is not None and series:
        owners = actions.loc[actions["ticker"].isin({s.vendor_ticker for s in series})]
        present = set(owners["ticker"].map(normalise_ticker))
        for predecessor in series:
            if predecessor.vendor_ticker not in present:
                context.log.warning(
                    "level factor: no selected predecessor split/spinoff rows for %s of %s (CIK %s); actions before %s are kept as fetched",
                    predecessor.vendor_ticker,
                    predecessor.ticker,
                    predecessor.cik,
                    predecessor.valid_to.date() if predecessor.valid_to is not None else "-",
                )
        actions = predecessor_actions(actions.loc[actions["ticker"].isin(names)], owners, series)
    if vendor is not None:
        vendor = vendor.assign(date=pd.to_datetime(vendor["date"]))
    quotes = prepare_quote_anchors(raw, vendor, quote_anchors, bugfix) if quote_anchors else None
    basis = resolve_share_events(actions, splits, bugfix, shares=vendor, quotes=quotes, tickers=names)
    bars = (
        raw.loc[raw["ticker"].isin(class_names), ["ticker", "date", "volume"]]
        if raw is not None
        else pd.DataFrame(columns=["ticker", "date", "volume"])
    )
    bars = apply_volume_scale(bars, bugfix)
    return shares, splits, basis, as_traded_close(frames.close_split, frames.level_factor, basis), bars
