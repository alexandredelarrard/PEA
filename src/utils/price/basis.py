"""Scoped source-anchor loading for the shared, pure share-basis resolver."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import pandas as pd

from src.constants.constants import SHARADAR_ACTION_SPINOFF, SHARADAR_ACTION_SPLIT
from src.context import Context
from src.data_store.schema import Tables
from src.utils.cutover_continuity import PredecessorSeries, predecessor_actions, predecessor_series, register_windows
from src.utils.identity_tables import identity_columns, read_identity
from src.utils.predecessor_series import load_vendor_series, with_overrides
from src.utils.price.adjustments import ShareBasis, quantity_basis, resolve_share_events
from src.utils.price.repairs import apply_return_seams, apply_split_vintage


def load_price_identity(context: Context, tickers: list[str]) -> tuple[tuple[PredecessorSeries, ...], pd.DataFrame | None]:
    """Load predecessor ownership and ADR category evidence in one metadata projection."""
    store = context.store
    columns = ["canonical_ticker", "cik", "role", "valid_from", "valid_to", "sources"]
    df_lineage = (
        read_identity(store, Tables.universe_cik_symbol, columns=columns, where={"role": "cik_window"}, optional=True)
        if "role" in identity_columns(store, Tables.universe_cik_symbol)
        else None
    )
    security_columns = ["ticker", "secfilings", "lastquarter"]
    if "category" in store.columns(Tables.sharadar_tickers):
        security_columns.append("category")
    df_security = store.load(Tables.sharadar_tickers, columns=security_columns, optional=True)
    derived = () if df_lineage is None or df_security is None else predecessor_series(df_security, register_windows(df_lineage), tickers)
    declared = tuple(item for item in load_vendor_series(context.config_dir) if item.ticker in tickers)
    return with_overrides(derived, declared), df_security


def load_quantity_basis(
    context: Context,
    tickers: Sequence[str],
    bugfix: dict,
    identity: tuple[tuple[PredecessorSeries, ...], pd.DataFrame | None] | None = None,
) -> ShareBasis:
    """Load sparse quantity events and identity once, without price anchors or repairs."""
    names = sorted(set(tickers))
    store = context.store
    series, df_security = load_price_identity(context, names) if identity is None else identity
    df_splits = store.load(Tables.prices_splits, columns=["ticker", "date", "ratio"], where={"ticker": names}, optional=True)
    df_actions = store.load(
        Tables.sharadar_actions,
        columns=["ticker", "date", "action", "value"],
        where={
            "ticker": sorted(set(names) | {s.vendor_ticker for s in series}),
            "action": [SHARADAR_ACTION_SPLIT, SHARADAR_ACTION_SPINOFF, "adrratiosplit"],
        },
        optional=True,
    )
    if df_splits is None and store.exists(Tables.prices_splits):
        df_splits = pd.DataFrame(columns=["ticker", "date", "ratio"])
    if df_actions is None and store.exists(Tables.sharadar_actions):
        df_actions = pd.DataFrame(columns=["ticker", "date", "action", "value"])
    if df_actions is not None and series:
        df_units = df_actions.loc[df_actions["action"].eq("adrratiosplit") & df_actions["ticker"].isin(names)]
        df_economics = df_actions.loc[df_actions["action"].isin([SHARADAR_ACTION_SPLIT, SHARADAR_ACTION_SPINOFF])]
        df_owners = df_economics.loc[df_economics["ticker"].isin({s.vendor_ticker for s in series})]
        df_actions = predecessor_actions(df_economics.loc[df_economics["ticker"].isin(names)], df_owners, series)
        df_actions = pd.concat([df_actions, df_units], ignore_index=True)
    return quantity_basis(df_actions, df_splits, bugfix, names, df_security)


def _quote_anchors(context: Context, anchors: list[tuple[str, dict]], bugfix: dict, quotes: pd.DataFrame | None) -> pd.DataFrame | None:
    """Use covered repaired bars, otherwise read named history through its close-repair boundaries."""
    if quotes is not None:
        keys = set(zip(quotes["ticker"], pd.to_datetime(quotes["date"]), strict=True))
        if all((ticker, pd.Timestamp(anchor["date"])) in keys for ticker, anchor in anchors):
            return quotes
    names = sorted({ticker for ticker, _ in anchors})
    legacy = {ticker: [entry for entry in bugfix.get("return_seams", {}).get(ticker, []) if "fields" not in entry] for ticker in names}
    vintage = {ticker: bugfix.get("split_vintage", {}).get(ticker, []) for ticker in names}
    dates = [pd.Timestamp(anchor["date"]) for _, anchor in anchors]
    boundaries = [pd.Timestamp(entry["date"]) for entries in legacy.values() for entry in entries]
    boundaries.extend(pd.Timestamp(entry["before"]) for entries in vintage.values() for entry in entries)
    raw = context.store.load(
        Tables.prices,
        columns=["ticker", "date", "close_split"],
        where={"ticker": names},
        since=min(dates) - pd.Timedelta(days=7),
        until=max([*dates, *boundaries]) + pd.Timedelta(days=7),
        optional=True,
    )
    vendor = (
        context.store.load(
            Tables.sharadar_fundamentals,
            columns=["ticker", "date", "price"],
            where={"ticker": names, "dimension": "ARQ"},
            since=min(dates) - pd.Timedelta(days=7),
            until=max([*dates, *boundaries]) + pd.Timedelta(days=7),
            optional=True,
        )
        if any(vintage.values())
        else None
    )
    return prepare_quote_anchors(raw, vendor, anchors, bugfix)


def prepare_quote_anchors(
    raw: pd.DataFrame | None, vendor: pd.DataFrame | None, anchors: list[tuple[str, dict]], bugfix: dict
) -> pd.DataFrame | None:
    """Prepare already-loaded quote anchors with the existing guarded price repairs."""
    if raw is None or raw.empty:
        return raw
    names = {ticker for ticker, _ in anchors}
    raw = raw.loc[raw["ticker"].isin(names)].copy()
    raw["date"] = pd.to_datetime(raw["date"])
    if vendor is not None:
        vendor = vendor.assign(date=pd.to_datetime(vendor["date"]))
    wide = {"close_split": raw.pivot(index="date", columns="ticker", values="close_split")}
    legacy = {ticker: [entry for entry in bugfix.get("return_seams", {}).get(ticker, []) if "fields" not in entry] for ticker in names}
    vintage = {ticker: bugfix.get("split_vintage", {}).get(ticker, []) for ticker in names}
    repairs = {"split_vintage": vintage, "return_seams": legacy}
    apply_split_vintage(wide, repairs, vendor)
    apply_return_seams(wide, repairs)
    stacked = cast(pd.Series, wide["close_split"].rename_axis(index="date", columns="ticker").stack(future_stack=True))
    return stacked.rename("close_split").reset_index()


def load_share_basis(
    context: Context,
    tickers: Sequence[str],
    bugfix: dict,
    actions: pd.DataFrame | None,
    yf_splits: pd.DataFrame | None,
    *,
    shares: pd.DataFrame | None = None,
    quotes: pd.DataFrame | None = None,
    base_events: pd.DataFrame | None = None,
) -> ShareBasis:
    """Load only the scoped sources and repaired anchors needed even by a tail build."""
    names = list(tickers)
    policies = {ticker: policy for ticker, policy in bugfix.get("share_events", {}).items() if ticker in names}
    if policies and yf_splits is None and context.store.exists(Tables.prices_splits):
        # Optional reads collapse empty and missing; remeasure the available scoped source.
        observed = context.store.load(Tables.prices_splits, columns=["ticker", "date", "ratio"], where={"ticker": names}, optional=True)
        yf_splits = pd.DataFrame(columns=["ticker", "date", "ratio"]) if observed is None else observed
    anchors = [(ticker, anchor) for ticker, policy in policies.items() for anchor in policy["anchors"]]
    share_anchors = [(ticker, anchor) for ticker, anchor in anchors if "ordinary_shares" in anchor]
    if share_anchors and shares is None:
        dates = [anchor["date"] for _, anchor in share_anchors]
        shares = context.store.load(
            Tables.sharadar_fundamentals,
            columns=["ticker", "date", "sharesbas"],
            where={"ticker": sorted({ticker for ticker, _ in share_anchors}), "dimension": "ARQ"},
            since=min(dates),
            until=max(dates),
            optional=True,
        )
    price_anchors = [(ticker, anchor) for ticker, anchor in anchors if "quoted_price" in anchor]
    if price_anchors:
        quotes = _quote_anchors(context, price_anchors, bugfix, quotes)
    return resolve_share_events(actions, yf_splits, bugfix, shares=shares, quotes=quotes, tickers=names, base_events=base_events)
