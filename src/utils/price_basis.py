"""Scoped source-anchor loading for the shared, pure share-basis resolver."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import pandas as pd

from src.context import Context
from src.data_store.schema import Tables
from src.utils.price_adjustments import ShareBasis, resolve_share_events
from src.utils.price_repairs import apply_return_seams, apply_split_vintage


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
    if raw is None or raw.empty:
        return raw
    raw = raw.copy()
    raw["date"] = pd.to_datetime(raw["date"])
    vendor = None
    if any(vintage.values()):
        vendor = context.store.load(
            Tables.sharadar_fundamentals,
            columns=["ticker", "date", "price"],
            where={"ticker": names, "dimension": "ARQ"},
            since=raw["date"].min(),
            until=raw["date"].max(),
            optional=True,
        )
        if vendor is not None:
            vendor = vendor.assign(date=pd.to_datetime(vendor["date"]))
    wide = {"close_split": raw.pivot(index="date", columns="ticker", values="close_split")}
    repairs = {"split_vintage": vintage, "return_seams": legacy}
    apply_split_vintage(wide, repairs, vendor, context.log.info)
    apply_return_seams(wide, repairs, context.log.info)
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
