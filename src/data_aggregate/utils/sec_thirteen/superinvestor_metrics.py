"""Shared descriptive and matched-quarter equations for raw-CIK cohorts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.data_aggregate.utils.sec_thirteen.superinvestor_books import PublicBook, PublicBookSnapshot
from src.utils.price_adjustments import ShareBasis


@dataclass(frozen=True)
class BehaviorMarket:
    """Existing as-traded quotes, authoritative sessions and genuine share actions."""

    trading_index: pd.DatetimeIndex
    as_traded_close: pd.DataFrame | None
    basis: ShareBasis


@dataclass(frozen=True)
class BookSummary:
    df_positions: pd.DataFrame
    df_tickers: pd.DataFrame
    total_value: float


class BookSummaries:
    """Cache full summaries and compact tenure evidence per immutable vintage."""

    def __init__(self) -> None:
        self._cache: dict[PublicBook, BookSummary] = {}
        self._held_cache: dict[PublicBook, pd.Index] = {}

    def get(self, book: PublicBook) -> BookSummary:
        if book not in self._cache:
            self._cache[book] = summarize_book(book)
            self._held_cache.pop(book, None)
        return self._cache[book]

    def held(self, book: PublicBook) -> pd.Index:
        """Reuse full evidence or retain only verified-held tickers for history."""
        if book in self._cache:
            df_tickers = self._cache[book].df_tickers
            return df_tickers.index[df_tickers.held & df_tickers.held_known]
        if book not in self._held_cache:
            # Project before copying; optional provenance keeps its existing defaults.
            columns = ["ticker", "position_type", "shares", "value_usd"]
            columns.extend(name for name in ("value_basis_shares_source", "value_basis_inverse_unit_unverified") if name in book.df_positions)
            df_positions = _holding_evidence(book.df_positions[columns])
            grouped = df_positions.groupby("ticker", sort=False)
            held = grouped.held.any()
            self._held_cache[book] = held.index[held & grouped.held_known.all()]
        return self._held_cache[book]


@dataclass(frozen=True)
class PairSample:
    df_positions: pd.DataFrame
    managers: frozenset[str]
    period: pd.Timestamp | None
    prior_period: pd.Timestamp | None


def _holding_evidence(df_positions: pd.DataFrame) -> pd.DataFrame:
    """Coerce common-leg amounts and preserve conservative row holding evidence."""
    df_positions = df_positions.copy()
    for name in ("shares", "value_usd"):
        df_positions[name] = pd.to_numeric(df_positions[name], errors="coerce").astype("float64")

    # A typed amendment row can retain an explicitly unknown common leg.
    unknown_common = df_positions[["shares", "value_usd"]].isna().any(axis=1)
    df_positions = df_positions.loc[df_positions.position_type.eq("common") | unknown_common].copy()

    # Estimates can describe value, but cannot establish observed share trading.
    quantity_source = df_positions.get("value_basis_shares_source", pd.Series("unknown", index=df_positions.index))
    df_positions["quantity_known"] = quantity_source.eq("filed") & np.isfinite(df_positions.shares) & df_positions.shares.ge(0)
    unverified = df_positions.get("value_basis_inverse_unit_unverified", pd.Series(False, index=df_positions.index)).fillna(True)
    df_positions["value_known"] = np.isfinite(df_positions.value_usd) & df_positions.value_usd.ge(0) & ~unverified
    df_positions["held"] = df_positions.shares.gt(0) | df_positions.value_usd.gt(0)
    df_positions["held_known"] = df_positions.held | (df_positions.quantity_known & df_positions.value_known)
    return df_positions


def summarize_book(book: PublicBook) -> BookSummary:
    """Keep whole-book totals, per-security evidence and per-ticker levels together."""
    df_positions = _holding_evidence(book.df_positions)
    total_value = float(df_positions.value_usd.sum()) if df_positions.value_known.all() else np.nan

    # Rank all common positions, including those outside the output universe.
    df_positions = df_positions.sort_values(["value_usd", "cusip"], ascending=[False, True], kind="stable")
    df_positions["top10"] = np.arange(len(df_positions)) < 10
    df_positions["top10"] &= df_positions.held & np.isfinite(total_value) & (total_value > 0)
    grouped = df_positions.groupby("ticker", sort=False)
    df_tickers = grouped[["shares", "value_usd"]].sum(min_count=1)
    df_tickers["held"] = grouped.held.any()
    df_tickers["held_known"] = grouped.held_known.all()
    df_tickers["top10"] = grouped.top10.any()
    df_tickers["value_usd"] = df_tickers.value_usd.where(grouped.value_known.all())

    # Keep comparison evidence narrow; PublicBook retains full provenance.
    comparison_columns = ["cusip", "ticker", "shares", "quantity_known"]
    if "class_evidence" in df_positions:
        comparison_columns.append("class_evidence")
    return BookSummary(df_positions[comparison_columns].copy(), df_tickers, total_value)


def _supported_book(book: PublicBook | None) -> Any[PublicBook]:
    return book is not None and book.support.book_known and book.support.zero_eligible and book.support.comparison_eligible


def _holding_duration(book: PublicBook, snapshot: PublicBookSnapshot, summaries: BookSummaries, universe: tuple[str, ...]) -> pd.Series:
    """Count adjacent observed report quarters; missing quarters break the verified run."""
    held = summaries.held(book)
    duration = pd.Series(1.0, index=held[held.isin(universe)])
    active = duration.index
    period = book.period.to_period("Q") - 1
    while len(active):
        prior = snapshot.archive.get((book.cik, period.end_time.normalize()))
        if not _supported_book(prior):
            break
        active = active.intersection(summaries.held(prior))
        duration.loc[active] += 1
        period -= 1
    return duration


def level_metrics(
    snapshot: PublicBookSnapshot, cohort: frozenset[str], universe: tuple[str, ...], summaries: BookSummaries
) -> tuple[dict[str, pd.Series], dict[str, int]]:
    """Use each manager's latest fresh complete book for descriptive levels."""
    books = [book for cik, book in snapshot.level_books.items() if cik in cohort and _supported_book(book)]
    names = ("holders", "conviction_weight", "max_conviction", "top10_holders", "quarters_held", "pooled_weight")
    result = {name: pd.Series(np.nan, index=universe, dtype="float64") for name in names}
    if not books:
        return result, {"observed": 0, "valued": 0}

    # Structural zeros are allowed only because each included book is complete.
    held_rows, top_rows, weight_rows, value_rows, durations, totals = [], [], [], [], [], []
    for book in books:
        summary = summaries.get(book)
        df_tickers = summary.df_tickers.reindex(universe)
        absent = ~df_tickers.index.isin(summary.df_tickers.index)
        held_rows.append(df_tickers.held.astype(float).where(df_tickers.held_known.fillna(False)).mask(absent, 0.0))
        durations.append(_holding_duration(book, snapshot, summaries, universe).reindex(universe))
        if np.isfinite(summary.total_value):
            top_rows.append(df_tickers.top10.astype(float).mask(absent, 0.0))
        if np.isfinite(summary.total_value) and summary.total_value > 0:
            values = df_tickers.value_usd.mask(absent, 0.0)
            value_rows.append(values)
            weight_rows.append(values / summary.total_value)
            totals.append(summary.total_value)

    df_held = pd.DataFrame(held_rows)
    result["holders"] = df_held.sum(min_count=1)
    result["quarters_held"] = pd.DataFrame(durations).median()
    if top_rows:
        result["top10_holders"] = pd.DataFrame(top_rows).sum(min_count=1)
    if weight_rows:
        df_weights = pd.DataFrame(weight_rows)
        result["conviction_weight"] = df_weights.mean()
        result["max_conviction"] = df_weights.where(df_weights.gt(0)).max()
        result["pooled_weight"] = pd.DataFrame(value_rows).sum(min_count=1) / sum(totals)
    return result, {"observed": len(books), "valued": len(totals)}


def _join_positions(current: BookSummary, prior: BookSummary) -> pd.DataFrame:
    df_pair = current.df_positions.merge(prior.df_positions, on="cusip", how="outer", suffixes=("_current", "_prior"), indicator=True)
    current_absent = df_pair._merge.eq("right_only")
    prior_absent = df_pair._merge.eq("left_only")
    df_pair["ticker"] = df_pair.ticker_current.combine_first(df_pair.ticker_prior)
    df_pair["current"] = df_pair.shares_current.mask(current_absent, 0.0)
    df_pair["prior"] = df_pair.shares_prior.mask(prior_absent, 0.0)
    df_pair["known"] = (df_pair.quantity_known_current.fillna(False) | current_absent) & (df_pair.quantity_known_prior.fillna(False) | prior_absent)

    # A changed identity/class is not an evidenced split or a verified round trip.
    both = df_pair._merge.eq("both")
    df_pair["known"] &= ~both | df_pair.ticker_current.eq(df_pair.ticker_prior)
    if "class_evidence_current" in df_pair:
        df_pair["known"] &= ~both | df_pair.class_evidence_current.eq(df_pair.class_evidence_prior)
    grouped = df_pair.groupby("ticker", sort=False)
    changed = df_pair.current.gt(0).ne(df_pair.prior.gt(0)).groupby(df_pair.ticker).any()
    both_held = grouped.current.max().gt(0) & grouped.prior.max().gt(0)
    changed_identity = changed & both_held & grouped.cusip.nunique().gt(1)
    df_pair["known"] &= ~df_pair.ticker.isin(changed_identity.index[changed_identity])
    return df_pair[["cusip", "ticker", "current", "prior", "known"]]


def _attach_reference_prices(df_pair: pd.DataFrame, market: BehaviorMarket, prior_period: pd.Timestamp, period: pd.Timestamp) -> pd.DataFrame:
    """Rebase prior shares and the single pre-comparison quote vector inversely."""
    df_actions = market.basis.actual
    actions = df_actions.loc[df_actions.date.gt(prior_period) & df_actions.date.le(period)]
    ratios = actions.groupby("ticker").value.prod()
    ratio = df_pair.ticker.map(ratios).fillna(1.0)
    df_pair["prior"] *= ratio

    # Exact reference-session quotes only; missing securities are never dropped or filled.
    position = int(market.trading_index.searchsorted(prior_period, side="right")) - 1
    prices = pd.Series(dtype="float64")
    if position >= 0 and market.as_traded_close is not None:
        reference = market.trading_index[position]
        prices = market.as_traded_close.reindex(index=[reference]).iloc[0]
    df_pair["price"] = df_pair.ticker.map(prices) / ratio
    dates = pd.Series(period, index=df_pair.index)
    units = market.basis.factor(df_pair.ticker, dates, "units")
    df_pair["known"] &= units.eq(1.0) & np.isfinite(ratio) & ratio.gt(0)
    return df_pair


def prepare_pair_sample(snapshot: PublicBookSnapshot, lag: int, summaries: BookSummaries, market: BehaviorMarket) -> PairSample:
    """Prepare genuine exact-quarter pairs once for both cohort reductions."""
    if snapshot.common_period is None:
        return PairSample(pd.DataFrame(), frozenset(), None, None)
    period = snapshot.common_period
    prior_period = (period.to_period("Q") - lag).end_time.normalize()
    parts, managers = [], set()
    for cik, current in snapshot.level_books.items():
        prior = snapshot.archive.get((cik, prior_period))
        if current.period != period or not _supported_book(current) or not _supported_book(prior):
            continue
        df_pair = _join_positions(summaries.get(current), summaries.get(prior))
        parts.append(df_pair.assign(cik=cik))
        managers.add(cik)
    if not parts:
        return PairSample(pd.DataFrame(), frozenset(), period, prior_period)
    df_positions = pd.concat(parts, ignore_index=True)
    df_positions = _attach_reference_prices(df_positions, market, prior_period, period)
    return PairSample(df_positions, frozenset(managers), period, prior_period)


def quantity_metrics(sample: PairSample, cohort: frozenset[str], universe: tuple[str, ...]) -> dict[str, pd.Series]:
    """Compare matched filed quantities, keeping entries, exits and unknowns distinct."""
    result = {name: pd.Series(np.nan, index=universe, dtype="float64") for name in ("breadth_chg", "shares_chg", "trade_balance")}
    count = len(sample.managers & cohort)
    if not count:
        return result
    df_positions = sample.df_positions.loc[sample.df_positions.cik.isin(cohort)]
    grouped = df_positions.groupby(["cik", "ticker"], sort=False)
    df_tickers = grouped[["current", "prior"]].sum(min_count=1)
    df_tickers["known"] = grouped.known.all()
    unknown = (~df_tickers.known).groupby(level="ticker").sum().reindex(universe, fill_value=0)
    df_known = df_tickers.loc[df_tickers.known]

    # The denominator includes matched known nonholders, never absent books.
    holder_change = df_known.current.gt(0).astype(float) - df_known.prior.gt(0).astype(float)
    holders = holder_change.groupby(level="ticker").sum().reindex(universe, fill_value=0)
    denominator = (count - unknown).astype(float)
    result["breadth_chg"] = holders / denominator.where(denominator.gt(0))
    totals = df_known.groupby(level="ticker")[["current", "prior"]].sum().reindex(universe, fill_value=0)
    result["shares_chg"] = totals.current / totals.prior.where(totals.prior.gt(0)) - 1
    delta = df_known.current - df_known.prior
    active = ~np.isclose(delta, 0.0, rtol=1e-10, atol=1e-10)
    votes = pd.Series(np.sign(delta), index=delta.index).where(active, 0.0).groupby(level="ticker").sum().reindex(universe, fill_value=0)
    voters = pd.Series(active, index=delta.index).groupby(level="ticker").sum().reindex(universe, fill_value=0)
    result["trade_balance"] = votes / voters.replace(0, np.nan)
    return result


def whole_book_metrics(sample: PairSample, cohort: frozenset[str], universe: tuple[str, ...]) -> tuple[dict[str, pd.Series], int]:
    """Remove endpoint price moves with one common fixed-price whole-book union."""
    result = {name: pd.Series(np.nan, index=universe, dtype="float64") for name in ("relative_position_chg", "pooled_weight_chg")}
    if not sample.managers & cohort:
        return result, 0
    df_positions = sample.df_positions.loc[sample.df_positions.cik.isin(cohort)].copy()
    priced = np.isfinite(df_positions.price) & df_positions.price.gt(0)
    supported = df_positions.known & priced
    eligible = supported.groupby(df_positions.cik).all()
    df_positions = df_positions.loc[df_positions.cik.isin(eligible.index[eligible])]
    if df_positions.empty:
        return result, int(eligible.sum())

    # A missing union position excludes its manager rather than shrinking the book.
    df_positions["prior_value"] = df_positions.prior * df_positions.price
    df_positions["current_value"] = df_positions.current * df_positions.price
    df_books = df_positions.groupby("cik")[["prior_value", "current_value"]].sum()
    prior_total, current_total = df_books.prior_value.sum(), df_books.current_value.sum()
    df_tickers = df_positions.groupby(["cik", "ticker"])[["prior", "current", "prior_value", "current_value"]].sum()
    growth = df_books.current_value / df_books.prior_value.where(df_books.prior_value.gt(0)) - 1
    relative = df_tickers.current / df_tickers.prior.where(df_tickers.prior.gt(0)) - 1 - df_tickers.index.get_level_values("cik").map(growth)
    result["relative_position_chg"] = relative.groupby(level="ticker").mean().reindex(universe)
    if prior_total > 0 and current_total > 0:
        pooled = df_tickers.groupby(level="ticker")[["prior_value", "current_value"]].sum().reindex(universe, fill_value=0)
        result["pooled_weight_chg"] = pooled.current_value / current_total - pooled.prior_value / prior_total
    return result, int(eligible.sum())
