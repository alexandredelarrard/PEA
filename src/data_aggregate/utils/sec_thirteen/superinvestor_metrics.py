"""Shared descriptive and matched-quarter equations for raw-CIK cohorts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data_aggregate.utils.common.quotes import lookup_asof
from src.data_aggregate.utils.sec_thirteen.superinvestor_books import BookKey, PublicBook, PublicBookSnapshot
from src.utils.price.levels import convert_quantities

type BookPair = tuple[str, PublicBook, PublicBook]


@dataclass(frozen=True)
class BehaviorMarket:
    """Saved as-traded quotes (`close_pit`), authoritative sessions and the forward quantity basis F (`security_basis`)."""

    trading_index: pd.DatetimeIndex
    close_pit: pd.DataFrame | None
    basis: pd.DataFrame


@dataclass(frozen=True)
class BookTable:
    """Every revision's comparison positions, whole-book value and per-output-ticker level rows, by revision code."""

    codes: Mapping[PublicBook, int]
    previous: Mapping[pd.Timestamp, pd.Timestamp]
    df_positions: pd.DataFrame
    starts: np.ndarray
    has_class: np.ndarray
    held: np.ndarray
    top: np.ndarray
    value: np.ndarray
    total: np.ndarray


@dataclass(frozen=True)
class PairTable:
    """Every compared revision pair's CUSIP-union rows with comparison support, by pair code."""

    codes: Mapping[tuple[PublicBook, PublicBook], int]
    df_rows: pd.DataFrame
    starts: np.ndarray


@dataclass(frozen=True)
class PairSample:
    df_positions: pd.DataFrame
    df_tickers: pd.DataFrame
    managers: frozenset[str]


@dataclass(frozen=True)
class LevelSample:
    df_held: pd.DataFrame
    df_top: pd.DataFrame
    df_weights: pd.DataFrame
    df_values: pd.DataFrame
    df_durations: pd.DataFrame
    totals: pd.Series


@dataclass(frozen=True)
class WholeBookSample:
    df_tickers: pd.DataFrame
    df_books: pd.DataFrame


def summarize_books(books: Sequence[PublicBook], universe: pd.Index) -> BookTable:
    """Summarize each revision once: per-security evidence, whole-book value and per-output-ticker levels."""
    columns = ["cusip", "ticker", "position_type", "shares", "value_usd"]
    columns += ["class_evidence", "value_basis_shares_source"]
    frames = [book.df_positions.reindex(columns=columns) for book in books]
    df_positions = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
    df_positions["_book"] = np.repeat(np.arange(len(books)), [len(book.df_positions) for book in books])
    has_class = np.array(["class_evidence" in book.df_positions for book in books], dtype=bool)

    # A missing flag column is verified; a null flag is unverified.
    flag = "value_basis_inverse_unit_unverified"
    verified = [~book.df_positions.get(flag, pd.Series(False, index=book.df_positions.index)).fillna(True).astype(bool) for book in books]
    df_positions["_verified"] = pd.concat(verified, ignore_index=True).to_numpy(bool) if books else np.array([], dtype=bool)
    df_positions = df_positions.astype({"ticker": object, "class_evidence": object})
    df_positions["_cusip"] = pd.factorize(df_positions["cusip"], sort=True)[0]
    for name in ("shares", "value_usd"):
        df_positions[name] = pd.to_numeric(df_positions[name], errors="coerce").astype("float64")

    # A typed amendment row can retain an explicitly unknown common leg; estimates cannot establish observed trading.
    unknown_common = df_positions[["shares", "value_usd"]].isna().any(axis=1)
    df_positions = df_positions.loc[df_positions.position_type.eq("common") | unknown_common].reset_index(drop=True)
    filed = df_positions["value_basis_shares_source"].eq("filed")
    df_positions["quantity_known"] = filed & np.isfinite(df_positions.shares) & df_positions.shares.ge(0)
    df_positions["value_known"] = np.isfinite(df_positions.value_usd) & df_positions.value_usd.ge(0) & df_positions["_verified"]
    df_positions["held"] = df_positions.shares.gt(0) | df_positions.value_usd.gt(0)
    df_positions["held_known"] = df_positions.held | (df_positions.quantity_known & df_positions.value_known)
    starts = np.searchsorted(df_positions["_book"].to_numpy(), np.arange(len(books) + 1))
    values = df_positions.value_usd.to_numpy()
    valued = df_positions.value_known.groupby(df_positions["_book"]).all().reindex(range(len(books)), fill_value=True).to_numpy()
    total = np.array([values[start:stop].sum() if known else np.nan for start, stop, known in zip(starts[:-1], starts[1:], valued, strict=True)])

    # Rank all common positions per book, including those outside the output universe.
    df_positions = df_positions.sort_values(["_book", "value_usd", "cusip"], ascending=[True, False, True], kind="stable")
    rank = df_positions.groupby("_book").cumcount().to_numpy()
    book_total = total[df_positions["_book"].to_numpy()]
    df_positions["top10"] = (rank < 10) & df_positions.held & np.isfinite(book_total) & (book_total > 0)
    grouped = df_positions.groupby(["_book", "ticker"], sort=False)
    df_tickers = grouped[["shares", "value_usd"]].sum(min_count=1)
    df_tickers["value_usd"] = df_tickers.value_usd.where(grouped.value_known.all())

    # Output tickers absent from a book are known zeros.
    cols = universe.get_indexer(df_tickers.index.get_level_values("ticker"))
    inside = cols >= 0
    rows, cols = df_tickers.index.get_level_values("_book").to_numpy()[inside], cols[inside]
    levels = {name: np.zeros((len(books), len(universe))) for name in ("held", "top", "value")}
    levels["held"][rows, cols] = grouped.held.any().astype(float).where(grouped.held_known.all()).to_numpy()[inside]
    levels["top"][rows, cols] = grouped.top10.any().astype(float).to_numpy()[inside]
    levels["value"][rows, cols] = df_tickers.value_usd.to_numpy()[inside]
    df_positions["value"] = df_positions.value_usd.where(df_positions.value_known)
    kept = ["_book", "_cusip", "ticker", "shares", "value", "quantity_known", "class_evidence"]
    comparison = df_positions[kept].sort_values("_book", kind="stable")
    codes = {book: code for code, book in enumerate(books)}
    previous = {period: (period.to_period("Q") - 1).end_time.normalize() for period in {book.period for book in books}}
    return BookTable(codes, previous, comparison.reset_index(drop=True), starts, has_class, levels["held"], levels["top"], levels["value"], total)


def holding_duration(book: PublicBook, archive: Mapping[BookKey, PublicBook], table: BookTable) -> np.ndarray:
    """Count adjacent observed report quarters per held output ticker; missing quarters break the verified run."""
    active = table.held[table.codes[book]] == 1.0
    duration = np.where(active, 1.0, np.nan)
    prior = book
    while active.any():
        prior = archive.get((book.cik, table.previous[prior.period]))
        if prior is None or not prior.support.comparable:
            break
        active &= table.held[table.codes[prior]] == 1.0
        duration[active] += 1
    return duration


def prepare_level_sample(snapshot: PublicBookSnapshot, table: BookTable, durations: Mapping[str, np.ndarray], universe: pd.Index) -> LevelSample:
    """Stack the prepared rows of each supported fresh manager (those with a tenure row) for this event."""
    ciks = pd.Index([cik for cik in snapshot.level_books if cik in durations], dtype=object)
    codes = np.array([table.codes[snapshot.level_books[cik]] for cik in ciks], dtype="int64")
    total = table.total[codes]
    finite, valued = np.isfinite(total), np.isfinite(total) & (total > 0)
    valued_ciks = ciks[valued]
    return LevelSample(
        pd.DataFrame(table.held[codes], index=ciks, columns=universe),
        pd.DataFrame(table.top[codes[finite]], index=ciks[finite], columns=universe),
        pd.DataFrame(table.value[codes[valued]] / total[valued, None], index=valued_ciks, columns=universe),
        pd.DataFrame(table.value[codes[valued]], index=valued_ciks, columns=universe),
        pd.DataFrame(np.array([durations[cik] for cik in ciks]).reshape(len(ciks), len(universe)), index=ciks, columns=universe),
        pd.Series(total[valued], index=valued_ciks, dtype="float64"),
    )


def level_metrics(sample: LevelSample, cohort: frozenset[str], universe: pd.Index) -> tuple[dict[str, pd.Series], dict[str, int]]:
    """Reduce the same prepared manager rows under each fixed cohort mask."""
    names = ("holders", "conviction_weight", "max_conviction", "top10_holders", "quarters_held", "pooled_weight")
    result = {name: pd.Series(np.nan, index=universe, dtype="float64") for name in names}
    df_held = sample.df_held.loc[sample.df_held.index.isin(cohort)]
    totals = sample.totals.loc[sample.totals.index.isin(cohort)]
    if df_held.empty:
        return result, {"observed": 0, "valued": 0}
    result["holders"] = df_held.sum(min_count=1)
    result["quarters_held"] = sample.df_durations.loc[df_held.index].median()
    df_top = sample.df_top.loc[sample.df_top.index.isin(cohort)]
    if not df_top.empty:
        result["top10_holders"] = df_top.sum(min_count=1)
    if not totals.empty:
        df_weights = sample.df_weights.loc[totals.index]
        result["conviction_weight"] = df_weights.mean()
        result["max_conviction"] = df_weights.where(df_weights.gt(0)).max()
        result["pooled_weight"] = sample.df_values.loc[totals.index].sum(min_count=1) / sum(totals)
    return result, {"observed": len(df_held), "valued": len(totals)}


def comparable_pairs(snapshot: PublicBookSnapshot, lag: int) -> list[BookPair]:
    """Fresh managers whose common-quarter book and exact lagged archive book are both supported."""
    if snapshot.common_period is None:
        return []
    prior_period = (snapshot.common_period.to_period("Q") - lag).end_time.normalize()
    pairs = [(cik, current, snapshot.archive.get((cik, prior_period))) for cik, current in snapshot.level_books.items()]
    return [
        (cik, current, prior)
        for cik, current, prior in pairs
        if current.period == snapshot.common_period and current.support.comparable and prior is not None and prior.support.comparable
    ]


def join_pairs(pairs: Sequence[tuple[PublicBook, PublicBook]], table: BookTable) -> PairTable:
    """Outer-join each pair's current and prior positions by CUSIP (codes sort as the CUSIPs), in pair then CUSIP order."""
    sides = []
    for side in (0, 1):
        codes = [table.codes[pair[side]] for pair in pairs]
        rows = np.concatenate([np.arange(table.starts[code], table.starts[code + 1]) for code in codes] + [np.array([], dtype="int64")])
        sides.append(table.df_positions.iloc[rows].drop(columns="_book").assign(_pair=np.repeat(np.arange(len(codes)), np.diff(table.starts)[codes])))
    df_pair = sides[0].merge(sides[1], on=["_pair", "_cusip"], how="outer", suffixes=("_current", "_prior"), indicator=True)
    current_absent = df_pair._merge.eq("right_only")
    prior_absent = df_pair._merge.eq("left_only")
    df_pair["ticker"] = df_pair.ticker_current.combine_first(df_pair.ticker_prior)
    df_pair["current"] = df_pair.shares_current.mask(current_absent, 0.0)
    df_pair["prior"] = df_pair.shares_prior.mask(prior_absent, 0.0)
    df_pair["known"] = (df_pair.quantity_known_current.fillna(False) | current_absent) & (df_pair.quantity_known_prior.fillna(False) | prior_absent)

    # A changed identity/class is not an evidenced split or a verified round trip.
    both = df_pair._merge.eq("both")
    classed = np.array([table.has_class[table.codes[current]] and table.has_class[table.codes[prior]] for current, prior in pairs], dtype=bool)
    df_pair["known"] &= ~both | df_pair.ticker_current.fillna("").eq(df_pair.ticker_prior.fillna(""))
    df_pair["known"] &= ~both | ~classed[df_pair["_pair"].to_numpy()] | df_pair.class_evidence_current.eq(df_pair.class_evidence_prior)
    keys = [df_pair["_pair"], df_pair["ticker"]]
    grouped = df_pair.groupby(keys, sort=False)
    changed = df_pair.current.gt(0).ne(df_pair.prior.gt(0)).groupby(keys, sort=False).any()
    moved = changed & grouped.current.max().gt(0) & grouped.prior.max().gt(0) & grouped["_cusip"].nunique().gt(1)
    df_pair["known"] &= ~pd.MultiIndex.from_arrays(keys).isin(moved.index[moved])
    starts = np.searchsorted(df_pair["_pair"].to_numpy(), np.arange(len(pairs) + 1))
    columns = ["ticker", "current", "prior", "value_current", "value_prior", "known"]
    return PairTable({pair: code for code, pair in enumerate(pairs)}, df_pair[columns], starts)


def _implied_price(df_pair: pd.DataFrame) -> pd.Series:
    """Each position's prior value per share, else its current value per share."""
    current, prior = (df_pair[f"value_{side}"] / df_pair[side] for side in ("current", "prior"))
    current, prior = (price.where(np.isfinite(price) & price.gt(0)) for price in (current, prior))
    return prior.fillna(current)


def prepare_pair_sample(selected: list[BookPair], pairs: PairTable, market: BehaviorMarket) -> PairSample:
    """Prepare this event's genuine exact-quarter pairs once for both cohort reductions."""
    if not selected:
        return PairSample(pd.DataFrame(), pd.DataFrame(), frozenset())
    codes = [pairs.codes[(current, prior)] for _, current, prior in selected]
    rows = np.concatenate([np.arange(pairs.starts[code], pairs.starts[code + 1]) for code in codes])
    ciks = np.repeat(np.array([cik for cik, _, _ in selected], dtype=object), np.diff(pairs.starts)[codes])
    df_positions = pairs.df_rows.iloc[rows].reset_index(drop=True).assign(cik=ciks)
    _, current, prior = selected[0]

    # Prior shares move onto the current basis (`F_period / F_prior`) and the reference price inversely: universe tickers
    # use F and the exact reference-session quote, other positions their implied price at F = 1. No quote is filled.
    tickers = df_positions.ticker
    quoted = tickers.isin(market.close_pit.columns if market.close_pit is not None else [])
    prior_basis = lookup_asof(market.basis, tickers, pd.Series(prior.period, index=tickers.index)).where(quoted, 1.0)
    basis_now = lookup_asof(market.basis, tickers, pd.Series(current.period, index=tickers.index)).where(quoted, 1.0)
    df_positions["prior"] = convert_quantities(df_positions.prior, prior_basis, basis_now)
    position = int(market.trading_index.searchsorted(prior.period, side="right")) - 1
    prices = pd.Series(dtype="float64")
    if position >= 0 and market.close_pit is not None:
        prices = market.close_pit.reindex(index=[market.trading_index[position]]).iloc[0]
    df_positions["price"] = convert_quantities(tickers.map(prices).where(quoted, _implied_price(df_positions)), basis_now, prior_basis)
    df_positions["known"] &= np.isfinite(df_positions.prior) & df_positions.prior.ge(0)
    grouped = df_positions.groupby(["cik", "ticker"], sort=False)
    df_tickers = grouped[["current", "prior"]].sum(min_count=1)
    df_tickers["known"] = grouped.known.all()
    return PairSample(df_positions, df_tickers, frozenset(cik for cik, _, _ in selected))


def quantity_metrics(sample: PairSample, cohort: frozenset[str], universe: pd.Index) -> dict[str, pd.Series]:
    """Compare matched filed quantities, keeping entries, exits and unknowns distinct."""
    result = {name: pd.Series(np.nan, index=universe, dtype="float64") for name in ("breadth_chg", "shares_chg", "trade_balance")}
    count = len(sample.managers & cohort)
    if not count:
        return result
    df_tickers = sample.df_tickers.loc[sample.df_tickers.index.get_level_values("cik").isin(cohort)]
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


def prepare_whole_book_sample(sample: PairSample) -> WholeBookSample:
    """Prepare whole-manager eligibility and fixed-price amounts before cohort masks."""
    if sample.df_positions.empty:
        return WholeBookSample(pd.DataFrame(), pd.DataFrame())
    df_positions = sample.df_positions
    priced = np.isfinite(df_positions.price) & df_positions.price.gt(0)
    supported = df_positions.known & priced
    eligible = supported.groupby(df_positions.cik).all()
    df_positions = df_positions.loc[df_positions.cik.isin(eligible.index[eligible])]
    if df_positions.empty:
        return WholeBookSample(pd.DataFrame(), pd.DataFrame())

    # A missing union position excludes its manager rather than shrinking the book.
    df_positions = df_positions.assign(prior_value=df_positions.prior * df_positions.price, current_value=df_positions.current * df_positions.price)
    df_books = df_positions.groupby("cik")[["prior_value", "current_value"]].sum()
    df_tickers = df_positions.groupby(["cik", "ticker"])[["prior", "current", "prior_value", "current_value"]].sum()
    growth = df_books.current_value / df_books.prior_value.where(df_books.prior_value.gt(0)) - 1
    df_tickers["relative"] = (
        df_tickers.current / df_tickers.prior.where(df_tickers.prior.gt(0)) - 1 - df_tickers.index.get_level_values("cik").map(growth)
    )
    return WholeBookSample(df_tickers, df_books)


def whole_book_metrics(sample: WholeBookSample, cohort: frozenset[str], universe: pd.Index) -> tuple[dict[str, pd.Series], int]:
    """Remove endpoint price moves with one common fixed-price whole-book union."""
    result = {name: pd.Series(np.nan, index=universe, dtype="float64") for name in ("relative_position_chg", "pooled_weight_chg")}
    df_books = sample.df_books.loc[sample.df_books.index.isin(cohort)]
    if df_books.empty:
        return result, 0
    df_tickers = sample.df_tickers.loc[sample.df_tickers.index.get_level_values("cik").isin(cohort)]
    prior_total, current_total = df_books.prior_value.sum(), df_books.current_value.sum()
    result["relative_position_chg"] = df_tickers.relative.groupby(level="ticker").mean().reindex(universe)
    if prior_total > 0 and current_total > 0:
        pooled = df_tickers.groupby(level="ticker")[["prior_value", "current_value"]].sum().reindex(universe, fill_value=0)
        result["pooled_weight_chg"] = pooled.current_value / current_total - pooled.prior_value / prior_total
    return result, len(df_books)
