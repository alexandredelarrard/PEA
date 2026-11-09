"""Broad point-in-time 13F levels and matched-quarter changes."""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
import pandas as pd

from src.constants.constants import SEC_13F_FILING_LAG_DAYS
from src.data_aggregate.utils.common.availability import availability_date
from src.data_aggregate.utils.common.quotes import lookup_asof
from src.data_aggregate.utils.price.frames import PriceFrames
from src.data_aggregate.utils.sec_thirteen.value_basis import nullable_value_to_daily
from src.data_store.schema import Resume, Tables
from src.utils.price.levels import convert_quantities

logger = logging.getLogger(__name__)

#: The first period whose all-filer coverage describes the market rather than the fetch
#: (measured above: 63 -> 2,938 in-universe filers, post-band).
INST_LEVEL_FLOOR_PERIOD = pd.Timestamp(cast(str, cast(Resume, Tables.sec13f_hr.resume).coverage_start))

#: The first period whose PREDECESSOR is also post-break, so a QoQ delta is meaningful.
INST_DELTA_FLOOR_PERIOD = INST_LEVEL_FLOOR_PERIOD + pd.offsets.QuarterEnd(1)

#: Minimum number of holders a ticker must have in the PREVIOUS quarter for its
#: quarter-over-quarter features to be considered meaningful.
#:
#: This is the per-ticker counterpart to the universe-wide D16/D17 coverage
#: guards. A newly covered stock can move from a handful of holders to hundreds,
#: causing mechanically extreme share growth, buyer, exit, and flow measures
#: that describe coverage onset rather than institutional activity.
#:
#: When the prior quarter has fewer than this threshold, all `DELTA_FEATURES`
#: are nulled. Level features remain valid: a stock may
#: genuinely have few institutional holders in that quarter.
#:
#: The threshold of 100 is based on the measured break in delta stability. It is
#: treated as a fixed data-quality boundary, not a research parameter.
MIN_PRIOR_HOLDERS = 100

#: The quarter-over-quarter features. Nulled before `INST_DELTA_FLOOR_PERIOD` (D16) and on a
#: break quarter (D17). `new_buyer_ratio` / `exit_ratio` / `cluster_buying` are all statements
#: about the PREVIOUS quarter's filer set, so they belong here with the arithmetic differences.
DELTA_FEATURES: tuple[str, ...] = (
    "ic_inst_breadth_chg",
    "ic_inst_shares_chg",
    "ic_inst_new_buyer_ratio",
    "ic_inst_exit_ratio",
    "ic_inst_cluster_buying",
)

#: `ic_inst_ownership_pct` above this many times shares outstanding is a denominator defect,
#: not a holder, and the cell is NULLED (never clipped -- see `_capped_ownership`). 2.0 rather
#: than 1.0 because securities lending double-counts a position and 100-130% is a real, widely
#: published reading; 9.68x is not.
OWNERSHIP_CEILING = 2.0

#: The level features. Nulled before `INST_LEVEL_FLOOR_PERIOD` (D16).
LEVEL_FEATURES: tuple[str, ...] = (
    "ic_inst_holders",
    "ic_inst_concentration",
    "ic_inst_net_options_ratio",
    "ic_inst_ownership_pct",
    "ic_inst_value_to_mcap",
)

#: Preserve interpretable economic units. Peer variants are not emitted: the fixed candidate
#: has no target/OOS evidence with which to pass the approved retention gate.
EMISSION: dict[str, str] = {
    "ic_inst_holders": "raw",  # D28 share of the quarter's filers, in [0, 1]
    "ic_inst_breadth_chg": "raw",  # a difference of that share
    "ic_inst_shares_chg": "raw",  # an interpretable growth rate
    "ic_inst_new_buyer_ratio": "raw",  # bounded [0, 1]
    "ic_inst_exit_ratio": "raw",  # bounded [0, 1]
    "ic_inst_cluster_buying": "raw",  # bounded [-1, 1]
    "ic_inst_concentration": "raw",  # Herfindahl concentration
    "ic_inst_net_options_ratio": "raw",  # bounded [-1, 1]
    "ic_inst_ownership_pct": "raw",
    "ic_inst_value_to_mcap": "raw",
}


def _capped_ownership(ratio: pd.DataFrame) -> pd.DataFrame:
    """Null and report ownership ratios above the existing denominator-defect ceiling."""
    over = ratio.gt(OWNERSHIP_CEILING)
    n = int(over.to_numpy().sum())
    if n:
        logger.info(
            "ownership guard: %s of %s non-null `ic_inst_ownership_pct` cells (%.3f%%) "
            "above %.1fx saved PIT shares outstanding -> nulled; review source quantity/coverage compatibility; max seen %.2f",
            f"{n:,}",
            f"{int(ratio.notna().to_numpy().sum()):,}",
            100.0 * n / max(1, int(ratio.notna().to_numpy().sum())),
            OWNERSHIP_CEILING,
            float(np.nanmax(ratio.to_numpy(dtype="float64"))),
        )
    return ratio.mask(over)


def _daily_owned_shares(qf: pd.DataFrame, trading_index: pd.DatetimeIndex, df_basis: pd.DataFrame | None) -> pd.DataFrame:
    """Carry report-date quantities to each day's basis (`q * F_d / F_period`) without changing publication clocks."""
    if df_basis is None or trading_index.empty:
        df_daily = nullable_value_to_daily(qf, "inst_shares", trading_index)
        return df_daily.where(df_daily.eq(0))

    # Project first-bar-basis quantities (F = 1), then restate each day on its own basis.
    df_owned = qf.copy()
    df_owned["inst_shares"] = convert_quantities(df_owned["inst_shares"], lookup_asof(df_basis, df_owned["ticker"], df_owned["period"]), 1.0)
    df_daily = nullable_value_to_daily(df_owned, "inst_shares", trading_index).reindex(columns=sorted(df_owned["ticker"].dropna().unique()))
    return convert_quantities(df_daily, 1.0, df_basis.reindex(index=trading_index, columns=df_daily.columns))


def _daily_institutional_value_weight(df_features: pd.DataFrame, frames: PriceFrames) -> pd.DataFrame | None:
    """Scale historical institutional dollars by the supported saved issuer cap."""
    if frames.mcap_pit is None or frames.mcap_pit.empty or "inst_value" not in df_features:
        return None
    df_market_cap = frames.mcap_pit.reindex(frames.trading_index)
    df_positive_cap = df_market_cap.where(np.isfinite(df_market_cap) & df_market_cap.gt(0))
    df_value = nullable_value_to_daily(df_features, "inst_value", frames.trading_index)
    df_weight = (df_value / df_positive_cap).replace([np.inf, -np.inf], np.nan)
    return df_weight if df_weight.notna().any().any() else None


def _mask_daily_institutional_fields(fields: dict[str, pd.DataFrame], floors: pd.Series) -> dict[str, pd.DataFrame]:
    """Apply the publication-dated level and delta floors to every daily field; drop empty fields."""
    level_floor = pd.Timestamp(cast(Any, floors.get(INST_LEVEL_FLOOR_PERIOD, INST_LEVEL_FLOOR_PERIOD + pd.Timedelta(days=SEC_13F_FILING_LAG_DAYS))))
    delta_floor = pd.Timestamp(cast(Any, floors.get(INST_DELTA_FLOOR_PERIOD, INST_DELTA_FLOOR_PERIOD + pd.Timedelta(days=SEC_13F_FILING_LAG_DAYS))))
    if pd.isna(level_floor):
        level_floor = INST_LEVEL_FLOOR_PERIOD + pd.Timedelta(days=SEC_13F_FILING_LAG_DAYS)
    if pd.isna(delta_floor):
        delta_floor = INST_DELTA_FLOOR_PERIOD + pd.Timedelta(days=SEC_13F_FILING_LAG_DAYS)
    for name, df_field in list(fields.items()):
        if df_field is None or df_field.empty:
            fields.pop(name)
            continue
        floor = delta_floor if name in DELTA_FEATURES else level_floor
        fields[name] = df_field.where(pd.Series(df_field.index >= floor, index=df_field.index), axis=0)
    return {name: df_field for name, df_field in fields.items() if df_field.notna().any().any()}


def project_institutional_features(
    df_features: pd.DataFrame,
    frames: PriceFrames,
    settle_trading_days: int,
    *,
    trading_calendar: pd.DatetimeIndex,
) -> dict[str, pd.DataFrame]:
    """Project quarter features and the existing ownership/value denominators onto daily dates."""
    features = [column for column in EMISSION if column in df_features.columns]
    fields = {feature: nullable_value_to_daily(df_features, feature, frames.trading_index) for feature in features}
    if frames.shares_outstanding_pit is not None and not frames.shares_outstanding_pit.empty and "inst_shares" in df_features:
        df_owned = _daily_owned_shares(df_features, frames.trading_index, frames.security_basis)
        df_shares = frames.shares_outstanding_pit.reindex(frames.trading_index)
        if not df_shares.empty and df_shares.notna().any().any():
            df_supported_shares = df_shares.where(np.isfinite(df_shares) & df_shares.gt(0))
            fields["ic_inst_ownership_pct"] = _capped_ownership((df_owned / df_supported_shares).replace([np.inf, -np.inf], np.nan))
    df_weight = _daily_institutional_value_weight(df_features, frames)
    if df_weight is not None:
        fields["ic_inst_value_to_mcap"] = df_weight

    # Level and delta floors at their publication dates.
    floors = availability_date(
        pd.DatetimeIndex([INST_LEVEL_FLOOR_PERIOD, INST_DELTA_FLOOR_PERIOD]), trading_calendar, settle_trading_days=settle_trading_days
    )
    return _mask_daily_institutional_fields(fields, floors)


def _book_versions(df_states: pd.DataFrame) -> tuple[pd.DataFrame, pd.DatetimeIndex, list[tuple[pd.Timestamp, int, int]]]:
    """Each public book version valid on public days [start, stop) until the next version of its manager-quarter, and
    the runs of days sharing one active quarter, the latest report quarter with a known public book."""
    days = pd.DatetimeIndex(np.sort(df_states["_day"].unique()))
    df_versions = df_states.assign(start=days.get_indexer(pd.DatetimeIndex(df_states["_day"]))).sort_values(["cik", "period", "start"], kind="stable")
    df_versions["stop"] = df_versions.groupby(["cik", "period"], sort=False)["start"].shift(-1).fillna(len(days)).astype("int64")
    df_versions = df_versions.reset_index(drop=True)

    # The active quarter is a running maximum: the later versions of a known book stay known.
    latest = df_versions.loc[df_versions["book_known"].astype(bool)].groupby("start")["period"].max()
    active = latest.reindex(range(len(days))).ffill().cummax()
    first = np.flatnonzero(active.ne(active.shift()).to_numpy())
    last = np.r_[first[1:] - 1, len(days) - 1]
    return df_versions, days, [(active.iloc[lo], int(lo), int(hi)) for lo, hi in zip(first, last, strict=True) if pd.notna(active.iloc[lo])]


def _quarter_rows(df_compact: pd.DataFrame, df_versions: pd.DataFrame, quarters: list[pd.Timestamp], tickers: pd.Index) -> pd.DataFrame:
    """Compact rows of every version of the given report quarters, tagged with that version and its ticker slot."""
    df_books = df_versions.loc[df_versions["period"].isin(quarters)]
    lengths = (df_books["_stop"] - df_books["_start"]).to_numpy(dtype=np.int64)
    first = np.repeat(df_books["_start"].to_numpy(dtype=np.int64) - np.cumsum(lengths) + lengths, lengths)
    df_rows = df_compact.iloc[first + np.arange(lengths.sum())]
    return pd.DataFrame(
        {
            "version": np.repeat(df_books.index.to_numpy(), lengths),
            **{column: np.repeat(df_books[column].to_numpy(), lengths) for column in ("period", "start", "stop")},
            "slot": tickers.get_indexer(df_rows.index.get_level_values("ticker")),
            **{column: df_rows[column].to_numpy() for column in df_compact.columns},
        }
    )


def _level_items(df_rows: pd.DataFrame, df_books: pd.DataFrame, n_tickers: int) -> pd.DataFrame:
    """Additive level statistics of each current-quarter row, plus each known book's counts in the extra last slot."""
    shares, value, call, put = (df_rows[column].to_numpy(dtype="float64") for column in ("shares", "value_usd", "call_value", "put_value"))
    amounts = {"shares": shares, "value": value, "call": call, "put": put}
    df_positions = df_rows[["start", "stop", "slot"]].assign(
        rows=1.0,
        **{"known_" + name: ~np.isnan(amount) for name, amount in amounts.items()},
        **{name: np.nan_to_num(amount) for name, amount in amounts.items()},
        holders=shares > 0,
        value_sq=np.nan_to_num(value) ** 2,
        positive_value=value > 0,
        invested=(value > 0) | (call > 0) | (put > 0),
        observed=(shares > 0) | (call > 0) | (put > 0),
    )
    df_known = df_books.loc[df_books["book_known"].astype(bool)]
    df_counts = df_known[["start", "stop"]].assign(slot=n_tickers, books=1.0, open_books=~df_known["zero_eligible"].astype(bool))
    return pd.concat([df_positions.astype("float64"), df_counts.astype("float64")], ignore_index=True).fillna(0.0)


def _pair_segments(df_versions: pd.DataFrame, period: pd.Timestamp, prior: pd.Timestamp, n_days: int) -> pd.DataFrame:
    """Per manager, the current and prior versions in force on each [start, stop) segment where both books are
    comparison-eligible, and whether each side's absence means zero."""
    df_two = df_versions.loc[df_versions["period"].isin([period, prior]), ["cik", "period", "start"]].sort_values("start", kind="stable")
    df_segments = df_two[["cik", "start"]].drop_duplicates()
    for side, quarter in (("current", period), ("prior", prior)):
        df_side = df_two.loc[df_two["period"].eq(quarter), ["cik", "start"]].rename_axis(side).reset_index()
        df_segments = pd.merge_asof(df_segments, df_side, on="start", by="cik")
    df_segments = df_segments.dropna(subset=["current", "prior"]).sort_values(["cik", "start"], kind="stable")
    df_segments["stop"] = df_segments.groupby("cik", sort=False)["start"].shift(-1).fillna(n_days).astype("int64")

    # Both books must be comparison-eligible; absence is zero only on a zero-eligible side.
    current = df_versions.loc[df_segments["current"].astype("int64")]
    previous = df_versions.loc[df_segments["prior"].astype("int64")]
    df_segments["zero_current"] = current["zero_eligible"].to_numpy(dtype=bool)
    df_segments["zero_prior"] = previous["zero_eligible"].to_numpy(dtype=bool)
    eligible = current["comparison_eligible"].to_numpy(dtype=bool) & previous["comparison_eligible"].to_numpy(dtype=bool)
    return df_segments.loc[eligible].reset_index(drop=True)


def _pair_items(df_rows: pd.DataFrame, df_segments: pd.DataFrame, factors: np.ndarray, n_tickers: int) -> pd.DataFrame:
    """Additive matched-quarter statistics of each manager x ticker on each paired segment, plus each zero-eligible
    pair in the extra last slot."""
    sides = [df_segments[[side]].reset_index(names="segment").merge(df_rows, left_on=side, right_on="version") for side in ("current", "prior")]
    keep = ["segment", "slot", "shares", "_cusips"]
    df_pairs = sides[0][keep].merge(sides[1][keep], on=["segment", "slot"], how="outer", suffixes=("_current", "_prior"), indicator=True)
    segment, slot = df_pairs["segment"].to_numpy(dtype=np.int64), df_pairs["slot"].to_numpy(dtype=np.int64)
    safe = (df_segments["zero_current"] & df_segments["zero_prior"]).to_numpy(dtype=bool)

    # Zero-fill a side's absence on its zero-eligible book, restate prior shares, null a changed security.
    zero_current = df_pairs["_merge"].eq("right_only").to_numpy() & df_segments["zero_current"].to_numpy(dtype=bool)[segment]
    zero_prior = df_pairs["_merge"].eq("left_only").to_numpy() & df_segments["zero_prior"].to_numpy(dtype=bool)[segment]
    ids = df_pairs[["_cusips_current", "_cusips_prior"]]
    changed = (ids.notna().all(axis=1) & ids["_cusips_current"].ne(ids["_cusips_prior"])).to_numpy(dtype=bool)
    current = np.where(changed, np.nan, np.where(zero_current, 0.0, df_pairs["shares_current"].to_numpy(dtype="float64", na_value=np.nan)))
    prior = np.where(changed, np.nan, np.where(zero_prior, 0.0, df_pairs["shares_prior"].to_numpy(dtype="float64", na_value=np.nan)) * factors[slot])

    matched = ~np.isnan(current) & ~np.isnan(prior)
    held, was = matched & (current > 0), matched & (prior > 0)
    df_items = pd.DataFrame(
        {
            "start": df_segments["start"].to_numpy()[segment],
            "stop": df_segments["stop"].to_numpy()[segment],
            "slot": slot,
            "matched": matched,
            "shares_current": np.where(matched, current, 0.0),
            "shares_prior": np.where(matched, prior, 0.0),
            "held": held,
            "prior": was,
            "new": held & (prior == 0),
            "exit": matched & (current == 0) & was,
            "increase": held & was & (current > prior),
            "decrease": held & was & (current < prior),
            "support": (matched & ~safe[segment]).astype("float64") - (~matched & safe[segment]),
        },
        dtype="float64",
    )
    df_zero = df_segments.loc[safe, ["start", "stop"]].assign(slot=n_tickers, zero_pairs=1.0).astype("float64")
    return pd.concat([df_items, df_zero], ignore_index=True).fillna(0.0)


def _accumulate(df_items: pd.DataFrame, window: tuple[int, int], n_slots: int) -> dict[str, np.ndarray]:
    """Per (window day, slot) sums of the statistics each item adds on its `start` day and removes on its `stop` day."""
    lo, hi = window
    start = df_items["start"].clip(lo, hi + 1).to_numpy(dtype=np.int64) - lo
    stop = df_items["stop"].clip(lo, hi + 1).to_numpy(dtype=np.int64) - lo
    live = start < stop
    names = [name for name in df_items.columns if name not in ("start", "stop", "slot")]
    values = df_items.loc[live, names].to_numpy(dtype="float64")
    cells = np.concatenate([start[live], stop[live]]) * n_slots + np.tile(df_items["slot"].to_numpy(dtype=np.int64)[live], 2)
    size = (hi + 2 - lo) * n_slots
    return {
        name: np.bincount(cells, np.concatenate([values[:, k], -values[:, k]]), size).reshape(-1, n_slots).cumsum(axis=0)[:-1]
        for k, name in enumerate(names)
    }


def _window_features(
    state: dict[str, np.ndarray], seen: np.ndarray, tickers: pd.Index, as_of: pd.DatetimeIndex, period: pd.Timestamp, min_prior_holders: int
) -> tuple[pd.DataFrame, np.ndarray]:
    """Levels and matched-quarter changes of every ticker ever observed, on each day of one active-quarter window."""
    s = {name: values[:, :-1] for name, values in state.items()}
    books, open_books, zero_pairs = (state[name][:, -1:] for name in ("books", "open_books", "zero_pairs"))
    seen_days = np.logical_or.accumulate(s["observed"] > 0, axis=0) | seen
    present = s["rows"] > 0

    # Totals stay unknown with any unknown contributor; absence is zero only when every known book is zero-eligible.
    shares, value, call, put = (np.where(present & (s["known_" + name] == s["rows"]), s[name], np.nan) for name in ("shares", "value", "call", "put"))
    absent_zero = ~present & (open_books == 0)
    holders = np.where(absent_zero, 0.0, np.where(np.isnan(shares), np.nan, s["holders"] / books))
    shares, value = np.where(absent_zero, 0.0, shares), np.where(absent_zero, 0.0, value)

    # Matched-quarter flows over supported manager pairs; every threshold reads an integer count.
    matched = s["matched"] > 0
    held, was = np.where(matched, s["held"], np.nan), np.where(matched, s["prior"], np.nan)
    support = zero_pairs + s["support"]
    thin = np.nan_to_num(was) < min_prior_holders
    with np.errstate(divide="ignore", invalid="ignore"):
        flows = {
            "ic_inst_shares_chg": np.where(matched & (s["prior"] > 0), s["shares_current"] / s["shares_prior"] - 1, np.nan),
            "ic_inst_breadth_chg": (held - was) / np.where(support > 0, support, np.nan),
            "ic_inst_new_buyer_ratio": np.where(matched, s["new"], np.nan) / np.where(held > 0, held, np.nan),
            "ic_inst_exit_ratio": np.where(matched, s["exit"], np.nan) / np.where(was > 0, was, np.nan),
            "ic_inst_cluster_buying": np.where(matched, s["increase"] - s["decrease"], np.nan) / np.where(held > 0, held, np.nan),
        }
        columns = {
            "inst_shares": shares,
            "inst_value": value,
            "ic_inst_holders": holders,
            "ic_inst_concentration": np.where((s["positive_value"] > 0) & ~np.isnan(value), s["value_sq"] / value**2, np.nan),
            "ic_inst_net_options_ratio": (call - put) / np.where(s["invested"] > 0, value + call + put, np.nan),
            **{name: np.where(thin, np.nan, values) for name, values in flows.items()},
        }

    day, slot = np.nonzero(seen_days)
    df_window = pd.DataFrame({"ticker": tickers[slot], **{name: values[day, slot] for name, values in columns.items()}})
    df_window["period"], df_window["as_of"] = period, as_of[day]
    return df_window, seen_days[-1]


def compute_institutional_feature_history(
    df_compact: pd.DataFrame,
    df_states: pd.DataFrame,
    *,
    df_basis: pd.DataFrame | None,
    min_prior_holders: int = MIN_PRIOR_HOLDERS,
) -> pd.DataFrame:
    """Accumulate each public book version's per-ticker statistics from its public day until a later version of the
    same manager-quarter supersedes it, then reduce each active quarter's daily totals to levels and matched changes."""
    if df_states.empty:
        return pd.DataFrame()
    df_versions, days, windows = _book_versions(df_states)
    tickers = pd.Index(np.sort(df_compact.index.get_level_values("ticker").unique()))
    seen = np.zeros(len(tickers), dtype=bool)
    frames: list[pd.DataFrame] = []

    for period, lo, hi in windows:
        # Rows of the active quarter and its predecessor; F_period / F_prior carries prior shares onto this quarter.
        prior = (period.to_period("Q") - 1).end_time.normalize()
        df_rows = _quarter_rows(df_compact, df_versions, [period, prior], tickers)
        factors = np.ones(len(tickers))
        if df_basis is not None:
            keys = pd.Series(tickers, index=tickers)
            prior_basis, basis = (
                lookup_asof(df_basis, keys, pd.Series(quarter, index=tickers)).to_numpy(dtype="float64") for quarter in (prior, period)
            )
            factors = np.asarray(convert_quantities(factors, prior_basis, basis), dtype="float64")

        # Each day's totals are the running sum of contributions added when public and removed when superseded.
        df_levels = _level_items(df_rows.loc[df_rows["period"].eq(period)], df_versions.loc[df_versions["period"].eq(period)], len(tickers))
        df_pairs = _pair_items(df_rows, _pair_segments(df_versions, period, prior, len(days)), factors, len(tickers))
        state = {**_accumulate(df_levels, (lo, hi), len(tickers) + 1), **_accumulate(df_pairs, (lo, hi), len(tickers) + 1)}
        df_window, seen = _window_features(state, seen, tickers, days[lo : hi + 1], period, min_prior_holders)
        frames.append(df_window)

    df_history = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    logger.info("13F immutable publication: %s ticker states; unfiled/restricted books never imply exits.", len(df_history))
    return df_history if len(df_history) else pd.DataFrame()
