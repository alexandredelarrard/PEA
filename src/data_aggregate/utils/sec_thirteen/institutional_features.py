"""Broad point-in-time 13F levels and matched-quarter changes."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any, cast

import numpy as np
import pandas as pd

from src.constants.constants import SEC_13F_FILING_LAG_DAYS
from src.data_aggregate.utils.common.availability import availability_date
from src.data_aggregate.utils.common.pit import fundamentals_to_daily
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


def _mask_daily_institutional_fields(
    fields: dict[str, pd.DataFrame], trading_index: pd.DatetimeIndex, settle_trading_days: int
) -> dict[str, pd.DataFrame]:
    """Apply publication-based level and delta floors to every daily field."""
    floors = availability_date(
        pd.DatetimeIndex([INST_LEVEL_FLOOR_PERIOD, INST_DELTA_FLOOR_PERIOD]), trading_index, settle_trading_days=settle_trading_days
    )
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
    versioned: bool,
    trading_calendar: pd.DatetimeIndex,
) -> dict[str, pd.DataFrame]:
    """Project quarter features and the existing ownership/value denominators onto daily dates."""
    features = [column for column in EMISSION if column in df_features.columns]
    value_fields = {"ic_inst_concentration", "ic_inst_net_options_ratio"}
    fields = {
        feature: (nullable_value_to_daily if versioned or feature in value_fields else fundamentals_to_daily)(
            df_features, feature, frames.trading_index
        )
        for feature in features
    }
    if frames.shares_outstanding_pit is not None and not frames.shares_outstanding_pit.empty and "inst_shares" in df_features:
        df_owned = _daily_owned_shares(df_features, frames.trading_index, frames.security_basis)
        df_shares = frames.shares_outstanding_pit.reindex(frames.trading_index)
        if not df_shares.empty and df_shares.notna().any().any():
            df_supported_shares = df_shares.where(np.isfinite(df_shares) & df_shares.gt(0))
            fields["ic_inst_ownership_pct"] = _capped_ownership((df_owned / df_supported_shares).replace([np.inf, -np.inf], np.nan))
    df_weight = _daily_institutional_value_weight(df_features, frames)
    if df_weight is not None:
        fields["ic_inst_value_to_mcap"] = df_weight
    return _mask_daily_institutional_fields(fields, trading_calendar, settle_trading_days)


def _institutional_levels(df_current: pd.DataFrame, df_states: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Reduce numeric columns natively, preserving any unknown contributor."""
    grouped = df_current.groupby("ticker")
    numeric = ["shares", "value_usd", "call_value", "put_value"]
    df_totals = grouped[numeric].sum(min_count=1).where(grouped[numeric].count().eq(grouped.size(), axis=0))
    df_out = pd.DataFrame(index=pd.Index(names, name="ticker"))
    df_out["inst_shares"] = df_totals.shares
    df_out["inst_value"] = df_totals.value_usd
    holders = df_current.shares.gt(0).groupby("ticker").sum()
    df_out["ic_inst_holders"] = (holders / len(df_states)).where(df_totals.shares.notna())
    if df_states.zero_eligible.all():
        absent = ~df_out.index.isin(df_current.index.get_level_values("ticker"))
        df_out.loc[absent, ["inst_shares", "inst_value", "ic_inst_holders"]] = 0.0
    value_total = df_totals.value_usd.reindex(df_current.index.get_level_values("ticker")).set_axis(df_current.index)
    concentration = (df_current.value_usd / value_total.where(value_total.gt(0))).pow(2).groupby("ticker").sum(min_count=1)
    df_out["ic_inst_concentration"] = concentration.where(df_out.inst_value.notna())
    investment = df_out.inst_value + df_totals.call_value + df_totals.put_value
    df_out["ic_inst_net_options_ratio"] = (df_totals.call_value - df_totals.put_value) / investment.where(investment.gt(0))
    return df_out


def _institutional_votes(df_matched: pd.DataFrame) -> pd.DataFrame:
    """Share one native holder/trading reduction across the matched flow metrics."""
    held, prior = df_matched.shares_current.gt(0), df_matched.shares_prior.gt(0)
    return (
        pd.DataFrame(
            {
                "held": held,
                "prior": prior,
                "new": held & df_matched.shares_prior.eq(0),
                "exit": df_matched.shares_current.eq(0) & prior,
                "increase": held & prior & df_matched.shares_current.gt(df_matched.shares_prior),
                "decrease": held & prior & df_matched.shares_current.lt(df_matched.shares_prior),
            }
        )
        .groupby("ticker")
        .sum()
    )


def _quarter_basis_ratio(df_basis: pd.DataFrame | None, tickers: pd.Index, period: pd.Timestamp, prior_period: pd.Timestamp) -> pd.Series:
    """`F_period / F_prior` per ticker: the factor carrying prior-quarter shares onto the current quarter's basis."""
    if df_basis is None:
        return pd.Series(1.0, index=tickers)
    names = pd.Series(tickers, index=tickers)
    prior = lookup_asof(df_basis, names, pd.Series(prior_period, index=tickers))
    return convert_quantities(1.0, prior, lookup_asof(df_basis, names, pd.Series(period, index=tickers)))


def compute_institutional_feature_history(
    snapshots: Iterator[tuple[pd.Timestamp, pd.DataFrame, pd.DataFrame]],
    *,
    df_basis: pd.DataFrame | None,
    min_prior_holders: int = MIN_PRIOR_HOLDERS,
) -> pd.DataFrame:
    """Aggregate observed levels and identical eligible book cohorts at each public event."""
    rows: list[pd.DataFrame] = []
    seen: set[str] = set()
    numeric = ["shares", "value_usd", "call_value", "put_value"]
    split_period: pd.Timestamp | None = None
    split_factors = pd.Series(dtype="float64")
    for day, df_compact, states in snapshots:
        known = states.loc[states.book_known]
        if known.empty:
            continue
        period = known.period.max()
        prior_period = (period.to_period("Q") - 1).end_time.normalize()
        current_states = known.loc[known.period.eq(period)].set_index("cik")
        prior_states = known.loc[known.period.eq(prior_period)].set_index("cik")
        totals = df_compact.drop(columns="_cusips").sort_index()
        securities = df_compact["_cusips"].dropna().sort_index()
        current = (
            cast(pd.DataFrame, totals.xs(period)).copy() if period in totals.index.get_level_values("period") else totals.droplevel("period").iloc[:0]
        )
        prior = cast(pd.DataFrame, totals.xs(prior_period)).copy() if prior_period in totals.index.get_level_values("period") else current.iloc[:0]
        observed = current[["shares", "call_value", "put_value"]].gt(0).any(axis=1)
        seen.update(current.loc[observed].index.get_level_values("ticker"))
        names = sorted(seen)
        if not names:
            continue
        out = _institutional_levels(current, current_states, names)
        paired = current_states.index[current_states.comparison_eligible & prior_states.comparison_eligible.reindex(current_states.index).eq(True)]
        joined = current.add_suffix("_current").join(prior.add_suffix("_prior"), how="outer")
        joined = joined.loc[joined.index.get_level_values("cik").isin(paired)]
        for suffix, values, state in (("current", current, current_states), ("prior", prior, prior_states)):
            absent = ~joined.index.isin(values.index)
            zeros = state.zero_eligible.reindex(joined.index.get_level_values("cik")).eq(True).to_numpy()
            joined.loc[absent & zeros, [col + "_" + suffix for col in numeric]] = 0.0
        if split_period != period:
            split_period, split_factors = period, pd.Series(dtype="float64")
        tickers = joined.index.get_level_values("ticker")
        missing = tickers.unique().difference(split_factors.index, sort=False)
        if len(missing):
            split_factors = pd.concat([split_factors, _quarter_basis_ratio(df_basis, missing, period, prior_period)])
        joined["shares_prior"] *= split_factors.reindex(tickers).to_numpy()
        if period in securities.index.get_level_values("period") and prior_period in securities.index.get_level_values("period"):
            now_ids = securities.xs(period).reindex(joined.index)
            old_ids = securities.xs(prior_period).reindex(joined.index)
            changed = now_ids.notna() & old_ids.notna() & now_ids.ne(old_ids)
            joined.loc[changed, ["shares_current", "shares_prior"]] = np.nan
        supported = joined.shares_current.notna() & joined.shares_prior.notna()
        matched = joined.loc[supported]
        groups = matched.groupby("ticker")
        df_share_totals = groups[["shares_current", "shares_prior"]].sum(min_count=1)
        current_shares, previous_shares = df_share_totals.shares_current, df_share_totals.shares_prior
        df_votes = _institutional_votes(matched)
        holder_count, prior_count = df_votes.held, df_votes.prior
        out["ic_inst_shares_chg"] = current_shares / previous_shares.where(previous_shares.gt(0)) - 1
        zero_books = current_states.zero_eligible.reindex(paired) & prior_states.zero_eligible.reindex(paired)
        safe_absence = zero_books.reindex(joined.index.get_level_values("cik")).to_numpy(dtype=bool)
        support_count = (
            int(zero_books.sum())
            + pd.Series(
                (supported.to_numpy() & ~safe_absence).astype(int) - (~supported.to_numpy() & safe_absence).astype(int),
                index=joined.index,
            )
            .groupby("ticker")
            .sum()
        )
        out["ic_inst_breadth_chg"] = (holder_count - prior_count) / support_count.where(support_count.gt(0))
        out["ic_inst_new_buyer_ratio"] = df_votes["new"] / holder_count.where(holder_count.gt(0))
        out["ic_inst_exit_ratio"] = df_votes.exit / prior_count.where(prior_count.gt(0))
        out["ic_inst_cluster_buying"] = (df_votes.increase - df_votes.decrease) / holder_count.where(holder_count.gt(0))
        out.loc[prior_count.reindex(out.index).fillna(0).lt(min_prior_holders), list(DELTA_FEATURES)] = np.nan
        out["period"], out["as_of"] = period, day
        rows.append(out.reset_index())
    result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    logger.info("13F immutable publication: %s ticker states; unfiled/restricted books never imply exits.", len(result))
    return result
