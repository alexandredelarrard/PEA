"""Broad point-in-time 13F levels, matched-quarter changes, and publication previews."""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from typing import Any, cast

import numpy as np
import pandas as pd

from src.constants.constants import F13_SETTLE_TRADING_DAYS, SEC_13F_FILING_LAG_DAYS
from src.data_aggregate.utils.common.availability import availability_date, filing_public_dates, public_books
from src.data_aggregate.utils.common.pit import fundamentals_to_daily
from src.data_aggregate.utils.common.split_basis import split_cum_product
from src.data_aggregate.utils.price.frames import PriceFrames
from src.data_aggregate.utils.sec_thirteen.value_basis import nullable_value_to_daily
from src.data_store.schema import Resume, Tables
from src.utils.price.adjustments import ShareBasis
from src.utils.price.levels import rebase_security_quantities

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


def _daily_owned_shares(qf: pd.DataFrame, trading_index: pd.DatetimeIndex, share_basis: ShareBasis | None) -> pd.DataFrame:
    """Carry supported quoted quantities from report date without changing publication clocks."""
    if share_basis is None or trading_index.empty:
        df_daily = nullable_value_to_daily(qf, "inst_shares", trading_index)
        return df_daily.where(df_daily.eq(0))

    # Normalize every supported ticker before projecting public replacement observations.
    df_owned = qf.copy()
    periods = pd.to_datetime(df_owned["period"])
    terminal = pd.Timestamp(trading_index.max())
    targets = pd.Series(terminal, index=df_owned.index)
    units = share_basis.factor(df_owned["ticker"], periods, "units")
    quantities = rebase_security_quantities(share_basis, df_owned["inst_shares"], df_owned["ticker"], periods, targets, units)
    df_owned["inst_shares"] = quantities.where(units.eq(1.0))
    df_daily = nullable_value_to_daily(df_owned, "inst_shares", trading_index).reindex(columns=sorted(df_owned["ticker"].dropna().unique()))

    # Aggregated books lack the declared title required to establish nonunit ADS quantities.
    for ticker in df_daily.columns:
        names = pd.Series(ticker, index=trading_index)
        dates = pd.Series(trading_index, index=trading_index)
        source = pd.Series(terminal, index=trading_index)
        declared = share_basis.factor(names, source, "units")
        df_daily[ticker] = rebase_security_quantities(share_basis, df_daily[ticker], names, source, dates, declared)
        df_daily[ticker] = df_daily[ticker].where(share_basis.factor(names, dates, "units").eq(1.0))
    return df_daily


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
    share_basis: ShareBasis | None,
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
        df_owned = _daily_owned_shares(df_features, frames.trading_index, share_basis)
        df_shares = frames.shares_outstanding_pit.reindex(frames.trading_index)
        if not df_shares.empty and df_shares.notna().any().any():
            df_supported_shares = df_shares.where(np.isfinite(df_shares) & df_shares.gt(0))
            fields["ic_inst_ownership_pct"] = _capped_ownership((df_owned / df_supported_shares).replace([np.inf, -np.inf], np.nan))
    df_weight = _daily_institutional_value_weight(df_features, frames)
    if df_weight is not None:
        fields["ic_inst_value_to_mcap"] = df_weight
    return _mask_daily_institutional_fields(fields, trading_calendar, settle_trading_days)


def vintage_changes(
    manifests: pd.DataFrame,
    holdings: pd.DataFrame,
    as_of: object,
    trading_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Scoped manager/CUSIP preview: current levels and causally resolved matched-quarter flows."""
    books, states = public_books(manifests, holdings, as_of, trading_index)
    output: list[pd.DataFrame] = []
    for cik, group in states.groupby("cik", sort=True):
        usable = group.loc[group["book_known"]].sort_values("period")
        if usable.empty:
            continue
        current = usable.iloc[-1]
        prior_period = (pd.Timestamp(current.period).to_period("Q") - 1).end_time.normalize()
        prior_state = group.loc[group["period"].eq(prior_period) & group["book_known"]]
        cur = books.loc[books.cik.eq(cik) & books.period.eq(current.period)].set_index("cusip")
        prev = books.loc[books.cik.eq(cik) & books.period.eq(prior_period)].set_index("cusip")
        names = cur.index.union(prev.index)
        result = pd.DataFrame(index=names)
        result["shares"] = cur["shares"].reindex(names)
        result["prior_shares"] = prev["shares"].reindex(names)
        paired = not prior_state.empty and bool(current.comparison_eligible) and bool(prior_state.iloc[0].comparison_eligible)
        if current.zero_eligible:
            result.loc[~result.index.isin(cur.index), "shares"] = 0.0
        if not prior_state.empty and prior_state.iloc[0].zero_eligible:
            result.loc[~result.index.isin(prev.index), "prior_shares"] = 0.0
        result["comparison_known"] = paired & result["shares"].notna() & result["prior_shares"].notna()
        result["shares_change"] = (result["shares"] - result["prior_shares"]).where(result["comparison_known"])
        result["reported_exit"] = result["comparison_known"] & result["shares"].eq(0) & result["prior_shares"].gt(0)
        result["value_usd"] = cur["value_usd"].reindex(names)
        result["prior_value_usd"] = prev["value_usd"].reindex(names)
        if current.zero_eligible:
            result.loc[~result.index.isin(cur.index), "value_usd"] = 0.0
        if not prior_state.empty and prior_state.iloc[0].zero_eligible:
            result.loc[~result.index.isin(prev.index), "prior_value_usd"] = 0.0
        result["value_change"] = (result["value_usd"] - result["prior_value_usd"]).where(result["comparison_known"])
        result["cik"] = cik
        result["period"] = current.period
        result["prior_period"] = prior_period
        result["as_of"] = pd.Timestamp(as_of)
        result["current_public_at"] = current.public_at
        result["resolution_status"] = current.resolution_status
        result["publication_age_days"] = (pd.Timestamp(as_of) - pd.Timestamp(current.public_at)).days
        result["prior_public_at"] = prior_state.iloc[0].public_at if not prior_state.empty else pd.NaT
        result["prior_resolution_status"] = prior_state.iloc[0].resolution_status if not prior_state.empty else "missing_original"
        output.append(result.reset_index())
    columns = [
        "cusip",
        "shares",
        "prior_shares",
        "shares_change",
        "reported_exit",
        "comparison_known",
        "value_usd",
        "prior_value_usd",
        "value_change",
        "cik",
        "period",
        "prior_period",
        "as_of",
        "current_public_at",
        "publication_age_days",
        "prior_public_at",
        "resolution_status",
        "prior_resolution_status",
    ]
    return pd.concat(output, ignore_index=True)[columns] if output else pd.DataFrame(columns=columns)


def vintage_publications(
    manifests: pd.DataFrame,
    holdings: pd.DataFrame,
    trading_index: pd.DatetimeIndex,
    *,
    until: object,
) -> pd.DataFrame:
    """Emit per-manager snapshots for relevant public vintages, including value-only corrections."""
    public_dates = filing_public_dates(manifests, trading_index)
    frames: list[pd.DataFrame] = []
    for day in public_dates.dropna().drop_duplicates().sort_values():
        if day > pd.Timestamp(until):
            continue
        preview = vintage_changes(manifests, holdings, day, trading_index)
        active = preview[["cik", "period", "prior_period"]].drop_duplicates().rename(columns={"period": "current_period"})
        events = manifests.loc[public_dates.eq(day), ["cik", "period"]].copy()
        events["period"] = pd.to_datetime(events["period"])
        relevant = events.merge(active, on="cik", how="inner")
        matches = relevant["period"].eq(relevant["current_period"]) | relevant["period"].eq(relevant["prior_period"])
        ciks = relevant.loc[matches, "cik"]
        if not ciks.empty:
            frames.append(preview.loc[preview["cik"].isin(ciks)])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _cohort_quantities(df_books: pd.DataFrame, cusip: str, report: pd.Timestamp, column: str, zero: pd.Series, managers: pd.Index) -> pd.Series:
    """Read scoped book quantities; only verified public absence establishes zero."""
    df_selected = df_books.loc[df_books.period.eq(report) & df_books.cusip.eq(cusip)].set_index("cik")
    amounts = df_selected[column].reindex(managers).astype("float64")
    amounts.loc[~managers.isin(df_selected.index) & zero.eq(True)] = 0.0
    return amounts


def _cohort_total(amounts: pd.Series) -> float:
    """Preserve unknown contributions instead of treating them as zero."""
    return float(amounts.sum()) if len(amounts) and amounts.notna().all() else np.nan


def vintage_cohort_snapshot(
    manifests: pd.DataFrame,
    holdings: pd.DataFrame,
    as_of: object,
    trading_index: pd.DatetimeIndex,
    *,
    security_tickers: Mapping[str, str],
    period: object | None = None,
    splits: pd.DataFrame | None = None,
    settle_days: int = F13_SETTLE_TRADING_DAYS,
) -> pd.DataFrame:
    """Preview complete public books for verified stable CUSIPs and supplied genuine splits."""
    if len(set(security_tickers.values())) != len(security_tickers):
        raise ValueError("Preview requires one verified stable CUSIP per ticker; changed identities need a verified bridge")
    books, states = public_books(manifests, holdings, as_of, trading_index, settle_days)
    known = states.loc[states.book_known]
    if known.empty or not security_tickers:
        return pd.DataFrame()
    current_period = pd.Timestamp(period) if period is not None else known.period.max()
    prior_period = (current_period.to_period("Q") - 1).end_time.normalize()
    current_states = known.loc[known.period.eq(current_period)].set_index("cik")
    all_prior_states = known.loc[known.period.eq(prior_period)].set_index("cik")
    if current_states.empty:
        return pd.DataFrame()
    ciks = current_states.index
    prior_states = all_prior_states.reindex(ciks)
    paired = current_states.comparison_eligible & prior_states.comparison_eligible.eq(True)

    tickers = pd.Series(list(security_tickers.values()))
    factors = split_cum_product(splits, tickers, pd.Series(current_period, index=tickers.index)) / split_cum_product(
        splits, tickers, pd.Series(prior_period, index=tickers.index)
    )
    rows: list[dict[str, Any]] = []
    for (cusip, ticker), factor in zip(security_tickers.items(), factors, strict=True):
        current = _cohort_quantities(books, cusip, current_period, "shares", current_states.zero_eligible, ciks)
        prior = _cohort_quantities(books, cusip, prior_period, "shares", prior_states.zero_eligible, ciks) * factor
        value = _cohort_quantities(books, cusip, current_period, "value_usd", current_states.zero_eligible, ciks)
        prior_value = _cohort_quantities(books, cusip, prior_period, "value_usd", prior_states.zero_eligible, ciks)
        all_prior = _cohort_quantities(books, cusip, prior_period, "shares", all_prior_states.zero_eligible, all_prior_states.index) * factor
        all_prior_total = _cohort_total(all_prior)
        comparable = paired & current.notna() & prior.notna()
        cur, prev = current.loc[comparable], prior.loc[comparable]
        cur_total, prev_total = _cohort_total(cur), _cohort_total(prev)
        cur_value, prev_value = _cohort_total(value.loc[comparable]), _cohort_total(prior_value.loc[comparable])
        holders, old_holders = int(cur.gt(0).sum()), int(prev.gt(0).sum())
        both = cur.gt(0) & prev.gt(0)
        rows.append(
            {
                "as_of": pd.Timestamp(as_of),
                "period": current_period,
                "prior_period": prior_period,
                "cusip": cusip,
                "ticker": ticker,
                "public_books": len(ciks),
                "paired_books": int(paired.sum()),
                "matched_books": int(comparable.sum()),
                "share_support": int(current.notna().sum()),
                "value_support": int(value.notna().sum()),
                "shares": _cohort_total(current),
                "value_usd": _cohort_total(value),
                "public_prior_books": len(all_prior_states),
                "prior_share_support": int(all_prior.notna().sum()),
                "public_prior_shares": all_prior_total,
                "prior_share_coverage": float(prev.sum()) / all_prior_total if all_prior_total > 0 else np.nan,
                "publication_reference_at": current_states.public_at.median(),
                "publication_age_days": float((pd.Timestamp(as_of) - current_states.public_at).dt.days.median()),
                "matched_shares": cur_total,
                "matched_prior_shares": prev_total,
                "matched_value_usd": cur_value,
                "matched_prior_value_usd": prev_value,
                "shares_chg": cur_total / prev_total - 1 if prev_total > 0 else np.nan,
                "value_chg": cur_value / prev_value - 1 if prev_value > 0 else np.nan,
                "matched_holders": holders,
                "matched_prior_holders": old_holders,
                "new_ratio": float((cur.gt(0) & prev.eq(0)).sum()) / holders if holders else np.nan,
                "exit_ratio": float((cur.eq(0) & prev.gt(0)).sum()) / old_holders if old_holders else np.nan,
                "cluster_buying": float((both & cur.gt(prev)).sum() - (both & cur.lt(prev)).sum()) / holders if holders else np.nan,
                "breadth_chg": (holders - old_holders) / len(cur) if len(cur) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def vintage_cohort_publications(
    manifests: pd.DataFrame,
    holdings: pd.DataFrame,
    trading_index: pd.DatetimeIndex,
    *,
    until: object,
    security_tickers: Mapping[str, str],
    splits: pd.DataFrame | None = None,
    settle_days: int = F13_SETTLE_TRADING_DAYS,
) -> pd.DataFrame:
    """Publish changed cohort states, preserving unsupported quantities and values."""
    frames: list[pd.DataFrame] = []
    previous: pd.DataFrame | None = None
    days = filing_public_dates(manifests, trading_index, settle_days).dropna().drop_duplicates().sort_values()
    for day in days.loc[days.le(pd.Timestamp(until))]:
        snapshot = vintage_cohort_snapshot(
            manifests, holdings, day, trading_index, security_tickers=security_tickers, splits=splits, settle_days=settle_days
        )
        if snapshot.empty:
            continue
        # Passage of time alone is not a filing revision; age advances between events.
        state = snapshot.drop(columns=["as_of", "publication_age_days"])
        if previous is None or not state.equals(previous):
            frames.append(snapshot)
            previous = state
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def compute_institutional_feature_history(
    snapshots: Iterator[tuple[pd.Timestamp, pd.DataFrame, pd.DataFrame]],
    *,
    splits: pd.DataFrame | None,
    min_prior_holders: int = MIN_PRIOR_HOLDERS,
) -> pd.DataFrame:
    """Aggregate observed levels and identical eligible book cohorts at each public event."""
    rows: list[pd.DataFrame] = []
    seen: set[str] = set()
    numeric = ["shares", "value_usd", "call_value", "put_value"]
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
        current = totals.xs(period).copy() if period in totals.index.get_level_values("period") else totals.droplevel("period").iloc[:0]
        prior = totals.xs(prior_period).copy() if prior_period in totals.index.get_level_values("period") else current.iloc[:0]
        observed = current[["shares", "call_value", "put_value"]].gt(0).any(axis=1)
        seen.update(current.loc[observed].index.get_level_values("ticker"))
        names = sorted(seen)
        if not names:
            continue
        out = pd.DataFrame(index=pd.Index(names, name="ticker"))
        level = current.groupby("ticker")
        out["inst_shares"] = level.shares.agg(lambda amounts: amounts.sum(skipna=False))
        out["inst_value"] = level.value_usd.agg(lambda amounts: amounts.sum(skipna=False))
        holders = current.shares.gt(0).groupby("ticker").sum()
        out["ic_inst_holders"] = holders / len(current_states)
        unknown_quantity = current.shares.isna().groupby("ticker").any()
        out.loc[unknown_quantity.reindex(out.index).eq(True), "ic_inst_holders"] = np.nan
        if current_states.zero_eligible.all():
            empty_names = ~out.index.isin(current.index.get_level_values("ticker"))
            out.loc[empty_names, ["inst_shares", "inst_value", "ic_inst_holders"]] = 0.0
        value_total = current.value_usd.groupby("ticker").transform(lambda amounts: amounts.sum(skipna=False)).reindex(current.index)
        hhi = (current.value_usd / value_total.where(value_total.gt(0))).pow(2).groupby("ticker").sum(min_count=1)
        out["ic_inst_concentration"] = hhi.where(out.inst_value.notna())
        calls = level.call_value.agg(lambda amounts: amounts.sum(skipna=False))
        puts = level.put_value.agg(lambda amounts: amounts.sum(skipna=False))
        investment = out.inst_value + calls + puts
        out["ic_inst_net_options_ratio"] = (calls - puts) / investment.where(investment.gt(0))
        paired = current_states.index[current_states.comparison_eligible & prior_states.comparison_eligible.reindex(current_states.index).eq(True)]
        joined = current.add_suffix("_current").join(prior.add_suffix("_prior"), how="outer")
        joined = joined.loc[joined.index.get_level_values("cik").isin(paired)]
        for suffix, values, state in (("current", current, current_states), ("prior", prior, prior_states)):
            absent = ~joined.index.isin(values.index)
            zeros = state.zero_eligible.reindex(joined.index.get_level_values("cik")).eq(True).to_numpy()
            joined.loc[absent & zeros, [col + "_" + suffix for col in numeric]] = 0.0
        factors = split_cum_product(
            splits, pd.Series(joined.index.get_level_values("ticker")), pd.Series(period, index=range(len(joined)))
        ) / split_cum_product(splits, pd.Series(joined.index.get_level_values("ticker")), pd.Series(prior_period, index=range(len(joined))))
        joined["shares_prior"] *= factors
        if period in securities.index.get_level_values("period") and prior_period in securities.index.get_level_values("period"):
            now_ids = securities.xs(period).reindex(joined.index)
            old_ids = securities.xs(prior_period).reindex(joined.index)
            changed = now_ids.notna() & old_ids.notna() & now_ids.ne(old_ids)
            joined.loc[changed, ["shares_current", "shares_prior"]] = np.nan
        supported = joined.shares_current.notna() & joined.shares_prior.notna()
        matched = joined.loc[supported]
        groups = matched.groupby("ticker")
        current_shares = groups.shares_current.sum(min_count=1)
        previous_shares = groups.shares_prior.sum(min_count=1)
        now_held, was_held = matched.shares_current.gt(0), matched.shares_prior.gt(0)
        holder_count = now_held.groupby("ticker").sum()
        prior_count = was_held.groupby("ticker").sum()
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
        out["ic_inst_new_buyer_ratio"] = (now_held & matched.shares_prior.eq(0)).groupby("ticker").sum() / holder_count.where(holder_count.gt(0))
        out["ic_inst_exit_ratio"] = (matched.shares_current.eq(0) & was_held).groupby("ticker").sum() / prior_count.where(prior_count.gt(0))
        increases = (now_held & was_held & matched.shares_current.gt(matched.shares_prior)).groupby("ticker").sum()
        decreases = (now_held & was_held & matched.shares_current.lt(matched.shares_prior)).groupby("ticker").sum()
        out["ic_inst_cluster_buying"] = (increases - decreases) / holder_count.where(holder_count.gt(0))
        out.loc[prior_count.reindex(out.index).fillna(0).lt(min_prior_holders), list(DELTA_FEATURES)] = np.nan
        out["period"], out["as_of"] = period, day
        rows.append(out.reset_index())
    result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    logger.info("13F immutable publication: %s ticker states; unfiled/restricted books never imply exits.", len(result))
    return result
