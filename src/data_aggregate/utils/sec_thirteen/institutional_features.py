"""Broad point-in-time 13F levels, matched-quarter changes, and publication previews."""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from typing import Any, cast

import numpy as np
import pandas as pd

from src.constants.constants import F13_SETTLE_TRADING_DAYS, SEC_13F_FILING_LAG_DAYS
from src.data_aggregate.utils.common.availability import availability_date, filing_public_dates, public_books
from src.data_aggregate.utils.common.panel import build_peer_relative_panel
from src.data_aggregate.utils.common.pit import daily_market_cap, fundamentals_to_daily
from src.data_aggregate.utils.common.price_frames import PriceFrames
from src.data_aggregate.utils.common.split_basis import future_split_factor, split_cum_product
from src.data_aggregate.utils.common.typing import frame_column
from src.data_aggregate.utils.sec_thirteen.holdings_clean import clean_holdings as clean_holdings
from src.data_aggregate.utils.sec_thirteen.holdings_clean import clean_legacy_institutional_holdings, clean_public_holdings
from src.data_aggregate.utils.sec_thirteen.holdings_clean import stamp_legacy_publication as _stamp_availability
from src.data_aggregate.utils.sec_thirteen.institutional_books import compact_institutional_books
from src.data_aggregate.utils.sec_thirteen.institutional_vintages import iter_public_books
from src.data_aggregate.utils.sec_thirteen.value_basis import nullable_value_to_daily
from src.data_store.schema import Resume, Tables
from src.utils.price_adjustments import ShareBasis

logger = logging.getLogger(__name__)
F13_REVISION_MIN_MOVE = 0.001

#: The first period whose all-filer coverage describes the market rather than the fetch
#: (measured above: 63 -> 2,938 in-universe filers, post-band).
INST_LEVEL_FLOOR_PERIOD = pd.Timestamp(cast(str, cast(Resume, Tables.sec13f_hr.resume).coverage_start))

#: The first period whose PREDECESSOR is also post-break, so a QoQ delta is meaningful.
INST_DELTA_FLOOR_PERIOD = INST_LEVEL_FLOOR_PERIOD + pd.offsets.QuarterEnd(1)

#: D17. A quarter-on-quarter move in the universe-wide filer count larger than this is a
#: coverage discontinuity, not a market event. A research parameter, so the step passes the
#: configured value in; this is the declared default.
COVERAGE_BREAK_DEFAULT = 0.50

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


def _coverage_states(h: pd.DataFrame, break_pct: float) -> pd.DataFrame:
    """Public filer counts and causal coverage flags at each publication stamp."""
    first_public = cast(pd.DataFrame, h.groupby(["period", "cik"], as_index=False)["as_of"].min())
    new_filers = first_public.groupby(["period", "as_of"]).size().rename("new_filers")
    stamps = h[["period", "as_of"]].drop_duplicates().set_index(["period", "as_of"]).sort_index()
    table = stamps.join(new_filers, how="left")
    table["filers"] = table["new_filers"].fillna(0).groupby(level="period").cumsum()
    final_filers = table.groupby(level="period")["filers"].last()
    previous = final_filers.shift(1)
    table["prev"] = table.index.get_level_values("period").map(previous)
    ratio = (table["filers"] - table["prev"]).abs() / table["prev"]
    table["break"] = ratio > break_pct
    table["hole"] = table["filers"] < (1.0 - break_pct) * table["prev"]
    return table


def _split_factors(h: pd.DataFrame, splits: pd.DataFrame | None) -> dict:
    """Restate prior ticker-quarter shares by splits effective between its observed reports."""
    pairs = cast(pd.DataFrame, h[["ticker", "period"]]).drop_duplicates().sort_values(["ticker", "period"])
    if pairs.empty:
        return {}

    pairs["prev_period"] = pairs.groupby("ticker")["period"].shift(1)
    pairs = pairs.dropna(subset=["prev_period"])
    if pairs.empty:
        return {}
    f_now = future_split_factor(splits, frame_column(pairs, "ticker"), frame_column(pairs, "period"))
    f_prev = future_split_factor(splits, frame_column(pairs, "ticker"), frame_column(pairs, "prev_period"))
    factor = np.divide(f_prev, f_now, out=np.ones_like(f_prev), where=f_now > 0)
    n = int((np.abs(factor - 1.0) > 1e-9).sum())
    if n:
        logger.info(
            "13F split restatement: %s of %s (ticker, quarter) pairs had the prior quarter's share count rebased (V5 trigger count)", n, len(pairs)
        )
    return {(t, p): f for t, p, f in zip(pairs["ticker"], pairs["period"], factor, strict=False)}


def _capped_ownership(ratio: pd.DataFrame) -> pd.DataFrame:
    """Null and report ownership ratios above the existing denominator-defect ceiling."""
    over = ratio.gt(OWNERSHIP_CEILING)
    n = int(over.to_numpy().sum())
    if n:
        logger.info(
            "ownership guard: %s of %s non-null `ic_inst_ownership_pct` cells (%.3f%%) "
            "above %.1fx shares outstanding -> nulled (the `sharesOutstandingPit` split "
            "defect, not a holder); max seen %.2f",
            f"{n:,}",
            f"{int(ratio.notna().to_numpy().sum()):,}",
            100.0 * n / max(1, int(ratio.notna().to_numpy().sum())),
            OWNERSHIP_CEILING,
            float(np.nanmax(ratio.to_numpy(dtype="float64"))),
        )
    return ratio.mask(over)


def _assert_emission_windows_ordered(qf: pd.DataFrame) -> None:
    """Reject overlapping quarter publication windows and duplicate ticker/date emissions."""
    span = cast(pd.DataFrame, qf.groupby("period")["as_of"].agg(["min", "max"]).sort_index())
    # `bad` marks the LATER period of an offending pair -- the one whose first publication is
    # at or before its predecessor's last revision -- so the pair is `(index[i - 1], index[i])`.
    bad = np.flatnonzero((frame_column(span, "max").shift(1) >= frame_column(span, "min")).to_numpy())
    if len(bad):
        period_index: pd.DatetimeIndex = pd.DatetimeIndex(span.index)
        overlaps = [(str(period_index[int(i) - 1].date()), str(period_index[int(i)].date())) for i in bad]
        raise ValueError(
            "13F emission windows OVERLAP across consecutive periods -- a later quarter's "
            f"first publication is at or before an earlier one's last revision: {overlaps[:6]}. "
            "`fundamentals_to_daily` would resolve the collision with `aggfunc='last'` and the "
            "panel would step backwards. Narrow the late filing tolerance or widen the settle."
        )
    duplicated = int(qf.duplicated(["ticker", "as_of"]).sum())
    if duplicated:
        raise ValueError(
            f"13F emission: {duplicated:,} (ticker, as_of) pair(s) are emitted twice, so "
            "`fundamentals_to_daily` would silently keep one of them. Two periods share an "
            "availability date, which the window ordering above should have made impossible."
        )


def _report_first_publication_shortfall(h: pd.DataFrame) -> None:
    """Log share-weighted filings arriving after their quarter's first public date."""
    if not len(h) or not {"as_of", "first_pub"}.issubset(h.columns):
        logger.info("13F availability: no `as_of`/`first_pub` -> the first-publication shortfall cannot be measured on this read")
        return

    late = (frame_column(h, "as_of") > frame_column(h, "first_pub")).fillna(False)
    shares = cast(pd.Series, pd.to_numeric(frame_column(h, "shares"), errors="coerce"))
    total_sh = float(shares.sum())
    logger.info(
        "13F availability (settle %s trading day(s)): %s of %s filer-row(s) (%.2f%%) and "
        "%.2f%% of SHARES arrive after their period's first publication -> emitted as "
        "revisions on their own filing date, never back-dated",
        F13_SETTLE_TRADING_DAYS,
        f"{int(late.sum()):,}",
        f"{len(h):,}",
        100.0 * float(late.mean()),
        100.0 * float(cast(pd.Series, shares.loc[late]).sum()) / total_sh if total_sh > 0 else 0.0,
    )

    per_q = cast(
        pd.DataFrame,
        pd.DataFrame({"period": frame_column(h, "period"), "shares": shares, "behind": shares.where(late, 0.0)}).groupby("period").sum(),
    )
    shortfall = (frame_column(per_q, "behind") / frame_column(per_q, "shares").replace(0.0, np.nan)).dropna().sort_values(ascending=False)
    worst = shortfall.head(6)
    logger.info(
        "13F availability: worst 6 quarters by share of SHARES behind the first publication: %s (universe-wide median %.2f%%)",
        {str(pd.Timestamp(cast(Any, p)).date()): f"{100.0 * v:.2f}%" for p, v in worst.items()},
        100.0 * float(shortfall.median()),
    )


def _availability_coverage(h: pd.DataFrame) -> pd.Series:
    """Measure prior-quarter shares represented by filers public at the current deadline."""

    if not len(h) or "first_pub" not in h.columns:
        return pd.Series(dtype="float64")
    # ⚠ AN UNPROJECTED `shares` IS THE FAILURE MODE THIS GUARD EXISTS FOR, and it has already
    # happened once. `clean_holdings` CREATES a missing numeric leg as 0.0 (see `_NUMERIC_13F`),
    # so a read that projected `[ticker, period, cik, filing_date]` arrives here with an
    # all-zero share column, every weight sums to 0, and the coverage comes back EMPTY -- which
    # a caller reads as "no quarter has a prior quarter", a true-sounding sentence about the
    # wrong thing. Raising the distinction to the caller is the point: no data and no basis for
    # the measurement are different answers.
    shares = cast(pd.Series, pd.to_numeric(frame_column(h, "shares"), errors="coerce"))
    if float(shares.abs().sum()) == 0.0:
        logger.info(
            "13F availability coverage: the `shares` column is all zero -- it was almost "
            "certainly projected away and zero-filled by `clean_holdings`, so there is no "
            "basis to weight the diagnostic on. Not measured."
        )
        return pd.Series(dtype="float64")

    # One row per (cik, period): the filer's universe-wide share count, and the first date any
    # of its filings for that period was public.
    per_filer = h.groupby(["cik", "period"], sort=False).agg(shares=("shares", "sum"), pub=("as_of", "min")).reset_index()
    first_pub = cast(pd.Series, h.groupby("period")["first_pub"].first())
    periods = pd.Index(sorted(first_pub.dropna().index))
    pos = {p: i for i, p in enumerate(periods)}
    per_filer["pi"] = frame_column(per_filer, "period").map(pos)
    per_filer = per_filer.dropna(subset=["pi"])

    # `q-1`'s filers, aligned onto `q`, then matched against the filers public at `q`'s own
    # first publication. `pub <= first_pub(q)` IS "reported by the availability date".
    prior = cast(pd.DataFrame, per_filer[["cik", "pi", "shares"]].copy())
    prior["pi"] = frame_column(prior, "pi") + 1
    public_by_deadline = frame_column(per_filer, "pub") <= frame_column(per_filer, "period").map(first_pub)
    reported = cast(pd.DataFrame, per_filer.loc[public_by_deadline, ["cik", "pi"]]).assign(_seen=True)
    joined = prior.merge(reported, on=["cik", "pi"], how="left")
    # ⚠ `pi + 1` PUTS THE LAST QUARTER'S FILERS ONE PAST THE END, and that row has no quarter
    # to describe. Dropping it BEFORE the groupby rather than trimming the index afterwards is
    # what keeps the index assignment length-safe -- the earlier form filtered the list and not
    # the Series, which would raise on exactly the frame that triggered it.
    joined = cast(pd.DataFrame, joined.loc[frame_column(joined, "pi") < len(periods)])
    if joined.empty:
        return pd.Series(dtype="float64")
    seen = frame_column(joined, "_seen").fillna(False).to_numpy(dtype=bool)
    weight = cast(pd.Series, joined.groupby("pi")["shares"].sum()).replace(0.0, np.nan)
    held = cast(pd.Series, joined.loc[seen].groupby("pi")["shares"].sum())
    coverage = (held.reindex(weight.index).fillna(0.0) / weight).dropna()
    coverage.index = pd.DatetimeIndex([periods[int(i)] for i in coverage.index])
    return coverage.sort_index()


def _report_availability_coverage(h: pd.DataFrame) -> None:
    """Log `_availability_coverage` beside the filer-count coverage table."""
    coverage = _availability_coverage(h)
    if coverage.empty:
        return
    thin = cast(pd.Series, coverage.loc[coverage < 0.90])
    logger.info(
        "13F availability coverage (share of q-1's SHARES held by filers public by as_of(q)): "
        "median %.2f%%, p05 %.2f%%, min %.2f%% on %s; %s of %s quarter(s) below 90%%%s",
        100.0 * float(coverage.median()),
        100.0 * float(coverage.quantile(0.05)),
        100.0 * float(coverage.min()),
        str(pd.Timestamp(cast(Any, coverage.idxmin())).date()),
        len(thin),
        len(coverage),
        (" -> " + ", ".join(f"{pd.Timestamp(cast(Any, p)).date()}={100.0 * v:.1f}%" for p, v in thin.sort_values().head(8).items()))
        if len(thin)
        else "",
    )


def _legacy_publication_walk(
    df_holdings: pd.DataFrame, splits: pd.DataFrame | None, break_pct: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict]:
    """Prepare ordered publication rows, causal coverage, and prior-quarter split factors."""
    if "as_of" in df_holdings.columns:
        df_walk = df_holdings
    else:
        logger.info(
            "13F emission: no `as_of` column -> one emission per period on the bare "
            "period+%sd deadline, with no availability cutoff (every filing for the "
            "period is inside it, including the late ones)",
            SEC_13F_FILING_LAG_DAYS,
        )
        df_walk = df_holdings.assign(as_of=df_holdings["period"] + pd.Timedelta(days=SEC_13F_FILING_LAG_DAYS))
    df_coverage = _coverage_states(df_walk, break_pct)
    df_holes = df_coverage[df_coverage["hole"]]
    df_breaks = df_coverage[df_coverage["break"]]
    if not df_holes.empty or not df_breaks.empty:
        logger.info(
            "13F coverage guard: %s publication state(s) are holes (every feature nulled), %s are breaks (deltas nulled)",
            len(df_holes),
            len(df_breaks),
        )
    final_filers = df_coverage.groupby(level="period")["filers"].last()
    # Coverage retains pre-floor history; arithmetic starts at the declared floor.
    df_walk = cast(pd.DataFrame, df_walk.loc[frame_column(df_walk, "period") >= INST_LEVEL_FLOOR_PERIOD])
    factors = _split_factors(df_walk, splits)
    df_walk = df_walk.sort_values(["ticker", "period", "as_of"], kind="stable")
    return df_walk, df_coverage, final_filers, factors


def _advance_manager_counts(
    previous: dict,
    ciks: list,
    shares: list,
    factor: float,
    n_both: int,
    increases: int,
    decreases: int,
) -> tuple[int, int, int]:
    """Advance matched-manager counts in source order for newly public holdings."""
    for cik, held in zip(ciks, shares, strict=False):
        was = previous.get(cik)
        if was is None:
            continue
        n_both += 1
        restated = was * factor
        if held > restated:
            increases += 1
            continue
        if held < restated:
            decreases += 1
    return n_both, increases, decreases


def _legacy_period_features(
    ticker: object,
    period: object,
    df_period: pd.DataFrame,
    previous: dict,
    previous_total: float,
    previous_period: pd.Timestamp | None,
    df_coverage: pd.DataFrame,
    final_filers: pd.Series,
    factors: dict,
    revision_min_move: float,
) -> tuple[list[dict], int, int, dict]:
    """Emit cumulative public states for one ticker-quarter against its revised predecessor."""
    ciks = df_period["cik"].to_numpy()
    shares = df_period["shares"].to_numpy(dtype="float64")
    values = df_period["value_usd"].to_numpy(dtype="float64")
    calls = df_period["call_value"].to_numpy(dtype="float64")
    puts = df_period["put_value"].to_numpy(dtype="float64")
    stamps = df_period["as_of"].to_numpy()
    period_stamps = pd.DatetimeIndex(df_coverage.loc[cast(Any, period)].index)
    emission_stamps = period_stamps[period_stamps >= pd.Timestamp(stamps[0])]
    ends = np.searchsorted(stamps, emission_stamps.to_numpy(), side="right")
    cik_order = np.argsort(ciks, kind="stable")
    cik_list, share_list = ciks.tolist(), shares.tolist()
    n_prev = len(previous)
    has_prev = n_prev > 0
    factor = factors.get((ticker, period), 1.0) if has_prev else 1.0
    prev_shares = previous_total * factor if has_prev else np.nan
    prev_pool = float(cast(Any, final_filers.get(previous_period, np.nan))) if previous_period is not None else np.nan
    prev_share = n_prev / prev_pool if (has_prev and prev_pool > 0) else np.nan
    rows: list[dict] = []
    n_revisions = n_skipped = 0
    n_both = inc = dec = at = 0
    last_emitted = last_share = np.nan
    last_hole = last_break = None
    for stamp, k in zip(emission_stamps, ends, strict=True):
        n_both, inc, dec = _advance_manager_counts(previous, cik_list[at:k], share_list[at:k], factor, n_both, inc, dec)
        at = int(k)
        # Re-reduce the exact prefix: incremental floating sums would depend on emission count.
        inst_shares = float(shares[:k].sum())
        holders = int(k)
        coverage_state = df_coverage.loc[(pd.Timestamp(cast(Any, period)), stamp)]
        pool = float(cast(Any, coverage_state["filers"]))
        share = holders / pool if pool > 0 else np.nan
        is_hole, is_break = bool(coverage_state["hole"]), bool(coverage_state["break"])
        material = True
        is_revision = np.isfinite(last_emitted)
        if is_revision:
            move = abs(inst_shares - last_emitted) / last_emitted if last_emitted > 0 else np.inf
            state_changed = share != last_share or is_hole != last_hole or is_break != last_break
            material = not (move < revision_min_move and not state_changed)
        if not material:
            n_skipped += 1
            continue
        if is_revision:
            n_revisions += 1
        inst_value = float(values[:k].sum())
        call_v = float(calls[:k].sum())
        put_v = float(puts[:k].sum())
        total_invested = inst_value + call_v + put_v
        opt_ratio = ((call_v - put_v) / total_invested) if total_invested > 0 else np.nan
        mv = values[cik_order[cik_order < k]]
        tot_mv = float(mv.sum())
        hhi = float(((mv / tot_mv) ** 2).sum()) if tot_mv > 0 else np.nan
        rows.append(
            {
                "ticker": ticker,
                "period": pd.Timestamp(cast(Any, period)),
                "as_of": stamp,
                "_prev_holders": float(n_prev) if has_prev else np.nan,
                "_coverage_hole": is_hole,
                "_coverage_break": is_break,
                "ic_inst_holders": share,
                "inst_shares": inst_shares,
                "inst_value": inst_value,
                "ic_inst_breadth_chg": (share - prev_share) if np.isfinite(prev_share) else np.nan,
                "ic_inst_shares_chg": (inst_shares / prev_shares - 1.0) if (has_prev and prev_shares > 0) else np.nan,
                "ic_inst_new_buyer_ratio": ((holders - n_both) / holders) if (has_prev and holders > 0) else np.nan,
                "ic_inst_exit_ratio": ((n_prev - n_both) / n_prev) if has_prev else np.nan,
                "ic_inst_cluster_buying": ((inc - dec) / holders) if (has_prev and holders > 0) else np.nan,
                "ic_inst_net_options_ratio": opt_ratio,
                "ic_inst_concentration": hhi,
            }
        )
        last_emitted, last_share = inst_shares, share
        last_hole, last_break = is_hole, is_break
    return rows, n_revisions, n_skipped, dict(zip(cik_list, share_list, strict=False))


def _legacy_ticker_features(
    ticker: object,
    df_ticker: pd.DataFrame,
    df_coverage: pd.DataFrame,
    final_filers: pd.Series,
    factors: dict,
    revision_min_move: float,
) -> tuple[list[dict], int, int]:
    """Walk one ticker's report quarters while retaining each fully revised predecessor."""
    rows: list[dict] = []
    n_revisions = n_skipped = 0
    previous: dict = {}
    previous_total = np.nan
    previous_period = None
    for period, df_period in df_ticker.groupby("period", sort=True):
        emitted, revisions, skipped, previous = _legacy_period_features(
            ticker, period, df_period, previous, previous_total, previous_period, df_coverage, final_filers, factors, revision_min_move
        )
        rows.extend(emitted)
        n_revisions += revisions
        n_skipped += skipped
        previous_total = float(sum(previous.values()))
        previous_period = pd.Timestamp(cast(Any, period))
    return rows, n_revisions, n_skipped


def _report_legacy_revisions(n_rows: int, n_revisions: int, n_skipped: int, revision_min_move: float) -> None:
    """Report emitted publications and immaterial candidates deferred to the next state."""
    if n_revisions or n_skipped:
        logger.info(
            "13F emission: %s first publication(s) + %s revision(s) = %s row(s); %s "
            "candidate revision date(s) skipped as immaterial (< %.3f%% of the last "
            "emitted share count -- DEFERRED to the next emission, never dropped)",
            f"{n_rows - n_revisions:,}",
            f"{n_revisions:,}",
            f"{n_rows:,}",
            f"{n_skipped:,}",
            100.0 * revision_min_move,
        )


def _mask_legacy_quarter_features(df_features: pd.DataFrame, min_prior_holders: int) -> pd.DataFrame:
    """Apply publication coverage, period floors, and per-ticker predecessor support."""
    if df_features.empty:
        return df_features
    is_hole = df_features.pop("_coverage_hole").astype(bool)
    is_break = df_features.pop("_coverage_break").astype(bool) | is_hole
    value_cols = ["inst_shares", "inst_value"]
    for column in DELTA_FEATURES:
        if column in df_features.columns:
            df_features.loc[is_break, column] = np.nan
    for column in LEVEL_FEATURES + tuple(value_cols):
        if column in df_features.columns:
            df_features.loc[is_hole, column] = np.nan
    df_features.loc[
        df_features["period"] < INST_LEVEL_FLOOR_PERIOD, [column for column in LEVEL_FEATURES if column in df_features.columns] + value_cols
    ] = np.nan
    df_features.loc[df_features["period"] < INST_DELTA_FLOOR_PERIOD, [column for column in DELTA_FEATURES if column in df_features.columns]] = np.nan
    delta_cols = [column for column in DELTA_FEATURES if column in df_features.columns]
    thin = df_features["_prev_holders"] < min_prior_holders
    removed = thin & df_features[delta_cols].notna().any(axis=1)
    if removed.any():
        logger.info(
            "13F per-ticker onset guard: %s of %s (ticker, quarter) deltas nulled "
            "-- the prior quarter carried fewer than %s filers for that name "
            "(%s ticker(s))",
            f"{int(removed.sum()):,}",
            f"{len(df_features):,}",
            min_prior_holders,
            df_features.loc[removed, "ticker"].nunique(),
        )
    df_features.loc[thin, delta_cols] = np.nan
    return df_features.drop(columns=["_prev_holders"])


def _quarter_features(
    h: pd.DataFrame,
    splits: pd.DataFrame | None = None,
    break_pct: float = COVERAGE_BREAK_DEFAULT,
    min_prior_holders: int = MIN_PRIOR_HOLDERS,
    revision_min_move: float = F13_REVISION_MIN_MOVE,
) -> pd.DataFrame:
    """Reduce legacy manager holdings to cumulative public ticker-quarter features."""
    df_walk, df_coverage, final_filers, factors = _legacy_publication_walk(h, splits, break_pct)
    rows: list[dict] = []
    n_revisions = n_skipped = 0
    for ticker, df_ticker in df_walk.groupby("ticker", sort=False):
        emitted, revisions, skipped = _legacy_ticker_features(ticker, df_ticker, df_coverage, final_filers, factors, revision_min_move)
        rows.extend(emitted)
        n_revisions += revisions
        n_skipped += skipped
    _report_legacy_revisions(len(rows), n_revisions, n_skipped, revision_min_move)
    return _mask_legacy_quarter_features(pd.DataFrame(rows), min_prior_holders)


def _daily_owned_shares(qf: pd.DataFrame, trading_index: pd.DatetimeIndex, share_basis: ShareBasis | None) -> pd.DataFrame:
    """Carry reported quantities in actual units without changing their publication clock."""
    if share_basis is None or not share_basis.policies:
        return nullable_value_to_daily(qf, "inst_shares", trading_index)
    df_owned = qf.copy()
    configured = df_owned["ticker"].isin(share_basis.policies)
    periods = pd.to_datetime(df_owned["period"])
    factor = share_basis.factor(df_owned["ticker"], periods, "actual")
    df_owned.loc[configured, "inst_shares"] *= factor.loc[configured]
    # Aggregated books without an explicit security title cannot establish ADS units.
    units = share_basis.factor(df_owned["ticker"], periods, "units")
    df_owned.loc[configured & units.ne(1.0), "inst_shares"] = np.nan
    daily = nullable_value_to_daily(df_owned, "inst_shares", trading_index)
    for ticker in share_basis.policies.keys() & set(daily.columns):
        names = pd.Series(ticker, index=trading_index)
        dates = pd.Series(trading_index, index=trading_index)
        daily[ticker] /= share_basis.factor(names, dates, "actual")
        daily[ticker] = daily[ticker].where(share_basis.factor(names, dates, "units").eq(1.0))
    return daily


def build_institutional_feature_panel(
    frames: PriceFrames,
    holdings: pd.DataFrame | None,
    *,
    shares_out_history: pd.DataFrame | None = None,
    splits: pd.DataFrame | None = None,
    break_pct: float = COVERAGE_BREAK_DEFAULT,
    min_prior_holders: int = MIN_PRIOR_HOLDERS,
    settle_trading_days: int = F13_SETTLE_TRADING_DAYS,
    revision_min_move: float = F13_REVISION_MIN_MOVE,
    valuation_close: pd.DataFrame | None = None,
    value_basis: Mapping[str, float] | None = None,
    filings: pd.DataFrame | None = None,
    share_basis: ShareBasis | None = None,
) -> pd.DataFrame:
    """Prepare raw holdings and build the broad institutional daily feature panel."""
    need = {"cik", "period", "ticker", "shares"}
    if holdings is None or holdings.empty or not need.issubset(holdings.columns):
        return pd.DataFrame(columns=["date", "ticker"])
    df_holdings = _prepare_institutional_holdings(holdings, filings, frames.trading_index, valuation_close, value_basis, settle_trading_days)
    if df_holdings.empty:
        return pd.DataFrame(columns=["date", "ticker"])
    return build_prepared_institutional_feature_panel(
        frames,
        df_holdings,
        shares_out_history=shares_out_history,
        splits=splits,
        break_pct=break_pct,
        min_prior_holders=min_prior_holders,
        settle_trading_days=settle_trading_days,
        revision_min_move=revision_min_move,
        filings=filings,
        share_basis=share_basis,
    )


def _prepare_institutional_holdings(
    df_holdings: pd.DataFrame,
    filings: pd.DataFrame | None,
    trading_index: pd.DatetimeIndex,
    valuation_close: pd.DataFrame | None,
    value_basis: Mapping[str, float] | None,
    settle_trading_days: int,
) -> pd.DataFrame:
    """Preserve distinct immutable-vintage and legacy holdings preparation policies."""
    if filings is not None:
        return clean_public_holdings(df_holdings, valuation_close, value_basis)
    df_holdings = clean_legacy_institutional_holdings(df_holdings, valuation_close, value_basis)
    if df_holdings.empty:
        return df_holdings
    df_holdings = _stamp_availability(df_holdings, trading_index, settle_trading_days=settle_trading_days)
    _report_first_publication_shortfall(df_holdings)
    _report_availability_coverage(df_holdings)
    return df_holdings


def _daily_institutional_value_weight(df_features: pd.DataFrame, frames: PriceFrames, shares_out_history: pd.DataFrame | None) -> pd.DataFrame | None:
    """Scale known institutional dollars by the existing daily market-cap basis."""
    if shares_out_history is None or frames.close_split is None or frames.close_split.empty:
        return None
    df_market_cap = daily_market_cap(shares_out_history, frames.close_split, level_factor=frames.level_factor)
    if df_market_cap.empty:
        logger.warning(
            "daily_market_cap returned no columns (shares_out_history has %s; "
            "it needs `sharesOutstanding`, the VENDOR basis) -> "
            "ic_inst_value_to_mcap is skipped.",
            sorted(shares_out_history.columns),
        )
        return None
    df_positive_cap = df_market_cap.where(df_market_cap > 0)
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
    shares_out_history: pd.DataFrame | None,
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
    if shares_out_history is not None and not shares_out_history.empty:
        df_owned = _daily_owned_shares(df_features, frames.trading_index, share_basis)
        df_shares = fundamentals_to_daily(shares_out_history, "sharesOutstandingPit", frames.trading_index, share_basis=share_basis)
        if not df_shares.empty and df_shares.notna().any().any():
            fields["ic_inst_ownership_pct"] = _capped_ownership((df_owned / df_shares.where(df_shares > 0)).replace([np.inf, -np.inf], np.nan))
    df_weight = _daily_institutional_value_weight(df_features, frames, shares_out_history)
    if df_weight is not None:
        fields["ic_inst_value_to_mcap"] = df_weight
    return _mask_daily_institutional_fields(fields, trading_calendar, settle_trading_days)


def build_prepared_institutional_feature_panel(
    frames: PriceFrames,
    holdings: pd.DataFrame,
    *,
    shares_out_history: pd.DataFrame | None = None,
    splits: pd.DataFrame | None = None,
    break_pct: float = COVERAGE_BREAK_DEFAULT,
    min_prior_holders: int = MIN_PRIOR_HOLDERS,
    settle_trading_days: int = F13_SETTLE_TRADING_DAYS,
    revision_min_move: float = F13_REVISION_MIN_MOVE,
    filings: pd.DataFrame | None = None,
    share_basis: ShareBasis | None = None,
) -> pd.DataFrame:
    """Compute quarter features from prepared holdings, then project the supported daily panel."""
    if splits is None or splits.empty:
        logger.warning("No `prices_splits` -> 13F share changes are NOT split-restated; a 20-for-1 split will read as +1,900%% accumulation.")
    if filings is not None:
        df_features = _versioned_quarter_features(
            filings, holdings, frames.trading_index, splits=splits, settle_days=settle_trading_days, min_prior_holders=min_prior_holders
        )
    else:
        df_features = _quarter_features(
            holdings, splits=splits, break_pct=break_pct, min_prior_holders=min_prior_holders, revision_min_move=revision_min_move
        )
    if df_features.empty:
        return pd.DataFrame(columns=["date", "ticker"])
    if filings is None:
        _assert_emission_windows_ordered(df_features)
    fields = project_institutional_features(
        df_features,
        frames,
        shares_out_history,
        share_basis,
        settle_trading_days,
        versioned=filings is not None,
        trading_calendar=frames.trading_index,
    )
    if not fields:
        return pd.DataFrame(columns=["date", "ticker"])
    emission = {name: EMISSION[name] for name in fields}
    return build_peer_relative_panel(fields, frames.peers, emission=emission, availability=frames.availability)


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


def _versioned_quarter_features(
    manifests: pd.DataFrame,
    holdings: pd.DataFrame,
    trading_index: pd.DatetimeIndex,
    *,
    splits: pd.DataFrame | None,
    settle_days: int,
    min_prior_holders: int,
) -> pd.DataFrame:
    """Adapt bounded raw callers to the same compact public-snapshot arithmetic."""
    snapshots = (
        (day, compact_institutional_books(books), states)
        for day, books, states in iter_public_books(manifests, holdings, trading_index, settle_days=settle_days)
    )
    return compute_institutional_feature_history(snapshots, splits=splits, min_prior_holders=min_prior_holders)


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
