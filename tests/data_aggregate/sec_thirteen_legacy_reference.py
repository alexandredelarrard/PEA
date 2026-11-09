"""Historical broad 13F arithmetic for B0 regression; current steps use immutable books."""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.constants.constants import F13_SETTLE_TRADING_DAYS, SEC_13F_FILING_LAG_DAYS
from src.data_aggregate.utils.common.availability import availability_date
from src.data_aggregate.utils.common.data_utils import to_day
from src.data_aggregate.utils.common.panel import build_peer_relative_panel
from src.data_aggregate.utils.common.quotes import lookup_asof
from src.data_aggregate.utils.common.typing import frame_column
from src.data_aggregate.utils.price.features import security_basis_frame
from src.data_aggregate.utils.price.frames import PriceFrames
from src.data_aggregate.utils.sec_thirteen.holdings_clean import _DATES
from src.data_aggregate.utils.sec_thirteen.institutional_books import compact_institutional_books
from src.data_aggregate.utils.sec_thirteen.institutional_features import (
    DELTA_FEATURES,
    EMISSION,
    INST_DELTA_FLOOR_PERIOD,
    INST_LEVEL_FLOOR_PERIOD,
    LEVEL_FEATURES,
    MIN_PRIOR_HOLDERS,
    project_institutional_features,
)
from src.data_aggregate.utils.sec_thirteen.value_basis import log_register, repair_value_basis
from src.utils.institutional_vintages import prepare_book_vintages, resolve_prepared_books
from src.utils.price.adjustments import ShareBasis
from src.utils.price.levels import convert_quantities, split_cum_product

logger = logging.getLogger(__name__)


# Retired per-day public-book cache (no production caller): legacy broad snapshots replay through it.
@dataclass(frozen=True, slots=True)
class _PublicationSources:
    """Prepared source frames and their fixed filing, position and publication indices."""

    df_manifest: pd.DataFrame
    df_holdings: pd.DataFrame
    filing_groups: Mapping[tuple[str, pd.Timestamp], NDArray[np.intp]]
    position_indices: Mapping[str, NDArray[np.intp]]
    publications: Mapping[pd.Timestamp, NDArray[np.intp]]


def _publication_schedule(df_manifest: pd.DataFrame) -> tuple[dict, pd.DatetimeIndex]:
    """Schedule public filings in chronological order."""
    publications = cast(dict[pd.Timestamp, Any], df_manifest.groupby("_day", sort=True).indices)
    days = pd.DatetimeIndex(sorted(publications))
    return publications, days


def _history_floor(active: Any) -> pd.Timestamp:
    """The report quarter before the latest known one is the oldest dependency kept."""
    return (active.to_period("Q") - 1).end_time.normalize() if pd.notna(active) else pd.Timestamp.min


def _reconcile_publication(
    sources: _PublicationSources,
    cache: dict[tuple[str, pd.Timestamp], tuple[pd.DataFrame, pd.DataFrame]],
    active: Any,
    day: pd.Timestamp,
) -> tuple[dict[tuple[str, pd.Timestamp], tuple[pd.DataFrame, pd.DataFrame]], Any, list[tuple[str, pd.Timestamp]]]:
    """Resolve changed books and retain the dependency quarters of the latest known report."""
    df_events = sources.df_manifest.iloc[sources.publications.get(day, np.array([], dtype=int))]
    floor = _history_floor(active)
    changed = []
    for key in df_events[["cik", "period"]].drop_duplicates().itertuples(index=False, name=None):
        if key[1] < floor:
            continue
        df_versions = sources.df_manifest.iloc[sources.filing_groups[key]]
        indices = np.concatenate([sources.position_indices.get(accession, np.array([], dtype=int)) for accession in df_versions.accession_number])
        cache[key] = resolve_prepared_books(df_versions, sources.df_holdings.iloc[indices], day)
        changed.append(key)

    usable = [key[1] for key, (_, df_state) in cache.items() if df_state.book_known.any()]
    if usable:
        active = max(usable)
        floor = _history_floor(active)
    cache = {key: value for key, value in cache.items() if key[1] >= floor}
    return cache, active, changed


def _cached_snapshot(cache: dict[tuple[str, pd.Timestamp], tuple[pd.DataFrame, pd.DataFrame]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine retained books and their states in the cache's existing insertion order."""
    df_books = pd.concat([value[0] for value in cache.values()], ignore_index=True)
    df_states = pd.concat([value[1] for value in cache.values()], ignore_index=True)
    return df_books, df_states


def _iter_public_book_cache(
    manifests: pd.DataFrame,
    holdings: pd.DataFrame,
) -> Iterator[
    tuple[
        pd.Timestamp,
        dict[tuple[str, pd.Timestamp], tuple[pd.DataFrame, pd.DataFrame]],
        dict[tuple[str, pd.Timestamp], tuple[pd.DataFrame, pd.DataFrame]],
    ]
]:
    """Schedule prepared book replacements, retaining unchanged book objects."""
    df_manifest = manifests.copy()
    df_manifest["_day"] = df_manifest["public_at"]
    df_manifest = df_manifest.loc[df_manifest["_day"].notna()]
    filing_groups = df_manifest.groupby(["cik", "period"], sort=False).indices
    position_indices = holdings.groupby("accession_number", sort=False).indices
    publications, days = _publication_schedule(df_manifest)
    sources = _PublicationSources(
        df_manifest,
        holdings,
        cast(Mapping[tuple[str, pd.Timestamp], NDArray[np.intp]], filing_groups),
        cast(Mapping[str, NDArray[np.intp]], position_indices),
        publications,
    )
    cache: dict[tuple[str, pd.Timestamp], tuple[pd.DataFrame, pd.DataFrame]] = {}
    active = pd.NaT
    for day in days:
        cache, active, changed = _reconcile_publication(sources, cache, active, day)
        if cache:
            yield pd.Timestamp(day), cache, {key: cache[key] for key in changed if key in cache}


def iter_public_books(
    manifests: pd.DataFrame,
    holdings: pd.DataFrame,
    trading_index: pd.DatetimeIndex,
    *,
    settle_days: int = F13_SETTLE_TRADING_DAYS,
) -> Iterator[tuple[pd.Timestamp, pd.DataFrame, pd.DataFrame]]:
    """Adapt raw vintages to the existing full reconciled-book snapshot interface."""
    df_manifest = prepare_book_vintages(manifests, holdings, trading_index, settle_days)
    for day, cache, _ in _iter_public_book_cache(df_manifest, holdings):
        df_books, df_states = _cached_snapshot(cache)
        yield day, df_books, df_states


F13_REVISION_MIN_MOVE = 0.001
COVERAGE_BREAK_DEFAULT = 0.50


def _apply_filing_band(h: pd.DataFrame, band: tuple[int, int]) -> pd.DataFrame:
    """Keep legacy rows within the declared deadline band, including unknown dates."""
    early, late = band
    if "filing_date" not in h.columns or "period" not in h.columns:
        logger.info("13F clean: no `filing_date`/`period` -> the [-%s, +%s] filing band CANNOT be applied on this read; every row kept", early, late)
        return h

    lag = (h["filing_date"] - (h["period"] + pd.Timedelta(days=SEC_13F_FILING_LAG_DAYS))).dt.days
    unknown = lag.isna()
    drop = ~lag.between(-early, late) & ~unknown
    if not drop.any():
        return h

    value = pd.to_numeric(h["value_usd"], errors="coerce") if "value_usd" in h.columns else None
    total = float(value.sum()) if value is not None else 0.0
    logger.info(
        "13F filing band [-%sd, +%sd] vs deadline: %s of %s row(s) dropped (%.3f%%) "
        "across %s filer(s), carrying %.3f%% of as-filed value (PRE unit-repair, so "
        "that share is only indicative); %s early, %s late, %s row(s) undated and kept",
        early,
        late,
        f"{int(drop.sum()):,}",
        f"{len(h):,}",
        100.0 * float(drop.mean()),
        f"{h.loc[drop, 'cik'].nunique():,}" if "cik" in h.columns else "?",
        100.0 * float(value[drop].sum()) / total if (value is not None and total > 0) else 0.0,
        f"{int((lag < -early).sum()):,}",
        f"{int((lag > late).sum()):,}",
        f"{int(unknown.sum()):,}",
    )
    return h[~drop]


def clean_holdings(
    holdings: pd.DataFrame,
    *,
    key: Sequence[str],
    filing_band: tuple[int, int] | None = None,
) -> pd.DataFrame:
    """Normalize legacy keys, then apply the filing band and amendment order.

    This latest-only adapter does not certify immutable filing completeness."""
    h = holdings.copy()
    for col in _DATES:
        if col in h.columns:
            h[col] = to_day(h[col])
    for col in ("shares", "value_usd", "call_value", "put_value"):
        h[col] = pd.to_numeric(h[col], errors="coerce") if col in h.columns else pd.Series(np.nan if col == "value_usd" else 0.0, index=h.index)
        if col == "shares":
            h[col] = h[col].fillna(0.0)

    h = h.dropna(subset=list(key))
    if filing_band is not None:
        h = _apply_filing_band(h, filing_band)

    if "filing_date" in h.columns:
        h = h.sort_values("filing_date")
    else:
        logger.info(
            "13F clean: no `filing_date` column -> amendments cannot be ordered, so `keep='last'` falls back to source order for %s duplicate key(s)",
            f"{int(h.duplicated(list(key)).sum()):,}",
        )
    return h.drop_duplicates(list(key), keep="last")


def _basis_grid(trading_index: pd.DatetimeIndex, holdings: pd.DataFrame) -> tuple[pd.DatetimeIndex, pd.Index]:
    """The saved-basis grid the production part would hold: every session plus each report period, per held ticker."""
    periods = pd.DatetimeIndex(pd.to_datetime(holdings["period"]).dropna().unique())
    return pd.DatetimeIndex(trading_index).union(periods), pd.Index(sorted(holdings["ticker"].dropna().astype(str).unique()))


def legacy_basis_frame(share_basis: ShareBasis | None, trading_index: pd.DatetimeIndex, holdings: pd.DataFrame) -> pd.DataFrame | None:
    """`security_basis` F of the legacy `ShareBasis` input, for the production 13F projections."""
    return None if share_basis is None else security_basis_frame(share_basis, *_basis_grid(trading_index, holdings))


def split_basis_frame(splits: pd.DataFrame | None, trading_index: pd.DatetimeIndex, holdings: pd.DataFrame) -> pd.DataFrame | None:
    """F from a legacy `splits` table (`ratio` column), for the production matched-quarter arithmetic."""
    if splits is None:
        return None
    index, tickers = _basis_grid(trading_index, holdings)
    keys = pd.MultiIndex.from_product([index, tickers])
    values = split_cum_product(splits, pd.Series(keys.get_level_values(1)), pd.Series(keys.get_level_values(0)))
    return pd.DataFrame(values.reshape(len(index), len(tickers)), index=index, columns=tickers)


def stamp_legacy_publication(
    holdings: pd.DataFrame, trading_index: pd.DatetimeIndex, *, settle_trading_days: int = F13_SETTLE_TRADING_DAYS
) -> pd.DataFrame:
    """Retain legacy late revisions; remove periods not yet public on the calendar."""
    h = holdings.copy()
    h["first_pub"] = availability_date(frame_column(h, "period"), trading_index, settle_trading_days=settle_trading_days)
    if "filing_date" not in h.columns:
        logger.info(
            "13F availability: no `filing_date` column -> `as_of` is the period's "
            "availability date for every row; the cutoff CANNOT be applied on this read, so "
            "a late filing is inside the first publication",
        )
        h["as_of"] = h["first_pub"]
    else:
        filing_date = frame_column(h, "filing_date")
        first_pub = frame_column(h, "first_pub")
        later = filing_date.notna() & first_pub.notna() & (filing_date > first_pub)
        h["as_of"] = first_pub.where(~later, filing_date)
    unavailable = frame_column(h, "as_of").isna()
    if unavailable.any():
        # An unpublishable predecessor must not enter the next quarter's differences.
        logger.info(
            "13F availability: %s row(s) across period(s) %s have no availability date inside the trading calendar (it ends %s) -> not emitted",
            f"{int(unavailable.sum()):,}",
            sorted({str(pd.Timestamp(p).date()) for p in cast(pd.Series, h.loc[unavailable, "period"]).unique()})[:6],
            str(pd.Timestamp(cast(Any, pd.DatetimeIndex(trading_index).max())).date()) if len(trading_index) else "(empty)",
        )
        h = cast(pd.DataFrame, h.loc[~unavailable])
    return cast(pd.DataFrame, h)


def _repair_legacy_values(df_holdings: pd.DataFrame, valuation_close: pd.DataFrame | None, value_basis: Mapping[str, Any] | None) -> pd.DataFrame:
    """Preserve legacy whole-row exclusions and its as-filed repair summary."""
    value_before = float(df_holdings["value_usd"].sum())
    df_holdings, df_register = repair_value_basis(df_holdings, valuation_close, value_basis)
    log_register(df_register, value_before)
    if "value_basis_excluded" in df_holdings:
        df_holdings = df_holdings.loc[~df_holdings["value_basis_excluded"]]
    return df_holdings


def clean_legacy_institutional_holdings(
    df_holdings: pd.DataFrame, valuation_close: pd.DataFrame | None, value_basis: Mapping[str, float] | None
) -> pd.DataFrame:
    """Prepare the existing broad no-manifest contract without certifying vintages."""
    df_holdings = clean_holdings(df_holdings, key=("ticker", "cik", "period"), filing_band=(45, 60))
    return _repair_legacy_values(df_holdings, valuation_close, value_basis)


_stamp_availability = stamp_legacy_publication


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


def _future_split_factor(splits: pd.DataFrame | None, tickers: pd.Series, dates: pd.Series) -> np.ndarray:
    """Legacy terminal factor: PROD(ratio) over splits effective after each date = total / prefix."""
    at = split_cum_product(splits, tickers, dates)
    if splits is None or splits.empty or not {"date", "ticker", "ratio"}.issubset(splits.columns):
        return np.ones(len(tickers), dtype="float64")
    s = splits.dropna(subset=["date", "ticker", "ratio"])
    s = s[pd.to_numeric(s["ratio"], errors="coerce") > 0]
    total_by_ticker = s.groupby(s["ticker"].astype(str))["ratio"].prod()
    return pd.Series(np.asarray(tickers, dtype=object)).astype(str).map(total_by_ticker).fillna(1.0).to_numpy() / at


def _split_factors(h: pd.DataFrame, splits: pd.DataFrame | None) -> dict:
    """Restate prior ticker-quarter shares by splits effective between its observed reports."""
    pairs = cast(pd.DataFrame, h[["ticker", "period"]]).drop_duplicates().sort_values(["ticker", "period"])
    if pairs.empty:
        return {}

    pairs["prev_period"] = pairs.groupby("ticker")["period"].shift(1)
    pairs = pairs.dropna(subset=["prev_period"])
    if pairs.empty:
        return {}
    f_now = _future_split_factor(splits, frame_column(pairs, "ticker"), frame_column(pairs, "period"))
    f_prev = _future_split_factor(splits, frame_column(pairs, "ticker"), frame_column(pairs, "prev_period"))
    factor = np.divide(f_prev, f_now, out=np.ones_like(f_prev), where=f_now > 0)
    n = int((np.abs(factor - 1.0) > 1e-9).sum())
    if n:
        logger.info(
            "13F split restatement: %s of %s (ticker, quarter) pairs had the prior quarter's share count rebased (V5 trigger count)", n, len(pairs)
        )
    return {(t, p): f for t, p, f in zip(pairs["ticker"], pairs["period"], factor, strict=False)}


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
    # happened once. `clean_holdings` CREATES a missing numeric leg as 0.0,
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


def build_institutional_feature_panel(
    frames: PriceFrames,
    holdings: pd.DataFrame | None,
    *,
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
    if filings is not None and df_holdings.empty:
        return df_holdings
    if filings is not None:
        df_holdings, df_register = repair_value_basis(df_holdings, valuation_close, value_basis)
        log_register(df_register, float(df_holdings.value_usd.sum()))
        if "value_basis_excluded" in df_holdings:
            df_holdings.loc[df_holdings.value_basis_excluded, ["shares", "value_usd"]] = np.nan
        return df_holdings
    df_holdings = clean_legacy_institutional_holdings(df_holdings, valuation_close, value_basis)
    if df_holdings.empty:
        return df_holdings
    df_holdings = _stamp_availability(df_holdings, trading_index, settle_trading_days=settle_trading_days)
    _report_first_publication_shortfall(df_holdings)
    _report_availability_coverage(df_holdings)
    return df_holdings


def build_prepared_institutional_feature_panel(
    frames: PriceFrames,
    holdings: pd.DataFrame,
    *,
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
        replace(frames, security_basis=legacy_basis_frame(share_basis, frames.trading_index, holdings)),
        settle_trading_days,
        trading_calendar=frames.trading_index,
    )
    if not fields:
        return pd.DataFrame(columns=["date", "ticker"])
    emission = {name: EMISSION[name] for name in fields}
    return build_peer_relative_panel(fields, frames.peers, emission=emission, availability=frames.availability)


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
    df_basis = split_basis_frame(splits, trading_index, holdings)
    return snapshot_feature_history(snapshots, df_basis=df_basis, min_prior_holders=min_prior_holders)


def iter_snapshot_books(df_compact: pd.DataFrame, df_states: pd.DataFrame) -> Iterator[tuple[pd.Timestamp, pd.DataFrame, pd.DataFrame]]:
    """Replay compact replacements against the global active-quarter floor: the full public snapshot of each day."""
    cache: dict[tuple[str, pd.Timestamp], int] = {}
    active: Any = pd.NaT
    if df_states.empty:
        return

    publications = df_states.groupby("_day", sort=True).indices
    days = pd.DatetimeIndex(df_states["_day"].unique()).sort_values()
    for day in days:
        df_events = df_states.iloc[publications[day]]
        floor = _history_floor(active)
        for event in df_events.loc[df_events.period.ge(floor)].itertuples():
            cache[(cast(str, event.cik), cast(pd.Timestamp, event.period))] = cast(int, event.Index)
        df_current = df_states.iloc[list(cache.values())]
        known = df_current.loc[df_current.book_known, "period"]
        if not known.empty:
            active = known.max()
        floor = _history_floor(active)
        cache = {key: index for key, index in cache.items() if key[1] >= floor}
        if not cache:
            continue
        df_current = df_states.iloc[list(cache.values())]
        bounds = df_current[["_start", "_stop"]].to_numpy(dtype=np.intp)
        indices = np.concatenate([np.arange(start, stop, dtype=np.intp) for start, stop in bounds])
        yield pd.Timestamp(day), df_compact.iloc[indices], df_current.drop(columns=["_day", "_start", "_stop"]).reset_index(drop=True)


def _snapshot_levels(df_current: pd.DataFrame, df_states: pd.DataFrame, names: list[str]) -> pd.DataFrame:
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


def _snapshot_votes(df_matched: pd.DataFrame) -> pd.DataFrame:
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
    keys = pd.Series(tickers, index=tickers)
    prior = lookup_asof(df_basis, keys, pd.Series(prior_period, index=tickers))
    now = lookup_asof(df_basis, keys, pd.Series(period, index=tickers))
    return convert_quantities(1.0, prior, now)


def snapshot_feature_history(
    snapshots: Iterator[tuple[pd.Timestamp, pd.DataFrame, pd.DataFrame]],
    *,
    df_basis: pd.DataFrame | None,
    min_prior_holders: int = MIN_PRIOR_HOLDERS,
) -> pd.DataFrame:
    """The replaced production engine: recompute levels and matched cohorts from the full snapshot of each public day."""
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
        out = _snapshot_levels(current, current_states, names)
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
        df_votes = _snapshot_votes(matched)
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
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
