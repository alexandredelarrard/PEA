"""
short_flow_features.py  (src/data_aggregate/utils/institutionals/short_flow_features.py)
----------------------------------------------------------------------------------------
Short-flow features: FINRA RegSHO daily short-sale VOLUME (`ic_shortvol_*`, not short interest)
and SEC fails-to-deliver (`ic_ftd_*`). Ratios are volume-weighted (SUM short / SUM total over N
days); `ic_shortvol_market_coverage` is RegSHO's measured share of tape volume. Both tape-volume
denominators (ADV20 and coverage) sum the issuer's secondary share classes, as the class-summed
numerators do (`secondary_class_volume`). Point-in-time:
every `ic_shortvol_*` leg is shifted `SHORTVOL_PUB_LAG` trading day; an FTD ZIP becomes knowable as
a whole at half-month end plus the configured calendar-day lag (estimated_policy_21d),
moved to the next actual session and held until the next ZIP. This is an estimated policy,
not an observed historical SEC release timestamp. Verified issuer absence on a covered,
dated-supported source is 0; unsupported absence remains NaN.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from src.constants.constants import CANONICAL_ROLES
from src.data_aggregate.utils.common.data_utils import to_day
from src.data_aggregate.utils.common.errors import _absent, _empty_panel
from src.data_aggregate.utils.common.panel import build_peer_relative_panel
from src.data_aggregate.utils.common.price_frames import PriceFrames
from src.data_aggregate.utils.common.xs import self_history_z
from src.data_aggregate.utils.institutionals.inputs import QuantityBasis
from src.utils.string import yahoo_symbol

logger = logging.getLogger(__name__)

#: RegSHO: day-t short volume is public on t+1.
SHORTVOL_PUB_LAG = 1

#: Internal self-history z window for the price-regime and FTD-persistence legs (the z itself is not emitted).
Z_WINDOW = 252
Z_MIN_PERIODS = 126

#: Count of the last 30 trading days with standardized FTD pressure above `Z_HIGH`.
PERSISTENCE_WINDOW = 30
Z_HIGH = 1.0

#: Volume-weighted ratio windows; `BASE_WINDOW` backs the z-score, acceleration and price interactions.
RATIO_WINDOWS: tuple[int, ...] = (5, 20, 60)
BASE_WINDOW = 20

#: Total-return window the two price-interaction features condition on.
RET_WINDOW = 20

#: Every leg is emitted in raw economic units.
EMISSION: dict[str, str] = {
    "ic_shortvol_ratio_5d": "raw",
    "ic_shortvol_ratio_20d": "raw",
    "ic_shortvol_ratio_60d": "raw",
    "ic_shortvol_acceleration": "raw",
    "ic_shortvol_turnover_20d": "raw",
    "ic_shortvol_high_x_weak_price": "raw",
    "ic_shortvol_high_x_strong_price": "raw",
    "ic_shortvol_market_coverage": "raw",
    "ic_ftd_to_adv20": "raw",
    "ic_ftd_persistence_30d": "raw",
}


@dataclass(frozen=True)
class ShortFlowCapital:
    """PIT outstanding counts and terminal-restated ordinary-share tape on one target basis."""

    shares_out: pd.DataFrame | None
    tape: pd.DataFrame | None
    tape_supported: pd.DataFrame | None = None


@dataclass(frozen=True)
class FtdSettlement:
    """Target-basis outstanding fails and independently observed source dates."""

    quantities: pd.DataFrame
    on_file: pd.Series


@dataclass(frozen=True)
class FtdPublication:
    """Global settlement-date/ZIP coverage and each ZIP's existing public availability day."""

    coverage: pd.DataFrame
    available_by_period: dict[str, pd.Timestamp]


def _min_periods(window: int) -> int:
    """Half the window, floored at 3: enough of the window present to be a window."""
    return max(3, window // 2)


def _pivot(hist: pd.DataFrame, value: str, idx: pd.DatetimeIndex) -> pd.DataFrame:
    wide = hist.groupby(["date", "ticker"])[value].sum(min_count=1).unstack("ticker")
    wide.index = pd.to_datetime(wide.index).normalize()
    return wide.reindex(idx)


def _ftd_period_end(period: str) -> pd.Timestamp:
    """Parse a source ZIP tag, never inferring its half from a settlement date."""
    if not isinstance(period, str) or re.fullmatch(r"\d{6}[ab]", period) is None:
        raise ValueError(f"Invalid FTD ZIP period: {period!r}")
    try:
        month = pd.Timestamp(year=int(period[:4]), month=int(period[4:6]), day=1)
    except ValueError as exc:
        raise ValueError(f"Invalid FTD ZIP period: {period!r}") from exc
    return month + pd.Timedelta(days=14) if period[-1] == "a" else month + pd.offsets.MonthEnd(0)


def _ftd_estimated_available_date(period: str, calendar_days: int) -> pd.Timestamp:
    """Estimated ZIP policy day; the publisher snaps it to an actual market session."""
    return _ftd_period_end(period) + pd.Timedelta(days=calendar_days)


def _ftd_available_dates(
    fails_hist: pd.DataFrame,
    calendar_days: int,
) -> dict[str, pd.Timestamp]:
    """Derive deterministic estimated policy days from persisted ZIP periods and the owner setting."""
    if isinstance(calendar_days, bool) or not isinstance(calendar_days, int) or calendar_days < 1:
        raise ValueError("FTD availability calendar days must be a positive integer")
    if "period" not in fails_hist:
        raise ValueError("FTD source is missing its persisted ZIP period")
    if "date" not in fails_hist:
        raise ValueError("FTD source is missing settlement date")
    source_periods = fails_hist[["date", "period"]].drop_duplicates().copy()
    source_periods["date"] = to_day(source_periods["date"])
    if source_periods.isna().any().any():
        raise ValueError("FTD source has a null settlement date or ZIP period")
    if source_periods.groupby("date")["period"].nunique().gt(1).any():
        raise ValueError("FTD settlement date belongs to multiple ZIP periods")
    available = {period: _ftd_estimated_available_date(period, calendar_days) for period in source_periods["period"].unique()}
    return dict(sorted(available.items()))


def _publish_ftd_zip_states(
    settlement_state: pd.DataFrame,
    fails_hist: pd.DataFrame,
    idx: pd.DatetimeIndex,
    available_by_period: dict[str, pd.Timestamp],
) -> pd.DataFrame:
    """Expose each ZIP's latest cumulative FTD state from its first tradable session on or after its
    publication day until the next ZIP; raises when a ZIP would publish before its own settlement dates.
    """
    periods: dict[str, list[pd.Timestamp]] = {}
    source_periods = fails_hist[["date", "period"]].drop_duplicates().copy()
    source_periods["date"] = to_day(source_periods["date"])
    for day, period in source_periods.itertuples(index=False, name=None):
        periods.setdefault(str(period), []).append(day)

    events: list[tuple[int, str, pd.Series]] = []
    for period, days in periods.items():
        observed_days = pd.DatetimeIndex(days).intersection(cast(pd.DatetimeIndex, settlement_state.index)).sort_values()
        if observed_days.empty:
            continue
        if period not in available_by_period:
            raise ValueError(f"FTD available date missing period {period}")
        available_date = available_by_period[period]
        if pd.isna(available_date):
            raise ValueError(f"FTD available date is missing for {period}")
        if available_date < observed_days[-1]:
            raise ValueError(f"FTD ZIP {period} becomes available before its latest settlement date")
        publish_at = int(idx.searchsorted(available_date, side="left"))
        if publish_at >= len(idx):
            continue
        latest = settlement_state.reindex(observed_days).iloc[-1]
        events.append((publish_at, period, latest))

    published = pd.DataFrame(np.nan, index=idx, columns=settlement_state.columns)
    events.sort(key=lambda event: (event[0], event[1]))
    current_period = ""
    forward_events = []
    for event in events:
        if event[1] > current_period:
            forward_events.append(event)
            current_period = event[1]
    for position, (start, _, state) in enumerate(forward_events):
        stop = forward_events[position + 1][0] if position + 1 < len(forward_events) else len(idx)
        if stop > start:
            published.iloc[start:stop] = state.to_numpy()
    return published


def secondary_class_volume(lines: pd.DataFrame, bars: pd.DataFrame, idx: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Wide secondary-class volume in canonical shares, and company-days lacking supported active-class input.

    A class counts only on its `security_master` dates. Yahoo volume is restated to today's split basis, so a class
    enters at its latest conversion ratio: a dated ratio step is a split of one class (BRK-B's 50:1, BRK-A 30 -> 1,500)
    that Yahoo already restated. Ratios must be finite and positive; bars and converted quantities finite and nonnegative.
    Unsupported active legs add nothing and flag the complete company denominator as unknown.
    """
    idx = pd.DatetimeIndex(idx).normalize()
    if lines.empty:
        return pd.DataFrame(index=idx, dtype="float64"), pd.DataFrame(index=idx, dtype="bool")
    work = lines.assign(
        company=lines["canonical_company"].astype(str),
        symbol=lines["market_symbol"].map(yahoo_symbol),
        ratio=pd.to_numeric(lines["conversion_ratio"], errors="coerce"),
        start=pd.to_datetime(lines["valid_from"]).dt.normalize(),
        end=pd.to_datetime(lines["valid_to"]).dt.normalize(),
    ).sort_values("start", kind="stable")
    tape = bars.assign(date=to_day(bars["date"])).drop_duplicates(["ticker", "date"], keep="last")
    tape = tape.pivot(index="date", columns="ticker", values="volume").reindex(idx) if not tape.empty else pd.DataFrame(index=idx)
    companies = sorted(work["company"].unique())
    volume = pd.DataFrame(0.0, index=idx, columns=companies)
    missing = pd.DataFrame(False, index=idx, columns=companies)
    gaps: list[str] = []
    for (company, symbol), group in work.groupby(["company", "symbol"], sort=True):
        active = np.zeros(len(idx), dtype=bool)
        for start, end in zip(group["start"], group["end"], strict=True):
            window = idx >= start if pd.notna(start) else np.ones(len(idx), dtype=bool)
            active |= window & (idx < end) if pd.notna(end) else window
        bar = pd.to_numeric(tape[symbol], errors="coerce") if symbol in tape.columns else pd.Series(np.nan, index=idx)
        ratio = float(group["ratio"].iloc[-1])
        raw = bar.to_numpy()
        with np.errstate(over="ignore", invalid="ignore"):
            converted = ratio * raw
            supported = np.isfinite(ratio) & (ratio > 0) & np.isfinite(raw) & (raw >= 0) & np.isfinite(converted)
            volume[company] += np.where(active & supported, converted, 0.0)
        gap = active & ~supported
        missing[company] |= gap
        if gap.any():
            days = idx[gap]
            gaps.append(f"{company} {symbol} {int(gap.sum())} day(s) {days[0].date()}..{days[-1].date()}")
    if gaps:
        logger.warning(
            "secondary-class volume: %s company-day(s) have an active class without supported input, which adds no volume: %s",
            f"{int(missing.to_numpy().sum()):,}",
            "; ".join(gaps),
        )
    finite = np.isfinite(volume)
    missing |= ~finite
    return volume.where(finite, 0.0), missing


def with_class_volume(volume: pd.DataFrame | None, class_volume: pd.DataFrame | None) -> pd.DataFrame | None:
    """`volume` plus each company's secondary-class volume; a company with no class, or a day with no canonical bar, is unchanged."""
    if volume is None or class_volume is None or class_volume.empty:
        return volume
    columns = volume.columns.intersection(class_volume.columns)
    if columns.empty:
        return volume
    out = volume.copy()
    out[columns] = volume[columns] + class_volume.reindex(index=volume.index, columns=columns).fillna(0.0)
    return out


def _guard_coverage(cov: pd.DataFrame) -> pd.DataFrame:
    """NULL `ic_shortvol_market_coverage` above 1.0: off-exchange volume cannot exceed the tape, so
    such cells (typically reused-ticker windows) describe another security. Only this leg has a
    physical ceiling to detect that by.
    """
    over = cov.gt(1.0)
    n = int(over.to_numpy().sum())
    if n:
        bad = [c for c in cov.columns if bool(over[c].any())]
        logger.info(
            "coverage guard: %s of %s non-null `ic_shortvol_market_coverage` cells "
            "above 1.0 -> nulled (off-exchange volume cannot exceed the tape; these "
            "are reused-ticker windows). Tickers: %s",
            f"{n:,}",
            f"{int(cov.notna().to_numpy().sum()):,}",
            ", ".join(sorted(bad)),
        )
    return cov.mask(over)


def _market_identity_conflicts(work: pd.DataFrame, idx: pd.DatetimeIndex) -> dict[object, np.ndarray]:
    """A security has one dated owner and no simultaneous rejected canonical verdict, across every symbol key."""
    conflicts = {}
    for security, group in work.groupby("security_id", sort=False, dropna=False):
        owners = {}
        blocked = np.zeros(len(idx), dtype=bool)
        for row in group.itertuples(index=False):
            active = np.asarray((idx >= cast(pd.Timestamp, row.start)) & ((idx < cast(pd.Timestamp, row.end)) if pd.notna(row.end) else True))
            owners.setdefault(str(row.canonical_company), np.zeros(len(idx), dtype=bool))[:] |= active
            rejected = row.lineage_role not in (*CANONICAL_ROLES, "secondary_class", "acquired_constituent")
            blocked |= active & (rejected or (row.lineage_role in CANONICAL_ROLES and not row.accepted))
        conflicts[security] = blocked | (np.sum(list(owners.values()), axis=0) > 1)
    return conflicts


def _proven_tenure_mask(
    idx: pd.DatetimeIndex,
    columns: pd.Index,
    master: pd.DataFrame | None,
    finra: pd.DataFrame | None,
) -> pd.DataFrame:
    """Canonical market intervals first; accepted FINRA fallback supports only observed dates without a master."""
    mask = pd.DataFrame(False, index=idx, columns=columns)
    required = {
        "canonical_company",
        "security_id",
        "source",
        "source_symbol",
        "security_class",
        "conversion_ratio",
        "lineage_role",
        "valid_from",
        "valid_to",
    }
    work = pd.DataFrame(columns=["source", "key"])
    if master is not None and required.issubset(master):
        work = master.assign(
            start=pd.to_datetime(master["valid_from"], errors="coerce").dt.normalize(),
            end=pd.to_datetime(master["valid_to"], errors="coerce").dt.normalize(),
            key=master["source_symbol"].astype(str).str.upper().str.replace(r"[^A-Z0-9]", "", regex=True),
        )
        ratio = pd.to_numeric(work["conversion_ratio"], errors="coerce")
        work["accepted"] = (
            work["lineage_role"].isin(CANONICAL_ROLES)
            & work["security_class"].fillna("").str.fullmatch(r"common|class_[A-Z]")
            & np.isfinite(ratio)
            & ratio.gt(0)
            & work["security_id"].fillna("").ne("")
            & work["source_symbol"].fillna("").str.fullmatch(r"[A-Z0-9][A-Z0-9./-]*")
            & work["canonical_company"].notna()
            & work["canonical_company"].astype(str).str.strip().ne("")
            & work["start"].notna()
            & (master["valid_to"].isna() | work["end"].notna())
        )
    conflicts = _market_identity_conflicts(work, idx) if not work.empty else {}
    for _, group in work.loc[work["source"].eq("ftd")].groupby("key", sort=False):
        hits, summed, admitted = {}, {}, {}
        rejected = np.zeros(len(idx), dtype=bool)
        for row in group.itertuples(index=False):
            active = np.asarray((idx >= cast(pd.Timestamp, row.start)) & ((idx < cast(pd.Timestamp, row.end)) if pd.notna(row.end) else True))
            hits.setdefault(row.security_id, np.zeros(len(idx), dtype=bool))[:] |= active
            if row.lineage_role in (*CANONICAL_ROLES, "secondary_class"):
                summed.setdefault(row.security_id, np.zeros(len(idx), dtype=bool))[:] |= active
            if row.lineage_role not in (*CANONICAL_ROLES, "secondary_class", "acquired_constituent"):
                rejected |= active
            if row.accepted and row.canonical_company in columns:
                admitted.setdefault(row.canonical_company, np.zeros(len(idx), dtype=bool))[:] |= active & ~conflicts[row.security_id]
        resolved = (np.sum(list(hits.values()), axis=0) == 1) | ((np.sum(list(summed.values()), axis=0) == 1) & ~rejected)
        for ticker, active in admitted.items():
            mask.loc[active & resolved, ticker] = True
    known = set(master["canonical_company"].dropna()) if master is not None and "canonical_company" in master else set()
    if finra is not None and {"date", "ticker", "security_id", "lineage_role", "security_class"}.issubset(finra):
        selected = (
            finra["ticker"].isin(columns)
            & ~finra["ticker"].isin(known)
            & finra["lineage_role"].isin(CANONICAL_ROLES)
            & finra["security_class"].fillna("").str.fullmatch(r"common|class_[A-Z]")
            & finra["security_id"].fillna("").str.fullmatch(r"S\d{10}:[A-Z0-9][A-Z0-9.-]*")
        )
        for ticker, group in finra.loc[selected].groupby("ticker", sort=False):
            mask[ticker] |= idx.isin(pd.to_datetime(group["date"], errors="coerce").dt.normalize())
    return mask


def _shortvol_fields(
    hist: pd.DataFrame,
    idx: pd.DatetimeIndex,
    shares_out: pd.DataFrame | None,
    close_total: pd.DataFrame | None,
    volume: pd.DataFrame | None,
    tenure_mask: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """RegSHO ratio, acceleration, turnover, price-interaction and coverage legs, each shifted by the publication lag."""
    # Prepare issuer-valid daily quantities and reuse their aligned tenure mask.
    df_short = _pivot(hist, "short_volume", idx)
    df_total = _pivot(hist, "total_volume", idx)
    df_tenure = tenure_mask.reindex(index=idx, columns=df_short.columns, fill_value=False) if tenure_mask is not None else None
    if df_tenure is not None:
        df_short = df_short.where(df_tenure)
        df_total = df_total.where(df_tenure)
    df_observed = df_short.notna() & df_total.notna()

    # Compute weighted window ratios and reuse the base-window sum for turnover.
    ratios, df_turnover = _shortvol_window_fields(df_short, df_total, df_observed, idx, shares_out)
    fields = {f"ic_shortvol_ratio_{window}d": df_ratio for window, df_ratio in ratios.items()}
    fields["ic_shortvol_acceleration"] = ratios[BASE_WINDOW] - ratios[max(RATIO_WINDOWS)]
    if df_turnover is not None:
        fields["ic_shortvol_turnover_20d"] = df_turnover

    # Price interactions use total-return closes and the already-lagged ratio history.
    if close_total is not None and not close_total.empty:
        df_z = self_history_z(ratios[BASE_WINDOW], window=Z_WINDOW, min_periods=Z_MIN_PERIODS)
        df_returns = close_total.reindex(index=idx).reindex(columns=df_short.columns).pct_change(RET_WINDOW)
        df_high = df_z.clip(lower=0.0)
        fields["ic_shortvol_high_x_weak_price"] = df_high * (-df_returns).clip(lower=0.0)
        fields["ic_shortvol_high_x_strong_price"] = df_high * df_returns.clip(lower=0.0)
    if volume is not None and not volume.empty:
        df_tape = volume.reindex(index=idx).reindex(columns=df_short.columns)
        fields["ic_shortvol_market_coverage"] = _shortvol_market_coverage(df_total, df_tape)

    # Publication does not extend a ticker beyond its proven current-issuer tenure.
    if df_tenure is not None:
        fields = {name: df_field.where(df_tenure) for name, df_field in fields.items()}
    return fields


def _shortvol_window_fields(
    df_short: pd.DataFrame, df_total: pd.DataFrame, df_observed: pd.DataFrame, idx: pd.DatetimeIndex, shares_out: pd.DataFrame | None
) -> tuple[dict[int, pd.DataFrame], pd.DataFrame | None]:
    """Normalize window short-sale totals by total volume and optional PIT shares."""
    ratios: dict[int, pd.DataFrame] = {}
    df_turnover = None
    df_shares = shares_out.reindex(index=idx).reindex(columns=df_short.columns) if shares_out is not None and not shares_out.empty else None

    # Calculate each numerator once; the base sum also supplies share turnover.
    for window in RATIO_WINDOWS:
        min_periods = _min_periods(window)
        df_numerator = df_short.where(df_observed).rolling(window, min_periods=min_periods).sum()
        df_denominator = df_total.where(df_observed).rolling(window, min_periods=min_periods).sum()
        df_ratio = (df_numerator / df_denominator.where(df_denominator > 0)).replace([np.inf, -np.inf], np.nan).where(df_observed)
        ratios[window] = df_ratio.shift(SHORTVOL_PUB_LAG)
        if window == BASE_WINDOW and df_shares is not None:
            df_turnover = (
                (df_short.rolling(window, min_periods=min_periods).sum() / df_shares.where(df_shares > 0))
                .replace([np.inf, -np.inf], np.nan)
                .where(df_short.notna())
                .shift(SHORTVOL_PUB_LAG)
            )
    return ratios, df_turnover


def _shortvol_market_coverage(df_total: pd.DataFrame, df_tape: pd.DataFrame) -> pd.DataFrame:
    """Compute split-consistent off-exchange tape coverage and report its supported range."""
    min_periods = _min_periods(BASE_WINDOW)

    # Both prepared quantity legs already have the same ordinary-share basis.
    paired = df_total.notna() & df_tape.notna()
    df_numerator = df_total.where(paired).rolling(BASE_WINDOW, min_periods=min_periods).sum()
    df_denominator = df_tape.where(paired).rolling(BASE_WINDOW, min_periods=min_periods).sum()
    df_coverage = (df_numerator / df_denominator.where(df_denominator > 0)).replace([np.inf, -np.inf], np.nan).where(paired)
    df_coverage = _guard_coverage(df_coverage)

    # Keep the existing coverage diagnostic on settlement-day quantities.
    live = df_coverage.to_numpy(dtype="float64", na_value=np.nan).ravel()
    live = live[np.isfinite(live)]
    if len(live):
        p05, p50, p95 = np.percentile(live, [5, 50, 95])
        logger.info(
            "RegSHO market coverage (off-exchange share of tape volume): p50 %.1f%%, p05 %.1f%%, p95 %.1f%% over %s ticker-days",
            100 * p50,
            100 * p05,
            100 * p95,
            len(live),
        )
    return df_coverage.shift(SHORTVOL_PUB_LAG)


def _fails_fields(
    settlement: FtdSettlement,
    capital: ShortFlowCapital,
    publication: FtdPublication,
    tenure_mask: pd.DataFrame | None,
) -> dict[str, pd.DataFrame]:
    """Compute compatible quantity ratios and persistence before publishing each ZIP's state."""
    df_fails, on_file = settlement.quantities, settlement.on_file
    idx = pd.DatetimeIndex(df_fails.index)
    fields: dict[str, pd.DataFrame] = {}
    df_pct_shares = None
    df_to_adv = None

    if capital.shares_out is not None and not capital.shares_out.empty:
        df_shares = capital.shares_out.reindex(index=idx).reindex(columns=df_fails.columns)
        df_pct_shares = (df_fails / df_shares.where(df_shares > 0)).replace([np.inf, -np.inf], np.nan)
    if capital.tape is not None and not capital.tape.empty:
        df_adv = capital.tape.reindex(index=idx).reindex(columns=df_fails.columns).rolling(BASE_WINDOW, min_periods=_min_periods(BASE_WINDOW)).mean()
        df_supported = _rolling_tape_support(capital.tape_supported, idx, df_fails.columns)
        df_to_adv = (df_fails / df_adv.where(df_adv > 0)).replace([np.inf, -np.inf], np.nan).where(df_supported)
        fields["ic_ftd_to_adv20"] = _publish_ftd_zip_states(df_to_adv, publication.coverage, idx, publication.available_by_period)

    # Calculate settlement persistence, then expose each ZIP as one public state.
    basis = df_pct_shares if df_pct_shares is not None else df_to_adv
    if basis is not None:
        df_persistence = _ftd_settlement_persistence(basis, on_file, idx)
        fields["ic_ftd_persistence_30d"] = _publish_ftd_zip_states(df_persistence, publication.coverage, idx, publication.available_by_period)

    # Retain unknown values outside the issuer's applicable published history.
    if tenure_mask is not None:
        for name, df_field in fields.items():
            fields[name] = df_field.where(tenure_mask.reindex(index=df_field.index, columns=df_field.columns, fill_value=False))
    return fields


def _rolling_tape_support(supported: pd.DataFrame | None, idx: pd.DatetimeIndex, columns: pd.Index) -> pd.DataFrame:
    """An active class with missing input makes its complete rolling tape denominator unknown."""
    if supported is None:
        return pd.DataFrame(True, index=idx, columns=columns)
    df_supported = supported.reindex(index=idx, columns=columns, fill_value=True).fillna(False)
    return df_supported.astype(float).rolling(BASE_WINDOW, min_periods=1).min().eq(1)


def prepare_fails_quantities(
    fails_hist: pd.DataFrame,
    coverage: pd.DataFrame,
    idx: pd.DatetimeIndex,
    columns: pd.Index | None,
    tenure_mask: pd.DataFrame | None,
    basis: QuantityBasis,
) -> FtdSettlement:
    """Rebase explicit dated canonical fails, including verified source zeros, to target ordinary shares."""
    df_fails, on_file = _settlement_fails(fails_hist, coverage, idx, columns, tenure_mask)
    source = pd.Series(idx, index=idx)
    target = pd.Series(basis.target_date, index=idx)
    for ticker in df_fails.columns:
        names = pd.Series(ticker, index=idx)
        df_fails[ticker] = basis.events.rebase(df_fails[ticker], names, source, target)
    return FtdSettlement(df_fails, on_file)


def _settlement_fails(
    fails_hist: pd.DataFrame, coverage: pd.DataFrame, idx: pd.DatetimeIndex, columns: pd.Index | None, tenure_mask: pd.DataFrame | None
) -> tuple[pd.DataFrame, pd.Series]:
    """Align explicit issuer observations; global period metadata cannot create another issuer's zero."""
    df_fails = _pivot(fails_hist, "fails_quantity", idx) if not fails_hist.empty else pd.DataFrame(index=idx, dtype="float64")
    if columns is not None:
        df_fails = df_fails.reindex(columns=columns)
    covered = pd.DatetimeIndex(to_day(coverage["date"]).dropna().unique())
    on_file = pd.Series(idx.isin(covered), index=idx)

    # Preserve global-date coverage before applying ticker-specific issuer support.
    if tenure_mask is not None:
        df_fails = df_fails.where(tenure_mask.reindex(index=idx, columns=df_fails.columns, fill_value=False))

    # Report the same observed-date coverage used to distinguish zero from unknown.
    logger.info(
        "FTD period metadata covers %s of %s trading days (%.1f%%); missing issuer observations remain unknown",
        int(on_file.sum()),
        len(idx),
        100 * float(on_file.mean()),
    )
    return df_fails, on_file


def _ftd_settlement_persistence(basis: pd.DataFrame, on_file: pd.Series, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Count supported high-FTD events; observed no-fail flags are independent of undefined constant-series z."""
    df_z = self_history_z(basis, window=Z_WINDOW, min_periods=Z_MIN_PERIODS)
    df_covered = pd.DataFrame({column: on_file for column in basis.columns}, index=idx)

    # Complete observed zeros establish no-fail event flags; zero standard deviation leaves z undefined.
    df_applicable = df_covered & basis.notna().cummax()
    df_expected = df_applicable.astype("float64").rolling(Z_WINDOW, min_periods=1).sum()
    df_observed = basis.notna().astype("float64").rolling(Z_WINDOW, min_periods=1).sum()
    df_zeros = basis.eq(0.0).astype("float64").rolling(Z_WINDOW, min_periods=1).sum()
    df_neutral = df_applicable & basis.eq(0.0) & df_expected.ge(Z_MIN_PERIODS) & df_observed.eq(df_expected) & df_zeros.eq(df_expected)

    # Aggregate the same threshold flags on the settlement calendar before publication.
    df_flags = (df_z > Z_HIGH).astype("float64").where(df_z.notna())
    df_flags = df_flags.mask(df_z.isna() & df_neutral, 0.0)
    return df_flags.rolling(PERSISTENCE_WINDOW, min_periods=_min_periods(PERSISTENCE_WINDOW)).sum().where(df_covered)


def build_short_flow_feature_panel(
    frames: PriceFrames,
    short_history: pd.DataFrame | None,
    settlement: FtdSettlement | None,
    capital: ShortFlowCapital,
    publication: FtdPublication | None,
    tenure_mask: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute the short/FTD families from prepared common-basis quantities and public ZIP metadata."""
    if _absent(short_history) and settlement is None:
        return _empty_panel()
    idx = pd.DatetimeIndex(frames.trading_index).normalize().unique().sort_values()
    if capital.tape is not None:
        # Invalid company sums must not turn into finite partial rolling denominators.
        df_supported = ~capital.tape.isin([np.inf, -np.inf]) & ~capital.tape.lt(0)
        if capital.tape_supported is not None:
            df_supported &= capital.tape_supported.reindex(index=capital.tape.index, columns=capital.tape.columns, fill_value=True)
        capital = ShortFlowCapital(capital.shares_out, capital.tape, df_supported)
    fields: dict[str, pd.DataFrame] = {}
    if short_history is not None and not short_history.empty and {"short_volume", "total_volume"}.issubset(short_history.columns):
        fields.update(_shortvol_fields(short_history, idx, capital.shares_out, frames.close_total, capital.tape, tenure_mask))
        if "ic_shortvol_market_coverage" in fields:
            df_coverage = fields["ic_shortvol_market_coverage"]
            df_supported = _rolling_tape_support(capital.tape_supported, idx, df_coverage.columns)
            fields["ic_shortvol_market_coverage"] = df_coverage.where(df_supported.shift(SHORTVOL_PUB_LAG, fill_value=False))
    if settlement is not None and publication is not None:
        fields.update(_fails_fields(settlement, capital, publication, tenure_mask))

    # Preserve data-dependent field omission and the existing raw float32 emitter.
    for name in list(fields):
        df_field = fields[name]
        if df_field is None or df_field.empty or not df_field.notna().any().any():
            fields.pop(name)
    if not fields:
        return pd.DataFrame(columns=["date", "ticker"])
    emission = {name: EMISSION[name] for name in fields}
    return build_peer_relative_panel(fields, frames.peers, emission=emission, availability=frames.availability)
