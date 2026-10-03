"""
short_flow_features.py  (src/data_aggregate/utils/institutionals/short_flow_features.py)
----------------------------------------------------------------------------------------
Short-flow features: FINRA RegSHO daily short-sale VOLUME (`ic_shortvol_*`, not short interest)
and SEC fails-to-deliver (`ic_ftd_*`). Ratios are volume-weighted (SUM short / SUM total over N
days); `ic_shortvol_market_coverage` is RegSHO's measured share of tape volume. Point-in-time:
every `ic_shortvol_*` leg is shifted `SHORTVOL_PUB_LAG` trading day; an FTD ZIP becomes knowable as
a whole at period end plus `FTD_HISTORICAL_LAG_DAYS` past weekends (the fresh latest cached ZIP may
use its mtime in `MARKET_TIMEZONE`), moved to the next session and held until the next ZIP. An
absent FTD ticker is 0 only on a date the file covers; an uncovered date is NaN.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.constants.constants import (
    FTD_HISTORICAL_LAG_DAYS,
    FTD_LATEST_PERIOD_MAX_AGE_DAYS,
    FTD_RECENT_CACHE_DAYS,
    FTD_ZIP_NAME_TEMPLATE,
    MARKET_TIMEZONE,
)
from src.data_aggregate.utils.common.data_utils import to_day
from src.data_aggregate.utils.common.errors import _empty_panel
from src.data_aggregate.utils.common.panel import build_peer_relative_panel
from src.data_aggregate.utils.common.pit import fundamentals_to_daily
from src.data_aggregate.utils.common.price_frames import PriceFrames
from src.data_aggregate.utils.common.xs import self_history_z
from src.data_aggregate.utils.institutionals.availability import InstitutionalAvailability
from src.data_aggregate.utils.institutionals.split_basis import split_adjust_frame
from src.data_store.schema import Tables
from src.utils.string import pad_cik


def _absent(df: pd.DataFrame | None, need: set[str] | None = None) -> bool:
    """True when `df` is missing, empty, or lacks a column of `need` (only unconditionally read columns)."""
    if df is None or df.empty:
        return True
    return bool(need) and not need.issubset(df.columns)


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


def _min_periods(window: int) -> int:
    """Half the window, floored at 3: enough of the window present to be a window."""
    return max(3, window // 2)


def _pivot(hist: pd.DataFrame, value: str, idx: pd.DatetimeIndex) -> pd.DataFrame:
    wide = hist.pivot_table(index="date", columns="ticker", values=value, aggfunc="sum")
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


def _ftd_estimated_available_date(period: str) -> pd.Timestamp:
    """Historical ZIP estimate: period end plus 15 calendar days, past weekends."""
    available = _ftd_period_end(period) + pd.Timedelta(days=FTD_HISTORICAL_LAG_DAYS)
    while available.weekday() >= 5:
        available += pd.Timedelta(days=1)
    return available


def _ftd_available_dates(
    fails_hist: pd.DataFrame,
    cache_dir: Path | None,
    *,
    stored_periods: Sequence[str] | None = None,
    today: pd.Timestamp | None = None,
) -> dict[str, pd.Timestamp]:
    """Derive one publication day per stored ZIP, using only a fresh latest cache mtime."""
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
    period_ends = {period: _ftd_period_end(period) for period in source_periods["period"].unique()}
    periods = sorted(period_ends)
    available = {period: _ftd_estimated_available_date(period) for period in periods}
    if cache_dir is None or not periods:
        return available

    ny_today = pd.Timestamp.now(tz=MARKET_TIMEZONE).date() if today is None else pd.Timestamp(today).date()
    candidate_ends = period_ends if stored_periods is None else {period: _ftd_period_end(period) for period in stored_periods}
    eligible = [period for period, end in candidate_ends.items() if 0 <= (ny_today - end.date()).days <= FTD_LATEST_PERIOD_MAX_AGE_DAYS]
    if not eligible:
        return available
    latest = max(eligible)
    if latest not in available:
        return available
    cached = cache_dir / FTD_ZIP_NAME_TEMPLATE.format(period=latest)
    if cached.is_file():
        cache_date = datetime.fromtimestamp(cached.stat().st_mtime, MARKET_TIMEZONE).date()
        if 0 <= (ny_today - cache_date).days < FTD_RECENT_CACHE_DAYS:
            available[latest] = pd.Timestamp(cache_date)
    return available


def _publish_ftd_zip_states(
    settlement_state: pd.DataFrame,
    fails_hist: pd.DataFrame,
    idx: pd.DatetimeIndex,
    available_by_period: dict[str, pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """Expose each ZIP's latest cumulative FTD state from its first tradable session on or after its
    publication day until the next ZIP; raises when a ZIP would publish before its own settlement dates.
    """
    if available_by_period is None:
        available_by_period = _ftd_available_dates(fails_hist, None)
    periods: dict[str, list[pd.Timestamp]] = {}
    source_periods = fails_hist[["date", "period"]].drop_duplicates().copy()
    source_periods["date"] = to_day(source_periods["date"])
    for day, period in source_periods.itertuples(index=False, name=None):
        periods.setdefault(str(period), []).append(day)

    events: list[tuple[int, str, pd.Series]] = []
    for period, days in periods.items():
        observed_days = pd.DatetimeIndex(days).intersection(settlement_state.index).sort_values()
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


def _current_ticker_by_cik(ticker_ciks: pd.DataFrame) -> dict[str, str]:
    grouped: dict[str, set[str]] = {}
    for row in ticker_ciks.itertuples(index=False):
        grouped.setdefault(pad_cik(row.cik), set()).add(str(row.ticker))
    return {cik: next(iter(tickers)) for cik, tickers in grouped.items() if cik and len(tickers) == 1}


def _lineage_target(symbol: object, issuer_cik: object, exact: dict[str, str], unique: dict[str, str]) -> str | None:
    source_symbol, source_cik = str(symbol), pad_cik(issuer_cik)
    if exact.get(source_symbol) == source_cik:
        return source_symbol
    return unique.get(source_cik)


def _proven_tenure_mask(
    idx: pd.DatetimeIndex,
    columns: pd.Index,
    symbol_tenure: pd.DataFrame | None,
    ticker_ciks: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Current-issuer symbol tenure, when both lineage inputs are available."""
    if symbol_tenure is None or ticker_ciks is None or symbol_tenure.empty or ticker_ciks.empty:
        return None
    if not {"symbol", "issuer_cik", "valid_from", "valid_to"}.issubset(symbol_tenure) or not {"ticker", "cik"}.issubset(ticker_ciks):
        return None
    current = _current_ticker_by_cik(ticker_ciks)
    exact = {str(row.ticker): pad_cik(row.cik) for row in ticker_ciks.itertuples(index=False)}
    mask = pd.DataFrame(False, index=idx, columns=columns)
    for row in symbol_tenure.itertuples(index=False):
        target = _lineage_target(row.symbol, row.issuer_cik, exact, current)
        if target not in mask.columns:
            continue
        start = pd.to_datetime(row.valid_from, errors="coerce")
        end = pd.to_datetime(row.valid_to, errors="coerce")
        if pd.isna(start):
            continue
        valid = idx >= pd.Timestamp(start).normalize()
        if pd.notna(end):
            valid &= idx < pd.Timestamp(end).normalize()
        mask.loc[valid, target] = True
    return mask


def _shortvol_fields(
    hist: pd.DataFrame,
    idx: pd.DatetimeIndex,
    shares_out: pd.DataFrame | None,
    close_total: pd.DataFrame | None,
    volume: pd.DataFrame | None,
    splits: pd.DataFrame | None = None,
    tenure_mask: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """RegSHO ratio, acceleration, turnover, price-interaction and coverage legs, each shifted by the publication lag."""
    short = _pivot(hist, "short_volume", idx)
    total = _pivot(hist, "total_volume", idx)
    if tenure_mask is not None:
        source_tenure = tenure_mask.reindex(index=idx, columns=short.columns, fill_value=False)
        short = short.where(source_tenure)
        total = total.where(source_tenure)
    observed = short.notna() & total.notna()
    f_dict: dict[str, pd.DataFrame] = {}

    ratios: dict[int, pd.DataFrame] = {}
    for w in RATIO_WINDOWS:
        mp = _min_periods(w)
        num = short.rolling(w, min_periods=mp).sum()
        den = total.rolling(w, min_periods=mp).sum()
        ratio = (num / den.where(den > 0)).replace([np.inf, -np.inf], np.nan).where(observed)
        ratios[w] = ratio.shift(SHORTVOL_PUB_LAG)
        f_dict[f"ic_shortvol_ratio_{w}d"] = ratios[w]

    base = ratios[BASE_WINDOW]
    z = self_history_z(base, window=Z_WINDOW, min_periods=Z_MIN_PERIODS)
    f_dict["ic_shortvol_acceleration"] = ratios[BASE_WINDOW] - ratios[max(RATIO_WINDOWS)]

    if shares_out is not None and not shares_out.empty:
        so = shares_out.reindex(index=idx).reindex(columns=short.columns)
        turn = short.rolling(BASE_WINDOW, min_periods=_min_periods(BASE_WINDOW)).sum()
        f_dict["ic_shortvol_turnover_20d"] = (turn / so.where(so > 0)).replace([np.inf, -np.inf], np.nan).where(observed).shift(SHORTVOL_PUB_LAG)

    if close_total is not None and not close_total.empty:
        # A return, so `close_total`; `close_split` is reserved for levels.
        ret = close_total.reindex(index=idx).reindex(columns=short.columns).pct_change(RET_WINDOW)
        high = z.clip(lower=0.0)
        f_dict["ic_shortvol_high_x_weak_price"] = high * (-ret).clip(lower=0.0)
        f_dict["ic_shortvol_high_x_strong_price"] = high * ret.clip(lower=0.0)

    if volume is not None and not volume.empty:
        tape = volume.reindex(index=idx).reindex(columns=short.columns)
        num = (total * split_adjust_frame(splits, total)).rolling(BASE_WINDOW, min_periods=_min_periods(BASE_WINDOW)).sum()
        den = tape.rolling(BASE_WINDOW, min_periods=_min_periods(BASE_WINDOW)).sum()
        cov = (num / den.where(den > 0)).replace([np.inf, -np.inf], np.nan).where(observed)
        cov = _guard_coverage(cov)
        f_dict["ic_shortvol_market_coverage"] = cov.shift(SHORTVOL_PUB_LAG)
        live = cov.to_numpy(dtype="float64", na_value=np.nan).ravel()
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
    if tenure_mask is not None:
        for name, frame in f_dict.items():
            f_dict[name] = frame.where(tenure_mask.reindex(index=frame.index, columns=frame.columns, fill_value=False))
    return f_dict


def _fails_fields(
    fails_hist: pd.DataFrame,
    idx: pd.DatetimeIndex,
    shares_out: pd.DataFrame | None,
    volume: pd.DataFrame | None,
    splits: pd.DataFrame | None = None,
    tenure_mask: pd.DataFrame | None = None,
    ftd_cache_dir: Path | None = None,
    ftd_stored_periods: Sequence[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """FTD-to-ADV20 and persistence legs, published per ZIP; zero-filled only on dates the FTD file covers."""
    available_by_period = _ftd_available_dates(fails_hist, ftd_cache_dir, stored_periods=ftd_stored_periods)
    fails = _pivot(fails_hist, "fails_quantity", idx)
    covered = pd.DatetimeIndex(to_day(fails_hist["date"]).dropna().unique())
    on_file = pd.Series(idx.isin(covered), index=idx)
    logger.info(
        "FTD file covers %s of %s trading days in the window (%.1f%%); an absent ticker on a covered date is 0 fails, an absent date is NaN",
        int(on_file.sum()),
        len(idx),
        100 * float(on_file.mean()),
    )
    # 0 on a covered date, NaN on an unpublished one; the per-date flag is broadcast to the ticker axis explicitly.
    covered_wide = pd.DataFrame({c: on_file for c in fails.columns}, index=idx)
    fails = fails.mask(covered_wide & fails.isna(), 0.0)
    if tenure_mask is not None:
        source_tenure = tenure_mask.reindex(index=idx, columns=fails.columns, fill_value=False)
        fails = fails.where(source_tenure)
        covered_wide &= source_tenure

    f_dict: dict[str, pd.DataFrame] = {}
    pct_so = None
    to_adv = None
    if shares_out is not None and not shares_out.empty:
        so = shares_out.reindex(index=idx).reindex(columns=fails.columns)
        pct_so = (fails / so.where(so > 0)).replace([np.inf, -np.inf], np.nan)
    if volume is not None and not volume.empty:
        adv = volume.reindex(index=idx).reindex(columns=fails.columns).rolling(BASE_WINDOW, min_periods=_min_periods(BASE_WINDOW)).mean()
        # Restate as-traded fails onto the split-adjusted basis of yfinance volume.
        fails_adj = fails * split_adjust_frame(splits, fails)
        to_adv = (fails_adj / adv.where(adv > 0)).replace([np.inf, -np.inf], np.nan)
        f_dict["ic_ftd_to_adv20"] = _publish_ftd_zip_states(to_adv, fails_hist, idx, available_by_period)
    # Prefer the shares-outstanding basis; fall back to ADV when fundamentals are absent.
    basis = pct_so if pct_so is not None else to_adv
    if basis is not None:
        z = self_history_z(basis, window=Z_WINDOW, min_periods=Z_MIN_PERIODS)
        # An undefined z is 0 when every covered observation in the window is present and exactly zero.
        covered_basis = pd.DataFrame({c: on_file for c in basis.columns}, index=idx)
        # Applicability starts at the ticker's first computable basis; a later hole blocks neutralization.
        applicable = covered_basis & basis.notna().cummax()
        expected = applicable.astype("float64").rolling(Z_WINDOW, min_periods=1).sum()
        observed = basis.notna().astype("float64").rolling(Z_WINDOW, min_periods=1).sum()
        zeros = basis.eq(0.0).astype("float64").rolling(Z_WINDOW, min_periods=1).sum()
        neutral = applicable & basis.eq(0.0) & expected.ge(Z_MIN_PERIODS) & observed.eq(expected) & zeros.eq(expected)
        z = z.mask(z.isna() & neutral, 0.0)
        flag = (z > Z_HIGH).astype("float64").where(z.notna())
        persistence = flag.rolling(PERSISTENCE_WINDOW, min_periods=_min_periods(PERSISTENCE_WINDOW)).sum().where(covered_basis)
        f_dict["ic_ftd_persistence_30d"] = _publish_ftd_zip_states(persistence, fails_hist, idx, available_by_period)
    if tenure_mask is not None:
        for name, frame in f_dict.items():
            f_dict[name] = frame.where(tenure_mask.reindex(index=frame.index, columns=frame.columns, fill_value=False))
    return f_dict


def build_short_flow_feature_panel(
    frames: PriceFrames,
    short_history: pd.DataFrame | None,
    *,
    fails_history: pd.DataFrame | None = None,
    ftd_cache_dir: Path | None = None,
    ftd_stored_periods: Sequence[str] | None = None,
    shares_out_history: pd.DataFrame | None = None,
    splits: pd.DataFrame | None = None,
    symbol_tenure: pd.DataFrame | None = None,
    ticker_ciks: pd.DataFrame | None = None,
    availability: InstitutionalAvailability | None = None,
    sink=None,
) -> pd.DataFrame:
    """Long-format short-flow panel (`f_<name>` per `EMISSION`); empty if neither source is available.

    `frames.volume` backs ADV20 and coverage, `shares_out_history` (`sharesOutstandingPit`) the
    share-count-scaled legs, `frames.close_total` the price interactions; each is optional and its
    absence removes only the legs that need it (hence no `frames.require`).
    """
    peer_dict = frames.peers
    trading_index = frames.trading_index
    volume = frames.volume
    close_total = frames.close_total
    # Both sources are optional; the panel is empty only when neither arrived.
    if _absent(short_history) and _absent(fails_history):
        return _empty_panel()

    idx = pd.DatetimeIndex(trading_index).normalize().unique().sort_values()
    columns = pd.Index(sorted(map(str, frames.universe)), name="ticker")
    tenure_mask = _proven_tenure_mask(idx, columns, symbol_tenure, ticker_ciks)
    shares_out = None
    if shares_out_history is not None and not shares_out_history.empty:
        # Point-in-time share count, matching the as-traded basis of fails and short sales.
        shares_out = fundamentals_to_daily(shares_out_history, "sharesOutstandingPit", idx)
        if shares_out.empty or not shares_out.notna().any().any():
            shares_out = None

    fields: dict[str, pd.DataFrame] = {}
    if short_history is not None and not short_history.empty and {"short_volume", "total_volume"}.issubset(short_history.columns):
        fields.update(_shortvol_fields(short_history, idx, shares_out, close_total, volume, splits, tenure_mask))
    if fails_history is not None and not fails_history.empty and "fails_quantity" in fails_history.columns:
        fields.update(_fails_fields(fails_history, idx, shares_out, volume, splits, tenure_mask, ftd_cache_dir, ftd_stored_periods))

    for name in list(fields):
        frame = fields[name]
        if frame is None or frame.empty or not frame.notna().any().any():
            fields.pop(name)
    if not fields:
        return pd.DataFrame(columns=["date", "ticker"])
    if sink is not None:
        # Only the confirmed-short-flow leg feeds the cross-source bearish family count (no actor here).
        name = "ic_shortvol_high_x_weak_price"
        signal_fields = dict(fields)
        signal_masks: dict[str, pd.DataFrame] = {}
        if name in fields:
            columns = pd.Index(sorted(map(str, frames.universe)), name="ticker")
            raw = fields[name].reindex(index=idx, columns=columns)
            if close_total is not None and not close_total.empty:
                listed = close_total.reindex(index=idx, columns=columns).notna()
            else:
                listed = pd.DataFrame(True, index=idx, columns=columns)
            mask = (
                availability.derived_mask(
                    name,
                    idx,
                    columns,
                    dependencies=((Tables.short_interest, None),),
                    requirements=(listed, raw.notna()),
                )
                if availability is not None
                else raw.notna()
            )
            signal_masks[name] = mask
            signal_fields[name] = raw.where(mask)
        sink.keep_signals(signal_fields, signal_masks)
    emission = {name: EMISSION[name] for name in fields}
    return build_peer_relative_panel(fields, peer_dict, emission=emission, availability=frames.availability)
