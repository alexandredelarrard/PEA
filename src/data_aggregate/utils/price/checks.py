"""Price output diagnostics and independent consumer seam masks."""

from __future__ import annotations

import logging
from typing import cast

import numpy as np
import pandas as pd

from src.data_aggregate.utils.price.clean import NULL_RET_STEP_TOL
from src.data_aggregate.utils.price.frames import ALL_FIELDS
from src.data_store.schema import Tables
from src.data_store.store import DataStore
from src.utils.price.adjustments import SPLIT_MATCH_DAYS, ShareBasis
from src.utils.price.repairs import BUGFIX_WEDGE_TOL, LEVEL_SNAP_TOL, _total_seam_step, observed_wedge

logger = logging.getLogger(__name__)


def describe(factor: pd.DataFrame, top: int = 10) -> str:
    """One log line's worth of "what did S actually do", for `StepCubePrices`.

    Ranked by `|log S|` so a 0.5 and a 2.0 read as equally large, which they are. Call it on
    the MASKED frame -- on the raw one the ranking is dominated by cells from before a ticker
    had a single bar, which are never stored and never read."""
    off = factor.ne(1.0) & factor.notna()
    rows, tickers = int(off.to_numpy().sum()), int(off.any().sum())
    cells = int(factor.notna().to_numpy().sum())
    if not rows or not cells:
        return "level_factor: 1.0 everywhere -- no spinoff factor to undo"
    extreme = factor.where(off).stack(future_stack=True).dropna()
    ranked = extreme.groupby(level=1).max().pipe(lambda s: s.reindex(s.apply(np.log).abs().sort_values(ascending=False).index)).head(top)
    return (
        f"level_factor: {rows:,} of {cells:,} cells != 1.0 "
        f"({rows / cells:.2%}) across {tickers} of {factor.shape[1]} tickers; "
        f"largest |log S|: " + ", ".join(f"{t} x{v:.4f}" for t, v in ranked.items())
    )


def measure_seams(close_total: pd.DataFrame, bugfix: dict) -> dict[str, list[pd.Timestamp]]:
    """Verify repaired totals, then measure unresolved seams on close_total.

    Consumers refuse an applicable unrebuilt total-only repair. Remaining null_ret
    seams are measured independently of the already-nulled daily return.
    """
    for ticker, entries in bugfix.get("return_seams", {}).items():
        for entry in entries:
            if "fields" not in entry:
                continue
            observed = _total_seam_step(close_total, ticker, pd.Timestamp(entry["date"]))
            if observed is not None and abs(observed / entry["target_step"] - 1.0) > LEVEL_SNAP_TOL:
                raise ValueError(
                    f"{ticker}: close_total repair is not verified in the persisted input; rebuild historical prices before consuming returns"
                )
    seams: dict[str, list[pd.Timestamp]] = {}
    for ticker, entries in (bugfix.get("null_ret") or {}).items():
        for entry in entries:
            when, expected = pd.Timestamp(entry["date"]), float(entry["expect_ret"])
            if ticker not in close_total.columns or when not in close_total.index:
                logger.info("seam mask: %s %s SKIPPED -- outside this build's window", ticker, when.date())
                continue
            series = close_total[ticker]
            position = int(series.index.get_loc(when))
            if position == 0:
                logger.info("seam mask: %s %s SKIPPED -- first bar, no step to measure", ticker, when.date())
                continue
            prior, here = float(series.iloc[position - 1]), float(series.iloc[position])
            if not (np.isfinite(prior) and np.isfinite(here)) or prior <= 0:
                logger.info("seam mask: %s %s SKIPPED -- no usable bar pair", ticker, when.date())
                continue
            observed = here / prior - 1.0
            # the one-bar STEP, the identical quantity `apply_null_ret` compares
            if abs((1.0 + observed) / (1.0 + expected) - 1.0) > NULL_RET_STEP_TOL:
                logger.info(
                    "seam mask: %s %s SKIPPED -- the defect is GONE or CHANGED: observed ret %+.5f, register says %+.5f. Re-measure the entry.",
                    ticker,
                    when.date(),
                    observed,
                    expected,
                )
                continue
            seams.setdefault(ticker, []).append(when)
            logger.info(
                "seam mask: %s %s MEASURED -- a %+.2f%% one-bar basis change in close_total; every straddling lookback window will be masked",
                ticker,
                when.date(),
                observed * 100,
            )
    return seams


def measure_quote_seams(close_split: pd.DataFrame, bugfix: dict) -> dict[str, list[pd.Timestamp]]:
    """Independently retain quote discontinuities after a close_total-only correction."""
    entries = {
        ticker: [
            {"date": entry["date"], "expect_ret": float(entry["expect_quote_step"]) - 1.0}
            for entry in values
            if "fields" in entry and "expect_quote_step" in entry
        ]
        for ticker, values in bugfix.get("return_seams", {}).items()
    }
    return measure_seams(close_split, {"null_ret": entries})


def mask_seam_windows(frame: pd.DataFrame, seams: dict[str, list[pd.Timestamp]], back: int, skip: int = 0) -> pd.DataFrame:
    """NaN the cells of `frame` whose lookback window STRADDLES a measured seam. Returns a copy.

    A feature at `t` reads its input over index positions `[t-back, t-skip]`, so it straddles a
    seam at position `s` exactly when `t-back < s <= t-skip` -- i.e. for `t` in
    `[s+skip, s+back-1]`, both ends inclusive. Both ends are load-bearing. Verified on the live
    panel at `DHR`/2016-07-05: `peer_mom_63` (`back=63, skip=0`) is pinned at rank 0.9978
    through `s+62` and reads 0.1891 at `s+63`; `mom_12_1` (`back=252, skip=21`) plateaus at
    0.9934 from `s+21`, holds to 0.9457 at `s+251` and craters to 0.2928 at `s+252`.

    Positional on `frame`'s OWN index, not on calendar arithmetic, so an incrementally trimmed
    window masks the same trading days as a full build. A seam outside the frame is a no-op, and
    a mask running off the end of the frame is clipped by the slice.
    """
    out = frame.copy()
    for ticker, dates in (seams or {}).items():
        if ticker not in out.columns:
            continue
        column = int(cast(int, out.columns.get_loc(ticker)))
        for when in dates:
            if when not in out.index:
                continue
            position = int(cast(int, out.index.get_loc(when)))
            start = position + skip
            stop = position + back  # exclusive: s+back-1 inclusive
            if stop <= start:
                continue
            out.iloc[max(start, 0) : stop, column] = np.nan
    return out


def verified_level_support(df_level: pd.DataFrame, df_vendor: pd.DataFrame | None, bugfix: dict) -> pd.DataFrame:
    """Verify registered hidden levels without using source prices to fit a new factor."""
    df_supported = pd.DataFrame(True, index=df_level.index, columns=df_level.columns)
    if df_vendor is not None and not {"ticker", "date", "price"}.issubset(df_vendor):
        df_vendor = None
    for ticker, policy in bugfix.get("level_factor", {}).items():
        if ticker not in df_level:
            continue
        lower = pd.Timestamp.min
        for segment in sorted(policy.get("segments", []), key=lambda item: item["before"]):
            upper = pd.Timestamp(segment["before"])
            observed = observed_wedge(ticker, upper, df_vendor, df_level, since=lower)
            window = (df_level.index >= lower) & (df_level.index < upper)
            lower = upper
            if observed is None or abs(observed - 1) > BUGFIX_WEDGE_TOL:
                df_supported.loc[window, ticker] = False
                logger.warning(
                    "PIT quote support: %s before %s remains unknown; compatible vendor/price level ratio=%s", ticker, upper.date(), observed
                )
    return df_supported


def _price_keys(df_rows: pd.DataFrame) -> pd.MultiIndex:
    """Validate ordered daily security keys before matching or publishing rows."""
    keys = list(Tables.cube_part_prices.pk)
    df_keys = df_rows[keys].copy()
    df_keys["date"] = pd.to_datetime(df_keys["date"], errors="coerce")
    if df_keys.isna().any().any():
        raise ValueError("Invalid price keys")
    index = pd.MultiIndex.from_frame(df_keys)
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("Invalid duplicate or unordered price keys")
    return index


def verified_price_factor_support(df_close: pd.DataFrame, basis: ShareBasis) -> pd.DataFrame:
    """Keep ambiguous pre-restatement quotes unknown when Yahoo has no evidenced factor."""
    df_supported = pd.DataFrame(True, index=df_close.index, columns=df_close.columns)
    price_dates = {ticker: pd.to_datetime(df_group["date"]) for ticker, df_group in basis.price.groupby("ticker", sort=False)}
    df_events = basis.vendor.loc[basis.vendor["value"].ne(1), ["ticker", "date"]].drop_duplicates()
    for ticker, date in df_events.itertuples(index=False, name=None):
        if ticker not in df_close:
            continue
        when = pd.Timestamp(date)
        declared = [event.get("vendor", event["actual"]) for event in basis.policies.get(ticker, {}).get("events", [])]
        if any(leg is not None and pd.Timestamp(leg["date"]) == when for leg in declared):
            continue
        dates = price_dates.get(ticker)
        if dates is not None and ((dates - when).abs() <= pd.Timedelta(days=SPLIT_MATCH_DAYS)).any():
            continue
        prefix = df_close.index < when
        if df_close.loc[prefix, ticker].notna().any():
            df_supported.loc[prefix, ticker] = False
            logger.warning("PIT quote support: %s before %s unknown; vendor restatement has no evidenced Yahoo price factor", ticker, when.date())
    return df_supported


def check_candidate(df_rows: pd.DataFrame) -> None:
    """Check the complete candidate before any price-part mutation."""
    if df_rows.empty or set(df_rows) != {*Tables.cube_part_prices.pk, *ALL_FIELDS}:
        raise ValueError("Invalid empty or incomplete price candidate")
    _price_keys(df_rows)
    for field in ALL_FIELDS:
        values = df_rows[field]
        if np.isinf(values.to_numpy(dtype="float64")).any():
            raise ValueError(f"Invalid infinite price field: {field}")
        positive = field.startswith("close_") or field in {"open", "high", "low", "level_factor", "shares_outstanding_pit", "mcap_pit"}
        positive |= field == "security_basis"
        if (positive and values.le(0).any()) or (field in {"volume", "volume_pit"} and values.lt(0).any()):
            raise ValueError(f"Price field must be positive: {field}")
    if (df_rows["mcap_pit"].notna() & df_rows[["close_pit", "shares_outstanding_pit"]].isna().any(axis=1)).any():
        raise ValueError("Invalid market cap without a supported quote/count pair")
    for field in ("close_pit", "shares_outstanding_pit", "mcap_pit", "security_basis", "volume_pit"):
        logger.info("Price candidate %s coverage: %d / %d observed security sessions", field, int(df_rows[field].notna().sum()), len(df_rows))


def earliest_price_change(store: DataStore, df_rows: pd.DataFrame, existing_columns: list[str]) -> pd.Timestamp | None:
    """Stream the saved part once and find the first key, null or exact value difference."""
    keys = list(Tables.cube_part_prices.pk)
    stream = store.iter_load(Tables.cube_part_prices, columns=existing_columns, order_by=keys)
    try:
        if set(existing_columns) != set(df_rows):
            raise ValueError("Historical price comparison requires identical schemas")
        index = _price_keys(df_rows)
        fields = [field for field in df_rows if field not in keys]
        values = df_rows[fields].to_numpy(dtype="float64")
        seen = np.zeros(len(df_rows), dtype=bool)
        earliest: pd.Timestamp | None = None
        last_key: tuple | None = None
        for df_chunk in stream:
            stored_index = _price_keys(df_chunk)
            if df_chunk.empty:
                continue
            if last_key is not None and stored_index[0] <= last_key:
                raise ValueError("Invalid repeated or unordered saved price keys across chunks")
            last_key = stored_index[-1]
            dates = pd.DatetimeIndex(stored_index.get_level_values("date"))
            positions = index.get_indexer(stored_index)
            matched = positions >= 0
            if seen[positions[matched]].any():
                raise ValueError("Invalid repeated saved price keys")
            seen[positions[matched]] = True
            changed = ~matched
            old = df_chunk.loc[matched, fields].to_numpy(dtype="float64")
            new = values[positions[matched]]
            changed[matched] = ~((old == new) | (np.isnan(old) & np.isnan(new))).all(axis=1)
            if changed.any():
                first = dates[changed].min()
                earliest = first if earliest is None else min(earliest, first)
        if (~seen).any():
            first = pd.Timestamp(df_rows.loc[~seen, "date"].min())
            earliest = first if earliest is None else min(earliest, first)
        return earliest
    finally:
        stream.close()
