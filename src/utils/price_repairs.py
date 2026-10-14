"""Pure price repair math shared by builders and independent anchor loading."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import pandas as pd

LEVEL_SNAP_TOL = 1e-12


BUGFIX_WEDGE_TOL = 0.02


BUGFIX_STEP_TOL = 0.005


VINTAGE_FLIP_TOL = 0.10


REPAIRED_PRICE_FIELDS = ("close_split", "close_total", "open", "high", "low")


def _rescale(wide: dict[str, pd.DataFrame], ticker: str, factor: pd.Series | tuple[pd.Series | np.ndarray, float]) -> None:
    """Multiply every stored price leg of one ticker by `factor`, which is either a scalar or
    a per-bar Series. Legs absent from this build's `wide` are skipped."""
    for field in REPAIRED_PRICE_FIELDS:
        target = wide.get(field)
        if target is None or ticker not in target.columns:
            continue
        if isinstance(factor, pd.Series):
            target[ticker] = target[ticker] * factor.reindex(target.index).fillna(1.0)
        else:
            mask = factor[0]
            target.loc[mask, ticker] = target.loc[mask, ticker] * factor[1]


def apply_split_vintage(wide: dict[str, pd.DataFrame], bugfix: dict, vendor_price: pd.DataFrame | None, log: Callable[..., None]) -> int:
    """Repair mixed split vintages by walking backward from the event anchor.

    Independent vendor prices guard the expected wedge; all price legs use the
    same per-bar multipliers. Volume and post-event bars stay unchanged.
    """
    applied = 0
    for ticker, entries in (bugfix.get("split_vintage") or {}).items():
        for entry in entries:
            when = pd.Timestamp(entry["before"])
            ratio, expect = float(entry["ratio"]), float(entry["expect_wedge"])
            close = wide.get("close_split")
            if close is None or ticker not in close.columns:
                continue
            series = close[ticker].dropna()
            ahead = series.index[series.index >= when]
            behind = series.index[series.index < when]
            if not len(ahead) or not len(behind):
                log(
                    "price bugfix: %s split-vintage SKIPPED -- this build's window does not straddle %s, so there is no anchor bar to walk back from",
                    ticker,
                    when.date(),
                )
                continue

            was = observed_wedge(ticker, when, vendor_price, close)
            if was is None:
                log("price bugfix: %s split-vintage SKIPPED -- no vendor price before %s, so the defect cannot be re-verified", ticker, when.date())
                continue
            if abs(was / expect - 1.0) > BUGFIX_WEDGE_TOL:
                log(
                    "price bugfix: %s split-vintage SKIPPED at %s -- the defect is GONE or CHANGED: "
                    "observed wedge %.5f, register expects %.5f. Re-measure the entry.",
                    ticker,
                    when.date(),
                    was,
                    expect,
                )
                continue

            multiplier, flips = _vintage_multiplier(series, ahead[0], ratio)
            _rescale(wide, ticker, multiplier)

            applied += 1
            now = observed_wedge(ticker, when, vendor_price, wide["close_split"])
            log(
                "price bugfix: %s split-vintage REPAIRED across %s (ratio %g) -- %d flip(s), %d of %d bar(s) rescaled; wedge %.5f -> %s",
                ticker,
                when.date(),
                ratio,
                flips,
                int((multiplier != 1.0).sum()),
                len(series),
                was,
                f"{now:.5f}" if now is not None else "unmeasurable",
            )
            if now is not None and abs(now - 1.0) > BUGFIX_WEDGE_TOL:
                log(
                    "price bugfix: %s split-vintage did NOT reach parity -- wedge %.5f after "
                    "the walk. The repair ran; treat the residual as unexplained.",
                    ticker,
                    now,
                )
    return applied


def _vintage_multiplier(series: pd.Series, anchor: pd.Timestamp, ratio: float) -> tuple[pd.Series, int]:
    """Walk backward from the verified newest basis, changing only its prefix."""
    values = series.to_numpy(dtype="float64")
    out = np.ones(values.size, dtype="float64")
    flips = 0
    for i in range(int(cast(int, series.index.get_loc(anchor))) - 1, -1, -1):
        here = values[i] * out[i + 1]
        if not np.isfinite(here) or here <= 0:
            out[i] = out[i + 1]
            continue
        step = (values[i + 1] * out[i + 1]) / here
        if abs(step * ratio - 1.0) < VINTAGE_FLIP_TOL:  # this bar reads `ratio` too HIGH
            out[i], flips = out[i + 1] / ratio, flips + 1
        elif abs(step / ratio - 1.0) < VINTAGE_FLIP_TOL:  # ... or `ratio` too LOW
            out[i], flips = out[i + 1] * ratio, flips + 1
        else:
            out[i] = out[i + 1]
    return pd.Series(out, index=series.index), flips


def _total_seam_step(frame: pd.DataFrame | None, ticker: str, when: pd.Timestamp) -> float | None:
    """Measure an applicable adjacent pair, refusing holes inside an affected prefix."""
    if frame is None or ticker not in frame.columns or frame.empty:
        return None
    series = frame[ticker]
    if when < series.index.min() or when > series.index.max() or not series.loc[series.index < when].notna().any():
        return None
    if when not in series.index:
        raise ValueError(f"{ticker} {when.date()}: total-return repair REFUSED; missing interior event bar")
    position = int(cast(int, series.index.get_loc(when)))
    prior, here = float(series.iloc[position - 1]), float(series.iloc[position])
    if not np.isfinite([prior, here]).all() or prior <= 0 or here <= 0:
        raise ValueError(f"{ticker} {when.date()}: total-return repair REFUSED; unusable adjacent pair")
    return here / prior


def _apply_total_seam(wide: dict[str, pd.DataFrame], ticker: str, entry: dict, log: Callable[..., None]) -> int:
    when = pd.Timestamp(entry["date"])
    expected, target = float(entry["expect_step"]), float(entry["target_step"])
    if entry["fields"] != ["close_total"] or "step" in entry or not np.isfinite([expected, target]).all() or min(expected, target) <= 0:
        raise ValueError(f"{ticker}: invalid total-return repair fields or steps")
    frame = wide.get("close_total")
    scope = frame if frame is not None else wide.get("close_split")
    observed = _total_seam_step(scope, ticker, when)
    if observed is None:
        log("price bugfix: %s %s total-return NOT_APPLICABLE -- no affected prefix/pair in this window", ticker, when.date())
        return 0
    if frame is None:
        raise ValueError(f"{ticker}: total-return repair REFUSED; close_total unavailable")
    if abs(observed / target - 1.0) <= LEVEL_SNAP_TOL:
        log("price bugfix: %s %s total-return ALREADY_VERIFIED at gross %.12g", ticker, when.date(), observed)
        return 0
    if abs(observed / expected - 1.0) > BUGFIX_STEP_TOL:
        raise ValueError(f"{ticker} {when.date()}: total-return repair REFUSED; observed gross {observed:.12g}, expected {expected:.12g}; remeasure")
    multiplier = observed / target
    frame.loc[frame.index < when, ticker] *= multiplier
    log("price bugfix: %s %s close_total-only APPLIED; prefix x%.12g, event gross %.12g -> %.12g", ticker, when.date(), multiplier, observed, target)
    return 1


def apply_return_seams(wide: dict[str, pd.DataFrame], bugfix: dict, log: Callable[..., None]) -> int:
    """Apply guarded total-only repairs or the unchanged legacy all-price splice.

    New entries preserve their supported nonzero gross target and refuse unknown
    input before a builder write. Legacy entries guard close_split and target one.
    """
    applied = 0
    for ticker, entries in (bugfix.get("return_seams") or {}).items():
        for entry in entries:
            if "fields" in entry:
                applied += _apply_total_seam(wide, ticker, entry, log)
                continue
            when, expected = pd.Timestamp(entry["date"]), float(entry["step"])
            frame = wide.get("close_split")
            if frame is None or ticker not in frame.columns or when not in frame.index:
                log("price bugfix: %s %s seam SKIPPED -- outside this build's window", ticker, when.date())
                continue
            series = frame[ticker]
            position = int(series.index.get_loc(when))
            if position == 0:
                log("price bugfix: %s %s seam SKIPPED -- first bar, no step to measure", ticker, when.date())
                continue
            prior = float(cast(float, series.iloc[position - 1]))
            here = float(cast(float, series.iloc[position]))
            if not (np.isfinite(prior) and np.isfinite(here)) or prior <= 0:
                log("price bugfix: %s %s seam SKIPPED -- no usable bar pair", ticker, when.date())
                continue
            observed = here / prior
            if abs(observed / expected - 1.0) > BUGFIX_STEP_TOL:
                log(
                    "price bugfix: %s %s seam SKIPPED -- the defect is GONE or CHANGED: "
                    "observed step %.6f, register says %.6f. Re-measure the entry.",
                    ticker,
                    when.date(),
                    observed,
                    expected,
                )
                continue
            _rescale(wide, ticker, (frame.index < when, observed))
            applied += 1
            log(
                "price bugfix: %s %s seam REPAIRED -- every bar before it rescaled by %.6f (it was a %+.2f%% one-bar 'return' that never happened)",
                ticker,
                when.date(),
                observed,
                (observed - 1) * 100,
            )
    return applied


def observed_wedge(
    ticker: str, before: pd.Timestamp, vendor_price: pd.DataFrame | None, close_split: pd.DataFrame, *, since: pd.Timestamp = pd.Timestamp.min
) -> float | None:
    """Median vendor-price/repaired-close ratio in [since,before).

    Filing dates use the last quote at or before publication. Nonoverlapping
    segment bounds prevent an older era dominating a later plateau.
    """
    if vendor_price is None or vendor_price.empty or ticker not in close_split.columns:
        return None
    rows = vendor_price[(vendor_price["ticker"] == ticker) & (vendor_price["date"] >= since) & (vendor_price["date"] < before)]
    rows = rows[rows["price"] > 0]
    series = close_split[ticker].dropna()
    if rows.empty or series.empty:
        return None
    positions = series.index.searchsorted(rows["date"].to_numpy(), side="right") - 1
    keep = positions >= 0
    if not keep.any():
        return None
    wedge = rows["price"].to_numpy()[keep] / series.to_numpy()[positions[keep]]
    wedge = wedge[np.isfinite(wedge) & (wedge > 0)]
    return float(np.median(wedge)) if wedge.size else None
