"""Price levels, legacy correction guards and independent return/quote seam masks.

S=Y*H/(F*U) reconciles adjusted prices with vendor counts. It never enters returns.
Shared pure price repairs and event resolution retain compatibility imports here.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import numpy as np
import pandas as pd

from src.utils.predecessor_series import load_level_actions as level_actions
from src.utils.price_adjustments import (
    BUGFIX_FILENAME as BUGFIX_FILENAME,
)
from src.utils.price_adjustments import (
    ShareBasis,
)
from src.utils.price_adjustments import (
    load_bugfix as load_bugfix,
)
from src.utils.price_adjustments import (
    split_events as genuine_splits,
)
from src.utils.price_adjustments import (
    suffix_factor as suffix_factor,
)
from src.utils.price_basis import load_share_basis as load_share_basis
from src.utils.price_repairs import (
    BUGFIX_STEP_TOL,
    BUGFIX_WEDGE_TOL,
    LEVEL_SNAP_TOL,
    REPAIRED_PRICE_FIELDS,
    VINTAGE_FLIP_TOL,
    _apply_total_seam,
    _rescale,
    _total_seam_step,
    _vintage_multiplier,
    apply_return_seams,
    apply_split_vintage,
    observed_wedge,
)

logger = logging.getLogger(__name__)

__all__ = [
    "genuine_splits",
    "level_actions",
    "BUGFIX_STEP_TOL",
    "BUGFIX_WEDGE_TOL",
    "LEVEL_SNAP_TOL",
    "REPAIRED_PRICE_FIELDS",
    "VINTAGE_FLIP_TOL",
    "_apply_total_seam",
    "_rescale",
    "_total_seam_step",
    "_vintage_multiplier",
    "apply_return_seams",
    "apply_split_vintage",
    "observed_wedge",
]


def level_factor(
    index: pd.DatetimeIndex,
    universe: Sequence[str],
    yf_splits: pd.DataFrame | None,
    genuine_splits: pd.DataFrame | ShareBasis,
) -> pd.DataFrame:
    """Return S=Y/(F*U), before hidden wedges, aligned to the requested date/ticker grid.

    A ShareBasis supplies separately verified price, vendor-count and quoted-unit views.
    The legacy DataFrame argument preserves the default event-product behavior.
    """
    if isinstance(genuine_splits, ShareBasis):
        basis = genuine_splits
        events = basis.vendor
        yf_splits = basis.price
    else:
        basis = None
        events = genuine_splits
    columns = sorted(set(map(str, universe)))
    idx = pd.DatetimeIndex(index)
    frame = pd.DataFrame(1.0, index=idx, columns=columns, dtype="float64")
    frame.index.name, frame.columns.name = "date", "ticker"

    stamps = idx.to_numpy(dtype="datetime64[ns]")
    numerator = suffix_factor(
        (yf_splits.rename(columns={"ratio": "value"}) if yf_splits is not None and "ratio" in yf_splits.columns else yf_splits), columns, stamps
    )
    denominator = suffix_factor(events, columns, stamps)

    for ticker in set(numerator) | set(denominator):
        up = numerator.get(ticker)
        down = denominator.get(ticker)
        if up is None:
            assert down is not None
            frame[ticker] = 1.0 / down
        elif down is None:
            frame[ticker] = up
        else:
            frame[ticker] = up / down

    if basis is not None:
        for ticker in basis.policies.keys() & set(columns):
            units = basis.factor(pd.Series(ticker, index=idx), pd.Series(idx, index=idx), "units")
            frame[ticker] = frame[ticker] / units

    # `copy=True`: pandas hands back a READ-ONLY view of a single-block frame, so writing the
    # snap into it raises rather than silently doing nothing.
    values = frame.to_numpy(dtype="float64", copy=True)
    values[np.abs(values - 1.0) < LEVEL_SNAP_TOL] = 1.0
    return pd.DataFrame(values, index=frame.index, columns=frame.columns)


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


NULL_RET_STEP_TOL = BUGFIX_STEP_TOL


VOLUME_JUMP_TOL = 3.0
VOLUME_JUMP_BARS = 20


def _volume_jump(volume: pd.Series, when: pd.Timestamp) -> float | None:
    """Median volume of the `VOLUME_JUMP_BARS` bars before `when` over that of the bars at or after
    it; None when either side is missing or the later median is not positive."""
    before = volume[volume.index < when].tail(VOLUME_JUMP_BARS).median()
    after = volume[volume.index >= when].head(VOLUME_JUMP_BARS).median()
    if pd.isna(before) or pd.isna(after) or after <= 0:
        return None
    return float(before / after)


def apply_volume_scale(bars: pd.DataFrame, bugfix: Mapping[str, Any]) -> pd.DataFrame:
    """Repair a vendor volume UNIT defect: the `volume` of a registered ticker's bars dated before an
    entry's `before` is multiplied by its `factor`. `bars` is long (`ticker`, `date`, `volume`); the
    input is returned untouched when no entry applies, else a copy.

    Re-verified before it fires: the observed jump across `before` (`_volume_jump`) must sit within
    `VOLUME_JUMP_TOL` of `expect_jump`, else the entry is skipped and logged. A frame with no bar
    before `before` has nothing to repair."""
    entries = bugfix.get("volume_scale") or {}
    if not entries or bars.empty:
        return bars
    out = bars
    dates = pd.to_datetime(bars["date"])
    for ticker, items in entries.items():
        mine = bars["ticker"].astype(str).eq(ticker).to_numpy()
        if not mine.any():
            continue
        for entry in items:
            when = pd.Timestamp(entry["before"])
            target = mine & (dates < when).to_numpy()
            if not target.any():
                continue
            volume = pd.Series(pd.to_numeric(out.loc[mine, "volume"], errors="coerce").to_numpy(), index=dates[mine]).sort_index()
            jump, expect = _volume_jump(volume, when), float(entry["expect_jump"])
            if jump is None or not expect / VOLUME_JUMP_TOL <= jump <= expect * VOLUME_JUMP_TOL:
                logger.info("price bugfix: volume_scale %s before %s SKIPPED -- observed jump %s, expected %.1f", ticker, when.date(), jump, expect)
                continue
            if out is bars:
                out = bars.copy()
                out["volume"] = pd.to_numeric(out["volume"], errors="coerce").astype("float64")
            out.loc[target, "volume"] = out.loc[target, "volume"] * float(entry["factor"])
            logger.info(
                "price bugfix: volume_scale %s x%s on %d bar(s) before %s (observed jump %.1f)",
                ticker,
                entry["factor"],
                int(target.sum()),
                when.date(),
                jump,
            )
    return out


def apply_null_ret(ret: pd.DataFrame, bugfix: dict, log: Callable[..., None]) -> int:
    """Null an evidenced but unresolved one-bar return, preserving every price leg.

    The five retained events have no supported economic target. A changed expectation
    is skipped with a diagnostic; rolling consumers independently mask their windows.
    """
    applied = 0
    for ticker, entries in (bugfix.get("null_ret") or {}).items():
        for entry in entries:
            when, expected = pd.Timestamp(entry["date"]), float(entry["expect_ret"])
            if ticker not in ret.columns or when not in ret.index:
                log("price bugfix: %s %s null_ret SKIPPED -- outside this build's window", ticker, when.date())
                continue
            observed = float(cast(float, ret.at[when, ticker]))
            if not np.isfinite(observed):
                log("price bugfix: %s %s null_ret SKIPPED -- already not a number", ticker, when.date())
                continue
            # compared as a one-bar STEP, the same quantity `apply_return_seams` compares
            if abs((1.0 + observed) / (1.0 + expected) - 1.0) > NULL_RET_STEP_TOL:
                log(
                    "price bugfix: %s %s null_ret SKIPPED -- the defect is GONE or CHANGED: "
                    "observed ret %+.5f, register says %+.5f. Re-measure the entry.",
                    ticker,
                    when.date(),
                    observed,
                    expected,
                )
                continue
            ret.at[when, ticker] = np.nan
            applied += 1
            log(
                "price bugfix: %s %s null_ret APPLIED -- a %+.2f%% one-bar 'return' deleted (the price legs are left exactly as published)",
                ticker,
                when.date(),
                observed * 100,
            )
    return applied


def measure_seams(close_total: pd.DataFrame, bugfix: dict, log: Callable[..., None]) -> dict[str, list[pd.Timestamp]]:
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
                log("seam mask: %s %s SKIPPED -- outside this build's window", ticker, when.date())
                continue
            series = close_total[ticker]
            position = int(series.index.get_loc(when))
            if position == 0:
                log("seam mask: %s %s SKIPPED -- first bar, no step to measure", ticker, when.date())
                continue
            prior, here = float(series.iloc[position - 1]), float(series.iloc[position])
            if not (np.isfinite(prior) and np.isfinite(here)) or prior <= 0:
                log("seam mask: %s %s SKIPPED -- no usable bar pair", ticker, when.date())
                continue
            observed = here / prior - 1.0
            # the one-bar STEP, the identical quantity `apply_null_ret` compares
            if abs((1.0 + observed) / (1.0 + expected) - 1.0) > NULL_RET_STEP_TOL:
                log(
                    "seam mask: %s %s SKIPPED -- the defect is GONE or CHANGED: observed ret %+.5f, register says %+.5f. Re-measure the entry.",
                    ticker,
                    when.date(),
                    observed,
                    expected,
                )
                continue
            seams.setdefault(ticker, []).append(when)
            log(
                "seam mask: %s %s MEASURED -- a %+.2f%% one-bar basis change in close_total; every straddling lookback window will be masked",
                ticker,
                when.date(),
                observed * 100,
            )
    return seams


def measure_quote_seams(close_split: pd.DataFrame, bugfix: dict, log: Callable[..., None]) -> dict[str, list[pd.Timestamp]]:
    """Independently retain quote discontinuities after a close_total-only correction."""
    entries = {
        ticker: [
            {"date": entry["date"], "expect_ret": float(entry["expect_quote_step"]) - 1.0}
            for entry in values
            if "fields" in entry and "expect_quote_step" in entry
        ]
        for ticker, values in bugfix.get("return_seams", {}).items()
    }
    return measure_seams(close_split, {"null_ret": entries}, log)


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


def apply_level_bugfix(
    factor: pd.DataFrame, bugfix: dict, vendor_price: pd.DataFrame | None, close_split: pd.DataFrame, log: Callable[..., None]
) -> pd.DataFrame:
    """Fold the registered LEVEL wedges into `S`, re-measuring each one first.

    These are the cases `S` is structurally BLIND to: Yahoo adjusted the price and its splits
    feed never said so, so a factor derived from that feed reads exactly 1.0. IP is the
    clearest -- 107 of its 127 filing rows fail invariant 1 and `prices_splits` has no IP row
    at all.

    ⚠ NEVER TOUCHES A RETURN. It multiplies `S`, which reaches `cube_part_prices.level_factor`
    and every market cap built from it, and nothing else. That is the whole distinction from
    the two repairs above: a smooth back-adjustment leaves returns CORRECT and only the level
    short, so moving the returns would break what is currently right.

    THE VERIFICATION IS THE POINT. `vendor_price` (`fundamentals_sharadar.price`) is the only
    independent statement of what a stock actually traded at, so the observed wedge is
    `median(price / close_split)` over the segment's own rows. An entry whose wedge has moved
    outside `BUGFIX_WEDGE_TOL` is REFUSED rather than applied: a moved wedge means either
    Yahoo fixed it (the repair is no longer needed) or something else changed (the repair is
    no longer correct), and both call for a human rather than a silent multiply.

    ⚠ This makes invariant 2 partly SELF-REFERENTIAL for the registered tickers -- the factor
    is derived from the same vendor price the invariant scores against. The independent
    alternative is to derive the wedge from the spun-off child's own price via
    `sharadar_actions.contraticker`, which needs price history for securities outside the
    universe and is therefore its own task.
    """
    entries = bugfix.get("level_factor") or {}
    if not entries:
        return factor
    out = factor.copy()
    for ticker, spec in entries.items():
        if ticker not in out.columns:
            continue
        # ⚠ A SEGMENT IS A WINDOW, NOT A PREFIX, and `[lower, before)` is what makes both
        # halves of this loop correct. Measured on the live tables: IP reads 1.07078 before
        # its Veritiv spinoff and 1.05600 between Veritiv and Sylvamo, and HBAN steps
        # 1.61051 / 1.46410 / 1.33100 / 1.21000 / 1.10005 through its five stock dividends.
        # Every one of those is the WHOLE wedge for its own era, so segments must not
        # compound -- and an unbounded window would re-measure the era before it, which
        # skipped three of HBAN's five entries and applied IP's second on a 1.4% coincidence.
        lower = pd.Timestamp.min
        for segment in sorted(spec.get("segments", []), key=lambda s: s["before"]):
            before, claimed = pd.Timestamp(segment["before"]), float(segment["factor"])
            observed = observed_wedge(ticker, before, vendor_price, close_split, since=lower)
            window, lower = (out.index >= lower) & (out.index < before), before
            if observed is None:
                log("price bugfix: %s < %s SKIPPED -- no vendor price in the segment, so the defect cannot be re-verified", ticker, before.date())
                continue
            if abs(observed / claimed - 1.0) > BUGFIX_WEDGE_TOL:
                log(
                    "price bugfix: %s < %s SKIPPED -- the defect is GONE or CHANGED: observed wedge %.5f, register says %.5f. Re-measure the entry.",
                    ticker,
                    before.date(),
                    observed,
                    claimed,
                )
                continue
            eligible = window & close_split[ticker].reindex(out.index).notna()
            out.loc[eligible, ticker] = out.loc[eligible, ticker] * claimed
            log(
                "price bugfix: %s < %s level x%.5f APPLIED to %d bar(s) -- observed wedge %.5f (%s)",
                ticker,
                before.date(),
                claimed,
                int(eligible.sum()),
                observed,
                segment.get("event", "no event recorded"),
            )
    return out


def as_traded_close(
    close_split: pd.DataFrame | None,
    level_factor: pd.DataFrame | None,
    split_events: pd.DataFrame | ShareBasis,
) -> pd.DataFrame | None:
    """Restore quoted-security prices using vendor restatements and evidenced quote units."""
    if close_split is None or level_factor is None:
        return None
    df_close = close_split * level_factor.reindex_like(close_split)
    if isinstance(split_events, ShareBasis):
        basis = split_events
        events = basis.vendor
    else:
        basis = None
        events = split_events
    factors = suffix_factor(events, list(df_close.columns), df_close.index.to_numpy(dtype="datetime64[ns]"))
    for ticker, factor in factors.items():
        df_close[ticker] *= factor
    if basis is not None:
        for ticker in basis.policies.keys() & set(df_close.columns):
            df_close[ticker] *= basis.factor(pd.Series(ticker, index=df_close.index), pd.Series(df_close.index, index=df_close.index), "units")
    return df_close.where(np.isfinite(df_close) & df_close.gt(0))


def _period_close(close: pd.DataFrame, periods: pd.Series, max_age_days: int) -> pd.Series:
    """Last observed valuation-day quote, bounded by age and never from the future."""
    wanted = pd.DatetimeIndex(sorted(periods.dropna().unique()))
    df_quotes = close.sort_index().where(np.isfinite(close) & close.gt(0))
    union = df_quotes.index.union(wanted)
    df_asof = df_quotes.reindex(union).ffill().reindex(wanted)
    for ticker in df_quotes:
        observed = pd.Series(df_quotes.index, index=df_quotes.index).where(df_quotes[ticker].notna())
        last = observed.reindex(union).ffill().reindex(wanted)
        df_asof[ticker] = df_asof[ticker].where((pd.Series(wanted, index=wanted) - last).le(pd.Timedelta(days=max_age_days)))
    result = cast(pd.Series, df_asof.stack(future_stack=True))
    result.index = result.index.set_names(["period", "ticker"])
    return result
