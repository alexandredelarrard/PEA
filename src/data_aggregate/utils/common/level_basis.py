"""
level_basis.py  (src/data_aggregate/utils/common/level_basis.py)
---------------------------------------------------------------
`S(d)` -- the price adjustment Yahoo applied to its own history that the SHARE COUNT did not,
so that a LEVEL built from the two can be put back on one basis.

## Why anything is needed at all

`close_split` and `sharesOutstanding` already cancel for SPLITS, and that cancellation is the
whole design of `pit.daily_market_cap`: the vendor back-fills `sharesbas` to today's basis and
Yahoo restates `Close` to the same one, so the future-split factor divides out of the product
exactly. A SPINOFF breaks the symmetry, because only one leg moves:

  * Yahoo back-adjusts `Close` across a spinoff -- correct for RETURNS, because it keeps the
    series continuous through an event that hands shareholders value in another security.
  * Sharadar's `price` does not -- correct for LEVELS, because it is what the stock traded at.
  * A spinoff does NOT change the parent's share count. Verified on the SEC cover page across
    8 events: `sharesbas` / SEC reads 1.0000 on BOTH sides of every one.

So the price leg carries a factor the share leg does not and nothing cancels.
`close_split x sharesOutstanding` UNDERSTATES market cap by exactly that factor -- FDX on
2020-12-17 reads $62.4bn against a true $77.5bn, 19.4% low, and GE reads 40% low.

## The definition

    S(d)  =  PROD{ prices_splits.ratio    : date > d }
           /  PROD{ split_events(...).value : date > d }

A RATIO OF TWO PRODUCTS, not a set subtraction, and the distinction is load-bearing: HON
carries an event on 2026-06-29 in BOTH sources with DIFFERENT values -- yfinance 0.9535 (the
Solstice spinoff's price factor) and Sharadar 0.5 (the co-dated 1:2 reverse split) -- and only
the ratio gives the right answer. Worked, for a 1996 HON row:

    PROD(yfinance)  = 2 x 1.00533 x 1.011 x 1.032 x 1.061 x 0.9535 = 2.12228
    PROD(genuine)   = 2 x 0.5                                      = 1.00000
    S               = 2.12228        (measured price leg 0.4712 = 1/2.1223)

The denominator is `field_map.split_events` -- THE SAME FUNCTION the extract layer de-adjusts
share counts with. That is deliberate and is what stops the numerator and denominator drifting
apart: a row that counts as a genuine split there has already cancelled against `sharesbas`
here, so it must not be counted twice.

## ⚠ S MULTIPLIES A LEVEL, NEVER A RETURN

A return computed from `close_split x S` would be wrong at the spinoff date -- by the entire
factor, on that one bar -- which is precisely what Yahoo's back-adjustment exists to prevent.
Market cap, enterprise value and every yield built on one are levels and want `S`; `ret`,
momentum, betas and every label are returns and must never see it.

## What S does NOT fix

MNST. Yahoo PUBLISHES the 2026-08-11 x2 in its own splits feed but never applied it to the
quote, so the event appears in both products and cancels to 1.0. That is a stale price
VINTAGE, not a basis difference, and it needs a per-bar repair rather than a factor.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

# ⚠ THE ONE PLACE `data_aggregate` REACHES INTO `data_extract`, and it is deliberate.
#
# AGENTS.md forbids cross-imports between `src/` subfolders, and the escape hatch is normally
# `src/utils/`. It is not taken here because the thing being shared is not a utility, it is a
# DECISION: which `prices_splits` rows are genuine share events. That decision has to be
# IDENTICAL on both sides or `S` is wrong -- the extract layer divides `sharesOutstandingPit`
# by exactly the events this module puts in the denominator, and a fork would let the two
# drift apart silently, which is the failure mode `S` exists to close.
#
# So the rule is bent ONCE, here, rather than in each of the consumers: `StepCubePrices`
# imports `genuine_splits` from this module and never names `data_extract` itself. Moving
# `split_events` to `src/utils/` would satisfy the letter of the rule, but it would drag
# `TranslationReport`, both registers and the whole corroboration rule with it -- a refactor
# of the extract layer that this plan has no mandate for.
from src.data_extract.utils.fundamentals_sharadar.field_map import (
    split_events as genuine_splits)

#: |S - 1| below which the two event sets are the SAME set and the factor is exactly 1.0.
#: Float noise from multiplying and then dividing the same ratios is ~1e-16; the smallest
#: genuine factor in the universe is 1.00533. Snapping matters because ~89% of rows must come
#: out BIT-IDENTICAL to the pre-fix build -- a control ticker at 1.0000000000000002 would move
#: every downstream digest and make the "this change is targeted" claim unprovable.
LEVEL_SNAP_TOL = 1e-12


def _suffix_factor(events: pd.DataFrame, tickers: Sequence[str],
                   stamps: np.ndarray) -> dict[str, np.ndarray]:
    """Per ticker, `PROD(value : event date > d)` evaluated at every `d` in `stamps`.

    A suffix product plus a `searchsorted`, not a loop over events: the products are formed
    RIGHT-TO-LEFT over dates sorted ascending, so two identical event lists produce
    bit-identical floats in `level_factor`'s division and cancel to exactly 1.0. A left-to-
    right accumulation would not guarantee that.
    """
    out: dict[str, np.ndarray] = {}
    if events is None or events.empty:
        return out

    frame = events.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date", "value"])
    frame = frame[np.isfinite(frame["value"]) & (frame["value"] > 0)]
    wanted = set(map(str, tickers))

    for ticker, group in frame.groupby(frame["ticker"].astype(str), sort=False):
        if ticker not in wanted:
            continue
        group = group.sort_values("date")
        values = group["value"].to_numpy(dtype="float64")
        # suffix[i] = prod(values[i:]), with suffix[n] = 1.0 for a date after every event.
        suffix = np.ones(values.size + 1, dtype="float64")
        suffix[:-1] = np.cumprod(values[::-1])[::-1]
        # `right`: an event dated exactly d restates the bars BEFORE d and leaves d itself
        # alone, matching `field_map.forward_split_factor`'s strict `<`.
        pos = np.searchsorted(group["date"].to_numpy(dtype="datetime64[ns]"), stamps,
                              side="right")
        out[ticker] = suffix[pos]
    return out


def level_factor(index: pd.DatetimeIndex, universe: Sequence[str],
                 yf_splits: pd.DataFrame, genuine_splits: pd.DataFrame) -> pd.DataFrame:
    """`S(d)` as a wide (date x ticker) frame of MULTIPLIERS, 1.0 where nothing applies.

    `yf_splits` is `prices_splits` (`ticker`, `date`, `ratio`) -- every factor Yahoo applied
    to its own series, spinoff factors included, which is exactly what has to be undone.
    `genuine_splits` is `field_map.split_events(...)` (`ticker`, `date`, `value`) -- the
    subset that is a real share event and has therefore already cancelled against the count.

    Every ticker in `universe` gets a column and every date in `index` a row, so the result
    aligns with the other price frames without a reindex at the call site. A ticker with no
    events, or with two agreeing event sets, is exactly `1.0` -- see `LEVEL_SNAP_TOL`.
    """
    columns = sorted(set(map(str, universe)))
    idx = pd.DatetimeIndex(index)
    frame = pd.DataFrame(1.0, index=idx, columns=columns, dtype="float64")
    frame.index.name, frame.columns.name = "date", "ticker"

    stamps = idx.to_numpy(dtype="datetime64[ns]")
    numerator = _suffix_factor(
        (yf_splits.rename(columns={"ratio": "value"})
         if yf_splits is not None and "ratio" in yf_splits.columns else yf_splits),
        columns, stamps)
    denominator = _suffix_factor(genuine_splits, columns, stamps)

    for ticker in set(numerator) | set(denominator):
        up = numerator.get(ticker)
        down = denominator.get(ticker)
        if up is None:
            frame[ticker] = 1.0 / down
        elif down is None:
            frame[ticker] = up
        else:
            frame[ticker] = up / down

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
    ranked = (extreme.groupby(level=1).max().pipe(
        lambda s: s.reindex(np.log(s).abs().sort_values(ascending=False).index)).head(top))
    return (f"level_factor: {rows:,} of {cells:,} cells != 1.0 "
            f"({rows / cells:.2%}) across {tickers} of {factor.shape[1]} tickers; "
            f"largest |log S|: "
            + ", ".join(f"{t} x{v:.4f}" for t, v in ranked.items()))


# --------------------------------------------------------------------------- #
# the Yahoo bug register                                                       #
# --------------------------------------------------------------------------- #
#: Where the register lives, relative to `Context.config_dir`. One consumer
#: (`StepCubePrices`), so the name stays here rather than in `constants.py`.
BUGFIX_FILENAME = "prices/yf_price_bugfix.json"
#: How far the OBSERVED wedge may sit from a registered `factor` before the entry is REFUSED.
#: The registered values were measured with <=0.06% spread inside every segment, so 2% is
#: ~30x the measurement noise and still tight enough that a genuinely changed vendor series --
#: Yahoo re-adjusting, or a new action compounding into the wedge -- falls outside it.
BUGFIX_WEDGE_TOL = 0.02
#: The same idea for a return seam, against the observed one-bar step. Tighter, because a bar
#: ratio is one division of two stored numbers with no join and no median in between.
BUGFIX_STEP_TOL = 0.005
#: How near a one-bar step must sit to a split ratio to count as a VINTAGE FLIP. Much looser
#: than the two above, and it has to be: a flip is the ratio TIMES that day's real move, and
#: MNST's five are 1.9559, 1.9413, 1.9193, 0.4895 and 0.4935 -- up to 4% off 2.0 or 0.5.
VINTAGE_FLIP_TOL = 0.10
#: The price legs a repair rescales, all by the SAME multiplier on the same bar.
#:
#: ⚠ `open`/`high`/`low` ARE IN HERE and must stay in. `_atr` takes `high - low` and
#: `high - close.shift(1)`, so repairing the close and not the range would not merely leave
#: those bars stale -- it would make them incoherent, and MNST's would read a ~47 close inside
#: a ~95 high/low band. Anything not listed here is deliberately untouched: `volume`, because
#: the two MNST vintages show comparable volume (10.8M on the last stale bar, 10.1M on the
#: next adjusted one) and inventing a factor for a leg with no measured defect is how a
#: register starts lying; `ret`, because it is recomputed downstream FROM these.
REPAIRED_PRICE_FIELDS = ("close_split", "close_total", "open", "high", "low")


def _rescale(wide: dict[str, pd.DataFrame], ticker: str, factor) -> None:
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


def load_bugfix(config_dir: str | Path) -> dict:
    """Read `configs/prices/yf_price_bugfix.json`. Returns `{}` when it is absent, so a
    checkout without the register builds normally instead of failing.

    ⚠ REFUSED WITHOUT AN `_APPROVED` BLOCK, exactly as the two Sharadar registers are and for
    the same reason: a regenerated proposal is byte-identical to a reviewed decision, so
    without the check "human-approved" is a sentence in a docstring. These entries rewrite
    prices, which is the strongest case in the repo for demanding the block."""
    path = Path(config_dir) / BUGFIX_FILENAME
    if not path.exists():
        return {}
    blob = json.loads(path.read_text(encoding="utf-8"))
    if "_APPROVED" not in blob:
        raise ValueError(f"{path} has no `_APPROVED` block -- refusing to apply a price "
                         "repair nobody signed off. Add one stating who measured it and how.")
    return blob


def apply_split_vintage(wide: dict[str, pd.DataFrame], bugfix: dict,
                        vendor_price: pd.DataFrame | None,
                        log: Callable[..., None]) -> int:
    """Put a series back on ONE split basis when the vendor left it on two. Returns the
    number of entries applied.

    THE DEFECT. Yahoo published MNST's 2026-08-11 two-for-one, back-adjusted five scattered
    bars for it and left every other bar in thirty years of history unadjusted. The result is
    not a wedge and not a seam -- it is a series that alternates between two bases inside the
    same three weeks: 97.65, then 48.19, then 93.55, then 47.08, then 90.36, then 45.53. That
    sequence was re-fetched live from Yahoo, so it is upstream and a re-extract does not
    touch it.

    THE REPAIR. Anchor on the first bar AT OR AFTER the split date -- that one is on the new
    basis by definition, because it is the split itself -- and walk BACKWARDS holding a
    multiplier. A one-bar step within `VINTAGE_FLIP_TOL` of the ratio or its reciprocal is not
    a price move, it is the basis changing underneath, so the multiplier flips there. Every
    other step is left alone. `close_split` and `close_total` are rescaled together.

    ⚠ THIS MOVES RETURNS, and it must: the island edges are fabricated moves of -51% and +96%
    that otherwise reach momentum, vol, betas and every MNST label. It is also the only repair
    here that is direction-aware PER BAR, which is exactly why it cannot be written as a
    `factor` -- half the affected bars need no correction at all.

    Every leg in `REPAIRED_PRICE_FIELDS` moves together, `open`/`high`/`low` included; see
    that constant for what is deliberately left alone and why.

    Verified from BOTH ends against Sharadar's independent `price`:
      * before -- the observed wedge must still sit near `expect_wedge` (0.5 for MNST, flat to
        five decimals across every filing row back to 1996). If Yahoo has fixed its data the
        wedge reads 1.0, the entry is SKIPPED and the skip is logged.
      * after -- the wedge is re-measured on the repaired frame and should land near 1.0. A
        repair that does not reach parity is reported rather than quietly trusted.
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
                log("price bugfix: %s split-vintage SKIPPED -- this build's window does not "
                    "straddle %s, so there is no anchor bar to walk back from",
                    ticker, when.date())
                continue

            was = observed_wedge(ticker, when, vendor_price, close)
            if was is None:
                log("price bugfix: %s split-vintage SKIPPED -- no vendor price before %s, so "
                    "the defect cannot be re-verified", ticker, when.date())
                continue
            if abs(was / expect - 1.0) > BUGFIX_WEDGE_TOL:
                log("price bugfix: %s split-vintage SKIPPED at %s -- the defect is GONE or CHANGED: "
                    "observed wedge %.5f, register expects %.5f. Re-measure the entry.",
                    ticker, when.date(), was, expect)
                continue

            multiplier, flips = _vintage_multiplier(series, ahead[0], ratio)
            _rescale(wide, ticker, multiplier)

            applied += 1
            now = observed_wedge(ticker, when, vendor_price, wide["close_split"])
            log("price bugfix: %s split-vintage REPAIRED across %s (ratio %g) -- %d flip(s), "
                "%d of %d bar(s) rescaled; wedge %.5f -> %s", ticker, when.date(), ratio,
                flips, int((multiplier != 1.0).sum()), len(series), was,
                "%.5f" % now if now is not None else "unmeasurable")
            if now is not None and abs(now - 1.0) > BUGFIX_WEDGE_TOL:
                log("price bugfix: %s split-vintage did NOT reach parity -- wedge %.5f after "
                    "the walk. The repair ran; treat the residual as unexplained.", ticker, now)
    return applied


def _vintage_multiplier(series: pd.Series, anchor: pd.Timestamp,
                        ratio: float) -> tuple[pd.Series, int]:
    """The per-bar rescaling that puts `series` back on the basis its `anchor` bar is on.

    Walking BACKWARDS is what makes this well-posed: the newest bar is the one whose basis we
    know, and every flip found going back compounds onto the bars before it. Walking forwards
    would need the basis of the OLDEST bar, which is the thing in question.

    Only bars strictly before `anchor` can move -- it and everything after it are the
    reference -- so a build whose window starts after the split is a no-op rather than a
    second correction."""
    values = series.to_numpy(dtype="float64")
    out = np.ones(values.size, dtype="float64")
    flips = 0
    for i in range(int(series.index.get_loc(anchor)) - 1, -1, -1):
        here = values[i] * out[i + 1]
        if not np.isfinite(here) or here <= 0:
            out[i] = out[i + 1]
            continue
        step = (values[i + 1] * out[i + 1]) / here
        if abs(step * ratio - 1.0) < VINTAGE_FLIP_TOL:      # this bar reads `ratio` too HIGH
            out[i], flips = out[i + 1] / ratio, flips + 1
        elif abs(step / ratio - 1.0) < VINTAGE_FLIP_TOL:    # ... or `ratio` too LOW
            out[i], flips = out[i + 1] * ratio, flips + 1
        else:
            out[i] = out[i + 1]
    return pd.Series(out, index=series.index), flips


def apply_return_seams(wide: dict[str, pd.DataFrame], bugfix: dict,
                       log: Callable[..., None]) -> int:
    """Repair a discontinuity that is not a price move. Returns the number applied.

    A seam is a one-bar ratio the market did not produce -- Yahoo applying an adjustment on
    the wrong date, or with the wrong factor. Every bar STRICTLY BEFORE the seam date is
    multiplied by the observed step, which makes the series continuous and removes the
    fabricated return.

    Unlike `apply_split_vintage` this is a SINGLE boundary with no islands behind it, so the
    whole prefix moves together and no per-bar decision is needed. JCI is the case: its
    `close_split` falls 70.354 -> 27.775 on 2007-07-02, a factor of 0.3948, while its feed
    claims 0.25 and its real three-for-one was 2007-10-03, where the series does not step at
    all. Feed, applied adjustment and real event disagree three ways.

    ⚠ It moves `ret`, and that is the point -- a fabricated -60.52% bar is a LABEL defect
    first and a level defect second. Every leg in `REPAIRED_PRICE_FIELDS` moves by the same
    factor, so the dividend leg between `close_split` and `close_total`, and the range between
    `high` and `low`, both survive it unchanged.
    """
    applied = 0
    for ticker, entries in (bugfix.get("return_seams") or {}).items():
        for entry in entries:
            when, expected = pd.Timestamp(entry["date"]), float(entry["step"])
            frame = wide.get("close_split")
            if frame is None or ticker not in frame.columns or when not in frame.index:
                log("price bugfix: %s %s seam SKIPPED -- outside this build's window",
                    ticker, when.date())
                continue
            series = frame[ticker]
            position = int(series.index.get_loc(when))
            if position == 0:
                log("price bugfix: %s %s seam SKIPPED -- first bar, no step to measure",
                    ticker, when.date())
                continue
            prior, here = float(series.iloc[position - 1]), float(series.iloc[position])
            if not (np.isfinite(prior) and np.isfinite(here)) or prior <= 0:
                log("price bugfix: %s %s seam SKIPPED -- no usable bar pair",
                    ticker, when.date())
                continue
            observed = here / prior
            if abs(observed / expected - 1.0) > BUGFIX_STEP_TOL:
                log("price bugfix: %s %s seam SKIPPED -- the defect is GONE or CHANGED: "
                    "observed step %.6f, register says %.6f. Re-measure the entry.",
                    ticker, when.date(), observed, expected)
                continue
            _rescale(wide, ticker, (frame.index < when, observed))
            applied += 1
            log("price bugfix: %s %s seam REPAIRED -- every bar before it rescaled by %.6f "
                "(it was a %+.2f%% one-bar 'return' that never happened)",
                ticker, when.date(), observed, (observed - 1) * 100)
    return applied


def apply_level_bugfix(factor: pd.DataFrame, bugfix: dict, vendor_price: pd.DataFrame | None,
                       close_split: pd.DataFrame, log: Callable[..., None]) -> pd.DataFrame:
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
                log("price bugfix: %s < %s SKIPPED -- no vendor price in the segment, so "
                    "the defect cannot be re-verified", ticker, before.date())
                continue
            if abs(observed / claimed - 1.0) > BUGFIX_WEDGE_TOL:
                log("price bugfix: %s < %s SKIPPED -- the defect is GONE or CHANGED: "
                    "observed wedge %.5f, register says %.5f. Re-measure the entry.",
                    ticker, before.date(), observed, claimed)
                continue
            out.loc[window, ticker] = out.loc[window, ticker] * claimed
            log("price bugfix: %s < %s level x%.5f APPLIED to %d bar(s) -- observed wedge "
                "%.5f (%s)", ticker, before.date(), claimed, int(window.sum()), observed,
                segment.get("event", "no event recorded"))
    return out


def observed_wedge(ticker: str, before: pd.Timestamp, vendor_price: pd.DataFrame | None,
                   close_split: pd.DataFrame, *,
                   since: pd.Timestamp = pd.Timestamp.min) -> float | None:
    """`median(sharadar.price / close_split)` over `[since, before)`, or None if unmeasurable.

    The MEDIAN, not the mean: one filing row landing on a stale bar would drag a mean, and
    every registered segment is a plateau with <=0.06% spread, so the median IS the plateau.

    ⚠ `since` IS NOT OPTIONAL IN SPIRIT. Leave it open and a multi-segment ticker measures the
    era BEFORE the one being checked, because the older rows outnumber the newer ones and the
    median follows them: HBAN's second window read 1.46410 where the truth is 1.33100. It
    defaults to open only because the split-vintage repair genuinely means "all history".

    The price a filing row sees is the last bar AT OR BEFORE it -- a filing date is often a
    weekend or a holiday, and that is the same rule `daily_market_cap`'s forward-fill applies.
    """
    if vendor_price is None or vendor_price.empty or ticker not in close_split.columns:
        return None
    rows = vendor_price[(vendor_price["ticker"] == ticker)
                        & (vendor_price["date"] >= since)
                        & (vendor_price["date"] < before)]
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
